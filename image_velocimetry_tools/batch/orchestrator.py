"""BatchOrchestrator service for coordinating batch video processing workflows.

This service orchestrates the complete workflow for processing videos in batch
mode, combining multiple services to extract frames, rectify, compute velocities,
and calculate discharge.
"""

import os
import time
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Callable, List
import numpy as np

from image_velocimetry_tools.services.base_service import BaseService
from image_velocimetry_tools.services.video_service import VideoService
from image_velocimetry_tools.services.orthorectification_service import OrthorectificationService
from image_velocimetry_tools.services.stiv_service import STIVService
from image_velocimetry_tools.services.discharge_service import DischargeService
from image_velocimetry_tools.services.project_service import ProjectService
from image_velocimetry_tools.batch.config import (
    BatchVideoConfig,
    ProcessingResult,
    BatchResult,
    VideoConfig,
    ScaffoldConfig,
)


class BatchOrchestrator(BaseService):
    """Orchestrates batch video processing workflow.

    This service coordinates multiple services to process videos through the
    complete workflow:
    1. Extract frames from video (VideoService)
    2. Rectify frames to world coordinates (OrthorectificationService)
    3. Compute velocities using STIV (STIVService)
    4. Calculate discharge (DischargeService)
    5. Save results to .ivy project and CSV (ProjectService)

    The orchestrator is designed for both single-video and batch processing,
    with detailed error handling and progress reporting.

    Attributes:
        video_service: Service for video operations
        ortho_service: Service for orthorectification
        stiv_service: Service for STIV velocimetry
        discharge_service: Service for discharge calculations
        project_service: Service for project save/load
    """

    def __init__(
        self,
        video_service: Optional[VideoService] = None,
        ortho_service: Optional[OrthorectificationService] = None,
        stiv_service: Optional[STIVService] = None,
        discharge_service: Optional[DischargeService] = None,
        project_service: Optional[ProjectService] = None,
    ):
        """Initialize BatchOrchestrator with service dependencies.

        Args:
            video_service: VideoService instance (creates new if None)
            ortho_service: OrthorectificationService instance (creates new if None)
            stiv_service: STIVService instance (creates new if None)
            discharge_service: DischargeService instance (creates new if None)
            project_service: ProjectService instance (creates new if None)
        """
        super().__init__()

        # Dependency injection - create services if not provided
        self.video_service = video_service or VideoService()
        self.ortho_service = ortho_service or OrthorectificationService()
        self.stiv_service = stiv_service or STIVService()
        self.discharge_service = discharge_service or DischargeService()
        self.project_service = project_service or ProjectService()

    def process_video(
        self,
        config: BatchVideoConfig,
        output_directory: str,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        cleanup_temp_files: bool = True,
    ) -> ProcessingResult:
        """Process a single video through complete workflow.

        Args:
            config: Combined scaffold and video configuration
            output_directory: Directory for output files
            progress_callback: Optional callback(percent, message) for progress
            cleanup_temp_files: Whether to delete temporary frame files after processing

        Returns:
            ProcessingResult with outputs and metrics
        """
        start_time = time.time()

        # Create video-specific output directory
        video_name = Path(config.video.video_path).stem
        video_output_dir = os.path.join(output_directory, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        # Initialize result
        result = ProcessingResult(
            video_path=config.video.video_path,
            success=False,
        )

        try:
            # ============================================================
            # Stage 1: Extract frames from video
            # ============================================================
            if progress_callback:
                progress_callback(0, f"Processing {video_name}: Extracting frames...")

            self.logger.info(f"Stage 1: Extracting frames from {config.video.video_path}")

            frames_dir = os.path.join(video_output_dir, "frames")

            def frames_progress(pct, msg):
                if progress_callback:
                    # Map frames extraction to 0-20% of total progress
                    progress_callback(int(pct * 0.2), f"{video_name}: {msg}")

            try:
                frame_files, frame_metadata = self.video_service.extract_frames(
                    video_path=config.video.video_path,
                    output_directory=frames_dir,
                    start_time=config.video.start_time,
                    end_time=config.video.end_time,
                    frame_step=config.video.frame_step,
                    max_frames=config.video.max_frames,
                    progress_callback=frames_progress,
                )

                result.num_frames_extracted = len(frame_files)
                result.frames_directory = frames_dir
                result.frame_files = frame_files
                result.timestep_seconds = frame_metadata["timestep_ms"] / 1000.0

                self.logger.info(f"Extracted {len(frame_files)} frames")

            except Exception as e:
                result.error_stage = "frames"
                result.error_message = str(e)
                self.logger.error(f"Frame extraction failed: {e}")
                raise

            # ============================================================
            # Stage 2: Get video metadata
            # ============================================================
            if progress_callback:
                progress_callback(20, f"{video_name}: Getting video metadata...")

            self.logger.info("Stage 2: Getting video metadata")

            try:
                video_metadata = self.video_service.get_video_metadata(
                    config.video.video_path
                )
                result.video_metadata = video_metadata

            except Exception as e:
                result.error_stage = "metadata"
                result.error_message = str(e)
                self.logger.error(f"Metadata extraction failed: {e}")
                raise

            # ============================================================
            # Stage 3: Rectify frames to world coordinates
            # ============================================================
            if progress_callback:
                progress_callback(25, f"{video_name}: Rectifying frames...")

            self.logger.info("Stage 3: Rectifying frames")

            rectified_dir = os.path.join(video_output_dir, "rectified")

            def rectify_progress(pct, msg):
                if progress_callback:
                    # Map rectification to 25-50% of total progress
                    progress_callback(25 + int(pct * 0.25), f"{video_name}: {msg}")

            try:
                # Prepare rectification parameters
                # For camera matrix method, override WSE with video-specific value
                rectification_params = config.scaffold.rectification_params.copy()
                if config.scaffold.rectification_method == "camera matrix":
                    rectification_params["water_surface_elevation"] = config.video.water_surface_elevation
                    self.logger.debug(
                        f"Using video-specific WSE for camera matrix: "
                        f"{config.video.water_surface_elevation}m"
                    )

                rectified_files = self.ortho_service.rectify_frames_batch(
                    frame_paths=frame_files,
                    output_directory=rectified_dir,
                    method=config.scaffold.rectification_method,
                    rectification_params=rectification_params,
                    progress_callback=rectify_progress,
                )

                result.rectified_frames_directory = rectified_dir
                result.rectified_frame_files = rectified_files

                self.logger.info(f"Rectified {len(rectified_files)} frames")

            except Exception as e:
                result.error_stage = "rectify"
                result.error_message = str(e)
                self.logger.error(f"Rectification failed: {e}")
                raise

            # ============================================================
            # Stage 4: Extract grid points and GSD from scaffold
            # ============================================================
            if progress_callback:
                progress_callback(50, f"{video_name}: Preparing STIV grid...")

            self.logger.info("Stage 4: Extracting grid points")

            try:
                # Get grid points and GSD from scaffold
                # Grid points are generated automatically from cross-section line
                # when scaffold is loaded (see ProjectService._extract_grid_params)
                grid_data = config.scaffold.grid_params
                grid_points = grid_data.get("grid_points")

                if grid_points is None:
                    raise ValueError(
                        "Grid points not found in scaffold. Ensure scaffold has "
                        "a cross-section line defined."
                    )

                # Get pixel GSD
                pixel_gsd = grid_data.get("pixel_gsd", 0.1)
                result.pixel_gsd = pixel_gsd

                self.logger.debug(
                    f"Using {len(grid_points)} grid points with GSD={pixel_gsd:.4f} m/px"
                )

            except Exception as e:
                result.error_stage = "grid"
                result.error_message = str(e)
                self.logger.error(f"Grid extraction failed: {e}")
                raise

            # ============================================================
            # Stage 5: Process STIV velocimetry
            # ============================================================
            if progress_callback:
                progress_callback(55, f"{video_name}: Computing velocities (STIV)...")

            self.logger.info("Stage 5: Processing STIV")

            def stiv_progress(pct, msg):
                if progress_callback:
                    # Map STIV to 55-85% of total progress
                    progress_callback(55 + int(pct * 0.30), f"{video_name}: {msg}")

            try:
                stiv_results = self.stiv_service.process_stiv(
                    frame_files=rectified_files,
                    grid_points=grid_points,
                    phi_origin=config.scaffold.stiv_params["phi_origin"],
                    phi_range=config.scaffold.stiv_params["phi_range"],
                    dphi=config.scaffold.stiv_params["dphi"],
                    num_pixels=config.scaffold.stiv_params["num_pixels"],
                    pixel_gsd=pixel_gsd,
                    timestep_seconds=result.timestep_seconds,
                    progress_callback=stiv_progress,
                )

                result.stiv_magnitudes = stiv_results["magnitudes_mps"]
                result.stiv_directions = stiv_results["directions_deg"]

                self.logger.info("STIV processing complete")

            except Exception as e:
                result.error_stage = "stiv"
                result.error_message = str(e)
                self.logger.error(f"STIV processing failed: {e}")
                raise

            # ============================================================
            # Stage 6: Calculate discharge
            # ============================================================
            if progress_callback:
                progress_callback(85, f"{video_name}: Calculating discharge...")

            self.logger.info("Stage 6: Calculating discharge")

            try:
                # Get cross-section from scaffold
                xs_data = config.scaffold.cross_section_data
                xs_survey = xs_data.get("line")  # Cross-section line object

                if xs_survey is None:
                    raise ValueError("Cross-section data not found in scaffold")

                # Process discharge workflow
                discharge_results = self.discharge_service.process_discharge_workflow(
                    xs_survey=xs_survey,
                    grid_points=grid_points,
                    water_surface_elevation=config.video.water_surface_elevation,
                    stiv_results=stiv_results,
                    alpha=config.video.alpha,
                )

                result.total_discharge = discharge_results["total_discharge"]
                result.total_area = discharge_results["total_area"]
                result.mean_velocity = discharge_results["mean_velocity"]
                result.discharge_dataframe = discharge_results["discharge_dataframe"]
                result.discharge_uncertainty = discharge_results["uncertainty"]

                self.logger.info(
                    f"Discharge: Q={result.total_discharge:.4f} m³/s, "
                    f"A={result.total_area:.4f} m², "
                    f"V={result.mean_velocity:.4f} m/s"
                )

            except Exception as e:
                result.error_stage = "discharge"
                result.error_message = str(e)
                self.logger.error(f"Discharge calculation failed: {e}")
                raise

            # ============================================================
            # Stage 7: Save results
            # ============================================================
            if progress_callback:
                progress_callback(90, f"{video_name}: Saving results...")

            self.logger.info("Stage 7: Saving results")

            try:
                # Save discharge CSV
                csv_filename = f"{video_name}_discharge.csv"
                csv_path = os.path.join(video_output_dir, csv_filename)
                result.discharge_dataframe.to_csv(csv_path, index=False)
                result.output_csv_path = csv_path

                # Build and save .ivy project
                ivy_filename = f"{video_name}_results.ivy"
                ivy_path = os.path.join(video_output_dir, ivy_filename)

                project_dict = self._build_project_dict(
                    scaffold_config=config.scaffold,
                    video_config=config.video,
                    video_metadata=result.video_metadata,
                    stiv_results=stiv_results,
                    discharge_results=discharge_results,
                    grid_points=grid_points,
                    pixel_gsd=pixel_gsd,
                )

                # Save project using ProjectService
                self.project_service.save_project(
                    project_dict=project_dict,
                    save_path=ivy_path,
                    swap_directory=config.scaffold.swap_directory
                )

                result.output_project_path = ivy_path

                self.logger.info(f"Results saved to {video_output_dir}")

            except Exception as e:
                result.error_stage = "save"
                result.error_message = str(e)
                self.logger.error(f"Saving results failed: {e}")
                raise

            # ============================================================
            # Stage 8: Cleanup temporary files
            # ============================================================
            if cleanup_temp_files:
                if progress_callback:
                    progress_callback(95, f"{video_name}: Cleaning up...")

                self.logger.info("Stage 8: Cleaning up temporary files")

                try:
                    # Remove raw frames (keep rectified frames)
                    if os.path.exists(frames_dir):
                        shutil.rmtree(frames_dir)
                        self.logger.debug(f"Removed temporary frames: {frames_dir}")

                except Exception as e:
                    self.logger.warning(f"Cleanup failed (non-fatal): {e}")

            # ============================================================
            # Success!
            # ============================================================
            result.success = True
            result.processing_time_seconds = time.time() - start_time

            if progress_callback:
                progress_callback(100, f"{video_name}: Complete!")

            self.logger.info(
                f"Processing complete: {video_name} "
                f"({result.processing_time_seconds:.1f}s)"
            )

        except Exception as e:
            # Error already logged and captured in result
            result.success = False
            result.processing_time_seconds = time.time() - start_time

            if progress_callback:
                progress_callback(
                    0, f"{video_name}: Failed at {result.error_stage} - {result.error_message}"
                )

            self.logger.error(
                f"Processing failed: {video_name} at stage {result.error_stage}"
            )

        return result

    def process_batch(
        self,
        scaffold_config: ScaffoldConfig,
        video_configs: List[VideoConfig],
        output_directory: str,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        cleanup_temp_files: bool = True,
    ) -> BatchResult:
        """Process multiple videos in batch mode.

        Args:
            scaffold_config: Scaffold template configuration
            video_configs: List of video-specific configurations
            output_directory: Root output directory for batch
            progress_callback: Optional callback(percent, message) for progress
            cleanup_temp_files: Whether to delete temporary files after processing

        Returns:
            BatchResult with aggregated results
        """
        start_time = time.time()
        total_videos = len(video_configs)

        self.logger.info(f"Starting batch processing: {total_videos} videos")

        # Create output directory
        os.makedirs(output_directory, exist_ok=True)

        # Initialize batch result
        batch_result = BatchResult(
            total_videos=total_videos,
            successful=0,
            failed=0,
            output_directory=output_directory,
        )

        # Process each video
        for i, video_config in enumerate(video_configs):
            video_num = i + 1
            video_name = Path(video_config.video_path).stem

            self.logger.info(f"Processing video {video_num}/{total_videos}: {video_name}")

            # Create combined config
            combined_config = BatchVideoConfig(
                scaffold=scaffold_config,
                video=video_config,
            )

            # Progress callback for this video
            def video_progress(pct, msg):
                if progress_callback:
                    # Calculate overall progress
                    video_progress_pct = (i / total_videos) * 100
                    current_video_pct = (pct / 100) * (100 / total_videos)
                    overall_pct = int(video_progress_pct + current_video_pct)
                    progress_callback(overall_pct, f"[{video_num}/{total_videos}] {msg}")

            # Process video
            result = self.process_video(
                config=combined_config,
                output_directory=output_directory,
                progress_callback=video_progress,
                cleanup_temp_files=cleanup_temp_files,
            )

            # Update batch result
            batch_result.video_results.append(result)
            if result.success:
                batch_result.successful += 1
            else:
                batch_result.failed += 1

            self.logger.info(f"Video {video_num}/{total_videos}: {result}")

        # Save batch summary CSV
        if progress_callback:
            progress_callback(95, "Saving batch summary...")

        self._save_batch_summary(batch_result, output_directory)

        # Cleanup scaffold temp directory if needed
        if scaffold_config.temp_cleanup_required:
            try:
                temp_dir = os.path.dirname(scaffold_config.swap_directory)
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    self.logger.debug(f"Cleaned up scaffold temp directory: {temp_dir}")
            except Exception as e:
                self.logger.warning(f"Scaffold cleanup failed (non-fatal): {e}")

        batch_result.processing_time_seconds = time.time() - start_time

        if progress_callback:
            progress_callback(100, f"Batch complete: {batch_result}")

        self.logger.info(f"Batch processing complete: {batch_result}")

        return batch_result

    def _save_batch_summary(self, batch_result: BatchResult, output_directory: str):
        """Save batch summary CSV with all video results.

        Args:
            batch_result: BatchResult to save
            output_directory: Output directory for CSV
        """
        import csv

        csv_path = os.path.join(output_directory, "batch_summary.csv")

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "video_path",
                "success",
                "error_stage",
                "error_message",
                "discharge_m3s",
                "area_m2",
                "mean_velocity_ms",
                "num_frames",
                "processing_time_s",
            ])

            # Data rows
            for result in batch_result.video_results:
                writer.writerow([
                    result.video_path,
                    result.success,
                    result.error_stage or "",
                    result.error_message or "",
                    f"{result.total_discharge:.4f}" if result.success else "",
                    f"{result.total_area:.4f}" if result.success else "",
                    f"{result.mean_velocity:.4f}" if result.success else "",
                    result.num_frames_extracted,
                    f"{result.processing_time_seconds:.1f}",
                ])

        batch_result.batch_csv_path = csv_path
        self.logger.info(f"Batch summary saved: {csv_path}")

    def _build_project_dict(
        self,
        scaffold_config: "ScaffoldConfig",
        video_config: "VideoConfig",
        video_metadata: Dict[str, Any],
        stiv_results: Dict[str, np.ndarray],
        discharge_results: Dict,
        grid_points: np.ndarray,
        pixel_gsd: float,
    ) -> Dict[str, Any]:
        """Build complete project dictionary for .ivy file.

        Combines scaffold template with video-specific results to create
        a complete project that can be saved and reopened in IVyTools GUI.

        Args:
            scaffold_config: Scaffold configuration
            video_config: Video-specific configuration
            video_metadata: Video metadata from VideoService
            stiv_results: STIV processing results
            discharge_results: Discharge calculation results
            grid_points: Grid points used for analysis
            pixel_gsd: Ground sample distance in m/px

        Returns:
            Complete project dictionary ready to save
        """
        # Start with scaffold project dict as base
        project_dict = scaffold_config.project_dict.copy()

        # Update with video-specific information
        project_dict.update({
            # Video info
            "video_path": video_config.video_path,
            "video_width": video_metadata.get("width", 0),
            "video_height": video_metadata.get("height", 0),
            "video_frame_rate": video_metadata.get("avg_frame_rate", 0),
            "video_duration": video_metadata.get("duration", 0),

            # Measurement parameters
            "water_surface_elevation_m": video_config.water_surface_elevation,
            "alpha_coefficient": video_config.alpha,
            "measurement_date": video_config.measurement_date,
            "comments": video_config.comments,

            # Grid info
            "number_grid_points_along_xs_line": len(grid_points),
            "ortho_pixel_gsd": pixel_gsd,

            # STIV results
            "stiv_magnitudes": stiv_results.get("magnitudes_mps"),
            "stiv_directions": stiv_results.get("directions_deg"),

            # Discharge results
            "total_discharge": discharge_results.get("total_discharge", 0),
            "total_area": discharge_results.get("total_area", 0),
            "mean_velocity": discharge_results.get("mean_velocity", 0),
            "discharge_uncertainty": discharge_results.get("uncertainty", {}),
        })

        self.logger.debug("Built project dictionary with batch results")

        return project_dict
