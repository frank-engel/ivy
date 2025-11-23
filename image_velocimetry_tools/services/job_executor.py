"""Service for executing individual batch processing jobs.

This service orchestrates the complete image velocimetry workflow for a single
batch job, from video processing through discharge computation.
"""

import os
import glob
import time
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from dataclasses import dataclass
import json

from image_velocimetry_tools.services.base_service import BaseService
from image_velocimetry_tools.services.video_service import VideoService
from image_velocimetry_tools.services.orthorectification_service import (
    OrthorectificationService,
)
from image_velocimetry_tools.services.grid_service import GridService
from image_velocimetry_tools.services.image_stack_service import (
    ImageStackService,
)
from image_velocimetry_tools.services.discharge_service import DischargeService
from image_velocimetry_tools.batch.models import BatchJob
from image_velocimetry_tools.batch.exceptions import JobExecutionError
from image_velocimetry_tools.stiv import two_dimensional_stiv_exhaustive
from image_velocimetry_tools.file_management import deserialize_numpy_array


@dataclass
class STIVResults:
    """Container for STIV processing results."""

    magnitudes_mps: np.ndarray
    directions: np.ndarray
    magnitude_normals_mps: np.ndarray  # Same as magnitudes for exhaustive
    stis: np.ndarray
    thetas: np.ndarray


class JobExecutor(BaseService):
    """Service for executing individual batch processing jobs.

    This service takes a BatchJob and scaffold configuration, then executes
    the complete processing pipeline:
    1. Extract frames from video
    2. Orthorectify images using camera matrix method
    3. Generate grid along cross-section
    4. Create image stack from rectified frames
    5. Run STIV exhaustive analysis
    6. Compute discharge from velocities

    The service creates a working directory for each job containing all
    intermediate and final results.
    """

    def __init__(self):
        """Initialize the JobExecutor service."""
        super().__init__()

        # Initialize dependent services
        self.video_service = VideoService()
        self.ortho_service = OrthorectificationService()
        self.grid_service = GridService()
        self.image_stack_service = ImageStackService()
        self.discharge_service = DischargeService()

    def execute_job(
        self,
        job: BatchJob,
        scaffold_config: Dict[str, Any],
        output_dir: str,
        save_ivy_project: bool = False,
    ) -> Dict[str, Any]:
        """Execute a complete batch processing job.

        Parameters
        ----------
        job : BatchJob
            Job specification with video path, WSE, and other parameters
        scaffold_config : dict
            Configuration loaded from scaffold project, containing:
            - project_data: Project configuration dict
            - extract_dir: Path to extracted scaffold
            - cross_section_path: Path to AC3 file
        output_dir : str
            Directory where job results should be written
        save_ivy_project : bool, default=False
            If True, save complete .ivy project archive for this job

        Returns
        -------
        dict
            Job results containing:
            - discharge: Computed discharge (m³/s)
            - area: Cross-sectional area (m²)
            - processing_time: Time taken (seconds)
            - job_output_dir: Path to job output directory

        Raises
        ------
        JobExecutionError
            If any step of the processing pipeline fails
        """
        start_time = time.time()

        self.logger.info(f"Starting job execution: {job.job_id}")
        self.logger.info(f"  Video: {job.video_path}")
        self.logger.info(f"  WSE: {job.water_surface_elevation} m")

        # Mark job as processing
        job.mark_processing()

        # Create job working directory structure
        try:
            job_dir = self._create_job_directory(output_dir, job.job_id)
        except Exception as e:
            raise JobExecutionError(
                f"Failed to create job directory: {e}"
            ) from e

        try:
            # Extract scaffold data
            project_data = scaffold_config["project_data"]
            scaffold_extract_dir = scaffold_config["extract_dir"]
            cross_section_path = scaffold_config["cross_section_path"]

            # Convert water_surface_elevation from display units to SI (meters)
            # Job WSE is in display units, but all processing needs SI
            from image_velocimetry_tools.common_functions import (
                units_conversion,
            )

            display_units = project_data.get("display_units", "Metric")
            conversion_factor = units_conversion(display_units)["L"]

            original_wse = job.water_surface_elevation
            job.water_surface_elevation = (
                job.water_surface_elevation / conversion_factor
            )

            self.logger.info(
                f"Converted WSE from {original_wse} {display_units} to "
                f"{job.water_surface_elevation:.3f} m (SI)"
            )

            # Step 1: Extract frames from video
            self.logger.info(
                f"[{job.job_id}] Step 1/6: Extracting frames from video"
            )
            frames_dir = self._extract_frames(job, job_dir, project_data)

            # Step 2: Orthorectify images
            self.logger.info(
                f"[{job.job_id}] Step 2/6: Orthorectifying images"
            )
            rectified_frames = self._orthorectify_frames(
                job, frames_dir, job_dir, project_data
            )

            # Step 3: Generate grid
            self.logger.info(
                f"[{job.job_id}] Step 3/6: Generating grid along cross-section"
            )
            grid_points = self._generate_grid(job_dir, project_data)

            # Step 4: Create image stack
            self.logger.info(f"[{job.job_id}] Step 4/6: Creating image stack")
            image_stack = self._create_image_stack(rectified_frames, job_dir)

            # Step 5: Run STIV
            self.logger.info(f"[{job.job_id}] Step 5/6: Running STIV analysis")
            stiv_results = self._run_stiv(
                image_stack, grid_points, project_data
            )

            # Step 6: Compute discharge
            self.logger.info(f"[{job.job_id}] Step 6/6: Computing discharge")
            display_units = project_data.get("display_units", "Metric")

            # Extract cross-section line endpoints
            xs_line = project_data.get("cross_section_line")
            xs_line = deserialize_numpy_array(xs_line)
            xs_line_endpoints = np.array(xs_line[0])  # [[x1, y1], [x2, y2]]

            discharge_result = self._compute_discharge(
                stiv_results,
                grid_points,
                cross_section_path,
                job.water_surface_elevation,
                job.alpha,
                xs_line_endpoints=xs_line_endpoints,
                display_units=display_units,
            )

            # Convert discharge from SI to display units
            # discharge_result contains SI units (m³/s, m²)
            discharge_si = discharge_result["total_discharge"]  # m³/s
            area_si = discharge_result["total_area"]  # m²

            # Convert to display units
            from image_velocimetry_tools.common_functions import (
                units_conversion,
            )

            conversion_factor_Q = units_conversion(display_units)[
                "Q"
            ]  # Flow rate conversion
            conversion_factor_A = units_conversion(display_units)[
                "A"
            ]  # Area conversion

            discharge_display = discharge_si * conversion_factor_Q
            area_display = area_si * conversion_factor_A

            # Units labels
            if display_units == "English":
                Q_units = "cfs"
                A_units = "ft²"
            else:
                Q_units = "m³/s"
                A_units = "m²"

            # Calculate processing time
            processing_time = time.time() - start_time

            # Prepare result details for job
            result_details = {}
            if discharge_result.get("summary_statistics"):
                stats = discharge_result["summary_statistics"]
                conv_L = (
                    conversion_factor_A**0.5
                )  # sqrt of area factor = length factor
                conv_V = units_conversion(display_units)["V"]

                result_details.update(
                    {
                        "average_velocity": stats["average_velocity"] * conv_V,
                        "average_surface_velocity": stats[
                            "average_surface_velocity"
                        ]
                        * conv_V,
                        "max_surface_velocity": stats["max_surface_velocity"]
                        * conv_V,
                    }
                )

            if discharge_result.get("max_depth") is not None:
                conv_L = units_conversion(display_units)["L"]
                result_details.update(
                    {
                        "max_depth": discharge_result["max_depth"] * conv_L,
                        "min_depth": discharge_result["min_depth"] * conv_L,
                    }
                )

            if discharge_result.get("uncertainty"):
                unc = discharge_result["uncertainty"]
                if unc.get("u_iso"):
                    result_details["uncertainty_iso_95pct"] = unc["u_iso"].get(
                        "u95_q"
                    )
                if unc.get("u_ive"):
                    result_details["uncertainty_ive_95pct"] = unc["u_ive"].get(
                        "u95_q"
                    )

            # Mark job as completed with discharge in display units
            job.mark_completed(
                discharge_value=discharge_display,
                area_value=area_display,
                result_details=result_details,
                processing_time=processing_time,
            )

            # Save results (with both SI and display units)
            self._save_job_results(
                job_dir,
                job,
                discharge_result,
                processing_time,
                discharge_display,
                area_display,
                display_units,
            )

            # Save .ivy project if requested
            if save_ivy_project:
                self.logger.info(
                    f"[{job.job_id}] Creating .ivy project archive..."
                )
                ivy_project_path = self._save_ivy_project(
                    job_dir,
                    job,
                    scaffold_config,
                    project_data,
                    discharge_result,
                    grid_points,
                    stiv_results,
                )
                self.logger.info(
                    f"[{job.job_id}] .ivy project saved: {ivy_project_path}"
                )

            self.logger.info(
                f"[{job.job_id}] Job completed successfully in {processing_time:.1f}s"
            )
            self.logger.info(
                f"[{job.job_id}] Discharge: {discharge_display:.3f} {Q_units} "
                f"({discharge_si:.3f} m³/s)"
            )

            return {
                "discharge": discharge_display,  # In display units
                "area": area_display,  # In display units
                "discharge_si": discharge_si,  # Keep SI for reference
                "area_si": area_si,  # Keep SI for reference
                "processing_time": processing_time,
                "job_output_dir": job_dir,
            }

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Job execution failed: {str(e)}"
            self.logger.error(f"[{job.job_id}] {error_msg}")

            # Mark job as failed
            job.mark_failed(
                error_message=error_msg, processing_time=processing_time
            )

            raise JobExecutionError(error_msg) from e

    def _create_job_directory(self, output_dir: str, job_id: str) -> str:
        """Create working directory structure for job.

        Parameters
        ----------
        output_dir : str
            Parent output directory
        job_id : str
            Job identifier

        Returns
        -------
        str
            Path to created job directory
        """
        job_dir = os.path.join(output_dir, f"job_{job_id}")
        os.makedirs(job_dir, exist_ok=True)

        # Create subdirectories
        os.makedirs(os.path.join(job_dir, "1-images"), exist_ok=True)
        os.makedirs(
            os.path.join(job_dir, "2-orthorectification"), exist_ok=True
        )
        os.makedirs(os.path.join(job_dir, "4-velocities"), exist_ok=True)
        os.makedirs(os.path.join(job_dir, "5-discharge"), exist_ok=True)

        self.logger.debug(f"Created job directory: {job_dir}")
        return job_dir

    def _extract_frames(
        self, job: BatchJob, job_dir: str, project_data: Dict
    ) -> str:
        """Extract frames from video.

        Parameters
        ----------
        job : BatchJob
            Job specification
        job_dir : str
            Job working directory
        project_data : dict
            Project configuration

        Returns
        -------
        str
            Path to directory containing extracted frames
        """
        from image_velocimetry_tools.ffmpeg_tools import ffmpeg_cmd
        import subprocess

        frames_dir = os.path.join(job_dir, "1-images")

        # Build FFmpeg command for frame extraction
        # Get video metadata first
        from image_velocimetry_tools.opencv_tools import (
            opencv_get_video_metadata,
        )

        try:
            video_metadata = opencv_get_video_metadata(job.video_path)
        except Exception as e:
            raise JobExecutionError(
                f"Failed to read video metadata: {e}"
            ) from e

        # Get ffmpeg parameters from scaffold
        ffmpeg_params = project_data.get("ffmpeg_parameters", {})

        # Build frame extraction parameters
        frame_rate = ffmpeg_params.get("frame_rate", 10)
        frame_step = ffmpeg_params.get("frame_step", 1)

        # Build output pattern
        output_pattern = os.path.join(frames_dir, "f%04d.jpg")

        # Build FFmpeg command using the discovered ffmpeg binary
        cmd = [
            ffmpeg_cmd,
            "-i",
            job.video_path,
        ]

        # Add time clipping if specified
        if job.start_time_seconds is not None:
            cmd.extend(["-ss", str(job.start_time_seconds)])
        if job.end_time_seconds is not None:
            duration = job.end_time_seconds - (job.start_time_seconds or 0)
            cmd.extend(["-t", str(duration)])

        # Add frame extraction parameters
        cmd.extend(
            [
                "-vf",
                f"select='not(mod(n\\,{frame_step}))'",
                "-vsync",
                "0",
                "-q:v",
                "2",  # Quality
                output_pattern,
            ]
        )

        # Execute FFmpeg
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as e:
            raise JobExecutionError(
                f"FFmpeg frame extraction failed: {e.stderr}"
            ) from e

        # Verify frames were extracted
        frames = glob.glob(os.path.join(frames_dir, "f*.jpg"))
        if not frames:
            raise JobExecutionError("No frames were extracted from video")

        self.logger.info(f"Extracted {len(frames)} frames to {frames_dir}")
        return frames_dir

    def _orthorectify_frames(
        self, job: BatchJob, frames_dir: str, job_dir: str, project_data: Dict
    ) -> list:
        """Orthorectify extracted frames using camera matrix method.

        Parameters
        ----------
        job : BatchJob
            Job specification with water_surface_elevation
        frames_dir : str
            Directory containing extracted frames
        job_dir : str
            Job working directory
        project_data : dict
            Project configuration

        Returns
        -------
        list
            List of paths to rectified frame files
        """
        from image_velocimetry_tools.orthorectification import (
            rectify_many_camera,
            CameraHelper,
        )

        # Get frames to process
        frames = sorted(glob.glob(os.path.join(frames_dir, "f*.jpg")))

        if not frames:
            raise JobExecutionError("No frames found for orthorectification")

        # Get rectification parameters
        rect_params = project_data["rectification_parameters"]

        # Extract world coords and pixel coords (these are GCPs/ICPs in real .ivy files)
        gcps = np.array(rect_params["world_coords"])  # N x 3
        icps = np.array(rect_params["pixel_coords"])  # N x 2

        # Extract the extent
        extent = rect_params["extent"]

        # Get water surface elevation (Z plane for rectification) from job
        # Note: WSE is already in SI (meters) after conversion in execute_job()
        z_plane = job.water_surface_elevation

        self.logger.info(
            f"Using water surface elevation: {z_plane:.3f} m for rectification"
        )

        # Create camera helper to get projection matrix
        camera_helper = CameraHelper()
        camera_helper.add_space_points(gcps)
        camera_helper.add_image_points(icps)
        projection_matrix, rmse = camera_helper.get_camera_matrix()

        # Prepare batch config for rectify_many_camera
        batch_config = [
            frames_dir,  # Input folder
            z_plane,  # Z elevation for rectification plane
            projection_matrix,  # 3x4 projection matrix
            extent,  # 4x1 world extent bbox
        ]

        # Call rectify_many_camera
        try:
            # This function writes rectified frames as t*.jpg in the same directory
            rectify_many_camera(batch_config)
        except Exception as e:
            raise JobExecutionError(f"Orthorectification failed: {e}") from e

        # Get list of rectified frames
        rectified_frames = sorted(
            glob.glob(os.path.join(frames_dir, "t*.jpg"))
        )

        if not rectified_frames:
            raise JobExecutionError("No rectified frames were produced")

        self.logger.info(f"Orthorectified {len(rectified_frames)} frames")

        # Apply flips if specified in project_data
        flip_x = project_data.get("is_ortho_flip_x", False)
        flip_y = project_data.get("is_ortho_flip_y", False)

        if flip_x or flip_y:
            from image_velocimetry_tools.image_processing_tools import (
                flip_image_array,
            )
            from imageio.v3 import imread, imwrite

            self.logger.info(
                f"Applying flips to rectified frames: flip_x={flip_x}, flip_y={flip_y}"
            )

            for frame_path in rectified_frames:
                # Read the rectified frame
                img = imread(frame_path)

                # Apply flips
                flipped_img = flip_image_array(
                    img, flip_x=flip_x, flip_y=flip_y
                )

                # Write back the flipped image
                imwrite(frame_path, flipped_img)

            self.logger.info(
                f"Applied flips to {len(rectified_frames)} rectified frames"
            )
        return rectified_frames

    def _generate_grid(self, job_dir: str, project_data: Dict) -> np.ndarray:
        """Generate grid points along cross-section.

        Parameters
        ----------
        job_dir : str
            Job working directory
        project_data : dict
            Project configuration

        Returns
        -------
        np.ndarray
            Grid points array (N x 2) in pixel coordinates
        """
        # In real .ivy files, grid parameters are flat keys (not nested)
        # Get cross-section line (format: [[[x1, y1], [x2, y2]]])
        xs_line = project_data.get("cross_section_line")
        xs_line = deserialize_numpy_array(xs_line)
        if xs_line is None or len(xs_line) == 0:
            raise JobExecutionError(
                "Cross-section line not defined in project data"
            )

        # Extract start and end points from first line segment
        line_segment = xs_line[0]  # [[x1, y1], [x2, y2]]
        xs_line_start = line_segment[0]
        xs_line_end = line_segment[1]

        # Get number of grid points
        num_points = project_data.get("number_grid_points_along_xs_line", 50)

        # Generate evenly spaced points along the line
        x_start, y_start = xs_line_start
        x_end, y_end = xs_line_end

        x_points = np.linspace(x_start, x_end, num_points)
        y_points = np.linspace(y_start, y_end, num_points)

        grid_points = np.column_stack([x_points, y_points])

        # Apply masks if specified (flat key in real .ivy files)
        mask_polygons = project_data.get("mask_polygons", [])
        if mask_polygons:
            # Filter out points inside mask polygons
            # This is a simplified implementation
            # In production, you'd use proper point-in-polygon testing
            pass

        self.logger.info(f"Generated grid with {len(grid_points)} points")
        return grid_points

    def _create_image_stack(
        self, rectified_frames: list, job_dir: str
    ) -> np.ndarray:
        """Create grayscale image stack from rectified frames.

        Parameters
        ----------
        rectified_frames : list
            List of paths to rectified frame files
        job_dir : str
            Job working directory

        Returns
        -------
        np.ndarray
            3D image stack array (height x width x num_frames)
        """
        map_file_path = os.path.join(job_dir, "image_stack.dat")

        try:
            image_stack = self.image_stack_service.create_image_stack(
                image_paths=rectified_frames,
                map_file_path=map_file_path,
                map_file_size_thres=9e8,  # 900 MB threshold
            )
        except Exception as e:
            raise JobExecutionError(f"Image stack creation failed: {e}") from e

        self.logger.info(
            f"Created image stack: {image_stack.shape[0]}x{image_stack.shape[1]}x{image_stack.shape[2]}"
        )
        return image_stack

    def _run_stiv(
        self,
        image_stack: np.ndarray,
        grid_points: np.ndarray,
        project_data: Dict,
    ) -> STIVResults:
        """Run STIV exhaustive analysis.

        Parameters
        ----------
        image_stack : np.ndarray
            3D image stack
        grid_points : np.ndarray
            Grid points (N x 2)
        project_data : dict
            Project configuration

        Returns
        -------
        STIVResults
            STIV analysis results
        """
        # In real .ivy files, STIV parameters are flat keys (not nested)
        num_pixels = project_data.get("stiv_num_pixels", 20)
        phi_origin = project_data.get("stiv_phi_origin", 90)
        d_phi = project_data.get(
            "stiv_dphi", 1.0
        )  # Note: key is 'dphi' not 'd_phi'
        phi_range = project_data.get("stiv_phi_range", 90)
        max_vel_threshold = project_data.get(
            "stiv_max_vel_threshold_mps", 10.0
        )
        sigma = project_data.get("stiv_gaussian_blur_sigma", 0.5)

        # Get pixel GSD (ground scale distance) - flat key in real .ivy files
        pixel_gsd = project_data.get(
            "pixel_ground_scale_distance_m", 0.01
        )  # meters/pixel

        # Get frame interval (d_t) - flat key in real .ivy files
        timestep_ms = project_data.get("extraction_timestep_ms", 100)
        d_t = timestep_ms / 1000.0  # Convert to seconds

        # Extract x, y coordinates
        x_origin = grid_points[:, 0].astype(float)
        y_origin = grid_points[:, 1].astype(float)

        # Run STIV exhaustive
        try:
            magnitudes, directions, stis, thetas = (
                two_dimensional_stiv_exhaustive(
                    x_origin=x_origin,
                    y_origin=y_origin,
                    image_stack=image_stack,
                    num_pixels=num_pixels,
                    phi_origin=phi_origin,
                    d_phi=d_phi,
                    phi_range=phi_range,
                    pixel_gsd=pixel_gsd,
                    d_t=d_t,
                    sigma=sigma,
                    max_vel_threshold=max_vel_threshold,
                    progress_signal=None,  # No progress callback for batch
                )
            )
        except Exception as e:
            raise JobExecutionError(f"STIV analysis failed: {e}") from e

        self.logger.info(
            f"STIV analysis completed for {len(magnitudes)} points"
        )

        return STIVResults(
            magnitudes_mps=magnitudes,
            directions=directions,
            magnitude_normals_mps=magnitudes,  # Same for exhaustive method
            stis=stis,
            thetas=thetas,
        )

    def _compute_discharge(
        self,
        stiv_results: STIVResults,
        grid_points: np.ndarray,
        cross_section_path: str,
        water_surface_elevation: float,
        alpha: float,
        xs_line_endpoints: np.ndarray,
        display_units: str = "Metric",
    ) -> Dict[str, Any]:
        """Compute discharge from STIV results.

        Parameters
        ----------
        stiv_results : STIVResults
            STIV analysis results
        grid_points : np.ndarray
            Grid points (N x 2) in world coordinates
        cross_section_path : str
            Path to AC3 cross-section file
        water_surface_elevation : float
            Water surface elevation in SI units (m)
        alpha : float
            Velocity correction coefficient
        xs_line_endpoints : np.ndarray
            Cross-section line endpoints (2 x 2) [[x1, y1], [x2, y2]]
        display_units : str
            Display units from scaffold ("English" or "Metric")

        Returns
        -------
        dict
            Discharge computation results
        """
        # Load cross-section geometry using AreaComp backend directly
        # (avoiding GUI-dependent CrossSectionGeometry class)
        from areacomp.gui.areasurvey import AreaSurvey

        try:
            xs_survey = AreaSurvey()
            xs_survey.load_areacomp(cross_section_path, units=display_units)

            # Log the units from the AC3 file for debugging
            self.logger.info(
                f"Loaded AC3 cross-section: {cross_section_path}, "
                f"AC3 file units: {xs_survey.units}, IVy display units: {display_units}"
            )
        except Exception as e:
            raise JobExecutionError(
                f"Failed to load cross-section geometry: {e}"
            ) from e

        # Get station and depth from cross-section using batch-compatible method
        # Note: water_surface_elevation is already in SI (meters) after early conversion
        # The discharge_service will convert AC3 data to SI if needed
        stations, depths = (
            self.discharge_service.get_station_and_depth_from_grid(
                xs_survey,
                grid_points,
                water_surface_elevation,
                xs_line_endpoints,
            )
        )

        # Extract surface velocities (adds edge zeros)
        surface_velocities = self.discharge_service.extract_velocity_from_stiv(
            stiv_results, add_edge_zeros=True
        )

        # Add edge nodes to stations and depths to match edge velocities
        # Edge nodes are at the banks with zero depth
        if len(surface_velocities) == len(stations) + 2:
            # Estimate station spacing
            station_spacing = (
                stations[1] - stations[0] if len(stations) > 1 else 0
            )
            # Add edge station before first station
            edge_station_left = stations[0] - station_spacing
            # Add edge station after last station
            edge_station_right = stations[-1] + station_spacing
            # Insert edge nodes
            stations = np.insert(stations, 0, edge_station_left)
            stations = np.append(stations, edge_station_right)
            depths = np.insert(depths, 0, 0.0)  # Zero depth at left edge
            depths = np.append(depths, 0.0)  # Zero depth at right edge

        # Create discharge dataframe
        discharge_df = self.discharge_service.create_discharge_dataframe(
            stations=stations,
            depths=depths,
            surface_velocities=surface_velocities,
            alpha=alpha,
        )

        # Compute total discharge
        discharge_result = self.discharge_service.compute_discharge(
            discharge_df
        )

        # Compute uncertainty (requires rectification RMSE and scene width)
        # For batch processing, we'll use estimated values or skip if not available
        # TODO: Extract actual RMSE from rectification step
        rectification_rmse = 0.5  # Placeholder - meters
        scene_width = np.max(stations) - np.min(stations)  # meters

        try:
            uncertainty_result = self.discharge_service.compute_uncertainty(
                discharge_result["discharge_results"],
                discharge_result["total_discharge"],
                rectification_rmse,
                scene_width,
            )
            discharge_result["uncertainty"] = uncertainty_result
        except Exception as e:
            self.logger.warning(f"Failed to compute uncertainty: {e}")
            discharge_result["uncertainty"] = None

        # Compute summary statistics (velocities, depths, etc.)
        try:
            summary_stats = self.discharge_service.compute_summary_statistics(
                discharge_result["discharge_results"],
                discharge_result["total_discharge"],
                discharge_result["total_area"],
            )
            discharge_result["summary_statistics"] = summary_stats

            # Extract depth statistics from discharge dataframe
            depth_values = [
                float(result["Depth"])
                for result in discharge_result["discharge_results"].values()
                if result["Status"] == "Used"
            ]
            discharge_result["max_depth"] = (
                np.max(depth_values) if depth_values else 0
            )
            discharge_result["min_depth"] = (
                np.min(depth_values) if depth_values else 0
            )

        except Exception as e:
            self.logger.warning(f"Failed to compute summary statistics: {e}")
            discharge_result["summary_statistics"] = None
            discharge_result["max_depth"] = None
            discharge_result["min_depth"] = None

        return discharge_result

    def _save_job_results(
        self,
        job_dir: str,
        job: BatchJob,
        discharge_result: Dict,
        processing_time: float,
        discharge_display: float,
        area_display: float,
        display_units: str,
    ) -> None:
        """Save job results to output directory.

        Parameters
        ----------
        job_dir : str
            Job output directory
        job : BatchJob
            Job specification
        discharge_result : dict
            Discharge computation results (SI units)
        processing_time : float
            Processing time in seconds
        discharge_display : float
            Discharge in display units
        area_display : float
            Area in display units
        display_units : str
            Display units ("English" or "Metric")
        """
        import json
        from image_velocimetry_tools.common_functions import units_conversion

        # Get conversion factors for display units
        conv_L = units_conversion(display_units)["L"]  # Length
        conv_V = units_conversion(display_units)["V"]  # Velocity

        # Prepare results summary with both SI and display units
        results = {
            "job_id": job.job_id,
            "video_path": job.video_path,
            "water_surface_elevation": job.water_surface_elevation,
            "alpha": job.alpha,
            "display_units": display_units,
            # Primary results in display units
            "discharge": discharge_display,
            "area": area_display,
            "discharge_units": "cfs" if display_units == "English" else "m³/s",
            "area_units": "ft²" if display_units == "English" else "m²",
            # SI values for reference
            "discharge_m3s": discharge_result["total_discharge"],
            "area_m2": discharge_result["total_area"],
        }

        # Add summary statistics if available
        if discharge_result.get("summary_statistics"):
            stats = discharge_result["summary_statistics"]
            results.update(
                {
                    "average_velocity": stats["average_velocity"] * conv_V,
                    "average_velocity_units": (
                        "ft/s" if display_units == "English" else "m/s"
                    ),
                    "average_surface_velocity": stats[
                        "average_surface_velocity"
                    ]
                    * conv_V,
                    "max_surface_velocity": stats["max_surface_velocity"]
                    * conv_V,
                    "velocity_units": (
                        "ft/s" if display_units == "English" else "m/s"
                    ),
                }
            )

        # Add depth statistics if available
        if discharge_result.get("max_depth") is not None:
            results.update(
                {
                    "max_depth": discharge_result["max_depth"] * conv_L,
                    "min_depth": discharge_result["min_depth"] * conv_L,
                    "depth_units": "ft" if display_units == "English" else "m",
                }
            )

        # Add uncertainty if available
        if discharge_result.get("uncertainty"):
            unc = discharge_result["uncertainty"]
            if unc.get("u_iso"):
                results["uncertainty_iso_95pct"] = unc["u_iso"].get(
                    "u95_q", None
                )
            if unc.get("u_ive"):
                results["uncertainty_ive_95pct"] = unc["u_ive"].get(
                    "u95_q", None
                )

        # Add metadata
        results.update(
            {
                "processing_time_seconds": processing_time,
                "measurement_number": job.measurement_number,
                "measurement_date": job.measurement_date,
                "comments": job.comments,
            }
        )

        # Save results as JSON
        results_path = os.path.join(job_dir, "job_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.debug(f"Saved job results to {results_path}")

    def _save_ivy_project(
        self,
        job_dir: str,
        job: BatchJob,
        scaffold_config: Dict,
        project_data: Dict,
        discharge_result: Dict,
        grid_points: np.ndarray,
        stiv_results: STIVResults,
    ) -> str:
        """Save complete .ivy project archive for this job.

        Parameters
        ----------
        job_dir : str
            Job output directory
        job : BatchJob
            Job specification
        scaffold_config : dict
            Scaffold configuration
        project_data : dict
            Project data from scaffold
        discharge_result : dict
            Discharge computation results
        grid_points : np.ndarray
            Grid points used for STIV
        stiv_results : STIVResults
            STIV analysis results

        Returns
        -------
        str
            Path to created .ivy file
        """
        import zipfile
        import copy
        from image_velocimetry_tools.file_management import (
            serialize_numpy_array,
        )

        # Create job-specific project data by copying scaffold and updating
        job_project_data = copy.deepcopy(project_data)

        # Update with job-specific parameters
        job_project_data.update(
            {
                "job_id": job.job_id,
                "video_file": os.path.basename(job.video_path),
                "water_surface_elevation": job.water_surface_elevation,
                "alpha": job.alpha,
                "discharge": discharge_result.get("total_discharge"),
                "area": discharge_result.get("total_area"),
                "measurement_number": job.measurement_number,
                "measurement_date": job.measurement_date,
                "measurement_time": job.measurement_time,
                "gage_height": job.gage_height,
                "comments": job.comments,
            }
        )

        # Save STIV results
        job_project_data["stiv_magnitudes"] = serialize_numpy_array(
            stiv_results.magnitudes_mps
        )
        job_project_data["stiv_directions"] = serialize_numpy_array(
            stiv_results.directions
        )
        job_project_data["stiv_stis"] = serialize_numpy_array(
            stiv_results.stis
        )
        job_project_data["stiv_thetas"] = serialize_numpy_array(
            stiv_results.thetas
        )

        # Save grid points
        job_project_data["grid_points"] = serialize_numpy_array(grid_points)

        # Create temporary directory for .ivy contents
        ivy_temp_dir = os.path.join(job_dir, "_ivy_temp")
        os.makedirs(ivy_temp_dir, exist_ok=True)

        try:
            # Save project_data.json
            project_json_path = os.path.join(ivy_temp_dir, "project_data.json")
            with open(project_json_path, "w") as f:
                json.dump(job_project_data, f, indent=4)

            # Copy cross-section file from scaffold
            cross_section_src = scaffold_config["cross_section_path"]
            cross_section_filename = os.path.basename(cross_section_src)
            cross_section_dst = os.path.join(
                ivy_temp_dir, cross_section_filename
            )
            shutil.copy2(cross_section_src, cross_section_dst)

            # Copy all image directories (frames and rectified frames)
            for subdir in ["frames", "orthorectification", "image_stack"]:
                src_subdir = os.path.join(job_dir, subdir)
                if os.path.exists(src_subdir):
                    dst_subdir = os.path.join(ivy_temp_dir, subdir)
                    shutil.copytree(src_subdir, dst_subdir)

            # Copy discharge results
            discharge_csv_path = os.path.join(job_dir, "discharge_data.csv")
            if os.path.exists(discharge_csv_path):
                shutil.copy2(
                    discharge_csv_path,
                    os.path.join(ivy_temp_dir, "discharge_data.csv"),
                )

            # Create .ivy ZIP archive
            ivy_output_path = os.path.join(job_dir, f"{job.job_id}.ivy")

            with zipfile.ZipFile(
                ivy_output_path, "w", zipfile.ZIP_DEFLATED
            ) as zipf:
                for root, dirs, files in os.walk(ivy_temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, ivy_temp_dir)
                        zipf.write(file_path, arcname=arcname)

            self.logger.debug(f"Created .ivy project: {ivy_output_path}")

            return ivy_output_path

        finally:
            # Cleanup temporary directory
            if os.path.exists(ivy_temp_dir):
                shutil.rmtree(ivy_temp_dir)
