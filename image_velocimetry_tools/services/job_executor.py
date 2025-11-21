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

from image_velocimetry_tools.services.base_service import BaseService
from image_velocimetry_tools.services.video_service import VideoService
from image_velocimetry_tools.services.orthorectification_service import OrthorectificationService
from image_velocimetry_tools.services.grid_service import GridService
from image_velocimetry_tools.services.image_stack_service import ImageStackService
from image_velocimetry_tools.services.discharge_service import DischargeService
from image_velocimetry_tools.batch.models import BatchJob
from image_velocimetry_tools.batch.exceptions import JobExecutionError
from image_velocimetry_tools.stiv import two_dimensional_stiv_exhaustive
from image_velocimetry_tools.gui.xsgeometry import CrossSectionGeometry


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
        output_dir: str
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

            # Step 1: Extract frames from video
            self.logger.info(f"[{job.job_id}] Step 1/6: Extracting frames from video")
            frames_dir = self._extract_frames(job, job_dir, project_data)

            # Step 2: Orthorectify images
            self.logger.info(f"[{job.job_id}] Step 2/6: Orthorectifying images")
            rectified_frames = self._orthorectify_frames(
                frames_dir, job_dir, project_data
            )

            # Step 3: Generate grid
            self.logger.info(f"[{job.job_id}] Step 3/6: Generating grid along cross-section")
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
            discharge_result = self._compute_discharge(
                stiv_results,
                grid_points,
                cross_section_path,
                job.water_surface_elevation,
                job.alpha
            )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Save results
            self._save_job_results(
                job_dir, job, discharge_result, processing_time
            )

            self.logger.info(
                f"[{job.job_id}] Job completed successfully in {processing_time:.1f}s"
            )
            self.logger.info(
                f"[{job.job_id}] Discharge: {discharge_result['total_discharge']:.3f} m³/s"
            )

            return {
                "discharge": discharge_result["total_discharge"],
                "area": discharge_result["total_area"],
                "processing_time": processing_time,
                "job_output_dir": job_dir,
            }

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Job execution failed: {str(e)}"
            self.logger.error(f"[{job.job_id}] {error_msg}")

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
        os.makedirs(os.path.join(job_dir, "2-orthorectification"), exist_ok=True)
        os.makedirs(os.path.join(job_dir, "4-velocities"), exist_ok=True)
        os.makedirs(os.path.join(job_dir, "5-discharge"), exist_ok=True)

        self.logger.debug(f"Created job directory: {job_dir}")
        return job_dir

    def _extract_frames(
        self,
        job: BatchJob,
        job_dir: str,
        project_data: Dict
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
        from image_velocimetry_tools.ffmpeg_tools import create_ffmpeg_command
        import subprocess

        frames_dir = os.path.join(job_dir, "1-images")

        # Build FFmpeg command for frame extraction
        # Get video metadata first
        from image_velocimetry_tools.opencv_tools import opencv_get_video_metadata

        try:
            video_metadata = opencv_get_video_metadata(job.video_path)
        except Exception as e:
            raise JobExecutionError(f"Failed to read video metadata: {e}") from e

        # Get ffmpeg parameters from scaffold
        ffmpeg_params = project_data.get("ffmpeg_parameters", {})

        # Build frame extraction parameters
        frame_rate = ffmpeg_params.get("frame_rate", 10)
        frame_step = ffmpeg_params.get("frame_step", 1)

        # Build output pattern
        output_pattern = os.path.join(frames_dir, "f%04d.jpg")

        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-i", job.video_path,
        ]

        # Add time clipping if specified
        if job.start_time_seconds is not None:
            cmd.extend(["-ss", str(job.start_time_seconds)])
        if job.end_time_seconds is not None:
            duration = job.end_time_seconds - (job.start_time_seconds or 0)
            cmd.extend(["-t", str(duration)])

        # Add frame extraction parameters
        cmd.extend([
            "-vf", f"select='not(mod(n\\,{frame_step}))'",
            "-vsync", "0",
            "-q:v", "2",  # Quality
            output_pattern
        ])

        # Execute FFmpeg
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
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
        self,
        frames_dir: str,
        job_dir: str,
        project_data: Dict
    ) -> list:
        """Orthorectify extracted frames using camera matrix method.

        Parameters
        ----------
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
            CameraHelper
        )

        # Get frames to process
        frames = sorted(glob.glob(os.path.join(frames_dir, "f*.jpg")))

        if not frames:
            raise JobExecutionError("No frames found for orthorectification")

        # Get rectification parameters
        rect_params = project_data["rectification_parameters"]

        # Extract GCPs and ICPs
        gcps = np.array(rect_params["ground_control_points"])  # N x 3
        icps = np.array(rect_params["image_control_points"])  # N x 2

        # Get water surface elevation (Z plane for rectification)
        # For camera matrix, we need to specify the rectification plane elevation
        # Use the mean Z of GCPs as default, but this will be overridden if needed
        z_plane = np.mean(gcps[:, 2])

        # Create camera helper to get projection matrix
        camera_helper = CameraHelper()
        camera_helper.add_space_points(gcps)
        camera_helper.add_image_points(icps)
        projection_matrix, rmse = camera_helper.get_camera_matrix()

        # Prepare batch config for rectify_many_camera
        batch_config = [
            frames_dir,  # Input folder
            z_plane,  # Z elevation for rectification plane
            projection_matrix  # 3x4 projection matrix
        ]

        # Call rectify_many_camera
        try:
            # This function writes rectified frames as t*.jpg in the same directory
            rectify_many_camera(batch_config)
        except Exception as e:
            raise JobExecutionError(f"Orthorectification failed: {e}") from e

        # Get list of rectified frames
        rectified_frames = sorted(glob.glob(os.path.join(frames_dir, "t*.jpg")))

        if not rectified_frames:
            raise JobExecutionError("No rectified frames were produced")

        self.logger.info(f"Orthorectified {len(rectified_frames)} frames")
        return rectified_frames

    def _generate_grid(
        self,
        job_dir: str,
        project_data: Dict
    ) -> np.ndarray:
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
        grid_params = project_data.get("grid_parameters", {})

        # For batch processing, grid should be along cross-section line
        if not grid_params.get("use_cross_section_line", False):
            raise JobExecutionError(
                "Grid parameters must specify use_cross_section_line=True"
            )

        # Get cross-section line endpoints
        xs_line_start = grid_params.get("cross_section_line_start")
        xs_line_end = grid_params.get("cross_section_line_end")

        if xs_line_start is None or xs_line_end is None:
            raise JobExecutionError(
                "Cross-section line endpoints not specified in grid parameters"
            )

        # Get number of points or spacing
        num_points = grid_params.get("num_points", 50)

        # Generate evenly spaced points along the line
        x_start, y_start = xs_line_start
        x_end, y_end = xs_line_end

        x_points = np.linspace(x_start, x_end, num_points)
        y_points = np.linspace(y_start, y_end, num_points)

        grid_points = np.column_stack([x_points, y_points])

        # Apply masks if specified
        mask_polygons = grid_params.get("mask_polygons", [])
        if mask_polygons:
            # Filter out points inside mask polygons
            # This is a simplified implementation
            # In production, you'd use proper point-in-polygon testing
            pass

        self.logger.info(f"Generated grid with {len(grid_points)} points")
        return grid_points

    def _create_image_stack(
        self,
        rectified_frames: list,
        job_dir: str
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
                map_file_size_thres=9e8  # 900 MB threshold
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
        project_data: Dict
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
        stiv_params = project_data.get("stiv_parameters", {})
        rect_params = project_data.get("rectification_parameters", {})

        # Get STIV parameters
        num_pixels = stiv_params.get("num_pixels", 20)
        phi_origin = stiv_params.get("phi_origin", 90)
        d_phi = stiv_params.get("d_phi", 1.0)
        phi_range = stiv_params.get("phi_range", 90)
        max_vel_threshold = stiv_params.get("max_vel_threshold_mps", 10.0)
        sigma = stiv_params.get("gaussian_blur_sigma", 0.5)

        # Get pixel GSD (ground scale distance)
        pixel_gsd = rect_params.get("pixel_gsd", 0.01)  # meters/pixel

        # Get frame interval (d_t)
        extraction_params = project_data.get("extraction_parameters", {})
        timestep_ms = extraction_params.get("timestep_ms", 100)
        d_t = timestep_ms / 1000.0  # Convert to seconds

        # Extract x, y coordinates
        x_origin = grid_points[:, 0].astype(float)
        y_origin = grid_points[:, 1].astype(float)

        # Run STIV exhaustive
        try:
            magnitudes, directions, stis, thetas = two_dimensional_stiv_exhaustive(
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
                progress_signal=None  # No progress callback for batch
            )
        except Exception as e:
            raise JobExecutionError(f"STIV analysis failed: {e}") from e

        self.logger.info(f"STIV analysis completed for {len(magnitudes)} points")

        return STIVResults(
            magnitudes_mps=magnitudes,
            directions=directions,
            magnitude_normals_mps=magnitudes,  # Same for exhaustive method
            stis=stis,
            thetas=thetas
        )

    def _compute_discharge(
        self,
        stiv_results: STIVResults,
        grid_points: np.ndarray,
        cross_section_path: str,
        water_surface_elevation: float,
        alpha: float
    ) -> Dict[str, Any]:
        """Compute discharge from STIV results.

        Parameters
        ----------
        stiv_results : STIVResults
            STIV analysis results
        grid_points : np.ndarray
            Grid points (N x 2)
        cross_section_path : str
            Path to AC3 cross-section file
        water_surface_elevation : float
            Water surface elevation (m)
        alpha : float
            Velocity correction coefficient

        Returns
        -------
        dict
            Discharge computation results
        """
        # Load cross-section geometry
        try:
            xs_survey = CrossSectionGeometry()
            xs_survey.import_geometry(cross_section_path)
        except Exception as e:
            raise JobExecutionError(
                f"Failed to load cross-section geometry: {e}"
            ) from e

        # Get station and depth from cross-section
        stations, depths = self.discharge_service.get_station_and_depth(
            xs_survey, grid_points, water_surface_elevation
        )

        # Extract surface velocities
        surface_velocities = self.discharge_service.extract_velocity_from_stiv(
            stiv_results, add_edge_zeros=True
        )

        # Create discharge dataframe
        discharge_df = self.discharge_service.create_discharge_dataframe(
            stations=stations,
            depths=depths,
            surface_velocities=surface_velocities,
            alpha=alpha
        )

        # Compute total discharge
        discharge_result = self.discharge_service.compute_discharge(discharge_df)

        return discharge_result

    def _save_job_results(
        self,
        job_dir: str,
        job: BatchJob,
        discharge_result: Dict,
        processing_time: float
    ) -> None:
        """Save job results to output directory.

        Parameters
        ----------
        job_dir : str
            Job output directory
        job : BatchJob
            Job specification
        discharge_result : dict
            Discharge computation results
        processing_time : float
            Processing time in seconds
        """
        import json

        # Prepare results summary
        results = {
            "job_id": job.job_id,
            "video_path": job.video_path,
            "water_surface_elevation": job.water_surface_elevation,
            "alpha": job.alpha,
            "discharge_m3s": discharge_result["total_discharge"],
            "area_m2": discharge_result["total_area"],
            "processing_time_seconds": processing_time,
            "measurement_number": job.measurement_number,
            "measurement_date": job.measurement_date,
            "comments": job.comments,
        }

        # Save results as JSON
        results_path = os.path.join(job_dir, "job_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.debug(f"Saved job results to {results_path}")
