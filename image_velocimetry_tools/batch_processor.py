"""Batch processing module for IVyTools

This module provides headless batch processing capabilities for processing
multiple videos from the same fixed camera setup. It uses a scaffold .ivy
project containing camera calibration, GCPs, and cross-section data to
process multiple videos with different water surface elevations.

Typical workflow:
    1. Create a scaffold .ivy project with camera matrix, GCPs, cross-section
    2. Prepare a CSV file with video paths, WSE values, and metadata
    3. Run batch processor to generate discharge measurements for all videos
    4. Review individual .ivy files and batch summary CSV
"""

import csv
import json
import logging
import os
import shutil
import tempfile
import traceback
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from areacomp.gui.areasurvey import AreaSurvey

from image_velocimetry_tools.file_management import (
    deserialize_numpy_array,
    serialize_numpy_array,
)
from image_velocimetry_tools.common_functions import dict_arrays_to_list
from image_velocimetry_tools.batch_processing_helpers import (
    extract_frames_headless,
    get_video_metadata,
    generate_cross_section_grid,
    run_stiv_headless,
    calculate_discharge_headless,
    calculate_uncertainty_headless,
)


@dataclass
class ScaffoldData:
    """Holds reusable parameters extracted from a scaffold .ivy project

    This includes camera calibration, GCPs, cross-section geometry, and
    STIV processing parameters that remain constant across videos from
    the same fixed camera.
    """

    # Orthorectification parameters
    rectification_method: str = None
    rectification_parameters: Dict = None
    orthotable_dataframe: pd.DataFrame = None
    pixel_ground_scale_distance_m: float = None

    # Cross-section geometry
    bathymetry_ac3_filename: str = None
    cross_section_line: np.ndarray = None
    cross_section_start_bank: str = None
    cross_section_top_width_m: float = None
    cross_section_hydraulic_radius_m: float = None
    cross_section_length_pixels: float = None
    cross_section_line_exists: bool = False
    is_area_comp_loaded: bool = False

    # STIV parameters
    stiv_search_line_length_m: float = None
    stiv_phi_origin: float = None
    stiv_phi_range: float = None
    stiv_dphi: float = None
    stiv_num_pixels: int = None
    stiv_gaussian_blur_sigma: float = 0.0
    stiv_max_vel_threshold_mps: float = None
    process_step: str = "Process STIV Exhaustive"  # or "Process STIV Optimized"

    # Grid parameters
    number_grid_points_along_xs_line: int = 25
    horz_grid_size: int = 50
    vert_grid_size: int = 50

    # Video/frame extraction defaults
    extraction_frame_rate: float = None
    extraction_frame_step: int = 1
    extraction_num_frames: int = 200

    # Measurement template
    measurement_info: Dict = field(default_factory=dict)

    # Units
    display_units: str = "English"
    survey_units: Dict = field(default_factory=dict)

    # Misc
    ffmpeg_cmd: str = None
    ffprobe_cmd: str = None
    mask_polygons: List = field(default_factory=list)

    # Rectification parameters
    rectification_method: str = None
    projection_matrix: np.ndarray = None
    orthorectification_extent: np.ndarray = None

    # Full project dict (for reference)
    original_project_dict: Dict = field(default_factory=dict)


@dataclass
class BatchVideoConfig:
    """Configuration for a single video in a batch"""

    video_path: str
    water_surface_elevation: float
    measurement_date: str = ""
    measurement_number: int = 0
    gage_height: float = 0.0
    start_time: str = ""  # Format: "HH:MM:SS" or "SS.sss"
    end_time: str = ""    # Format: "HH:MM:SS" or "SS.sss"
    comments: str = ""
    alpha: float = 0.85   # Surface to depth-averaged velocity coefficient (default 0.85)


@dataclass
class BatchResult:
    """Results from processing a single video"""

    video_filename: str
    video_path: str
    measurement_date: str
    measurement_number: int
    wse_m: float
    gage_height: float

    # Processing status
    processing_status: str = "pending"  # pending, success, failed
    error_message: str = ""

    # Discharge results (populated on success)
    total_discharge: Optional[float] = None
    total_area: Optional[float] = None
    iso_uncertainty: Optional[float] = None
    ive_uncertainty: Optional[float] = None
    cross_section_width: Optional[float] = None
    hydraulic_radius: Optional[float] = None
    num_stations_used: int = 0
    num_stations_total: int = 0

    # Mean velocity
    mean_velocity: Optional[float] = None

    # Output paths
    output_ivy_path: str = ""

    # Timing
    processing_start_time: Optional[datetime] = None
    processing_end_time: Optional[datetime] = None

    @property
    def processing_duration_seconds(self) -> Optional[float]:
        """Calculate processing duration in seconds"""
        if self.processing_start_time and self.processing_end_time:
            delta = self.processing_end_time - self.processing_start_time
            return delta.total_seconds()
        return None


class BatchProcessor:
    """Main batch processor for processing multiple videos with a scaffold project

    Example usage:
        processor = BatchProcessor(
            scaffold_ivy_path='template.ivy',
            batch_config_csv='batch_videos.csv'
        )
        results = processor.process_batch(output_directory='./output')
        processor.export_results_csv(results, 'batch_summary.csv')
    """

    def __init__(
        self,
        scaffold_ivy_path: str,
        batch_config_csv: Optional[str] = None,
        batch_configs: Optional[List[BatchVideoConfig]] = None
    ):
        """Initialize the batch processor

        Args:
            scaffold_ivy_path: Path to template .ivy project with camera matrix,
                GCPs, and cross-section
            batch_config_csv: Optional path to CSV file with video list and WSE
            batch_configs: Optional list of BatchVideoConfig objects (if not using CSV)
        """
        self.scaffold_ivy_path = scaffold_ivy_path
        self.scaffold_data: Optional[ScaffoldData] = None
        self.batch_configs: List[BatchVideoConfig] = []
        self.temp_directory = None

        # Load scaffold
        self.scaffold_data = self._load_scaffold_project(scaffold_ivy_path)

        # Load batch configuration
        if batch_config_csv:
            self.batch_configs = self._load_batch_config_csv(batch_config_csv)
        elif batch_configs:
            self.batch_configs = batch_configs
        else:
            logging.warning("No batch configuration provided. Use load_batch_config_csv() or set batch_configs manually.")

    def _load_scaffold_project(self, scaffold_ivy_path: str) -> ScaffoldData:
        """Load scaffold .ivy project and extract reusable parameters

        Args:
            scaffold_ivy_path: Path to scaffold .ivy file

        Returns:
            ScaffoldData object with extracted parameters
        """
        logging.info(f"Loading scaffold project: {scaffold_ivy_path}")

        # Create temporary directory for extraction
        temp_dir = tempfile.mkdtemp(prefix="ivy_scaffold_")

        try:
            # Extract .ivy zip file
            with zipfile.ZipFile(scaffold_ivy_path, 'r') as zipf:
                zipf.extractall(temp_dir)

            # Load project_data.json
            json_path = os.path.join(temp_dir, 'project_data.json')
            with open(json_path, 'r') as f:
                project_dict = json.load(f)

            # Extract scaffold data
            scaffold = ScaffoldData()
            scaffold.original_project_dict = project_dict

            # Orthorectification
            scaffold.rectification_method = project_dict.get('rectification_method')
            scaffold.rectification_parameters = project_dict.get('rectification_parameters')
            scaffold.pixel_ground_scale_distance_m = project_dict.get('pixel_ground_scale_distance_m')

            # Extract camera matrix and extent for orthorectification
            if scaffold.rectification_parameters:
                camera_matrix = scaffold.rectification_parameters.get('camera_matrix')
                if camera_matrix:
                    scaffold.projection_matrix = np.array(camera_matrix)
                extent = scaffold.rectification_parameters.get('extent')
                if extent:
                    scaffold.orthorectification_extent = np.array(extent)

            # GCP table
            ortho_df_dict = project_dict.get('orthotable_dataframe')
            if ortho_df_dict:
                scaffold.orthotable_dataframe = pd.DataFrame.from_dict(ortho_df_dict)

            # Cross-section
            scaffold.bathymetry_ac3_filename = project_dict.get('bathymetry_ac3_filename')
            cross_section_line = project_dict.get('cross_section_line')
            if cross_section_line:
                scaffold.cross_section_line = deserialize_numpy_array(cross_section_line)
            scaffold.cross_section_start_bank = project_dict.get('cross_section_start_bank')
            scaffold.cross_section_top_width_m = project_dict.get('cross_section_top_width_m')
            scaffold.cross_section_hydraulic_radius_m = project_dict.get('cross_section_hydraulic_radius_m')
            scaffold.cross_section_length_pixels = project_dict.get('cross_section_length_pixels')
            scaffold.cross_section_line_exists = project_dict.get('cross_section_line_exists', False)
            scaffold.is_area_comp_loaded = project_dict.get('is_area_comp_loaded', False)

            # STIV parameters
            scaffold.stiv_search_line_length_m = project_dict.get('stiv_search_line_length_m')
            scaffold.stiv_phi_origin = project_dict.get('stiv_phi_origin', 145)
            scaffold.stiv_phi_range = project_dict.get('stiv_phi_range', 15)
            scaffold.stiv_dphi = project_dict.get('stiv_dphi', 1.0)
            scaffold.stiv_num_pixels = project_dict.get('stiv_num_pixels', 44)
            scaffold.stiv_gaussian_blur_sigma = project_dict.get('stiv_gaussian_blur_sigma', 0.0)
            scaffold.stiv_max_vel_threshold_mps = project_dict.get('stiv_max_vel_threshold_mps', 10.0)
            scaffold.process_step = project_dict.get('process_step', 'Process STIV Exhaustive')

            # Grid parameters
            scaffold.number_grid_points_along_xs_line = project_dict.get('number_grid_points_along_xs_line', 25)
            scaffold.horz_grid_size = project_dict.get('horz_grid_size', 50)
            scaffold.vert_grid_size = project_dict.get('vert_grid_size', 50)

            # Frame extraction defaults
            scaffold.extraction_frame_rate = project_dict.get('extraction_frame_rate')
            scaffold.extraction_frame_step = project_dict.get('extraction_frame_step', 1)
            scaffold.extraction_num_frames = project_dict.get('extraction_num_frames', 200)

            # Measurement template
            scaffold.measurement_info = project_dict.get('measurement_info', {})

            # Units
            scaffold.display_units = project_dict.get('display_units', 'English')
            scaffold.survey_units = project_dict.get('survey_units', {})

            # Misc
            scaffold.ffmpeg_cmd = project_dict.get('ffmpeg_cmd')
            scaffold.ffprobe_cmd = project_dict.get('ffprobe_cmd')

            # Mask polygons
            mask_polygons = project_dict.get('mask_polygons')
            if mask_polygons:
                scaffold.mask_polygons = [
                    deserialize_numpy_array(poly) if isinstance(poly, str) else poly
                    for poly in mask_polygons
                ]

            # Copy bathymetry file if it exists
            if scaffold.bathymetry_ac3_filename and os.path.exists(
                os.path.join(temp_dir, '5-discharge', 'cross_section_ac3.mat')
            ):
                # We'll need to copy this file to each output project
                scaffold.bathymetry_ac3_filename = os.path.join(
                    temp_dir, '5-discharge', 'cross_section_ac3.mat'
                )

            logging.info(f"Scaffold loaded successfully")
            logging.info(f"  Rectification method: {scaffold.rectification_method}")
            logging.info(f"  STIV method: {scaffold.process_step}")
            logging.info(f"  Cross-section loaded: {scaffold.is_area_comp_loaded}")
            logging.info(f"  GCP count: {len(scaffold.orthotable_dataframe) if scaffold.orthotable_dataframe is not None else 0}")

            # Store temp directory (we'll clean it up later)
            self.temp_directory = temp_dir

            return scaffold

        except Exception as e:
            # Clean up temp directory on error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            logging.error(f"Error loading scaffold project: {e}")
            logging.error(traceback.format_exc())
            raise

    def _load_batch_config_csv(self, csv_path: str) -> List[BatchVideoConfig]:
        """Load batch configuration from CSV file

        CSV format:
            video_path,water_surface_elevation,measurement_date,measurement_number,gage_height,start_time,end_time,comments,alpha

        Args:
            csv_path: Path to CSV configuration file

        Returns:
            List of BatchVideoConfig objects
        """
        logging.info(f"Loading batch configuration from: {csv_path}")

        # Get the directory containing the CSV file to resolve relative video paths
        csv_dir = os.path.dirname(os.path.abspath(csv_path))

        configs = []
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                try:
                    video_path = row['video_path'].strip()

                    # If video path is relative, resolve it relative to the CSV file location
                    if not os.path.isabs(video_path):
                        video_path = os.path.join(csv_dir, video_path)

                    # Normalize the path for the current OS
                    video_path = os.path.normpath(video_path)

                    # Parse alpha with fallback to 0.85 if not specified
                    alpha_str = row.get('alpha', '').strip()
                    alpha = float(alpha_str) if alpha_str else 0.85

                    config = BatchVideoConfig(
                        video_path=video_path,
                        water_surface_elevation=float(row['water_surface_elevation']),
                        measurement_date=row.get('measurement_date', '').strip(),
                        measurement_number=int(row.get('measurement_number', 0)),
                        gage_height=float(row.get('gage_height', 0.0)),
                        start_time=row.get('start_time', '').strip(),
                        end_time=row.get('end_time', '').strip(),
                        comments=row.get('comments', '').strip(),
                        alpha=alpha,
                    )
                    configs.append(config)
                    logging.debug(f"  Row {row_num}: {os.path.basename(config.video_path)} @ WSE={config.water_surface_elevation}m")

                except (KeyError, ValueError) as e:
                    logging.error(f"Error parsing CSV row {row_num}: {e}")
                    logging.error(f"  Row data: {row}")
                    continue

        logging.info(f"Loaded {len(configs)} video configurations")
        return configs

    def load_batch_config_csv(self, csv_path: str):
        """Load batch configuration from CSV file (public method)

        Args:
            csv_path: Path to CSV configuration file
        """
        self.batch_configs = self._load_batch_config_csv(csv_path)

    def process_batch(
        self,
        output_directory: str,
        progress_callback=None
    ) -> List[BatchResult]:
        """Process all videos in the batch

        Args:
            output_directory: Directory to save output .ivy files
            progress_callback: Optional callback function(current, total, message)

        Returns:
            List of BatchResult objects
        """
        logging.info(f"Starting batch processing of {len(self.batch_configs)} videos")
        logging.info(f"Output directory: {output_directory}")

        # Create output directory
        os.makedirs(output_directory, exist_ok=True)

        results = []

        for idx, config in enumerate(self.batch_configs, start=1):
            if progress_callback:
                progress_callback(idx, len(self.batch_configs),
                                f"Processing {os.path.basename(config.video_path)}")

            logging.info(f"\n{'='*60}")
            logging.info(f"Processing video {idx}/{len(self.batch_configs)}: {config.video_path}")
            logging.info(f"  WSE: {config.water_surface_elevation}m")

            # Create output path
            video_basename = Path(config.video_path).stem
            output_ivy_path = os.path.join(output_directory, f"{video_basename}.ivy")

            # Process single video
            result = self._process_single_video(
                config=config,
                output_ivy_path=output_ivy_path,
                progress_callback=lambda msg: progress_callback(idx, len(self.batch_configs), msg) if progress_callback else None
            )

            results.append(result)

            # Log result
            if result.processing_status == "success":
                logging.info(f"✓ Success: Q={result.total_discharge:.2f} {self.scaffold_data.survey_units.get('label_Q', 'units')}")
                logging.info(f"  Saved to: {output_ivy_path}")
            else:
                logging.error(f"✗ Failed: {result.error_message}")

        logging.info(f"\n{'='*60}")
        logging.info(f"Batch processing complete")

        # Summary
        successful = sum(1 for r in results if r.processing_status == "success")
        failed = sum(1 for r in results if r.processing_status == "failed")
        logging.info(f"  Successful: {successful}/{len(results)}")
        logging.info(f"  Failed: {failed}/{len(results)}")

        return results

    def _process_single_video(
        self,
        config: BatchVideoConfig,
        output_ivy_path: str,
        progress_callback=None
    ) -> BatchResult:
        """Process a single video (headless)

        This is the core processing function that processes a video through
        the complete workflow: frame extraction → STIV → discharge calculation.

        Args:
            config: BatchVideoConfig for this video
            output_ivy_path: Where to save the output .ivy file
            progress_callback: Optional callback for progress updates

        Returns:
            BatchResult object
        """
        # Create result object
        result = BatchResult(
            video_filename=os.path.basename(config.video_path),
            video_path=config.video_path,
            measurement_date=config.measurement_date,
            measurement_number=config.measurement_number,
            wse_m=config.water_surface_elevation,
            gage_height=config.gage_height,
            output_ivy_path=output_ivy_path,
            processing_start_time=datetime.now()
        )

        temp_frame_dir = None

        try:
            # Convert WSE from feet to meters if scaffold uses English units
            wse_feet = config.gage_height if config.gage_height else config.water_surface_elevation
            if self.scaffold_data.display_units == "English":
                wse_m = wse_feet / 3.28084  # Convert feet to meters
                logging.info(f"Converting WSE from {wse_feet:.3f} ft to {wse_m:.3f} m (English units)")
            else:
                wse_m = wse_feet
                logging.info(f"Using WSE: {wse_m:.3f} m (Metric units)")

            if progress_callback:
                progress_callback("Extracting frames...")

            # Step 1: Extract frames from video
            logging.info("Step 1/6: Extracting frames from video")
            temp_frame_dir = tempfile.mkdtemp(prefix="ivy_batch_frames_")

            frame_files, extraction_metadata = extract_frames_headless(
                video_path=config.video_path,
                output_directory=temp_frame_dir,
                start_time=config.start_time if config.start_time else None,
                end_time=config.end_time if config.end_time else None,
                frame_step=self.scaffold_data.extraction_frame_step,
                frame_rate=self.scaffold_data.extraction_frame_rate,
                max_frames=self.scaffold_data.extraction_num_frames,
                ffmpeg_cmd=self.scaffold_data.ffmpeg_cmd or "ffmpeg"
            )

            if len(frame_files) == 0:
                raise ValueError("No frames extracted from video")

            if progress_callback:
                progress_callback(f"Extracted {len(frame_files)} frames")

            # Step 1b: Orthorectify frames (if using camera matrix rectification)
            rectified_files = []
            if self.scaffold_data.rectification_method == "camera matrix":
                if progress_callback:
                    progress_callback("Orthorectifying frames...")

                logging.info("Step 1b/6: Orthorectifying frames")

                from image_velocimetry_tools.batch_processing_helpers import orthorectify_frames_headless
                rectified_files = orthorectify_frames_headless(
                    frame_directory=temp_frame_dir,
                    projection_matrix=self.scaffold_data.projection_matrix,
                    water_surface_elevation_m=wse_m,
                    extent=self.scaffold_data.orthorectification_extent
                )

                if progress_callback:
                    progress_callback(f"Orthorectified {len(rectified_files)} frames")

            # Step 2: Generate cross-section grid
            logging.info("Step 2/6: Generating cross-section grid")
            grid_pixel = generate_cross_section_grid(
                cross_section_line=self.scaffold_data.cross_section_line,
                num_points=self.scaffold_data.number_grid_points_along_xs_line,
                pixel_ground_scale_distance_m=self.scaffold_data.pixel_ground_scale_distance_m
            )

            if progress_callback:
                progress_callback("Running STIV processing...")

            # Step 3: Run STIV processing
            logging.info("Step 3/6: Running STIV velocity calculation")
            stiv_params = {
                'phi_origin': self.scaffold_data.stiv_phi_origin,
                'phi_range': self.scaffold_data.stiv_phi_range,
                'dphi': self.scaffold_data.stiv_dphi,
                'num_pixels': self.scaffold_data.stiv_num_pixels,
                'gaussian_blur_sigma': self.scaffold_data.stiv_gaussian_blur_sigma,
                'max_vel_threshold_mps': self.scaffold_data.stiv_max_vel_threshold_mps,
            }

            # Use rectified frames if available, otherwise use original frames
            frames_for_stiv = rectified_files if rectified_files else frame_files
            logging.info(f"Running STIV on {len(frames_for_stiv)} {'rectified' if rectified_files else 'original'} frames")

            magnitudes_mps, directions_deg = run_stiv_headless(
                frame_files=frames_for_stiv,
                grid=grid_pixel,
                stiv_params=stiv_params,
                pixel_gsd=self.scaffold_data.pixel_ground_scale_distance_m,
                timestep_ms=extraction_metadata['timestep_ms']
            )

            if progress_callback:
                progress_callback("Calculating discharge...")

            # Step 4: Load cross-section and calculate discharge
            logging.info("Step 4/6: Loading cross-section and calculating discharge")

            # Load AreaComp cross-section
            xs_survey = AreaSurvey()
            if self.scaffold_data.bathymetry_ac3_filename and os.path.exists(self.scaffold_data.bathymetry_ac3_filename):
                xs_survey.load_areacomp(self.scaffold_data.bathymetry_ac3_filename)
            else:
                raise ValueError("Cross-section bathymetry file not found in scaffold")

            # Calculate discharge (using wse_m converted earlier and alpha from config)
            logging.info(f"Using alpha = {config.alpha:.3f} for surface-to-average velocity conversion")
            discharge_results, discharge_summary = calculate_discharge_headless(
                magnitudes_mps=magnitudes_mps,
                directions_deg=directions_deg,
                grid_pixel=grid_pixel,
                xs_survey=xs_survey,
                cross_section_line=self.scaffold_data.cross_section_line,
                water_surface_elevation_m=wse_m,
                cross_section_start_bank=self.scaffold_data.cross_section_start_bank,
                alpha=config.alpha
            )

            if progress_callback:
                progress_callback("Calculating uncertainty...")

            # Step 5: Calculate uncertainty
            logging.info("Step 5/6: Calculating uncertainty")
            num_gcp = len(self.scaffold_data.orthotable_dataframe) if self.scaffold_data.orthotable_dataframe is not None else 0

            uncertainty_results = calculate_uncertainty_headless(
                discharge_summary=discharge_summary,
                discharge_results=discharge_results,
                survey_units=self.scaffold_data.survey_units,
                rectification_method=self.scaffold_data.rectification_method,
                num_gcp=num_gcp
            )

            if progress_callback:
                progress_callback("Saving project file...")

            # Step 6: Save .ivy project
            logging.info("Step 6/6: Saving .ivy project")
            self._save_ivy_project(
                output_path=output_ivy_path,
                config=config,
                frame_files=frame_files,
                rectified_files=rectified_files,
                extraction_metadata=extraction_metadata,
                grid_pixel=grid_pixel,
                magnitudes_mps=magnitudes_mps,
                directions_deg=directions_deg,
                discharge_results=discharge_results,
                discharge_summary=discharge_summary,
                uncertainty_results=uncertainty_results,
                temp_frame_dir=temp_frame_dir,
                wse_m=wse_m
            )

            # Update result with success data
            result.processing_status = "success"
            result.total_discharge = discharge_summary["total_discharge"]
            result.total_area = discharge_summary["total_area"]
            result.iso_uncertainty = discharge_summary.get("ISO_uncertainty")
            result.ive_uncertainty = discharge_summary.get("IVE_uncertainty")
            result.cross_section_width = self.scaffold_data.cross_section_top_width_m
            result.hydraulic_radius = self.scaffold_data.cross_section_hydraulic_radius_m

            # Count stations
            result.num_stations_total = len(discharge_results)
            result.num_stations_used = sum(1 for r in discharge_results.values() if r["Status"] == "Used")

            # Calculate mean velocity
            if result.total_area and result.total_area > 0:
                result.mean_velocity = result.total_discharge / result.total_area

            logging.info(f"✓ Processing complete: Q={result.total_discharge:.2f} m³/s")

        except Exception as e:
            logging.error(f"Error processing video: {e}")
            logging.error(traceback.format_exc())
            result.processing_status = "failed"
            result.error_message = str(e)

        finally:
            # Clean up temporary frame directory
            if temp_frame_dir and os.path.exists(temp_frame_dir):
                try:
                    shutil.rmtree(temp_frame_dir)
                    logging.debug(f"Cleaned up temp frames: {temp_frame_dir}")
                except Exception as e:
                    logging.warning(f"Failed to clean up temp directory: {e}")

            result.processing_end_time = datetime.now()

        return result

    def _save_ivy_project(
        self,
        output_path: str,
        config: BatchVideoConfig,
        frame_files: List[str],
        rectified_files: List[str],
        extraction_metadata: Dict,
        grid_pixel: np.ndarray,
        magnitudes_mps: np.ndarray,
        directions_deg: np.ndarray,
        discharge_results: Dict,
        discharge_summary: Dict,
        uncertainty_results: Dict,
        temp_frame_dir: str,
        wse_m: float
    ):
        """Save processed results as an .ivy project file

        Args:
            output_path: Path for output .ivy file
            config: Batch video configuration
            frame_files: List of extracted frame file paths
            extraction_metadata: Frame extraction metadata
            grid_pixel: Grid points in pixel coordinates
            magnitudes_mps: STIV velocity magnitudes
            directions_deg: STIV flow directions
            discharge_results: Discharge results dict (stations)
            discharge_summary: Discharge summary dict
            uncertainty_results: Uncertainty calculation results
            temp_frame_dir: Temporary directory containing frames
        """
        logging.debug(f"Saving .ivy project to: {output_path}")

        # Create temporary project directory
        project_temp_dir = tempfile.mkdtemp(prefix="ivy_batch_project_")

        try:
            # Create project directory structure
            swap_image_dir = os.path.join(project_temp_dir, "1-images")
            swap_ortho_dir = os.path.join(project_temp_dir, "2-orthorectification")
            swap_grids_dir = os.path.join(project_temp_dir, "3-grids")
            swap_vel_dir = os.path.join(project_temp_dir, "4-velocities")
            swap_discharge_dir = os.path.join(project_temp_dir, "5-discharge")
            swap_qaqc_dir = os.path.join(project_temp_dir, "6-qaqc")

            for dir_path in [swap_image_dir, swap_ortho_dir, swap_grids_dir,
                           swap_vel_dir, swap_discharge_dir, swap_qaqc_dir]:
                os.makedirs(dir_path, exist_ok=True)

            # Copy extracted frames to project
            for frame_file in frame_files:
                shutil.copy(frame_file, swap_image_dir)

            # Copy orthorectified frames to project (if they exist)
            if rectified_files and len(rectified_files) > 0:
                for rectified_file in rectified_files:
                    shutil.copy(rectified_file, swap_image_dir)
                logging.info(f"Copied {len(rectified_files)} orthorectified frames to project")

            # Copy bathymetry file if it exists
            if self.scaffold_data.bathymetry_ac3_filename and os.path.exists(
                self.scaffold_data.bathymetry_ac3_filename
            ):
                shutil.copy(
                    self.scaffold_data.bathymetry_ac3_filename,
                    os.path.join(swap_discharge_dir, "cross_section_ac3.mat")
                )

            # Build project_dict from scaffold + new data
            project_dict = self.scaffold_data.original_project_dict.copy()

            # Update with new video-specific data
            project_dict["__user__"] = os.environ.get("USERNAME", "batch_user")
            project_dict["uuid"] = str(datetime.now().timestamp())

            # Video info
            project_dict["video_file_name"] = config.video_path
            project_dict["extraction_num_frames"] = extraction_metadata["num_frames"]
            project_dict["extraction_frame_rate"] = extraction_metadata["frame_rate"]
            project_dict["extraction_frame_step"] = extraction_metadata["frame_step"]
            project_dict["extraction_timestep_ms"] = extraction_metadata["timestep_ms"]
            project_dict["is_frames_extracted"] = True
            project_dict["is_video_loaded"] = True

            # Measurement info
            project_dict["measurement_info"] = {
                "station_name": self.scaffold_data.measurement_info.get("station_name", ""),
                "station_number": self.scaffold_data.measurement_info.get("station_number", ""),
                "meas_date": config.measurement_date,
                "meas_number": config.measurement_number,
                "gage_ht": config.gage_height,
                "start_time": extraction_metadata["start_time"],
                "end_time": extraction_metadata["end_time"],
                "comments": config.comments,
            }

            # Water surface elevation (THIS IS CRITICAL - use converted wse_m)
            # wse_m has already been converted from feet to meters if needed
            project_dict["ortho_rectified_wse_m"] = wse_m
            project_dict["water_surface_elevation"] = wse_m

            # Update rectification parameters with new WSE
            if project_dict.get("rectification_parameters"):
                project_dict["rectification_parameters"]["water_surface_elev"] = wse_m

            # Image browser sequence
            project_dict["imagebrowser_sequence"] = [
                os.path.basename(f) for f in frame_files
            ]
            project_dict["imagebrowser_image_path"] = os.path.basename(frame_files[0]) if frame_files else ""

            # Grid and results
            project_dict["results_grid"] = serialize_numpy_array(grid_pixel)
            project_dict["is_cross_section_grid"] = True

            # STIV results
            project_dict["stiv_magnitudes"] = serialize_numpy_array(magnitudes_mps)
            project_dict["stiv_directions"] = serialize_numpy_array(directions_deg)
            project_dict["stiv_magnitude_normals"] = serialize_numpy_array(magnitudes_mps)
            project_dict["stiv_exists"] = True
            project_dict["is_stis"] = True

            # Discharge results
            project_dict["discharge_results"] = discharge_results
            project_dict["discharge_summary"] = discharge_summary

            # Uncertainty
            if uncertainty_results:
                project_dict["u_iso"] = uncertainty_results.get("u_iso", {})
                project_dict["u_iso_contribution"] = uncertainty_results.get("u_iso_contribution", {})
                project_dict["u_ive"] = uncertainty_results.get("u_ive", {})
                project_dict["u_ive_contribution"] = uncertainty_results.get("u_ive_contribution", {})

            # Swap directories
            project_dict["swap_directory"] = project_temp_dir
            project_dict["swap_image_directory"] = swap_image_dir
            project_dict["swap_orthorectification_directory"] = swap_ortho_dir
            project_dict["swap_grids_directory"] = swap_grids_dir
            project_dict["swap_velocities_directory"] = swap_vel_dir
            project_dict["swap_discharge_directory"] = swap_discharge_dir
            project_dict["swap_qaqc_directory"] = swap_qaqc_dir

            # Convert any numpy arrays in rectification_parameters to lists
            if "rectification_parameters" in project_dict:
                project_dict["rectification_parameters"] = dict_arrays_to_list(
                    project_dict["rectification_parameters"]
                )

            # Save project_data.json
            project_json_path = os.path.join(project_temp_dir, "project_data.json")
            with open(project_json_path, 'w') as f:
                json.dump(project_dict, f, indent=2)

            # Create .ivy zip archive
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from project temp directory
                for root, dirs, files in os.walk(project_temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, project_temp_dir)
                        zipf.write(file_path, arcname)

            logging.info(f"Saved .ivy project: {output_path}")

        finally:
            # Clean up project temp directory
            if os.path.exists(project_temp_dir):
                shutil.rmtree(project_temp_dir, ignore_errors=True)

    def export_results_csv(
        self,
        results: List[BatchResult],
        output_csv_path: str
    ):
        """Export batch results to CSV summary file

        Args:
            results: List of BatchResult objects
            output_csv_path: Path for output CSV file
        """
        logging.info(f"Exporting results to CSV: {output_csv_path}")

        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'video_filename',
                'measurement_date',
                'measurement_number',
                'wse_m',
                'gage_height',
                'total_discharge',
                'total_area',
                'iso_uncertainty',
                'ive_uncertainty',
                'cross_section_width',
                'hydraulic_radius',
                'mean_velocity',
                'num_stations_used',
                'num_stations_total',
                'processing_status',
                'processing_duration_sec',
                'error_message',
                'output_ivy_path',
            ])

            # Data rows
            for result in results:
                writer.writerow([
                    result.video_filename,
                    result.measurement_date,
                    result.measurement_number,
                    f"{result.wse_m:.4f}",
                    f"{result.gage_height:.4f}",
                    f"{result.total_discharge:.4f}" if result.total_discharge is not None else "",
                    f"{result.total_area:.4f}" if result.total_area is not None else "",
                    f"{result.iso_uncertainty:.4f}" if result.iso_uncertainty is not None else "",
                    f"{result.ive_uncertainty:.4f}" if result.ive_uncertainty is not None else "",
                    f"{result.cross_section_width:.4f}" if result.cross_section_width is not None else "",
                    f"{result.hydraulic_radius:.4f}" if result.hydraulic_radius is not None else "",
                    f"{result.mean_velocity:.4f}" if result.mean_velocity is not None else "",
                    result.num_stations_used,
                    result.num_stations_total,
                    result.processing_status,
                    f"{result.processing_duration_seconds:.2f}" if result.processing_duration_seconds else "",
                    result.error_message,
                    result.output_ivy_path,
                ])

        logging.info(f"CSV export complete: {len(results)} results")

    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_directory and os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory, ignore_errors=True)
            logging.debug(f"Cleaned up temp directory: {self.temp_directory}")

    def __del__(self):
        """Cleanup on object destruction"""
        self.cleanup()


def create_batch_config_template(output_path: str):
    """Create a template CSV file for batch configuration

    Args:
        output_path: Path for the template CSV file
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'video_path',
            'water_surface_elevation',
            'measurement_date',
            'measurement_number',
            'gage_height',
            'start_time',
            'end_time',
            'comments'
        ])
        writer.writerow([
            '/path/to/video1.mp4',
            '2.20',
            '2025-04-04',
            '1',
            '2.15',
            '00:00:00',
            '00:00:10',
            'High flow event'
        ])
        writer.writerow([
            '/path/to/video2.mp4',
            '1.85',
            '2025-04-05',
            '2',
            '1.80',
            '',
            '',
            'Recession'
        ])

    logging.info(f"Created batch config template: {output_path}")
