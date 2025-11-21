"""Python API for IVyTools batch video processing.

This module provides a high-level Python API for processing river videos
in batch mode using the IVyTools workflow:

1. Extract frames from video
2. Rectify frames to world coordinates
3. Compute surface velocities using STIV
4. Calculate discharge using velocity-area method

The API is designed for programmatic access from Python scripts, Jupyter
notebooks, or other Python applications.

Example usage:

    # Process a single video
    from image_velocimetry_tools.api import process_video

    result = process_video(
        scaffold_path="templates/scaffold.ivy",
        video_path="videos/river_20230615.mp4",
        water_surface_elevation=318.5,
        output_directory="results/",
        alpha=0.85,
        start_time=15.0,
        end_time=20.0,
    )

    print(f"Discharge: {result.total_discharge:.2f} m³/s")
    print(f"Mean velocity: {result.mean_velocity:.2f} m/s")

    # Process batch from CSV
    from image_velocimetry_tools.api import process_batch_csv

    batch_result = process_batch_csv(
        scaffold_path="templates/scaffold.ivy",
        batch_csv_path="batch_config.csv",
        output_directory="batch_results/",
    )

    print(f"Processed: {batch_result.successful}/{batch_result.total_videos}")
    print(f"Discharge summary: {batch_result.get_discharge_summary()}")
"""

import os
import csv
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List

from image_velocimetry_tools.batch import (
    BatchOrchestrator,
    ScaffoldConfig,
    VideoConfig,
    BatchVideoConfig,
    ProcessingResult,
    BatchResult,
)
from image_velocimetry_tools.services.project_service import ProjectService


def load_scaffold(
    scaffold_path: str,
    temp_dir: Optional[str] = None,
) -> ScaffoldConfig:
    """Load scaffold configuration from .ivy project file.

    The scaffold is a template .ivy project containing camera calibration,
    rectification parameters, STIV search parameters, cross-section bathymetry,
    and grid configuration. This configuration is shared across multiple videos
    in batch processing.

    Args:
        scaffold_path: Path to scaffold .ivy project file
        temp_dir: Optional temporary directory for extraction (auto-created if None)

    Returns:
        ScaffoldConfig dataclass with loaded configuration

    Raises:
        FileNotFoundError: If scaffold file doesn't exist
        ValueError: If scaffold is invalid or missing required fields

    Example:
        >>> scaffold = load_scaffold("templates/boneyard_scaffold.ivy")
        >>> print(f"Rectification: {scaffold.rectification_method}")
        >>> print(f"STIV params: {scaffold.stiv_params}")
    """
    if not os.path.exists(scaffold_path):
        raise FileNotFoundError(f"Scaffold file not found: {scaffold_path}")

    # Load using ProjectService
    project_service = ProjectService()
    config_dict = project_service.load_scaffold_configuration(
        scaffold_path,
        temp_dir=temp_dir
    )

    # Convert to ScaffoldConfig dataclass
    scaffold_config = ScaffoldConfig(
        scaffold_path=scaffold_path,
        project_dict=config_dict["project_dict"],
        swap_directory=config_dict["swap_directory"],
        rectification_method=config_dict["rectification_method"],
        rectification_params=config_dict["rectification_params"],
        stiv_params=config_dict["stiv_params"],
        cross_section_data=config_dict["cross_section_data"],
        grid_params=config_dict["grid_params"],
        display_units=config_dict["display_units"],
        temp_cleanup_required=config_dict["temp_cleanup_required"],
    )

    return scaffold_config


def process_video(
    scaffold_path: str,
    video_path: str,
    water_surface_elevation: float,
    output_directory: str,
    measurement_date: str = "",
    alpha: float = 0.85,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    frame_step: int = 1,
    max_frames: Optional[int] = None,
    comments: str = "",
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cleanup_temp_files: bool = True,
) -> ProcessingResult:
    """Process a single video through complete workflow.

    This is the main entry point for processing a single video. It loads the
    scaffold configuration, extracts and rectifies frames, computes velocities
    using STIV, calculates discharge, and saves results.

    Args:
        scaffold_path: Path to scaffold .ivy template project
        video_path: Path to input video file
        water_surface_elevation: Water surface elevation in meters
        output_directory: Directory for output files
        measurement_date: Date of measurement (YYYY-MM-DD format, default: "")
        alpha: Alpha coefficient for velocity adjustment (default: 0.85)
        start_time: Start time in seconds (None = start of video)
        end_time: End time in seconds (None = end of video)
        frame_step: Extract every Nth frame (default: 1 = all frames)
        max_frames: Maximum number of frames to extract (None = no limit)
        comments: Optional comments about this measurement
        progress_callback: Optional callback(percent, message) for progress updates
        cleanup_temp_files: Whether to delete temporary files (default: True)

    Returns:
        ProcessingResult with outputs, metrics, and any errors

    Raises:
        FileNotFoundError: If scaffold or video file doesn't exist
        ValueError: If parameters are invalid

    Example:
        >>> result = process_video(
        ...     scaffold_path="scaffold.ivy",
        ...     video_path="river.mp4",
        ...     water_surface_elevation=318.5,
        ...     output_directory="results/",
        ...     alpha=0.85,
        ...     start_time=15.0,
        ...     end_time=20.0,
        ... )
        >>> if result.success:
        ...     print(f"Q = {result.total_discharge:.2f} m³/s")
        ... else:
        ...     print(f"Failed: {result.error_message}")
    """
    # Load scaffold
    scaffold_config = load_scaffold(scaffold_path)

    # Create video config
    video_config = VideoConfig(
        video_path=video_path,
        water_surface_elevation=water_surface_elevation,
        measurement_date=measurement_date,
        alpha=alpha,
        start_time=start_time,
        end_time=end_time,
        frame_step=frame_step,
        max_frames=max_frames,
        comments=comments,
    )

    # Create combined config
    combined_config = BatchVideoConfig(
        scaffold=scaffold_config,
        video=video_config,
    )

    # Create orchestrator and process
    orchestrator = BatchOrchestrator()

    result = orchestrator.process_video(
        config=combined_config,
        output_directory=output_directory,
        progress_callback=progress_callback,
        cleanup_temp_files=cleanup_temp_files,
    )

    return result


def process_batch(
    scaffold_path: str,
    video_configs: List[Dict[str, Any]],
    output_directory: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cleanup_temp_files: bool = True,
) -> BatchResult:
    """Process multiple videos in batch mode.

    This processes multiple videos using a shared scaffold configuration.
    Each video can have different water surface elevation, alpha coefficient,
    and time window.

    Args:
        scaffold_path: Path to scaffold .ivy template project
        video_configs: List of video configuration dictionaries, each containing:
            - video_path: Path to video file (required)
            - water_surface_elevation: WSE in meters (required)
            - measurement_date: Date string (optional, default: "")
            - alpha: Alpha coefficient (optional, default: 0.85)
            - start_time: Start time in seconds (optional)
            - end_time: End time in seconds (optional)
            - frame_step: Frame step (optional, default: 1)
            - max_frames: Max frames (optional)
            - comments: Comments string (optional, default: "")
        output_directory: Root output directory for batch
        progress_callback: Optional callback(percent, message) for progress updates
        cleanup_temp_files: Whether to delete temporary files (default: True)

    Returns:
        BatchResult with aggregated results and statistics

    Raises:
        FileNotFoundError: If scaffold file doesn't exist
        ValueError: If video configs are invalid

    Example:
        >>> configs = [
        ...     {
        ...         "video_path": "video1.mp4",
        ...         "water_surface_elevation": 318.5,
        ...         "alpha": 0.85,
        ...     },
        ...     {
        ...         "video_path": "video2.mp4",
        ...         "water_surface_elevation": 318.7,
        ...         "alpha": 0.85,
        ...     },
        ... ]
        >>> batch_result = process_batch(
        ...     scaffold_path="scaffold.ivy",
        ...     video_configs=configs,
        ...     output_directory="batch_results/",
        ... )
        >>> print(f"{batch_result.successful}/{batch_result.total_videos} successful")
    """
    # Load scaffold
    scaffold_config = load_scaffold(scaffold_path)

    # Convert config dictionaries to VideoConfig objects
    video_config_objects = []
    for config in video_configs:
        video_config = VideoConfig(
            video_path=config["video_path"],
            water_surface_elevation=config["water_surface_elevation"],
            measurement_date=config.get("measurement_date", ""),
            alpha=config.get("alpha", 0.85),
            start_time=config.get("start_time"),
            end_time=config.get("end_time"),
            frame_step=config.get("frame_step", 1),
            max_frames=config.get("max_frames"),
            comments=config.get("comments", ""),
        )
        video_config_objects.append(video_config)

    # Create orchestrator and process batch
    orchestrator = BatchOrchestrator()

    batch_result = orchestrator.process_batch(
        scaffold_config=scaffold_config,
        video_configs=video_config_objects,
        output_directory=output_directory,
        progress_callback=progress_callback,
        cleanup_temp_files=cleanup_temp_files,
    )

    return batch_result


def process_batch_csv(
    scaffold_path: str,
    batch_csv_path: str,
    output_directory: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cleanup_temp_files: bool = True,
) -> BatchResult:
    """Process batch from CSV configuration file.

    This is a convenience function for processing videos defined in a CSV file.
    The CSV should have the following columns:
    - video_path: Path to video file (required)
    - water_surface_elevation: WSE in meters (required)
    - measurement_date: Date string (optional)
    - alpha: Alpha coefficient (optional, default: 0.85)
    - start_time: Start time in seconds (optional)
    - end_time: End time in seconds (optional)
    - frame_step: Frame step (optional, default: 1)
    - max_frames: Max frames (optional)
    - comments: Comments string (optional)

    Args:
        scaffold_path: Path to scaffold .ivy template project
        batch_csv_path: Path to batch configuration CSV file
        output_directory: Root output directory for batch
        progress_callback: Optional callback(percent, message) for progress updates
        cleanup_temp_files: Whether to delete temporary files (default: True)

    Returns:
        BatchResult with aggregated results and statistics

    Raises:
        FileNotFoundError: If scaffold or CSV file doesn't exist
        ValueError: If CSV format is invalid

    Example:
        >>> batch_result = process_batch_csv(
        ...     scaffold_path="scaffold.ivy",
        ...     batch_csv_path="batch_config.csv",
        ...     output_directory="batch_results/",
        ... )
        >>> print(f"Processed: {batch_result.successful}/{batch_result.total_videos}")
        >>> summary = batch_result.get_discharge_summary()
        >>> print(f"Mean discharge: {summary['mean']:.2f} m³/s")
    """
    if not os.path.exists(batch_csv_path):
        raise FileNotFoundError(f"Batch CSV not found: {batch_csv_path}")

    # Get CSV directory for resolving relative video paths
    csv_dir = os.path.dirname(os.path.abspath(batch_csv_path))

    # Parse CSV into video configs
    video_configs = []

    with open(batch_csv_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Required fields
            if "video_path" not in row or "water_surface_elevation" not in row:
                raise ValueError(
                    "CSV must have 'video_path' and 'water_surface_elevation' columns"
                )

            # Resolve video path relative to CSV directory if not absolute
            video_path = row["video_path"]
            if not os.path.isabs(video_path):
                # Try relative to CSV directory first
                video_path = os.path.join(csv_dir, video_path)

            # Parse optional time fields (support both float and HH:MM:SS format)
            start_time = None
            end_time = None

            if "start_time" in row and row["start_time"]:
                try:
                    start_time = float(row["start_time"])
                except ValueError:
                    # TODO: Parse HH:MM:SS format
                    start_time = None

            if "end_time" in row and row["end_time"]:
                try:
                    end_time = float(row["end_time"])
                except ValueError:
                    # TODO: Parse HH:MM:SS format
                    end_time = None

            # Build config dict
            config = {
                "video_path": video_path,
                "water_surface_elevation": float(row["water_surface_elevation"]),
                "measurement_date": row.get("measurement_date", ""),
                "alpha": float(row.get("alpha", 0.85)),
                "start_time": start_time,
                "end_time": end_time,
                "frame_step": int(row.get("frame_step", 1)),
                "max_frames": int(row["max_frames"]) if row.get("max_frames") else None,
                "comments": row.get("comments", ""),
            }

            video_configs.append(config)

    if not video_configs:
        raise ValueError(f"No video configurations found in CSV: {batch_csv_path}")

    # Process batch
    return process_batch(
        scaffold_path=scaffold_path,
        video_configs=video_configs,
        output_directory=output_directory,
        progress_callback=progress_callback,
        cleanup_temp_files=cleanup_temp_files,
    )


# Convenience exports
__all__ = [
    "process_video",
    "process_batch",
    "process_batch_csv",
    "load_scaffold",
    "ProcessingResult",
    "BatchResult",
    "ScaffoldConfig",
    "VideoConfig",
]
