"""Dataclasses for batch processing configuration and results.

These dataclasses provide type-safe configuration and result structures
for the batch processing workflow.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np


@dataclass
class ScaffoldConfig:
    """Configuration loaded from scaffold .ivy project.

    This contains all the template configuration that applies to multiple
    videos: camera calibration, rectification parameters, STIV search
    parameters, cross-section bathymetry, and grid points.

    Attributes:
        scaffold_path: Path to scaffold .ivy file
        project_dict: Full project dictionary from scaffold
        swap_directory: Temporary directory for extracted scaffold files
        rectification_method: Rectification method ("homography", "camera matrix", "scale")
        rectification_params: Method-specific rectification parameters
        stiv_params: STIV search parameters (phi_origin, phi_range, dphi, num_pixels)
        cross_section_data: Cross-section bathymetry and line data
        grid_params: Grid generation parameters (num_points, extent, etc.)
        display_units: Units for display ("m", "ft")
        temp_cleanup_required: Whether temp directory needs cleanup
    """
    scaffold_path: str
    project_dict: Dict[str, Any]
    swap_directory: str
    rectification_method: str
    rectification_params: Dict[str, Any]
    stiv_params: Dict[str, float]
    cross_section_data: Dict[str, Any]
    grid_params: Dict[str, Any]
    display_units: str
    temp_cleanup_required: bool

    def __post_init__(self):
        """Validate required fields."""
        required_stiv = ["phi_origin", "phi_range", "dphi", "num_pixels"]
        for key in required_stiv:
            if key not in self.stiv_params:
                raise ValueError(f"Missing required STIV parameter: {key}")

        if self.rectification_method not in ["homography", "camera matrix", "scale"]:
            raise ValueError(f"Invalid rectification method: {self.rectification_method}")


@dataclass
class VideoConfig:
    """Configuration for processing a single video.

    This contains video-specific parameters that vary between videos in
    a batch: video path, water surface elevation, measurement metadata,
    alpha coefficient, and time window.

    Attributes:
        video_path: Path to video file
        water_surface_elevation: Water surface elevation in meters
        measurement_date: Date of measurement (YYYY-MM-DD format)
        alpha: Alpha coefficient for velocity adjustment (default: 0.85)
        start_time: Start time in seconds (optional, default: start of video)
        end_time: End time in seconds (optional, default: end of video)
        frame_step: Extract every Nth frame (default: 1 = all frames)
        max_frames: Maximum number of frames to extract (optional)
        comments: Optional comments about this video
    """
    video_path: str
    water_surface_elevation: float
    measurement_date: str
    alpha: float = 0.85
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    frame_step: int = 1
    max_frames: Optional[int] = None
    comments: str = ""

    def __post_init__(self):
        """Validate video configuration."""
        if not Path(self.video_path).exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError(f"Alpha must be in range (0, 1], got {self.alpha}")

        if self.frame_step < 1:
            raise ValueError(f"Frame step must be >= 1, got {self.frame_step}")

        if self.start_time is not None and self.end_time is not None:
            if self.end_time <= self.start_time:
                raise ValueError(
                    f"End time ({self.end_time}) must be > start time ({self.start_time})"
                )


@dataclass
class BatchVideoConfig:
    """Combined configuration for batch processing a single video.

    This combines the scaffold template configuration with video-specific
    configuration for a complete processing specification.

    Attributes:
        scaffold: Scaffold template configuration
        video: Video-specific configuration
    """
    scaffold: ScaffoldConfig
    video: VideoConfig


@dataclass
class ProcessingResult:
    """Results from processing a single video.

    Contains all outputs, metrics, and error information from processing
    one video through the complete workflow.

    Attributes:
        video_path: Path to input video file
        success: Whether processing completed successfully
        error_message: Error message if processing failed (None if success)
        error_stage: Stage where error occurred ("frames", "rectify", "stiv", "discharge")

        # Frame extraction results
        num_frames_extracted: Number of frames extracted
        frames_directory: Directory containing extracted frames
        frame_files: List of frame file paths

        # Video metadata
        video_metadata: Video metadata (width, height, fps, duration)
        timestep_seconds: Time between frames in seconds

        # Rectification results
        rectified_frames_directory: Directory containing rectified frames
        rectified_frame_files: List of rectified frame file paths
        pixel_gsd: Ground sample distance in meters/pixel

        # STIV results
        stiv_magnitudes: Velocity magnitudes in m/s (2D array)
        stiv_directions: Velocity directions in degrees (2D array)

        # Discharge results
        total_discharge: Total discharge in m³/s
        total_area: Total cross-sectional area in m²
        mean_velocity: Mean velocity in m/s
        discharge_dataframe: Discharge calculation dataframe
        discharge_uncertainty: Uncertainty analysis dictionary

        # Output files
        output_project_path: Path to saved .ivy project file
        output_csv_path: Path to discharge CSV file

        # Processing metadata
        processing_time_seconds: Total processing time
    """
    video_path: str
    success: bool
    error_message: Optional[str] = None
    error_stage: Optional[str] = None

    # Frame extraction
    num_frames_extracted: int = 0
    frames_directory: str = ""
    frame_files: List[str] = field(default_factory=list)

    # Video metadata
    video_metadata: Dict[str, Any] = field(default_factory=dict)
    timestep_seconds: float = 0.0

    # Rectification
    rectified_frames_directory: str = ""
    rectified_frame_files: List[str] = field(default_factory=list)
    pixel_gsd: float = 0.0

    # STIV
    stiv_magnitudes: Optional[np.ndarray] = None
    stiv_directions: Optional[np.ndarray] = None

    # Discharge
    total_discharge: float = 0.0
    total_area: float = 0.0
    mean_velocity: float = 0.0
    discharge_dataframe: Optional[Any] = None  # pandas DataFrame
    discharge_uncertainty: Dict[str, float] = field(default_factory=dict)

    # Output files
    output_project_path: str = ""
    output_csv_path: str = ""

    # Metadata
    processing_time_seconds: float = 0.0

    def __str__(self) -> str:
        """Human-readable summary of results."""
        if not self.success:
            return f"FAILED: {self.error_stage} - {self.error_message}"

        return (
            f"SUCCESS: Q={self.total_discharge:.4f} m³/s, "
            f"A={self.total_area:.4f} m², "
            f"V={self.mean_velocity:.4f} m/s "
            f"({self.processing_time_seconds:.1f}s)"
        )


@dataclass
class BatchResult:
    """Results from batch processing multiple videos.

    Contains aggregated results, statistics, and outputs from processing
    multiple videos in a batch.

    Attributes:
        total_videos: Total number of videos in batch
        successful: Number of videos processed successfully
        failed: Number of videos that failed
        video_results: List of ProcessingResult for each video
        output_directory: Root output directory for batch
        batch_csv_path: Path to aggregated batch results CSV
        processing_time_seconds: Total batch processing time
    """
    total_videos: int
    successful: int
    failed: int
    video_results: List[ProcessingResult] = field(default_factory=list)
    output_directory: str = ""
    batch_csv_path: str = ""
    processing_time_seconds: float = 0.0

    def __str__(self) -> str:
        """Human-readable summary of batch results."""
        success_rate = (self.successful / self.total_videos * 100) if self.total_videos > 0 else 0
        return (
            f"Batch: {self.successful}/{self.total_videos} successful ({success_rate:.1f}%) "
            f"in {self.processing_time_seconds:.1f}s"
        )

    def get_successful_results(self) -> List[ProcessingResult]:
        """Get list of successful processing results."""
        return [r for r in self.video_results if r.success]

    def get_failed_results(self) -> List[ProcessingResult]:
        """Get list of failed processing results."""
        return [r for r in self.video_results if not r.success]

    def get_discharge_summary(self) -> Dict[str, float]:
        """Get summary statistics for discharge values.

        Returns:
            Dictionary with min, max, mean, std of discharge values
        """
        successful = self.get_successful_results()
        if not successful:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}

        discharges = [r.total_discharge for r in successful]
        return {
            "min": min(discharges),
            "max": max(discharges),
            "mean": sum(discharges) / len(discharges),
            "std": np.std(discharges),
            "count": len(discharges)
        }
