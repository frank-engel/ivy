"""BatchJob model representing a single batch processing job."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pathlib import Path


class JobStatus(Enum):
    """Enumeration of possible job statuses."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchJob:
    """Model representing a single batch processing job.

    This class encapsulates all configuration and state for processing
    a single video through the complete image velocimetry workflow.

    Parameters
    ----------
    job_id : str
        Unique identifier for this job (typically row index or video name)
    video_path : str
        Full path to the video file for this job
    water_surface_elevation : float
        Water surface elevation in the same units as GCPs (typically meters)
    start_time : Optional[str], default=None
        Start time for video clip (format: "ss" or "hh:mm:ss.s")
        None means start from beginning
    end_time : Optional[str], default=None
        End time for video clip (format: "ss" or "hh:mm:ss.s")
        None means process to end
    alpha : float, default=0.85
        Coefficient to correct surface velocity to mean channel velocity
    measurement_number : Optional[int], default=None
        Reference number for the measurement
    measurement_date : Optional[str], default=None
        Date of measurement (format: "YYYY-MM-DD")
    measurement_time : Optional[str], default=None
        Time of measurement (format: "HH:MM:SS")
    gage_height : Optional[float], default=None
        USGS gage height reference (may differ from water_surface_elevation)
    comments : Optional[str], default=None
        Job-specific comments

    Attributes
    ----------
    status : JobStatus
        Current processing status of the job
    discharge_value : Optional[float]
        Computed discharge result in m³/s (set after completion)
    error_message : Optional[str]
        Error message if job failed
    processing_time : Optional[float]
        Time taken to process job in seconds
    start_time_seconds : Optional[float]
        Start time converted to seconds (computed from start_time)
    end_time_seconds : Optional[float]
        End time converted to seconds (computed from end_time)

    Notes
    -----
    - All internal computations use SI units (meters, seconds)
    - Time strings are parsed and converted to seconds during validation
    - Status transitions: PENDING → PROCESSING → COMPLETED/FAILED
    """

    # Required fields
    job_id: str
    video_path: str
    water_surface_elevation: float
    display_units: str = "Metric"  # Units that water_surface_elevation is in

    # Optional time clipping
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    # Optional measurement parameters
    alpha: float = 0.85
    measurement_number: Optional[int] = None
    measurement_date: Optional[str] = None
    measurement_time: Optional[str] = None
    gage_height: Optional[float] = None
    comments: Optional[str] = None

    # Job state (not from CSV)
    status: JobStatus = field(default=JobStatus.PENDING, init=False)
    discharge_value: Optional[float] = field(default=None, init=False)
    area_value: Optional[float] = field(default=None, init=False)
    result_details: Optional[Dict[str, Any]] = field(default=None, init=False)
    error_message: Optional[str] = field(default=None, init=False)
    processing_time: Optional[float] = field(default=None, init=False)
    start_time_seconds: Optional[float] = field(default=None, init=False)
    end_time_seconds: Optional[float] = field(default=None, init=False)

    def __post_init__(self):
        """Validate and process job parameters after initialization."""
        self._validate()
        self._parse_times()

    def _validate(self) -> None:
        """Validate job parameters.

        Raises
        ------
        ValueError
            If any required field is invalid
        """
        if not self.job_id:
            raise ValueError("job_id cannot be empty")

        if not self.video_path:
            raise ValueError("video_path cannot be empty")

        # Validate alpha is in reasonable range
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(
                f"alpha must be between 0 and 1, got {self.alpha}"
            )

        # Validate measurement_number if provided
        if self.measurement_number is not None and self.measurement_number < 0:
            raise ValueError(
                f"measurement_number must be non-negative, got {self.measurement_number}"
            )

    def _parse_times(self) -> None:
        """Parse time strings to seconds for internal use.

        Converts start_time and end_time strings to seconds.
        Supports formats: "ss", "ss.s", "mm:ss", "hh:mm:ss", "hh:mm:ss.s"
        """
        if self.start_time is not None:
            self.start_time_seconds = self._time_string_to_seconds(
                self.start_time, "start_time"
            )

        if self.end_time is not None:
            self.end_time_seconds = self._time_string_to_seconds(
                self.end_time, "end_time"
            )

        # Validate end_time > start_time if both provided
        if (
            self.start_time_seconds is not None
            and self.end_time_seconds is not None
        ):
            if self.end_time_seconds <= self.start_time_seconds:
                raise ValueError(
                    f"end_time ({self.end_time}) must be greater than "
                    f"start_time ({self.start_time})"
                )

    @staticmethod
    def _time_string_to_seconds(time_str: str, field_name: str) -> float:
        """Convert time string to seconds.

        Parameters
        ----------
        time_str : str
            Time string in format "ss", "mm:ss", "hh:mm:ss", or with decimals
        field_name : str
            Name of field for error messages

        Returns
        -------
        float
            Time in seconds

        Raises
        ------
        ValueError
            If time string format is invalid
        """
        try:
            parts = time_str.split(":")

            if len(parts) == 1:
                # Format: "ss" or "ss.s"
                return float(parts[0])
            elif len(parts) == 2:
                # Format: "mm:ss" or "mm:ss.s"
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            elif len(parts) == 3:
                # Format: "hh:mm:ss" or "hh:mm:ss.s"
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
            else:
                raise ValueError(f"Invalid time format: {time_str}")

        except (ValueError, AttributeError) as e:
            raise ValueError(
                f"{field_name} has invalid format '{time_str}'. "
                f"Expected 'ss', 'mm:ss', or 'hh:mm:ss' (with optional decimals)"
            ) from e

    def mark_processing(self) -> None:
        """Mark job as currently processing."""
        self.status = JobStatus.PROCESSING

    def mark_completed(
        self,
        discharge_value: float,
        processing_time: float,
        area_value: Optional[float] = None,
        result_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark job as successfully completed.

        Parameters
        ----------
        discharge_value : float
            Computed discharge (in display units)
        processing_time : float
            Time taken to process job in seconds
        area_value : float, optional
            Computed cross-sectional area (in display units)
        result_details : dict, optional
            Additional result statistics (velocities, depths, uncertainty, etc.)
        """
        self.status = JobStatus.COMPLETED
        self.discharge_value = discharge_value
        self.area_value = area_value
        self.result_details = result_details or {}
        self.processing_time = processing_time
        self.error_message = None

    def mark_failed(
        self, error_message: str, processing_time: float = 0.0
    ) -> None:
        """Mark job as failed.

        Parameters
        ----------
        error_message : str
            Description of what went wrong
        processing_time : float, default=0.0
            Time spent before failure in seconds
        """
        self.status = JobStatus.FAILED
        self.error_message = error_message
        self.processing_time = processing_time
        self.discharge_value = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary representation.

        Returns
        -------
        dict
            Dictionary containing all job data including results

        Notes
        -----
        This is useful for serialization and creating summary reports.
        """
        return {
            "job_id": self.job_id,
            "video_path": self.video_path,
            "water_surface_elevation": self.water_surface_elevation,
            "display_units": self.display_units,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_time_seconds": self.start_time_seconds,
            "end_time_seconds": self.end_time_seconds,
            "alpha": self.alpha,
            "measurement_number": self.measurement_number,
            "measurement_date": self.measurement_date,
            "measurement_time": self.measurement_time,
            "gage_height": self.gage_height,
            "comments": self.comments,
            "status": self.status.value,
            "discharge_value": self.discharge_value,
            "error_message": self.error_message,
            "processing_time": self.processing_time,
        }

    def get_video_filename(self) -> str:
        """Get the video filename without path or extension.

        Returns
        -------
        str
            Video filename stem (without extension)

        Notes
        -----
        Useful for creating output filenames and directories.
        """
        return Path(self.video_path).stem

    def __repr__(self) -> str:
        """Return string representation of job."""
        return (
            f"BatchJob(job_id='{self.job_id}', "
            f"video='{self.get_video_filename()}', "
            f"status={self.status.value})"
        )
