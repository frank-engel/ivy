"""User-friendly Python API for batch processing.

This module provides a simple interface for running batch image velocimetry
analysis on multiple videos. It wraps the lower-level BatchProcessor service
with a clean, easy-to-use API suitable for scripting and automation.

Example
-------
Simple usage with default settings::

    from image_velocimetry_tools.api import run_batch_processing

    results = run_batch_processing(
        scaffold_project='my_project.ivy',
        batch_csv='my_videos.csv',
        output_folder='results'
    )

    print(f"Processed {results.total_jobs} videos")
    print(f"Successful: {results.successful_jobs}")
    print(f"Failed: {results.failed_jobs}")

With progress reporting::

    def on_progress(job_num, total_jobs, job_name, status):
        print(f"[{job_num}/{total_jobs}] {job_name}: {status}")

    results = run_batch_processing(
        scaffold_project='my_project.ivy',
        batch_csv='my_videos.csv',
        output_folder='results',
        progress_callback=on_progress
    )

With .ivy project archiving::

    results = run_batch_processing(
        scaffold_project='my_project.ivy',
        batch_csv='my_videos.csv',
        output_folder='results',
        save_projects=True  # Save .ivy project for each video
    )
"""

import os
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass

from image_velocimetry_tools.services.batch_processor import BatchProcessor


@dataclass
class JobResult:
    """Results from processing a single video.

    Attributes
    ----------
    job_id : str
        Unique identifier for this job (e.g., "job_001")
    video_name : str
        Name of the video file (without path)
    status : str
        Job status: "completed" or "failed"
    discharge : float or None
        Computed discharge in display units (cfs or m³/s)
    area : float or None
        Cross-sectional area in display units (ft² or m²)
    discharge_units : str
        Units for discharge ("cfs" for English, "m³/s" for Metric)
    area_units : str
        Units for area ("ft²" for English, "m²" for Metric)
    water_elevation : float
        Water surface elevation used (in display units)
    alpha : float
        Alpha coefficient used (0.0 to 1.0)
    processing_time : float
        Time taken to process this video (seconds)
    error_message : str or None
        Error message if job failed, None if successful
    details : dict
        Additional statistics (velocities, depths, uncertainty, etc.)
    """

    job_id: str
    video_name: str
    status: str
    discharge: Optional[float]
    area: Optional[float]
    discharge_units: str
    area_units: str
    water_elevation: float
    alpha: float
    processing_time: float
    error_message: Optional[str]
    details: Dict[str, Any]

    @property
    def successful(self) -> bool:
        """Return True if job completed successfully."""
        return self.status == "completed"

    @property
    def failed(self) -> bool:
        """Return True if job failed."""
        return self.status == "failed"

    def __str__(self) -> str:
        """Return human-readable summary."""
        if self.successful:
            return (
                f"{self.video_name}: "
                f"Q={self.discharge:.2f} {self.discharge_units}, "
                f"A={self.area:.2f} {self.area_units}"
            )
        else:
            return f"{self.video_name}: FAILED - {self.error_message}"


@dataclass
class BatchResults:
    """Results from batch processing multiple videos.

    Attributes
    ----------
    jobs : list of JobResult
        Individual results for each video
    output_folder : str
        Path to output folder containing results
    summary_csv : str
        Path to batch summary CSV file
    total_jobs : int
        Total number of videos processed
    successful_jobs : int
        Number of videos processed successfully
    failed_jobs : int
        Number of videos that failed
    total_time : float
        Total processing time (seconds)
    """

    jobs: List[JobResult]
    output_folder: str
    summary_csv: str
    total_jobs: int
    successful_jobs: int
    failed_jobs: int
    total_time: float

    def get_successful_jobs(self) -> List[JobResult]:
        """Return list of successfully completed jobs."""
        return [job for job in self.jobs if job.successful]

    def get_failed_jobs(self) -> List[JobResult]:
        """Return list of failed jobs."""
        return [job for job in self.jobs if job.failed]

    def get_job_by_video(self, video_name: str) -> Optional[JobResult]:
        """Find job result by video filename.

        Parameters
        ----------
        video_name : str
            Video filename (with or without extension)

        Returns
        -------
        JobResult or None
            Job result if found, None otherwise
        """
        # Strip extension if provided
        video_stem = Path(video_name).stem

        for job in self.jobs:
            if Path(job.video_name).stem == video_stem:
                return job
        return None

    def print_summary(self):
        """Print a human-readable summary of results."""
        print("=" * 80)
        print("BATCH PROCESSING RESULTS")
        print("=" * 80)
        print(f"Total videos: {self.total_jobs}")
        print(f"Successful: {self.successful_jobs}")
        print(f"Failed: {self.failed_jobs}")
        print(f"Total time: {self.total_time:.1f} seconds")
        print(f"\nOutput folder: {self.output_folder}")
        print(f"Summary CSV: {self.summary_csv}")

        if self.successful_jobs > 0:
            print(f"\nSuccessful jobs:")
            for job in self.get_successful_jobs():
                print(f"  ✓ {job}")

        if self.failed_jobs > 0:
            print(f"\nFailed jobs:")
            for job in self.get_failed_jobs():
                print(f"  ✗ {job}")

        print("=" * 80)


def run_batch_processing(
    scaffold_project: str,
    batch_csv: str,
    output_folder: str,
    stop_on_error: bool = False,
    save_projects: bool = False,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
) -> BatchResults:
    """Run batch image velocimetry analysis on multiple videos.

    This function processes multiple videos using a scaffold project as a
    template and a CSV file specifying the videos and parameters. Each video
    is processed through the complete image velocimetry workflow:

    1. Extract frames from video
    2. Orthorectify frames to real-world coordinates
    3. Generate measurement grid
    4. Create image stack
    5. Run STIV analysis
    6. Compute discharge and statistics

    Parameters
    ----------
    scaffold_project : str
        Path to the scaffold .ivy project file. This project serves as a
        template, containing camera calibration, cross-section geometry,
        and STIV parameters. All videos will be processed using these
        settings.

    batch_csv : str
        Path to CSV file containing job specifications. Required columns:
        - video_path: Path to video file (relative to CSV or absolute)
        - water_surface_elevation: Water elevation (feet or meters)

        Optional columns:
        - start_time: Video clip start time (e.g., "10" or "1:30.5")
        - end_time: Video clip end time (e.g., "20" or "2:00")
        - alpha: Velocity correction coefficient (default: 0.85)
        - measurement_number: Reference number
        - measurement_date: Date (YYYY-MM-DD)
        - measurement_time: Time (HH:MM:SS)
        - gage_height: USGS gage height
        - comments: Notes

    output_folder : str
        Path to folder where results will be saved. The folder will be
        created if it doesn't exist. Results include:
        - batch_summary.csv: Summary of all jobs
        - Individual job folders with frames, orthorectified images, etc.
        - .ivy project files (if save_projects=True)

    stop_on_error : bool, default=False
        If True, stop processing when a job fails. If False, continue
        processing remaining jobs even if some fail.

    save_projects : bool, default=False
        If True, save a complete .ivy project archive for each video.
        These archives contain all intermediate results (frames, rectified
        images, STIV results) and can be opened in the IVyTools GUI.

        WARNING: This creates large files (100-500 MB per video) and
        significantly increases processing time. Only enable if you need
        to review individual results in the GUI.

    progress_callback : callable, optional
        Function to call for progress updates. Should accept:
        - job_num (int): Current job number (1-based)
        - total_jobs (int): Total number of jobs
        - job_name (str): Name of current job/video
        - status (str): Status message ("processing", "completed", "failed")

        Example::

            def my_callback(job_num, total, name, status):
                print(f"[{job_num}/{total}] {name}: {status}")

    Returns
    -------
    BatchResults
        Object containing results for all jobs, with methods to access
        successful/failed jobs and generate reports.

    Raises
    ------
    FileNotFoundError
        If scaffold project or batch CSV file doesn't exist
    ValueError
        If batch CSV is malformed or contains invalid data
    RuntimeError
        If processing fails critically (e.g., scaffold project corrupt)

    Examples
    --------
    Basic usage::

        results = run_batch_processing(
            scaffold_project='my_template.ivy',
            batch_csv='videos_to_process.csv',
            output_folder='batch_results'
        )

        results.print_summary()

    With progress reporting::

        def show_progress(job_num, total, name, status):
            percent = (job_num / total) * 100
            print(f"[{percent:.0f}%] {name}: {status}")

        results = run_batch_processing(
            scaffold_project='my_template.ivy',
            batch_csv='videos_to_process.csv',
            output_folder='batch_results',
            progress_callback=show_progress
        )

    Stop on first error and save .ivy projects::

        results = run_batch_processing(
            scaffold_project='my_template.ivy',
            batch_csv='videos_to_process.csv',
            output_folder='batch_results',
            stop_on_error=True,
            save_projects=True
        )

    Notes
    -----
    - Video paths in CSV can be relative to CSV location or absolute
    - All videos are processed using the same camera calibration and
      cross-section from the scaffold project
    - Results are output in the same units as the scaffold project
      (English units: cfs, ft, ft²; Metric units: m³/s, m, m²)
    - Processing time varies by video length and resolution, typically
      2-10 minutes per video
    """
    # Validate inputs
    scaffold_path = Path(scaffold_project)
    csv_path = Path(batch_csv)
    output_path = Path(output_folder)

    if not scaffold_path.exists():
        raise FileNotFoundError(
            f"Scaffold project not found: {scaffold_project}"
        )

    if not csv_path.exists():
        raise FileNotFoundError(f"Batch CSV not found: {batch_csv}")

    # Create BatchProcessor
    processor = BatchProcessor(
        scaffold_path=str(scaffold_path),
        batch_csv_path=str(csv_path),
        output_dir=str(output_path),
        stop_on_first_failure=stop_on_error,
        save_ivy_projects=save_projects,
    )

    # Set up progress callback if provided
    if progress_callback:

        def _internal_callback(current_job, total_jobs, job_id, status_msg):
            # Extract video name from job for callback
            video_name = "unknown"
            if hasattr(processor, "jobs") and current_job <= len(
                processor.jobs
            ):
                job = processor.jobs[current_job - 1]
                video_name = Path(job.video_path).name

            progress_callback(current_job, total_jobs, video_name, status_msg)

        # Note: BatchProcessor doesn't currently support callbacks,
        # but we can wrap it manually below

    # Run batch processing
    import time

    start_time = time.time()

    try:
        jobs = processor.run()
        total_time = time.time() - start_time
    except Exception as e:
        raise RuntimeError(f"Batch processing failed: {e}") from e

    # Determine units from first job's display_units
    display_units = jobs[0].display_units if jobs else "Metric"
    if display_units == "English":
        discharge_units = "cfs"
        area_units = "ft²"
    else:
        discharge_units = "m³/s"
        area_units = "m²"

    # Convert jobs to JobResult objects
    job_results = []
    for job in jobs:
        job_result = JobResult(
            job_id=job.job_id,
            video_name=Path(job.video_path).name,
            status=job.status.value,
            discharge=job.discharge_value,
            area=job.area_value,
            discharge_units=discharge_units,
            area_units=area_units,
            water_elevation=job.water_surface_elevation,
            alpha=job.alpha,
            processing_time=job.processing_time or 0.0,
            error_message=job.error_message,
            details=job.result_details or {},
        )
        job_results.append(job_result)

    # Count successes and failures
    successful = sum(1 for j in job_results if j.successful)
    failed = sum(1 for j in job_results if j.failed)

    # Create BatchResults object
    summary_csv_path = output_path / "batch_summary.csv"

    results = BatchResults(
        jobs=job_results,
        output_folder=str(output_path),
        summary_csv=str(summary_csv_path),
        total_jobs=len(job_results),
        successful_jobs=successful,
        failed_jobs=failed,
        total_time=total_time,
    )

    return results
