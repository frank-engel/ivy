"""Service for orchestrating batch processing of multiple jobs.

This service coordinates the complete batch processing workflow, from loading
configuration files through executing all jobs and generating summary reports.
"""

import os
import time
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from image_velocimetry_tools.services.base_service import BaseService
from image_velocimetry_tools.services.batch_csv_parser import BatchCSVParser
from image_velocimetry_tools.services.scaffold_loader import ScaffoldLoader
from image_velocimetry_tools.services.job_executor import JobExecutor
from image_velocimetry_tools.batch.models import BatchConfig, BatchJob, JobStatus
from image_velocimetry_tools.batch.exceptions import (
    BatchProcessingError,
    InvalidBatchCSVError,
    InvalidScaffoldError,
    JobExecutionError,
    BatchValidationError
)


class BatchProcessor(BaseService):
    """Service for orchestrating batch processing of river discharge analysis.

    This is the main entry point for batch processing. It coordinates:
    1. Configuration validation (scaffold + CSV)
    2. Loading and parsing input files
    3. Sequential execution of jobs
    4. Progress tracking and error handling
    5. Summary report generation

    The batch processor can be used programmatically or via CLI.

    Examples
    --------
    >>> from image_velocimetry_tools.services.batch_processor import BatchProcessor
    >>> processor = BatchProcessor(
    ...     scaffold_path="scaffold.ivy",
    ...     batch_csv_path="jobs.csv",
    ...     output_dir="./results"
    ... )
    >>> results = processor.run()
    >>> print(f"Processed {len(results)} jobs")
    """

    def __init__(
        self,
        scaffold_path: str,
        batch_csv_path: str,
        output_dir: str,
        stop_on_first_failure: bool = False
    ):
        """Initialize the BatchProcessor.

        Parameters
        ----------
        scaffold_path : str
            Path to scaffold project file (.ivy)
        batch_csv_path : str
            Path to batch CSV file
        output_dir : str
            Directory for output files
        stop_on_first_failure : bool, default=False
            If True, stop processing when first job fails
            If False, continue processing remaining jobs

        Raises
        ------
        BatchValidationError
            If configuration validation fails
        """
        super().__init__()

        # Create configuration
        self.config = BatchConfig(
            scaffold_path=scaffold_path,
            batch_csv_path=batch_csv_path,
            output_dir=output_dir,
            stop_on_first_failure=stop_on_first_failure
        )

        # Validate configuration
        validation_errors = self.config.validate()
        if validation_errors:
            error_summary = "\n".join(validation_errors)
            raise BatchValidationError(
                f"Configuration validation failed:\n{error_summary}"
            )

        # Initialize services
        self.csv_parser = BatchCSVParser()
        self.scaffold_loader = ScaffoldLoader()
        self.job_executor = JobExecutor()

        # State
        self.jobs: List[BatchJob] = []
        self.scaffold_config: Optional[Dict[str, Any]] = None
        self.batch_start_time: Optional[float] = None
        self.batch_end_time: Optional[float] = None

    def run(self) -> List[BatchJob]:
        """Execute the complete batch processing workflow.

        This method:
        1. Creates output directory
        2. Loads scaffold project
        3. Parses batch CSV
        4. Executes each job sequentially
        5. Generates summary report
        6. Cleans up temporary files

        Returns
        -------
        list of BatchJob
            List of all jobs with updated status and results

        Raises
        ------
        BatchProcessingError
            If batch setup or execution fails
        """
        self.batch_start_time = time.time()

        self.logger.info("="*60)
        self.logger.info("Starting batch processing")
        self.logger.info(f"  Scaffold: {self.config.scaffold_path}")
        self.logger.info(f"  Batch CSV: {self.config.batch_csv_path}")
        self.logger.info(f"  Output: {self.config.output_dir}")
        self.logger.info("="*60)

        try:
            # Step 1: Create output directory
            self._setup_output_directory()

            # Step 2: Load scaffold
            self._load_scaffold()

            # Step 3: Parse batch CSV
            self._parse_batch_csv()

            # Step 4: Execute jobs
            self._execute_all_jobs()

            # Step 5: Generate summary
            self._generate_summary()

            self.batch_end_time = time.time()
            total_time = self.batch_end_time - self.batch_start_time

            # Print final summary
            self._print_final_summary(total_time)

            return self.jobs

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            raise BatchProcessingError(f"Batch processing failed: {e}") from e

        finally:
            # Cleanup
            self._cleanup()

    def _setup_output_directory(self) -> None:
        """Create output directory structure."""
        try:
            self.config.create_output_directory()
            self.logger.info(f"Created output directory: {self.config.output_dir}")
        except Exception as e:
            raise BatchProcessingError(
                f"Failed to create output directory: {e}"
            ) from e

        # Setup logging to file
        log_path = self.config.get_batch_log_path()
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logger.info(f"Batch log: {log_path}")

    def _load_scaffold(self) -> None:
        """Load and validate scaffold project."""
        self.logger.info("Loading scaffold project...")

        try:
            self.scaffold_config = self.scaffold_loader.load_scaffold(
                scaffold_path=str(self.config.scaffold_path_resolved)
            )

            self.logger.info(
                f"Scaffold loaded successfully from: "
                f"{self.config.scaffold_path_resolved.name}"
            )
            self.logger.info(
                f"  Extracted to: {self.scaffold_config['extract_dir']}"
            )
            self.logger.info(
                f"  Cross-section: {Path(self.scaffold_config['cross_section_path']).name}"
            )

        except InvalidScaffoldError as e:
            raise BatchProcessingError(f"Invalid scaffold: {e}") from e
        except Exception as e:
            raise BatchProcessingError(f"Failed to load scaffold: {e}") from e

    def _parse_batch_csv(self) -> None:
        """Parse batch CSV file into BatchJob objects."""
        self.logger.info("Parsing batch CSV...")

        try:
            self.jobs = self.csv_parser.parse_csv(
                csv_path=str(self.config.batch_csv_path_resolved)
            )

            self.logger.info(f"Parsed {len(self.jobs)} jobs from CSV")

            # Get display units from scaffold
            display_units = self.scaffold_config["project_data"].get("display_units", "Metric")
            self.logger.info(f"Scaffold display units: {display_units}")

            # Resolve video paths and set display units for all jobs
            for job in self.jobs:
                resolved_path = self.config.resolve_video_path(job.video_path)
                job.video_path = str(resolved_path)

                # Set display units from scaffold
                job.display_units = display_units

                # Validate video file exists
                if not resolved_path.exists():
                    self.logger.warning(
                        f"[{job.job_id}] Video file not found: {job.video_path}"
                    )

        except InvalidBatchCSVError as e:
            raise BatchProcessingError(f"Invalid batch CSV: {e}") from e
        except Exception as e:
            raise BatchProcessingError(f"Failed to parse batch CSV: {e}") from e

    def _execute_all_jobs(self) -> None:
        """Execute all jobs sequentially."""
        self.logger.info("")
        self.logger.info("="*60)
        self.logger.info(f"Executing {len(self.jobs)} jobs")
        self.logger.info("="*60)

        completed = 0
        failed = 0

        for idx, job in enumerate(self.jobs, 1):
            self.logger.info("")
            self.logger.info(f"Job {idx}/{len(self.jobs)}: {job.job_id}")
            self.logger.info("-"*60)

            # Get job output directory
            job_output_dir = self.config.get_job_output_dir(job.job_id)

            try:
                # Execute job
                job_start_time = time.time()

                result = self.job_executor.execute_job(
                    job=job,
                    scaffold_config=self.scaffold_config,
                    output_dir=str(self.config.output_dir_resolved)
                )

                job_end_time = time.time()
                processing_time = job_end_time - job_start_time

                # Mark job as completed
                job.mark_completed(
                    discharge_value=result["discharge"],
                    processing_time=processing_time
                )

                completed += 1

                self.logger.info(
                    f"[{job.job_id}] SUCCESS - "
                    f"Discharge: {result['discharge']:.3f} m³/s, "
                    f"Time: {processing_time:.1f}s"
                )

            except JobExecutionError as e:
                job_end_time = time.time()
                processing_time = job_end_time - job_start_time

                # Mark job as failed
                job.mark_failed(
                    error_message=str(e),
                    processing_time=processing_time
                )

                failed += 1

                self.logger.error(f"[{job.job_id}] FAILED - {e}")

                # Stop on first failure if configured
                if self.config.stop_on_first_failure:
                    self.logger.error(
                        "Stopping batch processing due to job failure "
                        "(stop_on_first_failure=True)"
                    )
                    # Mark remaining jobs as pending
                    break

            except Exception as e:
                job_end_time = time.time()
                processing_time = job_end_time - job_start_time

                job.mark_failed(
                    error_message=f"Unexpected error: {e}",
                    processing_time=processing_time
                )

                failed += 1

                self.logger.error(f"[{job.job_id}] FAILED - Unexpected error: {e}")

                if self.config.stop_on_first_failure:
                    self.logger.error(
                        "Stopping batch processing due to job failure "
                        "(stop_on_first_failure=True)"
                    )
                    break

        self.logger.info("")
        self.logger.info("="*60)
        self.logger.info(f"Batch execution complete: {completed} succeeded, {failed} failed")
        self.logger.info("="*60)

    def _generate_summary(self) -> None:
        """Generate batch summary CSV file."""
        summary_path = self.config.get_batch_summary_path()

        self.logger.info(f"Generating summary report: {summary_path}")

        try:
            with open(summary_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                writer.writerow([
                    "job_id",
                    "status",
                    "video_path",
                    "water_surface_elevation",
                    "alpha",
                    "discharge_m3s",
                    "processing_time_seconds",
                    "error_message",
                    "measurement_number",
                    "measurement_date",
                    "comments"
                ])

                # Write job results
                for job in self.jobs:
                    writer.writerow([
                        job.job_id,
                        job.status.value,
                        job.video_path,
                        job.water_surface_elevation,
                        job.alpha,
                        job.discharge_value if job.discharge_value else "",
                        f"{job.processing_time:.2f}" if job.processing_time else "",
                        job.error_message if job.error_message else "",
                        job.measurement_number if job.measurement_number else "",
                        job.measurement_date if job.measurement_date else "",
                        job.comments if job.comments else ""
                    ])

            self.logger.info(f"Summary report saved: {summary_path}")

        except Exception as e:
            self.logger.error(f"Failed to generate summary: {e}")
            # Don't raise - summary is nice to have but not critical

    def _print_final_summary(self, total_time: float) -> None:
        """Print final summary to console and log."""
        # Count status
        completed = sum(1 for job in self.jobs if job.status == JobStatus.COMPLETED)
        failed = sum(1 for job in self.jobs if job.status == JobStatus.FAILED)
        pending = sum(1 for job in self.jobs if job.status == JobStatus.PENDING)

        # Calculate total discharge
        total_discharge = sum(
            job.discharge_value for job in self.jobs
            if job.discharge_value is not None
        )

        summary = f"""
{'='*60}
BATCH PROCESSING COMPLETE
{'='*60}
Total Jobs:     {len(self.jobs)}
  Completed:    {completed}
  Failed:       {failed}
  Pending:      {pending}

Total Discharge: {total_discharge:.3f} m³/s
Total Time:      {total_time:.1f} seconds ({total_time/60:.1f} minutes)

Output Directory: {self.config.output_dir_resolved}
Summary Report:   {self.config.get_batch_summary_path().name}
Batch Log:        {self.config.get_batch_log_path().name}
{'='*60}
"""
        self.logger.info(summary)
        print(summary)  # Also print to console

    def _cleanup(self) -> None:
        """Clean up temporary files."""
        if self.scaffold_config:
            extract_dir = self.scaffold_config.get("extract_dir")
            if extract_dir:
                self.scaffold_loader.cleanup_scaffold(extract_dir)
                self.logger.debug(f"Cleaned up scaffold directory: {extract_dir}")

    def get_progress(self) -> Dict[str, Any]:
        """Get current batch processing progress.

        Returns
        -------
        dict
            Dictionary containing:
            - total_jobs: Total number of jobs
            - completed: Number of completed jobs
            - failed: Number of failed jobs
            - pending: Number of pending jobs
            - processing: Number of jobs currently processing
            - percent_complete: Percentage complete
        """
        if not self.jobs:
            return {
                "total_jobs": 0,
                "completed": 0,
                "failed": 0,
                "pending": 0,
                "processing": 0,
                "percent_complete": 0.0
            }

        total = len(self.jobs)
        completed = sum(1 for job in self.jobs if job.status == JobStatus.COMPLETED)
        failed = sum(1 for job in self.jobs if job.status == JobStatus.FAILED)
        processing = sum(1 for job in self.jobs if job.status == JobStatus.PROCESSING)
        pending = sum(1 for job in self.jobs if job.status == JobStatus.PENDING)

        percent = ((completed + failed) / total * 100) if total > 0 else 0.0

        return {
            "total_jobs": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "processing": processing,
            "percent_complete": percent
        }
