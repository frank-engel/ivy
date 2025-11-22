"""BatchConfig model representing batch processing configuration."""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class BatchConfig:
    """Model representing configuration for a batch processing run.

    This class encapsulates all settings needed to execute a batch of
    image velocimetry jobs, including input files, output location,
    and processing options.

    Parameters
    ----------
    scaffold_path : str
        Path to the scaffold project file (.ivy)
        Contains common configuration for all jobs
    batch_csv_path : str
        Path to the batch CSV file
        Each row defines one job with job-specific parameters
    output_dir : str
        Directory where results will be written
        Will be created if it doesn't exist
    stop_on_first_failure : bool, default=False
        If True, stop processing immediately when any job fails
        If False, continue processing remaining jobs after failures

    Attributes
    ----------
    scaffold_path_resolved : Path
        Resolved absolute path to scaffold file
    batch_csv_path_resolved : Path
        Resolved absolute path to batch CSV file
    output_dir_resolved : Path
        Resolved absolute path to output directory
    batch_csv_dir : Path
        Directory containing the batch CSV file
        Used for resolving relative video paths in CSV

    Notes
    -----
    - All paths are converted to absolute paths during initialization
    - Video paths in the CSV file can be relative to the CSV location
    - The output directory structure will be:
        output_dir/
        ├── job_001/
        │   ├── project_data.json
        │   └── 1-images/, 2-orthorectification/, etc.
        ├── job_002/
        └── batch_summary.csv

    Examples
    --------
    >>> config = BatchConfig(
    ...     scaffold_path="scaffold_project.ivy",
    ...     batch_csv_path="jobs.csv",
    ...     output_dir="./results"
    ... )
    >>> config.validate()
    []  # Empty list means no errors
    """

    scaffold_path: str
    batch_csv_path: str
    output_dir: str
    stop_on_first_failure: bool = False
    save_ivy_projects: bool = False  # Save .ivy project for each job (large files, slow)

    def __post_init__(self):
        """Resolve and validate paths after initialization."""
        # Resolve all paths to absolute
        self.scaffold_path_resolved = Path(self.scaffold_path).resolve()
        self.batch_csv_path_resolved = Path(self.batch_csv_path).resolve()
        self.output_dir_resolved = Path(self.output_dir).resolve()

        # Store CSV directory for resolving relative video paths
        self.batch_csv_dir = self.batch_csv_path_resolved.parent

    def validate(self) -> list:
        """Validate configuration parameters.

        Returns
        -------
        list of str
            List of validation error messages
            Empty list if all validations pass

        Notes
        -----
        This method checks:
        - Scaffold file exists and is readable
        - Batch CSV file exists and is readable
        - Output directory can be created (if it doesn't exist)
        - File extensions are correct
        """
        errors = []

        # Validate scaffold path
        if not self.scaffold_path:
            errors.append("scaffold_path cannot be empty")
        elif not self.scaffold_path_resolved.exists():
            errors.append(
                f"Scaffold file does not exist: {self.scaffold_path_resolved}"
            )
        elif not self.scaffold_path_resolved.is_file():
            errors.append(
                f"Scaffold path is not a file: {self.scaffold_path_resolved}"
            )
        elif not self.scaffold_path_resolved.suffix == ".ivy":
            errors.append(
                f"Scaffold file must have .ivy extension, "
                f"got: {self.scaffold_path_resolved.suffix}"
            )

        # Validate batch CSV path
        if not self.batch_csv_path:
            errors.append("batch_csv_path cannot be empty")
        elif not self.batch_csv_path_resolved.exists():
            errors.append(
                f"Batch CSV file does not exist: {self.batch_csv_path_resolved}"
            )
        elif not self.batch_csv_path_resolved.is_file():
            errors.append(
                f"Batch CSV path is not a file: {self.batch_csv_path_resolved}"
            )
        elif not self.batch_csv_path_resolved.suffix == ".csv":
            errors.append(
                f"Batch file must have .csv extension, "
                f"got: {self.batch_csv_path_resolved.suffix}"
            )

        # Validate output directory
        if not self.output_dir:
            errors.append("output_dir cannot be empty")
        else:
            # Check if we can create the output directory
            if not self.output_dir_resolved.exists():
                try:
                    # Try to create parent directory to test permissions
                    parent = self.output_dir_resolved.parent
                    if not parent.exists():
                        errors.append(
                            f"Output directory parent does not exist and "
                            f"cannot be created: {parent}"
                        )
                    elif not os.access(parent, os.W_OK):
                        errors.append(
                            f"No write permission for output directory parent: {parent}"
                        )
                except (OSError, PermissionError) as e:
                    errors.append(
                        f"Cannot create output directory: {e}"
                    )
            elif not self.output_dir_resolved.is_dir():
                errors.append(
                    f"Output path exists but is not a directory: "
                    f"{self.output_dir_resolved}"
                )
            elif not os.access(self.output_dir_resolved, os.W_OK):
                errors.append(
                    f"No write permission for output directory: "
                    f"{self.output_dir_resolved}"
                )

        return errors

    def create_output_directory(self) -> None:
        """Create the output directory if it doesn't exist.

        Raises
        ------
        OSError
            If directory cannot be created due to permissions or other errors
        """
        self.output_dir_resolved.mkdir(parents=True, exist_ok=True)

    def get_job_output_dir(self, job_id: str) -> Path:
        """Get the output directory path for a specific job.

        Parameters
        ----------
        job_id : str
            Unique identifier for the job

        Returns
        -------
        Path
            Absolute path to job's output directory

        Notes
        -----
        Directory name is sanitized to remove invalid filesystem characters.
        """
        # Sanitize job_id for filesystem use
        safe_job_id = "".join(
            c if c.isalnum() or c in ('-', '_') else '_'
            for c in job_id
        )
        return self.output_dir_resolved / f"job_{safe_job_id}"

    def get_batch_summary_path(self) -> Path:
        """Get path for batch summary CSV file.

        Returns
        -------
        Path
            Absolute path to batch_summary.csv in output directory
        """
        return self.output_dir_resolved / "batch_summary.csv"

    def get_batch_log_path(self) -> Path:
        """Get path for batch processing log file.

        Returns
        -------
        Path
            Absolute path to batch.log in output directory
        """
        return self.output_dir_resolved / "batch.log"

    def resolve_video_path(self, video_path: str) -> Path:
        """Resolve video path (may be relative to CSV location).

        Parameters
        ----------
        video_path : str
            Video path from CSV (may be relative or absolute)

        Returns
        -------
        Path
            Resolved absolute path to video file

        Notes
        -----
        If video_path is relative, it's resolved relative to the
        directory containing the batch CSV file.
        """
        video_path_obj = Path(video_path)

        if video_path_obj.is_absolute():
            return video_path_obj
        else:
            # Resolve relative to CSV location
            return (self.batch_csv_dir / video_path_obj).resolve()

    def __repr__(self) -> str:
        """Return string representation of configuration."""
        return (
            f"BatchConfig(\n"
            f"  scaffold='{self.scaffold_path_resolved.name}',\n"
            f"  batch_csv='{self.batch_csv_path_resolved.name}',\n"
            f"  output_dir='{self.output_dir_resolved}',\n"
            f"  stop_on_first_failure={self.stop_on_first_failure}\n"
            f")"
        )
