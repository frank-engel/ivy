"""Batch processing module for Image Velocimetry Tools.

This module provides functionality to process multiple river discharge
analysis jobs from a single command, using a scaffold project template
and a CSV file defining per-job variations.

Key Components
--------------
- Models: BatchJob, BatchConfig, JobStatus
- Exceptions: BatchProcessingError and subclasses
- Services: Will be implemented in image_velocimetry_tools.services

Examples
--------
>>> from image_velocimetry_tools.batch.models import BatchConfig
>>> config = BatchConfig(
...     scaffold_path="scaffold.ivy",
...     batch_csv_path="jobs.csv",
...     output_dir="./results"
... )
>>> errors = config.validate()
>>> if not errors:
...     print("Configuration is valid!")
"""

from image_velocimetry_tools.batch.models import (
    BatchJob,
    JobStatus,
    BatchConfig,
)
from image_velocimetry_tools.batch.exceptions import (
    BatchProcessingError,
    InvalidScaffoldError,
    InvalidBatchCSVError,
    JobExecutionError,
    BatchValidationError,
)

__all__ = [
    # Models
    "BatchJob",
    "JobStatus",
    "BatchConfig",
    # Exceptions
    "BatchProcessingError",
    "InvalidScaffoldError",
    "InvalidBatchCSVError",
    "JobExecutionError",
    "BatchValidationError",
]

__version__ = "1.0.0"
