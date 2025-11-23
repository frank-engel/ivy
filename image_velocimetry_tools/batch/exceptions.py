"""Custom exceptions for batch processing operations."""


class BatchProcessingError(Exception):
    """Base exception for all batch processing errors.

    This is the parent exception class for all batch-related errors.
    Catching this exception will catch all batch processing errors.
    """

    pass


class InvalidScaffoldError(BatchProcessingError):
    """Raised when scaffold project file is invalid or missing required data.

    Examples
    --------
    - Scaffold file doesn't exist
    - Scaffold file is corrupted
    - Required configuration missing (GCPs, cross-section, etc.)
    """

    pass


class InvalidBatchCSVError(BatchProcessingError):
    """Raised when batch CSV file is malformed or contains invalid data.

    Examples
    --------
    - CSV file doesn't exist
    - Required columns missing
    - Invalid data types or values
    - Malformed time strings
    """

    pass


class JobExecutionError(BatchProcessingError):
    """Raised when a batch job fails during execution.

    This exception is used for runtime errors during job processing,
    such as video processing failures, STIV computation errors, etc.
    """

    pass


class BatchValidationError(BatchProcessingError):
    """Raised when batch configuration validation fails.

    Examples
    --------
    - Output directory cannot be created
    - Invalid file paths
    - Conflicting configuration options
    """

    pass
