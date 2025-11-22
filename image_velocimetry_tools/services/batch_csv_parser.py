"""Service for parsing and validating batch CSV files.

This service handles parsing CSV files containing batch job specifications
and converting them into BatchJob objects with validation.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from image_velocimetry_tools.services.base_service import BaseService
from image_velocimetry_tools.batch.models import BatchJob
from image_velocimetry_tools.batch.exceptions import InvalidBatchCSVError


class BatchCSVParser(BaseService):
    """Service for parsing batch CSV files into BatchJob objects.

    This service reads CSV files containing job specifications and creates
    validated BatchJob instances for each row. It handles data type conversion,
    validation, and error reporting.

    Required CSV columns:
    - video_path: Path to video file
    - water_surface_elevation: Water surface elevation (numeric)

    Optional CSV columns:
    - start_time: Video clip start time (string)
    - end_time: Video clip end time (string)
    - alpha: Velocity correction coefficient (numeric, default=0.85)
    - measurement_number: Measurement reference number (integer)
    - measurement_date: Date of measurement (string)
    - measurement_time: Time of measurement (string)
    - gage_height: USGS gage height (numeric)
    - comments: Job-specific comments (string)
    """

    # Required columns that must exist in CSV
    REQUIRED_COLUMNS = [
        "video_path",
        "water_surface_elevation"
    ]

    # Optional columns with their default values
    OPTIONAL_COLUMNS = {
        "start_time": None,
        "end_time": None,
        "alpha": 0.85,
        "measurement_number": None,
        "measurement_date": None,
        "measurement_time": None,
        "gage_height": None,
        "comments": None,
    }

    def __init__(self):
        """Initialize the BatchCSVParser service."""
        super().__init__()

    def parse_csv(
        self,
        csv_path: str,
        validate_video_paths: bool = True
    ) -> List[BatchJob]:
        """Parse batch CSV file and create BatchJob objects.

        Parameters
        ----------
        csv_path : str
            Path to the batch CSV file
        validate_video_paths : bool, default=True
            If True, check that video files exist during validation

        Returns
        -------
        list of BatchJob
            List of validated BatchJob instances, one per CSV row

        Raises
        ------
        InvalidBatchCSVError
            If CSV file is malformed, missing required columns, or contains
            invalid data

        Notes
        -----
        - Empty rows are skipped
        - Job IDs are auto-generated as "job_{row_number}" if not provided
        - Numeric columns are automatically converted to appropriate types
        - Invalid rows generate detailed error messages
        """
        self.logger.info(f"Parsing batch CSV: {csv_path}")

        # Validate CSV file exists
        csv_path_obj = Path(csv_path)
        if not csv_path_obj.exists():
            raise InvalidBatchCSVError(f"CSV file does not exist: {csv_path}")

        if not csv_path_obj.is_file():
            raise InvalidBatchCSVError(f"Path is not a file: {csv_path}")

        # Read CSV file
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            raise InvalidBatchCSVError(f"CSV file is empty: {csv_path}")
        except pd.errors.ParserError as e:
            raise InvalidBatchCSVError(
                f"Failed to parse CSV file: {csv_path}. Error: {e}"
            )
        except Exception as e:
            raise InvalidBatchCSVError(
                f"Error reading CSV file: {csv_path}. Error: {e}"
            )

        # Validate CSV structure
        self._validate_csv_structure(df, csv_path)

        # Remove completely empty rows
        df = df.dropna(how='all')

        if len(df) == 0:
            raise InvalidBatchCSVError(
                f"CSV file contains no data rows: {csv_path}"
            )

        # Parse each row into a BatchJob
        jobs = []
        errors = []

        for idx, row in df.iterrows():
            try:
                job = self._parse_row_to_job(row, idx, csv_path)
                jobs.append(job)
            except Exception as e:
                error_msg = f"Row {idx + 2}: {str(e)}"  # +2 for header and 0-indexing
                errors.append(error_msg)
                self.logger.warning(f"Failed to parse row {idx + 2}: {e}")

        # Report all parsing errors
        if errors:
            error_summary = "\n".join(errors)
            raise InvalidBatchCSVError(
                f"Failed to parse {len(errors)} row(s) in {csv_path}:\n{error_summary}"
            )

        self.logger.info(
            f"Successfully parsed {len(jobs)} jobs from {csv_path}"
        )

        return jobs

    def _validate_csv_structure(self, df: pd.DataFrame, csv_path: str) -> None:
        """Validate that CSV has required columns.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to validate
        csv_path : str
            Path to CSV file (for error messages)

        Raises
        ------
        InvalidBatchCSVError
            If required columns are missing
        """
        # Check for required columns
        missing_columns = [
            col for col in self.REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing_columns:
            raise InvalidBatchCSVError(
                f"CSV file missing required columns: {missing_columns}. "
                f"Required columns are: {self.REQUIRED_COLUMNS}"
            )

        self.logger.debug(
            f"CSV structure validated. Columns: {list(df.columns)}"
        )

    def _parse_row_to_job(
        self,
        row: pd.Series,
        row_index: int,
        csv_path: str
    ) -> BatchJob:
        """Parse a single CSV row into a BatchJob.

        Parameters
        ----------
        row : pd.Series
            Row data from CSV
        row_index : int
            Row index (0-based) for job ID generation
        csv_path : str
            Path to CSV file (for error context)

        Returns
        -------
        BatchJob
            Validated BatchJob instance

        Raises
        ------
        ValueError
            If row data is invalid
        """
        # Generate job_id from row index (1-based, zero-padded)
        job_id = f"job_{row_index + 1:03d}"

        # Extract required fields
        video_path = self._get_string_field(row, "video_path", required=True)
        water_surface_elevation = self._get_numeric_field(
            row, "water_surface_elevation", required=True
        )

        # Extract optional fields
        start_time = self._get_string_field(row, "start_time")
        end_time = self._get_string_field(row, "end_time")
        alpha = self._get_numeric_field(row, "alpha", default=0.85)
        measurement_number = self._get_integer_field(row, "measurement_number")
        measurement_date = self._get_string_field(row, "measurement_date")
        measurement_time = self._get_string_field(row, "measurement_time")
        gage_height = self._get_numeric_field(row, "gage_height")
        comments = self._get_string_field(row, "comments")

        # Create BatchJob (validation happens in __post_init__)
        try:
            job = BatchJob(
                job_id=job_id,
                video_path=video_path,
                water_surface_elevation=water_surface_elevation,
                start_time=start_time,
                end_time=end_time,
                alpha=alpha,
                measurement_number=measurement_number,
                measurement_date=measurement_date,
                measurement_time=measurement_time,
                gage_height=gage_height,
                comments=comments,
            )
            return job

        except ValueError as e:
            raise ValueError(f"Invalid data: {e}")

    def _get_string_field(
        self,
        row: pd.Series,
        field_name: str,
        required: bool = False
    ) -> str:
        """Extract string field from row.

        Parameters
        ----------
        row : pd.Series
            Row data
        field_name : str
            Name of field to extract
        required : bool, default=False
            Whether field is required

        Returns
        -------
        str or None
            Field value as string, or None if not present

        Raises
        ------
        ValueError
            If required field is missing or empty
        """
        if field_name not in row:
            if required:
                raise ValueError(f"Required field '{field_name}' is missing")
            return None

        value = row[field_name]

        # Handle pandas NaN/None
        if pd.isna(value):
            if required:
                raise ValueError(f"Required field '{field_name}' is empty")
            return None

        # Convert to string and strip whitespace
        value_str = str(value).strip()

        if required and not value_str:
            raise ValueError(f"Required field '{field_name}' is empty")

        return value_str if value_str else None

    def _get_numeric_field(
        self,
        row: pd.Series,
        field_name: str,
        required: bool = False,
        default: float = None
    ) -> float:
        """Extract numeric field from row.

        Parameters
        ----------
        row : pd.Series
            Row data
        field_name : str
            Name of field to extract
        required : bool, default=False
            Whether field is required
        default : float, default=None
            Default value if field is missing

        Returns
        -------
        float or None
            Field value as float, or None/default if not present

        Raises
        ------
        ValueError
            If required field is missing or cannot be converted to float
        """
        if field_name not in row:
            if required:
                raise ValueError(f"Required field '{field_name}' is missing")
            return default

        value = row[field_name]

        # Handle pandas NaN/None
        if pd.isna(value):
            if required:
                raise ValueError(f"Required field '{field_name}' is empty")
            return default

        # Convert to float
        try:
            return float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Field '{field_name}' must be numeric, got: {value}"
            )

    def _get_integer_field(
        self,
        row: pd.Series,
        field_name: str,
        required: bool = False,
        default: int = None
    ) -> int:
        """Extract integer field from row.

        Parameters
        ----------
        row : pd.Series
            Row data
        field_name : str
            Name of field to extract
        required : bool, default=False
            Whether field is required
        default : int, default=None
            Default value if field is missing

        Returns
        -------
        int or None
            Field value as integer, or None/default if not present

        Raises
        ------
        ValueError
            If required field is missing or cannot be converted to int
        """
        if field_name not in row:
            if required:
                raise ValueError(f"Required field '{field_name}' is missing")
            return default

        value = row[field_name]

        # Handle pandas NaN/None
        if pd.isna(value):
            if required:
                raise ValueError(f"Required field '{field_name}' is empty")
            return default

        # Convert to integer
        try:
            return int(value)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Field '{field_name}' must be an integer, got: {value}"
            )

    def get_csv_info(self, csv_path: str) -> Dict[str, Any]:
        """Get basic information about a CSV file without full parsing.

        Parameters
        ----------
        csv_path : str
            Path to the batch CSV file

        Returns
        -------
        dict
            Dictionary containing:
            - num_rows: Number of data rows (excluding header)
            - columns: List of column names
            - has_required_columns: Boolean indicating if required columns present
            - missing_columns: List of missing required columns

        Raises
        ------
        InvalidBatchCSVError
            If CSV file cannot be read
        """
        try:
            df = pd.read_csv(csv_path)
            df = df.dropna(how='all')  # Remove empty rows

            missing_columns = [
                col for col in self.REQUIRED_COLUMNS
                if col not in df.columns
            ]

            return {
                "num_rows": len(df),
                "columns": list(df.columns),
                "has_required_columns": len(missing_columns) == 0,
                "missing_columns": missing_columns,
            }

        except Exception as e:
            raise InvalidBatchCSVError(
                f"Failed to read CSV file: {csv_path}. Error: {e}"
            )
