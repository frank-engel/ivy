"""Tests for BatchCSVParser service."""

import pytest
import tempfile
import shutil
from pathlib import Path

from image_velocimetry_tools.services.batch_csv_parser import BatchCSVParser
from image_velocimetry_tools.batch.models import BatchJob
from image_velocimetry_tools.batch.exceptions import InvalidBatchCSVError


@pytest.fixture
def csv_parser():
    """Fixture providing a BatchCSVParser instance."""
    return BatchCSVParser()


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory that's cleaned up after use."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_csv_minimal(temp_dir):
    """Fixture providing a minimal valid CSV file."""
    csv_path = Path(temp_dir) / "batch_minimal.csv"
    csv_content = """video_path,water_surface_elevation
videos/test1.mp4,318.211
videos/test2.mp4,318.6
"""
    csv_path.write_text(csv_content)
    return str(csv_path)


@pytest.fixture
def sample_csv_full(temp_dir):
    """Fixture providing a full CSV file with all columns."""
    csv_path = Path(temp_dir) / "batch_full.csv"
    csv_content = """video_path,water_surface_elevation,start_time,end_time,alpha,measurement_number,measurement_date,gage_height,comments
videos/test1.mp4,318.211,15,20,0.88,1,2017-06-30,318.211,Test comment
videos/test2.mp4,318.6,00:00:15,00:00:20,0.93,2,2017-07-11,318.6,Another comment
"""
    csv_path.write_text(csv_content)
    return str(csv_path)


class TestBatchCSVParserParseCSV:
    """Tests for parse_csv method."""

    def test_parse_minimal_csv(self, csv_parser, sample_csv_minimal):
        """Test parsing a minimal CSV with only required columns."""
        jobs = csv_parser.parse_csv(sample_csv_minimal)

        assert len(jobs) == 2
        assert all(isinstance(job, BatchJob) for job in jobs)

        # Check first job
        assert jobs[0].job_id == "job_001"
        assert jobs[0].video_path == "videos/test1.mp4"
        assert jobs[0].water_surface_elevation == 318.211
        assert jobs[0].alpha == 0.85  # Default value
        assert jobs[0].start_time is None
        assert jobs[0].end_time is None

        # Check second job
        assert jobs[1].job_id == "job_002"
        assert jobs[1].video_path == "videos/test2.mp4"
        assert jobs[1].water_surface_elevation == 318.6

    def test_parse_full_csv(self, csv_parser, sample_csv_full):
        """Test parsing a CSV with all optional columns."""
        jobs = csv_parser.parse_csv(sample_csv_full)

        assert len(jobs) == 2

        # Check first job with all fields
        job = jobs[0]
        assert job.job_id == "job_001"
        assert job.video_path == "videos/test1.mp4"
        assert job.water_surface_elevation == 318.211
        assert job.start_time == "15"
        assert job.start_time_seconds == 15.0
        assert job.end_time == "20"
        assert job.end_time_seconds == 20.0
        assert job.alpha == 0.88
        assert job.measurement_number == 1
        assert job.measurement_date == "2017-06-30"
        assert job.gage_height == 318.211
        assert job.comments == "Test comment"

        # Check second job with time format hh:mm:ss
        job = jobs[1]
        assert job.start_time == "00:00:15"
        assert job.start_time_seconds == 15.0
        assert job.end_time == "00:00:20"
        assert job.end_time_seconds == 20.0

    def test_parse_csv_with_empty_rows(self, csv_parser, temp_dir):
        """Test that empty rows are skipped."""
        csv_path = Path(temp_dir) / "batch_empty_rows.csv"
        csv_content = """video_path,water_surface_elevation
videos/test1.mp4,318.211

videos/test2.mp4,318.6

"""
        csv_path.write_text(csv_content)

        jobs = csv_parser.parse_csv(str(csv_path))

        assert len(jobs) == 2  # Empty rows should be skipped

    def test_parse_nonexistent_file_raises_error(self, csv_parser, temp_dir):
        """Test that parsing non-existent file raises InvalidBatchCSVError."""
        csv_path = Path(temp_dir) / "does_not_exist.csv"

        with pytest.raises(InvalidBatchCSVError, match="does not exist"):
            csv_parser.parse_csv(str(csv_path))

    def test_parse_empty_file_raises_error(self, csv_parser, temp_dir):
        """Test that parsing empty file raises InvalidBatchCSVError."""
        csv_path = Path(temp_dir) / "empty.csv"
        csv_path.write_text("")

        with pytest.raises(InvalidBatchCSVError, match="empty"):
            csv_parser.parse_csv(str(csv_path))

    def test_parse_missing_required_column_raises_error(self, csv_parser, temp_dir):
        """Test that missing required columns raises InvalidBatchCSVError."""
        csv_path = Path(temp_dir) / "missing_column.csv"
        csv_content = """video_path
videos/test1.mp4
"""
        csv_path.write_text(csv_content)

        with pytest.raises(InvalidBatchCSVError, match="missing required columns"):
            csv_parser.parse_csv(str(csv_path))

    def test_parse_invalid_numeric_field_raises_error(self, csv_parser, temp_dir):
        """Test that invalid numeric values raise InvalidBatchCSVError."""
        csv_path = Path(temp_dir) / "invalid_numeric.csv"
        csv_content = """video_path,water_surface_elevation
videos/test1.mp4,not_a_number
"""
        csv_path.write_text(csv_content)

        with pytest.raises(InvalidBatchCSVError, match="must be numeric"):
            csv_parser.parse_csv(str(csv_path))

    def test_parse_invalid_alpha_raises_error(self, csv_parser, temp_dir):
        """Test that alpha outside valid range raises InvalidBatchCSVError."""
        csv_path = Path(temp_dir) / "invalid_alpha.csv"
        csv_content = """video_path,water_surface_elevation,alpha
videos/test1.mp4,318.211,1.5
"""
        csv_path.write_text(csv_content)

        with pytest.raises(InvalidBatchCSVError, match="alpha must be between 0 and 1"):
            csv_parser.parse_csv(str(csv_path))

    def test_parse_invalid_time_format_raises_error(self, csv_parser, temp_dir):
        """Test that invalid time format raises InvalidBatchCSVError."""
        csv_path = Path(temp_dir) / "invalid_time.csv"
        csv_content = """video_path,water_surface_elevation,start_time
videos/test1.mp4,318.211,invalid_time
"""
        csv_path.write_text(csv_content)

        with pytest.raises(InvalidBatchCSVError, match="Invalid time format"):
            csv_parser.parse_csv(str(csv_path))

    def test_parse_end_time_before_start_time_raises_error(self, csv_parser, temp_dir):
        """Test that end_time < start_time raises InvalidBatchCSVError."""
        csv_path = Path(temp_dir) / "invalid_time_order.csv"
        csv_content = """video_path,water_surface_elevation,start_time,end_time
videos/test1.mp4,318.211,20,15
"""
        csv_path.write_text(csv_content)

        with pytest.raises(InvalidBatchCSVError, match="must be greater than"):
            csv_parser.parse_csv(str(csv_path))

    def test_parse_csv_with_whitespace_in_values(self, csv_parser, temp_dir):
        """Test that whitespace in values is handled correctly."""
        csv_path = Path(temp_dir) / "whitespace.csv"
        csv_content = """video_path,water_surface_elevation,comments
  videos/test1.mp4  ,  318.211  ,  Test comment
"""
        csv_path.write_text(csv_content)

        jobs = csv_parser.parse_csv(str(csv_path))

        assert len(jobs) == 1
        # Whitespace should be stripped from strings
        assert jobs[0].video_path == "videos/test1.mp4"
        assert jobs[0].comments == "Test comment"

    def test_parse_csv_with_nan_optional_fields(self, csv_parser, temp_dir):
        """Test that NaN/empty optional fields are handled as None."""
        csv_path = Path(temp_dir) / "nan_fields.csv"
        csv_content = """video_path,water_surface_elevation,start_time,comments
videos/test1.mp4,318.211,,
"""
        csv_path.write_text(csv_content)

        jobs = csv_parser.parse_csv(str(csv_path))

        assert len(jobs) == 1
        assert jobs[0].start_time is None
        assert jobs[0].comments is None

    def test_parse_malformed_csv_raises_error(self, csv_parser, temp_dir):
        """Test that malformed CSV raises InvalidBatchCSVError."""
        csv_path = Path(temp_dir) / "malformed.csv"
        csv_content = """video_path,water_surface_elevation
videos/test1.mp4,318.211,"unclosed quote
"""
        csv_path.write_text(csv_content)

        with pytest.raises(InvalidBatchCSVError):
            csv_parser.parse_csv(str(csv_path))


class TestBatchCSVParserGetCSVInfo:
    """Tests for get_csv_info method."""

    def test_get_info_valid_csv(self, csv_parser, sample_csv_full):
        """Test getting info from a valid CSV file."""
        info = csv_parser.get_csv_info(sample_csv_full)

        assert info["num_rows"] == 2
        assert "video_path" in info["columns"]
        assert "water_surface_elevation" in info["columns"]
        assert info["has_required_columns"] is True
        assert len(info["missing_columns"]) == 0

    def test_get_info_missing_required_columns(self, csv_parser, temp_dir):
        """Test get_csv_info detects missing required columns."""
        csv_path = Path(temp_dir) / "incomplete.csv"
        csv_content = """video_path
videos/test1.mp4
"""
        csv_path.write_text(csv_content)

        info = csv_parser.get_csv_info(str(csv_path))

        assert info["has_required_columns"] is False
        assert "water_surface_elevation" in info["missing_columns"]

    def test_get_info_nonexistent_file_raises_error(self, csv_parser, temp_dir):
        """Test that get_csv_info raises error for non-existent file."""
        csv_path = Path(temp_dir) / "does_not_exist.csv"

        with pytest.raises(InvalidBatchCSVError, match="Failed to read"):
            csv_parser.get_csv_info(str(csv_path))


class TestBatchCSVParserFieldExtraction:
    """Tests for internal field extraction methods."""

    def test_parse_integer_field(self, csv_parser, temp_dir):
        """Test that integer fields are parsed correctly."""
        csv_path = Path(temp_dir) / "integer_field.csv"
        csv_content = """video_path,water_surface_elevation,measurement_number
videos/test1.mp4,318.211,42
"""
        csv_path.write_text(csv_content)

        jobs = csv_parser.parse_csv(str(csv_path))

        assert jobs[0].measurement_number == 42
        assert isinstance(jobs[0].measurement_number, int)

    def test_parse_float_field(self, csv_parser, temp_dir):
        """Test that float fields are parsed correctly."""
        csv_path = Path(temp_dir) / "float_field.csv"
        csv_content = """video_path,water_surface_elevation,gage_height
videos/test1.mp4,318.211,318.5
"""
        csv_path.write_text(csv_content)

        jobs = csv_parser.parse_csv(str(csv_path))

        assert jobs[0].gage_height == 318.5
        assert isinstance(jobs[0].gage_height, float)

    def test_parse_with_negative_measurement_number_raises_error(self, csv_parser, temp_dir):
        """Test that negative measurement_number raises error."""
        csv_path = Path(temp_dir) / "negative_measurement.csv"
        csv_content = """video_path,water_surface_elevation,measurement_number
videos/test1.mp4,318.211,-1
"""
        csv_path.write_text(csv_content)

        with pytest.raises(InvalidBatchCSVError, match="must be non-negative"):
            csv_parser.parse_csv(str(csv_path))


class TestBatchCSVParserValidation:
    """Tests for CSV validation logic."""

    def test_required_columns_constant(self, csv_parser):
        """Test that REQUIRED_COLUMNS constant is defined correctly."""
        assert "video_path" in csv_parser.REQUIRED_COLUMNS
        assert "water_surface_elevation" in csv_parser.REQUIRED_COLUMNS
        assert len(csv_parser.REQUIRED_COLUMNS) == 2

    def test_optional_columns_constant(self, csv_parser):
        """Test that OPTIONAL_COLUMNS constant includes expected fields."""
        assert "alpha" in csv_parser.OPTIONAL_COLUMNS
        assert "start_time" in csv_parser.OPTIONAL_COLUMNS
        assert "end_time" in csv_parser.OPTIONAL_COLUMNS
        assert csv_parser.OPTIONAL_COLUMNS["alpha"] == 0.85  # Default value
