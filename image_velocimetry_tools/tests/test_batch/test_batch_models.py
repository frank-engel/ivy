"""Tests for batch processing models."""

import pytest
import tempfile
import os
from pathlib import Path

from image_velocimetry_tools.batch.models import (
    BatchJob,
    JobStatus,
    BatchConfig,
)
from image_velocimetry_tools.batch.exceptions import BatchValidationError


class TestBatchJob:
    """Tests for BatchJob model."""

    def test_create_minimal_job(self):
        """Test creating a job with minimal required fields."""
        job = BatchJob(
            job_id="test_001",
            video_path="/path/to/video.mp4",
            water_surface_elevation=318.211,
        )

        assert job.job_id == "test_001"
        assert job.video_path == "/path/to/video.mp4"
        assert job.water_surface_elevation == 318.211
        assert job.alpha == 0.85  # Default value
        assert job.status == JobStatus.PENDING
        assert job.discharge_value is None
        assert job.error_message is None

    def test_create_full_job(self):
        """Test creating a job with all fields."""
        job = BatchJob(
            job_id="test_002",
            video_path="/path/to/video.mp4",
            water_surface_elevation=318.6,
            start_time="00:00:15",
            end_time="00:00:20",
            alpha=0.88,
            measurement_number=3,
            measurement_date="2025-04-06",
            measurement_time="10:35:00",
            gage_height=318.6,
            comments="Test measurement",
        )

        assert job.alpha == 0.88
        assert job.measurement_number == 3
        assert job.comments == "Test measurement"

    def test_time_parsing_seconds_only(self):
        """Test parsing time strings with seconds only."""
        job = BatchJob(
            job_id="test",
            video_path="/test.mp4",
            water_surface_elevation=100.0,
            start_time="15",
            end_time="20",
        )

        assert job.start_time_seconds == 15.0
        assert job.end_time_seconds == 20.0

    def test_time_parsing_with_decimals(self):
        """Test parsing time strings with decimal seconds."""
        job = BatchJob(
            job_id="test",
            video_path="/test.mp4",
            water_surface_elevation=100.0,
            start_time="15.5",
            end_time="20.75",
        )

        assert job.start_time_seconds == 15.5
        assert job.end_time_seconds == 20.75

    def test_time_parsing_mmss_format(self):
        """Test parsing time strings in mm:ss format."""
        job = BatchJob(
            job_id="test",
            video_path="/test.mp4",
            water_surface_elevation=100.0,
            start_time="01:30",
            end_time="02:45",
        )

        assert job.start_time_seconds == 90.0  # 1*60 + 30
        assert job.end_time_seconds == 165.0  # 2*60 + 45

    def test_time_parsing_hhmmss_format(self):
        """Test parsing time strings in hh:mm:ss format."""
        job = BatchJob(
            job_id="test",
            video_path="/test.mp4",
            water_surface_elevation=100.0,
            start_time="00:01:30",
            end_time="00:02:45.5",
        )

        assert job.start_time_seconds == 90.0  # 0*3600 + 1*60 + 30
        assert job.end_time_seconds == 165.5  # 0*3600 + 2*60 + 45.5

    def test_end_time_before_start_time_raises_error(self):
        """Test that end_time < start_time raises ValueError."""
        with pytest.raises(ValueError, match="end_time.*must be greater than"):
            BatchJob(
                job_id="test",
                video_path="/test.mp4",
                water_surface_elevation=100.0,
                start_time="20",
                end_time="15",
            )

    def test_invalid_time_format_raises_error(self):
        """Test that invalid time format raises ValueError."""
        with pytest.raises(ValueError, match="has invalid format"):
            BatchJob(
                job_id="test",
                video_path="/test.mp4",
                water_surface_elevation=100.0,
                start_time="1:2:3:4",  # Too many colons
            )

    def test_empty_job_id_raises_error(self):
        """Test that empty job_id raises ValueError."""
        with pytest.raises(ValueError, match="job_id cannot be empty"):
            BatchJob(
                job_id="",
                video_path="/test.mp4",
                water_surface_elevation=100.0,
            )

    def test_empty_video_path_raises_error(self):
        """Test that empty video_path raises ValueError."""
        with pytest.raises(ValueError, match="video_path cannot be empty"):
            BatchJob(
                job_id="test", video_path="", water_surface_elevation=100.0
            )

    def test_invalid_alpha_raises_error(self):
        """Test that alpha outside valid range raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            BatchJob(
                job_id="test",
                video_path="/test.mp4",
                water_surface_elevation=100.0,
                alpha=1.5,
            )

        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            BatchJob(
                job_id="test",
                video_path="/test.mp4",
                water_surface_elevation=100.0,
                alpha=0.0,
            )

    def test_negative_measurement_number_raises_error(self):
        """Test that negative measurement_number raises ValueError."""
        with pytest.raises(
            ValueError, match="measurement_number must be non-negative"
        ):
            BatchJob(
                job_id="test",
                video_path="/test.mp4",
                water_surface_elevation=100.0,
                measurement_number=-1,
            )

    def test_mark_processing(self):
        """Test marking job as processing."""
        job = BatchJob(
            job_id="test",
            video_path="/test.mp4",
            water_surface_elevation=100.0,
        )

        job.mark_processing()
        assert job.status == JobStatus.PROCESSING

    def test_mark_completed(self):
        """Test marking job as completed with results."""
        job = BatchJob(
            job_id="test",
            video_path="/test.mp4",
            water_surface_elevation=100.0,
        )

        job.mark_completed(discharge_value=5.32, processing_time=120.5)

        assert job.status == JobStatus.COMPLETED
        assert job.discharge_value == 5.32
        assert job.processing_time == 120.5
        assert job.error_message is None

    def test_mark_failed(self):
        """Test marking job as failed with error message."""
        job = BatchJob(
            job_id="test",
            video_path="/test.mp4",
            water_surface_elevation=100.0,
        )

        job.mark_failed(
            error_message="Video file not found", processing_time=5.0
        )

        assert job.status == JobStatus.FAILED
        assert job.error_message == "Video file not found"
        assert job.processing_time == 5.0
        assert job.discharge_value is None

    def test_to_dict(self):
        """Test converting job to dictionary."""
        job = BatchJob(
            job_id="test_003",
            video_path="/path/to/video.mp4",
            water_surface_elevation=318.6,
            start_time="15",
            end_time="20",
            alpha=0.88,
            comments="Test",
        )

        job_dict = job.to_dict()

        assert job_dict["job_id"] == "test_003"
        assert job_dict["video_path"] == "/path/to/video.mp4"
        assert job_dict["water_surface_elevation"] == 318.6
        assert job_dict["start_time_seconds"] == 15.0
        assert job_dict["end_time_seconds"] == 20.0
        assert job_dict["alpha"] == 0.88
        assert job_dict["comments"] == "Test"
        assert job_dict["status"] == "pending"

    def test_get_video_filename(self):
        """Test extracting video filename."""
        job = BatchJob(
            job_id="test",
            video_path="/path/to/my_video.mp4",
            water_surface_elevation=100.0,
        )

        assert job.get_video_filename() == "my_video"

    def test_repr(self):
        """Test string representation."""
        job = BatchJob(
            job_id="test_004",
            video_path="/path/to/video.mp4",
            water_surface_elevation=100.0,
        )

        repr_str = repr(job)
        assert "test_004" in repr_str
        assert "video" in repr_str
        assert "pending" in repr_str


class TestBatchConfig:
    """Tests for BatchConfig model."""

    @pytest.fixture
    def temp_files(self):
        """Create temporary files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test files
            scaffold_file = tmpdir_path / "scaffold.ivy"
            scaffold_file.write_text("test scaffold content")

            csv_file = tmpdir_path / "batch.csv"
            csv_file.write_text("video_path,water_surface_elevation\n")

            output_dir = tmpdir_path / "output"

            yield {
                "scaffold": str(scaffold_file),
                "csv": str(csv_file),
                "output": str(output_dir),
                "tmpdir": tmpdir_path,
            }

    def test_create_config(self, temp_files):
        """Test creating a valid configuration."""
        config = BatchConfig(
            scaffold_path=temp_files["scaffold"],
            batch_csv_path=temp_files["csv"],
            output_dir=temp_files["output"],
        )

        assert config.scaffold_path == temp_files["scaffold"]
        assert config.batch_csv_path == temp_files["csv"]
        assert config.output_dir == temp_files["output"]
        assert config.stop_on_first_failure is False

    def test_config_with_stop_on_failure(self, temp_files):
        """Test creating config with stop_on_first_failure flag."""
        config = BatchConfig(
            scaffold_path=temp_files["scaffold"],
            batch_csv_path=temp_files["csv"],
            output_dir=temp_files["output"],
            stop_on_first_failure=True,
        )

        assert config.stop_on_first_failure is True

    def test_validate_valid_config(self, temp_files):
        """Test validating a valid configuration."""
        config = BatchConfig(
            scaffold_path=temp_files["scaffold"],
            batch_csv_path=temp_files["csv"],
            output_dir=temp_files["output"],
        )

        errors = config.validate()
        assert errors == []

    def test_validate_missing_scaffold(self, temp_files):
        """Test validation fails when scaffold file missing."""
        config = BatchConfig(
            scaffold_path="/nonexistent/scaffold.ivy",
            batch_csv_path=temp_files["csv"],
            output_dir=temp_files["output"],
        )

        errors = config.validate()
        assert len(errors) > 0
        assert any("Scaffold file does not exist" in err for err in errors)

    def test_validate_missing_csv(self, temp_files):
        """Test validation fails when CSV file missing."""
        config = BatchConfig(
            scaffold_path=temp_files["scaffold"],
            batch_csv_path="/nonexistent/batch.csv",
            output_dir=temp_files["output"],
        )

        errors = config.validate()
        assert len(errors) > 0
        assert any("Batch CSV file does not exist" in err for err in errors)

    def test_validate_wrong_scaffold_extension(self, temp_files):
        """Test validation fails for wrong scaffold extension."""
        wrong_file = temp_files["tmpdir"] / "scaffold.txt"
        wrong_file.write_text("test")

        config = BatchConfig(
            scaffold_path=str(wrong_file),
            batch_csv_path=temp_files["csv"],
            output_dir=temp_files["output"],
        )

        errors = config.validate()
        assert len(errors) > 0
        assert any(".ivy extension" in err for err in errors)

    def test_validate_wrong_csv_extension(self, temp_files):
        """Test validation fails for wrong CSV extension."""
        wrong_file = temp_files["tmpdir"] / "batch.txt"
        wrong_file.write_text("test")

        config = BatchConfig(
            scaffold_path=temp_files["scaffold"],
            batch_csv_path=str(wrong_file),
            output_dir=temp_files["output"],
        )

        errors = config.validate()
        assert len(errors) > 0
        assert any(".csv extension" in err for err in errors)

    def test_validate_empty_paths(self):
        """Test validation fails for empty paths."""
        config = BatchConfig(
            scaffold_path="", batch_csv_path="", output_dir=""
        )

        errors = config.validate()
        assert len(errors) >= 3
        assert any("scaffold_path cannot be empty" in err for err in errors)
        assert any("batch_csv_path cannot be empty" in err for err in errors)
        assert any("output_dir cannot be empty" in err for err in errors)

    def test_create_output_directory(self, temp_files):
        """Test creating output directory."""
        config = BatchConfig(
            scaffold_path=temp_files["scaffold"],
            batch_csv_path=temp_files["csv"],
            output_dir=temp_files["output"],
        )

        assert not config.output_dir_resolved.exists()
        config.create_output_directory()
        assert config.output_dir_resolved.exists()
        assert config.output_dir_resolved.is_dir()

    def test_get_job_output_dir(self, temp_files):
        """Test getting job-specific output directory."""
        config = BatchConfig(
            scaffold_path=temp_files["scaffold"],
            batch_csv_path=temp_files["csv"],
            output_dir=temp_files["output"],
        )

        job_dir = config.get_job_output_dir("test_001")
        assert "job_test_001" in str(job_dir)
        assert job_dir.parent == config.output_dir_resolved

    def test_get_job_output_dir_sanitizes_special_chars(self, temp_files):
        """Test job ID sanitization for filesystem."""
        config = BatchConfig(
            scaffold_path=temp_files["scaffold"],
            batch_csv_path=temp_files["csv"],
            output_dir=temp_files["output"],
        )

        job_dir = config.get_job_output_dir("test/with:special*chars")
        # Should replace invalid characters with underscores
        assert "/" not in job_dir.name
        assert ":" not in job_dir.name
        assert "*" not in job_dir.name

    def test_get_batch_summary_path(self, temp_files):
        """Test getting batch summary file path."""
        config = BatchConfig(
            scaffold_path=temp_files["scaffold"],
            batch_csv_path=temp_files["csv"],
            output_dir=temp_files["output"],
        )

        summary_path = config.get_batch_summary_path()
        assert summary_path.name == "batch_summary.csv"
        assert summary_path.parent == config.output_dir_resolved

    def test_get_batch_log_path(self, temp_files):
        """Test getting batch log file path."""
        config = BatchConfig(
            scaffold_path=temp_files["scaffold"],
            batch_csv_path=temp_files["csv"],
            output_dir=temp_files["output"],
        )

        log_path = config.get_batch_log_path()
        assert log_path.name == "batch.log"
        assert log_path.parent == config.output_dir_resolved

    def test_resolve_absolute_video_path(self, temp_files):
        """Test resolving absolute video path."""
        config = BatchConfig(
            scaffold_path=temp_files["scaffold"],
            batch_csv_path=temp_files["csv"],
            output_dir=temp_files["output"],
        )

        video_path = "/absolute/path/to/video.mp4"
        resolved = config.resolve_video_path(video_path)
        # On Windows, absolute paths include drive letter (e.g., C:/)
        # Use as_posix() to normalize path separators for comparison
        assert resolved.as_posix().endswith("absolute/path/to/video.mp4")

    def test_resolve_relative_video_path(self, temp_files):
        """Test resolving relative video path (relative to CSV location)."""
        config = BatchConfig(
            scaffold_path=temp_files["scaffold"],
            batch_csv_path=temp_files["csv"],
            output_dir=temp_files["output"],
        )

        # Create a video file relative to CSV
        video_file = temp_files["tmpdir"] / "videos" / "test.mp4"
        video_file.parent.mkdir(exist_ok=True)
        video_file.write_text("test")

        resolved = config.resolve_video_path("videos/test.mp4")
        assert resolved.exists()
        assert resolved.name == "test.mp4"

    def test_repr(self, temp_files):
        """Test string representation."""
        config = BatchConfig(
            scaffold_path=temp_files["scaffold"],
            batch_csv_path=temp_files["csv"],
            output_dir=temp_files["output"],
        )

        repr_str = repr(config)
        assert "BatchConfig" in repr_str
        assert "scaffold" in repr_str
        assert "batch_csv" in repr_str
        assert "output_dir" in repr_str
