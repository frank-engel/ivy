"""Tests for BatchProcessor service.

Note: These are unit tests focusing on the batch processor orchestration logic.
Full end-to-end integration tests should be done separately.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import csv

from image_velocimetry_tools.services.batch_processor import BatchProcessor
from image_velocimetry_tools.batch.models import BatchJob, JobStatus
from image_velocimetry_tools.batch.exceptions import (
    BatchProcessingError,
    BatchValidationError,
    InvalidBatchCSVError,
    InvalidScaffoldError,
)


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory that's cleaned up after use."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def valid_scaffold_file(temp_dir):
    """Fixture providing a valid scaffold file path."""
    import zipfile
    import json

    scaffold_path = Path(temp_dir) / "scaffold.ivy"

    # Create a minimal valid scaffold
    with zipfile.ZipFile(scaffold_path, 'w') as zipf:
        project_data = {
            "rectification_parameters": {
                "method": "camera matrix",
                "ground_control_points": [[0, 0, 100]] * 6,
                "image_control_points": [[100, 100]] * 6,
            },
            "grid_parameters": {"use_cross_section_line": True},
            "stiv_parameters": {
                "num_pixels": 20,
                "phi_origin": 90,
                "d_phi": 1.0,
                "phi_range": 90,
                "max_vel_threshold_mps": 10.0,
            },
            "cross_section_geometry_path": "5-discharge/xs.mat",
            "ffmpeg_parameters": {},
        }
        zipf.writestr("project_data.json", json.dumps(project_data))
        zipf.writestr("5-discharge/cross_section.mat", "dummy")

    return str(scaffold_path)


@pytest.fixture
def valid_batch_csv(temp_dir):
    """Fixture providing a valid batch CSV file."""
    csv_path = Path(temp_dir) / "batch.csv"
    csv_content = """video_path,water_surface_elevation
videos/test1.mp4,318.211
videos/test2.mp4,318.6
"""
    csv_path.write_text(csv_content)
    return str(csv_path)


@pytest.fixture
def output_dir(temp_dir):
    """Fixture providing an output directory path."""
    return str(Path(temp_dir) / "output")


class TestBatchProcessorInitialization:
    """Tests for BatchProcessor initialization."""

    def test_init_with_valid_config(self, valid_scaffold_file, valid_batch_csv, output_dir):
        """Test initialization with valid configuration."""
        processor = BatchProcessor(
            scaffold_path=valid_scaffold_file,
            batch_csv_path=valid_batch_csv,
            output_dir=output_dir
        )

        assert processor.config is not None
        assert processor.config.scaffold_path == valid_scaffold_file
        assert processor.config.batch_csv_path == valid_batch_csv
        assert processor.config.output_dir == output_dir
        assert processor.config.stop_on_first_failure is False

    def test_init_with_stop_on_first_failure(
        self, valid_scaffold_file, valid_batch_csv, output_dir
    ):
        """Test initialization with stop_on_first_failure flag."""
        processor = BatchProcessor(
            scaffold_path=valid_scaffold_file,
            batch_csv_path=valid_batch_csv,
            output_dir=output_dir,
            stop_on_first_failure=True
        )

        assert processor.config.stop_on_first_failure is True

    def test_init_with_invalid_scaffold_raises_error(self, valid_batch_csv, output_dir, temp_dir):
        """Test that invalid scaffold path raises BatchValidationError."""
        invalid_scaffold = str(Path(temp_dir) / "nonexistent.ivy")

        with pytest.raises(BatchValidationError, match="Configuration validation failed"):
            BatchProcessor(
                scaffold_path=invalid_scaffold,
                batch_csv_path=valid_batch_csv,
                output_dir=output_dir
            )

    def test_init_with_invalid_csv_raises_error(self, valid_scaffold_file, output_dir, temp_dir):
        """Test that invalid CSV path raises BatchValidationError."""
        invalid_csv = str(Path(temp_dir) / "nonexistent.csv")

        with pytest.raises(BatchValidationError, match="Configuration validation failed"):
            BatchProcessor(
                scaffold_path=valid_scaffold_file,
                batch_csv_path=invalid_csv,
                output_dir=output_dir
            )

    def test_init_initializes_services(self, valid_scaffold_file, valid_batch_csv, output_dir):
        """Test that initialization creates service instances."""
        processor = BatchProcessor(
            scaffold_path=valid_scaffold_file,
            batch_csv_path=valid_batch_csv,
            output_dir=output_dir
        )

        assert hasattr(processor, 'csv_parser')
        assert hasattr(processor, 'scaffold_loader')
        assert hasattr(processor, 'job_executor')
        assert processor.csv_parser is not None
        assert processor.scaffold_loader is not None
        assert processor.job_executor is not None


class TestBatchProcessorSetup:
    """Tests for batch processor setup methods."""

    @patch.object(BatchProcessor, '_cleanup')
    @patch.object(BatchProcessor, '_generate_summary')
    @patch.object(BatchProcessor, '_execute_all_jobs')
    @patch.object(BatchProcessor, '_parse_batch_csv')
    @patch.object(BatchProcessor, '_load_scaffold')
    def test_setup_output_directory(
        self,
        mock_load_scaffold,
        mock_parse_csv,
        mock_execute,
        mock_summary,
        mock_cleanup,
        valid_scaffold_file,
        valid_batch_csv,
        output_dir
    ):
        """Test that _setup_output_directory creates the directory."""
        processor = BatchProcessor(
            scaffold_path=valid_scaffold_file,
            batch_csv_path=valid_batch_csv,
            output_dir=output_dir
        )

        # Mock methods to avoid full execution
        mock_load_scaffold.return_value = None
        mock_parse_csv.return_value = None
        mock_execute.return_value = None

        processor.run()

        # Verify output directory was created
        assert Path(output_dir).exists()


class TestBatchProcessorRun:
    """Tests for run method."""

    @patch('image_velocimetry_tools.services.batch_processor.BatchCSVParser.parse_csv')
    @patch('image_velocimetry_tools.services.batch_processor.ScaffoldLoader.load_scaffold')
    @patch('image_velocimetry_tools.services.batch_processor.JobExecutor.execute_job')
    def test_run_successful_batch(
        self,
        mock_execute_job,
        mock_load_scaffold,
        mock_parse_csv,
        valid_scaffold_file,
        valid_batch_csv,
        output_dir
    ):
        """Test successful batch processing run."""
        # Mock scaffold loading
        mock_load_scaffold.return_value = {
            "project_data": {},
            "extract_dir": "/tmp/scaffold",
            "cross_section_path": "/tmp/scaffold/xs.mat",
        }

        # Mock CSV parsing
        mock_jobs = [
            BatchJob(
                job_id="test_001",
                video_path="videos/test1.mp4",
                water_surface_elevation=318.211
            ),
            BatchJob(
                job_id="test_002",
                video_path="videos/test2.mp4",
                water_surface_elevation=318.6
            )
        ]
        mock_parse_csv.return_value = mock_jobs

        # Mock job execution
        mock_execute_job.return_value = {
            "discharge": 5.32,
            "area": 12.5,
            "processing_time": 120.0,
            "job_output_dir": "/tmp/output/job_001"
        }

        # Run batch processor
        processor = BatchProcessor(
            scaffold_path=valid_scaffold_file,
            batch_csv_path=valid_batch_csv,
            output_dir=output_dir
        )

        jobs = processor.run()

        # Verify results
        assert len(jobs) == 2
        assert all(job.status == JobStatus.COMPLETED for job in jobs)
        assert mock_execute_job.call_count == 2

        # Verify summary was generated
        summary_path = Path(output_dir) / "batch_summary.csv"
        assert summary_path.exists()

    @patch('image_velocimetry_tools.services.batch_processor.BatchCSVParser.parse_csv')
    @patch('image_velocimetry_tools.services.batch_processor.ScaffoldLoader.load_scaffold')
    @patch('image_velocimetry_tools.services.batch_processor.JobExecutor.execute_job')
    def test_run_with_job_failures(
        self,
        mock_execute_job,
        mock_load_scaffold,
        mock_parse_csv,
        valid_scaffold_file,
        valid_batch_csv,
        output_dir
    ):
        """Test batch processing with some job failures."""
        from image_velocimetry_tools.batch.exceptions import JobExecutionError

        # Mock scaffold loading
        mock_load_scaffold.return_value = {
            "project_data": {},
            "extract_dir": "/tmp/scaffold",
            "cross_section_path": "/tmp/scaffold/xs.mat",
        }

        # Mock CSV parsing
        mock_jobs = [
            BatchJob(
                job_id="test_001",
                video_path="videos/test1.mp4",
                water_surface_elevation=318.211
            ),
            BatchJob(
                job_id="test_002",
                video_path="videos/test2.mp4",
                water_surface_elevation=318.6
            )
        ]
        mock_parse_csv.return_value = mock_jobs

        # First job succeeds, second job fails
        mock_execute_job.side_effect = [
            {
                "discharge": 5.32,
                "area": 12.5,
                "processing_time": 120.0,
                "job_output_dir": "/tmp/output/job_001"
            },
            JobExecutionError("Video file not found")
        ]

        # Run batch processor
        processor = BatchProcessor(
            scaffold_path=valid_scaffold_file,
            batch_csv_path=valid_batch_csv,
            output_dir=output_dir
        )

        jobs = processor.run()

        # Verify results
        assert len(jobs) == 2
        assert jobs[0].status == JobStatus.COMPLETED
        assert jobs[1].status == JobStatus.FAILED
        assert jobs[1].error_message is not None

    @patch('image_velocimetry_tools.services.batch_processor.BatchCSVParser.parse_csv')
    @patch('image_velocimetry_tools.services.batch_processor.ScaffoldLoader.load_scaffold')
    @patch('image_velocimetry_tools.services.batch_processor.JobExecutor.execute_job')
    def test_run_with_stop_on_first_failure(
        self,
        mock_execute_job,
        mock_load_scaffold,
        mock_parse_csv,
        valid_scaffold_file,
        valid_batch_csv,
        output_dir
    ):
        """Test batch processing stops on first failure when configured."""
        from image_velocimetry_tools.batch.exceptions import JobExecutionError

        # Mock scaffold loading
        mock_load_scaffold.return_value = {
            "project_data": {},
            "extract_dir": "/tmp/scaffold",
            "cross_section_path": "/tmp/scaffold/xs.mat",
        }

        # Mock CSV parsing - 3 jobs
        mock_jobs = [
            BatchJob(job_id=f"test_{i:03d}", video_path=f"video{i}.mp4",
                    water_surface_elevation=318.0)
            for i in range(1, 4)
        ]
        mock_parse_csv.return_value = mock_jobs

        # First job fails
        mock_execute_job.side_effect = JobExecutionError("Video file not found")

        # Run batch processor with stop_on_first_failure=True
        processor = BatchProcessor(
            scaffold_path=valid_scaffold_file,
            batch_csv_path=valid_batch_csv,
            output_dir=output_dir,
            stop_on_first_failure=True
        )

        jobs = processor.run()

        # Only first job should have been attempted
        assert mock_execute_job.call_count == 1
        assert jobs[0].status == JobStatus.FAILED
        assert jobs[1].status == JobStatus.PENDING  # Not executed
        assert jobs[2].status == JobStatus.PENDING  # Not executed


class TestBatchProcessorGenerateSummary:
    """Tests for _generate_summary method."""

    @patch('image_velocimetry_tools.services.batch_processor.BatchCSVParser.parse_csv')
    @patch('image_velocimetry_tools.services.batch_processor.ScaffoldLoader.load_scaffold')
    @patch('image_velocimetry_tools.services.batch_processor.JobExecutor.execute_job')
    def test_generate_summary_creates_csv(
        self,
        mock_execute_job,
        mock_load_scaffold,
        mock_parse_csv,
        valid_scaffold_file,
        valid_batch_csv,
        output_dir
    ):
        """Test that summary CSV is generated correctly."""
        # Mock scaffold loading
        mock_load_scaffold.return_value = {
            "project_data": {},
            "extract_dir": "/tmp/scaffold",
            "cross_section_path": "/tmp/scaffold/xs.mat",
        }

        # Mock CSV parsing
        mock_jobs = [
            BatchJob(
                job_id="test_001",
                video_path="videos/test1.mp4",
                water_surface_elevation=318.211,
                measurement_number=1,
                comments="Test job"
            )
        ]
        mock_parse_csv.return_value = mock_jobs

        # Mock job execution
        mock_execute_job.return_value = {
            "discharge": 5.32,
            "area": 12.5,
            "processing_time": 120.0,
            "job_output_dir": "/tmp/output/job_001"
        }

        # Run batch processor
        processor = BatchProcessor(
            scaffold_path=valid_scaffold_file,
            batch_csv_path=valid_batch_csv,
            output_dir=output_dir
        )

        processor.run()

        # Verify summary CSV exists
        summary_path = Path(output_dir) / "batch_summary.csv"
        assert summary_path.exists()

        # Read and verify content
        with open(summary_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["job_id"] == "test_001"
        assert rows[0]["status"] == "completed"
        assert float(rows[0]["discharge_m3s"]) == 5.32
        assert rows[0]["comments"] == "Test job"


class TestBatchProcessorGetProgress:
    """Tests for get_progress method."""

    @patch('image_velocimetry_tools.services.batch_processor.ScaffoldLoader.load_scaffold')
    @patch('image_velocimetry_tools.services.batch_processor.BatchCSVParser.parse_csv')
    def test_get_progress_no_jobs(
        self,
        mock_parse_csv,
        mock_load_scaffold,
        valid_scaffold_file,
        valid_batch_csv,
        output_dir
    ):
        """Test get_progress with no jobs."""
        processor = BatchProcessor(
            scaffold_path=valid_scaffold_file,
            batch_csv_path=valid_batch_csv,
            output_dir=output_dir
        )

        progress = processor.get_progress()

        assert progress["total_jobs"] == 0
        assert progress["completed"] == 0
        assert progress["failed"] == 0
        assert progress["pending"] == 0
        assert progress["processing"] == 0
        assert progress["percent_complete"] == 0.0

    def test_get_progress_with_mixed_status(
        self, valid_scaffold_file, valid_batch_csv, output_dir
    ):
        """Test get_progress with jobs in various states."""
        processor = BatchProcessor(
            scaffold_path=valid_scaffold_file,
            batch_csv_path=valid_batch_csv,
            output_dir=output_dir
        )

        # Manually create jobs with different statuses
        processor.jobs = [
            BatchJob(job_id=f"job_{i}", video_path=f"video{i}.mp4",
                    water_surface_elevation=318.0)
            for i in range(5)
        ]

        # Set different statuses
        processor.jobs[0].mark_completed(5.0, 100.0)
        processor.jobs[1].mark_completed(6.0, 110.0)
        processor.jobs[2].mark_failed("Error", 50.0)
        processor.jobs[3].mark_processing()
        # jobs[4] remains pending

        progress = processor.get_progress()

        assert progress["total_jobs"] == 5
        assert progress["completed"] == 2
        assert progress["failed"] == 1
        assert progress["processing"] == 1
        assert progress["pending"] == 1
        assert progress["percent_complete"] == 60.0  # (2 completed + 1 failed) / 5 * 100


class TestBatchProcessorErrorHandling:
    """Tests for error handling in BatchProcessor."""

    @patch('image_velocimetry_tools.services.batch_processor.ScaffoldLoader.load_scaffold')
    def test_run_with_scaffold_load_failure(
        self,
        mock_load_scaffold,
        valid_scaffold_file,
        valid_batch_csv,
        output_dir
    ):
        """Test that scaffold loading failure is handled."""
        mock_load_scaffold.side_effect = InvalidScaffoldError("Invalid scaffold")

        processor = BatchProcessor(
            scaffold_path=valid_scaffold_file,
            batch_csv_path=valid_batch_csv,
            output_dir=output_dir
        )

        with pytest.raises(BatchProcessingError, match="Invalid scaffold"):
            processor.run()

    @patch('image_velocimetry_tools.services.batch_processor.ScaffoldLoader.load_scaffold')
    @patch('image_velocimetry_tools.services.batch_processor.BatchCSVParser.parse_csv')
    def test_run_with_csv_parse_failure(
        self,
        mock_parse_csv,
        mock_load_scaffold,
        valid_scaffold_file,
        valid_batch_csv,
        output_dir
    ):
        """Test that CSV parsing failure is handled."""
        mock_load_scaffold.return_value = {
            "project_data": {},
            "extract_dir": "/tmp/scaffold",
            "cross_section_path": "/tmp/scaffold/xs.mat",
        }

        mock_parse_csv.side_effect = InvalidBatchCSVError("Invalid CSV")

        processor = BatchProcessor(
            scaffold_path=valid_scaffold_file,
            batch_csv_path=valid_batch_csv,
            output_dir=output_dir
        )

        with pytest.raises(BatchProcessingError, match="Invalid CSV"):
            processor.run()


class TestBatchProcessorPathResolution:
    """Tests for path resolution in batch processor."""

    @patch('image_velocimetry_tools.services.batch_processor.BatchCSVParser.parse_csv')
    @patch('image_velocimetry_tools.services.batch_processor.ScaffoldLoader.load_scaffold')
    @patch('image_velocimetry_tools.services.batch_processor.JobExecutor.execute_job')
    def test_resolve_relative_video_paths(
        self,
        mock_execute_job,
        mock_load_scaffold,
        mock_parse_csv,
        valid_scaffold_file,
        valid_batch_csv,
        output_dir,
        temp_dir
    ):
        """Test that relative video paths are resolved correctly."""
        # Mock scaffold loading
        mock_load_scaffold.return_value = {
            "project_data": {},
            "extract_dir": "/tmp/scaffold",
            "cross_section_path": "/tmp/scaffold/xs.mat",
        }

        # Create video file in CSV directory
        csv_dir = Path(valid_batch_csv).parent
        videos_dir = csv_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        video_file = videos_dir / "test1.mp4"
        video_file.write_text("dummy")

        # Mock CSV parsing with relative path
        mock_job = BatchJob(
            job_id="test_001",
            video_path="videos/test1.mp4",  # Relative path
            water_surface_elevation=318.211
        )
        mock_parse_csv.return_value = [mock_job]

        # Mock job execution
        mock_execute_job.return_value = {
            "discharge": 5.0,
            "area": 10.0,
            "processing_time": 100.0,
            "job_output_dir": "/tmp/output/job_001"
        }

        # Run batch processor
        processor = BatchProcessor(
            scaffold_path=valid_scaffold_file,
            batch_csv_path=valid_batch_csv,
            output_dir=output_dir
        )

        jobs = processor.run()

        # Verify path was resolved to absolute
        assert Path(jobs[0].video_path).is_absolute()
        assert jobs[0].video_path.endswith("test1.mp4")
