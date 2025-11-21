"""Tests for JobExecutor service.

Note: These are unit tests focusing on the job executor logic.
Integration tests with real video processing should be done separately.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from image_velocimetry_tools.services.job_executor import JobExecutor, STIVResults
from image_velocimetry_tools.batch.models import BatchJob, JobStatus
from image_velocimetry_tools.batch.exceptions import JobExecutionError


@pytest.fixture
def job_executor():
    """Fixture providing a JobExecutor instance."""
    return JobExecutor()


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory that's cleaned up after use."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_job():
    """Fixture providing a sample BatchJob."""
    return BatchJob(
        job_id="test_001",
        video_path="/path/to/video.mp4",
        water_surface_elevation=318.211,
        start_time="15",
        end_time="20",
        alpha=0.88
    )


@pytest.fixture
def sample_scaffold_config(temp_dir):
    """Fixture providing sample scaffold configuration."""
    extract_dir = Path(temp_dir) / "scaffold_extract"
    extract_dir.mkdir()

    return {
        "project_data": {
            "rectification_parameters": {
                "method": "camera matrix",
                "ground_control_points": [[0, 0, 100]] * 6,
                "image_control_points": [[100, 100]] * 6,
                "pixel_gsd": 0.01,
            },
            "grid_parameters": {
                "use_cross_section_line": True,
                "cross_section_line_start": [100, 100],
                "cross_section_line_end": [500, 100],
                "num_points": 50,
            },
            "stiv_parameters": {
                "num_pixels": 20,
                "phi_origin": 90,
                "d_phi": 1.0,
                "phi_range": 90,
                "max_vel_threshold_mps": 10.0,
                "gaussian_blur_sigma": 0.5,
            },
            "ffmpeg_parameters": {
                "frame_rate": 10,
                "frame_step": 1,
            },
            "extraction_parameters": {
                "timestep_ms": 100,
            },
        },
        "extract_dir": str(extract_dir),
        "cross_section_path": str(extract_dir / "cross_section.mat"),
    }


class TestJobExecutorCreateJobDirectory:
    """Tests for _create_job_directory method."""

    def test_create_job_directory(self, job_executor, temp_dir):
        """Test creating job directory structure."""
        job_dir = job_executor._create_job_directory(temp_dir, "test_001")

        assert Path(job_dir).exists()
        assert "job_test_001" in job_dir

        # Check subdirectories
        assert (Path(job_dir) / "1-images").exists()
        assert (Path(job_dir) / "2-orthorectification").exists()
        assert (Path(job_dir) / "4-velocities").exists()
        assert (Path(job_dir) / "5-discharge").exists()

    def test_create_job_directory_already_exists(self, job_executor, temp_dir):
        """Test that creating existing directory doesn't raise error."""
        job_dir = job_executor._create_job_directory(temp_dir, "test_001")

        # Create again - should not raise error
        job_dir2 = job_executor._create_job_directory(temp_dir, "test_001")

        assert job_dir == job_dir2
        assert Path(job_dir).exists()


class TestJobExecutorGenerateGrid:
    """Tests for _generate_grid method."""

    def test_generate_grid_from_cross_section_line(self, job_executor, temp_dir, sample_scaffold_config):
        """Test generating grid along cross-section line."""
        project_data = sample_scaffold_config["project_data"]

        grid = job_executor._generate_grid(temp_dir, project_data)

        assert isinstance(grid, np.ndarray)
        assert grid.shape == (50, 2)  # 50 points, x and y coordinates

        # Check endpoints match
        assert np.allclose(grid[0], [100, 100])
        assert np.allclose(grid[-1], [500, 100])

    def test_generate_grid_missing_cross_section_line_raises_error(
        self, job_executor, temp_dir, sample_scaffold_config
    ):
        """Test that missing cross-section line configuration raises error."""
        project_data = sample_scaffold_config["project_data"]
        del project_data["grid_parameters"]["cross_section_line_start"]

        with pytest.raises(JobExecutionError, match="endpoints not specified"):
            job_executor._generate_grid(temp_dir, project_data)

    def test_generate_grid_not_on_cross_section_raises_error(
        self, job_executor, temp_dir, sample_scaffold_config
    ):
        """Test that grid not on cross-section raises error."""
        project_data = sample_scaffold_config["project_data"]
        project_data["grid_parameters"]["use_cross_section_line"] = False

        with pytest.raises(JobExecutionError, match="use_cross_section_line"):
            job_executor._generate_grid(temp_dir, project_data)


class TestSTIVResults:
    """Tests for STIVResults dataclass."""

    def test_stiv_results_creation(self):
        """Test creating STIVResults instance."""
        magnitudes = np.array([1.0, 2.0, 3.0])
        directions = np.array([45, 90, 135])
        stis = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
        thetas = np.array([30, 60, 90])

        results = STIVResults(
            magnitudes_mps=magnitudes,
            directions=directions,
            magnitude_normals_mps=magnitudes,
            stis=stis,
            thetas=thetas
        )

        assert np.array_equal(results.magnitudes_mps, magnitudes)
        assert np.array_equal(results.directions, directions)
        assert np.array_equal(results.magnitude_normals_mps, magnitudes)


class TestJobExecutorExecuteJob:
    """Tests for execute_job method (integration-style with mocks)."""

    @patch('image_velocimetry_tools.services.job_executor.opencv_get_video_metadata')
    @patch('image_velocimetry_tools.services.job_executor.subprocess.run')
    @patch('image_velocimetry_tools.services.job_executor.glob.glob')
    @patch('image_velocimetry_tools.services.job_executor.rectify_many_camera')
    @patch('image_velocimetry_tools.services.job_executor.two_dimensional_stiv_exhaustive')
    @patch('image_velocimetry_tools.services.job_executor.CrossSectionGeometry')
    def test_execute_job_success(
        self,
        mock_xs_geometry,
        mock_stiv,
        mock_rectify,
        mock_glob,
        mock_subprocess,
        mock_opencv,
        job_executor,
        sample_job,
        sample_scaffold_config,
        temp_dir
    ):
        """Test successful job execution with mocked services."""
        # Mock video metadata
        mock_opencv.return_value = {
            "duration_ms": 30000,
            "frame_count": 300,
            "fps": 10
        }

        # Mock subprocess (FFmpeg)
        mock_subprocess.return_value = Mock(returncode=0)

        # Mock glob to return frame files
        frames_dir = Path(temp_dir) / "job_test_001" / "1-images"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_files = [str(frames_dir / f"f{i:04d}.jpg") for i in range(1, 11)]
        rectified_files = [str(frames_dir / f"t{i:04d}.jpg") for i in range(1, 11)]

        # Create dummy frame files
        for f in frame_files + rectified_files:
            Path(f).write_text("dummy")

        def glob_side_effect(pattern):
            if "f*.jpg" in pattern:
                return frame_files
            elif "t*.jpg" in pattern:
                return rectified_files
            return []

        mock_glob.side_effect = glob_side_effect

        # Mock STIV results
        mock_stiv.return_value = (
            np.array([1.5, 2.0, 1.8]),  # magnitudes
            np.array([45, 90, 135]),     # directions
            np.array([[[1, 2]]]),        # stis
            np.array([30, 60, 90])       # thetas
        )

        # Mock cross-section geometry
        mock_xs = Mock()
        mock_xs.get_pixel_xs.return_value = (
            np.array([0, 10, 20]),      # stations
            np.array([315, 316, 315])   # elevations
        )
        mock_xs_geometry.return_value = mock_xs

        # Mock image stack service
        mock_image_stack = np.zeros((100, 100, 10), dtype=np.uint8)
        job_executor.image_stack_service.create_image_stack = Mock(
            return_value=mock_image_stack
        )

        # Execute job
        result = job_executor.execute_job(
            job=sample_job,
            scaffold_config=sample_scaffold_config,
            output_dir=temp_dir
        )

        # Verify result
        assert "discharge" in result
        assert "area" in result
        assert "processing_time" in result
        assert result["processing_time"] > 0

        # Verify job status was updated
        assert sample_job.status == JobStatus.COMPLETED

    def test_execute_job_marks_job_as_processing(
        self, job_executor, sample_job, sample_scaffold_config, temp_dir
    ):
        """Test that job is marked as processing before execution."""
        # This will fail quickly due to missing mocks, but we can check the status
        assert sample_job.status == JobStatus.PENDING

        with pytest.raises(Exception):  # Will fail due to unmocked dependencies
            job_executor.execute_job(
                job=sample_job,
                scaffold_config=sample_scaffold_config,
                output_dir=temp_dir
            )

        # Job should have been marked as processing before the failure
        assert sample_job.status == JobStatus.PROCESSING


class TestJobExecutorErrorHandling:
    """Tests for error handling in JobExecutor."""

    def test_execute_job_directory_creation_failure_raises_error(
        self, job_executor, sample_job, sample_scaffold_config
    ):
        """Test that directory creation failure is handled."""
        # Use invalid output directory
        with pytest.raises(JobExecutionError, match="Failed to create job directory"):
            job_executor.execute_job(
                job=sample_job,
                scaffold_config=sample_scaffold_config,
                output_dir="/invalid/nonexistent/path"
            )

    @patch('image_velocimetry_tools.services.job_executor.opencv_get_video_metadata')
    def test_video_metadata_failure_raises_error(
        self, mock_opencv, job_executor, sample_job, sample_scaffold_config, temp_dir
    ):
        """Test that video metadata read failure is handled."""
        mock_opencv.side_effect = Exception("Failed to read video")

        with pytest.raises(JobExecutionError, match="Failed to read video metadata"):
            job_executor.execute_job(
                job=sample_job,
                scaffold_config=sample_scaffold_config,
                output_dir=temp_dir
            )


class TestJobExecutorSaveResults:
    """Tests for _save_job_results method."""

    def test_save_job_results(self, job_executor, sample_job, temp_dir):
        """Test saving job results to JSON."""
        job_dir = Path(temp_dir) / "job_test"
        job_dir.mkdir()

        discharge_result = {
            "total_discharge": 5.32,
            "total_area": 12.5,
            "discharge_results": {}
        }

        job_executor._save_job_results(
            job_dir=str(job_dir),
            job=sample_job,
            discharge_result=discharge_result,
            processing_time=120.5
        )

        # Verify file was created
        results_file = job_dir / "job_results.json"
        assert results_file.exists()

        # Verify content
        import json
        with open(results_file) as f:
            saved_results = json.load(f)

        assert saved_results["job_id"] == "test_001"
        assert saved_results["discharge_m3s"] == 5.32
        assert saved_results["processing_time_seconds"] == 120.5


class TestJobExecutorServiceInitialization:
    """Tests for JobExecutor initialization."""

    def test_executor_initializes_services(self, job_executor):
        """Test that JobExecutor initializes all required services."""
        assert hasattr(job_executor, 'video_service')
        assert hasattr(job_executor, 'ortho_service')
        assert hasattr(job_executor, 'grid_service')
        assert hasattr(job_executor, 'image_stack_service')
        assert hasattr(job_executor, 'discharge_service')

        # Verify they are actual service instances
        assert job_executor.video_service is not None
        assert job_executor.ortho_service is not None
        assert job_executor.grid_service is not None
        assert job_executor.image_stack_service is not None
        assert job_executor.discharge_service is not None
