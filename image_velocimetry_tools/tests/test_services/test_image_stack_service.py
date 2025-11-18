"""Tests for ImageStackService."""

import os
import tempfile
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from image_velocimetry_tools.services.image_stack_service import ImageStackService


@pytest.fixture
def image_stack_service():
    """Create an ImageStackService instance for testing."""
    return ImageStackService()


@pytest.fixture
def sample_image_paths(tmp_path):
    """Create sample image paths for testing."""
    # Create temporary image files
    image_paths = []
    for i in range(5):
        img_path = tmp_path / f"frame_{i:05d}.jpg"
        img_path.touch()  # Create empty file
        image_paths.append(str(img_path))
    return image_paths


class TestCreateImageStack:
    """Tests for create_image_stack method."""

    @patch('image_velocimetry_tools.services.image_stack_service.create_grayscale_image_stack')
    def test_create_image_stack_basic(self, mock_create_stack, image_stack_service, sample_image_paths):
        """Test basic image stack creation."""
        # Setup mock
        expected_stack = np.zeros((480, 640, 5), dtype=np.uint8)
        mock_create_stack.return_value = expected_stack

        # Call service
        result = image_stack_service.create_image_stack(sample_image_paths)

        # Verify
        assert result.shape == (480, 640, 5)
        mock_create_stack.assert_called_once_with(
            image_paths=sample_image_paths,
            progress_callback=None,
            map_file_path=None,
            map_file_size_thres=9e8
        )

    @patch('image_velocimetry_tools.services.image_stack_service.create_grayscale_image_stack')
    def test_create_image_stack_with_memory_map(
        self, mock_create_stack, image_stack_service, sample_image_paths, tmp_path
    ):
        """Test image stack creation with memory-mapped file."""
        # Setup
        map_file = str(tmp_path / "image_stack.dat")
        expected_stack = np.zeros((480, 640, 5), dtype=np.uint8)
        mock_create_stack.return_value = expected_stack

        # Call service
        result = image_stack_service.create_image_stack(
            sample_image_paths,
            map_file_path=map_file,
            map_file_size_thres=1e6
        )

        # Verify
        assert result.shape == (480, 640, 5)
        mock_create_stack.assert_called_once_with(
            image_paths=sample_image_paths,
            progress_callback=None,
            map_file_path=map_file,
            map_file_size_thres=1e6
        )

    @patch('image_velocimetry_tools.services.image_stack_service.create_grayscale_image_stack')
    def test_create_image_stack_with_progress_callback(
        self, mock_create_stack, image_stack_service, sample_image_paths
    ):
        """Test image stack creation with progress callback."""
        # Setup
        progress_values = []
        def progress_callback(value):
            progress_values.append(value)

        expected_stack = np.zeros((480, 640, 5), dtype=np.uint8)
        mock_create_stack.return_value = expected_stack

        # Call service
        result = image_stack_service.create_image_stack(
            sample_image_paths,
            progress_callback=progress_callback
        )

        # Verify
        assert result.shape == (480, 640, 5)
        mock_create_stack.assert_called_once()
        assert mock_create_stack.call_args[1]['progress_callback'] == progress_callback

    def test_create_image_stack_with_empty_paths(self, image_stack_service):
        """Test that empty image paths raise ValueError."""
        with pytest.raises(ValueError, match="Image paths list is empty"):
            image_stack_service.create_image_stack([])

    def test_create_image_stack_with_nonexistent_files(self, image_stack_service):
        """Test that nonexistent files raise ValueError."""
        fake_paths = ["/fake/path/image1.jpg", "/fake/path/image2.jpg"]
        with pytest.raises(ValueError, match="Image file not found"):
            image_stack_service.create_image_stack(fake_paths)


class TestValidateImageStackParameters:
    """Tests for validate_image_stack_parameters method."""

    def test_validate_valid_parameters(self, image_stack_service, sample_image_paths, tmp_path):
        """Test validation with valid parameters."""
        map_file = str(tmp_path / "stack.dat")
        errors = image_stack_service.validate_image_stack_parameters(
            sample_image_paths,
            map_file_path=map_file,
            map_file_size_thres=1e9
        )
        assert len(errors) == 0

    def test_validate_empty_image_paths(self, image_stack_service):
        """Test validation with empty image paths."""
        errors = image_stack_service.validate_image_stack_parameters([])
        assert "Image paths list is empty" in errors

    def test_validate_non_list_image_paths(self, image_stack_service):
        """Test validation with non-list image paths."""
        errors = image_stack_service.validate_image_stack_parameters("not_a_list")
        assert "Image paths must be a list" in errors

    def test_validate_nonexistent_files(self, image_stack_service):
        """Test validation with nonexistent files."""
        fake_paths = ["/fake/image1.jpg", "/fake/image2.jpg"]
        errors = image_stack_service.validate_image_stack_parameters(fake_paths)
        assert any("Image file not found" in error for error in errors)

    def test_validate_negative_threshold(self, image_stack_service, sample_image_paths):
        """Test validation with negative memory threshold."""
        errors = image_stack_service.validate_image_stack_parameters(
            sample_image_paths,
            map_file_size_thres=-1
        )
        assert "Memory threshold must be positive" in errors

    def test_validate_nonexistent_map_directory(self, image_stack_service, sample_image_paths):
        """Test validation with nonexistent map file directory."""
        map_file = "/nonexistent/directory/stack.dat"
        errors = image_stack_service.validate_image_stack_parameters(
            sample_image_paths,
            map_file_path=map_file
        )
        assert any("Memory map directory does not exist" in error for error in errors)

    def test_validate_reports_max_five_missing_files(self, image_stack_service):
        """Test that validation reports maximum 5 missing files."""
        fake_paths = [f"/fake/image{i}.jpg" for i in range(10)]
        errors = image_stack_service.validate_image_stack_parameters(fake_paths)
        # Should have error messages for first 5 files plus summary message
        missing_file_errors = [e for e in errors if "Image file not found" in e]
        assert len(missing_file_errors) <= 6  # 5 specific + 1 summary


class TestEstimateMemoryUsage:
    """Tests for estimate_memory_usage method."""

    @patch('image_velocimetry_tools.services.image_stack_service.estimate_image_stack_memory_usage')
    def test_estimate_memory_grayscale(self, mock_estimate, image_stack_service):
        """Test memory estimation for grayscale images."""
        mock_estimate.return_value = 1536000  # 640 * 480 * 5 * 1

        result = image_stack_service.estimate_memory_usage(480, 640, 5, num_bands=1)

        assert result == 1536000
        mock_estimate.assert_called_once_with(480, 640, 5, 1)

    @patch('image_velocimetry_tools.services.image_stack_service.estimate_image_stack_memory_usage')
    def test_estimate_memory_rgb(self, mock_estimate, image_stack_service):
        """Test memory estimation for RGB images."""
        mock_estimate.return_value = 4608000  # 640 * 480 * 5 * 3

        result = image_stack_service.estimate_memory_usage(480, 640, 5, num_bands=3)

        assert result == 4608000
        mock_estimate.assert_called_once_with(480, 640, 5, 3)


class TestShouldUseMemoryMap:
    """Tests for should_use_memory_map method."""

    @patch('image_velocimetry_tools.services.image_stack_service.estimate_image_stack_memory_usage')
    def test_should_use_memory_map_below_threshold(self, mock_estimate, image_stack_service):
        """Test memory map decision when below threshold."""
        mock_estimate.return_value = 5e8  # 500 MB

        result = image_stack_service.should_use_memory_map(480, 640, 100, map_file_size_thres=9e8)

        assert result is False

    @patch('image_velocimetry_tools.services.image_stack_service.estimate_image_stack_memory_usage')
    def test_should_use_memory_map_above_threshold(self, mock_estimate, image_stack_service):
        """Test memory map decision when above threshold."""
        mock_estimate.return_value = 1.5e9  # 1.5 GB

        result = image_stack_service.should_use_memory_map(1080, 1920, 500, map_file_size_thres=9e8)

        assert result is True

    @patch('image_velocimetry_tools.services.image_stack_service.estimate_image_stack_memory_usage')
    def test_should_use_memory_map_exactly_at_threshold(self, mock_estimate, image_stack_service):
        """Test memory map decision when exactly at threshold."""
        mock_estimate.return_value = 9e8

        result = image_stack_service.should_use_memory_map(480, 640, 100, map_file_size_thres=9e8)

        assert result is False  # Should be False when equal


class TestGetPreprocessingParameters:
    """Tests for get_preprocessing_parameters method."""

    def test_get_preprocessing_parameters_defaults(self, image_stack_service):
        """Test getting default preprocessing parameters."""
        params = image_stack_service.get_preprocessing_parameters()

        assert params["do_clahe"] is False
        assert params["clahe_parameters"] == (2.0, 8, 8)
        assert params["do_auto_contrast"] is False
        assert params["auto_contrast_percent"] is None

    def test_get_preprocessing_parameters_with_clahe(self, image_stack_service):
        """Test getting preprocessing parameters with CLAHE enabled."""
        params = image_stack_service.get_preprocessing_parameters(
            do_clahe=True,
            clahe_clip=3.0,
            clahe_horz_tiles=16,
            clahe_vert_tiles=16
        )

        assert params["do_clahe"] is True
        assert params["clahe_parameters"] == (3.0, 16, 16)

    def test_get_preprocessing_parameters_with_auto_contrast(self, image_stack_service):
        """Test getting preprocessing parameters with auto-contrast enabled."""
        params = image_stack_service.get_preprocessing_parameters(
            do_auto_contrast=True,
            auto_contrast_percent=2
        )

        assert params["do_auto_contrast"] is True
        assert params["auto_contrast_percent"] == 2

    def test_get_preprocessing_parameters_both_enabled(self, image_stack_service):
        """Test getting preprocessing parameters with both methods enabled."""
        params = image_stack_service.get_preprocessing_parameters(
            do_clahe=True,
            clahe_clip=2.5,
            clahe_horz_tiles=12,
            clahe_vert_tiles=12,
            do_auto_contrast=True,
            auto_contrast_percent=3
        )

        assert params["do_clahe"] is True
        assert params["clahe_parameters"] == (2.5, 12, 12)
        assert params["do_auto_contrast"] is True
        assert params["auto_contrast_percent"] == 3


class TestValidatePreprocessingParameters:
    """Tests for validate_preprocessing_parameters method."""

    def test_validate_valid_preprocessing_parameters(self, image_stack_service):
        """Test validation with valid preprocessing parameters."""
        errors = image_stack_service.validate_preprocessing_parameters(
            clahe_clip=2.0,
            clahe_horz_tiles=8,
            clahe_vert_tiles=8,
            auto_contrast_percent=2
        )
        assert len(errors) == 0

    def test_validate_negative_clahe_clip(self, image_stack_service):
        """Test validation with negative CLAHE clip limit."""
        errors = image_stack_service.validate_preprocessing_parameters(clahe_clip=-1.0)
        assert "CLAHE clip limit must be positive" in errors

    def test_validate_zero_clahe_clip(self, image_stack_service):
        """Test validation with zero CLAHE clip limit."""
        errors = image_stack_service.validate_preprocessing_parameters(clahe_clip=0)
        assert "CLAHE clip limit must be positive" in errors

    def test_validate_negative_tile_sizes(self, image_stack_service):
        """Test validation with negative tile sizes."""
        errors = image_stack_service.validate_preprocessing_parameters(
            clahe_horz_tiles=-1,
            clahe_vert_tiles=8
        )
        assert "CLAHE tile sizes must be positive" in errors

    def test_validate_zero_tile_sizes(self, image_stack_service):
        """Test validation with zero tile sizes."""
        errors = image_stack_service.validate_preprocessing_parameters(
            clahe_horz_tiles=0,
            clahe_vert_tiles=0
        )
        assert "CLAHE tile sizes must be positive" in errors

    def test_validate_auto_contrast_percent_below_range(self, image_stack_service):
        """Test validation with auto-contrast percentage below range."""
        errors = image_stack_service.validate_preprocessing_parameters(
            auto_contrast_percent=-1
        )
        assert "Auto-contrast percentage must be between 0 and 100" in errors

    def test_validate_auto_contrast_percent_above_range(self, image_stack_service):
        """Test validation with auto-contrast percentage above range."""
        errors = image_stack_service.validate_preprocessing_parameters(
            auto_contrast_percent=101
        )
        assert "Auto-contrast percentage must be between 0 and 100" in errors

    def test_validate_auto_contrast_percent_at_boundaries(self, image_stack_service):
        """Test validation with auto-contrast percentage at valid boundaries."""
        errors_zero = image_stack_service.validate_preprocessing_parameters(auto_contrast_percent=0)
        errors_hundred = image_stack_service.validate_preprocessing_parameters(auto_contrast_percent=100)

        assert len(errors_zero) == 0
        assert len(errors_hundred) == 0

    def test_validate_auto_contrast_percent_none(self, image_stack_service):
        """Test validation with None auto-contrast percentage."""
        errors = image_stack_service.validate_preprocessing_parameters(auto_contrast_percent=None)
        assert len(errors) == 0
