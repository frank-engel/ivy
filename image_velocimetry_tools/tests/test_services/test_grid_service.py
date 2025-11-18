"""Tests for GridService."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from image_velocimetry_tools.services.grid_service import GridService


@pytest.fixture
def grid_service():
    """Create a GridService instance for testing."""
    return GridService()


@pytest.fixture
def sample_mask_polygon():
    """Create a sample mask polygon."""
    return [np.array([
        [10, 10],
        [100, 10],
        [100, 100],
        [10, 100]
    ])]


class TestCreateMask:
    """Tests for create_mask method."""

    @patch('image_velocimetry_tools.services.grid_service.create_binary_mask')
    @patch('image_velocimetry_tools.services.grid_service.close_small_gaps')
    def test_create_mask_basic(self, mock_clean, mock_create, grid_service, sample_mask_polygon):
        """Test basic mask creation."""
        mock_create.return_value = np.ones((480, 640), dtype=bool)
        mock_clean.return_value = np.ones((480, 640), dtype=bool)

        result = grid_service.create_mask(sample_mask_polygon, 640, 480)

        assert result.shape == (480, 640)
        mock_create.assert_called_once_with(sample_mask_polygon, 640, 480)
        mock_clean.assert_called_once()

    @patch('image_velocimetry_tools.services.grid_service.create_binary_mask')
    def test_create_mask_without_cleaning(self, mock_create, grid_service, sample_mask_polygon):
        """Test mask creation without cleaning."""
        mock_create.return_value = np.ones((480, 640), dtype=bool)

        result = grid_service.create_mask(sample_mask_polygon, 640, 480, clean=False)

        assert result.shape == (480, 640)
        mock_create.assert_called_once()

    def test_create_mask_invalid_width(self, grid_service, sample_mask_polygon):
        """Test that invalid width raises ValueError."""
        with pytest.raises(ValueError, match="image_width must be positive"):
            grid_service.create_mask(sample_mask_polygon, 0, 480)

    def test_create_mask_invalid_height(self, grid_service, sample_mask_polygon):
        """Test that invalid height raises ValueError."""
        with pytest.raises(ValueError, match="image_height must be positive"):
            grid_service.create_mask(sample_mask_polygon, 640, -1)

    @patch('image_velocimetry_tools.services.grid_service.create_binary_mask')
    @patch('image_velocimetry_tools.services.grid_service.close_small_gaps')
    def test_create_mask_with_custom_cleaning_params(
        self, mock_clean, mock_create, grid_service, sample_mask_polygon
    ):
        """Test mask creation with custom cleaning parameters."""
        mock_create.return_value = np.ones((480, 640), dtype=bool)
        mock_clean.return_value = np.ones((480, 640), dtype=bool)

        grid_service.create_mask(
            sample_mask_polygon, 640, 480,
            kernel_size=7,
            area_threshold=0.05,
            blur_sigma=2.0
        )

        mock_clean.assert_called_once_with(
            mock_create.return_value,
            kernel_size=7,
            area_threshold=0.05,
            blur_sigma=2.0
        )


class TestGenerateRegularGrid:
    """Tests for generate_regular_grid method."""

    @patch('image_velocimetry_tools.services.grid_service.generate_grid')
    def test_generate_grid_basic(self, mock_generate, grid_service):
        """Test basic grid generation."""
        mock_points = np.array([[10, 10], [20, 10], [10, 20], [20, 20]])
        mock_generate.return_value = mock_points

        grid_points, binary_mask = grid_service.generate_regular_grid(
            640, 480, 10, 10
        )

        assert len(grid_points) == 4
        assert binary_mask.shape == (480, 640)
        mock_generate.assert_called_once()

    @patch('image_velocimetry_tools.services.grid_service.generate_grid')
    @patch('image_velocimetry_tools.services.grid_service.create_binary_mask')
    def test_generate_grid_with_mask(
        self, mock_create_mask, mock_generate, grid_service, sample_mask_polygon
    ):
        """Test grid generation with mask polygons."""
        mock_points = np.array([[10, 10], [20, 10]])
        mock_generate.return_value = mock_points
        mock_create_mask.return_value = np.ones((480, 640), dtype=bool)

        grid_points, binary_mask = grid_service.generate_regular_grid(
            640, 480, 10, 10,
            mask_polygons=sample_mask_polygon
        )

        assert len(grid_points) == 2
        mock_create_mask.assert_called_once()
        mock_generate.assert_called_once()

    def test_generate_grid_invalid_width(self, grid_service):
        """Test that invalid width raises ValueError."""
        with pytest.raises(ValueError, match="image_width must be positive"):
            grid_service.generate_regular_grid(0, 480, 10, 10)

    def test_generate_grid_invalid_height(self, grid_service):
        """Test that invalid height raises ValueError."""
        with pytest.raises(ValueError, match="image_height must be positive"):
            grid_service.generate_regular_grid(640, 0, 10, 10)

    def test_generate_grid_invalid_vertical_spacing(self, grid_service):
        """Test that invalid vertical spacing raises ValueError."""
        with pytest.raises(ValueError, match="vertical_spacing must be positive"):
            grid_service.generate_regular_grid(640, 480, 0, 10)

    def test_generate_grid_invalid_horizontal_spacing(self, grid_service):
        """Test that invalid horizontal spacing raises ValueError."""
        with pytest.raises(ValueError, match="horizontal_spacing must be positive"):
            grid_service.generate_regular_grid(640, 480, 10, -5)


class TestGenerateLineGrid:
    """Tests for generate_line_grid method."""

    @patch('image_velocimetry_tools.services.grid_service.generate_points_along_line')
    def test_generate_line_basic(self, mock_generate, grid_service):
        """Test basic line grid generation."""
        mock_points = np.array([[10, 10], [20, 20], [30, 30]])
        mock_generate.return_value = mock_points

        line_start = np.array([10, 10])
        line_end = np.array([100, 100])

        line_points, binary_mask = grid_service.generate_line_grid(
            640, 480, line_start, line_end, 3
        )

        assert len(line_points) == 3
        assert binary_mask.shape == (480, 640)
        mock_generate.assert_called_once()

    @patch('image_velocimetry_tools.services.grid_service.generate_points_along_line')
    @patch('image_velocimetry_tools.services.grid_service.create_binary_mask')
    def test_generate_line_with_mask(
        self, mock_create_mask, mock_generate, grid_service, sample_mask_polygon
    ):
        """Test line grid generation with mask polygons."""
        mock_points = np.array([[10, 10], [50, 50]])
        mock_generate.return_value = mock_points
        mock_create_mask.return_value = np.ones((480, 640), dtype=bool)

        line_start = np.array([10, 10])
        line_end = np.array([100, 100])

        line_points, binary_mask = grid_service.generate_line_grid(
            640, 480, line_start, line_end, 10,
            mask_polygons=sample_mask_polygon
        )

        assert len(line_points) == 2
        mock_create_mask.assert_called_once()
        mock_generate.assert_called_once()

    def test_generate_line_invalid_width(self, grid_service):
        """Test that invalid width raises ValueError."""
        line_start = np.array([10, 10])
        line_end = np.array([100, 100])

        with pytest.raises(ValueError, match="image_width must be positive"):
            grid_service.generate_line_grid(-1, 480, line_start, line_end, 10)

    def test_generate_line_invalid_num_points(self, grid_service):
        """Test that invalid num_points raises ValueError."""
        line_start = np.array([10, 10])
        line_end = np.array([100, 100])

        with pytest.raises(ValueError, match="num_points must be positive"):
            grid_service.generate_line_grid(640, 480, line_start, line_end, 0)

    def test_generate_line_invalid_line_start(self, grid_service):
        """Test that invalid line_start raises ValueError."""
        line_start = np.array([10])  # Wrong shape
        line_end = np.array([100, 100])

        with pytest.raises(ValueError, match="line_start must be a 2-element array"):
            grid_service.generate_line_grid(640, 480, line_start, line_end, 10)

    def test_generate_line_invalid_line_end(self, grid_service):
        """Test that invalid line_end raises ValueError."""
        line_start = np.array([10, 10])
        line_end = np.array([100, 100, 100])  # Wrong shape

        with pytest.raises(ValueError, match="line_end must be a 2-element array"):
            grid_service.generate_line_grid(640, 480, line_start, line_end, 10)


class TestCalculateGridStatistics:
    """Tests for calculate_grid_statistics method."""

    def test_calculate_stats_basic(self, grid_service):
        """Test basic grid statistics calculation."""
        grid_points = np.array([
            [10, 20],
            [50, 30],
            [90, 40],
            [30, 60]
        ])

        stats = grid_service.calculate_grid_statistics(grid_points)

        assert stats["num_points"] == 4
        assert stats["x_min"] == 10.0
        assert stats["x_max"] == 90.0
        assert stats["y_min"] == 20.0
        assert stats["y_max"] == 60.0
        assert stats["x_range"] == 80.0
        assert stats["y_range"] == 40.0

    def test_calculate_stats_with_gsd(self, grid_service):
        """Test statistics calculation with pixel GSD."""
        grid_points = np.array([
            [0, 0],
            [100, 0],
            [0, 100],
            [100, 100]
        ])

        pixel_gsd = 0.02  # 2 cm/pixel

        stats = grid_service.calculate_grid_statistics(grid_points, pixel_gsd)

        assert stats["num_points"] == 4
        assert stats["x_range"] == 100.0
        assert stats["y_range"] == 100.0
        assert stats["x_range_m"] == 2.0  # 100 * 0.02
        assert stats["y_range_m"] == 2.0

    def test_calculate_stats_empty_grid(self, grid_service):
        """Test statistics calculation for empty grid."""
        grid_points = np.array([]).reshape(0, 2)

        stats = grid_service.calculate_grid_statistics(grid_points)

        assert stats["num_points"] == 0
        assert stats["x_min"] == 0
        assert stats["x_max"] == 0
        assert stats["x_range"] == 0

    def test_calculate_stats_single_point(self, grid_service):
        """Test statistics calculation for single point."""
        grid_points = np.array([[42, 84]])

        stats = grid_service.calculate_grid_statistics(grid_points)

        assert stats["num_points"] == 1
        assert stats["x_min"] == 42.0
        assert stats["x_max"] == 42.0
        assert stats["x_range"] == 0.0


class TestValidateGridParameters:
    """Tests for validate_grid_parameters method."""

    def test_validate_valid_regular_grid(self, grid_service):
        """Test validation of valid regular grid parameters."""
        errors = grid_service.validate_grid_parameters(640, 480, 20, "regular")

        assert len(errors) == 0

    def test_validate_valid_line_grid(self, grid_service):
        """Test validation of valid line grid parameters."""
        errors = grid_service.validate_grid_parameters(640, 480, 50, "line")

        assert len(errors) == 0

    def test_validate_invalid_width(self, grid_service):
        """Test validation catches invalid width."""
        errors = grid_service.validate_grid_parameters(0, 480, 20, "regular")

        assert len(errors) > 0
        assert any("width" in error.lower() for error in errors)

    def test_validate_invalid_height(self, grid_service):
        """Test validation catches invalid height."""
        errors = grid_service.validate_grid_parameters(640, -10, 20, "regular")

        assert len(errors) > 0
        assert any("height" in error.lower() for error in errors)

    def test_validate_invalid_spacing(self, grid_service):
        """Test validation catches invalid spacing."""
        errors = grid_service.validate_grid_parameters(640, 480, 0, "regular")

        assert len(errors) > 0
        assert any("spacing" in error.lower() for error in errors)

    def test_validate_invalid_num_points(self, grid_service):
        """Test validation catches invalid number of points."""
        errors = grid_service.validate_grid_parameters(640, 480, -5, "line")

        assert len(errors) > 0
        assert any("number of points" in error.lower() for error in errors)

    def test_validate_spacing_too_large(self, grid_service):
        """Test validation catches spacing larger than image."""
        errors = grid_service.validate_grid_parameters(640, 480, 1000, "regular")

        assert len(errors) > 0
        assert any("larger than" in error.lower() for error in errors)

    def test_validate_multiple_errors(self, grid_service):
        """Test validation returns multiple errors when present."""
        errors = grid_service.validate_grid_parameters(0, 0, 0, "regular")

        # Should have at least 3 errors: width, height, and spacing
        assert len(errors) >= 3
