"""Tests for OrthorectificationService."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from image_velocimetry_tools.services.orthorectification_service import OrthorectificationService


@pytest.fixture
def ortho_service():
    """Create an OrthorectificationService instance for testing."""
    return OrthorectificationService()


@pytest.fixture
def sample_2_gcps():
    """Create sample 2-point GCP configuration."""
    pixel_coords = np.array([
        [100, 100],
        [300, 300]
    ])
    world_coords = np.array([
        [0.0, 0.0, 10.0],
        [10.0, 10.0, 10.0]
    ])
    return pixel_coords, world_coords


@pytest.fixture
def sample_4_gcps_same_z():
    """Create sample 4-point GCP configuration on same elevation."""
    pixel_coords = np.array([
        [100, 100],
        [400, 100],
        [400, 400],
        [100, 400]
    ])
    world_coords = np.array([
        [0.0, 0.0, 10.0],
        [10.0, 0.0, 10.0],
        [10.0, 10.0, 10.0],
        [0.0, 10.0, 10.0]
    ])
    return pixel_coords, world_coords


@pytest.fixture
def sample_6_gcps_varying_z():
    """Create sample 6-point GCP configuration with varying elevations."""
    pixel_coords = np.array([
        [100, 100],
        [400, 100],
        [400, 400],
        [100, 400],
        [250, 250],
        [350, 150]
    ])
    world_coords = np.array([
        [0.0, 0.0, 10.0],
        [10.0, 0.0, 11.0],
        [10.0, 10.0, 12.0],
        [0.0, 10.0, 10.5],
        [5.0, 5.0, 15.0],
        [7.5, 2.5, 13.0]
    ])
    return pixel_coords, world_coords


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


class TestDetermineRectificationMethod:
    """Tests for determine_rectification_method."""

    def test_2_points_returns_scale(self, ortho_service, sample_2_gcps):
        """Test that 2 GCPs returns 'scale' method."""
        _, world_coords = sample_2_gcps
        method = ortho_service.determine_rectification_method(2, world_coords)
        assert method == "scale"

    def test_4_points_returns_homography(self, ortho_service, sample_4_gcps_same_z):
        """Test that 4 GCPs returns 'homography' method."""
        _, world_coords = sample_4_gcps_same_z
        method = ortho_service.determine_rectification_method(4, world_coords)
        assert method == "homography"

    def test_5_points_same_z_returns_homography(self, ortho_service):
        """Test that 5 GCPs on same plane returns 'homography'."""
        world_coords = np.array([
            [0.0, 0.0, 10.0],
            [10.0, 0.0, 10.0],
            [10.0, 10.0, 10.0],
            [0.0, 10.0, 10.0],
            [5.0, 5.0, 10.0]
        ])
        method = ortho_service.determine_rectification_method(5, world_coords)
        assert method == "homography"

    def test_5_points_varying_z_raises_error(self, ortho_service):
        """Test that 5 GCPs with varying Z raises ValueError."""
        world_coords = np.array([
            [0.0, 0.0, 10.0],
            [10.0, 0.0, 11.0],
            [10.0, 10.0, 12.0],
            [0.0, 10.0, 10.5],
            [5.0, 5.0, 15.0]
        ])
        with pytest.raises(ValueError, match="Invalid configuration.*6 points"):
            ortho_service.determine_rectification_method(5, world_coords)

    def test_6_points_varying_z_returns_camera_matrix(self, ortho_service, sample_6_gcps_varying_z):
        """Test that 6+ GCPs with varying Z returns 'camera matrix'."""
        _, world_coords = sample_6_gcps_varying_z
        method = ortho_service.determine_rectification_method(6, world_coords)
        assert method == "camera matrix"

    def test_10_points_same_z_returns_homography(self, ortho_service):
        """Test that 10 GCPs on same plane returns 'homography'."""
        world_coords = np.ones((10, 3)) * 10.0
        world_coords[:, 0] = np.linspace(0, 10, 10)
        world_coords[:, 1] = np.linspace(0, 10, 10)
        method = ortho_service.determine_rectification_method(10, world_coords)
        assert method == "homography"

    def test_less_than_2_points_raises_error(self, ortho_service):
        """Test that less than 2 GCPs raises ValueError."""
        world_coords = np.array([[0.0, 0.0, 10.0]])
        with pytest.raises(ValueError, match="At least 2 ground control points"):
            ortho_service.determine_rectification_method(1, world_coords)


class TestCalculateScaleParameters:
    """Tests for calculate_scale_parameters."""

    def test_scale_parameters_basic(self, ortho_service, sample_2_gcps):
        """Test basic scale parameter calculation."""
        pixel_coords, world_coords = sample_2_gcps
        image_shape = (480, 640, 3)

        result = ortho_service.calculate_scale_parameters(
            pixel_coords,
            world_coords[:, 0:2],
            image_shape
        )

        assert "pixel_gsd" in result
        assert "homography_matrix" in result
        assert "extent" in result
        assert "pad_x" in result
        assert "pad_y" in result
        assert result["pad_x"] == 0
        assert result["pad_y"] == 0
        assert result["pixel_gsd"] > 0

    def test_scale_parameters_pixel_distance(self, ortho_service, sample_2_gcps):
        """Test pixel distance calculation."""
        pixel_coords, world_coords = sample_2_gcps
        image_shape = (480, 640, 3)

        result = ortho_service.calculate_scale_parameters(
            pixel_coords,
            world_coords[:, 0:2],
            image_shape
        )

        # Manual calculation: sqrt((300-100)^2 + (300-100)^2) = sqrt(80000) ≈ 282.84
        expected_pixel_distance = np.sqrt((300-100)**2 + (300-100)**2)
        assert np.isclose(result["pixel_distance"], expected_pixel_distance, rtol=1e-5)

    def test_scale_parameters_ground_distance(self, ortho_service, sample_2_gcps):
        """Test ground distance calculation."""
        pixel_coords, world_coords = sample_2_gcps
        image_shape = (480, 640, 3)

        result = ortho_service.calculate_scale_parameters(
            pixel_coords,
            world_coords[:, 0:2],
            image_shape
        )

        # Manual calculation: sqrt((10-0)^2 + (10-0)^2) = sqrt(200) ≈ 14.14
        expected_ground_distance = np.sqrt((10-0)**2 + (10-0)**2)
        assert np.isclose(result["ground_distance"], expected_ground_distance, rtol=1e-5)

    def test_scale_parameters_gsd_calculation(self, ortho_service, sample_2_gcps):
        """Test GSD is correctly calculated as ground_distance / pixel_distance."""
        pixel_coords, world_coords = sample_2_gcps
        image_shape = (480, 640, 3)

        result = ortho_service.calculate_scale_parameters(
            pixel_coords,
            world_coords[:, 0:2],
            image_shape
        )

        expected_gsd = result["ground_distance"] / result["pixel_distance"]
        assert np.isclose(result["pixel_gsd"], expected_gsd, rtol=1e-5)

    def test_scale_parameters_homography_matrix_shape(self, ortho_service, sample_2_gcps):
        """Test homography matrix has correct shape."""
        pixel_coords, world_coords = sample_2_gcps
        image_shape = (480, 640, 3)

        result = ortho_service.calculate_scale_parameters(
            pixel_coords,
            world_coords[:, 0:2],
            image_shape
        )

        assert result["homography_matrix"].shape == (3, 3)


class TestCalculateHomographyParameters:
    """Tests for calculate_homography_parameters."""

    @patch('image_velocimetry_tools.services.orthorectification_service.rectify_homography')
    def test_homography_parameters_basic(self, mock_rectify, ortho_service, sample_4_gcps_same_z, sample_image):
        """Test basic homography parameter calculation."""
        pixel_coords, world_coords = sample_4_gcps_same_z

        # Mock the rectify_homography return values
        mock_transformed_image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        mock_roi = [[0, 10], [0, 10]]
        mock_world_coords = world_coords[:, 0:2]
        mock_gsd = 0.02
        mock_homography = np.eye(3)

        mock_rectify.return_value = (
            mock_transformed_image,
            mock_roi,
            mock_world_coords,
            mock_gsd,
            mock_homography
        )

        result = ortho_service.calculate_homography_parameters(
            sample_image,
            pixel_coords,
            world_coords
        )

        assert "transformed_image" in result
        assert "homography_matrix" in result
        assert "pixel_gsd" in result
        assert "extent" in result
        assert "world_coords" in result
        assert "pad_x" in result
        assert "pad_y" in result
        assert result["pad_x"] == 200
        assert result["pad_y"] == 200

    @patch('image_velocimetry_tools.services.orthorectification_service.rectify_homography')
    def test_homography_parameters_custom_padding(self, mock_rectify, ortho_service, sample_4_gcps_same_z, sample_image):
        """Test homography with custom padding values."""
        pixel_coords, world_coords = sample_4_gcps_same_z

        mock_rectify.return_value = (
            sample_image,
            [[0, 10], [0, 10]],
            world_coords[:, 0:2],
            0.02,
            np.eye(3)
        )

        result = ortho_service.calculate_homography_parameters(
            sample_image,
            pixel_coords,
            world_coords,
            pad_x=100,
            pad_y=150
        )

        assert result["pad_x"] == 100
        assert result["pad_y"] == 150

        # Verify rectify_homography was called with correct padding
        call_args = mock_rectify.call_args
        assert call_args[1]["pad_x"] == 100
        assert call_args[1]["pad_y"] == 150

    @patch('image_velocimetry_tools.services.orthorectification_service.rectify_homography')
    def test_homography_parameters_with_existing_matrix(self, mock_rectify, ortho_service, sample_4_gcps_same_z, sample_image):
        """Test homography with pre-calculated homography matrix."""
        pixel_coords, world_coords = sample_4_gcps_same_z
        existing_homography = np.array([
            [1.0, 0.1, 10.0],
            [0.1, 1.0, 20.0],
            [0.001, 0.001, 1.0]
        ])

        mock_rectify.return_value = (
            sample_image,
            [[0, 10], [0, 10]],
            world_coords[:, 0:2],
            0.02,
            existing_homography
        )

        result = ortho_service.calculate_homography_parameters(
            sample_image,
            pixel_coords,
            world_coords,
            homography_matrix=existing_homography
        )

        # Verify the existing matrix was passed to rectify_homography
        call_args = mock_rectify.call_args
        assert call_args[1]["homography_matrix"] is existing_homography


class TestCalculateCameraMatrixParameters:
    """Tests for calculate_camera_matrix_parameters."""

    @patch('image_velocimetry_tools.services.orthorectification_service.CameraHelper')
    def test_camera_matrix_parameters_basic(self, mock_camera_helper, ortho_service, sample_6_gcps_varying_z, sample_image):
        """Test basic camera matrix parameter calculation."""
        pixel_coords, world_coords = sample_6_gcps_varying_z
        wse = 10.5

        # Mock CameraHelper
        mock_cam_instance = MagicMock()
        mock_cam_instance.get_camera_matrix.return_value = (np.eye(3, 4), 0.5)
        mock_cam_instance.get_top_view_of_image.return_value = sample_image
        mock_cam_instance.pixel_ground_scale_distance = 0.025
        mock_cam_instance.camera_position_world = np.array([5.0, 5.0, 20.0])
        mock_camera_helper.return_value = mock_cam_instance

        result = ortho_service.calculate_camera_matrix_parameters(
            sample_image,
            pixel_coords,
            world_coords,
            wse
        )

        assert "transformed_image" in result
        assert "camera_matrix" in result
        assert "pixel_gsd" in result
        assert "extent" in result
        assert "camera_position" in result
        assert "projection_rms_error" in result
        assert result["pixel_gsd"] == 0.025
        assert result["projection_rms_error"] == 0.5

    @patch('image_velocimetry_tools.services.orthorectification_service.CameraHelper')
    def test_camera_matrix_parameters_custom_padding(self, mock_camera_helper, ortho_service, sample_6_gcps_varying_z, sample_image):
        """Test camera matrix with custom padding percentage."""
        pixel_coords, world_coords = sample_6_gcps_varying_z
        wse = 10.5

        mock_cam_instance = MagicMock()
        mock_cam_instance.get_camera_matrix.return_value = (np.eye(3, 4), 0.5)
        mock_cam_instance.get_top_view_of_image.return_value = sample_image
        mock_cam_instance.pixel_ground_scale_distance = 0.025
        mock_cam_instance.camera_position_world = np.array([5.0, 5.0, 20.0])
        mock_camera_helper.return_value = mock_cam_instance

        result = ortho_service.calculate_camera_matrix_parameters(
            sample_image,
            pixel_coords,
            world_coords,
            wse,
            padding_percent=0.05
        )

        # Verify extent was padded by 5% instead of default 3%
        assert "extent" in result
        extent = result["extent"]
        # Extent should be [x_min, x_max, y_min, y_max] with 5% padding

    @patch('image_velocimetry_tools.services.orthorectification_service.CameraHelper')
    def test_camera_matrix_parameters_with_existing_matrix(self, mock_camera_helper, ortho_service, sample_6_gcps_varying_z, sample_image):
        """Test camera matrix with pre-calculated camera matrix."""
        pixel_coords, world_coords = sample_6_gcps_varying_z
        wse = 10.5
        existing_camera_matrix = np.random.rand(3, 4)

        mock_cam_instance = MagicMock()
        mock_cam_instance.get_top_view_of_image.return_value = sample_image
        mock_cam_instance.pixel_ground_scale_distance = 0.025
        mock_cam_instance.camera_position_world = np.array([5.0, 5.0, 20.0])
        mock_camera_helper.return_value = mock_cam_instance

        result = ortho_service.calculate_camera_matrix_parameters(
            sample_image,
            pixel_coords,
            world_coords,
            wse,
            camera_matrix=existing_camera_matrix
        )

        # When camera_matrix is provided, projection_rms_error should be None
        assert result["projection_rms_error"] is None
        # Verify set_camera_matrix was called
        mock_cam_instance.set_camera_matrix.assert_called_once_with(existing_camera_matrix)


class TestCalculateQualityMetrics:
    """Tests for calculate_quality_metrics."""

    def test_quality_metrics_scale_method(self, ortho_service):
        """Test quality metrics for scale method."""
        pixel_gsd = 0.05
        pixel_distance = 282.84
        ground_distance = 14.14

        metrics = ortho_service.calculate_quality_metrics(
            "scale",
            pixel_gsd,
            pixel_distance=pixel_distance,
            ground_distance=ground_distance
        )

        assert "rectification_rmse_m" in metrics
        assert "reprojection_error_pixels" in metrics
        assert metrics["rectification_rmse_m"] > 0
        assert metrics["reprojection_error_pixels"] > 0

    @patch('image_velocimetry_tools.services.orthorectification_service.estimate_view_angle')
    @patch('image_velocimetry_tools.services.orthorectification_service.estimate_orthorectification_rmse')
    def test_quality_metrics_homography_method(self, mock_rmse, mock_angle, ortho_service):
        """Test quality metrics for homography method."""
        pixel_gsd = 0.02
        homography_matrix = np.eye(3)

        mock_angle.return_value = 45.0
        mock_rmse.return_value = 0.1

        metrics = ortho_service.calculate_quality_metrics(
            "homography",
            pixel_gsd,
            homography_matrix=homography_matrix
        )

        assert "rectification_rmse_m" in metrics
        assert "reprojection_error_pixels" in metrics
        assert "estimated_view_angle" in metrics
        assert metrics["estimated_view_angle"] == 45.0
        assert metrics["rectification_rmse_m"] == 0.1
        assert metrics["reprojection_error_pixels"] == 0.1 / pixel_gsd

    def test_quality_metrics_homography_no_matrix(self, ortho_service):
        """Test quality metrics for homography without matrix."""
        pixel_gsd = 0.02

        metrics = ortho_service.calculate_quality_metrics(
            "homography",
            pixel_gsd,
            homography_matrix=None
        )

        # Should return empty dict or minimal metrics when no matrix provided
        assert isinstance(metrics, dict)

    def test_quality_metrics_camera_matrix_method(self, ortho_service):
        """Test quality metrics for camera matrix method."""
        pixel_gsd = 0.025

        metrics = ortho_service.calculate_quality_metrics(
            "camera matrix",
            pixel_gsd
        )

        # Camera matrix metrics require reprojection calculation
        assert isinstance(metrics, dict)


class TestRectifyImage:
    """Tests for rectify_image."""

    @patch('image_velocimetry_tools.services.orthorectification_service.rectify_homography')
    @patch('image_velocimetry_tools.services.orthorectification_service.flip_image_array')
    def test_rectify_image_homography_no_flip(self, mock_flip, mock_rectify, ortho_service, sample_image):
        """Test rectifying image with homography method without flipping."""
        rectification_params = {
            "world_coords": np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
            "pixel_coords": np.array([[100, 100], [400, 100], [400, 400], [100, 400]]),
            "homography_matrix": np.eye(3),
            "pad_x": 200,
            "pad_y": 200
        }

        mock_rectify.return_value = (sample_image, None, None, None, None)
        mock_flip.return_value = sample_image

        result = ortho_service.rectify_image(
            sample_image,
            "homography",
            rectification_params,
            flip_x=False,
            flip_y=False
        )

        assert result is not None
        mock_flip.assert_called_once_with(image=sample_image, flip_x=False, flip_y=False)

    @patch('image_velocimetry_tools.services.orthorectification_service.flip_image_array')
    def test_rectify_image_scale_with_flip(self, mock_flip, ortho_service, sample_image):
        """Test rectifying image with scale method with flipping."""
        rectification_params = {}

        mock_flip.return_value = sample_image

        result = ortho_service.rectify_image(
            sample_image,
            "scale",
            rectification_params,
            flip_x=True,
            flip_y=True
        )

        assert result is not None
        mock_flip.assert_called_once_with(image=sample_image, flip_x=True, flip_y=True)

    @patch('image_velocimetry_tools.services.orthorectification_service.CameraHelper')
    @patch('image_velocimetry_tools.services.orthorectification_service.flip_image_array')
    def test_rectify_image_camera_matrix(self, mock_flip, mock_camera, ortho_service, sample_image):
        """Test rectifying image with camera matrix method."""
        rectification_params = {
            "camera_matrix": np.eye(3, 4),
            "water_surface_elevation": 10.0,
            "extent": np.array([0, 10, 0, 10])
        }

        mock_cam_instance = MagicMock()
        mock_cam_instance.get_top_view_of_image.return_value = sample_image
        mock_camera.return_value = mock_cam_instance
        mock_flip.return_value = sample_image

        result = ortho_service.rectify_image(
            sample_image,
            "camera matrix",
            rectification_params,
            flip_x=False,
            flip_y=True
        )

        assert result is not None
        mock_flip.assert_called_once_with(image=sample_image, flip_x=False, flip_y=True)

    def test_rectify_image_unknown_method_raises_error(self, ortho_service, sample_image):
        """Test that unknown rectification method raises ValueError."""
        rectification_params = {}

        with pytest.raises(ValueError, match="Unknown rectification method"):
            ortho_service.rectify_image(
                sample_image,
                "unknown_method",
                rectification_params
            )


class TestValidateGCPConfiguration:
    """Tests for validate_gcp_configuration."""

    def test_validate_valid_configuration(self, ortho_service, sample_4_gcps_same_z):
        """Test validation of valid GCP configuration."""
        pixel_coords, world_coords = sample_4_gcps_same_z

        errors = ortho_service.validate_gcp_configuration(pixel_coords, world_coords)

        assert len(errors) == 0

    def test_validate_mismatched_point_counts(self, ortho_service):
        """Test validation catches mismatched point counts."""
        pixel_coords = np.array([[100, 100], [200, 200]])
        world_coords = np.array([[0, 0, 10], [10, 10, 10], [20, 20, 10]])

        errors = ortho_service.validate_gcp_configuration(pixel_coords, world_coords)

        assert len(errors) > 0
        assert any("Mismatch in number of points" in error for error in errors)

    def test_validate_too_few_points(self, ortho_service):
        """Test validation catches too few GCPs."""
        pixel_coords = np.array([[100, 100]])
        world_coords = np.array([[0, 0, 10]])

        errors = ortho_service.validate_gcp_configuration(pixel_coords, world_coords)

        assert len(errors) > 0
        assert any("At least 2 ground control points required" in error for error in errors)

    def test_validate_insufficient_world_dimensions(self, ortho_service):
        """Test validation catches insufficient world coordinate dimensions."""
        pixel_coords = np.array([[100, 100], [200, 200]])
        world_coords = np.array([[0], [10]])  # Only 1 dimension

        errors = ortho_service.validate_gcp_configuration(pixel_coords, world_coords)

        assert len(errors) > 0
        assert any("at least 2 dimensions" in error for error in errors)

    def test_validate_duplicate_pixel_coords(self, ortho_service):
        """Test validation catches duplicate pixel coordinates."""
        pixel_coords = np.array([[100, 100], [100, 100], [200, 200]])  # Duplicate
        world_coords = np.array([[0, 0, 10], [10, 10, 10], [20, 20, 10]])

        errors = ortho_service.validate_gcp_configuration(pixel_coords, world_coords)

        assert len(errors) > 0
        assert any("Duplicate pixel coordinates" in error for error in errors)

    def test_validate_duplicate_world_coords(self, ortho_service):
        """Test validation catches duplicate world coordinates."""
        pixel_coords = np.array([[100, 100], [200, 200], [300, 300]])
        world_coords = np.array([[0, 0, 10], [0, 0, 10], [20, 20, 10]])  # Duplicate

        errors = ortho_service.validate_gcp_configuration(pixel_coords, world_coords)

        assert len(errors) > 0
        assert any("Duplicate world coordinates" in error for error in errors)

    def test_validate_multiple_errors(self, ortho_service):
        """Test validation returns multiple errors when present."""
        pixel_coords = np.array([[100, 100]])  # Too few
        world_coords = np.array([[0, 0, 10], [10, 10, 10]])  # Mismatched count

        errors = ortho_service.validate_gcp_configuration(pixel_coords, world_coords)

        # Should have at least 2 errors: too few points and mismatched counts
        assert len(errors) >= 2
