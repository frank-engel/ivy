"""
OrthorectificationService - Business logic for image orthorectification

This service handles the business logic for orthorectifying images using
ground control points (GCPs). It supports three rectification methods:
- Scale-based (2 GCPs, nadir view)
- Homography (4+ GCPs on same elevation plane)
- Camera matrix (6+ GCPs with varying elevations)
"""

import logging
from typing import Dict, Any, Tuple, Optional, Callable, List
import numpy as np

from image_velocimetry_tools.services.base_service import BaseService
from image_velocimetry_tools.orthorectification import (
    rectify_homography,
    estimate_orthorectification_rmse,
    estimate_view_angle,
    estimate_scale_based_rmse,
    calculate_homography_matrix_simple,
    CameraHelper,
)
from image_velocimetry_tools.image_processing_tools import flip_image_array
from image_velocimetry_tools.common_functions import bounding_box_naive


class OrthorectificationService(BaseService):
    """Service for handling orthorectification business logic."""

    def __init__(self, logger_name: str = "OrthorectificationService"):
        """Initialize the OrthorectificationService.

        Args:
            logger_name: Name for the logger instance
        """
        super().__init__(logger_name)

    def determine_rectification_method(
        self,
        num_points: int,
        world_coords: np.ndarray
    ) -> str:
        """Determine which rectification method to use based on GCPs.

        Args:
            num_points: Number of ground control points
            world_coords: World coordinates array (N x 3)

        Returns:
            Rectification method: "scale", "homography", or "camera matrix"

        Raises:
            ValueError: If point configuration is invalid
        """
        if num_points < 2:
            raise ValueError(
                f"At least 2 ground control points are required, got {num_points}"
            )

        # Case 1: 2 GCPs - assume nadir view, use scale method
        if num_points == 2:
            self.logger.info(
                f"Using scale-based rectification (2 GCPs, nadir assumption)"
            )
            return "scale"

        # Check if all points are on the same Z-plane
        all_same_z = np.all(world_coords[:, -1] == world_coords[0, -1])

        # Case 2: 4 GCPs - use homography
        if num_points == 4:
            self.logger.info(
                f"Using homography rectification (4 GCPs, same Z={all_same_z})"
            )
            return "homography"

        # Case 3: >4 points, all on same elevation - use homography
        if num_points > 4 and all_same_z:
            self.logger.info(
                f"Using homography rectification ({num_points} GCPs, all on same Z-plane)"
            )
            return "homography"

        # Case 4: 5 points not on same plane - invalid
        if num_points == 5 and not all_same_z:
            raise ValueError(
                f"Invalid configuration: {num_points} GCPs with varying elevations. "
                f"Need at least 6 points for camera matrix method or all points on same plane for homography."
            )

        # Case 5: 6+ points with varying elevations - use camera matrix
        if num_points >= 6 and not all_same_z:
            self.logger.info(
                f"Using camera matrix rectification ({num_points} GCPs with varying elevations)"
            )
            return "camera matrix"

        # Default fallback to homography for edge cases
        self.logger.warning(
            f"Unexpected configuration: {num_points} GCPs, same_z={all_same_z}. "
            f"Defaulting to homography method."
        )
        return "homography"

    def calculate_scale_parameters(
        self,
        pixel_coords: np.ndarray,
        world_coords: np.ndarray,
        image_shape: Tuple[int, int, int]
    ) -> Dict[str, Any]:
        """Calculate parameters for scale-based rectification.

        Args:
            pixel_coords: Pixel coordinates (2 x 2)
            world_coords: World coordinates (2 x 2, X and Y only)
            image_shape: Shape of the image (height, width, channels)

        Returns:
            Dictionary containing:
                - pixel_gsd: Ground sample distance in meters/pixel
                - homography_matrix: Simple transformation matrix
                - extent: Bounding box of transformed image
                - pad_x: X padding (0 for scale method)
                - pad_y: Y padding (0 for scale method)
        """
        # Calculate pixel distance
        p1 = pixel_coords[0]
        p2 = pixel_coords[1]
        pixel_distance = np.sqrt(np.sum((p2 - p1) ** 2))

        # Calculate ground distance
        p1_world = world_coords[0]
        p2_world = world_coords[1]
        ground_distance = np.sqrt(np.sum((p2_world - p1_world) ** 2))

        # Calculate pixel GSD
        pixel_gsd = ground_distance / pixel_distance

        # Create point pairs for homography calculation
        point_pairs = [
            (
                pixel_coords[i][0],
                pixel_coords[i][1],
                world_coords[i][0],
                world_coords[i][1],
            )
            for i in range(len(world_coords))
        ]

        # Calculate simple homography matrix
        homography_matrix = calculate_homography_matrix_simple(point_pairs)

        # Extent is just the bounding box of the original image
        extent = bounding_box_naive(
            [(0, image_shape[0]), (0, image_shape[1])]
        )

        return {
            "pixel_gsd": pixel_gsd,
            "homography_matrix": homography_matrix,
            "extent": extent,
            "pad_x": 0,
            "pad_y": 0,
            "pixel_distance": pixel_distance,
            "ground_distance": ground_distance,
        }

    def calculate_homography_parameters(
        self,
        image: np.ndarray,
        pixel_coords: np.ndarray,
        world_coords: np.ndarray,
        homography_matrix: Optional[np.ndarray] = None,
        pad_x: int = 200,
        pad_y: int = 200
    ) -> Dict[str, Any]:
        """Calculate parameters for homography-based rectification.

        Args:
            image: Input image array
            pixel_coords: Pixel coordinates in source image
            world_coords: Corresponding world coordinates (X, Y only)
            homography_matrix: Existing homography matrix (optional)
            pad_x: Padding in X direction
            pad_y: Padding in Y direction

        Returns:
            Dictionary containing:
                - transformed_image: Rectified image
                - homography_matrix: Calculated homography matrix
                - pixel_gsd: Ground sample distance
                - extent: Transformed ROI extent
                - world_coords: Scaled world coordinates
                - pad_x: X padding used
                - pad_y: Y padding used
        """
        self.logger.debug(
            f"Calculating homography with padding ({pad_x}, {pad_y})"
        )

        (
            transformed_image,
            transformed_roi,
            scaled_world_coordinates,
            pixel_gsd,
            homography_matrix,
        ) = rectify_homography(
            image=image,
            points_world_coordinates=world_coords[:, 0:2],
            points_perspective_image_coordinates=pixel_coords,
            homography_matrix=homography_matrix,
            pad_x=pad_x,
            pad_y=pad_y,
        )

        return {
            "transformed_image": transformed_image,
            "homography_matrix": homography_matrix,
            "pixel_gsd": pixel_gsd,
            "extent": transformed_roi,
            "world_coords": scaled_world_coordinates,
            "pad_x": pad_x,
            "pad_y": pad_y,
        }

    def calculate_camera_matrix_parameters(
        self,
        image: np.ndarray,
        pixel_coords: np.ndarray,
        world_coords: np.ndarray,
        water_surface_elevation: float,
        camera_matrix: Optional[np.ndarray] = None,
        padding_percent: float = 0.03
    ) -> Dict[str, Any]:
        """Calculate parameters for camera matrix-based rectification.

        Args:
            image: Input image array
            pixel_coords: Pixel coordinates in source image
            world_coords: Corresponding 3D world coordinates (X, Y, Z)
            water_surface_elevation: Water surface elevation
            camera_matrix: Existing camera matrix (optional)
            padding_percent: Percentage padding for extent

        Returns:
            Dictionary containing:
                - transformed_image: Rectified image
                - camera_matrix: Calculated camera matrix
                - pixel_gsd: Ground sample distance
                - extent: Extent of rectified image
                - camera_position: Camera position in world coordinates
                - projection_rms_error: RMS error of projection
        """
        self.logger.debug(
            f"Calculating camera matrix with WSE={water_surface_elevation}m"
        )

        # Create camera helper
        cam = CameraHelper(
            image=image,
            world_points=world_coords,
            image_points=pixel_coords,
            elevation=water_surface_elevation
        )

        # Calculate camera matrix if not provided
        if camera_matrix is None:
            camera_matrix, projection_rms_error = cam.get_camera_matrix()
            self.logger.info(
                f"Calculated camera matrix with RMS error: {projection_rms_error:.4f}"
            )
        else:
            cam.set_camera_matrix(camera_matrix)
            projection_rms_error = None

        # Calculate extent from bounding box of GCPs
        bbox = bounding_box_naive(world_coords)
        extent = np.array(
            [bbox[0][0], bbox[1][0], bbox[0][1], bbox[1][1]]
        ) * np.array([
            1 - padding_percent,
            1 + padding_percent,
            1 - padding_percent,
            1 + padding_percent
        ])

        # Get rectified image
        transformed_image = cam.get_top_view_of_image(
            image,
            Z=water_surface_elevation,
            extent=extent,
            do_plot=False,
        )

        return {
            "transformed_image": transformed_image,
            "camera_matrix": camera_matrix,
            "pixel_gsd": cam.pixel_ground_scale_distance,
            "extent": extent,
            "camera_position": cam.camera_position_world,
            "projection_rms_error": projection_rms_error,
        }

    def calculate_quality_metrics(
        self,
        method: str,
        pixel_gsd: float,
        homography_matrix: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate quality metrics for rectification.

        Args:
            method: Rectification method used
            pixel_gsd: Pixel ground sample distance
            homography_matrix: Homography matrix (for homography method)
            **kwargs: Additional parameters (pixel_distance, ground_distance for scale method)

        Returns:
            Dictionary containing:
                - rectification_rmse_m: RMSE in meters
                - reprojection_error_pixels: Reprojection error in pixels
                - estimated_view_angle: Estimated view angle (homography only)
        """
        metrics = {}

        if method == "scale":
            # For scale method, use scale-based RMSE estimation
            pixel_distance = kwargs.get("pixel_distance", 0)
            ground_distance = kwargs.get("ground_distance", 0)
            rmse = estimate_scale_based_rmse(
                pixel_gsd,
                ground_distance,
                pixel_error_per_point=2.0
            )
            metrics["rectification_rmse_m"] = rmse
            metrics["reprojection_error_pixels"] = rmse / pixel_gsd

        elif method == "homography":
            # For homography, estimate view angle and calculate RMSE
            if homography_matrix is not None:
                view_angle = estimate_view_angle(homography_matrix)
                rmse = estimate_orthorectification_rmse(view_angle, pixel_gsd)
                metrics["estimated_view_angle"] = view_angle
                metrics["rectification_rmse_m"] = rmse
                metrics["reprojection_error_pixels"] = rmse / pixel_gsd
                self.logger.info(
                    f"Estimated view angle: {view_angle:.2f}°, RMSE: {rmse:.4f}m"
                )
            else:
                self.logger.warning("Homography matrix not provided, cannot calculate metrics")

        elif method == "camera matrix":
            # For camera matrix, metrics depend on camera calibration
            # This would need reprojection error calculation
            self.logger.debug("Camera matrix quality metrics require reprojection calculation")

        return metrics

    def rectify_image(
        self,
        image: np.ndarray,
        method: str,
        rectification_params: Dict[str, Any],
        flip_x: bool = False,
        flip_y: bool = False
    ) -> np.ndarray:
        """Rectify a single image using calculated parameters.

        Args:
            image: Input image to rectify
            method: Rectification method to use
            rectification_params: Parameters calculated from calculate_*_parameters methods
            flip_x: Whether to flip horizontally
            flip_y: Whether to flip vertically

        Returns:
            Rectified image array
        """
        if method == "scale":
            # For scale method, image is already in correct form
            transformed_image = image

        elif method == "homography":
            # Apply homography transformation
            _, _, _, _, _ = rectify_homography(
                image=image,
                points_world_coordinates=rectification_params["world_coords"][:, 0:2],
                points_perspective_image_coordinates=rectification_params["pixel_coords"],
                homography_matrix=rectification_params["homography_matrix"],
                pad_x=rectification_params["pad_x"],
                pad_y=rectification_params["pad_y"],
            )
            transformed_image, _, _, _, _ = rectify_homography(
                image=image,
                points_world_coordinates=rectification_params["world_coords"][:, 0:2],
                points_perspective_image_coordinates=rectification_params["pixel_coords"],
                homography_matrix=rectification_params["homography_matrix"],
                pad_x=rectification_params["pad_x"],
                pad_y=rectification_params["pad_y"],
            )

        elif method == "camera matrix":
            # Apply camera matrix transformation
            cam = CameraHelper()
            cam.set_camera_matrix(rectification_params["camera_matrix"])
            transformed_image = cam.get_top_view_of_image(
                image,
                Z=rectification_params["water_surface_elevation"],
                extent=rectification_params["extent"],
                do_plot=False,
            )

        else:
            raise ValueError(f"Unknown rectification method: {method}")

        # Apply flipping if requested
        transformed_image = flip_image_array(
            image=transformed_image,
            flip_x=flip_x,
            flip_y=flip_y
        )

        return transformed_image

    def validate_gcp_configuration(
        self,
        pixel_coords: np.ndarray,
        world_coords: np.ndarray
    ) -> List[str]:
        """Validate ground control point configuration.

        Args:
            pixel_coords: Pixel coordinates array
            world_coords: World coordinates array

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if pixel_coords.shape[0] != world_coords.shape[0]:
            errors.append(
                f"Mismatch in number of points: "
                f"{pixel_coords.shape[0]} pixel coords vs {world_coords.shape[0]} world coords"
            )

        if pixel_coords.shape[0] < 2:
            errors.append(
                f"At least 2 ground control points required, got {pixel_coords.shape[0]}"
            )

        if world_coords.shape[1] < 2:
            errors.append(
                f"World coordinates must have at least 2 dimensions (X, Y), got {world_coords.shape[1]}"
            )

        # Check for duplicate points
        if len(np.unique(pixel_coords, axis=0)) != pixel_coords.shape[0]:
            errors.append("Duplicate pixel coordinates detected")

        if len(np.unique(world_coords, axis=0)) != world_coords.shape[0]:
            errors.append("Duplicate world coordinates detected")

        return errors
