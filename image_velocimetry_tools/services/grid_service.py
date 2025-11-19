"""
GridService - Business logic for grid generation

This service handles the business logic for generating computational grids
and line grids for image velocimetry analysis. It supports:
- Grid generation with user-defined spacing
- Line grid generation between two points
- Binary mask creation from polygons
- Mask cleaning and topology operations
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from image_velocimetry_tools.services.base_service import BaseService
from image_velocimetry_tools.image_processing_tools import (
    generate_grid,
    create_binary_mask,
    close_small_gaps,
    generate_points_along_line,
)


class GridService(BaseService):
    """Service for handling grid generation business logic."""

    def __init__(self, logger_name: str = "GridService"):
        """Initialize the GridService.

        Args:
            logger_name: Name for the logger instance
        """
        super().__init__(logger_name)

    def create_mask(
        self,
        mask_polygons: List[np.ndarray],
        image_width: int,
        image_height: int,
        clean: bool = True,
        kernel_size: int = 5,
        area_threshold: float = 0.03,
        blur_sigma: float = 1.0
    ) -> np.ndarray:
        """Create and optionally clean a binary mask from polygons.

        Args:
            mask_polygons: List of polygon arrays defining masked regions
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
            clean: Whether to apply topology cleaning to the mask
            kernel_size: Kernel size for morphological operations
            area_threshold: Threshold for small gap filling
            blur_sigma: Sigma for Gaussian blur

        Returns:
            Binary mask as numpy array (True = valid region, False = masked)

        Raises:
            ValueError: If image dimensions are invalid
        """
        self._validate_positive(image_width, "image_width")
        self._validate_positive(image_height, "image_height")

        self.logger.debug(
            f"Creating mask for {image_width}x{image_height} image "
            f"with {len(mask_polygons)} polygons"
        )

        # Create initial binary mask
        binary_mask = create_binary_mask(mask_polygons, image_width, image_height)

        # Clean the mask if requested
        if clean:
            binary_mask = close_small_gaps(
                binary_mask,
                kernel_size=kernel_size,
                area_threshold=area_threshold,
                blur_sigma=blur_sigma
            )
            self.logger.debug("Mask cleaned")

        return binary_mask

    def generate_regular_grid(
        self,
        image_width: int,
        image_height: int,
        vertical_spacing: int,
        horizontal_spacing: int,
        mask_polygons: Optional[List[np.ndarray]] = None,
        clean_mask: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a regular computational grid.

        Args:
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
            vertical_spacing: Vertical spacing between grid points in pixels
            horizontal_spacing: Horizontal spacing between grid points in pixels
            mask_polygons: Optional list of polygons to mask grid points
            clean_mask: Whether to clean the mask

        Returns:
            Tuple of (grid_points, binary_mask) where:
                - grid_points: Nx2 array of (x, y) coordinates
                - binary_mask: Binary mask array

        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_positive(image_width, "image_width")
        self._validate_positive(image_height, "image_height")
        self._validate_positive(vertical_spacing, "vertical_spacing")
        self._validate_positive(horizontal_spacing, "horizontal_spacing")

        self.logger.info(
            f"Generating regular grid: {image_width}x{image_height}, "
            f"spacing: {horizontal_spacing}x{vertical_spacing} px"
        )

        # Create mask (or empty mask if no polygons)
        if mask_polygons:
            binary_mask = self.create_mask(
                mask_polygons,
                image_width,
                image_height,
                clean=clean_mask
            )
        else:
            binary_mask = np.ones((image_height, image_width), dtype=bool)

        # Generate grid points
        grid_points = generate_grid(
            image_width,
            image_height,
            vertical_spacing,
            horizontal_spacing,
            binary_mask
        )

        self.logger.info(f"Generated {len(grid_points)} grid points")

        return grid_points, binary_mask

    def generate_line_grid(
        self,
        image_width: int,
        image_height: int,
        line_start: np.ndarray,
        line_end: np.ndarray,
        num_points: int,
        mask_polygons: Optional[List[np.ndarray]] = None,
        clean_mask: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate evenly spaced points along a line.

        Args:
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
            line_start: (x, y) coordinates of line start
            line_end: (x, y) coordinates of line end
            num_points: Number of points to generate along the line
            mask_polygons: Optional list of polygons to mask grid points
            clean_mask: Whether to clean the mask

        Returns:
            Tuple of (line_points, binary_mask) where:
                - line_points: Nx2 array of (x, y) coordinates
                - binary_mask: Binary mask array

        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_positive(image_width, "image_width")
        self._validate_positive(image_height, "image_height")
        self._validate_positive(num_points, "num_points")

        if line_start.shape != (2,):
            raise ValueError("line_start must be a 2-element array (x, y)")
        if line_end.shape != (2,):
            raise ValueError("line_end must be a 2-element array (x, y)")

        self.logger.info(
            f"Generating line grid: {num_points} points from "
            f"({line_start[0]:.1f}, {line_start[1]:.1f}) to "
            f"({line_end[0]:.1f}, {line_end[1]:.1f})"
        )

        # Create mask (or empty mask if no polygons)
        if mask_polygons:
            binary_mask = self.create_mask(
                mask_polygons,
                image_width,
                image_height,
                clean=clean_mask
            )
        else:
            binary_mask = np.ones((image_height, image_width), dtype=bool)

        # Generate line points
        line_points = generate_points_along_line(
            image_width,
            image_height,
            line_start,
            line_end,
            num_points,
            binary_mask
        )

        self.logger.info(f"Generated {len(line_points)} line points")

        return line_points, binary_mask

    def calculate_grid_statistics(
        self,
        grid_points: np.ndarray,
        pixel_gsd: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculate statistics about a generated grid.

        Args:
            grid_points: Nx2 array of grid point coordinates
            pixel_gsd: Optional pixel ground sample distance in meters/pixel

        Returns:
            Dictionary containing:
                - num_points: Total number of grid points
                - x_min, x_max: X coordinate bounds
                - y_min, y_max: Y coordinate bounds
                - x_range, y_range: Coordinate ranges
                - If pixel_gsd provided:
                    - x_range_m: X range in meters
                    - y_range_m: Y range in meters
        """
        if len(grid_points) == 0:
            return {
                "num_points": 0,
                "x_min": 0,
                "x_max": 0,
                "y_min": 0,
                "y_max": 0,
                "x_range": 0,
                "y_range": 0,
            }

        stats = {
            "num_points": len(grid_points),
            "x_min": float(np.min(grid_points[:, 0])),
            "x_max": float(np.max(grid_points[:, 0])),
            "y_min": float(np.min(grid_points[:, 1])),
            "y_max": float(np.max(grid_points[:, 1])),
        }

        stats["x_range"] = stats["x_max"] - stats["x_min"]
        stats["y_range"] = stats["y_max"] - stats["y_min"]

        if pixel_gsd is not None:
            stats["x_range_m"] = stats["x_range"] * pixel_gsd
            stats["y_range_m"] = stats["y_range"] * pixel_gsd

        return stats

    def validate_grid_parameters(
        self,
        image_width: int,
        image_height: int,
        spacing_or_num_points: int,
        grid_type: str = "regular"
    ) -> List[str]:
        """Validate grid generation parameters.

        Args:
            image_width: Width of the image
            image_height: Height of the image
            spacing_or_num_points: Grid spacing (for regular) or number of points (for line)
            grid_type: "regular" or "line"

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if image_width <= 0:
            errors.append(f"Image width must be positive, got {image_width}")

        if image_height <= 0:
            errors.append(f"Image height must be positive, got {image_height}")

        if spacing_or_num_points <= 0:
            param_name = "spacing" if grid_type == "regular" else "number of points"
            errors.append(f"Grid {param_name} must be positive, got {spacing_or_num_points}")

        if grid_type == "regular":
            # Check if spacing is reasonable (not larger than image)
            if spacing_or_num_points > min(image_width, image_height):
                errors.append(
                    f"Grid spacing ({spacing_or_num_points}) is larger than "
                    f"smallest image dimension ({min(image_width, image_height)})"
                )

        return errors
