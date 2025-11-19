"""Service for image stack creation and processing.

This service provides business logic for:
- Creating grayscale image stacks from image paths
- Validating image stack parameters
- Calculating memory usage estimates
- Managing memory-mapped file creation
"""

import os
from typing import List, Optional, Callable, Tuple
import numpy as np

from image_velocimetry_tools.services.base_service import BaseService
from image_velocimetry_tools.image_processing_tools import (
    create_grayscale_image_stack,
    estimate_image_stack_memory_usage,
)


class ImageStackService(BaseService):
    """Service for image stack creation and management."""

    def create_image_stack(
        self,
        image_paths: List[str],
        progress_callback: Optional[Callable[[int], None]] = None,
        map_file_path: Optional[str] = None,
        map_file_size_thres: float = 9e8,
    ) -> np.ndarray:
        """
        Create a grayscale image stack from a list of image paths.

        Args:
            image_paths: List of file paths to image files
            progress_callback: Optional callback function for progress updates (0-100)
            map_file_path: Optional path for memory-mapped file storage
            map_file_size_thres: Threshold in bytes for using memory-mapped files (default: 900 MB)

        Returns:
            3D numpy array (height, width, num_images) representing the image stack

        Raises:
            ValueError: If image paths list is empty or invalid
            FileNotFoundError: If image files don't exist
        """
        # Validate inputs
        validation_errors = self.validate_image_stack_parameters(
            image_paths, map_file_path, map_file_size_thres
        )
        if validation_errors:
            raise ValueError(f"Invalid parameters: {', '.join(validation_errors)}")

        # Create the image stack using the existing function
        image_stack = create_grayscale_image_stack(
            image_paths=image_paths,
            progress_callback=progress_callback,
            map_file_path=map_file_path,
            map_file_size_thres=map_file_size_thres,
        )

        return image_stack

    def validate_image_stack_parameters(
        self,
        image_paths: List[str],
        map_file_path: Optional[str] = None,
        map_file_size_thres: float = 9e8,
    ) -> List[str]:
        """
        Validate parameters for image stack creation.

        Args:
            image_paths: List of file paths to image files
            map_file_path: Optional path for memory-mapped file
            map_file_size_thres: Memory threshold in bytes

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Validate image paths
        if not image_paths:
            errors.append("Image paths list is empty")
            return errors

        if not isinstance(image_paths, list):
            errors.append("Image paths must be a list")
            return errors

        # Check if files exist
        for i, path in enumerate(image_paths):
            if not os.path.exists(path):
                errors.append(f"Image file not found: {path}")
                if i >= 5:  # Only report first 5 missing files
                    errors.append(f"... and {len(image_paths) - 5} more missing files")
                    break

        # Validate map file size threshold
        if map_file_size_thres <= 0:
            errors.append("Memory threshold must be positive")

        # Validate map file path if provided
        if map_file_path:
            map_dir = os.path.dirname(map_file_path)
            if map_dir and not os.path.exists(map_dir):
                errors.append(f"Memory map directory does not exist: {map_dir}")

        return errors

    def estimate_memory_usage(
        self, height: int, width: int, num_images: int, num_bands: int = 1
    ) -> int:
        """
        Estimate memory usage for an image stack.

        Args:
            height: Image height in pixels
            width: Image width in pixels
            num_images: Number of images in stack
            num_bands: Number of color bands (1 for grayscale, 3 for RGB)

        Returns:
            Estimated memory usage in bytes
        """
        return estimate_image_stack_memory_usage(height, width, num_images, num_bands)

    def should_use_memory_map(
        self,
        height: int,
        width: int,
        num_images: int,
        map_file_size_thres: float = 9e8,
    ) -> bool:
        """
        Determine if a memory-mapped file should be used based on estimated size.

        Args:
            height: Image height in pixels
            width: Image width in pixels
            num_images: Number of images in stack
            map_file_size_thres: Threshold in bytes for using memory map

        Returns:
            True if memory mapping is recommended, False otherwise
        """
        estimated_bytes = self.estimate_memory_usage(height, width, num_images, num_bands=1)
        return estimated_bytes > map_file_size_thres

    def get_preprocessing_parameters(
        self,
        do_clahe: bool = False,
        clahe_clip: float = 2.0,
        clahe_horz_tiles: int = 8,
        clahe_vert_tiles: int = 8,
        do_auto_contrast: bool = False,
        auto_contrast_percent: Optional[int] = None,
    ) -> dict:
        """
        Prepare preprocessing parameters for image processing.

        Args:
            do_clahe: Enable CLAHE preprocessing
            clahe_clip: CLAHE clip limit
            clahe_horz_tiles: CLAHE horizontal tile size
            clahe_vert_tiles: CLAHE vertical tile size
            do_auto_contrast: Enable auto-contrast preprocessing
            auto_contrast_percent: Auto-contrast clip percentage

        Returns:
            Dictionary with preprocessing parameters
        """
        return {
            "do_clahe": do_clahe,
            "clahe_parameters": (clahe_clip, clahe_horz_tiles, clahe_vert_tiles),
            "do_auto_contrast": do_auto_contrast,
            "auto_contrast_percent": auto_contrast_percent,
        }

    def validate_preprocessing_parameters(
        self,
        clahe_clip: float = 2.0,
        clahe_horz_tiles: int = 8,
        clahe_vert_tiles: int = 8,
        auto_contrast_percent: Optional[int] = None,
    ) -> List[str]:
        """
        Validate preprocessing parameters.

        Args:
            clahe_clip: CLAHE clip limit
            clahe_horz_tiles: CLAHE horizontal tile size
            clahe_vert_tiles: CLAHE vertical tile size
            auto_contrast_percent: Auto-contrast clip percentage

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Validate CLAHE parameters
        if clahe_clip <= 0:
            errors.append("CLAHE clip limit must be positive")

        if clahe_horz_tiles <= 0 or clahe_vert_tiles <= 0:
            errors.append("CLAHE tile sizes must be positive")

        # Validate auto-contrast parameters
        if auto_contrast_percent is not None:
            if not 0 <= auto_contrast_percent <= 100:
                errors.append("Auto-contrast percentage must be between 0 and 100")

        return errors
