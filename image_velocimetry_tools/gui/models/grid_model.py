"""Grid generation state model."""

import numpy as np
from typing import Optional
from PyQt5.QtCore import pyqtSignal

from image_velocimetry_tools.gui.models.base_model import BaseModel


class GridModel(BaseModel):
    """Model for grid generation state management.

    This model holds all state related to grid generation including:
    - Grid configuration (spacing, number of points)
    - Generated grid results (pixel and world coordinates)
    - Binary mask for region of interest
    - Line mode settings (simple line vs cross-section)

    Signals:
        grid_created: Emitted when a grid is successfully generated
        line_grid_created: Emitted when a line grid is successfully generated
        grid_cleared: Emitted when grid is cleared
    """

    # Qt Signals
    grid_created = pyqtSignal()
    line_grid_created = pyqtSignal(str)  # mode: "line" or "cross_section"
    grid_cleared = pyqtSignal()

    def __init__(self):
        """Initialize the grid model."""
        super().__init__()

        # Grid spacing configuration
        self._horz_grid_size: int = 50  # Horizontal spacing in pixels
        self._vert_grid_size: int = 50  # Vertical spacing in pixels

        # Line grid configuration
        self._number_grid_points_along_line: int = 25
        self._number_grid_points_along_xs_line: int = 25  # Cross-section line
        self._line_mode: str = "line"  # "line" or "cross_section"

        # Grid results
        self._results_grid: Optional[np.ndarray] = (
            None  # Pixel coordinates (N x 2)
        )
        self._results_grid_world: Optional[np.ndarray] = (
            None  # World coordinates (N x 2)
        )
        self._binary_mask: Optional[np.ndarray] = None  # ROI mask

        # Grid type flags
        self._is_cross_section_grid: bool = False

        # Region of interest
        self._region_of_interest_pixels: Optional[np.ndarray] = None

    # ==================== Grid Spacing Properties ====================

    @property
    def horz_grid_size(self) -> int:
        """Get horizontal grid spacing in pixels."""
        return self._horz_grid_size

    @horz_grid_size.setter
    def horz_grid_size(self, value: int):
        """Set horizontal grid spacing."""
        self._horz_grid_size = value

    @property
    def vert_grid_size(self) -> int:
        """Get vertical grid spacing in pixels."""
        return self._vert_grid_size

    @vert_grid_size.setter
    def vert_grid_size(self, value: int):
        """Set vertical grid spacing."""
        self._vert_grid_size = value

    # ==================== Line Grid Properties ====================

    @property
    def number_grid_points_along_line(self) -> int:
        """Get number of grid points along simple line."""
        return self._number_grid_points_along_line

    @number_grid_points_along_line.setter
    def number_grid_points_along_line(self, value: int):
        """Set number of grid points along simple line."""
        self._number_grid_points_along_line = value

    @property
    def number_grid_points_along_xs_line(self) -> int:
        """Get number of grid points along cross-section line."""
        return self._number_grid_points_along_xs_line

    @number_grid_points_along_xs_line.setter
    def number_grid_points_along_xs_line(self, value: int):
        """Set number of grid points along cross-section line."""
        self._number_grid_points_along_xs_line = value

    @property
    def line_mode(self) -> str:
        """Get line mode ('line' or 'cross_section')."""
        return self._line_mode

    @line_mode.setter
    def line_mode(self, value: str):
        """Set line mode."""
        if value not in ["line", "cross_section"]:
            raise ValueError(
                f"Invalid line mode: {value}. Must be 'line' or 'cross_section'"
            )
        self._line_mode = value

    # ==================== Grid Results Properties ====================

    @property
    def results_grid(self) -> Optional[np.ndarray]:
        """Get results grid (pixel coordinates)."""
        return self._results_grid

    @results_grid.setter
    def results_grid(self, value: Optional[np.ndarray]):
        """Set results grid."""
        self._results_grid = value

    @property
    def results_grid_world(self) -> Optional[np.ndarray]:
        """Get results grid in world coordinates."""
        return self._results_grid_world

    @results_grid_world.setter
    def results_grid_world(self, value: Optional[np.ndarray]):
        """Set results grid world coordinates."""
        self._results_grid_world = value

    @property
    def binary_mask(self) -> Optional[np.ndarray]:
        """Get binary mask for region of interest."""
        return self._binary_mask

    @binary_mask.setter
    def binary_mask(self, value: Optional[np.ndarray]):
        """Set binary mask."""
        self._binary_mask = value

    # ==================== Grid Type Properties ====================

    @property
    def is_cross_section_grid(self) -> bool:
        """Check if grid is for cross-section analysis."""
        return self._is_cross_section_grid

    @is_cross_section_grid.setter
    def is_cross_section_grid(self, value: bool):
        """Set cross-section grid flag."""
        self._is_cross_section_grid = value

    @property
    def region_of_interest_pixels(self) -> Optional[np.ndarray]:
        """Get region of interest pixels."""
        return self._region_of_interest_pixels

    @region_of_interest_pixels.setter
    def region_of_interest_pixels(self, value: Optional[np.ndarray]):
        """Set region of interest pixels."""
        self._region_of_interest_pixels = value

    # ==================== Model Methods ====================

    def reset(self):
        """Reset all grid state to initial values."""
        self._horz_grid_size = 50
        self._vert_grid_size = 50
        self._number_grid_points_along_line = 25
        self._number_grid_points_along_xs_line = 25
        self._line_mode = "line"
        self._results_grid = None
        self._results_grid_world = None
        self._binary_mask = None
        self._is_cross_section_grid = False
        self._region_of_interest_pixels = None

    def has_grid(self) -> bool:
        """Check if a grid has been generated.

        Returns:
            True if results_grid is not None and has points
        """
        return self._results_grid is not None and len(self._results_grid) > 0

    def get_grid_size(self) -> tuple:
        """Get grid spacing as (horizontal, vertical) tuple.

        Returns:
            Tuple of (horz_grid_size, vert_grid_size)
        """
        return (self._horz_grid_size, self._vert_grid_size)

    def get_num_grid_points(self) -> int:
        """Get number of points in the current grid.

        Returns:
            Number of grid points, or 0 if no grid exists
        """
        if self._results_grid is None:
            return 0
        return len(self._results_grid)
