"""Controller for grid generation UI coordination."""

import logging
import os
import csv
import numpy as np
from typing import Optional, Tuple
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtWidgets
from PIL import Image

from image_velocimetry_tools.gui.controllers.base_controller import BaseController
from image_velocimetry_tools.gui.models.grid_model import GridModel
from image_velocimetry_tools.services.grid_service import GridService
from image_velocimetry_tools.graphics import Instructions


class GridController(BaseController):
    """Controller for grid generation UI coordination.

    This controller coordinates between:
    - Grid UI widgets (spacing spinboxes, create buttons)
    - GridModel (state management for grid configuration and results)
    - GridService (business logic for grid generation)

    Responsibilities:
    - Grid generation (regular grid and line grid)
    - Binary mask creation from polygons
    - Grid visualization on image browsers
    - Grid export to CSV
    """

    def __init__(
        self,
        main_window,
        grid_model: GridModel,
        grid_service: GridService
    ):
        """Initialize the grid controller.

        Args:
            main_window: Reference to main window for widget access
            grid_model: Grid state model
            grid_service: Grid business logic service
        """
        super().__init__(main_window, grid_model, grid_service)
        self.grid_model = grid_model
        self.grid_service = grid_service

        # Connect signals after initialization
        self._connect_signals()

    def _connect_signals(self):
        """Connect UI signals to controller methods and model signals to UI updates."""
        # Model signals
        self.grid_model.grid_created.connect(self.on_model_grid_created)
        self.grid_model.line_grid_created.connect(self.on_model_line_grid_created)
        self.grid_model.grid_cleared.connect(self.on_model_grid_cleared)

        self.logger.debug("Grid controller signals connected")

    @pyqtSlot()
    def on_model_grid_created(self):
        """Handle grid created signal from model."""
        self.logger.info(f"Grid created with {self.grid_model.get_num_grid_points()} points")

    @pyqtSlot(str)
    def on_model_line_grid_created(self, mode: str):
        """Handle line grid created signal from model."""
        self.logger.info(f"Line grid created in {mode} mode with {self.grid_model.get_num_grid_points()} points")

    @pyqtSlot()
    def on_model_grid_cleared(self):
        """Handle grid cleared signal from model."""
        self.logger.info("Grid cleared")

    # ==================== Grid Spacing Configuration ====================

    def set_horizontal_spacing(self, value: int):
        """Set horizontal grid spacing.

        Args:
            value: Spacing in pixels
        """
        self.grid_model.horz_grid_size = value
        self.logger.debug(f"Horizontal grid spacing set to {value} px")

    def set_vertical_spacing(self, value: int):
        """Set vertical grid spacing.

        Args:
            value: Spacing in pixels
        """
        self.grid_model.vert_grid_size = value
        self.logger.debug(f"Vertical grid spacing set to {value} px")

    def set_line_num_points(self, value: int):
        """Set number of points along simple line.

        Args:
            value: Number of points
        """
        self.grid_model.number_grid_points_along_line = value
        self.logger.debug(f"Line grid points set to {value}")

    def set_xs_line_num_points(self, value: int):
        """Set number of points along cross-section line.

        Args:
            value: Number of points
        """
        self.grid_model.number_grid_points_along_xs_line = value
        self.logger.debug(f"Cross-section line grid points set to {value}")

    # ==================== Regular Grid Generation ====================

    def create_regular_grid(self) -> bool:
        """Create a regular computational grid.

        Returns:
            True if grid was successfully created, False otherwise
        """
        mw = self.main_window

        # Check if image is loaded
        if not mw.gridpreparation.imageBrowser.has_image():
            self.logger.warning("No image loaded in grid preparation browser")
            return False

        # Get parameters
        horz = self.grid_model.horz_grid_size
        vert = self.grid_model.vert_grid_size
        image = mw.gridpreparation.imageBrowser.scene.ndarray()
        mask_polygons = mw.gridpreparation.imageBrowser.polygons_ndarray()

        message = "GRID PREPARATION: Creating results grid"
        mw.update_statusbar(message)

        with mw.wait_cursor():
            # Generate grid using service
            height, width = image.shape[:2]

            # Handle mask_polygons - check for None before checking size
            has_mask = mask_polygons is not None and mask_polygons.size > 0

            try:
                grid, binary_mask = self.grid_service.generate_regular_grid(
                    width,
                    height,
                    vert,  # vertical_spacing
                    horz,  # horizontal_spacing
                    mask_polygons=mask_polygons if has_mask else None,
                    clean_mask=True
                )
            except Exception as e:
                self.logger.error(f"Error generating grid: {str(e)}")
                mw.update_statusbar(f"Error generating grid: {str(e)}")
                return False

            # Update model
            self.grid_model.results_grid = grid
            self.grid_model.binary_mask = binary_mask
            self.grid_model.results_grid_world = grid  # TODO: Transform to world coords
            self.grid_model.is_cross_section_grid = False

            # Clear existing visualizations
            mw.gridpreparation.imageBrowser.clearPoints()
            mw.gridpreparation.imageBrowser.clearLines()

            # Visualize grid on image browser
            mw.gridpreparation.imageBrowser.scene.set_current_instruction(
                Instructions.ADD_POINTS_INSTRUCTION,
                points=grid,
                labels=["" for _ in grid],
            )

            # Save grid to CSV
            self._save_grid_to_csv(grid, self.grid_model.results_grid_world)

            # Save binary mask
            self._save_binary_mask(binary_mask)

            # Update world units labels if pixel GSD is available
            try:
                pixel_gsd = mw.pixel_ground_scale_distance_m
                horz_world = pixel_gsd * horz
                vert_world = pixel_gsd * vert
                mw.labelHorzSpacingWorldUnits.setText(f"px ({horz_world:.2f} world)")
                mw.labelVertSpacingWorldUnits.setText(f"px ({vert_world:.2f} world)")
            except (AttributeError, TypeError):
                pass

            # Enable image velocimetry processing
            mw.groupboxSpaceTimeParameters.setEnabled(True)
            mw.groupboxSpaceTimeOptParameters.setEnabled(True)

        # Emit signal
        self.grid_model.grid_created.emit()

        message = f"GRID PREPARATION: Results grid created with {len(grid)} points"
        mw.update_statusbar(message)
        self.logger.info(f"Regular grid created: {len(grid)} points")

        return True

    # ==================== Line Grid Generation ====================

    def create_line_grid(self, mode: str = "line") -> bool:
        """Create points along a line.

        Args:
            mode: Grid mode - "line" or "cross_section"

        Returns:
            True if line grid was successfully created, False otherwise
        """
        mw = self.main_window

        # Check if image is loaded
        if not mw.gridpreparation.imageBrowser.has_image():
            self.logger.warning("No image loaded in grid preparation browser")
            return False

        # Validate mode
        if mode not in ["line", "cross_section"]:
            self.logger.error(f"Invalid grid mode: {mode}")
            return False

        # Update model
        self.grid_model.line_mode = mode

        # Get number of points based on mode
        if mode == "line":
            num_points = self.grid_model.number_grid_points_along_line
            message = "GRID PREPARATION: Creating points along digitized line"
        else:  # cross_section
            num_points = self.grid_model.number_grid_points_along_xs_line
            self.grid_model.is_cross_section_grid = True
            message = "GRID PREPARATION: Creating points along digitized cross-section line"

        # Get image and line endpoints
        image = mw.gridpreparation.imageBrowser.scene.ndarray()
        mask_polygons = mw.gridpreparation.imageBrowser.polygons_ndarray()
        line_eps = mw.gridpreparation.imageBrowser.lines_ndarray()

        if not np.any(line_eps):
            self.logger.warning("No line digitized on image")
            mw.update_statusbar("Please digitize a line first")
            return False

        line_start = line_eps[-1, 0]
        line_end = line_eps[-1, 1]

        mw.update_statusbar(message)

        with mw.wait_cursor():
            # Generate line grid using service
            height, width = image.shape[:2]

            # Handle mask_polygons - check for None before checking size
            has_mask = mask_polygons is not None and mask_polygons.size > 0

            try:
                line_grid, binary_mask = self.grid_service.generate_line_grid(
                    width,
                    height,
                    line_start,
                    line_end,
                    num_points,
                    mask_polygons=mask_polygons if has_mask else None,
                    clean_mask=True
                )
            except Exception as e:
                self.logger.error(f"Error generating line grid: {str(e)}")
                mw.update_statusbar(f"Error generating line grid: {str(e)}")
                return False

            # Update model
            self.grid_model.results_grid = line_grid
            self.grid_model.binary_mask = binary_mask
            self.grid_model.results_grid_world = line_grid  # TODO: Transform to world coords

            # Clear and visualize grid
            mw.gridpreparation.imageBrowser.clearPoints()
            labels = [str(i + 1) for i in range(line_grid.shape[0])]

            mw.gridpreparation.imageBrowser.scene.set_current_instruction(
                Instructions.ADD_POINTS_INSTRUCTION,
                points=line_grid,
                labels=labels,
            )
            mw.gridpreparation.imageBrowser.scene.set_current_instruction(
                Instructions.NO_INSTRUCTION
            )

            # Also add points to STIV image browsers
            mw.stiv.imageBrowser.clearPoints()
            mw.stiv.imageBrowser.scene.set_current_instruction(
                Instructions.ADD_POINTS_INSTRUCTION,
                points=line_grid,
                labels=labels,
            )

            mw.stiv_opt.imageBrowser.clearPoints()
            mw.stiv_opt.imageBrowser.scene.set_current_instruction(
                Instructions.ADD_POINTS_INSTRUCTION,
                points=line_grid,
                labels=labels,
            )

        # Emit signal
        self.grid_model.line_grid_created.emit(mode)

        message = f"GRID PREPARATION: Line grid created with {len(line_grid)} points"
        mw.update_statusbar(message)
        self.logger.info(f"Line grid created ({mode}): {len(line_grid)} points")

        return True

    # ==================== Grid Export ====================

    def _save_grid_to_csv(self, pixel_grid: np.ndarray, world_grid: np.ndarray):
        """Save grid to CSV file.

        Args:
            pixel_grid: Grid in pixel coordinates
            world_grid: Grid in world coordinates
        """
        mw = self.main_window

        try:
            filepath = os.path.join(mw.swap_grids_directory, "results_grid.csv")

            with open(filepath, "w", newline="") as csvfile:
                fieldnames = [
                    "world_coords_x",
                    "world_coords_y",
                    "pixel_coords_x",
                    "pixel_coords_y",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for world_coord, pixel_coord in zip(world_grid, pixel_grid):
                    row_data = {
                        "world_coords_x": world_coord[0],
                        "world_coords_y": world_coord[1],
                        "pixel_coords_x": pixel_coord[0],
                        "pixel_coords_y": pixel_coord[1],
                    }
                    writer.writerow(row_data)

            self.logger.debug(f"Grid saved to CSV: {filepath}")
        except Exception as e:
            self.logger.warning(f"Failed to save grid CSV: {e}")

    def _save_binary_mask(self, binary_mask: np.ndarray):
        """Save binary mask to image file.

        Args:
            binary_mask: Binary mask array
        """
        mw = self.main_window

        try:
            binary_mask_uint8 = binary_mask.astype(np.uint8) * 255
            image = Image.fromarray(binary_mask_uint8)
            filepath = os.path.join(mw.swap_grids_directory, "binary_mask.jpg")
            image.save(filepath)
            self.logger.debug(f"Binary mask saved: {filepath}")
        except Exception as e:
            self.logger.warning(f"Failed to save binary mask: {e}")

    def set_grid_background_image(self, image_filename: str):
        """Set background image for grid preparation.

        Args:
            image_filename: Path to image file
        """
        mw = self.main_window

        try:
            mw.gridpreparation.imageBrowser.open(image_filename)
            self.logger.info(f"Grid background image loaded: {image_filename}")
        except Exception as e:
            self.logger.error(f"Error loading grid background image: {str(e)}")
