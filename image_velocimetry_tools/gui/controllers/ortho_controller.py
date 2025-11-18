"""Controller for orthorectification UI coordination."""

import logging
import os
import shutil
import pandas as pd
from typing import Optional
from PyQt5.QtCore import pyqtSlot, QDir
from PyQt5 import QtWidgets

from image_velocimetry_tools.gui.controllers.base_controller import BaseController
from image_velocimetry_tools.gui.models.ortho_model import OrthoModel
from image_velocimetry_tools.services.orthorectification_service import OrthorectificationService


class OrthoController(BaseController):
    """Controller for orthorectification UI.

    This controller coordinates between:
    - Orthorectification UI widgets (GCP table, image viewers, dialogs)
    - OrthoModel (state management for GCPs and rectification params)
    - OrthorectificationService (business logic for rectification)

    Responsibilities:
    - GCP table loading and management
    - GCP image loading and digitization coordination
    - Rectification parameter calculation coordination
    - UI state updates based on model changes

    NOTE: This is a minimal implementation focusing on core delegation.
    Full orthorectification workflow integration is pending.
    """

    def __init__(
        self,
        main_window,
        ortho_model: OrthoModel,
        ortho_service: OrthorectificationService
    ):
        """Initialize the orthorectification controller.

        Args:
            main_window: Reference to main window for widget access
            ortho_model: Orthorectification state model
            ortho_service: Orthorectification business logic service
        """
        super().__init__(main_window, ortho_model, ortho_service)
        self.ortho_model = ortho_model
        self.ortho_service = ortho_service

        # Connect signals after initialization
        self._connect_signals()

    def _connect_signals(self):
        """Connect UI signals to controller methods and model signals to UI updates."""
        # Model signals
        self.ortho_model.gcp_table_loaded.connect(self.on_model_gcp_table_loaded)
        self.ortho_model.gcp_table_changed.connect(self.on_model_gcp_table_changed)
        self.ortho_model.gcp_image_loaded.connect(self.on_model_gcp_image_loaded)
        self.ortho_model.rectification_calculated.connect(self.on_model_rectification_calculated)

        self.logger.debug("Orthorectification controller signals connected")

    @pyqtSlot(str)
    def on_model_gcp_table_loaded(self, file_path: str):
        """Handle GCP table loaded signal from model."""
        self.logger.info(f"GCP table loaded: {file_path}")

    @pyqtSlot()
    def on_model_gcp_table_changed(self):
        """Handle GCP table changed signal from model."""
        self.logger.debug("GCP table data changed")

    @pyqtSlot(str)
    def on_model_gcp_image_loaded(self, file_path: str):
        """Handle GCP image loaded signal from model."""
        self.logger.info(f"GCP image loaded: {file_path}")

    @pyqtSlot(str)
    def on_model_rectification_calculated(self, method: str):
        """Handle rectification calculated signal from model."""
        self.logger.info(f"Rectification calculated using method: {method}")

    def load_gcp_image(self, image_filename: Optional[str] = None) -> bool:
        """Load GCP image for orthorectification.

        Args:
            image_filename: Path to GCP image file. If None, opens file dialog.

        Returns:
            True if image was successfully loaded, False otherwise
        """
        mw = self.main_window

        # Get last GCP image path from sticky settings
        try:
            ss = mw.sticky_settings.get("last_ortho_gcp_image_path")
            if ss is not None:
                last_path = ss
            else:
                last_path = QDir.homePath()
        except (KeyError, AttributeError):
            last_path = QDir.homePath()

        # Open file dialog if no filename provided
        if image_filename is None:
            filter_spec = "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;All files (*.*)"
            image_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                None,
                "Open GCP Image",
                last_path,
                filter_spec,
            )

            if not image_filename:
                return False  # User cancelled

        # Load image using the ortho_original_image widget
        try:
            mw.ortho_original_image.open(image_filename)
        except Exception as e:
            self.logger.error(f"Error loading GCP image: {str(e)}")
            mw.warning_dialog(
                "Error Loading GCP Image",
                f"An error occurred while loading the image: {str(e)}",
                style="ok",
                icon=mw.__icon_path__ + os.sep + "IVy_logo.ico",
            )
            return False

        # Save copy as calibration image
        try:
            destination_path = os.path.join(
                mw.swap_orthorectification_directory,
                "!calibration_image.jpg"
            )
            shutil.copy(image_filename, destination_path)
        except Exception as e:
            self.logger.warning(f"Failed to save calibration image copy: {e}")

        # Enable UI widgets
        mw.set_qwidget_state_by_name(
            [
                "groupboxOrthoOrigImageTools",
                "toolbuttonOrthoOrigImageDigitizePoint",
            ],
            True,
        )
        mw.ortho_original_image.setEnabled(True)

        # Update sticky settings
        try:
            mw.sticky_settings.set("last_ortho_gcp_image_path", image_filename)
        except (KeyError, AttributeError):
            try:
                mw.sticky_settings.new("last_ortho_gcp_image_path", image_filename)
            except AttributeError:
                pass

        # Emit signal
        self.ortho_model.gcp_image_loaded.emit(image_filename)

        # Update status bar
        message = (
            "GCP Image Loaded. Drag a box or use scroll wheel to zoom, right-click to reset. "
            "A GCP table must be loaded to continue."
        )
        mw.update_statusbar(message)

        self.logger.info(f"GCP image loaded: {image_filename}")
        return True

    def load_gcp_table(self, file_name: Optional[str] = None) -> bool:
        """Load GCP table from CSV file.

        Args:
            file_name: Path to CSV file. If None, opens file dialog.

        Returns:
            True if table was successfully loaded, False otherwise

        NOTE: This is a minimal implementation. Full GCP table loading with
        validation and UI updates is pending.
        """
        mw = self.main_window

        # Get last GCP table path from sticky settings
        try:
            ss = mw.sticky_settings.get("last_orthotable_file_name")
            if ss is not None:
                last_path = ss
            else:
                last_path = QDir.homePath()
        except (KeyError, AttributeError):
            last_path = QDir.homePath()

        # Open file dialog if no filename provided
        if file_name is None:
            filter_spec = "CSV files (*.csv);;All files (*.*)"
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                None,
                "Open GCP Table",
                last_path,
                filter_spec,
            )

            if not file_name:
                return False  # User cancelled

        # Load CSV file
        try:
            df = pd.read_csv(file_name)
        except Exception as e:
            self.logger.error(f"Error loading GCP table: {str(e)}")
            mw.warning_dialog(
                "Error Loading GCP Table",
                f"An error occurred while loading the table: {str(e)}",
                style="ok",
                icon=mw.__icon_path__ + os.sep + "IVy_logo.ico",
            )
            return False

        # Validate table structure
        # TODO: Add comprehensive validation

        # Update model
        self.ortho_model.orthotable_dataframe = df.copy(deep=True)
        self.ortho_model.orthotable_file_name = file_name
        self.ortho_model.orthotable_fname = os.path.splitext(os.path.basename(file_name))[0]
        self.ortho_model.orthotable_is_changed = False
        self.ortho_model.is_ortho_table_loaded = True

        # Update sticky settings
        try:
            mw.sticky_settings.set("last_orthotable_file_name", file_name)
        except (KeyError, AttributeError):
            try:
                mw.sticky_settings.new("last_orthotable_file_name", file_name)
            except AttributeError:
                pass

        # Emit signal
        self.ortho_model.gcp_table_loaded.emit(file_name)

        # Update status bar
        message = f"GCP table loaded: {df.shape[0]} points"
        mw.update_statusbar(message)

        self.logger.info(f"GCP table loaded: {file_name} ({df.shape[0]} points)")
        return True

    def save_gcp_table(self, file_name: Optional[str] = None) -> bool:
        """Save GCP table to CSV file.

        Args:
            file_name: Path to save CSV file. If None, opens file dialog.

        Returns:
            True if table was successfully saved, False otherwise
        """
        mw = self.main_window

        # Check if table has data
        if self.ortho_model.orthotable_dataframe.empty:
            mw.warning_dialog(
                "No Data to Save",
                "The GCP table is empty. Nothing to save.",
                style="ok",
                icon=mw.__icon_path__ + os.sep + "IVy_logo.ico",
            )
            return False

        # Get save filename
        if file_name is None:
            default_name = self.ortho_model.orthotable_file_name
            if not default_name:
                default_name = QDir.homePath() + os.sep + "gcp_table.csv"

            filter_spec = "CSV files (*.csv);;All files (*.*)"
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                None,
                "Save GCP Table",
                default_name,
                filter_spec,
            )

            if not file_name:
                return False  # User cancelled

        # Ensure .csv extension
        if not file_name.endswith('.csv'):
            file_name += '.csv'

        # Save to CSV
        try:
            self.ortho_model.orthotable_dataframe.to_csv(file_name, index=False)
        except Exception as e:
            self.logger.error(f"Error saving GCP table: {str(e)}")
            mw.warning_dialog(
                "Error Saving GCP Table",
                f"An error occurred while saving the table: {str(e)}",
                style="ok",
                icon=mw.__icon_path__ + os.sep + "IVy_logo.ico",
            )
            return False

        # Update model
        self.ortho_model.orthotable_file_name = file_name
        self.ortho_model.orthotable_fname = os.path.splitext(os.path.basename(file_name))[0]
        self.ortho_model.orthotable_is_changed = False

        # Update sticky settings
        try:
            mw.sticky_settings.set("last_orthotable_file_name", file_name)
        except (KeyError, AttributeError):
            pass

        # Update status bar
        message = f"GCP table saved: {file_name}"
        mw.update_statusbar(message)

        self.logger.info(f"GCP table saved: {file_name}")
        return True

    def set_flip_x(self, enabled: bool):
        """Set horizontal flip for rectified images.

        Args:
            enabled: True to enable horizontal flip
        """
        self.ortho_model.is_ortho_flip_x = enabled
        self.logger.debug(f"Horizontal flip set to: {enabled}")

    def set_flip_y(self, enabled: bool):
        """Set vertical flip for rectified images.

        Args:
            enabled: True to enable vertical flip
        """
        self.ortho_model.is_ortho_flip_y = enabled
        self.logger.debug(f"Vertical flip set to: {enabled}")

    def set_water_surface_elevation(self, elevation_m: float):
        """Set water surface elevation for camera matrix rectification.

        Args:
            elevation_m: Water surface elevation in meters
        """
        self.ortho_model.ortho_rectified_wse_m = elevation_m
        self.logger.debug(f"Water surface elevation set to: {elevation_m}m")

    def calculate_rectification_parameters(self) -> bool:
        """Calculate rectification parameters from loaded GCP table.

        Returns:
            True if parameters were successfully calculated, False otherwise

        NOTE: This is a placeholder for the full rectification calculation workflow.
        Implementation pending.
        """
        mw = self.main_window

        # Check if GCP table is loaded
        if not self.ortho_model.is_ortho_table_loaded:
            mw.warning_dialog(
                "No GCP Table",
                "Please load a GCP table before calculating rectification parameters.",
                style="ok",
                icon=mw.__icon_path__ + os.sep + "IVy_logo.ico",
            )
            return False

        # TODO: Implement full rectification calculation workflow
        # - Extract pixel and world coordinates from table
        # - Determine rectification method
        # - Calculate parameters using ortho_service
        # - Update ortho_model with results
        # - Emit rectification_calculated signal

        self.logger.warning(
            "calculate_rectification_parameters() is a placeholder - full implementation pending"
        )

        message = "Rectification parameter calculation not yet implemented"
        mw.update_statusbar(message)

        return False
