"""Controller for orthorectification UI coordination.

This controller handles all orthorectification-related functionality including:
- GCP image and table loading/saving
- GCP table management (add/remove/edit rows/columns)
- GCP point digitization on images
- Rectification parameter calculation (scale, homography, camera matrix methods)
- Single frame and multi-frame orthorectification
- Image zoom and view controls
"""

import logging
import os
import shutil
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from PyQt5.QtCore import pyqtSlot, QDir, Qt
from PyQt5 import QtWidgets, QtGui

from image_velocimetry_tools.gui.controllers.base_controller import BaseController
from image_velocimetry_tools.gui.models.ortho_model import OrthoModel
from image_velocimetry_tools.services.orthorectification_service import OrthorectificationService
from image_velocimetry_tools.graphics import Instructions
from image_velocimetry_tools.common_functions import (
    find_matches_between_two_lists,
    units_conversion,
    string_to_boolean,
)


class OrthoController(BaseController):
    """Controller for orthorectification UI coordination.

    This controller implements the full orthorectification workflow following
    the Model-View-Presenter pattern.
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

    # ==================== Signal Handlers ====================

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

    # ==================== Image Zoom and View Control ====================

    def set_original_image_zoom(self, zoom_value: float):
        """Set zoom level for original (GCP) image.

        Args:
            zoom_value: Zoom factor (1.0 = 100%)
        """
        mw = self.main_window
        self.ortho_model.ortho_original_image_zoom_factor = zoom_value
        mw.ortho_original_image.zoomEvent(zoom_value)

    def set_rectified_image_zoom(self, zoom_value: float):
        """Set zoom level for rectified image.

        Args:
            zoom_value: Zoom factor (1.0 = 100%)
        """
        mw = self.main_window
        self.ortho_model.ortho_rectified_image_zoom_factor = zoom_value
        mw.ortho_rectified_image.zoomEvent(zoom_value)

    def reset_original_image_zoom(self):
        """Reset original image to normal size (100%)."""
        mw = self.main_window
        mw.ortho_original_image.clearZoom()
        self.ortho_model.ortho_original_image_zoom_factor = 1.0

    def reset_rectified_image_zoom(self):
        """Reset rectified image to normal size (100%)."""
        mw = self.main_window
        mw.ortho_rectified_image.clearZoom()
        self.ortho_model.ortho_rectified_image_zoom_factor = 1.0

    # ==================== Water Surface Elevation ====================

    def update_water_surface_elevation(self):
        """Update water surface elevation from spinbox value."""
        mw = self.main_window
        item = mw.doubleSpinBoxRectificationWaterSurfaceElevation.value()

        # Convert to meters based on display units
        if mw.display_units == "Metric":
            elevation_m = float(item)
        else:  # English units
            elevation_m = float(item) * 1 / mw.survey_units["L"]

        self.ortho_model.ortho_rectified_wse_m = elevation_m

        # Emit signal for AC3 backend updates if needed
        mw.signal_wse_changed.emit(elevation_m)

        self.logger.debug(f"Water surface elevation updated: {elevation_m}m")

    # ==================== Flip Settings ====================

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

    # ==================== GCP Image Loading ====================

    def load_gcp_image(self, image_filename: Optional[str] = None) -> bool:
        """Load GCP image for orthorectification.

        Args:
            image_filename: Path to GCP image file. If None, opens file dialog.

        Returns:
            True if image was successfully loaded, False otherwise
        """
        mw = self.main_window

        # Handle bool from Qt signals (treat as None)
        if isinstance(image_filename, bool):
            image_filename = None

        # Get last GCP image path from sticky settings
        try:
            ss = mw.sticky_settings.get("last_ortho_gcp_image_path")
            if ss is not None and isinstance(ss, str):
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

    # ==================== GCP Table File Operations ====================

    def load_gcp_table_dialog(self) -> bool:
        """Open file dialog and load GCP table from CSV.

        Returns:
            True if table was successfully loaded, False otherwise
        """
        mw = self.main_window

        # Get last GCP table path from sticky settings
        try:
            ss = mw.sticky_settings.get("last_orthotable_file_name")
            last_path = ss if (ss is not None and isinstance(ss, str)) else QDir.homePath()
        except KeyError:
            last_path = QDir.homePath()

        # Check if table has unsaved changes
        if self.ortho_model.orthotable_is_changed:
            # Could add save prompt here
            pass

        # Open file dialog
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            mw,
            "Open CSV containing Ground Control Point data",
            last_path,
            "CSV (*.csv *.tsv *.txt)",
        )

        if not file_name:
            return False  # User cancelled

        # Update sticky settings
        try:
            mw.sticky_settings.set("last_orthotable_file_name", file_name)
        except KeyError:
            mw.sticky_settings.new("last_orthotable_file_name", file_name)

        # Load the table
        return self.load_gcp_table_from_file(file_name)

    def load_gcp_table_from_file(self, file_name: str) -> bool:
        """Load GCP table from CSV file (used by project loading and user dialog).

        Args:
            file_name: Path to CSV file

        Returns:
            True if table was successfully loaded, False otherwise
        """
        if not file_name:
            return False

        mw = self.main_window

        try:
            # Define unit prompt callback
            def prompt_units():
                choices = ("English", "Metric")
                idx = mw.custom_dialog_index(
                    title="Ground Control Points Unit Selection",
                    message="Units not detected in GCP file.\\nPlease select units used in the survey:",
                    choices=choices,
                )
                return choices[idx]

            # Load and parse GCP CSV using existing utility
            from image_velocimetry_tools.file_management import load_and_parse_gcp_csv

            df, units = load_and_parse_gcp_csv(
                file_name=file_name,
                swap_ortho_path=mw.swap_orthorectification_directory,
                unit_prompt_callback=prompt_units,
            )

            # Update model with survey units
            self.ortho_model.orthotable_file_survey_units = units

        except ValueError as e:
            QtWidgets.QErrorMessage().showMessage(str(e)).exec_()
            return False
        except Exception as e:
            mw.update_statusbar(f"Failed to load GCP CSV: {e}")
            return False

        # Clear existing points from image viewers
        mw.ortho_original_image.clearPoints()
        mw.ortho_original_image.clearPolygons()

        # Update model
        self.ortho_model.orthotable_dataframe = df.copy(deep=True)
        self.ortho_model.orthotable_file_name = file_name
        self.ortho_model.orthotable_fname = os.path.splitext(os.path.basename(file_name))[0]
        self.ortho_model.orthotable_is_changed = False
        self.ortho_model.is_ortho_table_loaded = True

        # Populate UI table
        self.populate_table_widget(df)

        # Update table appearance
        mw.orthoPointsTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        mw.orthoPointsTable.resizeColumnsToContents()
        mw.orthoPointsTable.resizeRowsToContents()
        mw.orthoPointsTable.selectRow(0)

        # Enable controls
        mw.toolbuttonOrthoOrigImageDigitizePoint.setEnabled(True)
        mw.groupboxExportOrthoFrames.setEnabled(True)

        # Emit signals
        mw.signal_orthotable_check_units.emit()
        self.ortho_model.gcp_table_loaded.emit(file_name)

        # Save copy to project directory
        try:
            dest = os.path.join(mw.swap_orthorectification_directory, "ground_control_points.csv")
            shutil.copy(file_name, dest)
        except Exception as e:
            mw.update_statusbar(f"Failed to save GCP table to project: {e}")

        self.logger.info(f"GCP table loaded: {file_name} ({df.shape[0]} points, {units} units)")
        return True

    def save_gcp_table_dialog(self) -> bool:
        """Open file dialog and save GCP table to CSV.

        Returns:
            True if table was successfully saved, False otherwise
        """
        mw = self.main_window

        # Get current filename or default
        try:
            ss = mw.sticky_settings.get("last_orthotable_file_name")
            default_name = ss if ss else f"{QDir.homePath()}{os.sep}IVy_Points_Table_{mw.display_units}.csv"
        except KeyError:
            default_name = f"{QDir.homePath()}{os.sep}IVy_Points_Table_{mw.display_units}.csv"

        # Open save dialog
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            mw,
            "Save CSV containing Ground Control Point data",
            default_name,
            "CSV (*.csv)",
        )

        if not file_name:
            return False  # User cancelled

        # Get data from UI table
        dict_data = mw.get_table_as_dict(mw.orthoPointsTable)

        # Save to CSV
        try:
            pd.DataFrame(dict_data).fillna("").to_csv(file_name, index=False)
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
        except KeyError:
            mw.sticky_settings.new("last_orthotable_file_name", file_name)

        self.logger.info(f"GCP table saved: {file_name}")
        return True

    # ==================== GCP Table Management ====================

    def populate_table_widget(self, dataframe: pd.DataFrame):
        """Populate the GCP table widget with dataframe data.

        Args:
            dataframe: DataFrame containing GCP data (in meters)
        """
        mw = self.main_window

        # Set table dimensions
        mw.orthoPointsTable.setColumnCount(len(dataframe.columns))
        mw.orthoPointsTable.setRowCount(len(dataframe.index))

        # Populate cells (converting units for display)
        header_list = dataframe.columns.tolist()
        for i in range(len(dataframe.index)):
            for j in range(len(dataframe.columns)):
                # Convert columns 1-3 (X, Y, Z) to display units
                if j >= 1 and j <= 3:
                    item = dataframe.iat[i, j] * mw.survey_units["L"]
                else:
                    item = dataframe.iat[i, j]
                mw.orthoPointsTable.setItem(i, j, QtWidgets.QTableWidgetItem(str(item)))

        # Set headers
        for j in range(len(dataframe.columns)):
            m = QtWidgets.QTableWidgetItem(header_list[j])
            mw.orthoPointsTable.setHorizontalHeaderItem(j, m)

        self.ortho_model._orthotable_has_headers = True
        mw.orthoPointsTable.setHorizontalHeaderLabels(header_list)

        # Update table headers with units
        self.update_table_headers()

        # Plot points on image
        self.plot_gcp_points_on_original_image()

    def init_table(self):
        """Initialize the GCP table with default headers."""
        mw = self.main_window

        headers = [
            "# ID",
            f"X {mw.survey_units['label_L']}",
            f"Y {mw.survey_units['label_L']}",
            f"Z {mw.survey_units['label_L']}",
            "X (pixel)",
            "Y (pixel)",
            "Error X (pixel)",
            "Error Y (pixel)",
            "Tot. Error (pixel)",
            "Use in Rectification",
            "Use in Validation",
        ]

        mw.orthoPointsTable.setColumnCount(len(headers))
        self.ortho_model._orthotable_has_headers = True
        mw.orthoPointsTable.setHorizontalHeaderLabels(headers)
        mw.orthoPointsTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.ortho_model.orthotable_is_changed = False
        mw.orthoPointsTable.resizeColumnsToContents()
        mw.orthoPointsTable.resizeRowsToContents()
        mw.orthoPointsTable.selectRow(0)
        self.ortho_model.is_ortho_table_loaded = False

    def update_table_headers(self):
        """Update table headers based on current survey units."""
        mw = self.main_window

        headers = [
            "# ID",
            f"X {mw.survey_units['label_L']}",
            f"Y {mw.survey_units['label_L']}",
            f"Z {mw.survey_units['label_L']}",
            "X (pixel)",
            "Y (pixel)",
            "Error X (pixel)",
            "Error Y (pixel)",
            "Tot. Error (pixel)",
            "Use in Rectification",
            "Use in Validation",
        ]

        mw.orthoPointsTable.setHorizontalHeaderLabels(headers)
        mw.orthoPointsTable.resizeColumnsToContents()
        mw.orthoPointsTable.resizeRowsToContents()

    def change_table_selection_mode(self):
        """Enable extended selection mode for GCP table."""
        mw = self.main_window
        mw.orthoPointsTable.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

    def update_table_cell(self):
        """Update a GCP table cell from the edit line widget."""
        mw = self.main_window

        if mw.orthoPointsTable.selectionModel().hasSelection():
            row = self.get_selected_row()
            column = self.get_selected_column()
            newtext = QtWidgets.QTableWidgetItem(mw.editLine.text())
            mw.orthoPointsTable.setItem(row, column, newtext)

    def get_selected_item(self):
        """Get the selected item from the GCP table and populate edit line."""
        mw = self.main_window

        if not mw.orthoPointsTable.selectedItems():
            return

        item = mw.orthoPointsTable.selectedItems()[0]
        name = item.text() if item is not None else ""
        mw.orthoPointsTableLineEdit.setText(name)

    def get_selected_row(self) -> Optional[int]:
        """Get the currently selected row number.

        Returns:
            Row number (int) or None if no selection
        """
        mw = self.main_window

        if mw.orthoPointsTable.selectionModel().hasSelection():
            row = mw.orthoPointsTable.selectionModel().selectedIndexes()[0].row()
            return int(row)
        return None

    def get_selected_column(self) -> Optional[int]:
        """Get the currently selected column number.

        Returns:
            Column number (int) or None if no selection
        """
        mw = self.main_window

        if mw.orthoPointsTable.selectionModel().hasSelection():
            column = mw.orthoPointsTable.selectionModel().selectedIndexes()[0].column()
            return int(column)
        return None

    def mark_table_changed(self):
        """Mark the GCP table as having unsaved changes."""
        mw = self.main_window

        self.ortho_model.orthotable_is_changed = True
        mw.signal_orthotable_changed.emit(True)

    def add_row(self):
        """Add a new row to the GCP table."""
        mw = self.main_window

        if mw.orthoPointsTable.rowCount() > 0:
            if mw.orthoPointsTable.selectionModel().hasSelection():
                row = self.get_selected_row()
                mw.orthoPointsTable.insertRow(row)
            else:
                mw.orthoPointsTable.insertRow(0)
                mw.orthoPointsTable.selectRow(0)
        else:
            mw.orthoPointsTable.setRowCount(1)

        if mw.orthoPointsTable.columnCount() == 0:
            self.add_column()
            mw.orthoPointsTable.selectRow(0)

        self.mark_table_changed()

    def remove_row(self):
        """Remove the selected row from the GCP table with confirmation."""
        mw = self.main_window

        if mw.orthoPointsTable.rowCount() == 0:
            return

        # Confirm removal
        remove = QtWidgets.QMessageBox()
        remove.setText("This will remove the selected row, and cannot be undone. Are you sure?")
        remove.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        result = remove.exec()

        if result == QtWidgets.QMessageBox.Yes:
            row = self.get_selected_row()
            if row is not None:
                mw.orthoPointsTable.removeRow(row)
                self.mark_table_changed()

    def add_column(self):
        """Add a new column to the GCP table."""
        mw = self.main_window

        count = mw.orthoPointsTable.columnCount()
        mw.orthoPointsTable.setColumnCount(count + 1)
        mw.orthoPointsTable.resizeColumnsToContents()
        self.mark_table_changed()

        if mw.orthoPointsTable.rowCount() == 0:
            self.add_row()
            mw.orthoPointsTable.selectRow(0)

    def remove_column(self):
        """Remove the selected column from the GCP table."""
        mw = self.main_window

        column = self.get_selected_column()
        if column is not None:
            mw.orthoPointsTable.removeColumn(column)
            self.mark_table_changed()

    def clear_table(self):
        """Clear all items in the GCP table."""
        mw = self.main_window

        mw.orthoPointsTable.clear()
        self.mark_table_changed()

    def make_all_cells_white(self):
        """Reset all table cell colors to white background."""
        mw = self.main_window

        if self.ortho_model._orthotable_cell_colored:
            for row in range(mw.orthoPointsTable.rowCount()):
                for column in range(mw.orthoPointsTable.columnCount()):
                    item = mw.orthoPointsTable.item(row, column)
                    if item is not None:
                        item.setForeground(Qt.black)
                        item.setBackground(QtGui.QColor("#e1e1e1"))

        self.ortho_model._orthotable_cell_colored = False

    # ==================== GCP Digitization ====================

    def toggle_digitize_mode(self):
        """Toggle point digitization mode on/off."""
        mw = self.main_window

        # Create crosshairs cursor
        pixmap = QtGui.QPixmap(mw.__icon_path__ + os.sep + "crosshairs-solid.svg")
        pixmap = pixmap.scaledToWidth(32)
        cursor = QtGui.QCursor(pixmap, hotX=16, hotY=16)

        if mw.toolbuttonOrthoOrigImageDigitizePoint.isChecked():
            # Enable digitization mode
            mw.ortho_original_image.setCursor(cursor)

            # Connect mouse release to digitize handler
            try:
                mw.ortho_original_image.leftMouseButtonReleased.disconnect()
            except TypeError:
                pass  # Wasn't connected
            mw.ortho_original_image.leftMouseButtonReleased.connect(self.handle_digitized_point)
        else:
            # Disable digitization mode
            mw.ortho_original_image.setCursor(Qt.ArrowCursor)

    def handle_digitized_point(self, x: float, y: float):
        """Handle a digitized point click on the GCP image.

        Args:
            x: X pixel coordinate
            y: Y pixel coordinate
        """
        mw = self.main_window

        row = int(y)
        column = int(x)

        logging.debug(f"Clicked on image pixel (row={row}, column={column})")

        # Store current pixel
        self.ortho_model._ortho_original_image_current_pixel = [x, y]

        logging.debug(f"Pixel Info: x: {x}, y: {y}")
        logging.debug(f"Current selected GCP table row: {mw.orthoPointsTable.currentRow()}")

        if mw.toolbuttonOrthoOrigImageDigitizePoint.isChecked():
            # Update table with pixel coordinates
            current_row = mw.orthoPointsTable.currentRow()
            mw.orthoPointsTable.setItem(
                current_row, 4, QtWidgets.QTableWidgetItem(f"{x:.3f}")
            )
            mw.orthoPointsTable.setItem(
                current_row, 5, QtWidgets.QTableWidgetItem(f"{y:.3f}")
            )

            # Plot updated points on image
            points_to_plot = self.get_points_to_plot()
            mw.signal_ortho_original_digitized_point.emit(points_to_plot)

            mw.ortho_original_image.clearPoints()
            mw.ortho_original_image.scene.set_current_instruction(
                Instructions.ADD_POINTS_INSTRUCTION,
                points=points_to_plot["points"],
                labels=points_to_plot["labels"],
            )

    # ==================== GCP Point Plotting ====================

    def refresh_original_image_plot(self, event):
        """Refresh the GCP points plotted on the original image.

        Args:
            event: Qt event that triggered the refresh
        """
        mw = self.main_window

        if not event:
            return

        # Update orthotable_dataframe with pixel info from GUI table
        if self.ortho_model.orthotable_dataframe.empty:
            ortho_dict = mw.get_table_as_dict(mw.orthoPointsTable)
            self.ortho_model.orthotable_dataframe = pd.DataFrame.from_dict(ortho_dict)

        with mw.wait_cursor():
            current_table_data = mw.get_table_as_dict(mw.orthoPointsTable)
            keys = list(current_table_data.keys())

            if len(keys) >= 11:
                xpixel_key = keys[4]
                ypixel_key = keys[5]
                rectification_key = "Use in Rectification"
                validation_key = "Use in Validation"

                # Get non-empty values for each column
                non_empty_xpixel = [
                    (index, value)
                    for index, value in enumerate(current_table_data[xpixel_key])
                    if value != ""
                ]
                non_empty_ypixel = [
                    (index, value)
                    for index, value in enumerate(current_table_data[ypixel_key])
                    if value != ""
                ]

                # Get rectification and validation flags
                try:
                    non_empty_rectification = [
                        (index, value)
                        for index, value in enumerate(current_table_data[rectification_key])
                        if value != ""
                    ]
                except KeyError:
                    non_empty_rectification = None

                try:
                    non_empty_validation = [
                        (index, value)
                        for index, value in enumerate(current_table_data[validation_key])
                        if value != ""
                    ]
                except KeyError:
                    non_empty_validation = None

                # Update dataframe with current table values
                if non_empty_xpixel and non_empty_ypixel:
                    for index_x, value_x in non_empty_xpixel:
                        self.ortho_model.orthotable_dataframe.iloc[
                            index_x,
                            self.ortho_model.orthotable_dataframe.columns.get_loc("X (pixel)"),
                        ] = value_x

                    for index_y, value_y in non_empty_ypixel:
                        self.ortho_model.orthotable_dataframe.iloc[
                            index_y,
                            self.ortho_model.orthotable_dataframe.columns.get_loc("Y (pixel)"),
                        ] = value_y

                    if non_empty_rectification is not None:
                        for index_r, value_r in non_empty_rectification:
                            self.ortho_model.orthotable_dataframe.iloc[
                                index_r,
                                self.ortho_model.orthotable_dataframe.columns.get_loc(rectification_key),
                            ] = value_r

                    if non_empty_validation is not None:
                        for index_v, value_v in non_empty_validation:
                            self.ortho_model.orthotable_dataframe.iloc[
                                index_v,
                                self.ortho_model.orthotable_dataframe.columns.get_loc(validation_key),
                            ] = value_v

        self.plot_gcp_points_on_original_image()

    def plot_gcp_points_on_original_image(self):
        """Plot GCP points on the original image viewer."""
        mw = self.main_window

        # Clear existing points
        mw.ortho_original_image.clearPoints()
        mw.ortho_original_image.clearPolygons()

        # Get points to plot
        rectification_points = self.get_points_to_plot(which_points="rectification")
        validation_points = self.get_points_to_plot(which_points="validation")

        # Plot rectification points
        if rectification_points is not None and rectification_points["points"]:
            mw.ortho_original_image.scene.set_current_instruction(
                Instructions.ADD_POINTS_INSTRUCTION,
                points=rectification_points["points"],
                labels=rectification_points["labels"],
            )

    def get_points_to_plot(self, which_points: str = "rectification") -> Optional[Dict[str, List]]:
        """Get GCP points from table for plotting on image.

        Args:
            which_points: Which points to get - "rectification" or "validation"

        Returns:
            Dictionary with "points" and "labels" lists, or None if no points
        """
        mw = self.main_window

        if self.ortho_model.orthotable_dataframe.empty:
            return None

        df = self.ortho_model.orthotable_dataframe

        # Determine which column to check
        if which_points == "rectification":
            use_column = "Use in Rectification"
        elif which_points == "validation":
            use_column = "Use in Validation"
        else:
            return None

        # Check if column exists
        if use_column not in df.columns:
            return None

        # Filter points based on flag using string_to_boolean
        try:
            # Convert string values to boolean (handles "y", "yes", "true", "1", etc.)
            bool_values = df[use_column].fillna("").astype(str).map(
                lambda x: string_to_boolean(x) if x else False
            )
            filtered_df = df[bool_values]
        except (ValueError, TypeError) as e:
            logging.warning(f"Error filtering points: {e}")
            return None

        if filtered_df.empty:
            return None

        # Extract pixel coordinates and labels
        try:
            points = list(zip(
                filtered_df["X (pixel)"].astype(float),
                filtered_df["Y (pixel)"].astype(float)
            ))
            labels = filtered_df["# ID"].astype(str).tolist()
        except KeyError as e:
            logging.warning(f"Missing required column: {e}")
            return None

        return {"points": points, "labels": labels}

    # ==================== Rectification Calculation ====================

    def rectify_single_frame(self):
        """Calculate rectification parameters and rectify the current frame.

        This is the main entry point for single-frame orthorectification.
        It:
        1. Validates GCP table is loaded
        2. Determines rectification method (scale, homography, or camera matrix)
        3. Calculates rectification parameters using OrthorectificationService
        4. Updates model with results
        5. Displays rectified image
        """
        mw = self.main_window

        # Check if GCP table is loaded
        if self.ortho_model.orthotable_dataframe.empty:
            mw.warning_dialog(
                "No GCP Table",
                "Please load a GCP table before rectifying.",
                style="ok",
                icon=mw.__icon_path__ + os.sep + "IVy_logo.ico",
            )
            return

        # Get GCP data from table (always in meters)
        gcp_table = self.ortho_model.orthotable_dataframe.to_dict()
        labels = list(gcp_table["# ID"].values())
        world_coords = tuple(
            zip(
                [float(item) for item in gcp_table["X"].values()],
                [float(item) for item in gcp_table["Y"].values()],
                [float(item) for item in gcp_table["Z"].values()],
            )
        )

        # Get points selected for rectification
        points_dict = self.get_points_to_plot(which_points="rectification")
        if points_dict is None or not points_dict["points"]:
            mw.warning_dialog(
                "No Rectification Points",
                "Please mark at least 2 points as 'Use in Rectification'.",
                style="ok",
                icon=mw.__icon_path__ + os.sep + "IVy_logo.ico",
            )
            return

        # Match selected points to full table
        matched_point_labels = find_matches_between_two_lists(points_dict["labels"], labels)

        pixel_coords = np.array(points_dict["points"])
        world_coords_array = np.array([world_coords[index[0]] for index in matched_point_labels])
        num_points = pixel_coords.shape[0]

        # Get water surface elevation
        water_surface_elev = self.ortho_model.ortho_rectified_wse_m
        if water_surface_elev == 0.0:
            water_surface_elev = 1.0e-5  # Avoid exactly 0.0

        # Save GCP table to project
        try:
            destination_path = os.path.join(
                mw.swap_orthorectification_directory,
                "ground_control_points.csv"
            )
            dict_data = mw.get_table_as_dict(mw.orthoPointsTable)
            pd.DataFrame(dict_data).fillna("").to_csv(destination_path, index=False)
        except Exception as e:
            mw.update_statusbar(f"Failed to save GCP table to project: {e}")

        # Validate GCP configuration
        validation_errors = self.ortho_service.validate_gcp_configuration(
            pixel_coords,
            world_coords_array
        )
        if validation_errors:
            error_msg = "GCP validation errors:\\n" + "\\n".join(validation_errors)
            logging.error(error_msg)
            mw.warning_dialog(
                "Invalid GCP Configuration",
                error_msg,
                style="ok",
                icon=mw.__icon_path__ + os.sep + "IVy_logo.ico"
            )
            return

        # Determine rectification method
        method = self.ortho_service.determine_rectification_method(
            num_points,
            world_coords_array
        )

        logging.debug(
            f"ORTHORECTIFICATION: Found {num_points} points. "
            f"All on same Z-plane? {np.all(world_coords_array[:, -1] == world_coords_array[0, -1])}"
        )
        logging.info(f"Rectification method: {method}")

        # Get original image
        image = mw.ortho_original_image.scene.ndarray()

        # Calculate parameters based on method
        if method == "scale":
            self._calculate_scale_rectification(image, pixel_coords, world_coords_array)
        elif method == "homography":
            self._calculate_homography_rectification(image, pixel_coords, world_coords_array)
        elif method == "camera matrix":
            self._calculate_camera_matrix_rectification(
                image, pixel_coords, world_coords_array, water_surface_elev
            )
        else:
            mw.warning_dialog(
                "Unknown Method",
                f"Unknown rectification method: {method}",
                style="ok"
            )
            return

        # Emit signal
        self.ortho_model.rectification_calculated.emit(method)

        logging.info("Single frame rectification complete")

    def _calculate_scale_rectification(
        self,
        image: np.ndarray,
        pixel_coords: np.ndarray,
        world_coords: np.ndarray
    ):
        """Calculate scale-based rectification parameters (2 GCPs, nadir view).

        Args:
            image: Input image array
            pixel_coords: Pixel coordinates (2 x 2)
            world_coords: World coordinates (2 x 3)
        """
        mw = self.main_window

        # Calculate scale parameters using service
        scale_params = self.ortho_service.calculate_scale_parameters(
            pixel_coords,
            world_coords[:, 0:2],
            image.shape
        )

        # Update model rectification parameters
        mw.rectification_parameters["homography_matrix"] = scale_params["homography_matrix"]
        mw.rectification_parameters["extent"] = scale_params["extent"]
        mw.rectification_parameters["pad_x"] = scale_params["pad_x"]
        mw.rectification_parameters["pad_y"] = scale_params["pad_y"]
        mw.rectification_parameters["pixel_coords"] = pixel_coords
        mw.rectification_parameters["world_coords"] = world_coords

        # Update state
        pixel_gsd = scale_params["pixel_gsd"]
        mw.pixel_ground_scale_distance_m = pixel_gsd
        mw.is_homography_matrix = True
        mw.scene_averaged_pixel_gsd_m = pixel_gsd
        mw.rectification_method = "scale"

        # Calculate quality metrics
        quality_metrics = self.ortho_service.calculate_quality_metrics(
            "scale",
            pixel_gsd,
            pixel_distance=scale_params["pixel_distance"],
            ground_distance=scale_params["ground_distance"]
        )
        mw.rectification_rmse_m = quality_metrics["rectification_rmse_m"]
        mw.reprojection_error_pixels = quality_metrics["reprojection_error_pixels"]

        # For scale method, image not transformed (nadir assumption)
        transformed_image = image

        # Apply flips if needed
        if self.ortho_model.is_ortho_flip_x or self.ortho_model.is_ortho_flip_y:
            from image_velocimetry_tools.image_processing_tools import flip_image_array
            transformed_image = flip_image_array(
                transformed_image,
                flip_x=self.ortho_model.is_ortho_flip_x,
                flip_y=self.ortho_model.is_ortho_flip_y
            )

        # Display rectified image
        mw.ortho_rectified_image.scene.clearImage()
        mw.ortho_rectified_image.scene.setImage(transformed_image)
        mw.ortho_rectified_image.setEnabled(True)

        # Update UI widgets
        mw.load_ndarray_into_qtablewidget(
            mw.rectification_parameters["homography_matrix"],
            mw.tablewidgetProjectiveMatrix,
        )
        mw.lineeditPixelGSD.setText(
            f"{pixel_gsd * units_conversion(mw.display_units)['L']:.3f}"
        )

        logging.info(
            f"Scale rectification complete. Pixel GSD: {pixel_gsd} m/pixel, "
            f"RMSE: {mw.rectification_rmse_m:.4f} m"
        )

    def _calculate_homography_rectification(
        self,
        image: np.ndarray,
        pixel_coords: np.ndarray,
        world_coords: np.ndarray
    ):
        """Calculate homography-based rectification (4+ GCPs on same plane).

        Args:
            image: Input image array
            pixel_coords: Pixel coordinates (N x 2)
            world_coords: World coordinates (N x 3)
        """
        mw = self.main_window

        # Set padding
        pad_x, pad_y = 200, 200
        logging.info(f"ORTHORECTIFICATION: Padding ({pad_x}, {pad_y})")

        # Check for existing homography matrix
        existing_homography = None
        if mw.is_homography_matrix:
            existing_homography = mw.rectification_parameters.get("homography_matrix")

        # Calculate homography parameters using service
        homography_params = self.ortho_service.calculate_homography_parameters(
            image,
            pixel_coords,
            world_coords,
            homography_matrix=existing_homography,
            pad_x=pad_x,
            pad_y=pad_y
        )

        # Get transformed image
        transformed_image = homography_params["transformed_image"]

        # Apply flips if needed
        if self.ortho_model.is_ortho_flip_x or self.ortho_model.is_ortho_flip_y:
            from image_velocimetry_tools.image_processing_tools import flip_image_array
            transformed_image = flip_image_array(
                transformed_image,
                flip_x=self.ortho_model.is_ortho_flip_x,
                flip_y=self.ortho_model.is_ortho_flip_y
            )

        # Update model rectification parameters
        mw.rectification_parameters["homography_matrix"] = homography_params["homography_matrix"]
        mw.rectification_parameters["extent"] = homography_params["extent"]
        mw.rectification_parameters["pad_x"] = homography_params["pad_x"]
        mw.rectification_parameters["pad_y"] = homography_params["pad_y"]
        mw.rectification_parameters["pixel_coords"] = pixel_coords
        mw.rectification_parameters["world_coords"] = world_coords

        # Update state
        pixel_gsd = homography_params["pixel_gsd"]
        mw.pixel_ground_scale_distance_m = pixel_gsd
        mw.is_homography_matrix = True
        mw.scene_averaged_pixel_gsd_m = pixel_gsd
        mw.rectification_method = "homography"

        # Calculate quality metrics
        quality_metrics = self.ortho_service.calculate_quality_metrics(
            "homography",
            pixel_gsd,
            homography_matrix=mw.rectification_parameters["homography_matrix"]
        )
        if "estimated_view_angle" in quality_metrics:
            mw.rectification_parameters["estimated_view_angle"] = quality_metrics["estimated_view_angle"]
            logging.info(f"Estimated view angle: {quality_metrics['estimated_view_angle']:.2f}°")

        mw.rectification_rmse_m = quality_metrics["rectification_rmse_m"]
        mw.reprojection_error_pixels = quality_metrics["reprojection_error_pixels"]

        # Display rectified image
        mw.ortho_rectified_image.scene.clearImage()
        mw.ortho_rectified_image.scene.setImage(transformed_image)
        mw.ortho_rectified_image.setEnabled(True)

        # Update UI widgets
        mw.load_ndarray_into_qtablewidget(
            mw.rectification_parameters["homography_matrix"],
            mw.tablewidgetProjectiveMatrix,
        )
        mw.lineeditPixelGSD.setText(
            f"{pixel_gsd * units_conversion(mw.display_units)['L']:.3f}"
        )

        logging.info(
            f"Homography rectification complete. Pixel GSD: {pixel_gsd} m/pixel, "
            f"RMSE: {mw.rectification_rmse_m:.4f} m"
        )

    def _calculate_camera_matrix_rectification(
        self,
        image: np.ndarray,
        pixel_coords: np.ndarray,
        world_coords: np.ndarray,
        water_surface_elev: float
    ):
        """Calculate camera matrix rectification (6+ GCPs with varying elevations).

        Args:
            image: Input image array
            pixel_coords: Pixel coordinates (N x 2)
            world_coords: World coordinates (N x 3) with varying Z
            water_surface_elev: Water surface elevation in meters
        """
        mw = self.main_window

        # Check for existing camera matrix
        existing_camera_matrix = None
        if mw.is_camera_matrix:
            existing_camera_matrix = mw.rectification_parameters.get("camera_matrix")

        # Calculate camera matrix parameters using service
        camera_params = self.ortho_service.calculate_camera_matrix_parameters(
            image,
            pixel_coords,
            world_coords,
            water_surface_elev,
            camera_matrix=existing_camera_matrix,
            padding_percent=0.03
        )

        # Get transformed image
        transformed_image = camera_params["transformed_image"]

        # Apply flips if needed
        if self.ortho_model.is_ortho_flip_x or self.ortho_model.is_ortho_flip_y:
            from image_velocimetry_tools.image_processing_tools import flip_image_array
            transformed_image = flip_image_array(
                transformed_image,
                flip_x=self.ortho_model.is_ortho_flip_x,
                flip_y=self.ortho_model.is_ortho_flip_y
            )

        # Update model rectification parameters
        mw.rectification_parameters["camera_matrix"] = camera_params["camera_matrix"]
        mw.rectification_parameters["extent"] = camera_params["extent"]
        mw.rectification_parameters["pixel_coords"] = pixel_coords
        mw.rectification_parameters["world_coords"] = world_coords
        mw.rectification_parameters["water_surface_elev"] = water_surface_elev

        # Update state
        pixel_gsd = camera_params["pixel_gsd"]
        mw.pixel_ground_scale_distance_m = pixel_gsd
        mw.is_camera_matrix = True
        mw.scene_averaged_pixel_gsd_m = pixel_gsd
        mw.camera_position = camera_params["camera_position"]
        mw.rectification_method = "camera matrix"

        # Quality metrics
        if camera_params.get("projection_rms_error"):
            mw.rectification_rmse_m = camera_params["projection_rms_error"]

        # Display rectified image
        mw.ortho_rectified_image.scene.clearImage()
        mw.ortho_rectified_image.scene.setImage(transformed_image)
        mw.ortho_rectified_image.setEnabled(True)

        # Update UI widgets
        mw.load_ndarray_into_qtablewidget(
            mw.rectification_parameters["camera_matrix"],
            mw.tablewidgetProjectiveMatrix,
        )
        mw.lineeditPixelGSD.setText(
            f"{pixel_gsd * units_conversion(mw.display_units)['L']:.3f}"
        )

        logging.info(
            f"Camera matrix rectification complete. Pixel GSD: {pixel_gsd} m/pixel, "
            f"Camera position: {camera_params['camera_position']}"
        )
