"""IVy module that manages the STIV and STIV Review Tabs"""

import logging
import os

import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QCheckBox,
    QFrame,
)
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QTableWidgetItem, QTableWidget

from image_velocimetry_tools.common_functions import (
    units_conversion,
    load_csv_with_numpy,
    component_in_direction,
    geographic_to_arithmetic,
)
from image_velocimetry_tools.graphics import AnnotationView, Instructions
from image_velocimetry_tools.stiv import (
    two_dimensional_stiv_exhaustive,
    two_dimensional_stiv_optimized,
)
from image_velocimetry_tools.services.stiv_service import STIVService

global icons_path
icon_path = "icons"


class STIVTab:
    """Main class for the STIV Tab"""

    def __init__(self, ivy_framework):
        """Class init

        Args:
            ivy_framework (IVyTools object): the main IVyTools object
        """
        self.ivy_framework = ivy_framework
        self.image_path = ""
        self.zoom_factor = 1
        self.image = None
        self.original_image = None
        self.imageBrowser = AnnotationView()
        self.grid = None
        self.image_glob = None
        self.image_stack = None
        self.num_pixels = 20
        self.phi_origin = 90
        self.d_phi = 1
        self.phi_range = 90
        self.pixel_gsd = None
        self.d_t = None
        self.map_file_path = ""
        self.magnitudes_mps = None
        self.directions = None
        self.magnitude_normals_mps = None
        self.sti_array = None
        self.thetas = None
        self.max_velocity_threshold_mps = 10
        self.tolerance = 0.5
        self.d_rho = 0.5
        self.d_theta = 0.5

    def zoom_image(self, zoom_value):
        """Zoom in and zoom out."""
        self.zoom_factor = zoom_value
        self.imageBrowser.zoomEvent(self.imagebrowser_zoom_factor)
        # self.toolbuttonZoomIn.setEnabled(self.imagebrowser_zoom_factor < 4.0)
        # self.toolbuttonZoomOut.setEnabled(self.imagebrowser_zoom_factor > 0.333)

    def normal_size(self):
        """View image with its normal dimensions."""
        self.imageBrowser.clearZoom()
        self.zoom_factor = 1.0

    def process_stiv_exhaustive(self, progress_callback):
        """Main process call that runs the STIV exhaustive computation

        Args:
            progress_callback (pyqtsignal): the callback for the signal

        Returns:
            tuple: the STIV magnitudes_mps and directions
        """
        magnitudes_mps, directions, stis, thetas = two_dimensional_stiv_exhaustive(
            x_origin=self.grid[:, 0].astype(float),
            y_origin=self.grid[:, 1].astype(float),
            image_stack=self.image_stack,
            num_pixels=self.num_pixels,
            phi_origin=self.phi_origin,
            d_phi=self.d_phi,
            phi_range=self.phi_range,
            pixel_gsd=self.pixel_gsd,
            d_t=self.d_t,
            sigma=self.ivy_framework.stiv_gaussian_blur_sigma,
            max_vel_threshold=self.max_velocity_threshold_mps,
            # map_file_path=self.map_file_path,
            progress_signal=progress_callback,
        )

        self.magnitudes_mps = magnitudes_mps
        self.magnitude_normals_mps = magnitudes_mps  # Copy them out
        self.directions = directions  # geo
        self.sti_array = stis
        self.thetas = thetas
        return magnitudes_mps, directions

    def process_stiv_optimized(self, progress_callback):
        """Main process call that runs the STIV optimized computation

        Args:
            progress_callback (pyqtsignal): the callback for the signal

        Returns:
            tuple: the STIV magnitudes_mps and directions
        """
        magnitudes, directions = two_dimensional_stiv_optimized(
            x_origin=self.grid[:, 0].astype(float),
            y_origin=self.grid[:, 1].astype(float),
            image_stack=self.image_stack,
            num_pixels=self.num_pixels,
            phi_origin=self.phi_origin,
            pixel_gsd=self.pixel_gsd,
            d_t=self.d_t,
            d_rho=self.d_rho,
            d_theta=self.d_theta,
            max_vel_threshold=self.max_velocity_threshold_mps,
            map_file_path=self.map_file_path,
            progress_signal=progress_callback,
        )
        self.magnitudes_mps = magnitudes
        self.directions = directions
        return magnitudes, directions


class EditableTableWidgetItem(QTableWidgetItem):
    """Overloaded instance of a QTableWidgetItem to allow editing

    Args:
        QTableWidgetItem (QTebleWidgetItem): the Item
    """

    def __init__(self, text=""):
        """Class init

        Args:
            text (str, optional): supplied string to insert as a default entry. Defaults to "".
        """
        super().__init__(text)

    def flags(self):
        """Get the flags for the selected row and column

        Returns:
            flag: the flag
        """
        if self.column() == 5:  # Adjust the column index as needed
            return super().flags() | QtCore.Qt.ItemIsEditable
        else:
            return super().flags()


class STIReviewTab:
    """Main class for managing the STI Review Tab"""

    def __init__(self, ivy_framework):
        """Class init

        Args:
            ivy_framework (IVyTools object): the IVyTools main object
        """
        self.ivy_framework = ivy_framework
        self.stiv_service = STIVService()
        self.original_magnitudes_mps = None
        self.original_directions = None
        self.sti_paths = None
        self.manual_sti_lines = []
        self.manual_average_directions = []
        self.display_units = self.ivy_framework.display_units
        self.survey_units = units_conversion(units_id=self.display_units)

        # Configure fonts
        self.font = QtGui.QFont()
        self.font.setPointSize(12)
        self.font.setBold(False)
        self.font.setWeight(50)
        self.font_bold = QtGui.QFont()
        self.font_bold.setBold(True)
        self.font_bold.setWeight(75)
        self.font_italic = QtGui.QFont()
        self.font_italic.setPointSize(12)
        self.font_italic.setItalic(True)

        # Set up the Velocities Table
        self.Table = QTableWidget()
        self.Table.setDragEnabled(False)
        self.table_init()
        self.table_is_changed = False
        self.is_table_loaded = False
        self.table_cell_colored = False
        self.table_has_headers = False

        # Configure table properties
        self.Table.setGridStyle(1)
        self.Table.setCornerButtonEnabled(False)
        self.Table.setShowGrid(True)
        self.Table.horizontalHeader().setBackgroundRole(QtGui.QPalette.Window)

        # Connect signals
        self.Table.selectionModel().selectionChanged.connect(self.table_make_all_white)
        self.Table.itemClicked.connect(self.table_get_item)
        self.Table.cellChanged.connect(self.table_finished_edit)
        self.TableLineEdit = QtWidgets.QLineEdit()
        self.TableLineEdit.setToolTip("edit and press ENTER")
        self.TableLineEdit.setStatusTip("edit and press ENTER")
        self.TableLineEdit.returnPressed.connect(self.table_update_cell)
        self.ivy_framework.layoutStiReviewTable.addWidget(self.Table)
        self.ivy_framework.pushbuttonApplyManualSTIVChanges.clicked.connect(
            self.update_discharge_results_in_tab
        )
        self.ivy_framework.pushbuttonReloadProcessedSTIVResults.clicked.connect(
            self.reset_discharge_results
        )

        # Connect double-click event to method
        self.Table.itemDoubleClicked.connect(self.on_table_item_double_clicked)

        # Disable editing for all cells except double-click editing
        self.Table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
            | QtWidgets.QAbstractItemView.DoubleClicked
        )

    def on_table_item_double_clicked(self, item):
        """Executes when the user double-clicks a row in the table

        Args:
            item (QItem): the clicked QItem
        """
        row = item.row()
        column = item.column()
        value = item.text()

        # If user double-clicked in the Manual tab, allow editing of the STI
        # streak angle.
        if column == 5:  # Manual Velocity Col
            if self.ivy_framework.stiv.thetas is not None:
                dialog = ImageDialog(
                    sti_image_path=self.sti_paths[row],
                    theta=self.ivy_framework.stiv.thetas[row],
                    parent=None,
                )
            else:
                dialog = ImageDialog(sti_image_path=self.sti_paths[row], parent=None)
            if dialog.exec_() == QDialog.Accepted:
                average_direction = dialog.average_direction
                # Delegate to service for velocity calculation
                gsd = self.ivy_framework.pixel_ground_scale_distance_m
                dt = self.ivy_framework.extraction_timestep_ms / 1000
                manual_velocity_mps = self.stiv_service.compute_velocity_from_manual_angle(
                    average_direction, gsd, dt, dialog.is_upstream
                )

                if not np.isnan(manual_velocity_mps):
                    logging.debug(f"Average direction: {average_direction} degrees")

                # Set manual velocity for the current row in the Table
                new_item = QtWidgets.QTableWidgetItem(
                    f"{manual_velocity_mps * self.survey_units['V']:.2f}"
                )
                self.Table.setItem(row, column, QtWidgets.QTableWidgetItem(new_item))

                # Draw the manual line on the image
                sti_image_path = self.sti_paths[row]

                if self.ivy_framework.stiv.thetas is not None:
                    theta_deg = self.ivy_framework.stiv.thetas[row]
                else:
                    theta_deg = np.nan

                sti_pixmap = self.draw_manual_lines_on_image(
                    sti_image_path, theta_deg, average_direction
                )

                sti_label = QtWidgets.QLabel()
                sti_label.setPixmap(sti_pixmap)

                # Save the manual line end points into a nested list
                # dialog.lines  # contains the line objects
                for line in dialog.lines:
                    self.manual_sti_lines[row].append(
                        [
                            [line.draft_line.x1(), line.draft_line.y1()],
                            [line.draft_line.x2(), line.draft_line.y2()],
                        ]
                    )
                # For the current row, save the average direction
                # Check for flow upstream flow indication.
                # If manual velocity direction, use it, otherwise use the
                # STIV output direction.
                # If user canceled or did not save a manual line,
                # average_direction will be np.nan
                self.manual_average_directions[row] = average_direction

                self.Table.setCellWidget(row, 1, sti_label)
                self.ivy_framework.is_manual_sti_corrections = True
                logging.debug(
                    f"Manual velocity:"
                    f" {manual_velocity_mps * self.survey_units['V']:.2f} "
                    f"{self.survey_units['label_V']}"
                )

        # If user is trying to save a comment, handle that edit
        elif column == 6:  # Comments
            current_text = item.text()

            if len(current_text) > 240:
                # Truncate the text to 120 characters
                truncated_text = current_text[:240]
                logging.warning(
                    f"Comment truncated to 240 characters. " f"Original: {current_text}"
                )

                # Update the table with the truncated text
                self.Table.setItem(
                    row, column, QtWidgets.QTableWidgetItem(truncated_text)
                )

                # Optionally, notify the user (e.g., via a tooltip or dialog)
                QtWidgets.QMessageBox.warning(
                    self.Table,
                    "Comment Too Long",
                    "The comment exceeded 240 characters and has been " "truncated.",
                )
            else:
                logging.debug(f"User added a comment: {current_text}")

        else:  # Do nothing
            pass

    @staticmethod
    def draw_manual_lines_on_image(sti_image_path, theta_deg, average_direction):
        """
        Draws manual lines on the image based on provided angles.

        Parameters
        ----------
        sti_image_path : str
            The file path to the STI image.
        theta_deg : float
            The angle (in degrees) for the yellow line. If NaN, the yellow line is not drawn.
        average_direction : float
            The average direction angle (in degrees) for the red line.

        Returns
        -------
        QPixmap
            The QPixmap object with the drawn lines.
        """
        orig_sti_pixmap = QPixmap(sti_image_path)

        if not np.isnan(theta_deg):
            sti_pixmap = draw_line_on_pixmap(
                orig_sti_pixmap, theta_deg, color=Qt.yellow
            )
        else:
            sti_pixmap = orig_sti_pixmap

        # Ensure line convention is correct for the STI drawing
        # In essence, invert the angle relative to 180°
        average_direction = 90 + (90 - average_direction)

        sti_pixmap = draw_line_on_pixmap(sti_pixmap, average_direction, color=Qt.red)

        return sti_pixmap

    def compute_sti_velocity(self, theta, gsd, dt):
        """Calculate velocity by bringing in the pixel size and frame
        interval See equation 16 of Fujita et al. (2007).

        Delegates to STIVService.
        """
        return self.stiv_service.compute_sti_velocity(theta, gsd, dt)

    def compute_sti_angle(self, velocity, gsd, dt):
        """Calculate angle in degrees of a STI image given the velocity,
        pixel size, and frame interval. See equation 16 of Fujita et al.
        (2007).

        Delegates to STIVService.
        """
        return self.stiv_service.compute_sti_angle(velocity, gsd, dt)

    def table_init(self):
        """Executes at startup, sets up the table."""
        units = self.display_units
        headers = [
            "ID",
            "STI",
            f"Velocity Direction (deg)",
            f"STI Streak Angle (deg)",
            f"Original Velocity Magnitude " f"{self.survey_units['label_V']}",
            f"Manual Velocity Magnitude " f"{self.survey_units['label_V']}",
            f"Comments",
        ]
        self.Table.setColumnCount(len(headers))
        self.table_has_headers = True
        self.Table.setHorizontalHeaderLabels(headers)
        self.Table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_is_changed = False
        self.Table.resizeColumnsToContents()
        self.Table.setColumnWidth(6, 120)
        self.Table.resizeRowsToContents()
        self.Table.clearSelection()
        self.is_table_loaded = False

        self.Table.horizontalHeader().setFont(self.font_bold)
        self.Table.verticalHeader().hide()

    def table_make_all_white(self):
        """Make the table background color white"""
        if self.table_cell_colored:
            for row in range(self.Table.rowCount()):
                for column in range(self.Table.columnCount()):
                    item = self.Table.item(row, column)
                    if item is not None:
                        item.setForeground(QtCore.Qt.black)
                        item.setBackground(QtGui.QColor("#e1e1e1"))
        self.table_cell_colored = False

    def table_get_item(self):
        """Get the selected Table item"""
        item = self.Table.selectedItems()[0]
        row = self.table_selected_row()
        column = self.table_selected_column()
        if not item == None:
            name = item.text()
        else:
            name = ""
        # self.msg("'" + name + "' on Row " + str(row + 1) + " Column " + str(column + 1))
        self.TableLineEdit.setText(name)

    def table_finished_edit(self):
        """Executes when table editing has completed"""
        self.table_is_changed = True
        self.ivy_framework.signal_dischargetable_changed.emit(True)
        # self.compute_discharge_results()

    def table_update_cell(self):
        """Executes when the table is being edited"""
        if self.Table.selectionModel().hasSelection():
            row = self.table_selected_row()
            column = self.table_selected_column()
            newtext = QtWidgets.QTableWidgetItem(self.editLine.text())
            self.Table.setItem(row, column, newtext)
            self.Table.resizeColumnsToContents()
            self.Table.setColumnWidth(6, 120)
            self.Table.resizeRowsToContents()
            # self.compute_discharge_results()

    def table_selected_row(self):
        """Return the currently selected row

        Returns:
            int: the current table row
        """
        if self.Table.selectionModel().hasSelection():
            row = self.Table.selectionModel().selectedIndexes()[0].row()
            return int(row)

    def table_selected_column(self):
        """Return the currently selected column

        Returns:
            int: the current table column
        """
        column = self.Table.selectionModel().selectedIndexes()[0].column()
        return int(column)

    def table_remove_row(self):
        """Remove the selected row(s)"""
        if self.Table.rowCount() > 0:
            remove = QtWidgets.QMessageBox()
            remove.setText(
                "This will remove the selected row, and cannot be undone. Are you sure?"
            )
            remove.setStandardButtons(
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel
            )
            remove = remove.exec()

            if remove == QtWidgets.QMessageBox.Yes:
                row = self.table_selected_row()
                self.Table.removeRow(row)
                self.table_is_changed = True
                self.ivy_framework.signal_dischargetable_changed.emit(True)
            else:
                pass

    def table_add_row(self):
        """Add a new row to the bottom of the table"""
        if self.Table.rowCount() > 0:
            if self.Table.selectionModel().hasSelection():
                row = self.table_selected_row()
                item = QtWidgets.QTableWidgetItem("")
                self.Table.insertRow(row)
            else:
                row = 0
                item = QtWidgets.QTableWidgetItem("")
                self.Table.insertRow(row)
                self.Table.selectRow(0)
        else:
            self.Table.setRowCount(1)
        if self.Table.columnCount() == 0:
            self.table_add_column()
            self.Table.selectRow(0)
        self.table_is_changed = True
        self.ivy_framework.signal_dischargetable_changed.emit(True)

    def table_remove_column(self):
        """Remove the selected table column"""
        self.Table.removeColumn(self.table_selected_column())
        self.table_is_changed = True
        self.ivy_framework.signal_dischargetable_changed.emit(True)

    def table_add_column(self):
        """Add a new column to the end of the table"""
        count = self.Table.columnCount()
        self.Table.setColumnCount(count + 1)
        self.Table.resizeColumnsToContents()
        self.Table.setColumnWidth(6, 120)
        self.table_is_changed = True
        self.ivy_framework.signal_dischargetable_changed.emit(True)
        if self.Table.rowCount() == 0:
            self.table_add_row()
            self.Table.selectRow(0)

    def table_clear_list(self):
        """Clear all selections in the table"""
        self.Table.clear()
        self.table_is_changed = True
        self.ivy_framework.signal_dischargetable_changed.emit(True)

    def table_load_data(self, sti_images=None):
        """Load new data into the STI table

        Args:
            sti_images (glob, optional): A glob of file paths to the STI images. Defaults to None.

        """
        # Clear existing data in dischargeTable
        self.Table.clearContents()

        # Load the STIV Results from CSV
        csv_file_path = (
            f"{self.ivy_framework.swap_velocities_directory}"
            f"{os.sep}"
            f"stiv_results.csv"
        )
        if os.path.isfile(csv_file_path):
            headers, data = load_csv_with_numpy(csv_file_path)
            try:
                magnitudes_mps = data[:, 4].astype(float)
                scalar_projections_mps = data[:, 5].astype(float)
                directions = data[:, 6].astype(float)
            except:
                pass

        if self.ivy_framework.stiv.thetas is not None:
            thetas = self.ivy_framework.stiv.thetas
        else:
            thetas = np.empty_like(magnitudes_mps)
            thetas[:] = np.nan

        # Ensure magnitudes_mps and directions have the same length
        if len(magnitudes_mps) != len(directions):
            raise ValueError(
                "Length of magnitudes_mps and directions arrays must be the same."
            )

        # Save the original data in the class object
        self.original_magnitudes_mps = magnitudes_mps
        self.original_directions = directions
        self.sti_paths = sti_images

        # Set the number of rows in the table
        num_rows = len(magnitudes_mps)
        self.Table.setRowCount(num_rows)

        # Populate the table with data
        for row, (magnitudes_mps, direction, theta, sti_image_path) in enumerate(
            zip(magnitudes_mps, directions, thetas, sti_images)
        ):
            # Assuming the data is in the order: ID, STI, Velocity
            # Direction, STI Angle, Original Velocity Magnitude, Manual
            # Velocity Magnitude, (Comment not part of this loop)
            item = QtWidgets.QTableWidgetItem(str(row + 1))
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)  # Make non-editable
            self.Table.setItem(row, 0, QtWidgets.QTableWidgetItem(item))  # ID
            item = QtWidgets.QTableWidgetItem(f"{direction:.1f}")
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)  # Make non-editable
            self.Table.setItem(row, 2, QtWidgets.QTableWidgetItem(item))  # Vel Dir
            item = QtWidgets.QTableWidgetItem(f"{theta:.1f}")
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)  # Make non-editable
            self.Table.setItem(row, 3, QtWidgets.QTableWidgetItem(item))  # STI Angle
            item = QtWidgets.QTableWidgetItem(
                f'{magnitudes_mps * self.survey_units["V"]:.2f}'
            )
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)  # Make non-editable
            self.Table.setItem(row, 4, QtWidgets.QTableWidgetItem(item))  # Orig Vel
            item = QtWidgets.QTableWidgetItem(
                f'{magnitudes_mps * self.survey_units["V"]:.2f}'
            )
            # item.setFlags(
            #     item.flags() & QtCore.Qt.ItemIsEditable)  # Make editable
            self.Table.setItem(row, 5, item)  # Manual Vel

            # Load the STI image and display it in the table
            orig_sti_pixmap = QPixmap(sti_image_path)

            if self.ivy_framework.stiv.thetas is not None:
                theta_deg = self.ivy_framework.stiv.thetas[row]
            else:
                theta_deg = np.nan
            sti_pixmap = draw_line_on_pixmap(
                orig_sti_pixmap, theta_deg, color=Qt.yellow
            )
            sti_label = QtWidgets.QLabel()
            sti_label.setPixmap(sti_pixmap)
            self.Table.setCellWidget(row, 1, sti_label)

            # While we are looping through each row, go ahead and create a
            # spot to hold any manual lines generated later
            self.manual_sti_lines.append([])
            self.manual_average_directions.append([])

        # Resize to fit the content
        self.Table.resizeColumnsToContents()
        self.Table.setColumnWidth(6, 120)
        self.Table.resizeRowsToContents()

    def extract_manual_velocity_data(self):
        """Extract any manual velocity changes or edits

        Returns:
            ndarray: the extracted manual velocity data
        """
        num_rows = self.Table.rowCount()
        manual_vel_index = 5

        # Iterate over each row and extract the data from the last column
        data = []
        for row in range(num_rows):
            item = self.Table.item(row, manual_vel_index)
            if item is not None:
                data.append(float(item.text()))  # Assuming the data is numeric
            else:
                # Handle empty cells or non-numeric data appropriately
                data.append(np.nan)  # For example, you can add NaN for empty cells

        # Convert the data list into a NumPy array
        data_array = np.array(data)

        # This array has to be returned in Metric
        if self.display_units == "English":
            c = 1 / self.survey_units["V"]
        else:
            c = 1
        data_array *= c

        return data_array

    def update_discharge_results_in_tab(self):
        """Update the discharge results using the current table contents.

        Delegates to STIVService for applying manual corrections.
        """
        # Find where manual changes are applied, if nothing changed,
        # do nothing else
        idx = [index for index, element in enumerate(self.manual_sti_lines) if element]
        if idx:
            # Load the original results using service
            csv_file_path = (
                f"{self.ivy_framework.swap_velocities_directory}"
                f"{os.sep}"
                f"stiv_results.csv"
            )
            stiv_data = self.stiv_service.load_stiv_results_from_csv(csv_file_path)

            # Extract manual velocities from the table
            manual_velocities = self.extract_manual_velocity_data()

            # Apply manual corrections using the service
            result = self.stiv_service.apply_manual_corrections(
                stiv_data=stiv_data,
                manual_velocities=manual_velocities,
                manual_indices=idx,
                tagline_direction=stiv_data['Tagline_Direction'][0]
            )

            # Send updated scalar magnitudes
            result_dict = {
                "idx": idx,
                "manual_velocity": result['scalar_projections'],
                "normal_direction_geo": stiv_data['Normal_Direction']
            }
            self.ivy_framework.stiv.magnitude_normals_mps = result['scalar_projections']
            self.ivy_framework.signal_manual_vectors.emit(result_dict)

            # Update the discharge results tab
            self.ivy_framework.dischargecomputaton.update_discharge_results()

    def reset_discharge_results(self):
        """Reset the STI table removing all edits"""
        self.table_load_data(sti_images=self.sti_paths)
        # Update the discharge results
        # self.ivy_tools.dischargecomputaton.update_discharge_results()
        self.ivy_framework.dischargecomputaton.table_load_data(reset_table=True)
        self.update_discharge_results_in_tab()


class ImageDialog(QDialog):
    """Main class for the Manual STI angle dialog

    Args:
        QDialog (QDialog): the Dialog
    """

    def __init__(self, sti_image_path, theta=np.nan, parent=None):
        """Class init

        Args:
            sti_image_path (str): path to the current STI image
            theta (ndarray, optional): the STIV algorithm produced streak angle (theta). Defaults to np.nan.
            parent (IVyTools object, optional): the main IVyTools object. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle(
            "STI Image Manual Angle: Click Add Line to "
            "manually draw STI angles. All lines will be "
            "averaged."
        )
        self.setFixedSize(600, 800)
        self.image = AnnotationView()
        self.image.scene.load_image(sti_image_path)
        self.lines = []
        self.directions = []
        self.theta = theta
        self.average_direction = None
        self.is_upstream = False

        # Vertical layout for dialog
        layout = QVBoxLayout(self)

        # Grid layout for image
        image_layout = QGridLayout()
        sti_label = QLabel()
        orig_sti_pixmap = QPixmap(sti_image_path)
        sti_pixmap = draw_line_on_pixmap(orig_sti_pixmap, self.theta, color=Qt.yellow)
        sti_label.setPixmap(sti_pixmap)
        self.image.scene.setImage(sti_pixmap)
        image_layout.addWidget(self.image, 0, 0)

        # Add image layout to vertical layout
        layout.addLayout(image_layout)

        # Horizontal layout for upstream flow checkbox
        special_conditions_layout = QHBoxLayout()
        self.upstream_flow_checkbox = QCheckBox("Mark node as upstream flow?")
        self.upstream_flow_checkbox.setChecked(False)
        special_conditions_layout.addWidget(self.upstream_flow_checkbox)
        self.upstream_flow_checkbox.clicked.connect(self.upstream_flow)

        # Horizontal layout for buttons
        button_layout = QHBoxLayout()

        # Add Line button
        add_line_button = QPushButton("Add Line")
        button_layout.addWidget(add_line_button)
        add_line_button.clicked.connect(self.add_line)

        # Vertical separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        button_layout.addWidget(separator)

        # OK button
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.okay)
        button_layout.addWidget(ok_button)

        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        # Adjust size policy to make the Add Line button expandable
        add_line_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Add stretch to push the buttons to the right side
        button_layout.addStretch()

        # Adjust size policy to make the OK and Cancel buttons fixed-size
        ok_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        cancel_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Add button layout to vertical layout
        layout.addLayout(special_conditions_layout)
        layout.addLayout(button_layout)

    def add_line(self):
        """Add a manual STI streak angle line when user clicks the Add Line button"""
        self.image.scene.set_current_instruction(Instructions.SIMPLE_LINE_INSTRUCTION)
        self.lines.append(self.image.scene.line_item[-1])
        self.image.scene.line_item[-1].setPen(QtGui.QPen(QtGui.QColor("yellow"), 1.5))

    def upstream_flow(self):
        """Set the velocity upstream or downstream according to the checkbox"""
        if self.upstream_flow_checkbox.isChecked():
            self.is_upstream = True
        else:
            self.is_upstream = False

    def okay(self):
        """Executes if user clicks the Okay button"""
        self.directions = []  # Ensure it's reset before adding values

        # Get line directions
        for line in self.lines:
            # Get the angle in degrees from the line's attribute
            # The angle will be in the range of values from 0.0 up to but
            # not including 360.0. The angles are measured counter-clockwise
            # from a point on the x-axis to the right of the origin (x > 0).
            angle_deg = line.line_angle_rad

            self.directions.append(angle_deg)

        # Compute the average direction
        # Using the unit vector here ensures that averaging for lines drawn
        # near 0 and/or 360 degrees are correctly averaged
        if self.directions:
            angles_rad = np.radians(self.directions)  # Convert to radians
            mean_x = np.mean(np.cos(angles_rad))  # Average cosine
            mean_y = np.mean(np.sin(angles_rad))  # Average sine

            # Compute the mean angle using atan2
            self.average_direction = np.degrees(np.arctan2(mean_y, mean_x))

            # Ensure the angle is in the range [0, 360]
            if self.average_direction < 0:
                self.average_direction += 360
        else:
            self.average_direction = self.theta  # Default if no lines are drawn

        # Close the dialog
        logging.debug(f"MANUAL STI: Avg. Line angle: {self.average_direction:.2f}")
        self.accept()


def draw_line_on_pixmap(original_pixmap, angle_degrees=np.nan, color=Qt.yellow):
    """Render a streak angle line on the supplied STI image.

    Args:
        original_pixmap (QPixmap): The STI image as a pixmap.
        angle_degrees (float, optional): Streak angle (theta) in degrees. Defaults to np.nan.
        color (QColor, optional): The color to render the line. Defaults to Qt.yellow.

    Returns:
        QPixmap: The new pixmap with the streak line rendered.
    """
    # If theta is NaN, just return the original pixmap
    if np.isnan(angle_degrees):
        return original_pixmap

    # Create a copy of the original QPixmap to modify
    modified_pixmap = original_pixmap.copy()

    # Determine the aspect ratio scaling factor
    width = modified_pixmap.width()
    height = modified_pixmap.height()
    scale_x = width / max(width, height)  # Normalize scaling based on the largest dimension
    scale_y = height / max(width, height)

    # Create a QPainter and set it to paint on the modified QPixmap
    painter = QPainter()
    painter.begin(modified_pixmap)

    # Set the pen color and style for drawing the line
    pen = QPen(QColor(color))
    pen.setWidth(2)  # Change the line width as needed
    painter.setPen(pen)

    # Calculate the center point of the pixmap
    center_x = width / 2
    center_y = height / 2

    # Define a reasonable line length
    line_length = min(width, height) / 2

    # Convert angle to radians
    angle_radians = np.deg2rad(angle_degrees)

    # Compute end points with correct scaling
    dx = line_length * np.cos(angle_radians) * scale_x
    dy = line_length * np.sin(angle_radians) * scale_y

    end_x = center_x + dx
    end_y = center_y + dy
    start_x = center_x - dx
    start_y = center_y - dy

    # Draw the line on the modified QPixmap
    # Convert numpy types to Python floats for PyQt5 compatibility
    painter.drawLine(float(start_x), float(start_y), float(end_x), float(end_y))

    # End painting
    painter.end()

    return modified_pixmap