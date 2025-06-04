"""IVy module for computing discharge
"""

import copy
import logging
import os

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtGui, QtCore
from areacomp.gui.mplcanvas import MplCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from image_velocimetry_tools.discharge_tools import (
    compute_discharge_midsection,
    convert_surface_velocity_rantz,
)
from image_velocimetry_tools.gui.dischargeplot import QPlot
from image_velocimetry_tools.gui.filesystem import TableWidgetDragRows, DataFrameModel
from image_velocimetry_tools.uncertainty import Uncertainty

global icons_path
icon_path = "icons"


class DischargeTab:

    def __init__(self, ivy_framework):
        self.ivy_framework = ivy_framework

        # Configure bold and normal fonts (used for tables)
        self.font = QtGui.QFont()
        self.font.setPointSize(12)
        self.font.setBold(False)
        self.font.setWeight(50)
        self.font_bold = QtGui.QFont()
        # self.font_bold.setPointSize(12)
        self.font_bold.setBold(True)
        self.font_bold.setWeight(75)
        self.font_italic = QtGui.QFont()
        self.font_italic.setPointSize(12)
        self.font_italic.setItalic(True)

        # plot objects
        self.plot_tb = None
        self.plot_canvas = None
        self.plot_fig = None

        # Create dict to store data
        self.discharge = {}
        self.discharge_data_dataframe = None
        self.ivy_uncertainty = None

        # Set up the Discharge Table
        self.dischargeTable = TableWidgetDragRows()
        self.dischargeTable.setDragEnabled(False)
        self.table_init()
        self.table_is_changed = False
        self.is_table_loaded = False
        self.table_cell_colored = False
        self.table_has_headers = False

        self.dischargeTable.setGridStyle(1)
        self.dischargeTable.setCornerButtonEnabled(False)
        self.dischargeTable.setShowGrid(True)
        self.dischargeTable.horizontalHeader().setBackgroundRole(QtGui.QPalette.Window)
        self.dischargeTable.selectionModel().selectionChanged.connect(
            self.table_make_all_white
        )
        self.dischargeTable.itemClicked.connect(self.table_get_item)
        self.dischargeTable.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked)
        self.dischargeTable.cellChanged.connect(self.table_finished_edit)
        self.dischargeTableLineEdit = QtWidgets.QLineEdit()
        self.dischargeTableLineEdit.setToolTip("edit and press ENTER")
        self.dischargeTableLineEdit.setStatusTip("edit and press ENTER")
        self.dischargeTableLineEdit.returnPressed.connect(self.table_update_cell)
        self.ivy_framework.layoutDischargeStationsTable.addWidget(self.dischargeTable)

        self.ivy_framework.pushbuttonUsedInMeasurement.clicked.connect(
            self.station_type_used
        )
        self.ivy_framework.pushbuttonNotUsedInMeasurement.clicked.connect(
            self.station_type_not_used
        )
        self.ivy_framework.pushbuttonResetDischargeStationsTable.clicked.connect(
            self.update_discharge_results
        )
        self.ivy_framework.doublespinboxGlobalAlhpa.editingFinished.connect(
            self.apply_alpha_to_table
        )

    def table_init(self):
        """Executes at startup, sets up the Discharge table."""
        headers = [
            "ID",
            "Status",
            "Station Distance",
            "Width",
            "Depth",
            "Area",
            "Surface Velocity",
            "α (alpha)",
            "Unit Discharge",
        ]
        self.dischargeTable.setColumnCount(len(headers))
        self.table_has_headers = True
        self.dischargeTable.setHorizontalHeaderLabels(headers)
        self.dischargeTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_is_changed = False
        self.dischargeTable.resizeColumnsToContents()
        self.dischargeTable.resizeRowsToContents()
        self.dischargeTable.clearSelection()
        self.is_table_loaded = False

        self.dischargeTable.horizontalHeader().setFont(self.font_bold)
        self.dischargeTable.verticalHeader().hide()

    def table_make_all_white(self):
        if self.table_cell_colored:
            for row in range(self.dischargeTable.rowCount()):
                for column in range(self.dischargeTable.columnCount()):
                    item = self.dischargeTable.item(row, column)
                    if item is not None:
                        item.setForeground(QtCore.Qt.black)
                        item.setBackground(QtGui.QColor("#e1e1e1"))
        self.table_cell_colored = False

    def table_get_item(self):
        item = self.dischargeTable.selectedItems()[0]
        row = self.table_selected_row()
        column = self.table_selected_column()
        if not item == None:
            name = item.text()
        else:
            name = ""
        # self.msg("'" + name + "' on Row " + str(row + 1) + " Column " + str(column + 1))
        self.dischargeTableLineEdit.setText(name)

    def table_finished_edit(self):
        self.table_is_changed = True
        self.ivy_framework.signal_dischargetable_changed.emit(True)
        # self.compute_discharge_results()

    def table_update_cell(self):
        if self.dischargeTable.selectionModel().hasSelection():
            row = self.table_selected_row()
            column = self.table_selected_column()
            newtext = QtWidgets.QTableWidgetItem(self.editLine.text())
            self.dischargeTable.setItem(row, column, newtext)
            # self.compute_discharge_results()

    def table_selected_row(self):
        if self.dischargeTable.selectionModel().hasSelection():
            row = self.dischargeTable.selectionModel().selectedIndexes()[0].row()
            return int(row)

    def table_selected_column(self):
        column = self.dischargeTable.selectionModel().selectedIndexes()[0].column()
        return int(column)

    def table_remove_row(self):
        if self.dischargeTable.rowCount() > 0:
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
                self.dischargeTable.removeRow(row)
                self.table_is_changed = True
                self.ivy_framework.signal_dischargetable_changed.emit(True)
            else:
                pass

    def table_add_row(self):
        if self.dischargeTable.rowCount() > 0:
            if self.dischargeTable.selectionModel().hasSelection():
                row = self.table_selected_row()
                item = QtWidgets.QTableWidgetItem("")
                self.dischargeTable.insertRow(row)
            else:
                row = 0
                item = QtWidgets.QTableWidgetItem("")
                self.dischargeTable.insertRow(row)
                self.dischargeTable.selectRow(0)
        else:
            self.dischargeTable.setRowCount(1)
        if self.dischargeTable.columnCount() == 0:
            self.table_add_column()
            self.dischargeTable.selectRow(0)
        self.table_is_changed = True
        self.ivy_framework.signal_dischargetable_changed.emit(True)

    def table_remove_column(self):
        self.dischargeTable.removeColumn(self.table_selected_column())
        self.table_is_changed = True
        self.ivy_framework.signal_dischargetable_changed.emit(True)

    def table_add_column(self):
        count = self.dischargeTable.columnCount()
        self.dischargeTable.setColumnCount(count + 1)
        self.dischargeTable.resizeColumnsToContents()
        self.table_is_changed = True
        self.ivy_framework.signal_dischargetable_changed.emit(True)
        if self.dischargeTable.rowCount() == 0:
            self.table_add_row()
            self.dischargeTable.selectRow(0)

    def table_clear_list(self):
        self.dischargeTable.clear()
        self.table_is_changed = True
        self.ivy_framework.signal_dischargetable_changed.emit(True)

    def table_load_data(self, new_alpha=None):

        # Get the current discharge table
        csv_file = (
                self.ivy_framework.swap_discharge_directory
                + os.sep
                + "discharge_table.csv"
        )
        discharge_results_df = self.load_discharge_csv_to_dataframe(csv_file)

        if new_alpha is not None:
            for row_idx, alpha_value in new_alpha.items():
                if row_idx < len(discharge_results_df):
                    discharge_results_df.at[row_idx, 'α (alpha)'] = alpha_value

            # Save the modified DataFrame
            discharge_results_df.to_csv(csv_file, index=False)
            self.discharge_data_dataframe = discharge_results_df


        # Clear existing data in dischargeTable
        self.dischargeTable.clearContents()
        self.dischargeTable.setRowCount(0)
        units = self.ivy_framework.survey_units

        # Populate dischargeTable with data, converting to display units as
        # needed

        # Mapping of column names to their unit keys in the `units` dictionary
        unit_column_map = {
            'Station Distance': 'L',
            'Width': 'L',
            'Depth': 'L',
            'Area': 'A',
            'Surface Velocity': 'V',
            'Unit Discharge': 'Q',
        }

        self.dischargeTable.setRowCount(len(discharge_results_df))
        for i, row in enumerate(discharge_results_df.values):
            for j, value in enumerate(row):
                column_name = discharge_results_df.columns[j]
                dtype = discharge_results_df.dtypes[column_name]

                # Check if the column should be converted
                unit_key = unit_column_map.get(column_name)
                if unit_key in units and dtype == "float64":
                    try:
                        # Apply unit conversion
                        value *= units[unit_key]
                    except Exception as e:
                        print(
                            f"Conversion error in column '{column_name}': {e}")

                # Format value based on data type
                if dtype == "float64":
                    formatted_value = f"{value:.3f}"
                elif dtype == "int32":
                    formatted_value = str(int(value))  # Station IDs
                else:
                    formatted_value = str(value)  # Status, ID, etc.

                item = QtWidgets.QTableWidgetItem(formatted_value)
                self.dischargeTable.setItem(i, j, item)

        # Set headers
        headers = discharge_results_df.columns.tolist()
        self.dischargeTable.setHorizontalHeaderLabels(headers)

        # Additional setup as needed
        self.dischargeTable.resizeColumnsToContents()
        self.dischargeTable.resizeRowsToContents()
        self.dischargeTable.clearSelection()
        self.compute_discharge_results()
        self.update_measurement_results_table()
        self.compute_uncertainty()
        self.create_plot()
        self.is_table_loaded = True

    @staticmethod
    def apply_unit_conversion(column, conversion_factor, unit_label):
        return column * conversion_factor, unit_label

    def station_type_used(self):
        self.station_set_type("Used")

    def station_type_not_used(self):
        self.station_set_type("Not Used")

    def station_set_type(self, m_type):
        """Update the  type of the selected stations."""

        idx = set(index.row() for index in self.dischargeTable.selectedIndexes())

        if len(idx) < 1:
            return

        for row in idx:
            # Update table widget
            self.dischargeTable.setItem(row, 1,
                                        QtWidgets.QTableWidgetItem(m_type))
            # Update backend dataframe
            self.discharge_data_dataframe.at[row, 'Status'] = m_type

        self.compute_discharge_results()
        self.update_measurement_results_table()
        self.compute_uncertainty()
        self.create_plot()

    def apply_alpha_to_table(self):
        """Apply alpha coefficient to the discharge table when user changes
        global value.

        Applies the new alpha either to the currently selected or all
        rows using the "Apply only to selected?" checkbox as a filter.

        """
        alpha = self.ivy_framework.doublespinboxGlobalAlhpa.value()
        alpha_column = 7
        unit_q_column = 8
        area_column = 5
        surf_vel_column = 6
        new_alphas = {}


        if self.ivy_framework.checkboxGlobalAlphaApplySelected.isChecked():
            idx = set(index.row() for index in self.dischargeTable.selectedIndexes())

            if len(idx) < 1:
                return
        else:
            idx = set(range(self.dischargeTable.rowCount()))

        for row in idx:
            area = float(self.dischargeTable.item(row, area_column).text())
            surf_vel = float(self.dischargeTable.item(row, surf_vel_column).text())
            self.dischargeTable.setItem(
                row, alpha_column, QtWidgets.QTableWidgetItem(f"{alpha}")
            )
            self.dischargeTable.setItem(
                row,
                unit_q_column,
                QtWidgets.QTableWidgetItem(f"{area * alpha * surf_vel}"),
            )
            new_alphas[row] = alpha  # Store new alpha for this row

        self.table_load_data(new_alpha=new_alphas)


    def compute_uncertainty(self):
        q_dict = copy.deepcopy(self.ivy_framework.discharge_results)
        ortho_info = {
            "rmse_m": self.ivy_framework.rectification_rmse_m,
            "scene_width_m": self.ivy_framework.cross_section_top_width_m

        }
        self.ivy_uncertainty = Uncertainty()
        self.ivy_uncertainty.compute_uncertainty(
            q_dict,
            total_discharge=self.ivy_framework.discharge_summary["total_discharge"],
            ortho_info=ortho_info
        )
        logging.info(
            f"DISCHARGE: Total measurement uncertainty at 95% CI (ISO 748):"
            f" {self.ivy_uncertainty.u_iso['u95_q'] * 100:.2f} %"
        )
        logging.info(
            f"DISCHARGE: Total measurement uncertainty at 95% CI (IVE):"
            f" {self.ivy_uncertainty.u_ive['u95_q'] * 100:.2f} %"
        )

        data = {
            "IVy-ISO (95%)": [self.ivy_uncertainty.u_iso["u95_q"] * 100],
            "IVy-IVE (95%)": [self.ivy_uncertainty.u_ive["u95_q"] * 100],
        }

        df = pd.DataFrame(data)

        # Write into the Measurement Uncertainty Table
        model = DataFrameModel(df)
        self.ivy_framework.measurementUncertaintyTable.setModel(model)
        # self.measurementResultsTable.setFont(self.font)
        # self.measurementResultsTable.horizontalHeader().setFont(self.font_bold)
        self.ivy_framework.measurementUncertaintyTable.resizeColumnsToContents()
        self.ivy_framework.measurementUncertaintyTable.resizeRowsToContents()
        self.ivy_framework.measurementUncertaintyTable.horizontalHeader().setFont(
            self.font_bold
        )
        self.ivy_framework.measurementUncertaintyTable.verticalHeader().hide()

        # Get the ISO Uncertainty (%) value
        iso_uncertainty = data["IVy-ISO (95%)"][0]

        # Determine the index based on the ISO Uncertainty (%)
        if iso_uncertainty < 3:
            index = 1
        elif 3 <= iso_uncertainty < 5:
            index = 2
        elif 5 <= iso_uncertainty < 8:
            index = 3
        else:
            index = 4

        # Set the current index of the combobox
        self.ivy_framework.comboboxUserRating.setCurrentIndex(index)

        # Set the results back to the parent object
        self.ivy_framework.u_iso = self.ivy_uncertainty.u_iso
        self.ivy_framework.u_iso_contribution = self.ivy_uncertainty.u_iso_contribution
        self.ivy_framework.u_ive = self.ivy_uncertainty.u_ive
        self.ivy_framework.u_ive_contribution = self.ivy_uncertainty.u_ive_contribution
        self.ivy_framework.discharge_summary["ISO_uncertainty"] = (
            self.ivy_uncertainty.u_iso["u95_q"]
        )
        self.ivy_framework.discharge_summary["IVE_uncertainty"] = (
            self.ivy_uncertainty.u_ive["u95_q"]
        )

    def compute_discharge_results(self):
        df = self.discharge_data_dataframe  # Work directly with the backend (metric) data

        if not df.empty:
            # Filter to only the rows marked as "Used"
            used_df = df[df["Status"] == "Used"]

            # Extract required columns and compute values
            cumulative_distances = used_df["Station Distance"].astype(
                float).values
            surface_velocities = used_df["Surface Velocity"].astype(
                float).values
            alpha_values = used_df["α (alpha)"].astype(float).values
            average_velocities = convert_surface_velocity_rantz(
                surface_velocities, alpha=alpha_values
            )
            vertical_depths = used_df["Depth"].astype(float).values

            # Compute discharge and area
            total_discharge, total_area = compute_discharge_midsection(
                cumulative_distances, average_velocities, vertical_depths
            )


            # Set the results back to the parent object
            # The discharge_results dict can be turned into a dataframe like this:
            #   df = pd.DataFrame.from_dict(q_dict, orient='index')
            self.ivy_framework.discharge_results = df.to_dict(orient="index")
            self.ivy_framework.discharge_summary = {
                "total_discharge": total_discharge,
                "total_area": total_area,
                "ISO_uncertainty": None,  # iso_uncertainty,
                "IVE_uncertainty": None,  # ive_uncertainty,
            }

            self.update_measurement_results_table()
            self.compute_uncertainty()
            self.create_plot()
        else:
            logging.error(f"update_discharge_results: no discharge data to update")

    def get_station_and_depth(self):
        xs = self.ivy_framework.xs_survey
        xy_pixel = self.ivy_framework.results_grid

        # AC3 will always process SI units, so we have to ensure the wse fed
        # to it is in SI
        wse = self.ivy_framework.rectification_parameters["water_surface_elev"]
        # if self.ivy_framework.display_units == "English":
        #     wse /= self.ivy_framework.survey_units["L"]

        stations, elevations = self.ivy_framework.xs_survey.get_pixel_xs(xy_pixel)

        # AreaComp will always return elevations (stage).
        # Convert elevations to depths
        depths = wse - elevations

        return stations, depths

    def update_discharge_results(self):
        """Called by the refresh results button"""
        # TODO: here is where we ensure there are data to load
        # Requirements
        # 1. In "cross section mode"
        # 2. AC3 cross section loaded
        # 3. Velocity results exist
        is_cross_section = self.ivy_framework.cross_section_line_exists
        is_stiv = self.ivy_framework.stiv_exists
        is_stiv_opt = self.ivy_framework.stiv_opt_exists
        is_openpiv = self.ivy_framework.openpiv_exists
        is_trivia = self.ivy_framework.trivia_exists
        is_velocity = is_stiv or is_stiv_opt or is_openpiv or is_trivia
        can_process_discharge = is_cross_section and is_velocity

        if can_process_discharge:
            # Task 1 - Cross-section station and depth
            # Get the stationing from the AC3 instance
            station, depth = self.get_station_and_depth()
            self.discharge["stations"] = station
            self.discharge["depths"] = depth
            # Task 2 - Surface Velocity
            what_source = "stiv"  # replace with call to check user comboBox
            if what_source == "stiv":
                # TODO: Load from the STIV file here
                D = np.radians(self.ivy_framework.stiv.directions)  # degrees
                U = self.ivy_framework.stiv.magnitudes_mps * np.cos(D)
                V = self.ivy_framework.stiv.magnitudes_mps * np.sin(D)
                if self.ivy_framework.stiv.magnitude_normals_mps is not None and np.any(
                    self.ivy_framework.stiv.magnitude_normals_mps
                ):
                    M = self.ivy_framework.stiv.magnitude_normals_mps
                else:
                    M = np.sqrt(U**2 + V**2)
                sur_vel = M
                sur_vel = np.insert(sur_vel, 0, 0)  # 0 vel at edge
                sur_vel = np.append(sur_vel, 0)  # 0 vel at edge
                self.discharge["surf_vel"] = sur_vel
            if what_source == "stiv_opt":
                D = np.radians(self.ivy_framework.stiv_opt.directions)
                U = self.ivy_framework.stiv_opt.magnitudes_mps * np.cos(D)
                V = self.ivy_framework.stiv_opt.magnitudes_mps * np.sin(D)
                M = np.sqrt(U**2 + V**2)
                sur_vel = M
                sur_vel = np.insert(sur_vel, 0, np.nan)
                sur_vel = np.append(sur_vel, np.nan)
                self.discharge["surf_vel"] = sur_vel
            if what_source == "openpiv":
                pass
            if what_source == "trivia":
                pass

            # Task 3 - populate the results into the stations dataframe
            self.create_discharge_data_df()
            self.table_load_data()

            # Task 4 - Update results and table
            # self.compute_discharge_results()

            # Task 5 - Update the plots
            # self.create_plot()

            self.ivy_framework.set_tab_icon(
                "pushbuttonRefreshDischargeStationsTable",
                "good",
                self.ivy_framework.tabWidget,
            )
            self.ivy_framework.enable_disable_tabs(
                self.ivy_framework.tabWidget, "tabReporting", True
            )

    def update_measurement_results_table(self):
        """Update the measurement results table.

        This function is called to update the Measurement Results table. It
        will pull the relevant items from the ivy results, and if necessary,
        converts units to match the current display units.
        """
        discharge_summary = copy.deepcopy(self.ivy_framework.discharge_summary)
        discharge_results = copy.deepcopy(self.ivy_framework.discharge_results)

        # Extract values from discharge_summary
        total_discharge = discharge_summary["total_discharge"]
        total_area = discharge_summary["total_area"]

        # Extract values from discharge_results
        unit_discharges = [
            float(result["Unit Discharge"]) for result in discharge_results.values()
        ]
        surface_velocities = [
            float(result["Surface Velocity"]) for result in discharge_results.values()
        ]
        alphas = [float(result["α (alpha)"]) for result in discharge_results.values()]

        # Calculate average velocity (Q/A)
        average_velocity = total_discharge / total_area

        # Calculate average alpha
        average_alpha = np.nansum(alphas) / len(alphas)

        # Calculate average surface velocity
        average_surface_velocity = np.nansum(surface_velocities) / len(
            surface_velocities
        )

        # Find max surface velocity
        max_surface_velocity = np.nanmax(surface_velocities)

        # Create DataFrame, convert values to display units here
        units = self.ivy_framework.survey_units
        data = {
            "Total Discharge": [total_discharge * units["Q"]],
            "Total Area": [total_area * units["A"]],
            "Average Velocity (Q/A)": [average_velocity * units["V"]],
            "Average Alpha": [average_alpha],
            "Average Surface Velocity": [average_surface_velocity * units[
                "V"]],
            "Max Surface Velocity": [max_surface_velocity * units["V"]],
        }
        df = pd.DataFrame(data)

        # Write into the Measurement Results Table
        model = DataFrameModel(df)
        self.ivy_framework.measurementResultsTable.setModel(model)
        # self.measurementResultsTable.setFont(self.font)
        # self.measurementResultsTable.horizontalHeader().setFont(self.font_bold)
        self.ivy_framework.measurementResultsTable.resizeColumnsToContents()
        self.ivy_framework.measurementResultsTable.resizeRowsToContents()
        self.ivy_framework.measurementResultsTable.horizontalHeader().setFont(
            self.font_bold
        )
        self.ivy_framework.measurementResultsTable.verticalHeader().hide()

    @staticmethod
    def clear_table_widget(table: QtWidgets.QTableWidget):
        """
        Removes all loaded data from a table widget.

        Parameters:
        -----------
        table : QtWidgets.QTableWidget
            The QTableWidget instance to be cleared.

        Note:
        -----
        This static method removes all rows and columns from the provided QTableWidget,
        effectively clearing all loaded data from the table.
        """
        if isinstance(table, QtWidgets.QTableWidget):
            table.setRowCount(0)
            n_cols = table.columnCount()
            for column in range(n_cols):
                table.removeColumn(column)

    def create_plot(self):
        """Generate cross-section, velocity, discharge plot."""
        units = self.ivy_framework.survey_units
        if self.plot_canvas is None:
            # set the plot layout so the data can be plotted.
            plot_layout = QtWidgets.QVBoxLayout(
                self.ivy_framework.labelDischargePlotsPlaceholder
            )
            plot_layout.setContentsMargins(1, 1, 1, 1)
            self.plot_canvas = MplCanvas(
                self.ivy_framework.labelDischargePlotsPlaceholder
            )
            plot_layout.addWidget(self.plot_canvas)
            self.plot_tb = NavigationToolbar(self.plot_canvas, self.ivy_framework)
            self.plot_tb.hide()

        # get q results
        self.plot_canvas.figure.clear()
        # df = pd.DataFrame.from_dict(self.ivy_tools.discharge_results,
        #                             orient='index')
        # used_df = df[df['Status'] == 'Used']
        # # Convert 'Station Distance' to float if it's not already
        # used_df['Station Distance'] = used_df['Station Distance'].astype(float)
        # total_q = np.nansum(used_df['Unit Discharge'].astype(float))
        df = pd.DataFrame.from_dict(
            self.ivy_framework.discharge_results, orient="index"
        )
        df["Station Distance"] = df["Station Distance"].astype(float)

        # The used_df should be in display units
        used_df = df.loc[df["Status"] == "Used"]
        unit_column_map = {
            'Station Distance': 'L',
            'Width': 'L',
            'Depth': 'L',
            'Area': 'A',
            'Surface Velocity': 'V',
            'Unit Discharge': 'Q',
        }
        # Convert units in-place on used_df
        for col, unit_key in unit_column_map.items():
            if col in used_df.columns:
                used_df[col] = used_df[col].astype(float) * units[unit_key]


        total_q = np.nansum(used_df["Unit Discharge"].astype(float))

        # Make the plot
        self.plot_fig = QPlot(
            canvas=self.plot_canvas, sum_tbl=self.dischargeTable, units="English"
        )
        self.plot_fig.load_data(
            cross_section=self.ivy_framework.xs_survey, discharge_summary=used_df
        )
        self.ivy_framework.discharge_plot_fig = self.plot_fig.create_plot()

        self.plot_canvas.draw()

    def create_discharge_data_df(self):
        """Create the discharge stations dataframe

        This dataframe will always be in metric units
        """
        num_stations = len(self.discharge["stations"])

        if self.is_table_loaded:
            df = pd.DataFrame.from_dict(
                self.ivy_framework.discharge_results, orient="index"
            )
            # Check if there are any rows with 'Unit Discharge' as "nan"
            if len(df[df["Unit Discharge"] == "nan"]) > 0:
                first_row_status = df["Status"].iloc[0]
                last_row_status = df["Status"].iloc[-1]

                # Mark 'Status' as 'Not Used' where 'Unit Discharge' is "nan"
                df["Status"].where(
                    df["Unit Discharge"] != "nan", other="Not Used", inplace=True
                )

                # Restore the first and last row status
                df["Status"].iloc[0] = first_row_status
                df["Status"].iloc[-1] = last_row_status

            used_df = df[df["Status"] == "Used"]
            used_index = df.index[df.index.isin(used_df.index)].to_numpy()
        else:
            used_index = np.arange(0, num_stations)

        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate data container
        data = {
            "ID": np.arange(0, num_stations),
            "Status": np.where(
                np.isin(np.arange(num_stations), used_index), "Used", "Not Used"
            ),
            "Station Distance": self.discharge["stations"],
            "Width": np.zeros(self.discharge["stations"].shape),
            # Monotonically increasing positive floats
            "Depth": self.discharge["depths"],
            "Area": np.zeros(self.discharge["stations"].shape),
            "Surface Velocity": self.discharge["surf_vel"],
            "α (alpha)": 0.85,
            "Unit Discharge": np.zeros(self.discharge["stations"].shape),
        }

        # compute widths and areas
        for value in range(1, len(data["Width"]) - 1):
            data["Width"][value] = (
                data["Station Distance"][value + 1]
                - data["Station Distance"][value - 1]
            ) / 2
            data["Area"][value] = (data["Width"][value]) * data["Depth"][value]

        data["Width"][0] = (
            data["Station Distance"][1] - data["Station " "Distance"][0]
        ) / 2
        data["Area"][0] = (data["Width"][0] / 2) * data["Depth"][0]
        data["Width"][-1] = (
            data["Station Distance"][-1] - data["Station Distance"][-2]
        ) / 2
        data["Area"][-1] = (data["Width"][-1] / 2) * data["Depth"][-1]
        surface_velocity = np.nan_to_num(data["Surface Velocity"], nan=0)
        data["Unit Discharge"] = data["Area"] * surface_velocity * data["α (alpha)"]

        # Create DataFrame
        self.discharge_data_dataframe = pd.DataFrame(data)

        # Reorder columns to match the desired order
        self.discharge_data_dataframe = self.discharge_data_dataframe[
            [
                "ID",
                "Status",
                "Station Distance",
                "Width",
                "Depth",
                "Area",
                "Surface Velocity",
                "α (alpha)",
                "Unit Discharge",
            ]
        ]

        # compute unit discharge
        self.discharge_data_dataframe["Unit Discharge"] = (
            self.discharge_data_dataframe["Area"]
            * self.discharge_data_dataframe["α (alpha)"]
            * self.discharge_data_dataframe["Surface Velocity"]
        )

        # Write this table to a CSV file
        csv_file = (
            self.ivy_framework.swap_discharge_directory + os.sep + "discharge_table.csv"
        )
        self.discharge_data_dataframe.to_csv(csv_file, index=False)

        return self.discharge_data_dataframe

    @staticmethod
    def load_discharge_csv_to_dataframe(filename):
        """
        Load a CSV file into a dataframe with specified data types.

        Parameters:
        filename (str): The name of the CSV file.

        Returns:
        pd.DataFrame: The loaded dataframe with the correct data types.
        """
        dtype_dict = {
            "ID": "int32",
            "Status": "object",
            "Station Distance": "float64",
            "Width": "float64",
            "Depth": "float64",
            "Area": "float64",
            "Surface Velocity": "float64",
            "α (alpha)": "float64",
            "Unit Discharge": "float64",
        }

        df = pd.read_csv(filename, dtype=dtype_dict)
        return df


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """Intializes the figure canvas and fig attribute.

        Parameters
        ----------
        parent: Object
            Parent of object class.
        width: float
            Width of figure in inches.
        height: float
            Height of figure in inches.
        dpi: float
            Screen resolution in dots per inch used to scale figure.
        """

        # Initialize figure
        self.figure = Figure(figsize=(width, height), tight_layout=True, dpi=dpi)

        # Configure FigureCanvas
        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(
            self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        FigureCanvas.updateGeometry(self)
