# Class adapted from main AreaComp3 gui class.
import logging
import os
import shutil

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtGui
from areacomp.gui.areasurvey import AreaSurvey
from areacomp.gui.loaddata import LoadData

# from areacomp.gui.loadqrev import LoadQrevUi
from areacomp.gui.loadsvmaq import ReadSvmaq
from areacomp.gui.manningninput import ManningNInput
from areacomp.gui.mplcanvas import MplCanvas
from areacomp.gui.plots.plotbathymetry import BathymetryPlot
from areacomp.gui.plots.plotcharratings import RatingChar
from areacomp.gui.projectdata import ProjectData
from areacomp.gui.selectcircuitstouse import SelectCircuits
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

from image_velocimetry_tools.common_functions import units_conversion
from image_velocimetry_tools.services.cross_section_service import CrossSectionService


# ToDo Add table to the gui to display grid line data in the bathymetry tab
# ToDo Add ability with the gui to update stage for grid line data.


class CrossSectionGeometry:
    """Main class for managing the Cross-section Geometry tab"""

    def __init__(self, parent):
        """Class init

        Args:
            parent (IVyTools object): the main IVyTools object
        """
        self.parent = parent
        self.xs_service = CrossSectionService()

        # Connect action for loading AreaComp3 files.
        self.parent.actionImport_Bathymetry.triggered.connect(self.load_areacomp)

        # Connect signals from the IVy parent app
        self.parent.signal_wse_changed.connect(self.ivy_wse_changed)

        # Dynamic Tab location for xs_survey
        self.ivy_xs_tab_index = 3  # Index location for tabCrossSectionGeometry

        # signals for sub surveys
        self.parent.stage_lineEdit.editingFinished.connect(self.update_stage_offset)
        self.parent.start_station_lineEdit.editingFinished.connect(
            self.update_starting_station
        )
        self.parent.cb_file_name.currentIndexChanged.connect(self.update_survey_offsets)
        self.parent.pb_subsurvey_delete.clicked.connect(self.remove_sub_survey)
        self.parent.pb_flip.clicked.connect(self.change_survey_start)
        self.parent.pb_xs_subsurvey_import.clicked.connect(self.load_sub_survey)

        self.parent.tableCrossSectionCharacteristics.cellDoubleClicked.connect(
            self.edit_manning_n
        )

        # bathymetry table
        self.parent.add_pushButton.clicked.connect(self.manually_add_stations)
        self.parent.delete_pushButton.clicked.connect(self.delete_station)

        # signal for computation stage change
        self.parent.char_stage_sb.editingFinished.connect(self.update_char_stage)

        # subsection signals
        self.parent.add_section_pb.clicked.connect(self.add_subsection)
        self.parent.rm_sect_pb.clicked.connect(self.remove_subsection)

        # object for gui for loading data
        self.new_window = None

        # create plot objects
        self.bathy_plot_canvas = None
        self.bathy_plot_fig = None
        self.bathy_plot_tb = None
        self.char_plot_fig = None
        self.char_plot_tb = None

        # plot signals
        self.parent.pb_bathymetry_home.clicked.connect(self.bathy_plot_home)
        self.parent.pb_bathymetry_zoom.clicked.connect(self.bathy_plot_zoom)
        self.parent.pb_bathymetry_pan.clicked.connect(self.bathy_plot_pan)
        self.parent.pb_bathymetry_picker.clicked.connect(self.bathy_plot_data_cursor)

        self.parent.bathy_rb.clicked.connect(self.create_bathy_plot)
        self.parent.area_rb.clicked.connect(self.check_bathy_rb)
        self.parent.radius_rb.clicked.connect(self.check_bathy_rb)
        self.parent.perm_rb.clicked.connect(self.check_bathy_rb)
        self.parent.conveyance_rb.clicked.connect(self.check_bathy_rb)
        self.parent.topwidtch_rb.clicked.connect(self.check_bathy_rb)

        # tab click signals
        self.parent.tabWidget.currentChanged.connect(self.update_ui)
        self.parent.tabWidget_2.currentChanged.connect(self.update_ui)

        # initiate AreaComp3 backend
        self.xs_survey = AreaSurvey()

        # dataframe of grid line data from manual image clicks.

        self.change = {
            "bathymetry": False,
            "plot": False,
            "sub-survey": False,
            "char": False,
        }

        # Hide the Measuresurement Stage
        self.parent.char_stage_sb.hide()
        self.parent.labelCrossSectionMeasurementStage.hide()

    def load_areacomp(self, fname=None):
        """Load cross-section data from AreaComp matlab file."""

        # Todo Add display of Stationing line with 0 depths () from Image
        #  manual creation (click and drop) and allow editing
        if fname is None or not fname:
            fname = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, "Select File", filter="AreaComp (*.mat);; All ()"
            )[0]

        if fname == "":
            return

        # Set the fname into the parent so it can be included in the save file
        self.parent.bathymetry_ac3_filename = fname

        # Attempt to write the AC3 file to the swap_discharge_directory so that
        # it is included in the project file
        # Save a copy of the GCP image as '!calibration_image.jpg'
        try:
            destination_path = os.path.join(
                self.parent.swap_discharge_directory, "cross_section_ac3.mat"
            )
            shutil.copy(fname, destination_path)
        except Exception as e:
            self.parent.update_statusbar(f"Failed to save calibration image: {e}")

        # try to load the matlab file.
        try:
            self.xs_survey.load_areacomp(fname, units=self.parent.display_units)
            self.update_backend()
            self.update_subsurvey_cb()

            # If we got here, we have loaded bathy data, go ahead and enable
            # the Cross-Section Geometry Tab
            self.parent.set_qwidget_state_by_name("tabCrossSectionGeometry", True)

            # Also, go ahead and set a xs_line top width
            top_width = float(self.xs_survey.channel_char.Top_Width[0])
            if self.parent.display_units == "English":
                top_width *= 0.3048
            self.parent.cross_section_top_width_m = top_width

            # Attempt to set the WSE based on the IVy WSE
            self.xs_survey.stage = self.parent.ortho_rectified_wse_m * 3.281
            self.xs_survey.max_stage = self.parent.ortho_rectified_wse_m * 3.281
            self.update_backend()

            # Assign channel char out to the parent
            self.parent.channel_char = self.xs_survey.channel_char

            self.parent.is_area_comp_loaded = True
            self.parent.toolBox_bathymetry.setEnabled(True)

        except BaseException as e:
            # Trigger message if file fails to load. This could result is
            # the file was not an AreaComp3 file.

            # Log the actual error for debugging
            logging.error(f"Failed to load AreaComp file: {fname}", exc_info=True)

            msg = QtWidgets.QMessageBox()
            msg.setInformativeText(
                f"File failed to load. \nPlease verify the file was an AreaComp file.\n\n"
                f"Error details: {type(e).__name__}: {str(e)}"
            )
            msg.setWindowTitle("Error")
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.exec_()

            return

    def update_subsurvey_cb(self):
        """Clear and add sub surveys to combobox."""

        survey_list = []
        for sur in self.xs_survey.survey_info:
            survey_list.append(sur[0])

        self.parent.cb_file_name.clear()
        self.parent.cb_file_name.addItems(survey_list)

    # sub-survey controls
    def load_sub_survey(self):
        """Load sub survey data."""

        # Hard setting units to English for now. Will revisit later.
        select = LoadData()
        select.exec()

        try:
            if select.file_type == "csv":
                if len(select.fname) < 1:
                    return
                fname = select.fname
                meas_type = select.meas_type
                start_edge = select.start_edge

                self.xs_survey.load_csv_survey(
                    path=fname,
                    m_type=meas_type,
                    start_edge=start_edge,
                    units=select.units,
                )

            elif select.file_type == "svmaq":
                fname = select.fname
                if len(fname) < 1:
                    return
                svmaq_file = ReadSvmaq(fname)

                choose_circuits = SelectCircuits(svmaq_file.circuit_dict)
                choose_circuits.exec()

                if len(choose_circuits.final_circuits) > 0:
                    self.xs_survey.load_svmaq_survey(
                        fname, choose_circuits.final_circuits, units=select.units
                    )

            elif select.file_type == "sontek" or select.file_type == "trdi":
                fname = select.fname
                if len(fname) < 1:
                    return

                self.xs_survey.load_adcp_survey(
                    path=fname,
                    adcp_type=select.file_type,
                    start_edge=select.start_edge,
                    units=select.units,
                )

            # Todo Add QRev import once dependencies are straightened out.
            #  Currently pip hangs up on the installed due to conflicting
            #  versions.
            # elif select.file_type == "qrev":
            #     fname = select.fname
            #     if len(fname) < 1:
            #         return
            #
            #     self.new_window = LoadQrevUi(caller=self)
            #     self.new_window.load_file(fname)
            #     self.hide()
            #     self.new_window.show()

            self.update_backend()
            self.update_subsection_sb()

            self.update_subsurvey_cb()

        except BaseException:
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Failed to load file.")
            msg.setWindowTitle("Error")
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.exec_()

            return

    def load_qrev_data(self):
        """Method from AC3 to load QRev data. Hard coding units to English
        for now."""

        if self.new_window.meas.adcp_data is not None:
            self.xs_survey.load_adcp_survey(
                path=None,
                adcp_type="qrev",
                data=self.new_window.meas.adcp_data,
                units="English",
            )
            # self.populate_metadata()
            # self.update_ui()
            self.parent.show()
            self.new_window = None

    def update_stage_offset(self):
        """Updates stage offset for subsurvey user enters into the lineedit."""
        if self.xs_survey.survey is None:
            return  # Nothing loaded
        idx = self.parent.cb_file_name.currentIndex()

        stage = self.parent.stage_lineEdit.text()
        old_stage = self.xs_survey.survey_info[idx][5]
        try:
            stage = float(stage)

            if stage != old_stage:
                self.xs_survey.update_stage_offset(idx, stage)
                self.update_backend()

        except ValueError:
            if stage != "":
                msg = QtWidgets.QMessageBox()
                msg.setInformativeText(
                    "Stage is not a number. Please enter " "a valid stage"
                )
                msg.setWindowTitle("Warning")
                msg.setIcon(QtWidgets.QMessageBox.Warning)
                msg.exec_()

    def update_starting_station(self):
        """Updates location of the starting station based on what the
        user enters into the line edit."""
        if self.xs_survey.survey is None:
            return  # Nothing loaded
        idx = self.parent.cb_file_name.currentIndex()
        start_station = self.parent.start_station_lineEdit.text()
        old_station = self.xs_survey.survey_info[idx][4]
        try:
            start_station = float(start_station)
            if start_station != old_station:
                self.xs_survey.apply_start_station(idx, start_station)

                self.update_backend()
                self.update_subsection_sb()

        except ValueError:
            if start_station != "":
                msg = QtWidgets.QMessageBox()
                msg.setInformativeText(
                    "Station is not a number. Please enter " "a valid station"
                )
                msg.setWindowTitle("Warning")
                msg.setIcon(QtWidgets.QMessageBox.Warning)
                msg.exec_()

    def update_survey_offsets(self):
        """Updates survey offsets based on what the user entered into the
        line edits"""

        idx = self.parent.cb_file_name.currentIndex()

        stage = self.xs_survey.survey_info[idx][5]
        self.parent.stage_lineEdit.setText(str(stage))

        station = self.xs_survey.survey_info[idx][4]
        self.parent.start_station_lineEdit.setText(str(station))

        self.update_backend()

        self.update_subsection_sb()

    def change_survey_start(self):
        """Flip selected survey start edge."""

        idx = self.parent.cb_file_name.currentIndex()

        try:
            self.xs_survey.flip_subsurvey_start(idx)
            self.update_backend()
            self.update_subsection_sb()

        except BaseException:
            pass

    def update_file_info_table(self):
        """Creates station stage table. Method adapted from AC3."""

        tbl = self.parent.tbl_survey_info
        tbl.clear()
        headers = [
            "File Name",
            "Survey Type",
            "Start Station",
            "End Station",
            "Station Offset",
            "Stage Offset",
        ]

        bold = QtGui.QFont()
        bold.setBold(True)
        tbl.setColumnCount(6)

        tbl.setHorizontalHeaderLabels(headers)
        tbl.horizontalHeader().setFont(bold)
        tbl.horizontalHeader().setVisible(True)
        tbl.verticalHeader().setVisible(False)

        if self.xs_survey.survey_info is None:
            return

        row_count = len(self.xs_survey.survey_info)
        tbl.setRowCount(row_count)

        # convert data frame to array then populate table
        if row_count > 0:
            for row in range(row_count):
                # populate file name
                tbl.setItem(
                    row,
                    0,
                    QtWidgets.QTableWidgetItem(str(self.xs_survey.survey_info[row][0])),
                )

                # populate type
                tbl.setItem(
                    row,
                    1,
                    QtWidgets.QTableWidgetItem(str(self.xs_survey.survey_info[row][1])),
                )

                # populate start station
                tbl.setItem(
                    row,
                    2,
                    QtWidgets.QTableWidgetItem(
                        "{:.2f}".format(self.xs_survey.survey_info[row][2])
                    ),
                )

                # populate end station
                tbl.setItem(
                    row,
                    3,
                    QtWidgets.QTableWidgetItem(
                        "{:.2f}".format(self.xs_survey.survey_info[row][3])
                    ),
                )

                # populate station offset
                tbl.setItem(
                    row,
                    4,
                    QtWidgets.QTableWidgetItem(
                        "{:.2f}".format(self.xs_survey.survey_info[row][4])
                    ),
                )

                # populate stage offset
                tbl.setItem(
                    row,
                    5,
                    QtWidgets.QTableWidgetItem(
                        "{:.2f}".format(self.xs_survey.survey_info[row][5])
                    ),
                )

        tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        tbl.resizeColumnsToContents()
        tbl.resizeRowsToContents()

    def remove_sub_survey(self):
        """Remove subsurvey then update gui."""

        survey = self.parent.cb_file_name.currentText()
        if survey:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Question)
            msg.setWindowTitle("Delete SubSurvey")
            msg.setText("Are you sure you want to remove " + survey + "?")
            msg.setStandardButtons(
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel
            )
            msg = msg.exec()

            if msg == QtWidgets.QMessageBox.Yes:
                self.xs_survey.remove_survey(survey)
                self.update_backend()

                self.update_subsection_sb()

                self.update_subsurvey_cb()

    def update_bathy_tbl(self):
        """Populate bathymetry table with cross-section data."""

        tbl = self.parent.tableCrossSectionBathymetry
        tbl.clear()
        headers = ["Station", "Stage"]
        msg = []

        bold = QtGui.QFont()
        bold.setBold(True)
        tbl.setColumnCount(2)

        tbl.setHorizontalHeaderLabels(headers)
        tbl.horizontalHeader().setFont(bold)
        tbl.horizontalHeader().setVisible(True)
        tbl.verticalHeader().setVisible(False)

        if self.xs_survey.survey is None:
            return

        tbl.setRowCount(self.xs_survey.survey.shape[0])
        duplicates = self.xs_survey.check_for_duplicate_stations(self.xs_survey.survey)
        dups = duplicates.index.tolist()

        # convert data frame to array then populate table
        if self.xs_survey.survey.shape[0] > 0:
            for row in range(self.xs_survey.survey.shape[0]):
                # populate stations
                station = self.xs_survey.survey.iloc[row]["Stations"]
                tbl.setItem(
                    row, 0, QtWidgets.QTableWidgetItem("{:.3f}".format(station))
                )
                if dups:
                    if row in dups:
                        tbl.item(row, 0).setBackground(QtGui.QColor(255, 255, 51))
                        # tbl.item(row, 0).setFont(self.parent.font_bold)
                        msg.append(str(station))

                # populate stage values
                tbl.setItem(
                    row,
                    1,
                    QtWidgets.QTableWidgetItem(
                        "{:.3f}".format(
                            self.xs_survey.survey.iloc[row]["AdjustedStage"]
                        )
                    ),
                )

        tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        tbl.resizeColumnsToContents()
        tbl.resizeRowsToContents()

    def update_channel_char_tbl(self):
        """Populate channel characteristics table with cross-section data."""

        tbl = self.parent.tableCrossSectionCharacteristics
        tbl.clear()
        headers = [
            "Section",
            "Start",
            "End",
            "Area",
            "% Total Area",
            "Wetted Perimeter",
            "% Total Wetted Perimeter",
            "Top Width",
            "Hydraulic Radius",
            "Mannings n",
            "Conveyance",
        ]

        bold = QtGui.QFont()
        bold.setBold(True)

        tbl.setRowCount(self.xs_survey.channel_char.shape[0])
        tbl.setColumnCount(len(headers))

        tbl.setHorizontalHeaderLabels(headers)
        tbl.horizontalHeader().setFont(bold)
        tbl.horizontalHeader().setVisible(True)
        tbl.verticalHeader().setVisible(False)

        e_cols = ["Mannings n"]
        for col in range(len(headers)):
            item = tbl.horizontalHeaderItem(col)
            if item.text() in e_cols:
                tbl.horizontalHeaderItem(col).setForeground(QtGui.QColor(0, 0, 255))
                tbl.horizontalHeaderItem(col).setToolTip("Doubleclick cells to edit.")

        # populate table if data is present
        if self.xs_survey.channel_char.shape[0] > 0:
            for row in range(self.xs_survey.channel_char.shape[0]):
                tbl.setItem(
                    row,
                    0,
                    QtWidgets.QTableWidgetItem(
                        str(self.xs_survey.channel_char.iloc[row]["Section"])
                    ),
                )

                tbl.setItem(
                    row,
                    1,
                    QtWidgets.QTableWidgetItem(
                        str(
                            "{:.2f}".format(
                                self.xs_survey.channel_char.iloc[row]["Start"]
                            )
                        )
                    ),
                )

                tbl.setItem(
                    row,
                    2,
                    QtWidgets.QTableWidgetItem(
                        str(
                            "{:.2f}".format(
                                self.xs_survey.channel_char.iloc[row]["End"]
                            )
                        )
                    ),
                )

                tbl.setItem(
                    row,
                    3,
                    QtWidgets.QTableWidgetItem(
                        "{:.2f}".format(self.xs_survey.channel_char.iloc[row]["Area"])
                    ),
                )

                tbl.setItem(
                    row,
                    4,
                    QtWidgets.QTableWidgetItem(
                        "{:.2f}".format(
                            self.xs_survey.channel_char.iloc[row]["Percent_Area"]
                        )
                    ),
                )

                tbl.setItem(
                    row,
                    5,
                    QtWidgets.QTableWidgetItem(
                        "{:.2f}".format(
                            self.xs_survey.channel_char.iloc[row]["Wetted_Perimeter"]
                        )
                    ),
                )

                tbl.setItem(
                    row,
                    6,
                    QtWidgets.QTableWidgetItem(
                        "{:.2f}".format(
                            self.xs_survey.channel_char.iloc[row]["Percent_WP"]
                        )
                    ),
                )

                tbl.setItem(
                    row,
                    7,
                    QtWidgets.QTableWidgetItem(
                        "{:.2f}".format(
                            self.xs_survey.channel_char.iloc[row]["Top_Width"]
                        )
                    ),
                )

                tbl.setItem(
                    row,
                    8,
                    QtWidgets.QTableWidgetItem(
                        "{:.2f}".format(
                            self.xs_survey.channel_char.iloc[row]["Hydraulic_Radius"]
                        )
                    ),
                )

                if (
                    np.isnan(self.xs_survey.channel_char.iloc[row]["ManningsN"])
                    or row == 0
                ):
                    value = ""
                else:
                    value = "{:.3f}".format(
                        self.xs_survey.channel_char.iloc[row]["ManningsN"]
                    )

                tbl.setItem(row, 9, QtWidgets.QTableWidgetItem(value))

                if np.isnan(self.xs_survey.channel_char.iloc[row]["k_factor"]):
                    value = ""
                else:
                    value = "{:.3f}".format(
                        self.xs_survey.channel_char.iloc[row]["k_factor"]
                    )

                tbl.setItem(row, 10, QtWidgets.QTableWidgetItem(value))

        tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        tbl.resizeColumnsToContents()
        tbl.resizeRowsToContents()

    def edit_manning_n(self, row, col):
        """Calls dialog to allow user to set the manning n for the
        cell that was clicked.

        Parameters:
        ----------
            row: int
            col: int

        """

        if row != 0 and col == 9:
            dlg = ManningNInput(self.xs_survey.n_values[row - 1])
            dlg.exec_()

            if dlg.n_value is not None:
                self.xs_survey.modify_nvalue(row - 1, dlg.n_value)

                # update computations
                self.xs_survey.compute_channel_char(self.xs_survey.stage)
                self.xs_survey.sub_section_chars()

                # update plot and table
                self.update_channel_char_tbl()
                if self.parent.conveyance_rb.isChecked():
                    self.check_bathy_rb()

    def create_bathy_plot(self):
        """Create bathymetry plot."""

        # If the canvas has not been previously created, create the canvas,
        # toolbar, and add the widgets.
        if self.bathy_plot_canvas is None:
            # set the plot layout so the data can be plotted.
            plot_layout = QtWidgets.QVBoxLayout(self.parent.label)
            plot_layout.setContentsMargins(1, 1, 1, 1)
            self.bathy_plot_canvas = MplCanvas(self.parent.label)
            plot_layout.addWidget(self.bathy_plot_canvas)
            self.bathy_plot_tb = NavigationToolbar(self.bathy_plot_canvas, self.parent)
            self.bathy_plot_tb.hide()

        # Initiate the plot
        self.bathy_plot_canvas.figure.clear()
        self.bathy_plot_fig = BathymetryPlot(canvas=self.bathy_plot_canvas)
        if self.xs_survey.survey is None:
            return  # Nothing loaded
        self.bathy_plot_fig.plot_bathymetry(
            self.xs_survey,
            True,
            True,
            False,
            stage=self.xs_survey.stage,
            subsection=self.xs_survey.subsections,
            units=self.parent.display_units,
        )

        self.bathy_plot_fig.set_survey_tbl(self.parent.tableCrossSectionBathymetry)

        # Draw canvas
        self.bathy_plot_canvas.draw()
        return self.bathy_plot_fig

    def create_characteristics_plot(self, plot_option):
        """Create the channel characteristics plot."""
        # If the canvas has not been previously created, create the canvas,
        # toolbar, and add the widgets.
        if self.bathy_plot_canvas is None:
            # set the plot layout so the data can be plotted.
            plot_layout = QtWidgets.QVBoxLayout(self.parent.label)
            plot_layout.setContentsMargins(1, 1, 1, 1)
            self.bathy_plot_canvas = MplCanvas(self.parent.label)
            plot_layout.addWidget(self.bathy_plot_canvas)
            self.bathy_plot_tb = NavigationToolbar(self.bathy_plot_canvas, self.parent)
            self.bathy_plot_tb.hide()

        self.bathy_plot_canvas.figure.clear()
        self.char_plot_fig = RatingChar(canvas=self.bathy_plot_canvas)
        if not self.xs_survey.chars:
            return  # Nothing is loaded
        self.char_plot_fig.plot_rating(
            self.xs_survey.chars, plot_option, units=self.parent.display_units
        )

        # Draw canvas
        self.bathy_plot_canvas.draw()

    def bathy_plot_clear_zphd(self):
        """Clears the zoom, pan, home, and data cursor"""

        if self.bathy_plot_fig is not None:
            self.parent.pb_bathymetry_picker.setChecked(False)
            self.bathy_plot_fig.set_hover_connection(False)
            if self.parent.pb_bathymetry_pan.isChecked():
                self.bathy_plot_tb.pan()
                self.parent.pb_bathymetry_pan.setChecked(False)
            if self.parent.pb_bathymetry_zoom.isChecked():
                self.bathy_plot_tb.zoom()
                self.parent.pb_bathymetry_zoom.setChecked(False)

    def bathy_plot_home(self):
        """Resets the view of the visible figure"""

        if self.bathy_plot_fig is not None:
            self.bathy_plot_clear_zphd()
            self.bathy_plot_tb.home()

    def bathy_plot_pan(self):
        """Enables paning for the visible figure"""

        if self.bathy_plot_fig is not None:
            self.bathy_plot_tb.pan()
            self.parent.pb_bathymetry_zoom.setChecked(False)
            self.parent.pb_bathymetry_picker.setChecked(False)
            self.bathy_plot_data_cursor()

    def bathy_plot_zoom(self):
        """Enables the zoom for the visible figure"""

        if self.bathy_plot_fig is not None:
            self.bathy_plot_tb.zoom()
            self.parent.pb_bathymetry_pan.setChecked(False)
            self.parent.pb_bathymetry_picker.setChecked(False)
            self.bathy_plot_data_cursor()

    def bathy_plot_data_cursor(self):
        """Triggers the data cursor for the plot in view."""

        if self.bathy_plot_fig is not None:
            if self.parent.pb_bathymetry_picker.isChecked():
                if self.parent.pb_bathymetry_pan.isChecked():
                    self.bathy_plot_tb.pan()
                    self.parent.pb_bathymetry_pan.setChecked(False)
                if self.parent.pb_bathymetry_zoom.isChecked():
                    self.bathy_plot_tb.zoom()
                    self.parent.pb_bathymetry_zoom.setChecked(False)
                self.parent.pb_bathymetry_picker.setChecked(True)
                self.bathy_plot_fig.set_hover_connection(True)
            else:
                self.bathy_plot_fig.set_hover_connection(False)

    def check_bathy_rb(self):
        """Check which radio button is checked."""

        if self.parent.bathy_rb.isChecked():
            self.create_bathy_plot()
        elif self.parent.area_rb.isChecked():
            self.create_characteristics_plot("area")
        elif self.parent.radius_rb.isChecked():
            self.create_characteristics_plot("hr")
        elif self.parent.perm_rb.isChecked():
            self.create_characteristics_plot("wp")
        elif self.parent.conveyance_rb.isChecked():
            self.create_characteristics_plot("k")
        elif self.parent.topwidtch_rb.isChecked():
            self.create_characteristics_plot("tw")

    def manually_add_stations(self):
        """Adds station stage values to survey"""

        try:
            station = float(self.parent.station_lineEdit.text())
            stage = float(self.parent.stage_lineEdit.text())
        except BaseException:
            return

        self.xs_survey.create_manual_survey(station=station, stage=stage)
        self.update_subsection_sb()
        self.update_backend()

    def delete_station(self):
        """Deletes station from the survey based on the selected row in the
        station stage table."""

        ss_tbl = self.parent.tableCrossSectionBathymetry

        # idx = ss_tbl.selectionModel().currentIndex().row()
        idx = set(index.row() for index in ss_tbl.selectedIndexes())

        if len(idx) < 1:
            return

        self.xs_survey.remove_data(idx)
        self.update_subsection_sb()
        self.update_backend()

    def get_pixel_xs(self, points):
        """Use X/Y grid line points to create conversion for cross-section
        from real world coordinates.

        This method pulls from the current AC3  xs_survey object,
        so whatever units the survey is in is what gets computed initially.
        To ensure the backend is always in Metric, this method checks the
        current display units and converts as needed.

        Parameters:
            points: nd.Array
                x/y values

        Returns:
            stations: nd.Array
                pixel points converted to distance
            elevations: nd.Array
                interpolated elevations for points.
        """
        # IVy display units
        units = self.parent.display_units

        # get pixel distance
        point_extents = self.parent.rectified_xs_image.lines_ndarray().reshape(2, 2)

        # combine passes points with grid extends then project
        new_arr = np.insert(point_extents, 1, points, axis=0)
        df = pd.DataFrame(new_arr, columns=["x", "y"])
        proj = ProjectData()
        proj.compute_data(df, rtn=True)

        # compute conversion factor
        pixel_dist = self.pixel_2_distance(point_extents)

        # compute the wetted width
        # xs_width = np.max(self.xs_survey.survey["Stations"].to_numpy())
        # xs_width = self.xs_survey.channel_char.Top_Width[0]  # Always meters
        crossings = self.find_station_for_adj_stage(
            self.xs_survey.survey["Stations"],
            self.xs_survey.survey["AdjustedStage"],
            self.xs_survey.stage,
            mode="firstlast",
        )
        wetted_width = crossings[-1] - crossings[0]

        p_conversion = self.pixel_to_rw_conversion(pixel_dist, wetted_width)

        # convert stations to distance, corrected for station start
        pixel_stations = proj.stations * p_conversion + crossings[0]

        # ensure p_df has LEW start bank
        if self.parent.cross_section_start_bank == "right":
            pixel_stations = (
                np.nanmax(pixel_stations)
                - pixel_stations
                - (0 - np.nanmin(pixel_stations))
            )

        # interpolate elevations for pixel stations
        # Station/Adj Stage follow xs_survey.survey.units
        elevations = self.xs_service.interpolate_elevations(
            self.xs_survey.survey["Stations"].to_numpy(),
            self.xs_survey.survey["AdjustedStage"].to_numpy(),
            pixel_stations,
        )

        # Ensure results are in Metric
        if units == "English":
            c = 1 / units_conversion(units)["L"]
            pixel_stations *= c
            elevations *= c

        return pixel_stations, elevations

    def ivy_wse_changed(self, value):
        """Executes if the water surface elevation edit box on the Orthorectification tab is changed

        Args:
            value (float): the new WSE value
        """
        print(f"Received WSE elevation: {value} meters")

        # IVy display units
        units = self.parent.display_units
        c = units_conversion(units)["L"]

        wse_ft = value * c

        try:
            self.xs_survey.max_stage = wse_ft
            self.xs_survey.stage = wse_ft
            self.update_backend()
            self.update_subsection_sb()
            self.update_subsurvey_cb()
            self.update_ui()
            crossings = self.find_station_for_adj_stage(
                self.xs_survey.survey["Stations"],
                self.xs_survey.survey["AdjustedStage"],
                self.xs_survey.stage,
                mode="firstlast",
            )
            logging.debug(
                f"WSE Changed: Recompute wetted width. Found the "
                f"following crossings: {crossings}"
            )
        except:
            pass

    def pixel_2_distance(self, points):
        """Compute distance between x/y points.

        Delegates to CrossSectionService.

        Parameters:
            points: nd.array
        """
        return self.xs_service.compute_pixel_distance(points)

    def pixel_to_rw_conversion(self, pixel_distance, width):
        """Compute conversion for real world distance to pixel distance.

        Delegates to CrossSectionService.
        """
        return self.xs_service.compute_pixel_to_real_world_conversion(
            pixel_distance, width
        )

    def find_station_for_adj_stage(
        self, stations, adjusted_stages, target_adj_stage, mode="all", epsilon=1e-1
    ):
        """
        Calculate the station values for a specified Adjusted Stage using linear interpolation.

        Delegates to CrossSectionService.

        Parameters
        ----------
        stations : list or numpy.ndarray
            A list or array of station values.
        adjusted_stages : list or numpy.ndarray
            A list or array of adjusted stage values corresponding to the station values.
        target_adj_stage : float
            The target adjusted stage value for which to find the corresponding station values.
        mode : str, optional
            Specifies whether to return all crossings ('all') or just the first and last crossing ('firstLast').
            Default is 'all'.
        epsilon : float, optional
            A small tolerance value to handle numerical precision issues. Default is 1e-1.

        Returns
        -------
        list of float
            A list of station values where the adjusted stage intersects the target adjusted stage.
        """
        return self.xs_service.find_station_crossings(
            stations, adjusted_stages, target_adj_stage, mode=mode, epsilon=epsilon
        )

    def update_char_stage(self):
        """Updates stage used for Channel Characteristics"""

        self.xs_survey.stage = self.parent.char_stage_sb.value()
        self.update_backend()

    # Subsection signals
    def update_subsection_sb(self):
        """Update max and min extremes for spin box."""

        self.parent.subsection_sb.setMaximum(self.xs_survey.subsections[-1])
        self.parent.subsection_sb.setMinimum(self.xs_survey.subsections[0])

    def remove_subsection(self):
        """Added subsection at station in spin box."""

        idx = self.parent.subsect_tbl.selectionModel().currentIndex().row()
        col = self.parent.subsect_tbl.selectionModel().currentIndex().column()
        try:
            item = self.parent.subsect_tbl.itemAt(idx, col).text()
        except AttributeError:
            return
        value = float(item)

        if idx != 0 or value != self.xs_survey.subsections[-1]:
            self.xs_survey.remove_subsection(idx)
            self.update_backend()

    def add_subsection(self):
        """Added subsection at station in spin box."""

        value = self.parent.subsection_sb.value()
        self.xs_survey.add_subsections(value)

        self.update_backend()

    def update_subsection_table(self):
        """Creates Subsection table."""

        tbl = self.parent.subsect_tbl
        tbl.clear()
        headers = ["Sub-Sections"]

        bold = QtGui.QFont()
        bold.setBold(True)

        tbl.setRowCount(len(self.xs_survey.subsections))
        tbl.setColumnCount(len(headers))

        tbl.setHorizontalHeaderLabels(headers)
        tbl.horizontalHeader().setFont(bold)
        tbl.horizontalHeader().setVisible(True)
        tbl.verticalHeader().setVisible(False)

        # populate table if data is present
        if len(self.xs_survey.subsections) > 1:
            for row in range(len(self.xs_survey.subsections)):
                tbl.setItem(
                    row,
                    0,
                    QtWidgets.QTableWidgetItem(
                        "{:.2f}".format(self.xs_survey.subsections[row])
                    ),
                )

        tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        tbl.resizeColumnsToContents()
        tbl.resizeRowsToContents()

    def update_backend(self):
        """Update computations in backend."""

        if self.xs_survey.survey is None:
            return  # Nothing is loaded
        self.xs_survey.compute_channel_char(self.xs_survey.stage)
        if self.xs_survey.surveys:
            self.xs_survey.sub_section_chars()

            self.change.update(
                {"bathymetry": True, "plot": True, "sub-survey": True, "char": True}
            )

            # call code to update gui
            self.update_ui()

    def update_ui(self):
        """Update tables and plots in gui."""

        # update tables and plot
        if self.parent.tabWidget.currentIndex() == self.ivy_xs_tab_index:
            if self.parent.tabWidget_2.currentIndex() == 0:
                if self.change["bathymetry"]:
                    self.update_bathy_tbl()
                    self.change.update({"bathymetry": False})

            elif self.parent.tabWidget_2.currentIndex() == 2:
                if self.change["char"]:
                    self.update_channel_char_tbl()
                    self.update_subsection_table()
                    self.change.update({"char": False})

            elif self.parent.tabWidget_2.currentIndex() == 1:
                if self.change["sub-survey"]:
                    self.update_file_info_table()
                    self.change.update({"sub-survey": False})

            if self.change["plot"]:
                self.check_bathy_rb()
                self.change.update({"plot": False})
