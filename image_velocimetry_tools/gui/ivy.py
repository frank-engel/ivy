"""The Main IVyTools Module"""

import atexit
import csv
import datetime
import getpass
import glob
import json
import logging
import os
import re
import shutil
import sys
import time
import webbrowser
import zipfile
from contextlib import contextmanager
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from PIL import Image
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import (
    QDir,
    QUrl,
    pyqtSignal,
    QProcess,
    Qt,
    QThreadPool,
    QTime,
    QDate,
)
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from tqdm import tqdm

from image_velocimetry_tools import __version__, __author__
from image_velocimetry_tools.comments import Comments
from image_velocimetry_tools.common_functions import (
    float_seconds_to_time_string,
    seconds_to_frame_number,
    seconds_to_hhmmss,
    resource_path,
    find_matches_between_two_lists,
    bounding_box_naive,
    find_key_from_first_value,
    compute_vectors_with_projections,
    load_csv_with_numpy,
    get_column_contents,
    set_column_contents,
    string_to_boolean,
    units_conversion,
    geographic_to_arithmetic,
    parse_creation_time,
    calculate_uv_components,
    calculate_endpoint,
    dict_arrays_to_list,
    hhmmss_to_seconds,
)
from image_velocimetry_tools.ffmpeg_tools import (
    create_ffmpeg_command,
    parse_ffmpeg_stdout_progress,
    ffmpeg_compute_motion_trajectories_from_frames_command,
    ffmpeg_remove_motion_from_frames_command,
    ffprobe_add_exif_metadata,
)
from image_velocimetry_tools.file_management import (
    create_temp_directory,
    clean_up_temp_directory,
    format_windows_path,
    safe_make_directory,
    serialize_numpy_array,
    deserialize_numpy_array,
    locate_video_file,
    set_date,
    set_time,
    set_value_if_not_none,
    set_text_if_not_none,
    compare_versions,
)
from image_velocimetry_tools.graphics import (
    AnnotationView,
    Instructions,
    quiver,
    plot_quivers,
)
from image_velocimetry_tools.gui.HomographyDistanceConversionTool import (
    HomographyDistanceConversionTool,
)
from image_velocimetry_tools.gui.LensCharacteristics import LensCharacteristics
from image_velocimetry_tools.gui.adddocumentation import AddDocumentation
from image_velocimetry_tools.gui.dialogs.IVy_GUI import Ui_MainWindow
from image_velocimetry_tools.gui.dialogs.settings import \
    Settings as Settings_Dialog
from image_velocimetry_tools.gui.discharge import DischargeTab
from image_velocimetry_tools.exportpdf import Report
from image_velocimetry_tools.gui.filesystem import (
    FileSystemModelManager,
    TableWidgetDragRows,
    Worker,
)
from image_velocimetry_tools.services.video_service import VideoService
from image_velocimetry_tools.services.project_service import ProjectService
from image_velocimetry_tools.services.orthorectification_service import OrthorectificationService
from image_velocimetry_tools.services.image_stack_service import ImageStackService
from image_velocimetry_tools.gui.models.video_model import VideoModel
from image_velocimetry_tools.gui.controllers.video_controller import VideoController
from image_velocimetry_tools.gui.gridding import GridPreparationTab, \
    GridGenerator
from image_velocimetry_tools.gui.image_browser import ImageBrowserTab
from image_velocimetry_tools.gui.reporting import ReportingTab
from image_velocimetry_tools.gui.stiv_helper import StivHelper
from image_velocimetry_tools.gui.stiv_processor import STIVTab, STIReviewTab
from image_velocimetry_tools.gui.xsgeometry import CrossSectionGeometry
from image_velocimetry_tools.image_processing_tools import (
    ImageProcessor,
    create_grayscale_image_stack,
)
from image_velocimetry_tools.image_processing_tools import (
    create_change_overlay_image,
    flip_image_array,
    flip_and_save_images,
)
from image_velocimetry_tools.image_processing_tools import \
    image_file_to_numpy_array
from image_velocimetry_tools.opencv_tools import opencv_get_video_metadata
from image_velocimetry_tools.orthorectification import (
    CameraHelper,
    get_homographic_coordinates_3D,
    calculate_homography_matrix_simple,
    rectify_homography,
    transform_points_with_homography,
    space_to_image,
    image_to_space,
    estimate_view_angle,
    estimate_orthorectification_rmse,
    estimate_scale_based_rmse,
)
from image_velocimetry_tools.settings import Settings

# prevent numpy exponential notation on print, default False
np.set_printoptions(suppress=True)

# Set the logging level
logging.basicConfig(filename=None, level=logging.DEBUG)

# Fix for Windows users to propagate the gui icon to the Taskbar. This is
# needed because Windows is presuming that Python is the application, but
# actually, python is just hosting the application. By telling Windows
# the "Application User Model" for the gui, the correct icon will be used.
# See: https://stackoverflow.com/a/1552105
import ctypes

myappid = (
    f"USGS.ImageVelocityTools." f"image_velocimetry_tools.{__version__}.{__author__}"
)
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

# if there is a splash screen close it
try:
    import pyi_splash

    pyi_splash.close()
except Exception as err:
    logging.error(err)


class IvyTools(QtWidgets.QMainWindow, Ui_MainWindow):
    """The Main Class for IVyTools

    Args:
        QtWidgets (QWidget): the QWidget for the UI
        Ui_MainWindow: the UI

    Returns:

    """

    global IVY_ENV
    IVY_ENV = os.environ.get("IVY_ENV")
    logging.debug(f"Current deployment environment: {IVY_ENV}")

    signal_stderr = pyqtSignal(str)
    signal_stdout = pyqtSignal(str)
    signal_ffmpeg_thread = pyqtSignal(bool)
    signal_ffmpeg_caller = pyqtSignal(str)
    signal_opencv_updates = pyqtSignal(str)
    signal_orthotable_changed = pyqtSignal(bool)
    signal_orthotable_check_units = pyqtSignal()
    signal_ortho_original_digitized_point = pyqtSignal(
        dict
    )  # Used to transmit the list of digitized points
    signal_dischargetable_changed = pyqtSignal(bool)
    pointsExistSignal = pyqtSignal(bool)  # Used to indicate that points exist
    # signal_previous_image = pyqtSignal(
    #     int)  # When user clicks the previous image button
    # signal_next_image = pyqtSignal(
    #     int)  # When user clicks the next image button
    signal_cross_section_exists = pyqtSignal(
        bool
    )  # True if a cross-section line has been created
    signal_image_processor_progress = pyqtSignal(
        int
    )  # emitted by various methods in the ImageProcessor
    signal_polygon_points = pyqtSignal(np.ndarray)
    signal_wse_changed = pyqtSignal(float)
    signal_manual_vectors = pyqtSignal(dict)

    def __init__(self, parent=None):
        """Class init

        Args:
            parent (IVyTools object, optional): the main IVyTools object. Defaults to None.
        """
        super(IvyTools, self).__init__(parent)
        self.setAcceptDrops(
            True
        )  # enables ability to drag and drop items into MainWindow
        self.setupUi(self)
        self.__program_name__ = "Image Velocimetry Tools (development)"
        self.__user__ = getpass.getuser()
        self.__version__ = __version__
        # Set icon path for development or production
        if IVY_ENV == "development":
            self.__icon_path__ = "image_velocimetry_tools/gui/icons"
        else:
            self.__icon_path__ = "icons"
        self.setWindowTitle(f"{self.__program_name__} v{self.__version__}")
        self.setWindowIcon(
            QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "IVy_logo.svg"))
        )

        # Init the temporary swap directory structure
        self.swap_directory = create_temp_directory()
        self.file_system_watcher = QtCore.QFileSystemWatcher([self.swap_directory])
        self.swap_image_directory = safe_make_directory(
            os.path.join(self.swap_directory, "1-images")
        )
        self.swap_orthorectification_directory = safe_make_directory(
            os.path.join(self.swap_directory, "2-orthorectification")
        )
        self.swap_grids_directory = safe_make_directory(
            os.path.join(self.swap_directory, "3-grids")
        )
        self.swap_velocities_directory = safe_make_directory(
            os.path.join(self.swap_directory, "4-velocities")
        )
        self.swap_discharge_directory = safe_make_directory(
            os.path.join(self.swap_directory, "5-discharge")
        )
        self.swap_qaqc_directory = safe_make_directory(
            os.path.join(self.swap_directory, "6-qaqc")
        )
        logging.debug(f"Swap directory location: {self.swap_directory}")

        # Init the Project Structure Tree Views
        self.file_system_model = FileSystemModelManager(self.swap_directory)
        self.setup_tree_view(self.treeviewProjectStructureImageryProcessingTab)
        self.setup_tree_view(self.treeviewProjectStructureOrthorectificationTab)
        self.setup_tree_view(self.treeviewProjectStructureGridPreparationTab)
        self.setup_tree_view(self.treeviewProjectStructureSTIVTab)
        self.setup_tree_view(self.treeviewProjectStructureSTIVTabOpt)
        self.tree_views = {
            "treeviewProjectStructureImageryProcessingTab": self.treeviewProjectStructureImageryProcessingTab,
            "treeviewProjectStructureOrthorectificationTab": self.treeviewProjectStructureOrthorectificationTab,
            "treeviewProjectStructureGridPreparationTab": self.treeviewProjectStructureGridPreparationTab,
            "treeviewProjectStructureSTIVTab": self.treeviewProjectStructureSTIVTab,
            "treeviewProjectStructureSTIVTabOpt": self.treeviewProjectStructureSTIVTabOpt,
        }

        # Register the cleanup function to be called when the application exits
        # This ensures that the swap directory gets deleted
        atexit.register(self.cleanup_temp_directory)

        # Init gui buttons and controls
        # Note: QToolbutton toggle style is set
        # in Qt Creator's global style sheet. To set, in Qt Creator
        # right-click on the bottom area of the gui file and choose "Change
        # Stylesheet"
        self.init_indicator_icons()
        self.init_toolbar_menu_icons()

        # Init the class attributes
        self.init_class_attributes()

        # Init Comments functions
        self.comments = Comments()

        # Init Reporting Tab
        self.reporting = ReportingTab(self)

        # Enable/disable functions during start up
        self.update_statusbar(self.status_message)
        to_disable = [
            "PointPage",
            "SimpleLinePage",
            "RegularGridPage",
        ]
        self.set_qwidget_state_by_name(to_disable, False)
        # self.tabWidget_ImageVelocimetryMethods.setEnabled(False)
        # self.actionImport_Bathymetry.setEnabled(False)

        # Connections and Slots
        self.init_connections_and_slots()

        # Ensure the units are correctly set
        self.change_units(self.display_units)

        # Set to first tab on startup every time
        self.tabWidget.setCurrentIndex(0)

        #######################################################################
        # Development lock down
        # Temporarily force the app to have reduced functionality
        #######################################################################
        # 1. Disable and hide extra tabs in the app that are still draft
        self.tabSTIVOptimized.setEnabled(False)
        self.tabWidget_ImageVelocimetryMethods.setTabVisible(2, False)
        self.toolButtonNewProject.setEnabled(False)
        self.actionNew_Project.setEnabled(False)

        # 2. Force English Units
        self.units_label = "English"
        self.display_units = "English"
        try:
            ss = self.sticky_settings.set("last_display_units", self.display_units)
        except KeyError:
            self.sticky_settings.new("last_display_units", self.display_units)
        self.actionUnits.setEnabled(False)

        to_disable = [
            "tabImageFrameProcessing",
            "tabOrthorectification",
            "tabCrossSectionGeometry",
            "tabGridPreparation",
            "tabImageVelocimetry",
            "tabDischarge",
            "tabReporting",
        ]
        for tab in to_disable:
            self.enable_disable_tabs(self.tabWidget, tab, False)
        self.toolBox_bathymetry.setEnabled(False)

        # 3. Suppress the matplotlib.font_manager and numba.core.ssa warnings
        # to make feedback more helpful
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        logging.getLogger("numba.core.ssa").setLevel(logging.ERROR)

    def setup_tree_view(self, tree_view):
        """Configures the tree view that is used by the Project Manager

        Args:
            tree_view (QTreeViewWidget): the tree view widget
        """
        tree_view.setModel(self.file_system_model.get_file_system_model())
        tree_view.setRootIndex(self.file_system_model.index)

        columns_to_hide = [1, 2, 3, 4]
        for column in columns_to_hide:
            tree_view.setColumnHidden(column, True)

        tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        tree_view.customContextMenuRequested.connect(
            lambda pos, view=tree_view: self.openContextMenu(pos, view)
        )

    def openContextMenu(self, position, view):
        """Open a context window in the Project Manager

        Args:
            position (QPos): mouse position clicked to initiate the context menu
            view (QTreeView): the current tree view
        """
        # Get the selected item's index from the specified tree view
        tree_view = self.tree_views.get(view.objectName(), None)

        is_image, image_path = self.is_contextmenu_event_image_file(tree_view)
        if tree_view is not None:
            index = tree_view.currentIndex()

            if index.isValid():
                # Create a context menu
                menu = QtWidgets.QMenu(self)

                # Add a "Save" action to the context menu
                save_action = QtWidgets.QAction("Save", self)
                save_action.triggered.connect(
                    lambda: self.save_project_structure_context_menu(view)
                )
                menu.addAction(save_action)

                if is_image:
                    # Add a line to make the UI clean looking
                    menu.addSeparator()

                    # Add Select as GCP Image
                    gcp_action = QtWidgets.QAction(
                        "Set as the GCP Background Image", self
                    )
                    gcp_action.triggered.connect(
                        lambda: self.set_gcp_background_image(image_path)
                    )
                    menu.addAction(gcp_action)

                    # Add Apply as XS Image
                    xs_action = QtWidgets.QAction(
                        "Set as Cross-section Background Image", self
                    )
                    xs_action.triggered.connect(
                        lambda: self.set_cross_section_background_image(image_path)
                    )
                    menu.addAction(xs_action)

                    # Add Apply as Grid Prep Image
                    grid_action = QtWidgets.QAction(
                        "Set as Grid Preparation Background Image", self
                    )
                    grid_action.triggered.connect(
                        lambda: self.set_grid_preparation_background_image(image_path)
                    )
                    menu.addAction(grid_action)

                    # Add Apply as Image Velocity Image
                    image_action = QtWidgets.QAction(
                        "Set as Image Velocimetry Background Image", self
                    )
                    image_action.triggered.connect(
                        lambda: self.set_image_velocimetry_background_image(image_path)
                    )
                    menu.addAction(image_action)

                # Show the context menu at the given position
                menu.exec_(view.mapToGlobal(position))

    def set_gcp_background_image(self, image_filename):
        """Set the GCP background image"""
        self.ortho_original_image.open(image_filename)
        self.warning_dialog(
            title="Background image set",
            message="Background image successfully applied",
            style="ok",
            icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
        )

    def set_cross_section_background_image(self, image_filename):
        """Set the Cross-section background image"""

        # Check to see if the user is trying to set a transformed image, if
        # so warn them and provide a cancel capability
        pattern = r"^t\d{5}\.jpg$"
        basename = os.path.basename(image_filename)
        if re.match(pattern, basename) is not None:
            # True means a t* image
            result = self.warning_dialog(
                title=f"Warning: Did you mean to supply a transformed image?",
                message=f"It looks like you are trying to apply a transformed "
                f"image to the perspective view of the Cross-Section "
                f"Geometry tab.\n\n This function sets the "
                f"perspective view only. If you want to set the "
                f"rectified view image, first set the desired Ground "
                f"Control Image and then click Rectify Current Frame."
                f"\n\n Do you want to continue?",
                style="YesCancel",
                icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
            )
            if result != "yes":
                return

        self.warning_dialog(
            title=f"Information: Sets the Perspective Image",
            message=f"This will set the perspective cross-section image in "
            f"the Cross Section Geometry tab. It does not set the "
            f"rectified image. To apply a new rectified image, set "
            f"the desired image as the Ground Control Points image "
            f"and rectify the current frame.",
            style="ok",
            icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
        )
        image = image_file_to_numpy_array(image_filename)
        self.perspective_xs_image.scene.setImage(image)
        self.warning_dialog(
            title="Background image set",
            message="Background image successfully applied",
            style="ok",
            icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
        )

    def set_grid_preparation_background_image(self, image_filename):
        """Set the Grid Preparation background image"""
        # Check to see if the user is trying to set a transformed image, if
        # so warn them and provide a cancel capability
        pattern = r"^t\d{5}\.jpg$"
        basename = os.path.basename(image_filename)
        if re.match(pattern, basename) is not None:
            # True means a t* image
            self.gridpreparation.imageBrowser.scene.load_image(image_filename)
            self.warning_dialog(
                title="Background image set",
                message="Background image successfully applied",
                style="ok",
                icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
            )
        else:
            result = self.warning_dialog(
                title=f"Warning: Did you mean to supply a perspective image?",
                message=f"It looks like you are trying to apply a perspective "
                f"image to the Grid Preparation Tab.\n\n This "
                f"function sets the rectified view only."
                f"\n\n Do you want to continue?",
                style="YesCancel",
                icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
            )
            if result == "yes":
                self.gridpreparation.imageBrowser.scene.load_image(image_filename)
                self.warning_dialog(
                    title="Background image set",
                    message="Background image successfully applied",
                    style="ok",
                    icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
                )
            else:
                return

    def set_image_velocimetry_background_image(self, image_filename):
        """Set teh Image Velocimetry background image"""
        # Check to see if the user is trying to set a transformed image, if
        # so warn them and provide a cancel capability
        pattern = r"^t\d{5}\.jpg$"
        basename = os.path.basename(image_filename)
        if re.match(pattern, basename) is not None:
            # True means a t* image
            self.stiv.imageBrowser.scene.load_image(image_filename)
            self.warning_dialog(
                title="Background image set",
                message="Background image successfully applied",
                style="ok",
                icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
            )
        else:
            result = self.warning_dialog(
                title=f"Warning: Did you mean to supply a perspective image?",
                message=f"It looks like you are trying to apply a perspective "
                f"image to the Grid Preparation Tab.\n\n This "
                f"function sets the rectified view only."
                f"\n\n Do you want to continue?",
                style="YesCancel",
                icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
            )
            if result == "yes":
                self.stiv.imageBrowser.scene.load_image(image_filename)
                self.warning_dialog(
                    title="Background image set",
                    message="Background image successfully applied",
                    style="ok",
                    icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
                )
            else:
                return

    def open_settings_dialog(self):
        """
        Open the settings dialog.

        Description:
            This function opens the settings dialog, allowing the user to change the display units
            for the survey or enable/disable under-ice mode. It initializes the dialog with the
            current display units and the icon path, waits for the dialog to be closed, and then
            retrieves the selected units. If the units have changed, it updates the display units,
            changes the plot settings accordingly, and converts any relevant data to the new units.

        Returns:
            None

        """

        # If the user had a previous display units, use it
        try:
            ss = self.sticky_settings.get("last_display_units")
            self.display_units = ss
        except KeyError:
            self.sticky_settings.new("last_display_units", self.display_units)
        display_units = self.display_units

        dialog = Settings_Dialog(
            units=display_units,
            icon_path=resource_path(
                resource_path(self.__icon_path__ + os.sep + "IVy_Logo.ico"),
            ),
            parent=self,
        )
        dialog.exec_()

        if dialog.units is not None:
            if dialog.units != self.display_units:
                self.display_units = dialog.units

                # Change plot settings
                self.units_label = dialog.units

                # Update file menu
                self.actionUnits.setText(f"Units: {self.units_label}")
                # if self.units_label == "English":
                #     self.comboBoxPlotOptions.setCurrentIndex(1)
                # if self.units_label == "Metric":
                #     self.comboBoxPlotOptions.setCurrentIndex(2)
                # self.update_plot_options()

                # Update the user's settings
                self.sticky_settings.set("last_display_units", self.display_units)

            # Convert any data
            self.survey_units = units_conversion(self.display_units)
            self.change_units(self.display_units)
            logging.debug(f"NEW display units: {self.display_units}")

    def change_units(self, units="English"):
        """
        Apply units conversions and labels globally to all loaded data and elements.

        Args:
            units (str, optional): The units to be applied. Defaults to "English".

        Description:
            This function applies units conversions and labels to all loaded data and elements in the GUI.
            It updates the labels for the Known Values and Discharge Calculator sections based on the specified units.
            The function also triggers the update of the tables and recalculates the discharge if applicable.

        Returns:
            None

        """

        # gui Labels
        self.labelWaterSurfaceElevation.setText(
            f"Water Surface Elevation {self.survey_units['label_L']}:"
        )
        self.labelPixelGSD.setText(f"Pixel GSD {self.survey_units['label_L']}:")
        self.labelStivMaxVelThreshold.setText(
            f"Max Vel. Threshold {self.survey_units['label_V']}:"
        )
        self.labelStivOptMaxVelThreshold.setText(
            f"Max Vel. Threshold {self.survey_units['label_V']}:"
        )
        self.labelCrossSectionMeasurementStage.setText(
            f"Measurement Stage {self.survey_units['label_L']}:"
        )
        self.stationStationLabel.setText(
            f"Starting Station {self.survey_units['label_L']}:"
        )
        self.gageHeightLabel.setText(f"Stage {self.survey_units['label_L']}:")

        # Orthorectification Points Table
        if self.is_ortho_table_loaded:
            self.orthotable_populate_table(self.orthotable_dataframe)
            self.orthotable_change_units()
        old = self.doubleSpinBoxRectificationWaterSurfaceElevation.value()
        self.doubleSpinBoxRectificationWaterSurfaceElevation.setValue(old * units_conversion(self.display_units)['L'])
        if self.pixel_ground_scale_distance_m is not None:
            old = float(self.lineeditPixelGSD.text())
            self.lineeditPixelGSD.setText(
                f"{old * units_conversion(self.display_units)['L']:.3f}"
            )

        # Cross-section Geometry Tab (AC3)
        if self.bathymetry_ac3_filename is not None:
            # Reloading will change units
            self.xs_survey.load_areacomp(self.bathymetry_ac3_filename)

            # Then delete the old subsurvey
            survey = self.xs_survey.xs_survey.surveys[-1].file_id
            self.xs_survey.xs_survey.remove_survey(survey)
            self.xs_survey.update_backend()

            # Update the AC3 gui elements and backend
            old = self.char_stage_sb.value()
            self.char_stage_sb.setValue(old * units_conversion(self.display_units)["L"])
            old = float(self.start_station_lineEdit.text())
            self.start_station_lineEdit.setText(
                f"{old * units_conversion(self.display_units)['L']:.2f}"
            )
            old = float(self.stage_lineEdit.text())
            self.stage_lineEdit.setText(
                f"{old * units_conversion(self.display_units)['L']:.2f}"
            )
            self.xs_survey.change = {
                "bathymetry": True,
                "plot": True,
                "sub-survey": True,
                "char": True,
            }
            self.xs_survey.update_ui()

        # Discharge Measurements Table
        pass

        # Discharge Results Table
        pass

    def set_qwidget_state_by_name(self, widget_names, state: bool) -> None:
        """
        Helper function to enable or disable one or more QWidget(s) based on objectName.

        Parameters:
        - widget_names: Either a single str or a list of str representing objectName(s) of the QWidget(s).
        - state (bool): True to enable the widget(s), False to disable it/them.

        Returns:
        - None
        """
        if isinstance(widget_names, str):
            widget_names = [widget_names]

        for widget_name in widget_names:
            widget = self.findChild(QtWidgets.QWidget, widget_name)
            if widget:
                widget.setEnabled(state)
            else:
                logging.error(
                    f"FUNC:set_qwidget_state_by_name: Widget with "
                    f"objectName '{widget_name}' not found."
                )

    def is_contextmenu_event_image_file(self, view):
        """Test if the selected item is an image file

        Args
            view (object): the current tree view

        Returns
            bool: True if the current tree view context selection is an image
            str:  or None: if current tree view contains an image, the path to
                  the image is returned, otherwise None
        """
        # Get the selected item's index from the specified tree view
        tree_view = self.tree_views.get(view.objectName(), None)
        if tree_view is not None:
            indexes = tree_view.selectedIndexes()
            if indexes:
                index = indexes[0]
                file_path = tree_view.model().filePath(index)
                if os.path.isfile(file_path):
                    # List of valid image file extensions
                    image_extensions = {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".gif",
                        ".bmp",
                        ".tiff",
                        ".webp",
                    }
                    # Check if file extension matches an image type
                    _, ext = os.path.splitext(file_path)
                    if ext.lower() in image_extensions:
                        return True, file_path
        return False, None

    def save_project_structure_context_menu(self, view):
        """Saves the selected files or folders in the Project Manager to disk

        Args:
            view (QTreeView): the tree view
        """
        # Get the selected item's index from the specified tree view
        tree_view = self.tree_views.get(view.objectName(), None)
        if tree_view is not None:
            indexes = tree_view.selectedIndexes()
            if indexes:
                index = indexes[0]
                file_path = tree_view.model().filePath(index)

                # Check if the selected item is a folder
                if os.path.isdir(file_path):
                    # Use QFileDialog to choose a save location for the entire folder
                    save_path = QtWidgets.QFileDialog.getExistingDirectory(
                        self, "Select a directory to save the folder"
                    )

                    if save_path:
                        # Create a new folder with the same name in the chosen location
                        save_folder_path = os.path.join(
                            save_path, os.path.basename(file_path)
                        )
                        try:
                            shutil.copytree(file_path, save_folder_path)
                        except Exception as e:
                            print(f"An error occurred while copying the folder: {e}")

                elif os.path.isfile(file_path):
                    # Use QFileDialog to choose a save location for the file
                    save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                        self, "Save File", os.path.basename(file_path)
                    )
                    if save_path:
                        # Copy the selected file to the save location
                        try:
                            shutil.copyfile(file_path, save_path)
                        except Exception as e:
                            print(f"An error occurred while copying the file: {e}")

    def new_project(self):
        """Create a new IVy Project, discarding all currently loaded data"""
        self.project_filename = f"{QDir.homePath()}{os.sep}New_IVy_Project.ivy"
        # 1. Unload video
        self.video_file_name = None
        self.video_player.setMedia(QMediaContent())
        logging.info("Video player reset. No media loaded.")

        # 2. Reinit class attributes
        # self.init_class_attributes()  # Maybe not needed?
        # 2. Clear images from browsers

        # 3. Unload AC3 file if present
        # 4. Reinit all Image Browsers
        # 5. Clear project trees
        # 6. Clear comments
        pass

    def _draw_cross_section_line(self):
        """Draw the rectified cross-section line on the image."""
        self.rectified_xs_image.clearLines()
        self.rectified_xs_image.scene.set_current_instruction(
            Instructions.ADD_LINE_BY_POINTS,
            points=self.cross_section_rectified_eps,
        )
        if self.rectified_xs_image.scene.line_item:
            self.rectified_xs_image.scene.line_item[-1].setPen(
                QtGui.QPen(QtGui.QColor("yellow"), 5)
            )

    def _update_cross_section_spinboxes(self):
        """Update spinboxes for left and right bank coordinates."""
        left, right = self.cross_section_rectified_eps
        self.sbLeftBankXRectifiedCoordPixels.setValue(left[0])
        self.sbLeftBankYRectifiedCoordPixels.setValue(left[1])
        self.sbRightBankRectifiedXCoordPixels.setValue(right[0])
        self.sbRightBankRectifiedYCoordPixels.setValue(right[1])
        self.cross_section_line_exists = True

    def _project_cross_section_line(self):
        """Trigger cross-section projection depending on UI state."""
        was_checked = self.radioButtonRectifiedImage.isChecked()
        self.radioButtonRectifiedImage.setChecked(True)
        self.cross_section_manager()
        if not was_checked:
            self.radioButtonOriginalImage.setChecked(True)

    def _load_stiv_results_from_csv(self, csv_file_path):
        """Load STIV results from CSV if available."""
        if not os.path.isfile(csv_file_path):
            return None

        headers, data = load_csv_with_numpy(csv_file_path)

        try:
            return {
                "X": data[:, 0].astype(float),
                "Y": data[:, 1].astype(float),
                "U": data[:, 2].astype(float),
                "V": data[:, 3].astype(float),
                "M": data[:, 4].astype(float),
                "scalar_projections": data[:, 5].astype(float),
                "D": data[:, 6].astype(float),
                "tagline_dir_geog": data[:, 7].astype(float),
            }
        except IndexError:
            logging.warning(
                f"Attempted to load {csv_file_path} but the file format was unexpected."
            )
            return None

    def open_project(self):
        """Open an IVy Project Session File"""

        # Load the last project filename
        try:
            ss = self.sticky_settings.get("last_project_filename")
            self.project_filename = ss
        except KeyError:
            self.project_filename = f"{QDir.homePath()}{os.sep}New_IVy_Project.ivy"
            self.sticky_settings.new("last_project_filename", self.project_filename)

        # Open a project file
        filter_spec = "IVy Project (*.ivy);;All files (*.*)"
        project_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Open IVy Project File",
            self.project_filename,
            # path
            filter_spec,
        )
        if project_filename:
            self.project_filename = project_filename
            logging.info(f"Project file loaded: {project_filename}")

            # Extract the zip file to the swap directory
            try:
                self.project_service.extract_project_archive(
                    self.project_filename,
                    self.swap_directory
                )
            except (zipfile.BadZipFile, IOError, FileNotFoundError, ValueError) as e:
                # Handle the exception, display a warning dialog, and log the error
                self.warning_dialog(
                    "Error Opening Project",
                    f"An error occurred while opening the project: {str(e)}",
                    style="ok",
                    icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
                )
                logging.error(f"Error opening project: {str(e)}")
                return

            # Load the project_dict from the JSON file in the swap directory
            json_filename = os.path.join(self.swap_directory, "project_data.json")
            try:
                project_dict = self.project_service.load_project_from_json(json_filename)
                project_dict["project_file_path"] = project_filename
            except (FileNotFoundError, ValueError, IOError) as e:
                # Handle the file not found error, display a warning dialog, and log the error
                self.warning_dialog(
                    "Error Opening Project",
                    f"An error occurred while opening the project: {str(e)}",
                    style="ok",
                    icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
                )
                logging.error(f"Error opening project: {str(e)}")
                return

            # Task 0 - Comments and Reporting items
            # Loaded first, because the comments may be useful for debugging or
            # review purposes.
            try:
                comments = project_dict.get("comments", None)
                if comments is not None:
                    if isinstance(comments, dict):
                        self.comments.load_dict(comments)
                        self.update_comment_tbl()
                else:
                    logging.debug(f"No comments in the supplied *.ivy Project file.")
            except Exception as e:
                logging.error(f"An error occurred while loading comments: {e}")

            # Reporting items
            reporting = project_dict.get("reporting", None)
            if reporting is not None:
                if isinstance(reporting, dict):
                    self.reporting.station_name = reporting.get("station_name")
                    set_text_if_not_none(
                        reporting.get("station_name"), self.stationNamelineEdit
                    )

                    self.reporting.station_number = reporting.get("station_number")
                    set_text_if_not_none(
                        reporting.get("station_number"), self.stationNumberLineEdit
                    )

                    self.reporting.party = reporting.get("party")
                    set_text_if_not_none(reporting.get("party"), self.partyLineEdit)

                    self.reporting.weather = reporting.get("weather")
                    set_text_if_not_none(reporting.get("weather"), self.weatherLineEdit)

                    self.reporting.meas_date = reporting.get("meas_date")
                    set_date(reporting.get("meas_date"), self.measDate)

                    self.reporting.gage_ht = reporting.get("gage_ht")
                    set_value_if_not_none(
                        reporting.get("gage_ht"), self.gageHeightdoubleSpinBox
                    )

                    self.reporting.start_time = reporting.get("start_time")
                    set_time(reporting.get("start_time"), self.measStartTime)

                    self.reporting.end_time = reporting.get("end_time")
                    set_time(reporting.get("end_time"), self.measEndTime)
                    try:
                        self.reporting.calc_mid_time()
                    except:
                        pass

                    self.reporting.meas_number = reporting.get("meas_number")
                    set_value_if_not_none(
                        reporting.get("meas_number"), self.measurementNumberspinBox, int
                    )

                    self.reporting.project_description = reporting.get(
                        "project_description"
                    )
                    set_text_if_not_none(
                        reporting.get("project_description"),
                        self.projectDescriptionTextEdit,
                    )

            # Task 1 - Video
            video_file_name = project_dict.get("video_file_name", None)
            if video_file_name is not None:
                self.video_file_name = locate_video_file(project_dict)
                if self.video_file_name is None:
                    self.video_metadata = project_dict["video_metadata"]
                    self.set_video_metadata(self.video_metadata)
            else:
                msg = (
                    f"OPEN PROJECT: Project file does not contain a video file location"
                )
                logging.error(msg)
                self.warning_dialog(
                    f"Error Opening Project",
                    msg,
                    style="ok",
                    icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
                )

            if self.video_file_name:  # will return False if no video
                logging.info(f"Video loaded: {self.video_file_name}")
                try:
                    # Use video controller to load the video
                    self.video_controller.load_video(self.video_file_name)
                    self.play_video()  # This pauses the video immediately

                except:
                    msg = f"OPEN PROJECT: Unable to load video"
                    logging.error(msg)
                    self.warning_dialog(
                        f"Error Opening Project",
                        msg,
                        style="ok",
                        icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
                    )
                    return
            try:
                start_time = hhmmss_to_seconds(project_dict["ffmpeg_parameters"]["start_time"]) * 1000  # ms
                end_time = hhmmss_to_seconds(project_dict["ffmpeg_parameters"]["end_time"]) * 1000   # ms

                # Update model - this will emit signals that update UI
                self.video_model.set_clip_times(start_time, end_time)
                # UI labels are updated automatically by VideoController signal handler
                self.set_menu_item_color("actionOpen_Video", "good")
                self.set_tab_icon("tabVideoPreProcessing", "good")
            except:
                msg = f"OPEN PROJECT: Unable to set video clip information"
                logging.error(msg)
                self.warning_dialog(
                    f"Error Opening Project",
                    msg,
                    style="ok",
                    icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
                )

            try:
                # FFMPEG settings
                ffmpeg_parameters = project_dict["ffmpeg_parameters"]
                self.video_rotation = ffmpeg_parameters["video_rotation"]
                self.comboboxFfmpegRotation.setCurrentText(
                    f"{ffmpeg_parameters['video_rotation']}"
                )
                self.video_flip = ffmpeg_parameters["video_flip"]
                self.comboboxFfmpegFlipVideo.setCurrentText(
                    f"{ffmpeg_parameters['video_flip']}"
                )
                self.video_strip_audio = ffmpeg_parameters["strip_audio"]
                self.checkboxStripAudio.setChecked(self.video_strip_audio)
                self.video_normalize_luma = ffmpeg_parameters["normalize_luma"]
                self.checkboxFfmpegNormalizeLuma.setChecked(
                    self.video_normalize_luma
                )
                self.video_curve_preset = ffmpeg_parameters["curve_preset"]
                self.comboboxFfmpegCurvePresets.setCurrentText(
                    f"{self.video_curve_preset}"
                )
                self.video_ffmpeg_stabilize = ffmpeg_parameters["stabilize"]
                self.checkboxFfmpeg2PassStabilization.setChecked(
                    self.video_ffmpeg_stabilize
                )
                temp_dict = dict(
                    [
                        (i, self.comboboxFfmpegRotation.itemText(i))
                        for i in range(self.comboboxFfmpegRotation.count())
                    ]
                )
                ind = find_key_from_first_value(
                    temp_dict, str(project_dict["video_rotation"])
                )
                self.comboboxFfmpegRotation.setCurrentIndex(ind)
                temp_dict = dict(
                    [
                        (i, self.comboboxFfmpegFlipVideo.itemText(i))
                        for i in range(self.comboboxFfmpegFlipVideo.count())
                    ]
                )
                ind = find_key_from_first_value(
                    temp_dict, str(project_dict["video_flip"])
                )
                self.comboboxFfmpegFlipVideo.setCurrentIndex(ind)
                temp_dict = dict(
                    [
                        (i, self.comboboxFfmpegCurvePresets.itemText(i))
                        for i in range(self.comboboxFfmpegCurvePresets.count())
                    ]
                )
                ind = find_key_from_first_value(
                    temp_dict, str(project_dict["video_curve_preset"])
                )
                self.comboboxFfmpegCurvePresets.setCurrentIndex(ind)

                # CRITICAL: Also update VideoModel with loaded processing settings
                # This ensures the model stays in sync with UI and ivy.py properties
                self.video_model.video_rotation = self.video_rotation
                self.video_model.video_flip = self.video_flip
                self.video_model.video_strip_audio = self.video_strip_audio
                self.video_model.video_normalize_luma = self.video_normalize_luma
                self.video_model.video_curve_preset = self.video_curve_preset
                self.video_model.video_ffmpeg_stabilize = self.video_ffmpeg_stabilize
            except:
                msg = f"OPEN PROJECT: Unable to load video metadata"
                logging.error(msg)
                self.warning_dialog(
                    f"Error Opening Project",
                    msg,
                    style="ok",
                    icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
                )

            # Lens Correction
            try:
                cx, cy, k1, k2 = (
                    float(ffmpeg_parameters["cx"]),
                    float(ffmpeg_parameters["cy"]),
                    float(ffmpeg_parameters["k1"]),
                    float(ffmpeg_parameters["k2"]),
                )
                self.lens_characteristics = LensCharacteristics(
                    self,
                    width=self.video_metadata["width"],
                    height=self.video_metadata["height"],
                    cx=cx,
                    cy=cy,
                    k1=k1,
                    k2=k2,
                )
                self.checkboxCorrectRadialDistortion.setChecked(
                    ffmpeg_parameters["calibrate_radial"]
                )
                self.labelLensCxValue.setText(
                    f"{int(self.lens_characteristics.width / 2):d}"
                )
                self.labelLensCyValue.setText(
                    f"{int(self.lens_characteristics.height / 2):d}"
                )
                self.labelLensK1Value.setText(f"{k1:.3f}")
                self.labelLensK2Value.setText(f"{k2:.3f}")
            except:
                msg = f"OPEN PROJECT: Unable to set Lens Characteristics"
                logging.error(msg)
                self.warning_dialog(
                    f"Error Opening Project",
                    msg,
                    style="ok",
                    icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
                )

            try:
                # Extract Frames settings
                self.lineeditFrameStepValue.setText(
                    f"{project_dict['extraction_frame_step']:d}"
                )
                self.frame_step_changed()
                self.update_ffmpeg_parameters()
                # CRITICAL: Update extraction UI with restored values
                # This ensures the "New Frame Rate", "New Timestep", and "New Num Frames"
                # labels are updated even if clip times haven't changed from defaults
                self.video_controller._show_clip_information()
            except:
                msg = (f"OPEN PROJECT: Unable to set frame extraction "
                       f"information")
                logging.error(msg)
                self.warning_dialog(
                    f"Error Opening Project",
                    msg,
                    style="ok",
                    icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
                )

            # Task 2 - Image Frames
            # If there are loaded frames, the project file should have an
            # imagebrowser_sequence and index, set those
            try:
                # There are image frames
                if project_dict["imagebrowser_sequence"] is not None:
                    self.imagebrowser.sequence_index = project_dict["sequence_index"]
                    self.imagebrowser.sequence = [
                        os.path.join(self.swap_image_directory, item)
                        for item in project_dict["imagebrowser_sequence"]
                    ]
                    self.imagebrowser.glob_pattern = f"f*.jpg"
                    self.imagebrowser.image_path = os.path.join(
                        self.swap_image_directory,
                        project_dict["imagebrowser_image_path"],
                    )
                    self.imagebrowser.folder_path = self.swap_image_directory
                    self.imagebrowser.reload = True  # Ensures the frame loader will not prompt user for a folder
                    self.imagebrowser.open_image_folder()
                    self.imagebrowser.reload = False
                    self.groupboxTools.setEnabled(True)
                    self.groupboxProcessing.setEnabled(True)

                    # Update Export Frames
                    self.set_button_color("buttonExtractVideoFrames", "good")
                    self.set_menu_item_color("actionOpen_Image_Folder", "good")
                    self.set_tab_icon("tabImageFrameProcessing", "good")
            except KeyError:
                pass

            # Take note of if there are already transformed frames
            if glob.glob(os.path.join(self.swap_image_directory, "t*.jpg")):
                self.is_transformed_frames = True

            # Task 3 - Orthorectification
            # Load a GCP image if available
            try:
                calibration_image_path = os.path.join(
                    self.swap_orthorectification_directory, "!calibration_image.jpg"
                )

                if os.path.exists(
                    calibration_image_path
                ):  # Check for !calibration_image.jpg
                    self.ortho_original_image.open(filepath=calibration_image_path)
                    self.ortho_original_image.setEnabled(True)
                    self.set_menu_item_color("actionOpen_Ground_Control_Image", "good")
                elif (
                    "last_ortho_gcp_image_path" in project_dict
                    and project_dict["last_ortho_gcp_image_path"] is not None
                ):  # Check for last known GCP image path
                    self.last_ortho_gcp_image_path = project_dict[
                        "last_ortho_gcp_image_path"
                    ]
                    if os.path.exists(self.last_ortho_gcp_image_path):
                        self.ortho_original_image.open(
                            filepath=self.last_ortho_gcp_image_path
                        )
                        self.ortho_original_image.setEnabled(True)
                        self.set_menu_item_color(
                            "actionOpen_Ground_Control_Image", "good"
                        )
                    else:  # Path to GCP image is invalid, ask the user
                        choices = ("Ok", "Cancel")
                        choice = self.custom_dialog_index(
                            title="Project File Issue",
                            message=(
                                "The last known path to the GCP image is not valid.\n\n"
                                "This may be because the GCP image file was moved or deleted. "
                                "Please select the GCP image file to continue loading the project, "
                                "or cancel to skip this step."
                            ),
                            choices=choices,
                        )
                        if choices[choice].lower() == "ok":
                            filter_spec = "Jpeg Image (*.jpg);;All files (*.*)"
                            gcp_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                                None,
                                "Select Ground Control Points (GCP) Image",
                                self.project_filename,  # path
                                filter_spec,
                            )
                            if gcp_path:
                                try:
                                    # Put image into the viewer
                                    self.ortho_original_load_gcp_image(gcp_path)
                                    self.sticky_settings.set(
                                        "last_ortho_gcp_image_path",
                                        self.ortho_original_image.image_file_path,
                                    )
                                except KeyError:
                                    self.sticky_settings.new(
                                        "last_ortho_gcp_image_path",
                                        self.ortho_original_image.image_file_path,
                                    )
                        else:
                            return
                else:  # No valid GCP image found, prompt user
                    choices = ("Ok", "Cancel")
                    choice = self.custom_dialog_index(
                        title="No GCP Image Found",
                        message=(
                            "No valid GCP image file could be found in the project directory "
                            "or in the swap orthorectification directory.\n\n"
                            "Please select the GCP image file to continue, or cancel to skip this step."
                        ),
                        choices=choices,
                    )
                    if choices[choice].lower() == "ok":
                        filter_spec = "Jpeg Image (*.jpg);;All files (*.*)"
                        gcp_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                            None,
                            "Select Ground Control Points (GCP) Image",
                            self.project_filename,  # path
                            filter_spec,
                        )
                        if gcp_path:
                            try:
                                # Put image into the viewer
                                self.ortho_original_load_gcp_image(gcp_path)
                                self.sticky_settings.set(
                                    "last_ortho_gcp_image_path",
                                    self.ortho_original_image.image_file_path,
                                )
                            except KeyError:
                                self.sticky_settings.new(
                                    "last_ortho_gcp_image_path",
                                    self.ortho_original_image.image_file_path,
                                )
                    else:
                        return
            except KeyError:
                pass

            # Load GCP Points table
            try:
                ground_control_points_path = os.path.join(
                    self.swap_orthorectification_directory, "ground_control_points.csv"
                )

                if os.path.exists(ground_control_points_path):
                    # Prioritize loading ground_control_points.csv if it exists
                    self.orthotable_load_csv_on_open(ground_control_points_path)
                elif "orthotable_dataframe" in project_dict:
                    # Fallback to project_dict["orthotable_dataframe"] if
                    # no CSV file
                    self.orthotable_dataframe = pd.DataFrame(
                        project_dict["orthotable_dataframe"]
                    )
                    self.orthotable_file_survey_units = "Metric"  # Always metric
                    self.orthotable_populate_table(self.orthotable_dataframe)
                    self.orthotable_update_table_headers()
                    self.orthotable_change_units()
                    self.is_ortho_table_loaded = True
                    self.set_button_color("toolbuttonOpenOrthoPointsTable", "good")
                    self.set_menu_item_color(
                        "actionImport_Ground_Control_Points_Table", "good"
                    )
                    # Write DataFrame to ground_control_points.csv. This
                    # file will always be in meters.
                    self.orthotable_dataframe.to_csv(
                        ground_control_points_path, index=False
                    )
                    logging.info(
                        f"PROJECT IMPORT: GCP file was not present in the "
                        f"project file "
                        f"(possibly an earlier version of IVy). A copy "
                        f"of the currently loaded GCP table saved to "
                        f"project structure. NOTE: units are METERS."
                    )
            except Exception as e:
                # Log or handle exception
                pass

            # Water surface elevation
            try:
                if project_dict.get("water_surface_elevation") is not None:
                    self.ortho_rectified_wse_m = project_dict["water_surface_elevation"]
                    self.doubleSpinBoxRectificationWaterSurfaceElevation.setValue(
                        self.ortho_rectified_wse_m *
                        units_conversion(self.display_units)['L'])
            except Exception as e:
                # Log or handle exception
                pass

            # Pixel ground scale distance
            try:
                if project_dict.get("pixel_ground_scale_distance_m") is not None:
                    self.pixel_ground_scale_distance_m = project_dict[
                        "pixel_ground_scale_distance_m"
                    ]
                    self.lineeditPixelGSD.setText(
                        f"{self.pixel_ground_scale_distance_m * units_conversion(self.display_units)['L']}"
                    )
            except Exception as e:
                # Log or handle exception
                pass

            if project_dict.get("is_ortho_flip_x") is not None:
                self.is_ortho_flip_x = project_dict.get("is_ortho_flip_x")
                self.checkBoxOrthoFlipX.setChecked(self.is_ortho_flip_x)
            if project_dict.get("is_ortho_flip_y") is not None:
                self.is_ortho_flip_y = project_dict.get("is_ortho_flip_y")
                self.checkBoxOrthoFlipY.setChecked(self.is_ortho_flip_y)

            # Rectify image?
            self.groupboxOrthoOrigImageTools.setEnabled(True)
            self.groupboxExportOrthoFrames.setEnabled(True)
            self.toolbuttonOrthoOrigImageDigitizePoint.setEnabled(
                True
            )  # must load table first
            self.toolbuttonOpenOrthoPointsTable.setEnabled(True)

            try:
                # if we have this, the user previous rectified, so go ahead
                # and trigger rectify single frame

                # Apply the same flip X & Y the user saved.


                if project_dict["pixel_ground_scale_distance_m"] is not None:
                    self.rectify_single_frame()
                    self.set_button_color("buttonRectifyCurrentImage", "good")
            except:
                pass

            # Task 4 - Cross-section geometry
            # Look for the AC3 file in 5-discharge and load from there. If
            # that file is not present, try t oload what is in the
            # project_dict. Otherwise skip this step.
            try:
                cross_section_file_path = os.path.join(
                    self.swap_discharge_directory, "cross_section_ac3.mat"
                )
                if os.path.exists(cross_section_file_path):
                    # Prioritize loading cross_section_ac3.mat if it exists
                    self.xs_survey.load_areacomp(cross_section_file_path)
                    self.is_area_comp_loaded = True
                    self.toolBox_bathymetry.setEnabled(True)
                elif project_dict["bathymetry_ac3_filename"] is not None:
                    fname = project_dict["bathymetry_ac3_filename"]
                    if os.path.exists(fname):
                        self.xs_survey.load_areacomp(fname)
                        self.is_area_comp_loaded = True
                        self.toolBox_bathymetry.setEnabled(True)
                    else:
                        self.warning_dialog(
                            "AreaComp File Not Found",
                            "Unable to load AreaComp3 file path "
                            "that was saved in the IVy Project "
                            "file. Please verify it exists. "
                            "\n\nYou can re-import bathymetry "
                            "with the Import Menu.",
                            style="ok",
                            icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
                        )

                cross_section_start_bank = project_dict.get(
                    "cross_section_start_bank")
                if cross_section_start_bank is not None:
                    self.cross_section_start_bank = cross_section_start_bank

                cross_section_line = project_dict.get("cross_section_line")
                if cross_section_line is not None:
                    self.cross_section_line = deserialize_numpy_array(
                        cross_section_line)
                    self.cross_section_rectified_eps = self.cross_section_line.reshape(
                        2, 2)

                    # Update the UI
                    self._draw_cross_section_line()
                    self._update_cross_section_spinboxes()
                    self._project_cross_section_line()

                rectification_method = project_dict.get("rectification_method")
                if rectification_method is not None:
                    self.rectification_method = rectification_method

                    self.set_menu_item_color("actionImport_Bathymetry", "good")
                    self.set_tab_icon("tabCrossSectionGeometry", "good")

            except (TypeError, AttributeError, KeyError) as e:
                logging.warning(f"Failed to load Cross-section parameters: {e}")

            # Task 5 - Grid Preparation
            try:
                # Handle mask polygons
                mask_polygons = project_dict.get("mask_polygons")
                if mask_polygons:
                    for polygon in mask_polygons:
                        self.gridpreparation.imageBrowser.scene.set_current_instruction(
                            Instructions.ADD_POLYGON_INSTRUCTION,
                            points=deserialize_numpy_array(polygon),
                        )

                # Handle cross-section grid
                if project_dict.get("is_cross_section_grid"):
                    self.is_cross_section_grid = True
                    num_points = project_dict.get("number_grid_points_along_xs_line")

                    if num_points:
                        self.rectify_single_frame()
                        self.number_grid_points_along_xs_line = num_points
                        self.spinbocXsLineNumPoints.setValue(num_points)

                        self.toolbuttonCreateXsLine.setChecked(True)
                        self.gridpreparation.add_line_of_given_length()
                        self.create_line_grid(mode="cross_section")
                        self.toolbuttonCreateXsLine.setChecked(False)
                        self.set_tab_icon("tabGridCreation", "good")

                # Handle results grid
                results_grid = project_dict.get("results_grid")
                if results_grid:
                    self.rectify_single_frame()
                    self.line_mode = project_dict.get("line_mode")

                    if self.cross_section_line_exists:
                        self.gridpreparation.add_line_of_given_length()
                        num_points = project_dict.get("number_grid_points_along_xs_line",
                                                      0)
                        self.spinbocXsLineNumPoints.setValue(num_points)
                        self.create_line_grid(mode="cross_section")

                # Handle horizontal grid spacing
                horz_grid_size = project_dict.get("horz_grid_size")
                if horz_grid_size is not None:
                    self.spinboxHorizGridSpacing.setValue(horz_grid_size)

            except (TypeError, AttributeError, KeyError) as e:
                logging.warning(f"Failed to load Grid Prep parameters: {e}")

            # Task 6 - STIV information
            try:
                # Set Keys that map directly to attributes and widgets
                stiv_params = [
                    ("stiv_dphi", "stiv_dphi", self.spinboxSTIVdPhi),
                    ("stiv_phi_origin", "stiv_phi_origin",
                     self.spinboxSTIVPhiOrigin),
                    ("stiv_phi_range", "stiv_phi_range",
                     self.spinboxSTIVPhiRange),
                    ("stiv_max_vel_threshold_mps", "stiv_max_vel_threshold_mps",
                     self.spinboxSTIVMaxVelThreshold),
                    ("stiv_gaussian_blur_sigma", "stiv_gaussian_blur_sigma",
                     self.doublespinboxStivGaussianBlurSigma),
                ]

                for key, attr, item in stiv_params:
                    value = project_dict.get(key)
                    if value is not None:
                        setattr(self, attr, value)
                        item.setValue(value)

                # Special handling for stiv_num_pixels since it has extra logic
                stiv_num_pixels = project_dict.get("stiv_num_pixels")
                stiv_search_line_length = project_dict.get(
                    "stiv_search_line_length_m")
                if stiv_num_pixels is not None and self.pixel_ground_scale_distance_m is not None:
                    self.stiv_num_pixels = stiv_num_pixels

                    if stiv_search_line_length is None:  # Will be meters
                        self.stiv_search_line_length_m = (
                                stiv_num_pixels * self.pixel_ground_scale_distance_m
                        )
                    else:
                        self.stiv_search_line_length_m = stiv_search_line_length

                    # Set in the UI
                    self.doublespinboxStivSearchLineDistance.setValue(
                        self.stiv_search_line_length_m * self.survey_units["L"]
                    )


            except (TypeError, AttributeError, KeyError) as e:
                logging.warning(f"Failed to load STIV parameters: {e}")

            # Task 7 - STIV Results
            try:
                # Load STIV Normals if available
                normals = project_dict.get("stiv_magnitude_normals")
                if normals is not None:
                    self.stiv.magnitude_normals_mps = deserialize_numpy_array(
                        normals)

                # Load STIV Magnitudes and Directions if available
                magnitudes_mps = project_dict.get("stiv_magnitudes")
                if magnitudes_mps is not None:
                    self.results_grid = deserialize_numpy_array(
                        project_dict["results_grid"])
                    self.stiv.magnitudes_mps = deserialize_numpy_array(magnitudes_mps)
                    self.stiv.directions = deserialize_numpy_array(
                        project_dict["stiv_directions"])

                    directions_rad = np.radians(self.stiv.directions)
                    X = self.results_grid[:, 0].astype(float)
                    Y = self.results_grid[:, 1].astype(float)

                    # Attempt to load from CSV (overrides the above if successful)
                    csv_file_path = os.path.join(
                        self.swap_velocities_directory, "stiv_results.csv")
                    csv_data = self._load_stiv_results_from_csv(csv_file_path)
                    if csv_data:
                        X = csv_data["X"]
                        Y = csv_data["Y"]

                    directions_deg_geo = self.stiv.directions

                    # Ensure the U,V components are in the correct quadrant
                    # orientation
                    U, V = calculate_uv_components(
                        self.stiv.magnitudes_mps, directions_deg_geo
                    )
                    M = np.sqrt(U**2 + V**2)

                    # Project the normals
                    (
                        vectors,
                        norm_vectors,
                        normal_unit_vector,
                        scalar_projections,
                        tagline_dir_geog_deg,
                        mean_flow_dir_geog_deg,
                    ) = compute_vectors_with_projections(X, Y, U, V)

                    # Clear existing vectors, then Plot the new vectors
                    # Check for NaN values in U and V arrays prior to plotting
                    nan_indices = np.logical_or(
                        np.isnan(vectors[:, 2]), np.isnan(vectors[:, 3])
                    )
                    self.stiv.imageBrowser.clearLines()
                    if np.all(nan_indices):
                        logging.warning(
                            "STIV: There are no valid velocities, check settings."
                        )
                        scalar_projections = np.zeros_like(M)
                    else:
                        vectors_draw = quiver(
                            vectors[~nan_indices, 0],
                            vectors[~nan_indices, 1],
                            vectors[~nan_indices, 2],
                            vectors[~nan_indices, 3],
                            global_scale=self.vector_scale,
                        )
                        norm_vectors_draw = quiver(
                            norm_vectors[~nan_indices, 0],
                            norm_vectors[~nan_indices, 1],
                            norm_vectors[~nan_indices, 2],
                            norm_vectors[~nan_indices, 3],
                            global_scale=self.vector_scale,
                        )
                        plot_quivers(self.stiv.imageBrowser, vectors_draw,
                                     "green", Qt.DotLine)
                        plot_quivers(self.stiv.imageBrowser, norm_vectors_draw,
                                     "yellow", Qt.SolidLine)
                    self.stiv.magnitude_normals_mps = np.abs(scalar_projections)
                    self.stiv_exists = True

                    self.set_qwidget_state_by_name("tabSTIVExhaustive", True)
                    self.set_tab_icon("tabImageVelocimetry", "good")
                    self.set_tab_icon(
                        "tabSTIVExhaustive",
                        "good",
                        self.tabWidget_ImageVelocimetryMethods,
                    )

                    # Load any STIs
                    if glob.glob(os.path.join(self.swap_image_directory, "STI*.jpg")):
                        thetas = project_dict.get("thetas", None)
                        if thetas is not None:
                            self.stiv.thetas = deserialize_numpy_array(thetas)
                        self.is_stis = True
                        sti_paths = glob.glob(
                            os.path.join(self.swap_image_directory, "STI*.jpg")
                        )
                        self.sti.table_load_data(sti_images=sti_paths)

                    # Add any manual STI streak angles
                    manual_sti_lines = project_dict.get("manual_sti_lines", None)
                    if manual_sti_lines is not None:
                        self.sti.manual_sti_lines = manual_sti_lines
                    manual_average_directions = project_dict.get(
                        "manual_average_directions", None
                    )
                    if manual_average_directions is not None:
                        self.sti.manual_average_directions = manual_average_directions
                        self.is_manual_sti_corrections = True
                        for row, average_direction in enumerate(
                            manual_average_directions
                        ):
                            if average_direction:
                                sti_path = sti_paths[row]
                                theta = self.stiv.thetas[row]
                                sti_pixmap = self.sti.draw_manual_lines_on_image(
                                    sti_path, theta, average_direction
                                )
                                sti_label = QtWidgets.QLabel()
                                sti_label.setPixmap(sti_pixmap)
                                self.sti.Table.setCellWidget(row, 1, sti_label)

                                # Set the manual velocity
                                # Convert to velocity
                                gsd = self.pixel_ground_scale_distance_m
                                dt = self.extraction_timestep_ms / 1000
                                manual_velocity = (
                                    -np.tan(np.deg2rad(average_direction)) * gsd / dt
                                )
                                new_item = QtWidgets.QTableWidgetItem(
                                    f"{manual_velocity * self.survey_units['V']:.2f}"
                                )
                                self.sti.Table.setItem(
                                    row, 5, QtWidgets.QTableWidgetItem(new_item)
                                )

                    # Load and STI Comments into the table
                    try:
                        if project_dict.get("sti_comments") is not None:
                            set_column_contents(
                                self.sti.Table,
                                column_index=6,
                                data=project_dict["sti_comments"],
                            )
                    except IndexError:
                        logging.warning(
                            f"OPEN PROJECT: Attempted to load "
                            f"STI comments from the project file "
                            f"but an invalid column index was "
                            f"passed."
                        )
                        self.set_tab_icon(
                            "tabSpaceTimeImageReview",
                            "good",
                            self.tabWidget_ImageVelocimetryMethods,
                        )

            except KeyError:
                pass

            # Task 8 - Discharge
            discharge_results = project_dict.get("discharge_results")
            if discharge_results is not None:
                self.dischargecomputaton.update_discharge_results()
                self.enable_disable_tabs(self.tabWidget, "tabDischarge", True)
                # If there are manual STI changes, they need to be applied
                if self.is_manual_sti_corrections:
                    self.sti.update_discharge_results_in_tab()

                # If we got here, enable the Report Button
                self.toolButtonExportPDF.setEnabled(True)


            # Update this associated with last process step
            process_step = project_dict.get("process_step")
            if process_step is not None:
                self.process_step = process_step
                if self.process_step == "Rectify Single Frame":
                    self.rectify_single_frame()
                if self.process_step == "Rectify Many Frames":
                    # Show result in the Grid Preparation tab
                    first_transformed_image = self.imagebrowser.sequence[0]

                    self.gridpreparation.imageBrowser.scene.load_image(
                        first_transformed_image
                    )
                    self.gridpreparation.imageBrowser.setEnabled(True)
                if (
                    self.process_step == "Process STIV Exhaustive"
                    or self.process_step == "Process STIV"
                ):  # Stay backwards compatible

                    pass
                if self.process_step == "Process STIV Optimized":
                    pass

            # Set to last used tab
            try:
                tab = project_dict["open_tab_index"]
                self.tabWidget.setCurrentIndex(tab)
            except KeyError:
                msg = f"OPEN PROJECT: Error setting last tab"
                logging.error(msg)

            # Disable the other Grid Prep methods
            to_disable = [
                "PointPage",
                "SimpleLinePage",
                "RegularGridPage",
            ]
            self.set_qwidget_state_by_name(to_disable, False)

            # Check for image_stack if there are vectors
            # Ask user if they want to make the stack
            image_stack_memory_map_exists = os.path.exists(
                os.path.join(self.swap_directory, "image_stack.dat")
            )
            if isinstance(self.stiv.image_stack, np.ndarray):
                image_stack_exists = True
            else:
                image_stack_exists = False
            stiv_exist = np.logical_or(
                "stiv_magnitudes" in project_dict, "stiv_opt_magnitudes" in project_dict
            )
            if stiv_exist and not image_stack_exists:
                message = (
                    f"Frames are extracted, but no Image Stack has been "
                    f"created. Would you like to create the Image Stack now?\n\n"
                    f"Note: An image stack is required before performing "
                    f"Image Velocimetry analyses."
                )
                result = self.warning_dialog(
                    "Create Image Stack?",
                    message,
                    "YesCancel",
                    icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
                )
                if result == "yes" and not self.image_processor_thread_is_running:
                    message = (
                        "ORTHORECTIFICATION: Creating Image Stack. Image "
                        "Velocimetry functions will be enabled when finished."
                    )
                    self.update_statusbar(message)
                    # Create an image_stack of the resultant frames
                    # Update the ImageProcessor instance with the arguments and run
                    logging.debug(
                        "orthorectify_many_thread_finished: About to "
                        "execute self.start_image_stack_process()"
                    )
                    self.start_image_stack_process()

                    # Disable the Export Projected Frames button while the stack
                    # process is running
                    # self.set_qwidget_state_by_name(
                    #     "pushbuttonExportProjectedFrames", False)
                if result == "cancel":
                    return

    def save_video_clip(self):
        """Executes when user clips save video clip. Prompts user for a save location for the video clip."""

        # Prompt user for a save file location
        try:
            ss = self.sticky_settings.get("last_video_clip_filename")
            self.video_clip_filename = ss
        except KeyError:
            self.video_clip_filename = f"{self.ffmpeg_parameters['output_video']}"
        except FileNotFoundError:
            # Create the settings file
            self.sticky_settings = Settings(self.ivy_settings_file)
        options = QtWidgets.QFileDialog.Options()
        self.video_clip_filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Video Clip",
            os.path.splitext(self.video_clip_filename)[0] + ".mp4",
            "All Files (*.*)",
            options=options,
        )
        if self.video_clip_filename:  # User did not hit cancel
            # Update the model with the selected filename
            self.video_model.video_clip_filename = self.video_clip_filename
            try:
                self.sticky_settings.set(
                    "last_video_clip_filename", self.video_clip_filename
                )
            except KeyError:
                self.sticky_settings.new(
                    "last_video_clip_filename", self.video_clip_filename
                )
            return True
        else:
            return False

    def create_summary_report(self):
        """Create a summary report for the current project"""
        # Prompt user for a save session file location and writes a JSON file containing all session information.
        try:
            ss = self.sticky_settings.get("last_export_pdf_file")
            export_filename = ss
        except KeyError:
            try:
                export_filename = (f"{QDir.homePath()}{os.sep}"
                                   f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')} - "
                                   f"{self.reporting.station_number} - "
                                   f"{self.reporting.station_name} - "
                                   f"IVy-Summary"
                                   f".pdf")
            except:
                export_filename = f"{QDir.homePath()}{os.sep}IVy Summary " f"Report.pdf"
        except FileNotFoundError:
            # Create the settings file
            self.sticky_settings = Settings(self.ivy_settings_file)

        options = QtWidgets.QFileDialog.Options()
        export_filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export IVy Project Summary Report",
            export_filename,
            "PDF (*.pdf)",
            options=options,
        )
        if export_filename:  # User did not hit cancel
            pdf = Report(export_filename, self)

            with self.wait_cursor():
                pdf.create()

    def save_project(self):
        """Save the current project as a zip archive"""
        # Grab a dict with all the user variables
        # if self.is_video_loaded:
        #     self.update_ffmpeg_parameters()
        project_dict = {
            key: value
            for (key, value) in zip(self.__dict__.keys(), self.__dict__.values())
            if type(value) == list
            or type(value) == str
            or type(value) == dict
            or type(value) == int
            or type(value) == float
            or type(value) == bool
            or value is None
        }

        # Fill in the missing gaps that are not caught from the __dict__ approach
        # Video

        # Reporting items
        project_dict["reporting"] = {
            "station_name": self.reporting.station_name,
            "station_number": self.reporting.station_number,
            "party": self.reporting.party,
            "weather": self.reporting.weather,
            "meas_date": (
                self.reporting.meas_date.toString("M/d/yyyy")
                if self.reporting.meas_date not in [None, ""]
                else ""
            ),
            "gage_ht": self.reporting.gage_ht,
            "start_time": (
                self.reporting.start_time.toString()
                if self.reporting.start_time not in [None, ""]
                else ""
            ),
            "end_time": (
                self.reporting.end_time.toString()
                if self.reporting.end_time not in [None, ""]
                else ""
            ),
            "mid_time": (
                self.reporting.mid_time.toString()
                if self.reporting.mid_time not in [None, ""]
                else ""
            ),
            "meas_number": self.reporting.meas_number,
            "project_description": self.reporting.project_description,
        }

        # Image Browser
        if self.imagebrowser.sequence is not None:
            project_dict["imagebrowser_sequence"] = [
                os.path.basename(item) for item in self.imagebrowser.sequence
            ]
            project_dict["imagebrowser_image_path"] = os.path.basename(
                self.imagebrowser.image_path
            )
            project_dict["sequence_index"] = self.imagebrowser.sequence_index

        if self.region_of_interest_pixels is not None:
            project_dict["region_of_interest_pixels"] = [
                serialize_numpy_array(arr) for arr in self.region_of_interest_pixels
            ]

        # Orthorectification
        if self.ortho_original_image.has_image():
            project_dict["last_ortho_gcp_image_path"] = (
                self.ortho_original_image.image_file_path
            )
        project_dict["water_surface_elevation"] = self.ortho_rectified_wse_m
        project_dict["orthotable_dataframe"] = self.orthotable_dataframe.to_dict()
        project_dict["rectification_parameters"] = dict_arrays_to_list(
            project_dict["rectification_parameters"]
        )
        project_dict[
            "pixel_ground_scale_distance_m"] = self.pixel_ground_scale_distance_m

        # Grid Preparation
        if self.results_grid is not None:
            project_dict["results_grid"] = serialize_numpy_array(self.results_grid)
        if self.results_grid_world is not None:
            project_dict["results_grid_world"] = serialize_numpy_array(
                self.results_grid_world
            )
        if self.gridpreparation.imageBrowser.polygons_ndarray() is not None:
            project_dict["mask_polygons"] = [
                serialize_numpy_array(polygon)
                for polygon in self.gridpreparation.imageBrowser.polygons_ndarray()
            ]
        else:
            project_dict["mask_polygons"] = None

        # Area Comp Related
        project_dict["bathymetry_ac3_filename"] = self.bathymetry_ac3_filename
        project_dict["cross_section_top_width_m"] = (
            self.cross_section_top_width_m
        )
        try:
            project_dict["cross_section_line"] = serialize_numpy_array(
                self.cross_section_line
            )
        except ValueError:
            pass

        # Image velocimetry
        # STIV
        if self.stiv_search_line_length_m is not None:
            project_dict[
                "stiv_search_line_length_m"] = self.stiv_search_line_length_m
        if self.stiv.magnitudes_mps is not None:
            project_dict["stiv_magnitudes"] = serialize_numpy_array(
                self.stiv.magnitudes_mps
            )
        if self.stiv.directions is not None:
            project_dict["stiv_directions"] = serialize_numpy_array(
                self.stiv.directions
            )
        if self.stiv.magnitude_normals_mps is not None and np.any(
            self.stiv.magnitude_normals_mps
        ):
            project_dict["stiv_magnitude_normals"] = serialize_numpy_array(
                self.stiv.magnitude_normals_mps
            )
        if self.stiv.thetas is not None:
            project_dict["thetas"] = serialize_numpy_array(self.stiv.thetas)
        if self.stiv_opt.magnitudes_mps is not None:
            project_dict["stiv_opt_magnitudes"] = serialize_numpy_array(
                self.stiv_opt.magnitudes_mps
            )
        if self.stiv_opt.directions is not None:
            project_dict["stiv_opt_directions"] = serialize_numpy_array(
                self.stiv_opt.directions
            )

        # Manual STI changes
        if self.sti.manual_sti_lines:  # True is there are EPs
            project_dict["manual_sti_lines"] = self.sti.manual_sti_lines
            project_dict["manual_average_directions"] = (
                self.sti.manual_average_directions
            )
        else:
            project_dict["manual_sti_lines"] = []
            project_dict["manual_average_directions"] = []

        # STI Comments, contained in column 6
        try:
            project_dict["sti_comments"] = get_column_contents(
                self.sti.Table, column_index=6
            )
        except IndexError:
            logging.warning(
                f"SAVE PROJECT: Attempted to save STI comments, "
                f"but an invalid column index was passed."
            )

        # Discharge
        if self.discharge_results:
            project_dict["discharge_results"] = self.discharge_results
        if self.discharge_summary:
            project_dict["discharge_summary"] = self.discharge_summary
        if self.dischargecomputaton.is_table_loaded:
            # Save a System comment to the project file
            dd = self.dischargecomputaton.ivy_framework.discharge_summary
            time_stamp = datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S")
            msg = (
                f"{self.__user__}, {time_stamp}, File Saved Q = "
                f"{dd['total_discharge'] * self.survey_units['Q']:.2f} "
                f"{self.survey_units['label_Q']} "
                f"(ISO Uncert.) {dd['ISO_uncertainty']*100:.2f}%"
            )
            self.comments.append_comment(key="System", comment=msg)
            self.update_comment_tbl()

        # Other
        project_dict["open_tab_index"] = self.tabWidget.currentIndex()
        _ = project_dict.pop("tree_views")
        project_dict["comments"] = self.comments.comments

        # Sort project dict (recreate a sorted dict)
        sorted_keys = sorted(project_dict.keys())
        project_dict = {key: project_dict[key] for key in sorted_keys}

        # Prompt user for a save session file location and writes a JSON file containing all session information.
        try:
            ss = self.sticky_settings.get("last_project_filename")
            self.project_filename = ss
        except KeyError:
            self.project_filename = f"{QDir.homePath()}{os.sep}New_IVy_Project.ivy"
        except FileNotFoundError:
            # Create the settings file
            self.sticky_settings = Settings(self.ivy_settings_file)
        options = QtWidgets.QFileDialog.Options()
        self.project_filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save IVy Project file",
            os.path.splitext(self.project_filename)[0] + ".ivy",
            "IVy Files (*.ivy)",
            options=options,
        )
        if self.project_filename:  # User did not hit cancel
            # Step 1: Write the data to a JSON file in the swap directory
            json_filename = os.path.join(self.swap_directory, "project_data.json")
            project_dict["project_filename"] = json_filename

            try:
                self.project_service.save_project_to_json(project_dict, json_filename)
            except (IOError, ValueError) as e:
                self.warning_dialog(
                    "Error Saving Project",
                    f"Failed to save project JSON: {str(e)}",
                    style="ok",
                    icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
                )
                logging.error(f"Error saving project JSON: {str(e)}")
                return

            # Step 2: Create a zip archive of the swap directory
            message = (
                f"SAVING PROJECT: Saving the current project file, "
                f"please be patient."
            )
            self.update_statusbar(message)
            self.progressBar.show()
            self.progressBar.setValue(0)

            # Progress callback for archive creation
            def update_progress(current, total):
                progress = int(100 * current / total) if total > 0 else 0
                self.progressBar.setValue(progress)

            with self.wait_cursor():
                try:
                    self.project_service.create_project_archive(
                        self.swap_directory,
                        self.project_filename,
                        progress_callback=update_progress,
                        exclude_extensions=[".dat"]
                    )
                except (IOError, FileNotFoundError) as e:
                    # Handle the exception, display a warning dialog, and log the error
                    self.warning_dialog(
                        "Error Saving Project",
                        f"An error occurred while saving the project: {str(e)}",
                        style="ok",
                        icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
                    )
                    logging.error(f"Error saving project: {str(e)}")
                    return
            self.progressBar.hide()
            message = (
                f"SAVING PROJECT: Successfully saved project file: "
                f"{format_windows_path(self.project_filename)}"
            )
            self.update_statusbar(message)
            try:
                self.sticky_settings.set("last_project_filename", self.project_filename)
            except KeyError:
                self.sticky_settings.new("last_project_filename", self.project_filename)
            logging.info(f"Saved project file: {self.project_filename}")

    def clear_project(self):
        """Clear the current project"""
        self.new_project()

    def open_video(self):
        """Open and play a video - delegates to video controller."""
        self.video_controller.open_video_dialog()

    def set_video_metadata(self, video_metadata):
        """Sets video metadata and updates related UI elements.

        Parameters
        ----------
        video_metadata : dict
            A dictionary containing metadata about the video.
            Expected keys include:
            - "duration" : float
                Duration of the video in milliseconds.
            - "width" : int
                Width of the video in pixels.
            - "height" : int
                Height of the video in pixels.
            - "avg_frame_rate" : float
                Average frame rate of the video in frames per second (fps).
            - "frame_count" : int
                Total number of frames in the video.
            - "avg_timestep_ms" : float
                Average time step between frames in milliseconds.

        Attributes Set
        --------------
        video_duration : float
            Duration of the video in milliseconds.
        video_resolution : str
            Video resolution in the format "widthxheight" (e.g.,
            "1920x1080").
        video_frame_rate : float
            Average frame rate of the video in fps.
        video_num_frames : int
            Total number of frames in the video.
        video_timestep_ms : float
            Average time step between frames in milliseconds.
        extraction_frame_rate : float
            Frame rate used for frame extraction, initialized to
            `video_frame_rate`.
        extraction_timestep_ms : float
            Time step for frame extraction, initialized to
            `video_timestep_ms`.
        extraction_video_file_name : str
            Name of the video file, initialized to `self.video_file_name`.

        Updates UI Elements
        -------------------
        Updates various UI elements, such as labels and group boxes, to
        display the video metadata values, including resolution,
        frame rate, timestep, and frame count.

        Notes
        -----
        The function also calculates derived values, such as start and
        end frames, based on the video duration and frame rate. It uses
        placeholders for lens correction parameters (K1 and K2) and
        hides certain elements, like video preload labels.

        See Also
        --------
        seconds_to_frame_number : Helper function to compute frame
        numbers from time.
        """
        # Store metadata first
        self.video_metadata = video_metadata

        self.video_duration = video_metadata["duration"]  # In milliseconds
        self.video_resolution = "{}x{}".format(
            video_metadata["width"], video_metadata["height"]
        )
        self.video_frame_rate = video_metadata["avg_frame_rate"]
        self.video_num_frames = video_metadata["frame_count"]
        self.video_timestep_ms = video_metadata["avg_timestep_ms"]
        self.extraction_frame_rate = self.video_frame_rate
        self.extraction_timestep_ms = self.video_timestep_ms
        self.extraction_video_file_name = self.video_file_name
        self.groupboxFrameExtraction.setEnabled(True)
        self.labelVideoFramerateValue.setText(f"{self.video_frame_rate:.3f} fps")
        self.labelVideoTimestepValue.setText(f"{self.video_timestep_ms:.4f} ms")
        self.labelNumOfFramesValue.setText(f"{self.video_num_frames:d}")
        self.labelVideoResolutionValue.setText(f"{self.video_resolution} px")
        self.labelStartFrameValue.setText("0")
        self.labelEndFrameValue.setText(
            f"{seconds_to_frame_number(float(self.video_duration), self.video_frame_rate)}"
        )
        self.labelNewFrameRateValue.setText(f"{self.extraction_frame_rate:.3f} fps")
        self.labelNewTimestepValue.setText(f"{self.extraction_timestep_ms:.4f} ms")
        self.labelNewNumFramesValue.setText(f"{self.video_num_frames}")
        # Use parameter instead of self.video_metadata
        self.labelLensCxValue.setText(f"{video_metadata['width'] / 2:.3f}")
        self.labelLensCyValue.setText(f"{video_metadata['height'] / 2:.3f}")
        self.labelLensK1Value.setText("0.000")
        self.labelLensK2Value.setText("0.000")
        self.labelVideoPreload.setHidden(True)

    def parse_video(self):
        """Parse video, load metadata, and start playing"""
        self.is_clip_created = False
        self.is_frames_extracted = False
        self.video_metadata = opencv_get_video_metadata(
            self.video_file_name,
            status_callback=self.signal_opencv_updates.emit
        )

        self.video_metadata = ffprobe_add_exif_metadata(
            self.video_file_name,
            self.video_metadata
        )

        self.set_video_metadata(self.video_metadata)
        logging.info("Video Metadata:")
        logging.info(json.dumps(self.video_metadata, sort_keys=False, indent=4))

        # At this point, I can populate the Reporting Tab
        creation_time = self.video_metadata.get("exif_creation_time", None)
        duration = self.video_metadata.get("duration", None)
        if creation_time is not None:
            dt = parse_creation_time(creation_time)

        else:  # Fallback: Try to smart parse the file name
            dt = parse_creation_time(os.path.basename(
                self.video_file_name))

        if dt is not None:
            # Set date
            self.measDate.setDate(
                QDate(dt.year, dt.month, dt.day))

            # Set start time (triggers self.start_time_change)
            self.measStartTime.setTime(
                QTime(dt.hour, dt.minute, dt.second))

            # Compute and set end time (triggers self.end_time_change)
            dt_end = dt + datetime.timedelta(seconds=duration)
            self.measEndTime.setTime(
                QTime(dt_end.hour, dt_end.minute, dt_end.second))


            # Warn the user to check times
            # Not sure if I want this dialog here or not. It locks the
            # UI until the user clicks OK.
            # msg = (f"Video date and duration was found in the file "
            #        f"metadata. IVy has attempted to load this "
            #        f"information into the Reporting Tab.\n\nIVy can make "
            #        f"mistakes, please check for accuracy and correct if "
            #        f"needed.")
            # self.warning_dialog(
            #     title="Video Metadata",
            #     message=msg,
            #     style="ok",
            #     icon=os.path.join(self.__icon_path__, "IVy_logo.ico"),
            # )
        else:
            self.measStartTime.setTime(QTime(0, 0))
            self.measEndTime.setTime(QTime(0, 0))






        self.video_working_path = self.swap_image_directory
        # self.video_working_path = os.path.splitext(self.video_file_name)[0]
        self.video_player.setMedia(
            QMediaContent(QUrl.fromLocalFile(self.video_file_name))
        )


        self.video_player.setVideoOutput(self.widgetVideo)
        self.buttonPlay.setEnabled(True)
        self.sliderVideoPlayHead.setEnabled(True)
        self.video_player.play()
        logging.debug(f"Playing {self.video_file_name}")

        self.tabWidget.setCurrentIndex(0)  # Activate the video tab upon playing video
        self.is_video_loaded = True

        to_enable = [
            "groupboxClipControl",
            "groupboxClipCreationControls",
            "buttonPlay",
            "sliderVideoPlayHead",
            "groupboxFramePreparation",
        ]
        self.set_qwidget_state_by_name(to_enable, True)
        # self.set_qwidget_state_by_name("buttonLoadLensCharacteristics", False)
        message = f"Playing: {os.path.basename(self.video_file_name)}"
        self.set_menu_item_color("actionOpen_Video", "good")
        self.update_statusbar(message)

    def play_video(self):
        """Controls the Play button in the video control panel"""
        if self.video_player.state() == QMediaPlayer.PlayingState:
            self.video_player.pause()
            logging.debug("Video paused.")
        else:
            self.video_player.play()
            logging.debug("Video playing.")

    def video_position_changed(self, position):
        """Tracks video position and updates playhead and time labels"""
        self.sliderVideoPlayHead.setValue(position)
        self.labelVideoPlayheadTime.setText(
            float_seconds_to_time_string(float(position / 1000), precision="second")
            + f" [{seconds_to_frame_number(float(position / 1000), self.video_frame_rate)}]"
        )

    def video_duration_changed(self, duration):
        """When video is loaded, tracks duration and updates slider/time labels"""
        self.sliderVideoPlayHead.setRange(0, duration)
        self.video_duration = duration
        self.video_clip_start_time = 0
        self.video_clip_end_time = 0
        try:
            self.labelVideoDuration.setText(
                float_seconds_to_time_string(float(duration / 1000), precision="second")
                + f" [{seconds_to_frame_number(float(duration / 1000), self.video_frame_rate)}]"
            )
        except:
            pass

    def media_state_changed(self):
        """Changes play button icon to play/pause depending on video player state"""
        if self.video_player.state() == QMediaPlayer.PlayingState:
            self.buttonPlay.setIcon(
                QtGui.QIcon(
                    resource_path(self.__icon_path__ + os.sep + "pause-solid.svg")
                )
            )
        else:
            self.buttonPlay.setIcon(
                QtGui.QIcon(
                    resource_path(self.__icon_path__ + os.sep + "play-solid.svg")
                )
            )

    def video_set_position(self, position):
        """Moves video position to match the playhead position"""
        self.video_player.setPosition(position)

    def video_error_handling(self):
        """Handles video errors on load"""
        self.buttonPlay.setEnabled(False)
        logging.error("There was a problem loading the supplied video.")
        message = f"ERROR: Cannot open the supplied video."
        self.update_statusbar(message)

    def frame_step_changed(self):
        """Update the frame step when user changes it."""
        new_frame_step = int(self.lineeditFrameStepValue.text())
        self.extraction_frame_step = new_frame_step
        self._show_clip_information()
        # Calc new frame rate and timestep given start/end frame

    def update_ffmpeg_parameters(self):
        """Update the ffmpeg parameters to match the current gui settings"""
        # Read from VideoModel for video state
        video_clip_end_time = self.video_model.video_clip_end_time
        video_clip_start_time = self.video_model.video_clip_start_time
        video_file_name = self.video_model.video_file_name
        video_duration = self.video_model.video_duration

        # Set default end time if not specified
        if video_clip_end_time == 0 or video_clip_end_time is None:
            video_clip_end_time = video_duration

        # Get current settings from UI and update model
        video_rotation = int(self.comboboxFfmpegRotation.currentText())
        video_flip = self.comboboxFfmpegFlipVideo.currentText()
        video_strip_audio = self.checkboxStripAudio.isChecked()
        video_normalize_luma = self.checkboxFfmpegNormalizeLuma.isChecked()
        video_curve_preset = self.comboboxFfmpegCurvePresets.currentText()
        video_ffmpeg_stabilize = self.checkboxFfmpeg2PassStabilization.isChecked()

        # Update model with current UI settings
        self.video_model.video_rotation = video_rotation
        self.video_model.video_flip = video_flip
        self.video_model.video_strip_audio = video_strip_audio
        self.video_model.video_normalize_luma = video_normalize_luma
        self.video_model.video_curve_preset = video_curve_preset
        self.video_model.video_ffmpeg_stabilize = video_ffmpeg_stabilize

        # CRITICAL: Also update ivy.py's own properties for stabilization workflow
        # The stabilization setup functions check self.video_ffmpeg_stabilize
        self.video_rotation = video_rotation
        self.video_flip = video_flip
        self.video_strip_audio = video_strip_audio
        self.video_normalize_luma = video_normalize_luma
        self.video_curve_preset = video_curve_preset
        self.video_ffmpeg_stabilize = video_ffmpeg_stabilize

        # Use VideoService to generate output filename
        # Determine output directory
        video_clip_filename = self.video_model.video_clip_filename
        if video_clip_filename:
            output_video = video_clip_filename
        else:
            video_dir = os.path.dirname(
                video_file_name) if video_file_name else self.swap_directory
            output_video = self.video_service.generate_clip_filename(
                input_video_path=video_file_name or "no_video_loaded.mp4",
                start_time_ms=video_clip_start_time,
                end_time_ms=video_clip_end_time,
                output_dir=video_dir,
                rotation=video_rotation,
                flip=video_flip,
                normalize_luma=video_normalize_luma,
                curve_preset=video_curve_preset,
                stabilize=video_ffmpeg_stabilize
            )

        # Use VideoService to build FFmpeg parameters
        self.ffmpeg_parameters = self.video_service.build_ffmpeg_parameters(
            input_video=video_file_name,
            output_video=output_video,
            start_time_ms=video_clip_start_time,
            end_time_ms=video_clip_end_time,
            rotation=video_rotation,
            flip=video_flip,
            strip_audio=video_strip_audio,
            normalize_luma=video_normalize_luma,
            curve_preset=video_curve_preset,
            stabilize=video_ffmpeg_stabilize,
            extract_frames=False,
            extract_frame_step=self.extraction_frame_step,
            extracted_frames_folder=self.extracted_frames_folder,
            extract_frame_pattern="f%05d.jpg",
            calibrate_radial=self.checkboxCorrectRadialDistortion.isChecked(),
            cx=self.lens_characteristics.cx_dim,
            cy=self.lens_characteristics.cy_dim,
            k1=self.lens_characteristics.k1,
            k2=self.lens_characteristics.k2,
        )

        logging.debug("FFMPEG parameters changed. New parameters:")
        logging.debug(json.dumps(self.ffmpeg_parameters, indent=4))

    def update_statusbar(self, message):
        """Update the statusbar with supplied text"""
        self.status_message = message
        self.statusbarMainWindow.showMessage(message)
        logging.debug(f"New Statusbar Message: {message}")

    def update_progress_bar_qthreads(self, progress: int):
        """Track QThreadPool Worker progress  and update the progress.


        This is the progress callback for items that are run with the Worker class.
        This function is NOT associated wit htracking progress for QProcess calls (e.g.
        the ffmpeg or imagemagick functionality).
        """
        self.progressBar.setValue(progress)

    def dragEnterEvent(self, event):
        """Drag and drop file support. Checks if the dragged file is of acceptable type."""
        filter = [".mp4", ".mov", ".wmv", ".avi", ".mkv"]
        event_file_name_url = event.mimeData().text()
        file_ext = os.path.splitext(event_file_name_url)[-1]
        if file_ext.lower() in filter:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Drag and drop file support. If dragged event accepted, load the video."""
        file_uri = event.mimeData().text()
        p = urlparse(file_uri)
        video_file_path = os.path.join(p.netloc, p.path)[1:]
        try:
            self.sticky_settings.set("last_video_file_name", video_file_path)
        except KeyError:  # key didn't exist, create it
            self.sticky_settings.new("last_video_file_name", video_file_path)
        # Use video controller to load the video
        with self.wait_cursor():
            self.video_controller.load_video(video_file_path)

    def set_clip_start_time(self):
        """When Clip Start Time button is pressed, log the playhead position"""
        if self.is_video_loaded:
            start_time = self.sliderVideoPlayHead.sliderPosition()
            end_time = self.video_clip_end_time
            if end_time is not None and end_time > 0 and start_time > end_time:
                start_time = end_time  # Cannot have a start time after end of video
            self.video_clip_start_time = start_time
            self.video_clip_end_time = end_time
            self.buttonClipStart.setText(
                f"Clip Start [{seconds_to_hhmmss(start_time / 1000, precision='high')}]"
            )
            start_frame = seconds_to_frame_number(
                self.video_clip_start_time / 1000, self.video_frame_rate
            )
            self.labelStartFrameValue.setText(f"{start_frame}")
            self._show_clip_information()

    def set_clip_end_time(self):
        """When Clip End Time button is pressed, log the playhead position"""
        if self.is_video_loaded:
            end_time = self.sliderVideoPlayHead.sliderPosition()
            start_time = self.video_clip_start_time
            if start_time is not None and start_time > end_time:
                start_time = end_time  # Cannot have a start time after end of video
            self.video_clip_start_time = start_time
            self.video_clip_end_time = end_time
            self.buttonClipEnd.setText(
                f"Clip End [{seconds_to_hhmmss(end_time / 1000, precision='high')}]"
            )
            end_frame = seconds_to_frame_number(
                self.video_clip_end_time / 1000, self.video_frame_rate
            )
            self.labelEndFrameValue.setText(f"{end_frame}")
            self._show_clip_information()

    def clear_clip_start_end_times(self):
        """Clear the clip start and end times when user clicks the Clear Clip button."""
        self.video_clip_start_time = 0
        self.video_clip_end_time = 0
        self.buttonClipStart.setText(f"Clip Start")
        self.buttonClipEnd.setText(f"Clip End")
        self.labelStartFrameValue.setText(f"0")
        self.labelEndFrameValue.setText(f"{self.video_num_frames}")
        if self.is_video_loaded:
            self._show_clip_information()

    def _show_clip_information(self):
        """Update the statusbar with the clip information"""
        if (
            self.video_clip_start_time is not None
            and self.video_clip_end_time is not None
        ):
            start_time_str = float_seconds_to_time_string(
                self.video_clip_start_time / 1000, precision="hundredth"
            )
            end_time_str = float_seconds_to_time_string(
                self.video_clip_end_time / 1000, precision="hundredth"
            )
            duration_sec = (
                self.video_clip_end_time - self.video_clip_start_time
            ) / 1000
            if duration_sec < 0:
                duration_sec = 0
            duration_sec_str = f"{duration_sec:.2f}"
            message = (
                f"Selected Clip Information: "
                f"Start time: [{start_time_str}] | "
                f"End time: [{end_time_str}] | "
                f"Duration: [{duration_sec_str} seconds]"
            )
            start_frame = seconds_to_frame_number(
                self.video_clip_start_time / 1000, self.video_frame_rate
            )
            end_frame = seconds_to_frame_number(
                self.video_clip_end_time / 1000, self.video_frame_rate
            )
            if end_frame == 0:
                end_frame = self.video_num_frames
            self.extraction_frame_rate = (
                self.video_frame_rate / self.extraction_frame_step
            )
            self.extraction_timestep_ms = 1 / self.extraction_frame_rate * 1000
            self.extraction_num_frames = int(
                (end_frame - start_frame) / self.extraction_frame_step
            )
            self.labelNewFrameRateValue.setText(f"{self.extraction_frame_rate:.3f} fps")
            self.labelNewTimestepValue.setText(f"{self.extraction_timestep_ms:.4f} ms")
            self.labelNewNumFramesValue.setText(f"{self.extraction_num_frames}")
            self.update_statusbar(message)

    def video_process_monitor(self):
        """Configure settings based on current process step."""
        if self.process_step == "create_video_clip":
            self.update_ffmpeg_parameters()
            # Read clip filename from model
            self.extraction_video_file_name = self.video_model.video_clip_filename
            self.update_ffmpeg_parameters()
            self.ffmpeg_thread_is_running = False
            self.ffmpeg_thread_is_finished = False
        if self.process_step == "extract_frames":
            self.update_ffmpeg_parameters()
            self.stabilize_step1_finished = False
            self.stabilize_step2_finished = False
            basename = f"{self.swap_image_directory}{os.sep}{os.path.splitext(os.path.basename(self.ffmpeg_parameters['output_video']))[0]}"
            self.extracted_frames_folder = safe_make_directory(
                self.swap_image_directory, overwrite=True
            )
            self.update_ffmpeg_parameters()
            self.ffmpeg_parameters["extract_frames"] = True
            logging.debug("ffmpeg_parameters['extract_frames'] set to True")
            self.ffmpeg_thread_is_running = False
            self.ffmpeg_thread_is_finished = False

    def create_video_clip(self):
        """Create the selected video clip. Drives the ffmpeg thread."""
        if self.is_video_loaded:
            self.process_step = "create_video_clip"
            status = self.save_video_clip()
            if status:
                self.video_process_monitor()
                self.progressBar.setValue(0)
                self.ffmpeg_run_clip_generation_process()
                message = f"PROCESSING: Creating sub-clip, please be patient"
                self.update_statusbar(message)
                self.progressBar.show()

    def extract_frames(self):
        """Extract select frames from a video. Drives the ffmpeg thread."""
        if self.is_video_loaded:
            # self.update_ffmpeg_parameters()
            self.process_step = "extract_frames"
            self.video_process_monitor()
            self.progressBar.setValue(0)
            self.ffmpeg_run_frame_extraction_process()
            message = f"PROCESSING: Extracting video frames, please be patient"
            self.update_statusbar(message)
            self.progressBar.show()

            # Write metadata to a file
            info_file = (
                self.ffmpeg_parameters["extracted_frames_folder"]
                + os.sep
                + "!frame_extraction_information.txt"
            )
            extract_info = {
                "source_video_file": self.ffmpeg_parameters["input_video"],
                "clip_video_file": self.ffmpeg_parameters["output_video"],
                "processed_date": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "user": os.environ["COMPUTERNAME"] + os.sep + os.environ["USERNAME"],
                "start_time": self.ffmpeg_parameters["start_time"],
                "start_frame": seconds_to_frame_number(
                    self.video_clip_start_time / 1000, self.extraction_frame_rate
                ),
                "end_time": self.ffmpeg_parameters["end_time"],
                "end_frame": seconds_to_frame_number(
                    self.video_clip_end_time / 1000, self.extraction_frame_rate
                ),
                "frame_step": self.ffmpeg_parameters["extract_frame_step"],
                "frames_location": self.ffmpeg_parameters["extracted_frames_folder"],
                "frame_name_pattern": self.ffmpeg_parameters["extract_frame_pattern"],
                "original_video_metadata": self.video_metadata,
            }
            with open(info_file, "w") as f:
                json.dump(extract_info, f, indent=4)

    def correct_radial_distortion(self):
        """Enable/disable lens corrections"""
        if self.checkboxCorrectRadialDistortion.isChecked():
            self.buttonLoadLensCharacteristics.setEnabled(True)
        else:
            self.buttonLoadLensCharacteristics.setEnabled(False)

    def get_lens_characteristics(self):
        """Grab the supplied lens characteristics from the sub-gui"""
        logging.debug("##### Opening lens_characteristics to get lens characteristics")
        if self.is_video_loaded:
            cx, cy, k1, k2 = (
                float(self.labelLensCxValue.text()),
                float(self.labelLensCyValue.text()),
                float(self.labelLensK1Value.text()),
                float(self.labelLensK2Value.text()),
            )
            self.lens_characteristics = LensCharacteristics(
                self,
                width=self.video_metadata["width"],
                height=self.video_metadata["height"],
                cx=cx,
                cy=cy,
                k1=k1,
                k2=k2,
            )
            res = self.lens_characteristics.exec_()

            self.labelLensCxValue.setText(f"{self.lens_characteristics.cx}")
            self.labelLensCyValue.setText(f"{self.lens_characteristics.cy}")
            self.labelLensK1Value.setText(f"{self.lens_characteristics.k1}")
            self.labelLensK2Value.setText(f"{self.lens_characteristics.k2}")
            self.update_ffmpeg_parameters()

    def ffmpeg_onready_read_stderr(self):
        """Parse the ffmpeg standard error message. This is used to grab job/thread status information."""
        data = self.ffmpeg_process.readAllStandardError()
        stderr = bytes(data).decode("utf8")
        logging.debug(f"{time.time()}: {stderr}")

        # Parse the ffmpeg output, if there's a time then it is processing
        # Use this to get a percentage and update the progressbar
        current_progress = parse_ffmpeg_stdout_progress(
            stderr, video_duration=self.video_duration / 1000
        )
        if current_progress is not None:
            self.progressBar.setValue(current_progress)
        self.signal_stderr.emit(stderr)

    def ffmpeg_onready_read_stdout(self):
        """Parse the ffmpeg standard out message."""
        data = self.ffmpeg_process.readAllStandardOutput()
        stout = bytes(data).decode("utf8")
        logging.debug("{time.time()}: {stout}")
        self.signal_stdout.emit(stout)

    def ffmpeg_run_clip_generation_process(self):
        """Ffmpeg thread watcher for clip generation."""
        if (
            not self.ffmpeg_thread_is_running
        ):  # Don't allow starting a new thread if one is already running
            self.ffmpeg_thread_is_running = True
            self.ffmpeg_process.start(create_ffmpeg_command(self.ffmpeg_parameters))

    def ffmpeg_run_frame_extraction_process(self):
        """Ffmpeg thread watcher for frame extraction."""
        if (
            not self.ffmpeg_thread_is_running
        ):  # Don't allow starting a new thread if one is already running
            self.ffmpeg_thread_is_running = True
            self.ffmpeg_parameters["output_video"] = (
                "null -",
            )  # ffmpeg decode without writing a file
            logging.debug(
                f"FFMPEG Extract Frames command: {create_ffmpeg_command(self.ffmpeg_parameters)}"
            )
            self.ffmpeg_process.start(create_ffmpeg_command(self.ffmpeg_parameters))

    def ffmpeg_run_stabilization_pass1(self):
        """Ffmpeg thread watcher for pass 1 stabilization."""
        if (
            not self.ffmpeg_thread_is_running
        ):  # Don't allow starting a new thread if one is already running
            self.ffmpeg_thread_is_running = True
            self.ffmpeg_process.start(
                ffmpeg_compute_motion_trajectories_from_frames_command(
                    self.extracted_frames_folder
                )
            )
            self.stabilize_step1_finished = True

    def ffmpeg_run_stabilization_pass2(self):
        """Ffmpeg thread watcher for pass 2 stabilization."""
        if (
            not self.ffmpeg_thread_is_running
        ):  # Don't allow starting a new thread if one is already running
            self.ffmpeg_thread_is_running = True
            self.ffmpeg_process.start(
                ffmpeg_remove_motion_from_frames_command(self.extracted_frames_folder)
            )
            self.stabilize_step2_finished = True

    def ffmpeg_thread_finished(self):
        """Ffmpeg thread watcher."""
        self.ffmpeg_thread_is_finished = True
        self.ffmpeg_thread_is_running = False
        message = f"PROCESSING: Complete ({self.process_step})"
        self.update_statusbar(message)

        prompt_to_load_gcp_image = False

        if (
            self.process_step == "stabilize_pass2" and self.stabilize_step2_finished
        ):  # If we are here, stabilization just finished
            logging.debug("Stabilization complete. Create a stabilization check image.")
            file_pattern = "f*.jpg"
            processed_frames = glob.glob(
                f"{self.extracted_frames_folder}" f"/{file_pattern}"
            )
            self.stabilization_check_image_path = create_change_overlay_image(
                [processed_frames[0], processed_frames[-1]]
            )
            prompt_to_load_gcp_image = True

        # Extract frames is complete, so do any last minute tasks
        if (
            self.process_step == "extract_frames"
            or self.process_step == "stabilize_pass2"
        ):
            self._show_clip_information()
            logging.debug(
                "Frames were extracted, attempt to load them into Image Browser"
            )
            self.imagebrowser.folder_path = self.ffmpeg_parameters[
                "extracted_frames_folder"
            ]
            self.imagebrowser.reload = (
                True  # Ensures the frame loader will not prompt user for a folder
            )
            self.imagebrowser.open_image_folder()
            self.imagebrowser.reload = False
            self.set_qwidget_state_by_name(
                [
                    "groupboxTools",
                    "groupboxProcessing",
                ],
                True,
            )
            if not self.checkboxFfmpeg2PassStabilization.isChecked():
                prompt_to_load_gcp_image = True

        self.progressBar.setValue(
            100
        )  # complete the progress bar if it wasn't completed already
        self.progressBar.hide()
        self.signal_ffmpeg_thread.emit(True)

        # Update the tab colors
        self.set_tab_icon("tabVideoPreProcessing", "good")
        self.set_tab_icon("tabImageFrameProcessing", "good")
        self.set_button_color("buttonExtractVideoFrames", "good")

        # Enable next tab(s)
        self.enable_disable_tabs(self.tabWidget, "tabImageFrameProcessing")
        self.enable_disable_tabs(self.tabWidget, "tabOrthorectification")

        if prompt_to_load_gcp_image:
            # Raise a dialog box asking if the user would like to
            # load a ground control image
            choices = ("Yes", "No", "Let me choose the image")
            result = self.custom_dialog_index(
                title="Frame extraction complete",
                message=f"Video frame extraction complete.\n\n"
                f"Would you like to select the first frame as the "
                f"Ground Control Image?",
                choices=choices,
            )
            logging.debug(
                f"Extract frames completed. User choice on GCP Image "
                f"Dialog: {result} | {choices[result]}"
            )
            if choices[result].lower() == "yes":
                try:
                    # Put image into the viewer
                    self.ortho_original_load_gcp_image(self.imagebrowser.image_path)
                    self.sticky_settings.set(
                        "last_ortho_gcp_image_path",
                        self.ortho_original_image.image_file_path,
                    )
                except KeyError:
                    self.sticky_settings.new(
                        "last_ortho_gcp_image_path",
                        self.ortho_original_image.image_file_path,
                    )
            elif choices[result].lower() == "no":
                pass
            elif choices[result].lower == "let me choose the image":
                self.ortho_original_load_gcp_image()

    def ffmpeg_setup_stabilization_pass1(self):
        """Ffmpeg pass 1 stabilization."""
        if self.ffmpeg_thread_is_finished:
            if self.video_ffmpeg_stabilize:  # User wants to stabilize
                if not self.stabilize_step1_finished:
                    self.process_step = "stabilize_pass1"
                    logging.debug("##### Starting ffmpeg thread for Pass 1")
                    self.progressBar.setValue(0)
                    self.ffmpeg_thread_is_running = False
                    self.ffmpeg_thread_is_finished = False
                    self.ffmpeg_run_stabilization_pass1()
                    message = f"PROCESSING: Stabilization Pass 1, please be patient"
                    self.update_statusbar(message)
                    self.progressBar.show()
                if (
                    not self.stabilize_step2_finished
                    and self.process_step == "stabilize_pass1"
                ):
                    self.process_step = "stabilize_pass2"
                    logging.debug("##### Starting ffmpeg thread for pass 2")

    def ffmpeg_setup_stabilization_pass2(self):
        """Ffmpeg pass 2 stabilization."""
        if self.ffmpeg_thread_is_finished:
            if self.video_ffmpeg_stabilize:  # User wants to stabilize
                if not self.stabilize_step2_finished:
                    self.process_step = "stabilize_pass2"
                    logging.debug("##### Starting ffmpeg thread for Pass 2")
                    self.progressBar.setValue(0)
                    self.ffmpeg_thread_is_running = False
                    self.ffmpeg_thread_is_finished = False
                    self.ffmpeg_run_stabilization_pass2()
                    message = f"PROCESSING: Stabilization Pass 2, please be patient"
                    self.update_statusbar(message)
                    self.progressBar.show()

    def start_image_stack_process(self):
        """Start the image stack creation thread"""
        if self.image_processor_thread_is_running:
            self.warning_dialog(
                "Image Stack Thread in progress",
                "Cannot start Image Stack Process as a "
                "thread is already running. Please wait "
                "until the existing process completes before "
                "attempting to recreate the image stack.",
                style="Ok",
            )
            return
        else:
            # from image_velocimetry_tools.gui.filesystem import \
            #     ImageStackTask
            self.progressBar.setValue(0)
            self.progressBar.show()
            map_file_path = os.path.join(self.swap_directory, "image_stack.dat")

            # processed_frames = self.imagebrowser.sequence
            processed_frames = glob.glob(
                os.path.join(self.swap_image_directory, "t*.jpg")
            )

            # Don't try to process if there are not any images
            if processed_frames:

                def progress_callback(progress):
                    # Update your GUI with the progress information
                    self.progressBar.setValue(progress)
                    # logging.debug(f"IMAGE STACK WORKER: Progress: {progress}%")

                # Use ImageStackService to create the image stack
                with self.wait_cursor():
                    map_file_size_thres = 9e8
                    try:
                        image_stack = self.image_stack_service.create_image_stack(
                            image_paths=processed_frames,
                            progress_callback=progress_callback,
                            map_file_path=map_file_path,
                            map_file_size_thres=map_file_size_thres,
                        )
                        self.image_stack_process_finished(image_stack)
                        self.set_button_color("pushbuttonCreateRefreshImageStackSTIV", "good")
                    except ValueError as e:
                        self.warning_dialog(
                            "Image Stack Creation Failed",
                            f"Failed to create image stack: {str(e)}",
                            style="ok",
                        )
                        logging.error(f"Image stack creation failed: {e}")
            else:
                self.warning_dialog(
                    "Cannot Create Image Stack: No Rectified Frames",
                    "There are no rectified (t*.jpg) frames in the project "
                    "structure 1-Images folder. Please use the Export "
                    "Projected Frames button in the Orthorectification Tab "
                    "to create rectified frames before generating the Image "
                    "Stack.",
                    style="ok",
                )
                self.pushbuttonExportProjectedFrames.setEnabled(True)
                self.groupboxExportOrthoFrames.setEnabled(True)

    def start_image_stack_process_finished(self):
        """Executes when the image stack process starting function has completed"""
        pass

    def image_stack_process_finished(self, image_stack):
        """Perform post tasks once the image stack process has completed."""
        self.image_processor_thread_is_running = False
        self.image_stack = image_stack

        # Enable the Image Velocimetry Tabs
        to_enable = [
            "groupboxSpaceTimeParameters",
            "groupboxSpaceTimeOptParameters",
            "pushbuttonExportProjectedFrames",
            "buttonSTIVProcessVelocities",
            "buttonSTIVOptProcessVelocities",
            "pushbuttonExportProjectedFrames",
        ]
        self.pushbuttonExportProjectedFrames.setEnabled(True)
        self.set_qwidget_state_by_name(to_enable, True)
        logging.debug("image_stack_process_finished: stack created successfully")
        message = "ORTHORECTIFICATION: Image Stack Created."
        self.update_statusbar(message)
        self.set_tab_icon("tabOrthorectification", "good")

        self.progressBar.setValue(100)
        self.progressBar.hide()
        self.progressBar.setValue(0)  # reset progress bar for next use

    def start_image_preprocessor_process(self):
        """Starts the image preprocessor image"""
        # Prepare the inputs to the image preprocessor
        image_paths = glob.glob(os.path.join(self.swap_image_directory, "f*.jpg"))

        # Get preprocessing parameters from UI
        do_clahe = False
        clahe_clip = 2.0
        clahe_horz_tiles = 8
        clahe_vert_tiles = 8
        do_auto_contrast = False
        auto_contrast_percent = None

        if self.checkboxApplyClahe.isChecked():
            do_clahe = True
            clahe_clip = float(self.lineeditClaheClipLimit.text())
            clahe_horz_tiles = int(self.lineeditClaheHorzTileSize.text())
            clahe_vert_tiles = int(self.lineeditClaheVertTileSize.text())
        if self.checkboxAutoContrast.isChecked():
            do_auto_contrast = True
            auto_contrast_percent = int(self.lineeditAutoContrastPercentClip.text())

        # Validate preprocessing parameters using service
        validation_errors = self.image_stack_service.validate_preprocessing_parameters(
            clahe_clip=clahe_clip,
            clahe_horz_tiles=clahe_horz_tiles,
            clahe_vert_tiles=clahe_vert_tiles,
            auto_contrast_percent=auto_contrast_percent,
        )
        if validation_errors:
            self.warning_dialog(
                "Invalid Preprocessing Parameters",
                "The following preprocessing parameters are invalid:\n" + "\n".join(validation_errors),
                style="ok",
            )
            return

        # Use service to get properly formatted parameters
        preprocessing_params = self.image_stack_service.get_preprocessing_parameters(
            do_clahe=do_clahe,
            clahe_clip=clahe_clip,
            clahe_horz_tiles=clahe_horz_tiles,
            clahe_vert_tiles=clahe_vert_tiles,
            do_auto_contrast=do_auto_contrast,
            auto_contrast_percent=auto_contrast_percent,
        )
        clahe_parameters = preprocessing_params["clahe_parameters"]

        # Create an instance of the ImageProcessor
        self.image_processor = ImageProcessor(
            image_paths=image_paths,
            clahe_parameters=clahe_parameters,
            auto_contrast_percent=auto_contrast_percent,
            do_clahe=do_clahe,
            do_auto_contrast=do_auto_contrast,
        )

        self.image_processor.progress.connect(self.handle_progress)
        self.image_processor.finished.connect(self.handle_processing_finished)

        # Start the image processing
        self.progressBar.setValue(0)
        self.progressBar.show()
        self.image_processor.preprocess_images(
            image_paths=image_paths,
            clahe_parameters=clahe_parameters,
            auto_contrast_percent=auto_contrast_percent,
            do_clahe=do_clahe,
            do_auto_contrast=do_auto_contrast,
        )

    def handle_progress(self, progress):
        """Updates the progress bar

        Args:
            progress (int): the current process as an integer representing percent (0-100)
        """
        # Slot to handle progress updates
        logging.debug(f"Image preprocessor progress: {progress}")
        self.progressBar.setValue(progress)

    def handle_processing_finished(self):
        """Executes when the image pre-processor is finished"""
        # Slot to handle processing finished
        logging.info("Preprocessing images finished.")
        self.progressBar.setValue(100)
        self.progressBar.hide()
        self.progressBar.setValue(0)

    def ortho_original_image_zoom_image(self, zoom_value):
        """Zoom in and zoom out."""
        self.ortho_original_image_zoom_factor = zoom_value
        self.ortho_original_image.zoomEvent(self.ortho_original_image_zoom_factor)
        # self.toolbuttonOrthoOrigImageZoomIn.setEnabled(self.ortho_original_image_zoom_factor < 4.0)
        # self.toolbuttonOrthoOrigImageZoomOut.setEnabled(self.ortho_original_image_zoom_factor > 0.333)

    def ortho_rectified_image_zoom_image(self, zoom_value):
        """Zoom in and zoom out."""
        self.ortho_rectified_image_zoom_factor = zoom_value
        self.ortho_rectified_image.zoomEvent(self.ortho_rectified_image_zoom_factor)
        # self.toolbuttonOrthoOrigImageZoomIn.setEnabled(self.ortho_original_image_zoom_factor < 4.0)
        # self.toolbuttonOrthoOrigImageZoomOut.setEnabled(self.ortho_original_image_zoom_factor > 0.333)

    def ortho_original_image_normal_size(self):
        """View image with its normal dimensions."""
        self.ortho_original_image.clearZoom()
        self.ortho_original_image_zoom_factor = 1.0

    def ortho_rectified_image_normal_size(self):
        """View image with its normal dimensions."""
        self.ortho_rectified_image.clearZoom()
        self.ortho_rectified_image_zoom_factor = 1.0

    def ortho_rectified_water_surface_elevation(self):
        """Logs changes in the specified WSE for rectification."""
        item = self.doubleSpinBoxRectificationWaterSurfaceElevation.value()
        if self.display_units == "Metric":
            self.ortho_rectified_wse_m = float(item)
        if self.display_units == "English":
            self.ortho_rectified_wse_m = float(item) * 1 / self.survey_units["L"]

        # If AC3 file is loaded, attempt to update AC3 backend
        self.signal_wse_changed.emit(self.ortho_rectified_wse_m)
        # self.xs_survey.update_backend()

    @staticmethod
    def load_ndarray_into_qtablewidget(
        ndarray: np.ndarray, table: QtWidgets.QTableWidget
    ):
        """Static method to place numpy array data into a given QtTableWidget."""
        rows, columns = ndarray.shape
        table.setRowCount(rows)
        table.setColumnCount(columns)
        # table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        for r in range(rows):
            for c in range(columns):
                item = QtWidgets.QTableWidgetItem(f"{ndarray[r, c]:.4f}")
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(r, c, item)
        # table.resizeColumnToContents()

    @staticmethod
    def qtablewidget_to_dataframe(table):
        """Convert data from a QTableWidget into a pandas DataFrame."""
        num_rows = table.rowCount()
        num_columns = table.columnCount()

        # Extract data from table
        data = []
        for row in range(num_rows):
            row_data = []
            for column in range(num_columns):
                item = table.item(row, column)
                if item is not None:
                    row_data.append(item.text())
                else:
                    row_data.append(None)
            data.append(row_data)

        # Construct DataFrame
        df = pd.DataFrame(data)
        return df

    def ortho_original_image_digitize_point(self):
        """Connect point digitizer to the GCP image pane."""
        pixmap = QtGui.QPixmap(self.__icon_path__ + os.sep + "crosshairs-solid.svg")
        pixmap = pixmap.scaledToWidth(32)
        cursor = QtGui.QCursor(pixmap, hotX=16, hotY=16)
        print(f"cursor hotspot: {cursor.hotSpot()}")
        if self.toolbuttonOrthoOrigImageDigitizePoint.isChecked():
            # self.imagebrowser_button_state_checker("eyedropper")
            self.ortho_original_image.setCursor(cursor)

            # Create the mouse event
            self.ortho_original_image.leftMouseButtonReleased.connect(
                self.ortho_original_image_get_pixel
            )
        else:
            self.ortho_original_image.setCursor(Qt.ArrowCursor)

    def ortho_original_image_get_pixel(self, x, y):
        """Extract the pixel information."""
        row = int(y)
        column = int(x)
        logging.debug(
            "Clicked on image pixel (row=" + str(row) + ", column=" + str(column) + ")"
        )
        # x = event.pos().x() / self.ortho_original_image_zoom_factor
        # y = event.pos().y() / self.ortho_original_image_zoom_factor
        # c = self.ortho_original_image.image.pixel(x, y)
        self.ortho_original_image_current_pixel = [x, y]
        logging.debug(
            f"##### Pixel Info: x: {self.ortho_original_image_current_pixel[0]}, "
            f"y: {self.ortho_original_image_current_pixel[1]}."
        )
        logging.debug(
            f"Current selected GCP table row: {self.orthoPointsTable.currentRow()}"
        )
        if self.toolbuttonOrthoOrigImageDigitizePoint.isChecked():
            self.orthoPointsTable.setItem(
                self.orthoPointsTable.currentRow(),
                4,
                QtWidgets.QTableWidgetItem(f"{x:.3f}"),
            )
            self.orthoPointsTable.setItem(
                self.orthoPointsTable.currentRow(),
                5,
                QtWidgets.QTableWidgetItem(f"{y:.3f}"),
            )

            # Grab the current points from the table
            what_to_plot = self.get_orthotable_points_to_plot()
            self.signal_ortho_original_digitized_point.emit(what_to_plot)

            self.ortho_original_image.clearPoints()
            # self.ortho_original_image.addLabeledPoint(what_to_plot)
            # self.ortho_original_image.paintEvent(what_to_plot)
            self.ortho_original_image.scene.set_current_instruction(
                Instructions.ADD_POINTS_INSTRUCTION,
                points=what_to_plot["points"],
                labels=what_to_plot["labels"],
            )

    def rectify_single_frame(self):
        """Executes when user presses 'Rectify Current Image' button."""

        # Setup
        # gcp_table = self.get_table_as_dict(self.orthoPointsTable)
        if self.orthotable_dataframe.empty:
            return
        gcp_table = self.orthotable_dataframe.to_dict()  # always meters
        labels = list(gcp_table["# ID"].values())
        world_coords = tuple(
            zip(
                [float(item) for item in gcp_table["X"].values()],  # X
                [float(item) for item in gcp_table["Y"].values()],  # Y
                [float(item) for item in gcp_table["Z"].values()],  # Z
            )
        )
        transformed_image = np.array([])
        rect_params = self.rectification_parameters
        rect_params["extent"] = np.array(None)
        rect_params["homography_matrix"] = np.array(rect_params["homography_matrix"])
        rect_params["camera_matrix"] = np.array(rect_params["camera_matrix"])
        rect_params["pixel_coords"] = np.array(rect_params["pixel_coords"])
        rect_params["world_coords"] = np.array(rect_params["world_coords"])
        rect_params["extent"] = np.array(rect_params["extent"])

        # Grab only the points user specified for rectification
        # These are the points user wants rectified, but only their pixel coords, so we have to compare
        # that to the entire table of points
        points_dict = self.get_orthotable_points_to_plot(which_points="rectification")
        matched_point_labels = find_matches_between_two_lists(
            points_dict["labels"], labels
        )
        self.rectification_parameters["pixel_coords"] = np.array(points_dict["points"])
        self.rectification_parameters["world_coords"] = np.array(
            [world_coords[index[0]] for index in matched_point_labels]
        )  # always meters
        num_points = self.rectification_parameters["pixel_coords"].shape[0]
        self.rectification_parameters["water_surface_elev"] = self.ortho_rectified_wse_m

        # Don't let the WSE be exactly 0.0
        if self.rectification_parameters["water_surface_elev"] == 0.0:
            self.rectification_parameters["water_surface_elev"] = 1.0e-5

            # Save a copy of the table in the project structure
        try:
            destination_path = os.path.join(
                self.swap_orthorectification_directory, "ground_control_points.csv"
            )
            dict = self.get_table_as_dict(self.orthoPointsTable)
            pd.DataFrame(dict).fillna("").to_csv(destination_path, index=False)
        except Exception as e:
            self.update_statusbar(
                f"Failed to save GCP table to project " f"structure: {e}"
            )

        # Validate GCP configuration
        validation_errors = self.ortho_service.validate_gcp_configuration(
            self.rectification_parameters["pixel_coords"],
            self.rectification_parameters["world_coords"]
        )
        if validation_errors:
            error_msg = "GCP validation errors:\n" + "\n".join(validation_errors)
            logging.error(error_msg)
            self.warning_dialog(
                "Invalid GCP Configuration",
                error_msg,
                style="ok",
                icon=self.__icon_path__ + os.sep + "IVy_logo.ico"
            )
            return

        # Determine rectification method using service
        self.rectification_method = self.ortho_service.determine_rectification_method(
            num_points,
            self.rectification_parameters["world_coords"]
        )

        logging.debug(
            f"ORTHORECTIFICATION: Found {num_points} to rectify. Are all points on the same Z-plane? "
            f"[{np.all(self.rectification_parameters['world_coords'][:, -1] == self.rectification_parameters['world_coords'][0, -1])}]"
        )
        logging.info(f"Attempting rectification method: {self.rectification_method}")

        if self.rectification_method == "scale":
            # Use service to calculate scale parameters
            image = self.ortho_original_image.scene.ndarray()
            scale_params = self.ortho_service.calculate_scale_parameters(
                self.rectification_parameters["pixel_coords"],
                self.rectification_parameters["world_coords"][:, 0:2],
                image.shape
            )

            # Update rectification parameters
            self.rectification_parameters["homography_matrix"] = scale_params["homography_matrix"]
            self.rectification_parameters["extent"] = scale_params["extent"]
            self.rectification_parameters["pad_x"] = scale_params["pad_x"]
            self.rectification_parameters["pad_y"] = scale_params["pad_y"]

            # Update state
            pixel_gsd = scale_params["pixel_gsd"]
            self.pixel_ground_scale_distance_m = pixel_gsd
            self.is_homography_matrix = True
            self.scene_averaged_pixel_gsd_m = pixel_gsd

            # Calculate quality metrics using service
            quality_metrics = self.ortho_service.calculate_quality_metrics(
                "scale",
                pixel_gsd,
                pixel_distance=scale_params["pixel_distance"],
                ground_distance=scale_params["ground_distance"]
            )
            self.rectification_rmse_m = quality_metrics["rectification_rmse_m"]
            self.reprojection_error_pixels = quality_metrics["reprojection_error_pixels"]

            # For scale method, image is not transformed (nadir assumption)
            transformed_image = image

            # For scale method, transformed points are the same as pixel coords
            transformed_points = self.rectification_parameters["pixel_coords"]

            # Update UI
            self.load_ndarray_into_qtablewidget(
                self.rectification_parameters["homography_matrix"],
                self.tablewidgetProjectiveMatrix,
            )
            self.lineeditPixelGSD.setText(
                f"{self.pixel_ground_scale_distance_m * units_conversion(self.display_units)['L']:.3f}"
            )
            logging.info(
                f"Pixel GSD: "
                f"{self.pixel_ground_scale_distance_m * units_conversion(self.display_units)['L']} "
                f"{units_conversion(self.display_units)['label_L']}"
            )

        if self.rectification_method == "homography":
            # Set padding parameters
            pad_x, pad_y = 200, 200
            logging.info(
                f"ORTHORECTIFICATION: Rectifying image: Padding ({pad_x}, {pad_y}). "
                f"Output image will be scaled so that all pixels are positive with specified padding."
            )

            # Check to see if we have a homography matrix already
            existing_homography = None
            if self.is_homography_matrix:
                existing_homography = self.rectification_parameters.get("homography_matrix")

            # Use service to calculate homography parameters
            image = self.ortho_original_image.scene.ndarray()
            homography_params = self.ortho_service.calculate_homography_parameters(
                image,
                self.rectification_parameters["pixel_coords"],
                self.rectification_parameters["world_coords"],
                homography_matrix=existing_homography,
                pad_x=pad_x,
                pad_y=pad_y
            )

            # Update rectification parameters
            transformed_image = homography_params["transformed_image"]
            self.rectification_parameters["homography_matrix"] = homography_params["homography_matrix"]
            self.rectification_parameters["extent"] = homography_params["extent"]
            self.rectification_parameters["pad_x"] = homography_params["pad_x"]
            self.rectification_parameters["pad_y"] = homography_params["pad_y"]

            # Update state
            pixel_gsd = homography_params["pixel_gsd"]
            self.pixel_ground_scale_distance_m = pixel_gsd
            self.is_homography_matrix = True
            self.scene_averaged_pixel_gsd_m = pixel_gsd

            # Calculate quality metrics using service
            quality_metrics = self.ortho_service.calculate_quality_metrics(
                "homography",
                pixel_gsd,
                homography_matrix=self.rectification_parameters["homography_matrix"]
            )
            if "estimated_view_angle" in quality_metrics:
                self.rectification_parameters["estimated_view_angle"] = quality_metrics["estimated_view_angle"]
                logging.info(f"Rectify Single Frame: estimated view angle: "
                           f"{self.rectification_parameters['estimated_view_angle']:.2f}°")
            self.rectification_rmse_m = quality_metrics["rectification_rmse_m"]
            self.reprojection_error_pixels = quality_metrics["reprojection_error_pixels"]

            # Update UI
            self.load_ndarray_into_qtablewidget(
                self.rectification_parameters["homography_matrix"],
                self.tablewidgetProjectiveMatrix,
            )
            self.lineeditPixelGSD.setText(
                f"{self.pixel_ground_scale_distance_m * units_conversion(self.display_units)['L']:.3f}"
            )
            logging.info(
                f"Pixel GSD: "
                f"{self.pixel_ground_scale_distance_m * units_conversion(self.display_units)['L']} "
                f"{units_conversion(self.display_units)['label_L']}"
            )

            logging.info(
                f"Homography matrix (perspective --> ortho): \n"
                f"{np.array2string(np.linalg.inv(self.rectification_parameters['homography_matrix']), precision=4, floatmode='fixed')}"
            )
            logging.info(
                f"Homography matrix (ortho --> perspective): \n"
                f"{np.array2string(self.rectification_parameters['homography_matrix'], precision=4, floatmode='fixed')}"
            )

            # Plot the Points table onto the rectified image
            H = self.rectification_parameters["homography_matrix"]
            points = self.ortho_original_image.points_ndarray()
            transformed_points = transform_points_with_homography(points, H)

        if self.rectification_method == "camera matrix":
            # Check to see if we have a camera matrix
            existing_camera_matrix = None
            if self.is_camera_matrix:
                existing_camera_matrix = self.rectification_parameters.get("camera_matrix")

            # Use service to calculate camera matrix parameters
            image = self.ortho_original_image.scene.ndarray()
            camera_params = self.ortho_service.calculate_camera_matrix_parameters(
                image,
                self.rectification_parameters["pixel_coords"],
                self.rectification_parameters["world_coords"],
                self.rectification_parameters["water_surface_elev"],
                camera_matrix=existing_camera_matrix,
                padding_percent=0.03
            )

            # Update rectification parameters
            transformed_image = camera_params["transformed_image"]
            self.rectification_parameters["camera_matrix"] = camera_params["camera_matrix"]
            self.rectification_parameters["extent"] = camera_params["extent"]

            # Update state
            self.camera_position = camera_params["camera_position"]
            pixel_gsd = camera_params["pixel_gsd"]
            self.pixel_ground_scale_distance_m = pixel_gsd

            # Log camera matrix info
            logging.info(f"Camera matrix:\n{camera_params['camera_matrix']}")
            if camera_params["projection_rms_error"] is not None:
                logging.info(f"Projection RMS error: {camera_params['projection_rms_error']:.4f}")

            # Update UI
            self.load_ndarray_into_qtablewidget(
                self.rectification_parameters["camera_matrix"],
                self.tablewidgetProjectiveMatrix,
            )
            self.lineeditPixelGSD.setText(
                f"{self.pixel_ground_scale_distance_m * units_conversion(self.display_units)['L']:.3f}"
            )
            logging.info(
                f"Pixel GSD: "
                f"{self.pixel_ground_scale_distance_m * units_conversion(self.display_units)['L']} "
                f"{units_conversion(self.display_units)['label_L']}"
            )
            logging.info(
                f"Camera position in world coordinates: "
                f"{self.camera_position * units_conversion(self.display_units)['L']} "
                f"{units_conversion(self.display_units)['label_L']}"
            )
            scaled_world_coordinates = world_coords

        # Flip the image if requested
        transformed_image = flip_image_array(
            image=transformed_image,
            flip_x=self.is_ortho_flip_x,
            flip_y=self.is_ortho_flip_y
        )

        self.ortho_rectified_image.scene.setImage(transformed_image)
        self.ortho_rectified_image.setEnabled(True)
        self.groupboxExportOrthoFrames.setEnabled(True)

        # Plot the reprojected points on the original image
        what_to_plot = self.get_orthotable_points_to_plot()
        if self.rectification_method == "homography":
            what_to_plot["points"] = self.rectification_parameters["world_coords"][
                :, 0:2
            ]
        if self.rectification_method == "camera matrix":
            original_points_to_plot = self.get_orthotable_points_to_plot()
            K = self.rectification_parameters["camera_matrix"]
            transformed_points = (
                K
                @ get_homographic_coordinates_3D(
                    np.array(original_points_to_plot["coordinates"])
                ).T
            )
            reprojected_points = transformed_points / transformed_points[2, :]
            reprojected_points = reprojected_points[:2, :].T

            # self.ortho_original_image.updateViewer()
            original_points = original_points_to_plot["points"]
            self.reprojection_error_gcp_pixel_xy = reprojected_points - original_points
            a_min_b = reprojected_points.T - original_points.T
            self.reprojection_error_gcp_pixel_total = np.sqrt(
                np.einsum("ij,ij->j", a_min_b, a_min_b)
            )
            self.reprojection_error_pixels = np.sqrt(
                np.mean((reprojected_points - original_points) ** 2)
            )
            logging.info(
                f"Reprojection error by point (pixels): \n{self.reprojection_error_gcp_pixel_total}"
            )
            pixel_gsd = self.pixel_ground_scale_distance_m
            self.rectification_rmse_m = (self.reprojection_error_pixels *
                                       pixel_gsd)

            # Plot the Points table onto the rectified image
            image_shape = self.ortho_rectified_image.scene.ndarray().shape
            space_gcp_on_wse = original_points_to_plot["coordinates"]
            space_gcp_on_wse[:, -1] = self.rectification_parameters[
                "water_surface_elev"
            ]
            gcp_projected_on_wse = space_to_image(
                get_homographic_coordinates_3D(space_gcp_on_wse), K
            )
            gcp_rectified_on_wse, _ = image_to_space(
                gcp_projected_on_wse,
                K,
                Z=self.rectification_parameters["water_surface_elev"],
            )

            # Show the water surface projection in the perspective image
            for i in range(len(original_points)):
                self.ortho_original_image.scene.set_current_instruction(
                    Instructions.ADD_LINE_BY_POINTS,
                    points=[original_points[i, :], gcp_projected_on_wse[i, :]],
                )
                self.ortho_original_image.scene.line_item[-1].setPen(
                    QtGui.QPen(QtGui.QColor("red"), 2, Qt.DotLine)
                )

            # Extract the x_min and y_min values from the extent
            extent = self.rectification_parameters["extent"]
            x_min, y_min = extent[0], extent[2]
            normalized_points = (
                gcp_rectified_on_wse[:, :2] - np.array([x_min, y_min])
            ) / (extent[1] - extent[0])

            # Convert normalized coordinates to pixel coordinates
            gcp_pixel_coordinates_rectified = normalized_points * np.array(
                image_shape[:2][::-1]
            )
            # self.ortho_rectified_image.clearPoints()
            # self.ortho_rectified_image.scene.set_current_instruction(
            #     Instructions.ADD_POINTS_INSTRUCTION,
            #     points=gcp_pixel_coordinates_rectified,
            #     labels=original_points_to_plot["labels"],
            # )

            # to ensure I get the same results in the instance attribute
            transformed_points = gcp_pixel_coordinates_rectified

        # Populate the reprojection errors into the Points Table
        # First, zero out all the errors
        # ortho_table_dict = self.get_table_as_dict(self.orthoPointsTable)
        ortho_table_dict = self.orthotable_dataframe.to_dict()
        for r in range(len(ortho_table_dict["# ID"])):
            self.orthoPointsTable.setItem(r, 6, QtWidgets.QTableWidgetItem(f"0"))
            self.orthoPointsTable.setItem(r, 7, QtWidgets.QTableWidgetItem(f"0"))
            self.orthoPointsTable.setItem(r, 8, QtWidgets.QTableWidgetItem(f"0"))

        errors = what_to_plot
        errors["reprojection_error_gcp_pixel_xy"] = self.reprojection_error_gcp_pixel_xy
        errors["reprojection_error_gcp_pixel_total"] = (
            self.reprojection_error_gcp_pixel_total
        )

        b_dict = {}
        for i, b in enumerate(list(ortho_table_dict["# ID"].values())):
            b_dict[b] = i

        ind = [
            i
            for i, value in enumerate(ortho_table_dict["Use in Rectification"].values())
            if string_to_boolean(value)
        ]
        if self.reprojection_error_gcp_pixel_total is not None:
            r = 0
            for i in ind:
                self.orthoPointsTable.setItem(
                    i,
                    8,
                    QtWidgets.QTableWidgetItem(
                        f"{self.reprojection_error_gcp_pixel_total[r]:.3f}"
                    ),
                )
                r += 1
        if self.reprojection_error_gcp_pixel_xy is not None:
            r = 0
            for i in ind:
                self.orthoPointsTable.setItem(
                    i,
                    6,
                    QtWidgets.QTableWidgetItem(
                        f"{self.reprojection_error_gcp_pixel_xy[r, 0]:.3f}"
                    ),
                )
                self.orthoPointsTable.setItem(
                    i,
                    7,
                    QtWidgets.QTableWidgetItem(
                        f"{self.reprojection_error_gcp_pixel_xy[r, 1]:.3f}"
                    ),
                )
                r += 1

        # Ensure the current frame is in the Grid Preparation Tab
        # Show result in the Grid Preparation tab
        self.gridpreparation.imageBrowser.scene.setImage(
            self.ortho_rectified_image.scene.ndarray()
        )
        self.gridpreparation.imageBrowser.setEnabled(True)
        self.set_qwidget_state_by_name(
            [
                "PointPage",
                "SimpleLinePage",
                # "CrossSectionPage",
                "RegularGridPage",
                "MaskingPage",
            ],
            True,
        )

        # self.groupBoxMasking.setEnabled(True)
        # self.groupboxGridGeneration.setEnabled(True)

        # Ensure RMSE error is loaded
        self.lineeditPixelRMSE.setText(f"{self.reprojection_error_pixels:.3f}")
        logging.info(
            f"Total reprojection RMSE is {self.reprojection_error_pixels:.3f} pixels"
        )

        # Ensure the current frame is in the STIV Tab
        # Show result in the Grid Preparation tab
        self.stiv.imageBrowser.scene.setImage(
            self.ortho_rectified_image.scene.ndarray()
        )
        # self.stiv.imageBrowser.setEnabled(True) # Debug : don't enable yet

        # Ensure the current frame is in the STIV Opt Tab
        # Show result in the Grid Preparation tab
        self.stiv_opt.imageBrowser.scene.setImage(
            self.ortho_rectified_image.scene.ndarray()
        )
        self.stiv_opt.imageBrowser.setEnabled(True)

        # Put a copy of the results into the Cross-section Geometry
        # AnnotationViews
        self.perspective_xs_image.scene.setImage(
            self.ortho_original_image.scene.ndarray()
        )
        self.rectified_xs_image.scene.setImage(
            self.ortho_rectified_image.scene.ndarray()
        )
        self.perspective_xs_image.setEnabled(True)
        self.rectified_xs_image.setEnabled(True)

        # Enable the Cross-section tab, and ability to import Bathymetry
        self.set_qwidget_state_by_name("tabCrossSectionGeometry", True)
        self.actionImport_Bathymetry.setEnabled(True)

        # Write the homography or camera matrix to the project structure
        try:
            if "homography_matrix" in self.rectification_parameters:
                with open(
                    os.path.join(
                        self.swap_orthorectification_directory, "homography_matrix.csv"
                    ),
                    "w",
                    newline="",
                ) as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerows(
                        self.rectification_parameters["homography_matrix"]
                    )
            if "camera_matrix" in self.rectification_parameters:
                # Check if this is the default camera matrix, ignore if so
                first_3_columns = self.rectification_parameters["camera_matrix"][:, :3]
                expected_identity = np.eye(3)  # Create a 3x3 identity matrix
                is_identity = np.array_equal(first_3_columns, np.eye(3))
                if not is_identity:
                    with open(
                        os.path.join(
                            self.swap_orthorectification_directory, "camera_matrix.csv"
                        ),
                        "w",
                        newline="",
                    ) as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerows(
                            self.rectification_parameters["camera_matrix"]
                        )

            # Write out the current Points Table
            # TODO: Units may not be right here
            dict = self.get_table_as_dict(self.orthoPointsTable)
            file_name = os.path.join(
                self.swap_orthorectification_directory, "rectification_points_table.csv"
            )
            pd.DataFrame(dict).fillna("").to_csv(file_name, index=False)

            with open(
                os.path.join(self.swap_orthorectification_directory, "pixel_gsd.csv"),
                "w",
                newline="",
            ) as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([self.pixel_ground_scale_distance_m])
            with open(
                os.path.join(
                    self.swap_orthorectification_directory,
                    "water_surface_elevation.csv",
                ),
                "w",
                newline="",
            ) as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(
                    [self.doubleSpinBoxRectificationWaterSurfaceElevation.value()]
                )
            if self.reprojection_error_gcp_pixel_total is None:
                rmse = -9999.0
            else:
                rmse = self.reprojection_error_gcp_pixel_total
            with open(
                os.path.join(self.swap_orthorectification_directory, "pixel_rmse.csv"),
                "w",
                newline="",
            ) as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([rmse])
        except:
            pass

        # Update the status bar
        if self.reprojection_error_pixels is not None:
            message = (
                f"ORTHORECTIFICATION: Perspective image rectified using ground control. "
                f"Total reprojection RMSE is {self.reprojection_error_pixels:.3f} pixels. "
                f"Check image results for accuracy."
            )
        else:
            message = (
                f"ORTHORECTIFICATION: Perspective image rectified using ground control. "
                f"Check image results for accuracy."
            )
        self.update_statusbar(message)

        # Serialize rectification parameters
        self.rectification_parameters = rect_params

        # Save the rectified transformed point
        self.rectified_transformed_gcp_points = serialize_numpy_array(
            transformed_points
        )

        # Note the process step
        self.process_step = "Rectify Single Frame"

    def orthorectify_many_thread_handler(self):
        """Executes when user presses the Export Projected Frames button.

        Use the Worker threadpool to process all the rectification calls.
          1. check to see which method to call
          2. pass the function to the Worker
          3. ensure the worker is computing progress as an int from 0 - 100
          4. show progress bar while thread is running
          5. hide progress bar when thread finishes (may have to do this in update_progress_bar?)

        """
        logging.debug(
            f"Processing rectification for all frames based on current settings."
        )

        # Check with user that the correct files will be processed
        images_to_process = self.imagebrowser.sequence

        message = f"ORTHORECTIFICATION: Exporting frames according to settings. Please be patient."
        self.update_statusbar(message)
        if True:  # dialog == QtWidgets.QMessageBox.Yes:
            # event.accept()  # If using the menu this fails, no clean exit, Issue #4
            # Pass the function to execute
            if self.rectification_method is None:
                return
            if self.rectification_method == "scale":
                worker = Worker(
                    self.rectify_many_scale
                )  # Any other args, kwargs are passed to the run function
            if self.rectification_method == "homography":
                worker = Worker(
                    self.rectify_many_homography
                )  # Any other args, kwargs are passed to the run function
            if self.rectification_method == "camera matrix":
                worker = Worker(
                    self.rectify_many_camera_matrix
                )  # Any other args, kwargs are passed to the run function
            worker.signals.result.connect(self.orthorectify_many_results)
            worker.signals.finished.connect(self.orthorectify_many_thread_finished)
            worker.signals.progress.connect(self.update_progress_bar_qthreads)

            # Execute
            self.progressBar.show()
            self.threadpool.start(worker)
        else:
            pass

    def stiv_thread_handler(self):
        """Executes when user presses the Process STIV button (Either method.

        Use the Worker threadpool to process all the stiv calls.
          1. check to see which method to call
          2. pass the function to the Worker
          3. ensure the worker is computing progress as an int from 0 - 100
          4. show progress bar while thread is running
          5. hide progress bar when thread finishes (may have to do this in update_progress_bar?)

        """
        logging.debug(f"Processing STIV for all frames based on current settings.")

        sender_button = self.sender()

        if self.image_stack is None:
            message = f"SPACE-TIME IMAGE VELOCIMETRY: No image stack."
            self.warning_dialog(
                "STIV PROCESS: Error, No Image Stack", message, style="ok"
            )
            return

        message = (
            f"SPACE-TIME IMAGE VELOCIMETRY: Processing STIV results. Please "
            f"be patient. (Press Shift+ESC to cancel)"
        )
        self.update_statusbar(message)

        # Configure stiv processor
        self.stiv_search_line_distance_changed()
        self.stiv.image_stack = self.image_stack
        self.stiv.grid = self.results_grid
        self.stiv.num_pixels = self.stiv_num_pixels
        self.stiv.phi_origin = self.stiv_phi_origin
        self.stiv.d_phi = self.stiv_dphi
        self.stiv.phi_range = self.stiv_phi_range
        self.stiv.pixel_gsd = self.pixel_ground_scale_distance_m
        self.stiv.d_t = self.extraction_timestep_ms / 1000  # in seconds
        self.stiv.max_velocity_threshold_mps = self.stiv_max_vel_threshold_mps
        self.stiv.map_file_path = os.path.join(self.swap_directory, "stiv_map_file.dat")

        # Configure stiv optimized processor
        # TODO: think about whether I need this second stiv_opt, or maybe
        #  I can just use the same instance?
        self.stiv_opt.image_stack = self.image_stack
        self.stiv_opt.grid = self.results_grid
        self.stiv_opt.num_pixels = self.stiv_num_pixels
        self.stiv_opt.phi_origin = self.stiv_phi_origin
        self.stiv_opt.d_phi = self.stiv_dphi
        self.stiv_opt.phi_range = self.stiv_phi_range
        self.stiv_opt.pixel_gsd = self.pixel_ground_scale_distance_m
        self.stiv_opt.d_t = self.extraction_timestep_ms / 1000  # in seconds
        self.stiv_opt.max_velocity_threshold_mps = self.stiv_opt_max_vel_threshold_mps
        self.stiv_opt.map_file_path = os.path.join(
            self.swap_directory, "stiv_map_file.dat"
        )

        # Define the processing function based on the sender_button
        processing_function = None
        if sender_button == self.buttonSTIVProcessVelocities:
            processing_function = self.stiv.process_stiv_exhaustive
            self.process_step = "Process STIV Exhaustive"
        elif sender_button == self.buttonSTIVOptProcessVelocities:
            processing_function = self.stiv_opt.process_stiv_optimized
            self.process_step = "Process STIV Optimized"

        if processing_function:
            worker = Worker(processing_function)
            worker.signals.result.connect(self.process_stiv_results)
            worker.signals.finished.connect(self.process_stiv_thread_finished)
            worker.signals.progress.connect(self.update_progress_bar_qthreads)

            # Execute
            self.progressBar.show()
            self.threadpool.start(worker)

    def stop_current_threadpool_task(self):
        """
        Cancel the currently running task in the thread pool.

        This method iterates through all the threads in the thread pool
        and checks if any of them are currently running. If a running
        thread is found, it is terminated by calling the `quit()` method
        followed by `wait()` to ensure proper termination.

        Returns:
            None

        Notes:
            Ensure that this method is called from the main GUI thread
            to prevent any thread-safety issues.
        """
        # Check if the thread pool has active threads
        if self.threadpool.activeThreadCount() == 0:
            logging.info("stop_current_threadpool_task: No active threads to " "stop.")
            return

        # Iterate through active threads and cancel them
        for thread in self.threadpool.children():
            if isinstance(thread, QtCore.QThread) and thread.isRunning():
                thread.quit()
                thread.wait()

    def orthorectify_many_results(self, thread_results):
        """Executes with the orthorectification process completes

        Args:
            thread_results (str): the results of the thread process
        """
        logging.debug(f"{thread_results}")

    def find_first_transformed_image(self):
        """Helper function to find the first transformed image in the Project Swap

        Returns:
            str or None: the first file, or if nothing is found, None
        """
        matching_files = [
            filename
            for filename in os.listdir(self.swap_image_directory)
            if filename.startswith("t")
        ]
        if matching_files:
            return os.path.join(self.swap_image_directory, matching_files[0])
        else:
            return None

    def orthorectify_many_thread_finished(self):
        """Executes when the orthorectification thread has completed."""
        logging.debug(f"The orthorectification thread has finished.")
        message = f"ORTHORECTIFICATION: Exporting complete. "
        self.process_step = "Rectify Many Frames"
        self.is_transformed_frames = True

        # Set the imageBrowser to show the transformed frames
        self.lineeditFrameFiltering.setText("t*.jpg")
        self.imagebrowser.apply_file_filter()

        # Show result in the Grid Preparation tab
        first_transformed_image = self.find_first_transformed_image()

        self.gridpreparation.imageBrowser.scene.load_image(first_transformed_image)
        self.gridpreparation.imageBrowser.setEnabled(True)
        self.update_statusbar(message)
        self.progressBar.hide()
        self.progressBar.setValue(
            0
        )  # ensures the progress bar is ready to go for next time

        self.pushbuttonExportProjectedFrames.setEnabled(True)
        self.enable_disable_tabs(self.tabWidget, "tabCrossSectionGeometry")

        # We just finished extracting all the required frames, so now, ask
        # the user if they want to create the image stack
        self.create_image_stack()

    def create_image_stack(self):
        """Prompt user if they want to create an image stack and start the
        task."""
        if not self.is_transformed_frames:
            message = (
                f"There are no transformed frames to make an image\n"
                f"stack from. Please use the orthorectification tab\n"
                f"to export projected frames before the creation of an\n"
                f"image stack."
            )
            self.warning_dialog(
                "No images to create image stack",
                message,
                style="ok",
                icon=self.__icon_path__ + os.sep + "IVy_logo.ico",
            )

            return
        message = (
            f"Frames are extracted, but no Image Stack has been "
            f"created. Would you like to create the Image Stack now?\n\n"
            f"Note: An image stack is required before performing "
            f"Image Velocimetry analyses."
        )
        result = self.warning_dialog("Create Image Stack?", message, "YesCancel")
        if result == "yes" and not self.image_processor_thread_is_running:
            message = (
                "ORTHORECTIFICATION: Creating Image Stack. Image "
                "Velocimetry functions will be enabled when finished."
            )
            self.update_statusbar(message)
            # Create an image_stack of the resultant frames
            # Update the ImageProcessor instance with the arguments and run
            logging.debug(
                "orthorectify_many_thread_finished: About to "
                "execute self.start_image_stack_process()"
            )
            self.start_image_stack_process()

            # Disable the Export Projected Frames button while the stack
            # process is running
            # self.set_qwidget_state_by_name("pushbuttonExportProjectedFrames",
            #                                False)
        if result == "cancel":
            return

    def rectify_many_scale(self, progress_callback):
        """Rectify all images using scale method (nadir assumption).

        For scale method, images are only flipped, not transformed.
        This is the function executed by the ThreadPool.
        """
        flip_and_save_images(
            image_folder=self.imagebrowser.folder_path,
            flip_x=self.is_ortho_flip_x,
            flip_y=self.is_ortho_flip_y,
            progress_callback=progress_callback
        )

        return "Done."

    def rectify_many_homography(self, progress_callback):
        """Rectify all images using homography method.

        Applies pre-calculated homography transformation to all extracted frames.
        This is the function executed by the ThreadPool.
        """
        # Get list of extracted frame images
        sequence = sorted(glob.glob(os.path.join(self.swap_image_directory, "f*.jpg")))
        num_images = len(sequence)

        # Get rectification parameters
        homography_matrix = self.rectification_parameters["homography_matrix"]
        world_coords = self.rectification_parameters["world_coords"][:, 0:2]
        pixel_coords = self.rectification_parameters["pixel_coords"]
        pad_x = self.rectification_parameters["pad_x"]
        pad_y = self.rectification_parameters["pad_y"]

        for idx, img_file in enumerate(tqdm(sequence, total=num_images)):
            # Load image
            img = np.array(Image.open(img_file))

            # Apply homography transformation
            transformed_image, _, _, _, _ = rectify_homography(
                image=img,
                points_world_coordinates=world_coords,
                points_perspective_image_coordinates=pixel_coords,
                homography_matrix=homography_matrix,
                pad_x=pad_x,
                pad_y=pad_y,
            )

            # Flip the image if requested
            transformed_image = flip_image_array(
                image=transformed_image,
                flip_x=self.is_ortho_flip_x,
                flip_y=self.is_ortho_flip_y
            )

            # Save rectified image
            output_path = os.path.join(self.swap_image_directory, f"t{idx:05d}.jpg")
            Image.fromarray(transformed_image).save(output_path)
            logging.debug(f"Saved rectified image: {output_path}")

            # Update progress
            progress_callback.emit(int(((idx + 1) / num_images) * 100))

        return "Done."

    def rectify_many_camera_matrix(self, progress_callback):
        """Rectify all images using camera matrix method.

        Applies pre-calculated camera matrix transformation to all selected images.
        This is the function executed by the ThreadPool.
        """
        # Get list of images to process
        images_to_process = self.imagebrowser.sequence
        image_folder = self.imagebrowser.folder_path
        num_images = len(images_to_process)

        # Initialize camera helper with pre-calculated camera matrix
        first_image = np.array(Image.open(images_to_process[0]))
        cam = CameraHelper(
            image=first_image,
            world_points=self.rectification_parameters["world_coords"],
            image_points=self.rectification_parameters["pixel_coords"],
        )
        cam.set_camera_matrix(self.rectification_parameters["camera_matrix"])

        # Get rectification parameters
        water_surface_elev = self.rectification_parameters["water_surface_elev"]
        extent = self.rectification_parameters["extent"]

        for idx, img_file in enumerate(tqdm(images_to_process, total=num_images)):
            # Load image
            img = np.array(Image.open(img_file))

            # Apply camera matrix transformation
            transformed_image = cam.get_top_view_of_image(
                img,
                Z=water_surface_elev,
                extent=extent,
                do_plot=False,
            )

            # Flip the image if requested
            transformed_image = flip_image_array(
                image=transformed_image,
                flip_x=self.is_ortho_flip_x,
                flip_y=self.is_ortho_flip_y
            )

            # Save rectified image
            output_path = os.path.join(image_folder, f"t{idx:05d}.jpg")
            Image.fromarray(transformed_image).save(output_path)
            logging.debug(f"Saved rectified image: {output_path}")

            # Update progress
            progress_callback.emit(int(((idx + 1) / num_images) * 100))

        return "Done."

    def create_line_grid(self, mode="line"):
        """Executed when user clicks Create Points on Line button in Grid Preparation tab.
        Depending on the mode ('line` or 'cross_section'), this function will
        create and add the points along the existing SimpleLineAnnotaton line.
        Points are added as GripItems into the scene.

        """
        if not self.gridpreparation.imageBrowser.has_image():
            return  # No image to draw on
        self.line_mode = mode
        if mode == "line":
            number_points = self.spinboxLineNumPoints.value()
            message = f"GRID PREPARATION: Creating points along digitized line"
        if mode == "cross_section":
            number_points = self.spinbocXsLineNumPoints.value()
            message = (
                f"GRID PREPARATION: Creating points along digitized "
                f"cross-section line"
            )
        image = self.gridpreparation.imageBrowser.scene.ndarray()
        mask_polygons = self.gridpreparation.imageBrowser.polygons_ndarray()
        line_eps = self.gridpreparation.imageBrowser.lines_ndarray()
        if np.any(line_eps):
            line_start = line_eps[-1, 0]
            line_end = line_eps[-1, 1]
        else:
            return

        with self.wait_cursor():
            self.update_statusbar(message)
            self.gridpreparation.imageBrowser.clearPoints()
            self.results_grid = self.gridgenerator.make_line(
                image, line_start, line_end, number_points, mask_polygons
            )
            labels = [str(i + 1) for i in range(self.results_grid.shape[0])]
            self.gridpreparation.imageBrowser.scene.set_current_instruction(
                Instructions.ADD_POINTS_INSTRUCTION,
                points=self.results_grid,
                labels=labels,
            )
            self.gridpreparation.imageBrowser.scene.set_current_instruction(
                Instructions.NO_INSTRUCTION
            )

            # Add points to the Image Vel images
            self.stiv.imageBrowser.clearPoints()
            self.stiv.imageBrowser.scene.set_current_instruction(
                Instructions.ADD_POINTS_INSTRUCTION,
                points=self.results_grid,
                labels=labels,
            )

            self.stiv_opt.imageBrowser.clearPoints()
            self.stiv_opt.imageBrowser.scene.set_current_instruction(
                Instructions.ADD_POINTS_INSTRUCTION,
                points=self.results_grid,
                labels=labels,
            )

    def create_grid(self):
        """Executed when user clicks Create Grid button in the Grid Preparation tab"""
        if not self.gridpreparation.imageBrowser.has_image():
            return  # No image to draw on
        horz = self.spinboxHorizGridSpacing.value()
        vert = self.spinboxVertGridSpacing.value()
        image = self.gridpreparation.imageBrowser.scene.ndarray()
        mask_polygons = self.gridpreparation.imageBrowser.polygons_ndarray()
        message = f"GRID PREPARATION: Creating results grid"

        with self.wait_cursor():
            self.update_statusbar(message)
            self.results_grid = self.gridgenerator.make_grid(image, mask_polygons)

            # Clear out any existing grid points or lines
            self.gridpreparation.imageBrowser.clearPoints()
            self.gridpreparation.imageBrowser.clearLines()

            self.gridpreparation.imageBrowser.scene.set_current_instruction(
                Instructions.ADD_POINTS_INSTRUCTION,
                points=self.results_grid,
                labels=["" for p in self.results_grid],
            )

            # Save the grid
            try:
                self.results_grid_world = self.results_grid  # TODO: fix this
                # self.results_grid_world = pixels_to_world(self.results_grid, self.rectification_parameters["homography_matrix"])

                # Write a CSV file with X,Y, x, y
                with open(
                    os.path.join(self.swap_grids_directory, "results_grid.csv"),
                    "w",
                    newline="",
                ) as csvfile:
                    fieldnames = [
                        "world_coords_x",
                        "world_coords_y",
                        "pixel_coords_x",
                        "pixel_coords_y",
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    # Write the header row
                    writer.writeheader()

                    # Write the data rows
                    for world_coord, pixel_coord in zip(
                        self.results_grid_world, self.results_grid
                    ):
                        row_data = {
                            "world_coords_x": world_coord[0],
                            "world_coords_y": world_coord[1],
                            "pixel_coords_x": pixel_coord[0],
                            "pixel_coords_y": pixel_coord[1],
                        }
                        writer.writerow(row_data)
            except:
                pass

            # Save the binary mask
            try:
                binary_mask_numpy = self.gridgenerator.binary_mask
                binary_mask_uint8 = binary_mask_numpy.astype(np.uint8) * 255
                image = Image.fromarray(binary_mask_uint8)

                # TODO: Save this in a "masks" directory instead of grids?
                image.save(os.path.join(self.swap_grids_directory, "binary_mask.jpg"))
            except:
                pass
        try:
            horz_units = f"px ({self.pixel_ground_scale_distance_m * horz:.2f} world)"
            vert_units = f"px ({self.pixel_ground_scale_distance_m * vert:.2f} world)"
            self.labelHorzSpacingWorldUnits.setText(horz_units)
            self.labelVertSpacingWorldUnits.setText(vert_units)
        except:
            pass
        message = f"GRID PREPARATION: Results grid created."
        self.update_statusbar(message)

        # We have a results grid, enable the image velocimetry processors
        self.groupboxSpaceTimeParameters.setEnabled(True)
        self.groupboxSpaceTimeOptParameters.setEnabled(True)

    def change_line_num_points(self):
        """Update the number of grid points along a simple line"""
        self.number_grid_points_along_line = self.spinboxLineNumPoints.value()

    def change_xs_line_num_points(self):
        """Update the number of grid points along a defined cross-section line"""
        self.number_grid_points_along_xs_line = self.spinbocXsLineNumPoints.value()

    def change_horz_grid_size(self):
        """Update the horizontal grid size"""
        self.horz_grid_size = self.spinboxHorizGridSpacing.value()

    def change_vert_grid_size(self):
        """Update the vertical grid size"""
        self.vert_grid_size = self.spinboxVertGridSpacing.value()

    @staticmethod
    def get_table_as_dict(table: QtWidgets.QTableWidget):
        """Exports a QTableWidget object into a dictionary with keys corresponding to column header names."""
        num_rows = table.rowCount()
        num_columns = table.columnCount()
        headers = []
        result = {key: None for key in headers}
        for c in range(num_columns):
            item = table.horizontalHeaderItem(c)
            headers.append(item.text())

        for c in range(num_columns):
            items = []
            for r in range(num_rows):
                item = table.item(r, c)
                if item is not None:
                    text = item.text()
                else:
                    text = ""
                items.append(text)
            result[headers[c]] = items
        return result

    def set_table_from_dict(self, dictionary: dict, table: QtWidgets.QTableWidget):
        """Imports a supplied dictionary into a QTableWidget."""
        keys = list(dictionary.keys())
        values = list(dictionary.values())
        num_rows = len(dictionary[keys[0]])
        num_columns = len(keys)
        table.setColumnCount(num_columns)
        table.setRowCount(num_rows)
        table.setHorizontalHeaderLabels(keys)

        for row in range(num_rows):  # add items from array to QTableWidget
            for column in range(num_columns):
                item = values[column][row]
                table.setItem(row, column, QtWidgets.QTableWidgetItem(item))

        table.resizeColumnsToContents()
        table.resizeRowsToContents()

    def ortho_original_load_gcp_image(self, image_filename=None):
        """
        Load the GCP image for orthorectification and save a copy as '!calibration_image.jpg'.

        Parameters
        ----------
        image_filename : str, optional
            The file path of the GCP image to load. If not provided, defaults to
            the directory of the last loaded GCP image path.

        Notes
        -----
        This function loads the specified GCP image, updates the application state,
        and saves a copy of the image as '!calibration_image.jpg' in the
        `self.swap_orthorectification` directory.
        """
        try:
            ss = self.sticky_settings.get("last_ortho_gcp_image_path")
            if ss is not None:
                self.last_ortho_gcp_image_path = ss
            else:
                self.last_ortho_gcp_image_path = QDir.homePath()
        except KeyError:
            self.last_ortho_gcp_image_path = QDir.homePath()

        if image_filename is not None:
            self.ortho_original_image.open(image_filename)
        else:
            self.ortho_original_image.open(
                os.path.dirname(self.last_ortho_gcp_image_path)
            )
        # Save a copy of the GCP image as '!calibration_image.jpg'
        try:
            destination_path = os.path.join(
                self.swap_orthorectification_directory, "!calibration_image.jpg"
            )
            shutil.copy(self.ortho_original_image.image_file_path, destination_path)
        except Exception as e:
            self.update_statusbar(f"Failed to save calibration image: {e}")

        self.set_qwidget_state_by_name(
            [
                # "ortho_original_image",
                "groupboxOrthoOrigImageTools",
                "toolbuttonOrthoOrigImageDigitizePoint",
            ],
            True,
        )
        self.ortho_original_image.setEnabled(True)
        message = (
            f"GCP Image Loaded. Drag a box or use scroll wheel to zoom, right-click to reset. A GCP table "
            f"must be loaded to continue."
        )
        self.update_statusbar(message)
        try:
            self.sticky_settings.set(
                "last_ortho_gcp_image_path", self.ortho_original_image.image_file_path
            )
        except KeyError:
            self.sticky_settings.new(
                "last_ortho_gcp_image_path", self.ortho_original_image.image_file_path
            )

    def ortho_flip_x_changed(self):
        self.is_ortho_flip_x = self.checkBoxOrthoFlipX.isChecked()

    def ortho_flip_y_changed(self):
        self.is_ortho_flip_y = self.checkBoxOrthoFlipY.isChecked()

    def homography_distance_conversion_tool(self):
        """Opens the Homogrpahy Distance Conversion tool"""
        logging.debug("##### Opening Homography Distance Conversion Tool")
        self.homography_tool = HomographyDistanceConversionTool(self)
        res = self.homography_tool.exec_()

    def estimate_stiv_sample_rate(self):
        """Opens the STIV Helper program"""
        logging.debug("#### Opening estimate STIV tool")
        if (
            self.extraction_frame_rate is not None
            and self.pixel_ground_scale_distance_m is not None
        ):
            self.estimate_stiv_frame_step_tool = StivHelper(
                frame_rate=self.extraction_frame_rate,
                gsd=self.pixel_ground_scale_distance_m,
            )
        else:
            self.estimate_stiv_frame_step_tool = StivHelper()
        res = self.estimate_stiv_frame_step_tool.exec_()

    def orthotable_change_selection(self):
        """Executes upon a change to the GCP Points table"""
        self.orthoPointsTable.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )

    def orthotable_update_cell(self):
        """Update the GCP Points Table cell"""
        if self.orthoPointsTable.selectionModel().hasSelection():
            row = self.orthotable_selected_row()
            column = self.orthotable_selected_column()
            newtext = QtWidgets.QTableWidgetItem(self.editLine.text())
            self.orthoPointsTable.setItem(row, column, newtext)

    def orthotable_get_item(self):
        """Get the selected item from the GCP Points Table"""
        item = self.orthoPointsTable.selectedItems()[0]
        row = self.orthotable_selected_row()
        column = self.orthotable_selected_column()
        if not item == None:
            name = item.text()
        else:
            name = ""
        # self.msg("'" + name + "' on Row " + str(row + 1) + " Column " + str(column + 1))
        self.orthoPointsTableLineEdit.setText(name)

    def orthotable_selected_row(self):
        """Return the current row of the GCP Points Table

        Returns:
            int: the current row
        """
        if self.orthoPointsTable.selectionModel().hasSelection():
            row = self.orthoPointsTable.selectionModel().selectedIndexes()[0].row()
            return int(row)

    def orthotable_selected_column(self):
        """Return the current columns of the GCP Points Table

        Returns:
            int: the current column
        """
        column = self.orthoPointsTable.selectionModel().selectedIndexes()[0].column()
        return int(column)

    def orthotable_finished_edit(self):
        """Update the main instance class attributes when the editing of the GCP table is complete"""
        self.orthotable_is_changed = True
        self.signal_orthotable_changed.emit(True)

    def orthotable_remove_row(self):
        """Remove the selected row"""
        if self.orthoPointsTable.rowCount() > 0:
            remove = QtWidgets.QMessageBox()
            remove.setText(
                "This will remove the selected row, and cannot be undone. Are you sure?"
            )
            remove.setStandardButtons(
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel
            )
            remove = remove.exec()

            if remove == QtWidgets.QMessageBox.Yes:
                row = self.orthotable_selected_row()
                self.orthoPointsTable.removeRow(row)
                self.orthotable_is_changed = True
                self.signal_orthotable_changed.emit(True)
            else:
                pass

    def orthotable_add_row(self):
        """Add a new row to the bottom of the table"""
        if self.orthoPointsTable.rowCount() > 0:
            if self.orthoPointsTable.selectionModel().hasSelection():
                row = self.orthotable_selected_row()
                item = QtWidgets.QTableWidgetItem("")
                self.orthoPointsTable.insertRow(row)
            else:
                row = 0
                item = QtWidgets.QTableWidgetItem("")
                self.orthoPointsTable.insertRow(row)
                self.orthoPointsTable.selectRow(0)
        else:
            self.orthoPointsTable.setRowCount(1)
        if self.orthoPointsTable.columnCount() == 0:
            self.orthotable_add_column()
            self.orthoPointsTable.selectRow(0)
        self.orthotable_is_changed = True
        self.signal_orthotable_changed.emit(True)

    def orthotable_clear_list(self):
        """Clear all selected items in the table"""
        self.orthoPointsTable.clear()
        self.orthotable_is_changed = True
        self.signal_orthotable_changed.emit(True)

    def orthotable_remove_column(self):
        """Remove the selected column from the table"""
        self.orthoPointsTable.removeColumn(self.orthotable_selected_column())
        self.orthotable_is_changed = True
        self.signal_orthotable_changed.emit(True)

    def orthotable_add_column(self):
        """Add a new column to the end of the table"""
        count = self.orthoPointsTable.columnCount()
        self.orthoPointsTable.setColumnCount(count + 1)
        self.orthoPointsTable.resizeColumnsToContents()
        self.orthotable_is_changed = True
        self.signal_orthotable_changed.emit(True)
        if self.orthoPointsTable.rowCount() == 0:
            self.orthotable_add_row()
            self.orthoPointsTable.selectRow(0)

    def orthotable_make_all_white(self):
        """Set the background of all table cells to white"""
        if self.orthotable_cell_colored:
            for row in range(self.orthoPointsTable.rowCount()):
                for column in range(self.orthoPointsTable.columnCount()):
                    item = self.orthoPointsTable.item(row, column)
                    if item is not None:
                        item.setForeground(Qt.black)
                        item.setBackground(QtGui.QColor("#e1e1e1"))
        self.orthotable_cell_colored = False

    def orthotable_load_csv_on_open(self, file_name):
        """Load the GCP CSV File into the application and UI."""
        if not file_name:
            return

        try:
            def prompt_units():
                choices = ("English", "Metric")
                idx = self.custom_dialog_index(
                    title="Ground Control Points Unit Selection",
                    message="Units not detected in GCP file.\nPlease select units used in the survey:",
                    choices=choices,
                )
                return choices[idx]

            from image_velocimetry_tools.file_management import load_and_parse_gcp_csv

            df, units = load_and_parse_gcp_csv(
                file_name=file_name,
                swap_ortho_path=self.swap_orthorectification_directory,
                unit_prompt_callback=prompt_units,
            )
            self.orthotable_file_survey_units = units

        except ValueError as e:
            QtWidgets.QErrorMessage().showMessage(str(e)).exec_()
            return
        except Exception as e:
            self.update_statusbar(f"Failed to load GCP CSV: {e}")
            return

        # Clear existing state
        self.ortho_original_image.clearPoints()
        self.ortho_original_image.clearPolygons()

        # Save DataFrame
        self.orthotable_dataframe = df.copy(deep=True)
        self.orthotable_populate_table(df)

        self.orthoPointsTable.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.orthotable_is_changed = False
        self.orthotable_set_current_file(file_name)
        self.orthoPointsTable.resizeColumnsToContents()
        self.orthoPointsTable.resizeRowsToContents()
        self.orthoPointsTable.selectRow(0)

        # Enable controls
        self.toolbuttonOrthoOrigImageDigitizePoint.setEnabled(True)
        self.groupboxExportOrthoFrames.setEnabled(True)
        self.is_ortho_table_loaded = True
        self.signal_orthotable_check_units.emit()

        # Save to project
        try:
            dest = os.path.join(self.swap_orthorectification_directory,
                                "ground_control_points.csv")
            shutil.copy(file_name, dest)
        except Exception as e:
            self.update_statusbar(
                f"Failed to save GCP table to project: {e}")

            # self.orthotable_dataframe = df.copy(deep=True)
            #
            # # Put the data into the gui Table, respecting display units
            # self.orthotable_populate_table(self.orthotable_dataframe)
            #
            # # Clean up
            # self.orthoPointsTable.setSelectionBehavior(
            #     QtWidgets.QAbstractItemView.SelectRows
            # )
            # self.orthotable_is_changed = False
            # self.ortho_original_image.clearPoints()
            # self.ortho_original_image.clearPolygons()
            # self.orthotable_set_current_file(file_name)
            # self.orthoPointsTable.resizeColumnsToContents()
            # self.orthoPointsTable.resizeRowsToContents()
            # self.orthoPointsTable.selectRow(0)
            # self.toolbuttonOrthoOrigImageDigitizePoint.setEnabled(True)
            # # self.doubleSpinBoxRectificationWaterSurfaceElevation.setEnabled(True)
            # self.groupboxExportOrthoFrames.setEnabled(True)
            # self.is_ortho_table_loaded = True
            # self.signal_orthotable_check_units.emit()
            #
            # # Save a copy of the table in the project structure
            # try:
            #     destination_path = os.path.join(
            #         self.swap_orthorectification_directory, "ground_control_points.csv"
            #     )
            #     shutil.copy(file_name, destination_path)
            # except Exception as e:
            #     self.update_statusbar(
            #         f"Failed to save GCP table to project " f"structure: {e}"
            #     )

    def orthotable_populate_table(self, dataframe):
        """Populate the GCP Table"""
        # Now we can populate the table
        self.orthoPointsTable.setColumnCount(len(dataframe.columns))
        self.orthoPointsTable.setRowCount(len(dataframe.index))

        # The GCP table in should always be meters in the IvyTools Instance
        header_list = dataframe.columns.tolist()
        for i in range(len(dataframe.index)):
            for j in range(len(dataframe.columns)):
                if j >= 1 and j <= 3:
                    item = dataframe.iat[i, j] * self.survey_units["L"]
                else:
                    item = dataframe.iat[i, j]
                self.orthoPointsTable.setItem(
                    i, j, QtWidgets.QTableWidgetItem(str(item))
                )
        for j in range(len(dataframe.columns)):
            m = QtWidgets.QTableWidgetItem(header_list[j])
            self.orthoPointsTable.setHorizontalHeaderItem(j, m)
        self.orthotable_has_headers = True
        self.orthoPointsTable.setHorizontalHeaderLabels(header_list)

        # Make sure the table headers and units are correct
        # self.orthotable_change_units()
        self.orthotable_update_table_headers()
        is_points_to_plot = True
        if is_points_to_plot:
            self.ortho_original_plot_points()

    def orthotable_init(self):
        """Executes at startup, sets up the GCP ortho table."""
        headers = [
            "# ID",
            f"X {self.survey_units['label_L']}",
            f"Y {self.survey_units['label_L']}",
            f"Z {self.survey_units['label_L']}",
            "X (pixel)",
            "Y (pixel)",
            "Error X (pixel)",
            "Error Y (pixel)",
            "Tot. Error (pixel)",
            "Use in Rectification",
            "Use in Validation",
        ]
        self.orthoPointsTable.setColumnCount(len(headers))
        self.orthotable_has_headers = True
        self.orthoPointsTable.setHorizontalHeaderLabels(headers)
        self.orthoPointsTable.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )
        self.orthotable_is_changed = False
        self.orthoPointsTable.resizeColumnsToContents()
        self.orthoPointsTable.resizeRowsToContents()
        self.orthoPointsTable.selectRow(0)
        self.is_ortho_table_loaded = False

    def orthotable_update_table_headers(self):
        """Update table headers based on survey units."""
        headers = [
            "# ID",
            f"X {self.survey_units['label_L']}",
            f"Y {self.survey_units['label_L']}",
            f"Z {self.survey_units['label_L']}",
            "X (pixel)",
            "Y (pixel)",
            "Error X (pixel)",
            "Error Y (pixel)",
            "Tot. Error (pixel)",
            "Use in Rectification",
            "Use in Validation",
        ]
        self.orthoPointsTable.setHorizontalHeaderLabels(headers)
        self.orthoPointsTable.resizeColumnsToContents()
        self.orthoPointsTable.resizeRowsToContents()

    def orthotable_change_units(self):
        """Apply unit conversions to data displayed in the table"""
        table_units = self.orthotable_file_survey_units
        if False:
            if table_units != self.display_units:
                if self.display_units == "Metric":
                    c = 1 / units_conversion("English")["L"]  # Eng to Metric
                if self.display_units == "English":
                    c = units_conversion("English")["L"]  # Metric to Eng
            else:
                c = 1
        c = 1
        orthotable_dataframe = self.orthotable_dataframe.copy(deep=True)
        orthotable_dataframe.iloc[:, 1:4] *= c
        self.orthotable_populate_table(orthotable_dataframe)

    def ortho_original_refresh_plot(self, event):
        """Refresh the original image pane view"""
        if event:
            # Update the orthotable_dataframe
            # with the X, Y pixel info from the gui table
            if self.orthotable_dataframe is None:
                ortho_dict = self.get_table_as_dict(self.orthoPointsTable)
                self.orthotable_dataframe = pd.DataFrame.from_dict(ortho_dict)

            with self.wait_cursor():
                current_table_data = self.get_table_as_dict(self.orthoPointsTable)
                keys = list(current_table_data.keys())
                if len(keys) >= 11:
                    xpixel_key = keys[4]
                    ypixel_key = keys[5]
                    rectification_key = "Use in Rectification"
                    validation_key = "Use in Validation"

                    # Check if any values are not empty for the xpixel_key
                    non_empty_values_xpixel = [
                        (index, value)
                        for index, value in enumerate(current_table_data[xpixel_key])
                        if value != ""
                    ]

                    # Check if any values are not empty for the ypixel_key
                    non_empty_values_ypixel = [
                        (index, value)
                        for index, value in enumerate(current_table_data[ypixel_key])
                        if value != ""
                    ]

                    # Check if any values are not empty for the rectification_key
                    try:
                        non_empty_values_rectification = [
                            (index, value)
                            for index, value in enumerate(
                                current_table_data[rectification_key]
                            )
                            if value != ""
                        ]
                    except KeyError:
                        non_empty_values_rectification = None
                    try:
                        non_empty_values_validation = [
                            (index, value)
                            for index, value in enumerate(
                                current_table_data[validation_key]
                            )
                            if value != ""
                        ]
                    except KeyError:
                        non_empty_values_validation = None

                    if non_empty_values_xpixel and non_empty_values_ypixel:
                        for index_x, value_x in non_empty_values_xpixel:
                            self.orthotable_dataframe.iloc[
                                index_x,
                                self.orthotable_dataframe.columns.get_loc("X (pixel)"),
                            ] = value_x
                        for index_y, value_y in non_empty_values_ypixel:
                            self.orthotable_dataframe.iloc[
                                index_y,
                                self.orthotable_dataframe.columns.get_loc("Y (pixel)"),
                            ] = value_y
                        if non_empty_values_rectification is not None:
                            for index_r, value_r in non_empty_values_rectification:
                                self.orthotable_dataframe.iloc[
                                    index_r,
                                    self.orthotable_dataframe.columns.get_loc(
                                        rectification_key
                                    ),
                                ] = value_r
                        if non_empty_values_validation is not None:
                            for index_v, value_v in non_empty_values_validation:
                                self.orthotable_dataframe.iloc[
                                    index_v,
                                    self.orthotable_dataframe.columns.get_loc(
                                        validation_key
                                    ),
                                ] = value_v

            self.ortho_original_plot_points()

    def ortho_original_plot_points(self):
        """Plot the current GCP points on the original image"""
        # Grab the current points from the table
        self.ortho_original_image.clearPoints()
        self.ortho_original_image.clearPolygons()
        rectification_points = self.get_orthotable_points_to_plot(
            which_points="rectification"
        )
        validation_points = self.get_orthotable_points_to_plot(
            which_points="validation"
        )
        # self.ortho_original_image.addLabeledPoint(rectification_points)
        # self.ortho_original_image.addLabeledPoint(validation_points)
        if rectification_points is not None:
            self.ortho_original_image.scene.set_current_instruction(
                Instructions.ADD_POINTS_INSTRUCTION,
                points=rectification_points["points"],
                labels=rectification_points["labels"],
            )

    def get_orthotable_points_to_plot(self, which_points="rectification"):
        """Get the points to plot from the GCP Points Table.

        Args:
            which_points (str, optional): specify which points to grab.
               Defaults to "rectification".

        Returns:
            dict: a dict containing the points and labels to plot
        """
        try:
            df = self.orthotable_dataframe.copy(deep=True)

            # Replace "N/A" with NaN to simplify filtering
            df.replace("N/A", np.nan, inplace=True)

            # Convert relevant columns to numeric if needed
            for col in ["X", "Y", "Z", "X (pixel)", "Y (pixel)"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Convert string booleans to real booleans
            if which_points.lower() == "rectification":
                use_col = "Use in Rectification"
                symbol = "fiducial 2"
                color = "red"
            elif which_points.lower() == "validation":
                use_col = "Use in Validation"
                symbol = "fiducial 2"
                color = "green"
            else:
                return  # Unknown option

            df[use_col] = df[use_col].map(string_to_boolean)

            # Filter rows meeting all conditions
            valid = df[["X (pixel)", "Y (pixel)"]].notna().all(axis=1) & df[
                use_col]

            # Get filtered data
            filtered = df[valid]
            points = {
                "points": filtered[["X (pixel)", "Y (pixel)"]].to_numpy(),
                "coordinates": filtered[["X", "Y", "Z"]].to_numpy(),
                "labels": filtered["# ID"].tolist(),
                "zoom": self.ortho_original_image_zoom_factor,
                "symbology": symbol,
                "symbol size": 15,
                "color": color,
            }
            return points

        except Exception as e:
            logging.warning(f"Error loading GCP points data in get_orthotable_points_to_plot: {e}")
            return

    def orthotable_load_csv(self):
        """Executes if the user clicks the load GCP points button (or accesses from the menu)"""
        try:
            ss = self.sticky_settings.get("last_orthotable_file_name")
            self.last_orthotable_file_name = ss
        except KeyError:
            self.last_orthotable_file_name = QDir.homePath()

        if self.orthotable_is_changed:
            # quit_msg = "The Document was changed.<br>Do you want to save changes?"
            # reply = QtWidgets.QMessageBox.question(self, 'Save Confirmation',
            #                                        quit_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            # if reply == QtWidgets.QMessageBox.Yes:
            #     # self.saveOnQuit()
            #     pass
            pass
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open CSV containing Ground Control Point data",
            self.last_orthotable_file_name,
            "CSV (*.csv *.tsv *.txt)",
        )
        if file_name:
            try:
                self.sticky_settings.set(
                    "last_orthotable_file_name", self.last_orthotable_file_name
                )
            except KeyError:
                self.sticky_settings.new(
                    "last_orthotable_file_name", self.last_orthotable_file_name
                )
            self.orthotable_load_csv_on_open(file_name)

    def orthotable_set_current_file(self, file_name):
        """Set the current GCP file

        Args:
            file_name (str): path to the file
        """
        self.orthotable_file_name = file_name
        self.orthotable_fname = os.path.splitext(str(file_name))[0].split("/")[-1]

    def orthotable_save_csv(self):
        """Save the GCP points table as is, in the display units"""
        try:
            ss = self.sticky_settings.get("last_orthotable_file_name")
        except KeyError:
            ss = f"{(QDir.homePath())}{os.sep}IVy_Points_Table_{self.display_units}.csv"

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save CSV containing Ground Control Point data",
            ss,
            "CSV (*.csv)",
        )
        if file_name is not None:  # User did not hit cancel
            dict = self.get_table_as_dict(self.orthoPointsTable)
            pd.DataFrame(dict).fillna("").to_csv(file_name, index=False)
            try:
                self.sticky_settings.set("last_orthotable_file_name", file_name)
            except KeyError:
                self.sticky_settings.new("last_orthotable_file_name", file_name)

    def draw_cross_section_line(self):
        """Draw the cross section line on the correct image"""
        if self.toolbuttonDrawCrossSection.isChecked():
            self.perspective_xs_image.clearLines()
            self.rectified_xs_image.clearLines()
            if self.radioButtonLeft.isChecked():
                self.cross_section_start_bank = "left"
            if self.radioButtonRight.isChecked():
                self.cross_section_start_bank = "right"

            # Check which image User is drawing the line on
            if self.radioButtonOriginalImage.isChecked():
                if self.rectification_method == "camera matrix":
                    self.warning_dialog(
                        "Cross-section Warning",
                        "When using the 3D camera solution, you "
                        "must verify that the the cross-section "
                        "line on the RECTIFIED image is in the "
                        "correct place.\n\n"
                        "Best results are to draw the cross-section "
                        "line on the RECTIFIED image.",
                        style="ok",
                    )
                    self.toolbuttonDrawCrossSection.setChecked(False)
                    self.toolbuttonDrawCrossSection.repaint()
                    self.signal_cross_section_exists.emit(False)
                    return

                # Draw the line, double left click to stop
                self.perspective_xs_image.scene.set_current_instruction(
                    Instructions.SIMPLE_LINE_INSTRUCTION
                )
                self.perspective_xs_image.scene.line_item[-1].setPen(
                    QtGui.QPen(QtGui.QColor("yellow"), 5)
                )
                message = (
                    f"CROSS-SECTION: Draw the cross-section location on the Original image (top), starting from"
                    f"the {self.cross_section_start_bank} bank. "
                    f"Uncheck the add cross-section button when finished."
                )

            if self.radioButtonRectifiedImage.isChecked():
                self.rectified_xs_image.scene.set_current_instruction(
                    Instructions.SIMPLE_LINE_INSTRUCTION
                )
                self.rectified_xs_image.scene.line_item[-1].setPen(
                    QtGui.QPen(QtGui.QColor("yellow"), 5)
                )
                message = (
                    f"CROSS-SECTION: Draw the cross-section location on the Rectified image (bottom), starting from"
                    f"the {self.cross_section_start_bank} bank. "
                    f"Uncheck the add cross-section button when finished."
                )
            self.update_statusbar(message)
        else:
            self.perspective_xs_image.scene.set_current_instruction(
                Instructions.NO_INSTRUCTION
            )
            self.rectified_xs_image.scene.set_current_instruction(
                Instructions.NO_INSTRUCTION
            )
            self.toolbuttonDrawCrossSection.setChecked(False)
            self.toolbuttonDrawCrossSection.repaint()

            if self.cross_section_top_width_m > 0:
                self.set_tab_icon("tabCrossSectionGeometry", "good")
            self.signal_cross_section_exists.emit(True)

    def set_cross_section_line(self):
        """Emit a call back when a cross-section line is set"""
        self.signal_cross_section_exists.emit(True)

    def clear_cross_section_line(self):
        """Clear the cross-section line"""
        self.perspective_xs_image.clearLines()
        self.rectified_xs_image.clearLines()
        message = "CROSS-SECTION: Cleared digitized cross section line."
        self.update_statusbar(message)

    def cross_section_manager_eps(self):
        """Executes when the User updates a Rectified Endpoints coordinate

        """
        method = self.rectification_method
        new_eps = np.array(
            [[
                [self.sbLeftBankXRectifiedCoordPixels.value(),
                 self.sbLeftBankYRectifiedCoordPixels.value()],
                [self.sbRightBankRectifiedXCoordPixels.value(),
                 self.sbRightBankRectifiedYCoordPixels.value()]
            ]],
        )
        self.rectified_xs_image.clearLines()
        self.rectified_xs_image.scene.set_current_instruction(
            Instructions.ADD_LINE_BY_POINTS,
            points=[tuple(point) for point in new_eps[0]],
        )
        self.rectified_xs_image.scene.line_item[-1].setPen(
            QtGui.QPen(QtGui.QColor("yellow"), 5)
        )

        # TODO: respect the EPs and plot in the other image (perspective or
        #  rectified).
        if method is None:
            return
        if self.radioButtonOriginalImage.isChecked():
            if method == "scale":
                pass
            if method == "homography":
                pass
            if method == "camera matrix":
                pass
        if self.radioButtonRectifiedImage.isChecked():
            if method == "scale":
                pass
            if method == "homography":
                pass
            if method == "camera matrix":
                pass

        # Set the line pixel length and enable the ability to draw the line
        # in the Grid Prep tab
        line_length = self.rectified_xs_image.scene.line_item[
            -1].line_length
        logging.debug(
            f"CROSS-SECTION: The drawn XS is {line_length} " f"pixels long.")
        self.cross_section_length_pixels = line_length
        self.cross_section_line = self.rectified_xs_image.lines_ndarray()
        self.cross_section_rectified_eps = (
            self.rectified_xs_image.lines_ndarray().reshape(2, 2)
        )
        self.cross_section_line_exists = True
        self.set_qwidget_state_by_name("CrossSectionPage", True)
        if self.is_area_comp_loaded:
            self.enable_disable_tabs(self.tabWidget,
                                     "tabGridPreparation", True)

    def cross_section_manager(self):
        """Executes when the User completes drawing a cross-section.

        This function is connected to the signal_cross_section_exists slot.
        """
        method = self.rectification_method
        if method is None:
            return
        if self.radioButtonOriginalImage.isChecked():
            logging.debug("XS drawn on Original Image")

            if method == "scale":
                # If scale, it is the same image, just copy the line
                line_eps = self.perspective_xs_image.lines_ndarray()
                self.rectified_xs_image.scene.set_current_instruction(
                    Instructions.ADD_LINE_BY_POINTS,
                    points=[tuple(point) for point in line_eps[0]],
                )
                self.rectified_xs_image.scene.line_item[-1].setPen(
                    QtGui.QPen(QtGui.QColor("yellow"), 5)
                )
            if method == "homography":
                H = np.array(self.rectification_parameters["homography_matrix"])
                line = self.perspective_xs_image.scene.line_item[-1]
                points = np.array([(point.x(), point.y()) for point in line.m_points])
                transformed_points = transform_points_with_homography(points, H)
                self.rectified_xs_image.clearLines()
                self.rectified_xs_image.scene.set_current_instruction(
                    Instructions.ADD_LINE_BY_POINTS,
                    points=transformed_points,
                )
                self.rectified_xs_image.scene.line_item[-1].setPen(
                    QtGui.QPen(QtGui.QColor("yellow"), 5)
                )
            if method == "camera matrix":
                if self.radioButtonOriginalImage.isChecked():
                    logging.error(
                        "cross_section_manager: User tried drawing XS "
                        "line on perspective image, but is using 3D "
                        "camera matrix orthorectification."
                    )
                    return
                from image_velocimetry_tools.orthorectification import (
                    get_homographic_coordinates_2D,
                    projective_matrix_to_camera_matrix,
                )

                line = self.perspective_xs_image.scene.line_item[-1]
                points = np.array([(point.x(), point.y()) for point in line.m_points])
                points_h = get_homographic_coordinates_2D(points).T

                # points_h = get_homographic_coordinates_3D(points)
                # points_h = np.vstack([points_h, points_h])
                P = np.array(self.rectification_parameters["camera_matrix"])

                C, r = projective_matrix_to_camera_matrix(P)
                transformed_points = points_h @ r
                transformed_points = transformed_points[:, :2]
                # Find the minimum coordinates along each axis
                min_x = np.min(transformed_points[:, 0])
                min_y = np.min(transformed_points[:, 1])

                # Calculate the translation vector to shift all points to the positive quadrant
                translation_vector = np.array([max(0, -min_x), max(0, -min_y)])

                # Translate the points
                translated_points = transformed_points + translation_vector
                self.rectified_xs_image.clearLines()
                self.rectified_xs_image.scene.set_current_instruction(
                    Instructions.ADD_LINE_BY_POINTS,
                    points=translated_points,
                )
                self.rectified_xs_image.scene.line_item[-1].setPen(
                    QtGui.QPen(QtGui.QColor("yellow"), 5)
                )

            line_eps = self.perspective_xs_image.lines_ndarray()
            line_length = self.perspective_xs_image.scene.line_item[-1].line_length
            self.is_cross_section_grid = True
        if self.radioButtonRectifiedImage.isChecked():
            # TODO: here, I need to project the drawn line onto the
            #  perspective_xs_image. I am debating disabling this, and forcing
            #  the user to draw on the original image, and just showing the
            #  rectified results
            method = self.rectification_method
            if method == "scale":
                # If scale, it is the same image, just copy the line
                line_eps = self.rectified_xs_image.lines_ndarray()
                self.perspective_xs_image.scene.set_current_instruction(
                    Instructions.ADD_LINE_BY_POINTS,
                    points=[tuple(point) for point in line_eps[0]],
                )
                self.perspective_xs_image.scene.line_item[-1].setPen(
                    QtGui.QPen(QtGui.QColor("yellow"), 5)
                )
            if method == "homography":
                H_inv = np.linalg.inv(
                    self.rectification_parameters["homography_matrix"]
                )
                line = self.rectified_xs_image.scene.line_item[-1]
                points = np.array([(point.x(), point.y()) for point in line.m_points])
                transformed_points = transform_points_with_homography(points, H_inv)
                self.perspective_xs_image.clearLines()
                self.perspective_xs_image.scene.set_current_instruction(
                    Instructions.ADD_LINE_BY_POINTS,
                    points=transformed_points,
                )
                self.perspective_xs_image.scene.line_item[-1].setPen(
                    QtGui.QPen(QtGui.QColor("yellow"), 5)
                )
            if method == "camera matrix":
                line = self.rectified_xs_image.scene.line_item[-1]
                points = np.array([(point.x(), point.y()) for point in line.m_points])
                # self.cam.get_inverse_top_view_point(points[0])
                # points_h = get_homographic_coordinates_2D(points).T
                # P = np.array(self.rectification_parameters["camera_matrix"])
                # C, r = projective_matrix_to_camera_matrix(P)
                # transformed_points = np.dot(C, points_h.T).T
                # transformed_points[:, 0] /= transformed_points[:, 2]
                # transformed_points[:, 1] /= transformed_points[:, 2]
                # transformed_points = transformed_points[:,:2]
                # self.perspective_xs_image.clearLines()
                # self.perspective_xs_image.scene.set_current_instruction(
                #     Instructions.ADD_LINE_BY_POINTS,
                #     points=transformed_points,
                # )

            logging.debug("XS drawn on Rectified Image")
            line_length = self.rectified_xs_image.scene.line_item[-1].line_length

        logging.debug(f"XS start bank: {self.cross_section_start_bank}")

        logging.debug(
            f"Cross-section pixel coords:\n {self.rectified_xs_image.lines_ndarray()}"
        )
        # Set the cross-section EP Spin-boxes
        if self.rectified_xs_image.scene.line_item:
            xs_line = self.rectified_xs_image.scene.line_item[-1]
            self.sbLeftBankXRectifiedCoordPixels.setValue(
                xs_line.m_points[0].x()
            )
            self.sbLeftBankYRectifiedCoordPixels.setValue(
                xs_line.m_points[0].y()
            )
            self.sbRightBankRectifiedXCoordPixels.setValue(
                xs_line.m_points[1].x()
            )
            self.sbRightBankRectifiedYCoordPixels.setValue(
                xs_line.m_points[1].y()
            )

        # Set the line pixel length and enable the ability to draw the line
        # in the Grid Prep tab
        line_length = self.rectified_xs_image.scene.line_item[-1].line_length
        logging.debug(f"CROSS-SECTION: The drawn XS is {line_length} " f"pixels long.")
        self.cross_section_length_pixels = line_length
        self.cross_section_line = self.rectified_xs_image.lines_ndarray()
        self.cross_section_rectified_eps = (
            self.rectified_xs_image.lines_ndarray().reshape(2, 2)
        )
        self.cross_section_line_exists = True
        self.set_qwidget_state_by_name("CrossSectionPage", True)
        if self.is_area_comp_loaded:
            self.enable_disable_tabs(self.tabWidget, "tabGridPreparation", True)

    def process_stiv_results(self, thread_results):
        """Executes when the STIV thread has return results

        Args:
            thread_results (str): the STIV results
        """
        logging.debug(f"{thread_results}")


    def process_stiv_thread_finished(self):
        """Executes when the STIV thread has completed"""
        logging.debug(f"The STIV thread has finished.")
        message = f"SPACE-TIME IMAGE VELOCIMETRY: Processing complete. "

        try:
            # Save a CSV of the results
            X = self.results_grid[:, 0].astype(float)
            Y = self.results_grid[:, 1].astype(float)

            # TODO: add a reference velocity vector
            if self.process_step == "Process STIV Exhaustive":
                directions_deg_geo = self.stiv.directions
                directions_rad_ari = np.radians(
                    geographic_to_arithmetic(directions_deg_geo)
                )
                mfd_geog = np.nanmean(directions_deg_geo)

                # Ensure the U,V components are in the correct quadrant
                # orientation
                U, V = calculate_uv_components(
                    self.stiv.magnitudes_mps, directions_deg_geo
                )

                M = np.sqrt(U**2 + V**2)
                D = self.stiv.directions  # geo
                csv_file_path = os.path.join(
                    self.swap_velocities_directory, "stiv_results.csv"
                )

                (
                    vectors,
                    norm_vectors,
                    normal_unit_vector,
                    scalar_projections,
                    tagline_dir_geog,
                    mean_flow_dir_geog,
                ) = compute_vectors_with_projections(X, Y, U, V)
                header = (
                    "X (pixel),"
                    "Y (pixel,"
                    "U (m/s),"
                    "V (m/s),"
                    "Magnitude (m/s),"
                    "Normal Magnitude (m/s),"
                    "Vector Direction (deg),"
                    "Tagline Direction (deg),"
                    "Mean Flow Direction (deg)"
                )
                logging.debug(
                    f"STIV: Tagline direction LEFT to RIGHT (geog): "
                    f"{tagline_dir_geog[0]:.2f}°"
                )
                logging.debug(
                    f"STIV: Mean flow direction (geog): "
                    f"{mean_flow_dir_geog[0]:.2f}°"
                )
                data = np.column_stack(
                    (
                        X,
                        Y,
                        U,
                        V,
                        M,
                        scalar_projections,
                        D,
                        tagline_dir_geog,
                        mean_flow_dir_geog,
                    )
                )
                np.savetxt(
                    csv_file_path,
                    data,
                    delimiter=",",
                    header=header,
                    comments="",
                    fmt="%1.3f",
                )

                # Clear existing vectors, then Plot the new vectors
                # Check for NaN values in U and V arrays prior to plotting
                nan_indices = np.logical_or(
                    np.isnan(vectors[:, 2]), np.isnan(vectors[:, 3])
                )
                self.stiv.imageBrowser.clearLines()
                if np.all(nan_indices):
                    logging.warning(
                        "STIV: There are no valid velocities, check settings."
                    )
                    scalar_projections = np.zeros_like(M)
                else:
                    vectors_draw = quiver(
                        vectors[~nan_indices, 0],
                        vectors[~nan_indices, 1],
                        vectors[~nan_indices, 2],
                        vectors[~nan_indices, 3],
                        global_scale=self.vector_scale,
                    )
                    norm_vectors_draw = quiver(
                        norm_vectors[~nan_indices, 0],
                        norm_vectors[~nan_indices, 1],
                        norm_vectors[~nan_indices, 2],
                        norm_vectors[~nan_indices, 3],
                        global_scale=self.vector_scale,
                    )
                    plot_quivers(self.stiv.imageBrowser, vectors_draw,
                                 "green", Qt.DotLine)
                    plot_quivers(self.stiv.imageBrowser, norm_vectors_draw,
                                 "yellow", Qt.SolidLine)
                self.stiv.magnitude_normals_mps = np.abs(scalar_projections)
                self.stiv_exists = True
                self.toolButtonExportPDF.setEnabled(True)
                self.actionSummary_Report_PDF.setEnabled(True)
                self.set_tab_icon("tabImageVelocimetry", "good")
                self.set_tab_icon(
                    "tabSTIVExhaustive", "good", self.tabWidget_ImageVelocimetryMethods
                )
                self.enable_disable_tabs(self.tabWidget, "tabDischarge", True)

                # Save the STIs if enabled
                if self.checkboxSaveSTIs.isChecked():
                    logging.debug("Saving the STI images")

                    for idx, image in enumerate(self.stiv.sti_array):
                        # scaled_image = exposure.rescale_intensity(image,
                        #                                           in_range='image',
                        #                                           out_range=(0,
                        #                                                      255)).astype(
                        #     np.uint8)
                        img = Image.fromarray(
                            image.astype(np.uint8)
                        )  # Convert to uint8 if necessary
                        img.save(
                            os.path.join(
                                self.swap_image_directory, f"STI_{idx:04d}.jpg"
                            )
                        )

                # Build the STI Review Table
                self.tabSpaceTimeImageReview.setEnabled(True)
                self.is_stis = True
                self.sti.table_load_data(
                    sti_images=glob.glob(
                        os.path.join(self.swap_image_directory, "STI*.jpg")
                    )
                )
                self.set_tab_icon(
                    "tabSpaceTimeImageReview",
                    "good",
                    self.tabWidget_ImageVelocimetryMethods,
                )

            elif self.process_step == "Process STIV Optimized":
                directions_rad_ari = np.radians(self.stiv_opt.directions)
                U = self.stiv_opt.magnitudes_mps * np.cos(directions_rad_ari)
                V = self.stiv_opt.magnitudes_mps * np.sin(directions_rad_ari)
                M = np.sqrt(U**2 + V**2)
                D = self.stiv_opt.directions
                csv_file_path = os.path.join(
                    self.swap_velocities_directory, "stiv_opt_results.csv"
                )

                # Clear existing vectors, then Plot the new vectors
                # Check for NaN values in U and V arrays prior to plotting
                nan_indices = np.isnan(U) | np.isnan(V)
                self.stiv_opt.imageBrowser.clearLines()

                if np.all(nan_indices):
                    logging.warning(
                        "STIV-Opt: There are no valid velocities, check settings."
                    )
                else:
                    vectors = quiver(
                        X[~nan_indices],
                        Y[~nan_indices],
                        U[~nan_indices],
                        V[~nan_indices],
                        global_scale=self.vector_scale,
                    )
                    for vector in vectors:
                        self.stiv_opt.imageBrowser.scene.set_current_instruction(
                            Instructions.ADD_LINE_BY_POINTS, points=vector
                        )
                        self.stiv_opt.imageBrowser.scene.line_item[-1].setPen(
                            QtGui.QPen(QtGui.QColor("yellow"), 3)
                        )
                self.stiv_opt_exists = True
            # data = np.column_stack((X, Y, U, V, M, D))

            # np.savetxt(
            #     csv_file_path, data, delimiter=",", header="X,Y,U,V,M,D", comments=""
            # )

            # Clean up
            self.update_statusbar(message)
            self.progressBar.hide()
            self.progressBar.setValue(
                0
            )  # ensures the progress bar is ready to go for next time
            # if self.process_step == "Process STIV Exhaustive":
            #     self.stiv.imageBrowser.scene.load_image(save_path)
            # elif self.process_step == "Process STIV Optimized":
            #     self.stiv_opt.imageBrowser.scene.load_image(save_path)
        except BaseException as e:
            logging.error(
                "STIV THREAD: Could not parse results. Do you have" " a grid already?"
            )
            logging.error(e)

    def plot_manual_vectors(self, event):
        """Triggered when the STI Review Tab table is updated. Plot any
        manual velocity vectors in the STIV tab image."""

        idx = event.get("idx", None)
        manual_velocity = event.get("manual_velocity", None)
        normal_direction_geo = event.get("normal_direction_geo", None)

        if (idx is not None and manual_velocity is not None and
                normal_direction_geo) is not None:
            color = "red"
            # Any of these colors
            # https://www.w3.org/TR/SVG11/types.html#ColorKeywords

            # Clear any lines in the scene of the currnet color
            self.stiv.imageBrowser.clearLinesByColor(color)

            X = self.results_grid[idx, 0].astype(float)
            Y = self.results_grid[idx, 1].astype(float)
            U, V = calculate_uv_components(
                manual_velocity[idx], normal_direction_geo[idx]
            )
            vectors = quiver(X, Y, U, -V, global_scale=self.vector_scale,
                             )

            plot_quivers(self.stiv.imageBrowser, vectors,
                         color, Qt.SolidLine)
        else:
            return


    def stiv_update_search_line_visual(self):
        """Update the search line results in the STIV tab"""
        # search_line = np.array([[0, 0], [0, 0]])
        if np.any(self.results_grid):
            self.stiv.imageBrowser.clearLines()
            for point in self.results_grid:
                # Create a line with start coord of point
                # take self.stiv_phi_origin as the line direction
                # take self.stiv_num_pixels as the line length
                x_start, y_start = point
                search_line = calculate_endpoint(
                    self.stiv_phi_origin, self.stiv_num_pixels, x_start, y_start
                )
                search_range1 = calculate_endpoint(
                    self.stiv_phi_origin - self.stiv_phi_range,
                    self.stiv_num_pixels,
                    x_start,
                    y_start,
                )
                search_range2 = calculate_endpoint(
                    self.stiv_phi_origin + self.stiv_phi_range,
                    self.stiv_num_pixels,
                    x_start,
                    y_start,
                )

                self.stiv.imageBrowser.scene.set_current_instruction(
                    Instructions.ADD_LINE_BY_POINTS, points=search_line
                )
                self.stiv.imageBrowser.scene.line_item[-1].setPen(
                    QtGui.QPen(QtGui.QColor("red"), 3)
                )
                self.stiv.imageBrowser.scene.set_current_instruction(
                    Instructions.ADD_LINE_BY_POINTS, points=search_range1
                )
                self.stiv.imageBrowser.scene.line_item[-1].setPen(
                    QtGui.QPen(QtGui.QColor("red"), 2, Qt.DotLine)
                )
                self.stiv.imageBrowser.scene.set_current_instruction(
                    Instructions.ADD_LINE_BY_POINTS, points=search_range2
                )
                self.stiv.imageBrowser.scene.line_item[-1].setPen(
                    QtGui.QPen(QtGui.QColor("red"), 2, Qt.DotLine)
                )

    def stiv_gaussian_blur_changed(self):
        """Set the STIV Gaussian blur"""
        self.stiv_gaussian_blur_sigma = self.doublespinboxStivGaussianBlurSigma.value()
        self.stiv_update_search_line_visual()

    def stiv_dphi_changed(self):
        """Set the STIV delta phi"""
        self.stiv_dphi = self.spinboxSTIVdPhi.value()
        self.stiv_update_search_line_visual()

    def stiv_search_line_distance_changed(self):
        """Set the STIV search line distance"""
        # Calculate the value
        search_len_ft = self.doublespinboxStivSearchLineDistance.value()
        stiv_num_pixels_value = int(
            (search_len_ft / self.survey_units["L"])
            / self.pixel_ground_scale_distance_m
        )

        # Ensure it's even
        if stiv_num_pixels_value % 2 != 0:
            stiv_num_pixels_value += 1

        self.stiv_num_pixels = stiv_num_pixels_value
        self.stiv_update_search_line_visual()
        self.stiv_search_line_length_m = (
            self.stiv_num_pixels * self.pixel_ground_scale_distance_m
        )

    def stiv_phi_origin_changed(self):
        """Set the STIV phi"""
        self.stiv_phi_origin = self.spinboxSTIVPhiOrigin.value()
        self.stiv_update_search_line_visual()

    def stiv_phi_range_changed(self):
        """Set the STIV phi range"""
        self.stiv_phi_range = self.spinboxSTIVPhiRange.value()
        self.stiv_update_search_line_visual()

    def stiv_max_vel_threshold_changed(self):
        """Set the STIV maximum velocity threshold"""
        new_value_m = (self.spinboxSTIVMaxVelThreshold.value() /
                       self.survey_units["L"])
        self.stiv_max_vel_threshold_mps = new_value_m
        self.stiv_update_search_line_visual()

    def stiv_opt_max_vel_threshold_changed(self):
        """Set the STIV optimized maximum velocity thresholds"""
        self.stiv_opt_max_vel_threshold_mps = self.spinboxSTIVOptMaxVelThreshold.value()

    def about_dialog(self):
        """Opens the IVyTools about dialog"""
        message = (
            f"<p>{self.__program_name__}</p>"
            f"<p>Version {self.__version__}</p>"
            f"<p>Created by: <br>"
            f"&emsp;< a href='https://www.usgs.gov/staff-profiles/frank-l-engel'>Frank L. Engel, USGS</a><br>"
            f"&emsp;< a href='https://www.usgs.gov/staff-profiles/travis-m-knight'>Travis M. Knight, USGS</a></p>"
            f"</blockquote>"
            f"<p>STIV algorithm design:</p>"
            f"<blockquote style='margin-left: 40px; text-indent: -20px;'>"
            f"<p>Legleiter, C.J., Kinzel, P.J., Engel, F.L., Harrison, L.R., and "
            f"Hewitt, G. 2024. A two-dimensional, "
            f"reach-scale implementation "
            f"of Space Time Image Velocimetry (STIV) and comparison to "
            f"Particle "
            f"Image Velocimetry (PIV). <em>Earth Surface "
            f"Processes and "
            f"Landforms.</em> "
            f"&emsp;< a href="
            f"'https://doi.org/10.1002/esp.5878'>https://doi.org/10.1002/esp"
            f".5878</a>.</p>"
            f"</blockquote>"
            # f"<p>For more information, visit <a
            # href='http://example.com'>example.com</a></p>"
        )

        QtWidgets.QMessageBox.about(self, f"About {self.__program_name__}", message)

    def check_for_updates(self):
        """Check for updates by checking the current version against the web page results"""
        version_url = "https://frank-engel.github.io/"

        version_msg = compare_versions(app_version=__version__, url=version_url)

        if version_msg is None:
            version_msg = (
                f"Unable to perform update check. An internet"
                f" connection is required to check for updates."
            )

        self.warning_dialog(
            title="Check for Updates",
            message=version_msg,
            style="ok",
            icon=os.path.join(self.__icon_path__, "IVy_logo.ico"),
        )

    def exit_application(self):
        """Separate logic needed to exit IVy cleanly from the File Menu"""
        self.close()

    def closeEvent(self, event):
        """Custom closeEvent for IVy"""
        close = QtWidgets.QMessageBox()
        close.setWindowIcon(
            QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "IVy_logo.ico"))
        )

        close.setWindowTitle("Close Image Velocimetry Tools (IVy Tools)?")
        close.setIcon(QtWidgets.QMessageBox.Warning)
        close.setText(
            "All unsaved progress will be lost. Are you sure you want to exit?"
        )
        close.setStandardButtons(
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel
        )

        close = close.exec()

        if close == QtWidgets.QMessageBox.Yes:
            # self.cleanup_temp_directory()
            event.accept()
        else:
            if event:
                event.ignore()

    def cleanup_temp_directory(self):
        """This function will be called when the application exits"""

        # Check for a map_file and deal with it
        map_file_path = os.path.join(self.swap_directory, "image_stack.dat")
        if os.path.exists(map_file_path):
            with self.wait_cursor():
                del self.image_stack

        try:
            if os.path.exists(self.swap_directory):
                # Stop watching the directory
                self.file_system_watcher.removePath(self.swap_directory)

                # Clean up the temp directory
                clean_up_temp_directory(self.swap_directory)
        except PermissionError:
            logging.error("Cannot delete image stack on exit.")

    @staticmethod
    def warning_dialog(title: str, message: str, style="YesCancel", icon=None,
                       modal=True):
        """
        Display a warning dialog with customizable buttons.

        Parameters:
        - title (str): The title of the warning dialog.
        - message (str): The message to be displayed in the dialog.
        - style (str, optional): The style of the dialog buttons.
            - "YesCancel": Display Yes and Cancel buttons.
            - "Ok": Display Ok button.
            Default is "YesCancel".
        - icon (str, optional): Path to an icon for the dialog title bar.
        - modal (bool, optional): If True, dialog blocks execution until user responds.
                                  If False, dialog is non-blocking (caller must handle signals).

        Returns:
        - str or None: User's choice as string for modal usage.
                       Returns None immediately for non-modal usage.

        Raises:
        - AttributeError: If an invalid style is provided.
        """
        dialog = QtWidgets.QMessageBox()
        dialog.setWindowTitle(title)
        if icon is not None:
            dialog.setWindowIcon(QtGui.QIcon(resource_path(icon)))
        dialog.setIcon(QtWidgets.QMessageBox.Warning)
        dialog.setText(message)

        if style.lower() == "yescancel":
            dialog.setStandardButtons(
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel
            )
        elif style.lower() == "ok":
            dialog.setStandardButtons(QtWidgets.QMessageBox.Ok)
        else:
            raise AttributeError(f"Invalid style argument: {style}")

        if modal:
            result = dialog.exec_()
            if result == QtWidgets.QMessageBox.Yes:
                return "yes"
            if result == QtWidgets.QMessageBox.Ok:
                return "ok"
            if result == QtWidgets.QMessageBox.Cancel:
                return "cancel"
            return None
        else:
            dialog.setModal(False)
            dialog.show()
            return None  # Caller must connect to dialog.buttonClicked if they care


    @staticmethod
    def custom_dialog_index(
        title: str,
        message: str,
        choices=("Yes", "No", "Option 1", "Option 2", "Cancel"),
    ):
        """
        Display a custom dialog with customizable choices and return the index of the chosen button.

        Parameters:
        - title (str): The title of the custom dialog.
        - message (str): The message to be displayed in the dialog.
        - choices (tuple, optional): The tuple of choices for the dialog buttons.

        Returns:
        - int or None: The index of the chosen button.
            - Returns the index of the selected choice.
            - None if the dialog was closed without a specific choice.

        Raises:
        - ValueError: If choices is not a tuple.
        """
        if not isinstance(choices, tuple):
            raise ValueError("Choices must be a tuple.")

        dialog = QtWidgets.QMessageBox()
        dialog.setWindowTitle(title)
        dialog.setIcon(QtWidgets.QMessageBox.Information)
        dialog.setText(message)

        # Add the Qt.WindowCloseButtonHint flag to enable the close button
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowCloseButtonHint)

        buttons = [
            dialog.addButton(choice, QtWidgets.QMessageBox.ActionRole)
            for choice in choices
        ]

        result = dialog.exec()

        if (
            result == -1
        ):  # -1 indicates that the dialog was closed without a specific choice
            return None

        # Find the index of the chosen button
        chosen_button = dialog.clickedButton()
        if chosen_button:
            return buttons.index(chosen_button)

        return None


    @contextmanager
    def wait_cursor(self):
        """
        Context manager that provides a busy cursor to the user while the code is processing.

        Yields:
        -------
        None

        Note:
        -----
        This context manager temporarily changes the cursor of the QApplication to a
        wait cursor (spinning circle) while the enclosed code block is being executed.
        After the code block completes, the cursor is reverted to its original state.

        Example:
        --------
        >>> with self.wait_cursor():
        ...     # Your time-consuming code here
        ...
        """
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            yield
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def set_tab_icon(self, tab, status="normal", tab_base=None):
        """Sets the text color and icon for the tab based on the quality
        status.

            Parameters
            ----------
            tab: str
                Tab identifier
            status: str
                Quality status
            tab_base: object
                Object of tab interface
        """

        if tab_base is None:
            tab_base = self.tabWidget

        tab_base.tabBar().setTabTextColor(
            tab_base.indexOf(tab_base.findChild(QtWidgets.QWidget, tab)),
            QtGui.QColor(140, 140, 140),
        )
        tab_base.setTabIcon(
            tab_base.indexOf(tab_base.findChild(QtWidgets.QWidget, tab)), QtGui.QIcon()
        )
        # Set appropriate icon
        if status == "good":
            tab_base.setTabIcon(
                tab_base.indexOf(tab_base.findChild(QtWidgets.QWidget, tab)),
                self.icon_good,
            )
            tab_base.tabBar().setTabTextColor(
                tab_base.indexOf(tab_base.findChild(QtWidgets.QWidget, tab)),
                QtGui.QColor(0, 153, 0),
            )
        elif status == "caution":
            tab_base.setTabIcon(
                tab_base.indexOf(tab_base.findChild(QtWidgets.QWidget, tab)),
                self.icon_caution,
            )
            tab_base.tabBar().setTabTextColor(
                tab_base.indexOf(tab_base.findChild(QtWidgets.QWidget, tab)),
                QtGui.QColor(230, 138, 0),
            )
        elif status == "warning":
            tab_base.setTabIcon(
                tab_base.indexOf(tab_base.findChild(QtWidgets.QWidget, tab)),
                self.icon_warning,
            )
            tab_base.tabBar().setTabTextColor(
                tab_base.indexOf(tab_base.findChild(QtWidgets.QWidget, tab)),
                QtGui.QColor(255, 77, 77),
            )
        elif status == "normal":
            # Remove icon and set text color back to black
            tab_base.setTabIcon(
                tab_base.indexOf(tab_base.findChild(QtWidgets.QWidget, tab)),
                QtGui.QIcon(),
            )
            tab_base.tabBar().setTabTextColor(
                tab_base.indexOf(tab_base.findChild(QtWidgets.QWidget, tab)),
                QtGui.QColor(0, 0, 0),
            )
        tab_base.setIconSize(QtCore.QSize(15, 15))

    def set_button_color(self, button_name, status="normal"):
        """Sets the background color of the button based on the status.

        Parameters
        ----------
        button_name: str
            Name of the button object
        status: str
            Status of the button

        """

        button = self.findChild(QtWidgets.QPushButton, button_name)

        if button:
            if status == "good":
                button.setStyleSheet("background-color: rgb(0, 153, 0);")
            elif status == "caution":
                button.setStyleSheet("background-color: rgb(230, 138, 0);")
            elif status == "warning":
                button.setStyleSheet("background-color: rgb(255, 77, 77);")
            elif status == "normal":
                button.setStyleSheet("")  # Reset to default style
        else:
            print(f"Button '{button_name}' not found.")

    def set_menu_item_color(self, action_name, status="normal"):
        """Sets the text color of the menu item based on the status.

        Parameters
        ----------
        action_name: str
            Name of the QAction associated with the menu item
        status: str
            Status of the menu item

        """

        # TODO: This function does not work!
        return
        action = self.findChild(QtWidgets.QAction, action_name)

        # if action:
        #     font = action.font()
        #     if status == "good":
        #         font.setBold(True)
        #         font.setItalic(False)
        #         font.setPointSize(10)  # Adjust font size as needed
        #         font.setColor(QtGui.QColor(0, 153, 0))
        #     elif status == "caution":
        #         font.setBold(True)
        #         font.setItalic(False)
        #         font.setPointSize(10)  # Adjust font size as needed
        #         font.setColor(QtGui.QColor(230, 138, 0))
        #     elif status == "warning":
        #         font.setBold(True)
        #         font.setItalic(False)
        #         font.setPointSize(10)  # Adjust font size as needed
        #         font.setColor(QtGui.QColor(255, 77, 77))
        #     elif status == "normal":
        #         font.setBold(False)
        #         font.setItalic(False)
        #         font.setPointSize(10)  # Adjust font size as needed
        #         font.setColor(QtGui.QColor(0, 0, 0))
        #     action.setFont(font)
        # else:
        #     print(f"Action '{action_name}' not found.")

    def save_gcp_template_file(self):
        """Save a GCP Points Table template CSV file"""
        contents = [
            [
                "# ID",
                "X",
                "Y",
                "Z",
                "X (pixel)",
                "Y (pixel)",
                "Error X (" "pixel)",
                "Error Y (pixel)",
                "Tot. Error (pixel)",
                "Use in Rectification",
                "Use in Validation",
            ],
            ["Point Name 1", "0", "0", "0", "", "", "", "", "", "", ""],
            ["Point Name 2", "0", "0", "0", "", "", "", "", "", "", ""],
        ]
        options = QtWidgets.QFileDialog.Options()
        # options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "Save a IVy " "GCP Template " "CSV " "File",
            "",
            "CSV Files (*.csv)",
            options=options,
        )

        if file_name:
            try:
                with open(file_name, "w", newline="") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    for row in contents:
                        csv_writer.writerow(row)
                QtWidgets.QMessageBox.information(
                    None, "Success", "CSV file saved successfully."
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    None, "Error", f"Failed to save CSV file: {str(e)}"
                )
        else:
            QtWidgets.QMessageBox.warning(
                None, "Warning", "No file selected for saving."
            )

    def init_toolbar_menu_icons(self):
        """Apply icons and shortcuts to toolbar and menu items."""

        # File Menus
        self.actionNew_Project.setIcon(
            QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "plus-solid.svg"))
        )
        self.actionNew_Project.setShortcut("Ctrl+N")
        self.actionOpen_Project.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "box-open-solid.svg")
            )
        )
        self.actionOpen_Project.setShortcut("Ctrl+O")
        self.actionSave_Project.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "floppy-disk-solid.svg")
            )
        )
        self.actionSave_Project.setShortcut("Ctrl+S")
        self.actionOpen_Video.setIcon(
            QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "film-solid.svg"))
        )
        self.actionOpen_Video.setShortcut("Ctrl+V")
        self.actionOpen_Image_Folder.setIcon(
            QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "image-solid.svg"))
        )
        self.actionOpen_Image_Folder.setShortcut("Ctrl+I")
        self.actionOpen_Ground_Control_Image.setIcon(
            QtGui.QIcon(
                resource_path(
                    self.__icon_path__ + os.sep + "map-location-dot-solid.svg"
                )
            )
        )
        self.actionOpen_Ground_Control_Image.setShortcut("Ctrl+G")
        self.actionImport_Ground_Control_Points_Table.setIcon(
            QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "table-solid.svg"))
        )
        self.actionImport_Ground_Control_Points_Table.setShortcut("Ctrl+T")
        self.actionImport_Bathymetry.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "draw-cross-section.svg")
            )
        )
        self.actionImport_Bathymetry.setShortcut("Ctrl+B")
        self.actionCompute_coordinates_from_4_point_distances.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "draw-polygon-solid.svg")
            )
        )
        self.actionCompute_coordinates_from_4_point_distances.setShortcut(
            "Ctrl+Shift+H"
        )
        self.actionSummary_Report_PDF.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "file-pdf-solid.svg")
            )
        )
        self.actionCheck_for_Updates.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "circle-question-solid.svg")
            )
        )
        self.actionAbout_IVy_Tools.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "circle-question-solid.svg")
            )
        )
        self.actionOpen_Help_Documentation.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "circle-question-solid.svg")
            )
        )
        self.actionOpen_Help_Documentation.setShortcut("F1")
        self.actionExit.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "door-open-solid.svg")
            )
        )

        # Toolbar
        self.toolButtonNewProject.setIcon(
            QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "plus-solid.svg"))
        )
        self.toolButtonOpenProject.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "box-open-solid.svg")
            )
        )
        self.toolButtonSaveProject.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "floppy-disk-solid.svg")
            )
        )
        self.toolButtonImportVideo.setIcon(
            QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "film-solid.svg"))
        )
        self.toolButtonImportFramesDirectory.setIcon(
            QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "image-solid.svg"))
        )
        self.toolButtonImportGroundControlImage.setIcon(
            QtGui.QIcon(
                resource_path(
                    self.__icon_path__ + os.sep + "map-location-dot-solid.svg"
                )
            )
        )
        self.toolButtonImportBathymetry.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "draw-cross-section.svg")
            )
        )
        self.toolButtonImportGCPTable.setIcon(
            QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "table-solid.svg"))
        )
        self.toolButtonExportPDF.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "file-pdf-solid.svg")
            )
        )

        ###
        self.toolButtonAddComment.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "comment-medical-solid.svg")
            )
        )
        self.toolButtonOpenSettings.setIcon(
            QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "gear-solid.svg"))
        )
        self.toolButtonOpenSettings.setEnabled(False)  # Temp disable
        self.toolButtonOpenHelp.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "circle-question-solid.svg")
            )
        )

        # In-app Icons
        self.buttonPlay.setIcon(
            QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "play-solid.svg"))
        )
        self.toolbuttonPreviousImage.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "play-backwards-solid.svg")
            )
        )
        self.toolbuttonNextImage.setIcon(
            QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "play-solid.svg"))
        )
        # self.PointPage.setItemIcon(
        #     QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "draw-points.svg"))
        # )
        self.toolbuttonCreatePoint.setIcon(
            QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "draw-points.svg"))
        )
        self.toolbuttonClearPoints.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "clear-draw-points.svg")
            )
        )
        # self.SimpleLinePage.setIcon(
        #     QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "draw-line-edit.svg"))
        # )
        self.toolbuttonCreateLine.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "draw-line-edit.svg")
            )
        )
        self.toolbuttonClearLine.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "clear-draw-line-edit.svg")
            )
        )
        # self.CrossSectionPage.setIcon(
        #     QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "draw-cross-section.svg"))
        # )
        self.toolbuttonCreateXsLine.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "draw-cross-section.svg")
            )
        )
        self.toolbuttonClearXsLine.setIcon(
            QtGui.QIcon(
                resource_path(
                    self.__icon_path__ + os.sep + "clear-draw-cross-section.svg"
                )
            )
        )
        # self.RegularGridPage.setIcon(
        #     QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "draw-grid.svg"))
        # )
        # self.MaskingPage.setIcon(
        #     QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "draw-polygon-solid.svg"))
        # )
        self.toolbuttonClearMask.setIcon(
            QtGui.QIcon(
                resource_path(
                    self.__icon_path__ + os.sep + "clear-draw-polygon-solid.svg"
                )
            )
        )
        self.toolbuttonCreateMask.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "draw-polygon-solid.svg")
            )
        )
        # self.toolbuttonEyeDropper.setIcon(
        #     QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "eye-dropper-solid.svg"))
        # )
        self.toolbuttonOpenOrthoPointsTable.setIcon(
            QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "table-solid.svg"))
        )
        self.toolbuttonSaveOrthoPointsTable.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "floppy-disk-solid.svg")
            )
        )
        self.toolbuttonOrthoOrigImageDigitizePoint.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "crosshairs-solid.svg")
            )
        )
        self.toolbuttonAddOrthotableRow.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "square-plus-solid.svg")
            )
        )
        self.toolbuttonRemoveOrthotableRow.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "square-minus-solid.svg")
            )
        )
        # self.toolbuttonCreateROI.setIcon(
        #     QtGui.QIcon(resource_path(icon_path + os.sep + "draw-polygon-solid.svg"))
        # )
        # self.toolbuttonClearROI.setIcon(
        #     QtGui.QIcon(
        #         resource_path(self.__icon_path__ + os.sep + "clear-draw-polygon-solid.svg")
        #     )
        # )
        self.toolbuttonDrawCrossSection.setIcon(
            QtGui.QIcon(
                resource_path(self.__icon_path__ + os.sep + "square-plus-solid.svg")
            )
        )
        # self.toolButtonClearCrossSection.setIcon(
        #     QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "square-minus-solid.svg"))
        # )
        self.progressBar = QtWidgets.QProgressBar()
        self.statusbarMainWindow.addPermanentWidget(self.progressBar)
        self.progressBar.setRange(0, 100)
        self.progressBar.hide()
        # self.toolbuttonEyeDropper.setCheckable(True)
        self.toolbuttonOrthoOrigImageDigitizePoint.setCheckable(True)

    def init_indicator_icons(self):
        """Initialize the tab status icons"""
        # Setup indicator icons
        self.icon_caution = QtGui.QIcon()
        self.icon_caution.addPixmap(
            QtGui.QPixmap(
                resource_path(
                    self.__icon_path__ + os.sep + "triangle-exclamation-solid.svg"
                )
            ),
            QtGui.QIcon.Normal,
            QtGui.QIcon.Off,
        )

        self.icon_warning = QtGui.QIcon()
        self.icon_warning.addPixmap(
            QtGui.QPixmap(
                resource_path(
                    self.__icon_path__ + os.sep + "square-exclamation-solid.svg"
                )
            ),
            QtGui.QIcon.Normal,
            QtGui.QIcon.Off,
        )

        self.icon_good = QtGui.QIcon()
        self.icon_good.addPixmap(
            QtGui.QPixmap(
                resource_path(self.__icon_path__ + os.sep + "check-solid.svg")
            ),
            QtGui.QIcon.Normal,
            QtGui.QIcon.Off,
        )

        self.icon_allChecked = QtGui.QIcon()
        self.icon_allChecked.addPixmap(
            QtGui.QPixmap(
                resource_path(self.__icon_path__ + os.sep + "check-solid.svg")
            ),
            QtGui.QIcon.Normal,
            QtGui.QIcon.Off,
        )

        self.icon_unChecked = QtGui.QIcon()
        self.icon_unChecked.addPixmap(
            QtGui.QPixmap(
                resource_path(self.__icon_path__ + os.sep + "check-solid.svg")
            ),
            QtGui.QIcon.Normal,
            QtGui.QIcon.Off,
        )

    def init_video_tab_attributes(self):
        """Initialize video-related attributes."""
        self.video_file_name = ""
        self.video_clip_filename = ""
        self.video_working_path = ""
        self.video_metadata = None
        self.video_duration = None
        self.video_timestep_ms = None
        self.video_num_frames = None
        self.video_frame_rate = None
        self.video_resolution = None
        self.video_clip_start_time = 0
        self.video_clip_end_time = 0
        self.video_rotation = 0
        self.video_flip = "none"
        self.video_strip_audio = True
        self.video_normalize_luma = False
        self.video_curve_preset = "none"
        self.video_ffmpeg_stabilize = False
        self.is_video_loaded = False
        self.is_clip_created = False
        self.is_frames_extracted = False
        self.ffmpeg_parameters = {
            "input_video": "../../examples/test_video.mp4",
            "output_video": "null -",  # ffmpeg decode without writing a file
            "start_time": seconds_to_hhmmss(self.video_clip_start_time),
            "end_time": None,
            "video_rotation": self.video_rotation,
            "video_flip": self.video_flip,
            "strip_audio": self.video_strip_audio,
            "normalize_luma": self.video_normalize_luma,
            "curve_preset": self.video_curve_preset,
            "stabilize": self.video_ffmpeg_stabilize,
            "extract_frames": False,
            "extract_frame_step": 1,
            "extracted_frames_folder": "../../examples/test_video",
            "extract_frame_pattern": "f%05d.jpg",
            "calibrate_radial": False,
            "cx": 0.5,
            "cy": 0.5,
            "k1": 0.0,
            "k2": 0.0,
        }
        self.extraction_frame_step = 1
        self.extraction_frame_rate = None
        self.extraction_timestep_ms = None
        self.extraction_num_frames = None
        self.extracted_frames_folder = None
        self.extraction_video_file_name = None
        self.lens_characteristics = LensCharacteristics()
        self.video_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.widgetVideo.setBackgroundRole(QtGui.QPalette.Dark)

    def init_threading_attributes(self):
        """Initialize threading-related attributes."""
        # Threading related items
        # There is one threadpool for all python executed thread needs,
        # but a seperate QProcess for each external program call (e.g. ffmpeg, imagemagick, etc)
        self.threadpool = QThreadPool()
        logging.info(
            f"Multithreading with maximum {self.threadpool.maxThreadCount()} threads"
        )
        self.ffmpeg_process = QProcess()
        self.ffmpeg_thread_is_running = False
        self.ffmpeg_thread_is_finished = False
        self.image_processor_thread_is_running = False
        self.image_processor_thread_is_running = False

        # All explicit image processing that's not in ffmpeg is handled
        # here
        self.image_processor = ImageProcessor()
        self.image_processor_process = QProcess()

        # Stabilization related items
        self.stabilize_step1_finished = False
        self.stabilize_step2_finished = False
        self.stabilize_step3_finished = False
        self.stabilize_step4_finished = False
        self.process_step = "n/a"

    def init_imagebrowser_attributes(self):
        """Initialize image browser-related attributes."""
        self.imagebrowser = ImageBrowserTab(self)
        self.layoutImageBrowser.addWidget(self.imagebrowser.imageBrowser)

    def init_orthorectification_attributes(self):
        """Initialize orthorectification-related attributes."""
        # Orthorectification Tab Image
        self.cam = None
        self.rectification_parameters = {
            "homography_matrix": np.eye(3),
            "camera_matrix": np.hstack((np.eye(3), np.ones((3, 1)))),
            "pixel_coords": None,
            "world_coords": None,
        }
        self.camera_position = None
        self.pixel_ground_scale_distance_m = None
        self.is_homography_matrix = False
        self.is_camera_matrix = False
        self.is_ortho_flip_x = False
        self.is_ortho_flip_y = False
        self.scene_averaged_pixel_gsd_m = None
        self.rectification_method = None
        self.ortho_original_image_digitized_points = []
        self.ortho_original_image_zoom_factor = 1.0
        self.ortho_rectified_image_zoom_factor = 1.0
        self.ortho_rectified_wse_m = 0.0
        self.ortho_original_image = AnnotationView()
        self.ortho_original_image.setEnabled(False)
        self.image_stack = None
        self.is_transformed_frames = False
        self.layoutOrthoOriginalImage.addWidget(self.ortho_original_image)
        self.layoutOrthoImagePane.addWidget(self.ortho_original_image)
        self.ortho_rectified_image = AnnotationView()
        self.ortho_rectified_image.setEnabled(False)
        self.layoutOrthoImagePane.addWidget(self.ortho_rectified_image)

        # Orthorectification Points Table
        self.orthotable_dataframe = pd.DataFrame()
        self.orthotable_is_changed = False
        self.is_ortho_table_loaded = False
        self.orthotable_cell_colored = False
        self.orthotable_file_name = ""
        self.orthotable_fname = ""
        self.orthotable_file_survey_units = "English"
        self.reprojection_error_pixels = None
        self.reprojection_error_gcp_pixel_xy = None
        self.reprojection_error_gcp_pixel_total = None
        self.rectified_transformed_gcp_points = None
        self.rectification_rmse_m = None
        self.orthoPointsTable = TableWidgetDragRows()
        self.orthotable_init()
        self.orthoPointsTable.setGridStyle(1)
        self.orthoPointsTable.setCornerButtonEnabled(False)
        self.orthoPointsTable.setShowGrid(True)
        self.orthoPointsTable.horizontalHeader().setBackgroundRole(
            QtGui.QPalette.Window
        )
        self.orthoPointsTable.setDropIndicatorShown(True)
        self.orthoPointsTableLineEdit = QtWidgets.QLineEdit()
        self.orthoPointsTableLineEdit.setToolTip("edit and press ENTER")
        self.orthoPointsTableLineEdit.setStatusTip("edit and press ENTER")

        self.layoutOrthoPointsTable.addWidget(self.orthoPointsTable)

        self.load_ndarray_into_qtablewidget(
            self.rectification_parameters["homography_matrix"],
            self.tablewidgetProjectiveMatrix,
        )

    def init_grid_preparation_attributes(self):
        """Initialize grid preparation-related attributes."""
        self.gridpreparation = GridPreparationTab(self)
        self.layoutGridGeneration.addWidget(self.gridpreparation.imageBrowser)

        self.is_cross_section_grid = False
        self.results_grid = None
        self.results_grid_world = None
        self.horz_grid_size = 50
        self.vert_grid_size = 50
        self.number_grid_points_along_line = 25
        self.number_grid_points_along_xs_line = 25
        self.line_mode = "line"
        self.region_of_interest_pixels = None
        self.spinboxHorizGridSpacing.setValue(self.horz_grid_size)
        self.spinboxVertGridSpacing.setValue(self.vert_grid_size)
        self.spinboxLineNumPoints.setValue(self.number_grid_points_along_line)
        self.spinbocXsLineNumPoints.setValue(self.number_grid_points_along_xs_line)
        self.gridgenerator = GridGenerator(self)
        self.toolboxGridCreation.setCurrentIndex(0)  # Set to "Point" page

    def init_stiv_attributes(self):
        """Initialize STIV-related attributes."""
        self.stiv = STIVTab(self)
        self.layoutSTIV.addWidget(self.stiv.imageBrowser)
        self.stiv_gaussian_blur_sigma = 0.0
        self.stiv_dphi = 1
        self.stiv_num_pixels = 20
        self.stiv_search_line_length_m = 5
        self.stiv_phi_origin = 0
        self.stiv_phi_range = 90
        self.stiv_max_vel_threshold_mps = 10.0
        self.stiv_opt_max_vel_threshold_mps = 10.0
        self.magnitudes_mps = None
        self.directions = None
        self.stiv_opt = STIVTab(self)
        self.layoutSTIVOpt.addWidget(self.stiv_opt.imageBrowser)
        self.stiv_exists = False
        self.stiv_opt_exists = False
        self.is_stis = False
        self.sti = STIReviewTab(self)
        self.is_manual_sti_corrections = False

    def init_cross_section_attributes(self):
        """Initialize cross-section-related attributes."""
        self.bathymetry_ac3_filename = None
        self.is_area_comp_loaded = False
        self.cross_section_line = None
        self.cross_section_start_bank = "left"
        self.cross_section_line_exists = False
        self.cross_section_top_width_m = 0.0
        self.cross_section_length_pixels = 1000
        self.cross_section_rectified_eps = None
        self.perspective_xs_image = AnnotationView()
        self.perspective_xs_image.setEnabled(False)
        self.layoutXsPerspectiveImage.addWidget(self.perspective_xs_image)
        self.layoutXsImagePane.addWidget(self.perspective_xs_image)
        self.rectified_xs_image = AnnotationView()
        self.rectified_xs_image.setEnabled(False)
        self.layoutXsRectifiedImage.addWidget(self.rectified_xs_image)
        self.layoutXsImagePane.addWidget(self.rectified_xs_image)
        self.xs_survey = CrossSectionGeometry(self)
        self.channel_char = {}

    def init_discharge_attributes(self):
        """Initialize discharge-related attributes."""
        self.dischargecomputaton = DischargeTab(self)
        self.discharge_results = {}
        self.discharge_summary = {}
        self.u_iso = {}
        self.u_iso_contribution = {}
        self.u_ive = {}
        self.u_ive_contribution = {}
        self.discharge_plot_fig = None

    def init_class_attributes(self):
        """Initialize IVy's class attributes to default values and settings."""

        # Initialize services and models
        self.video_service = VideoService()
        self.project_service = ProjectService()
        self.ortho_service = OrthorectificationService()
        self.image_stack_service = ImageStackService()
        self.video_model = VideoModel()

        # Global init related
        self.ivy_settings_file = "IVy_Settings"
        self.sticky_settings = Settings(self.ivy_settings_file)
        self.units = "English"  # TODO: remove this and use the units_label
        try:
            ss = self.sticky_settings.get("last_display_units")
            self.units_label = ss
            self.display_units = ss
        except KeyError:
            self.units_label = "English"
            self.display_units = "English"
            self.sticky_settings.new("last_display_units", self.display_units)
        self.survey_units = units_conversion(units_id=self.display_units)
        self.project_filename = f"{QDir.homePath()}{os.sep}New_IVy_Project.ivy"
        self.status_message = (
            "Open a video to begin | Drag and Drop -OR- File-->Open Video (Ctrl+O)"
        )
        self.progress_percent = 0
        self.vector_scale = 40

        # Video tab
        self.init_video_tab_attributes()

        # Initialize video controller (after video_player is created)
        self.video_controller = VideoController(
            self,
            self.video_model,
            self.video_service
        )

        # Threading
        self.init_threading_attributes()

        # Image Browser
        self.init_imagebrowser_attributes()

        # Orthorectification
        self.init_orthorectification_attributes()

        # Grid Preparation
        self.init_grid_preparation_attributes()

        # STIV Tabs
        self.init_stiv_attributes()

        # Other velocity methods
        self.openpiv_exists = False
        self.trivia_exists = False

        # Cross-Section Geometry
        self.init_cross_section_attributes()

        # Discharge Tab
        self.init_discharge_attributes()

    def init_connections_and_slots(self):
        """Initialize all IVy connections and slots."""
        # File Menu
        self.actionNew_Project.triggered.connect(self.clear_project)
        self.actionOpen_Project.triggered.connect(self.open_project)
        self.actionSave_Project.triggered.connect(self.save_project)
        self.actionOpen_Video.triggered.connect(self.open_video)
        self.actionUnits.setText(f"Units: {self.display_units}")
        self.actionUnits.triggered.connect(self.open_settings_dialog)
        self.actionSummary_Report_PDF.triggered.connect(self.create_summary_report)
        self.actionSummary_Report_PDF.setEnabled(False)
        self.actionOpen_Image_Folder.triggered.connect(
            self.imagebrowser.open_image_folder
        )
        self.actionOpen_Ground_Control_Image.triggered.connect(
            self.ortho_original_load_gcp_image
        )
        self.actionImport_Ground_Control_Points_Table.triggered.connect(
            self.orthotable_load_csv
        )
        self.actionCompute_coordinates_from_4_point_distances.triggered.connect(
            self.homography_distance_conversion_tool
        )
        self.actionEstimate_STIV_Video_Sample_Rate.triggered.connect(
            self.estimate_stiv_sample_rate
        )
        self.actionSave_Template_Ground_Control_Points_File.triggered.connect(
            self.save_gcp_template_file
        )
        self.actionCheck_for_Updates.triggered.connect(self.check_for_updates)
        self.actionAbout_IVy_Tools.triggered.connect(self.about_dialog)
        self.actionOpen_Help_Documentation.triggered.connect(
            self.launch_documentation_browser
        )
        self.menuMenu.setEnabled(True)

        # Toolbar
        self.toolButtonNewProject.clicked.connect(self.clear_project)
        self.toolButtonOpenProject.clicked.connect(self.open_project)
        self.toolButtonSaveProject.clicked.connect(self.save_project)
        self.toolButtonImportVideo.clicked.connect(self.open_video)
        self.toolButtonImportFramesDirectory.clicked.connect(
            self.imagebrowser.open_image_folder
        )
        self.toolButtonImportGroundControlImage.clicked.connect(
            self.ortho_original_load_gcp_image
        )
        self.toolButtonImportBathymetry.clicked.connect(self.xs_survey.load_areacomp)
        self.toolButtonImportGCPTable.clicked.connect(self.orthotable_load_csv)
        self.toolButtonExportPDF.clicked.connect(self.create_summary_report)
        self.toolButtonExportPDF.setEnabled(False)
        self.toolButtonAddComment.clicked.connect(self.update_comments)
        self.toolButtonOpenSettings.clicked.connect(self.open_settings_dialog)
        self.toolButtonOpenHelp.clicked.connect(self.launch_documentation_browser)

        # Video Tab - signals are connected by video_controller
        # Note: VideoController connects signals in its _connect_signals() method
        # These connections are retained for backwards compatibility:
        self.video_player.error.connect(self.video_error_handling)
        self.buttonCreateVideoClip.clicked.connect(self.create_video_clip)
        self.actionExit.triggered.connect(self.exit_application)
        self.lineeditFrameStepValue.editingFinished.connect(self.frame_step_changed)
        self.buttonExtractVideoFrames.clicked.connect(self.extract_frames)
        self.checkboxCorrectRadialDistortion.stateChanged.connect(
            self.correct_radial_distortion
        )
        self.buttonLoadLensCharacteristics.clicked.connect(
            self.get_lens_characteristics
        )
        self.signal_opencv_updates.connect(
            lambda msg: self.warning_dialog(
                "Import Video: Problem parsing video metadata",
                message=msg,
                style="ok",
                icon=os.path.join(self.__icon_path__, "IVy_logo.ico")
            )
        )

        # Threading
        self.ffmpeg_process.readyReadStandardError.connect(
            self.ffmpeg_onready_read_stderr
        )  # ffmpeg dumps stdout to the stderr signal for some reason...
        self.ffmpeg_process.readyReadStandardOutput.connect(
            self.ffmpeg_onready_read_stdout
        )
        self.ffmpeg_process.finished.connect(self.ffmpeg_thread_finished)
        self.signal_ffmpeg_thread.connect(self.ffmpeg_setup_stabilization_pass1)
        self.signal_ffmpeg_thread.connect(self.ffmpeg_setup_stabilization_pass2)
        self.image_processor_process.finished.connect(self.image_stack_process_finished)
        kill_thread_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence("Shift+Esc"), self
        )
        kill_thread_shortcut.activated.connect(self.stop_current_threadpool_task)

        # Image Browser
        self.toolbuttonNextImage.clicked.connect(self.imagebrowser.set_next_image)
        self.toolbuttonPreviousImage.clicked.connect(
            self.imagebrowser.set_previous_image
        )
        self.buttonApplyFileFilter.clicked.connect(
            self.imagebrowser.reload_image_folder
        )
        self.buttonApplyToThisFrame.clicked.connect(
            self.imagebrowser.apply_to_this_frame
        )
        self.buttonApplyToAllFrames.clicked.connect(
            self.imagebrowser.apply_to_all_frames
        )
        self.imagebrowser.imageBrowser.actionComplete.connect(
            self.imagebrowser.editing_complete
        )

        # Orthorectification
        self.signal_orthotable_changed.connect(self.ortho_original_refresh_plot)
        self.signal_orthotable_check_units.connect(self.orthotable_change_units)
        self.pushbuttonExportProjectedFrames.clicked.connect(
            self.orthorectify_many_thread_handler
        )
        self.toolbuttonOrthoOrigImageDigitizePoint.clicked.connect(
            self.ortho_original_image_digitize_point
        )
        self.doubleSpinBoxRectificationWaterSurfaceElevation.editingFinished.connect(
            self.ortho_rectified_water_surface_elevation
        )
        self.orthoPointsTable.selectionModel().selectionChanged.connect(
            self.orthotable_make_all_white
        )
        self.orthoPointsTable.itemClicked.connect(self.orthotable_get_item)
        self.orthoPointsTable.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked)
        self.orthoPointsTable.cellChanged.connect(self.orthotable_finished_edit)
        self.orthoPointsTableLineEdit.returnPressed.connect(self.orthotable_update_cell)
        self.toolbuttonOpenOrthoPointsTable.clicked.connect(self.orthotable_load_csv)
        self.toolbuttonSaveOrthoPointsTable.clicked.connect(self.orthotable_save_csv)
        self.orthotable_add_row()
        self.buttonRectifyCurrentImage.clicked.connect(self.rectify_single_frame)
        self.toolbuttonAddOrthotableRow.clicked.connect(self.orthotable_add_row)
        self.toolbuttonRemoveOrthotableRow.clicked.connect(self.orthotable_remove_row)
        self.checkBoxOrthoFlipX.stateChanged.connect(self.ortho_flip_x_changed)
        self.checkBoxOrthoFlipY.stateChanged.connect(self.ortho_flip_y_changed)

        # Grid Preparation
        self.toolbuttonCreatePoint.clicked.connect(self.gridpreparation.add_point)
        self.toolbuttonCreateLine.clicked.connect(self.gridpreparation.add_line)
        self.toolbuttonCreateXsLine.clicked.connect(
            self.gridpreparation.add_line_of_given_length
        )
        self.toolbuttonCreateMask.clicked.connect(self.gridpreparation.add_mask)
        self.toolbuttonClearPoints.clicked.connect(self.gridpreparation.clear_point)
        self.toolbuttonClearLine.clicked.connect(self.gridpreparation.clear_line)
        self.toolbuttonClearXsLine.clicked.connect(self.gridpreparation.clear_line)
        self.toolbuttonClearMask.clicked.connect(self.gridpreparation.clear_mask)
        self.gridpreparation.imageBrowser.polygonPoints.connect(
            self.gridpreparation.save_roi
        )
        self.buttonCreatePointsOnLine.clicked.connect(
            lambda: self.create_line_grid(mode="line")
        )
        self.buttonCreatePointsOnCrossSection.clicked.connect(
            lambda: self.create_line_grid(mode="cross_section")
        )
        self.buttonCreateGrid.clicked.connect(self.create_grid)
        self.spinboxLineNumPoints.editingFinished.connect(self.change_line_num_points)
        self.spinbocXsLineNumPoints.editingFinished.connect(
            self.change_xs_line_num_points
        )
        self.spinboxHorizGridSpacing.editingFinished.connect(self.change_horz_grid_size)
        self.spinboxVertGridSpacing.editingFinished.connect(self.change_vert_grid_size)

        # STIV
        self.buttonSTIVProcessVelocities.clicked.connect(self.stiv_thread_handler)
        self.buttonSTIVOptProcessVelocities.clicked.connect(self.stiv_thread_handler)
        self.doublespinboxStivGaussianBlurSigma.editingFinished.connect(
            self.stiv_gaussian_blur_changed
        )
        self.spinboxSTIVdPhi.editingFinished.connect(self.stiv_dphi_changed)
        self.doublespinboxStivSearchLineDistance.editingFinished.connect(
            self.stiv_search_line_distance_changed
        )
        self.spinboxSTIVPhiOrigin.editingFinished.connect(self.stiv_phi_origin_changed)
        self.spinboxSTIVPhiRange.editingFinished.connect(self.stiv_phi_range_changed)
        self.spinboxSTIVMaxVelThreshold.editingFinished.connect(
            self.stiv_max_vel_threshold_changed
        )
        self.spinboxSTIVOptMaxVelThreshold.editingFinished.connect(
            self.stiv_opt_max_vel_threshold_changed
        )
        self.pushbuttonCreateRefreshImageStackSTIV.clicked.connect(
            self.create_image_stack
        )
        self.pushbuttonCreateRefreshImageStackSTIVOpt.clicked.connect(
            self.create_image_stack
        )
        self.signal_manual_vectors.connect(self.plot_manual_vectors)

        # Cross-Section Geometry
        self.toolbuttonDrawCrossSection.clicked.connect(self.draw_cross_section_line)
        # self.toolButtonClearCrossSection.clicked.connect(self.clear_cross_section_line)
        self.signal_cross_section_exists.connect(self.cross_section_manager)
        self.sbLeftBankXRectifiedCoordPixels.editingFinished.connect(
            self.cross_section_manager_eps)
        self.sbLeftBankYRectifiedCoordPixels.editingFinished.connect(
            self.cross_section_manager_eps)
        self.sbRightBankRectifiedXCoordPixels.editingFinished.connect(
            self.cross_section_manager_eps)
        self.sbRightBankRectifiedYCoordPixels.editingFinished.connect(
            self.cross_section_manager_eps)

        # Discharge
        # All migrated to DischargeTab Class

    def update_comments(self):
        """Updates comments based on text entered into the line edit"""
        tab_name = self.tabWidget.tabText(self.tabWidget.currentIndex())
        # If active tab is image velocimetry, find out which sub-tab is active
        if tab_name == "Image Velocimetry":
            tab_name = self.tabWidget_ImageVelocimetryMethods.tabText(
                self.tabWidget_ImageVelocimetryMethods.currentIndex()
            )

        doc = AddDocumentation(tab_name=tab_name)
        doc.exec()

        if len(doc.comment) > 0:
            self.comments.append_comment(doc.category, doc.comment)
            self.update_comment_tbl()

    def update_comment_tbl(self):
        """Update the comments table"""
        tbl = self.comments_tbl
        tbl.clear()
        headers = ["Category", "Comment"]

        bold = QtGui.QFont()
        bold.setBold(True)

        row_cnt = 0
        for key in self.comments.comments:
            row_cnt += len(self.comments.comments[key])

        tbl.setRowCount(row_cnt)
        tbl.setColumnCount(2)

        tbl.setHorizontalHeaderLabels(headers)
        tbl.horizontalHeader().setVisible(True)
        tbl.horizontalHeader().setFont(bold)
        tbl.verticalHeader().setVisible(False)

        if row_cnt > 0:
            row = 0

            for key in self.comments.comments:
                for item in self.comments.comments[key]:
                    tbl.setItem(row, 0, QtWidgets.QTableWidgetItem(key))
                    tbl.setItem(row, 1, QtWidgets.QTableWidgetItem(item))

                    row += 1

        tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        tbl.resizeColumnsToContents()
        tbl.resizeRowsToContents()

    @staticmethod
    def enable_disable_tabs(tab_base, tab, enable=True):
        """Enable or diable tabs in a pyQt 5 UI

        Args:
            tab_base (QTabWidget): the main tab object containing tabls
            tab (QTab): the tab to update
            enable (bool, optional): enable if true, disable if false. Defaults to True.
        """
        return
        tab_index = tab_base.indexOf(tab_base.findChild(QtWidgets.QWidget, tab))
        if tab_index == -1:
            print(f"Tab '{tab}' not found.")
            return

        tab_widget = tab_base.widget(tab_index)
        if tab_widget is None:
            print(f"Widget for tab '{tab}' not found.")
            return

        tab_base.setTabEnabled(tab_index, enable)

    @staticmethod
    def launch_documentation_browser():
        """
        Launch the documentation browser to open the SurfVelTool documentation.

        Description:

            This static method launches the system default browser to open
            the IVy documentation. It determines the location of the
            documentation based on the running environment. In a development
            environment, it looks for the documentation in the '_build/html'
            directory relative to the current directory. In a production
            environment, it uses the 'documentation/index.html' file located
            in the 'resources' directory of the IVy installation. The
            documentation is opened using the 'file://' protocol.

        Returns:
            None

        """
        # Check if running in a development environment
        if os.environ.get("IVY_ENV") == "development":
            # Use development-specific settings
            current_dir = os.getcwd()
            parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
            docs_dir = "docs" + os.sep + "_build" + os.sep + "html" + os.sep
            documentation_landing_page = current_dir + os.sep + docs_dir + "index.html"
        else:
            # Use production-specific settings
            documentation_landing_page = (
                resource_path("documentation") + os.sep + "index.html"
            )
            logging.debug(documentation_landing_page)

        # Open the local HTML file in the system default browser
        webbrowser.open("file://" + documentation_landing_page)


if __name__ == "__main__":
    try:
        # import cProfile, pstats
        app = QtWidgets.QApplication(sys.argv)
        # cProfile.run('IvyTools()', 'PROFILE.txt')
        window = IvyTools()
        window.showMaximized()
        sys.exit(app.exec_())
        # p = pstats.Stats('PROFILE.txt')
        # p.sort_stats('cumulative').print_stats(100)
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        import traceback

        traceback.print_exc()
