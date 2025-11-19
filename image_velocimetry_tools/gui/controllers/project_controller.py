"""Controller for project management UI coordination."""

import logging
import os
import zipfile
from typing import Optional, Dict, Any
from PyQt5.QtCore import pyqtSlot, QDir
from PyQt5 import QtWidgets

from image_velocimetry_tools.gui.controllers.base_controller import BaseController
from image_velocimetry_tools.gui.models.project_model import ProjectModel
from image_velocimetry_tools.services.project_service import ProjectService


class ProjectController(BaseController):
    """Controller for project management UI.

    This controller coordinates between:
    - Project UI widgets (file dialogs, save/load operations)
    - ProjectModel (state management)
    - ProjectService (business logic)

    Responsibilities:
    - New project creation
    - Project loading and extraction
    - Project saving and archiving
    - Project structure management
    - UI state updates based on model changes
    """

    def __init__(
        self,
        main_window,
        project_model: ProjectModel,
        project_service: ProjectService
    ):
        """Initialize the project controller.

        Args:
            main_window: Reference to main window for widget access
            project_model: Project state model
            project_service: Project business logic service
        """
        super().__init__(main_window, project_model, project_service)
        self.project_model = project_model
        self.project_service = project_service

        # Connect signals after initialization
        self._connect_signals()

    def _connect_signals(self):
        """Connect UI signals to controller methods and model signals to UI updates."""
        # Model signals
        self.project_model.project_created.connect(self.on_model_project_created)
        self.project_model.project_loaded.connect(self.on_model_project_loaded)
        self.project_model.project_saved.connect(self.on_model_project_saved)
        self.project_model.project_closed.connect(self.on_model_project_closed)

        self.logger.debug("Project controller signals connected")

    @pyqtSlot()
    def on_model_project_created(self):
        """Handle project created signal from model."""
        self.logger.info("Project created")

    @pyqtSlot(str)
    def on_model_project_loaded(self, file_path: str):
        """Handle project loaded signal from model."""
        self.logger.info(f"Project loaded: {file_path}")
        # Update window title with project name
        mw = self.main_window
        project_name = os.path.splitext(os.path.basename(file_path))[0]
        mw.setWindowTitle(f"IVyTools - {project_name}")

    @pyqtSlot(str)
    def on_model_project_saved(self, file_path: str):
        """Handle project saved signal from model."""
        self.logger.info(f"Project saved: {file_path}")

    @pyqtSlot()
    def on_model_project_closed(self):
        """Handle project closed signal from model."""
        self.logger.info("Project closed")
        # Reset window title
        mw = self.main_window
        mw.setWindowTitle("IVyTools")

    def new_project(self):
        """Create a new IVy Project, discarding all currently loaded data."""
        mw = self.main_window

        # Set default project filename
        default_filename = f"{QDir.homePath()}{os.sep}New_IVy_Project.ivy"
        self.project_model.project_filename = default_filename

        # Clear project state
        self.clear_project()

        # Emit project created signal
        self.project_model.project_created.emit()

        self.logger.info("New project created")

    def open_project(self) -> bool:
        """Open an IVy Project Session File.

        Returns:
            True if project was successfully opened, False otherwise
        """
        mw = self.main_window

        # Load the last project filename from settings
        try:
            ss = mw.sticky_settings.get("last_project_filename")
            last_filename = ss
        except KeyError:
            last_filename = f"{QDir.homePath()}{os.sep}New_IVy_Project.ivy"
            mw.sticky_settings.new("last_project_filename", last_filename)

        # Open a project file dialog
        filter_spec = "IVy Project (*.ivy);;All files (*.*)"
        project_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Open IVy Project File",
            last_filename,
            filter_spec,
        )

        if not project_filename:
            return False  # User cancelled

        # Extract the zip file to the swap directory
        try:
            self.project_service.extract_project_archive(
                project_filename,
                mw.swap_directory
            )
        except (zipfile.BadZipFile, IOError, FileNotFoundError, ValueError) as e:
            mw.warning_dialog(
                "Error Opening Project",
                f"An error occurred while opening the project: {str(e)}",
                style="ok",
                icon=mw.__icon_path__ + os.sep + "IVy_logo.ico",
            )
            self.logger.error(f"Error opening project: {str(e)}")
            return False

        # Load the project_dict from the JSON file in the swap directory
        json_filename = os.path.join(mw.swap_directory, "project_data.json")
        try:
            project_dict = self.project_service.load_project_from_json(json_filename)
            project_dict["project_file_path"] = project_filename
        except (FileNotFoundError, ValueError, IOError) as e:
            mw.warning_dialog(
                "Error Opening Project",
                f"An error occurred while opening the project: {str(e)}",
                style="ok",
                icon=mw.__icon_path__ + os.sep + "IVy_logo.ico",
            )
            self.logger.error(f"Error opening project: {str(e)}")
            return False

        # Update model
        self.project_model.project_filename = project_filename
        self.project_model.project_name = os.path.splitext(os.path.basename(project_filename))[0]

        # Save to sticky settings
        mw.sticky_settings.set("last_project_filename", project_filename)

        # Emit project loaded signal
        self.project_model.project_loaded.emit(project_filename)

        self.logger.info(f"Project file loaded: {project_filename}")

        return True

    def save_project(self, project_dict: Dict[str, Any]) -> bool:
        """Save the current project as a zip archive.

        Args:
            project_dict: Dictionary containing all project data to save

        Returns:
            True if project was successfully saved, False otherwise
        """
        mw = self.main_window

        # Get project filename (may prompt user if not set)
        project_filename = self.project_model.project_filename
        if not project_filename:
            project_filename = f"{QDir.homePath()}{os.sep}New_IVy_Project.ivy"

        # Prompt user for save location
        filter_spec = "IVy Project (*.ivy);;All files (*.*)"
        save_filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "Save IVy Project File",
            project_filename,
            filter_spec,
        )

        if not save_filename:
            return False  # User cancelled

        # Ensure .ivy extension
        if not save_filename.endswith('.ivy'):
            save_filename += '.ivy'

        # Save project_dict to JSON in swap directory
        json_filename = os.path.join(mw.swap_directory, "project_data.json")
        try:
            self.project_service.save_project_to_json(project_dict, json_filename)
        except (ValueError, IOError) as e:
            mw.warning_dialog(
                "Error Saving Project",
                f"An error occurred while saving the project: {str(e)}",
                style="ok",
                icon=mw.__icon_path__ + os.sep + "IVy_logo.ico",
            )
            self.logger.error(f"Error saving project to JSON: {str(e)}")
            return False

        # Create project archive
        try:
            self.project_service.create_project_archive(
                mw.swap_directory,
                save_filename,
                progress_callback=None,
                exclude_extensions=[".dat"]
            )
        except (IOError, ValueError) as e:
            mw.warning_dialog(
                "Error Saving Project",
                f"An error occurred while creating the project archive: {str(e)}",
                style="ok",
                icon=mw.__icon_path__ + os.sep + "IVy_logo.ico",
            )
            self.logger.error(f"Error creating project archive: {str(e)}")
            return False

        # Update model
        self.project_model.project_filename = save_filename
        self.project_model.project_name = os.path.splitext(os.path.basename(save_filename))[0]

        # Save to sticky settings
        mw.sticky_settings.set("last_project_filename", save_filename)

        # Emit project saved signal
        self.project_model.project_saved.emit(save_filename)

        self.logger.info(f"Project saved: {save_filename}")

        return True

    def clear_project(self):
        """Clear all project data and reset to initial state."""
        mw = self.main_window

        # Reset project model
        self.project_model.reset()

        # Additional cleanup can be added here
        # (clearing image browsers, tables, etc.)

        self.logger.info("Project cleared")

    def get_project_dict_from_main_window(self) -> Dict[str, Any]:
        """Extract project data from main window state.

        This method collects all serializable state from the main window
        and returns it as a dictionary for saving.

        Returns:
            Dictionary containing all project data
        """
        mw = self.main_window

        # Create project dictionary from main window __dict__
        project_dict = {
            key: value
            for (key, value) in zip(mw.__dict__.keys(), mw.__dict__.values())
            if type(value) == list
            or type(value) == str
            or type(value) == dict
            or type(value) == int
            or type(value) == float
            or type(value) == bool
            or value is None
        }

        return project_dict

    def load_project_dict_to_main_window(self, project_dict: Dict[str, Any]):
        """Load project data into main window state.

        This method takes a project dictionary and updates the main window
        state with the loaded data.

        Args:
            project_dict: Dictionary containing project data
        """
        mw = self.main_window

        # This will be implemented to restore all project state
        # For now, just log that we would load the data
        self.logger.debug(f"Would load project dict with {len(project_dict)} keys")
