"""Model for project state management."""

from typing import Optional, Dict, Any
from PyQt5.QtCore import pyqtSignal

from image_velocimetry_tools.gui.models.base_model import BaseModel


class ProjectModel(BaseModel):
    """Model representing project state.

    This model holds all project-related state including:
    - Project file information (path, name)
    - Project loaded state
    - Project directory structure paths

    Signals:
        project_created: Emitted when a new project is created
        project_loaded: Emitted when a project is loaded (file_path)
        project_saved: Emitted when project is saved (file_path)
        project_closed: Emitted when project is closed
    """

    # Qt Signals
    project_created = pyqtSignal()
    project_loaded = pyqtSignal(str)  # file_path
    project_saved = pyqtSignal(str)  # file_path
    project_closed = pyqtSignal()

    def __init__(self):
        """Initialize the project model with default values."""
        super().__init__()

        # Project file information
        self._project_filename: Optional[str] = None
        self._is_project_loaded: bool = False
        self._project_name: Optional[str] = None

        # Project structure paths
        self._swap_directory: Optional[str] = None
        self._swap_image_directory: Optional[str] = None
        self._swap_grids_directory: Optional[str] = None
        self._swap_stiv_directory: Optional[str] = None

    # Project file properties
    @property
    def project_filename(self) -> Optional[str]:
        """Get the current project file path."""
        return self._project_filename

    @project_filename.setter
    def project_filename(self, path: Optional[str]):
        """Set the project file path."""
        if path != self._project_filename:
            self._project_filename = path
            self._is_project_loaded = path is not None
            self._emit_state_change("project_filename", path)

    @property
    def is_project_loaded(self) -> bool:
        """Check if a project is currently loaded."""
        return self._is_project_loaded

    @property
    def project_name(self) -> Optional[str]:
        """Get the current project name."""
        return self._project_name

    @project_name.setter
    def project_name(self, name: Optional[str]):
        """Set the project name."""
        if name != self._project_name:
            self._project_name = name
            self._emit_state_change("project_name", name)

    # Project directory properties
    @property
    def swap_directory(self) -> Optional[str]:
        """Get the swap directory path."""
        return self._swap_directory

    @swap_directory.setter
    def swap_directory(self, path: Optional[str]):
        """Set the swap directory path."""
        if path != self._swap_directory:
            self._swap_directory = path
            self._emit_state_change("swap_directory", path)

    @property
    def swap_image_directory(self) -> Optional[str]:
        """Get the swap image directory path."""
        return self._swap_image_directory

    @swap_image_directory.setter
    def swap_image_directory(self, path: Optional[str]):
        """Set the swap image directory path."""
        if path != self._swap_image_directory:
            self._swap_image_directory = path
            self._emit_state_change("swap_image_directory", path)

    @property
    def swap_grids_directory(self) -> Optional[str]:
        """Get the swap grids directory path."""
        return self._swap_grids_directory

    @swap_grids_directory.setter
    def swap_grids_directory(self, path: Optional[str]):
        """Set the swap grids directory path."""
        if path != self._swap_grids_directory:
            self._swap_grids_directory = path
            self._emit_state_change("swap_grids_directory", path)

    @property
    def swap_stiv_directory(self) -> Optional[str]:
        """Get the swap STIV directory path."""
        return self._swap_stiv_directory

    @swap_stiv_directory.setter
    def swap_stiv_directory(self, path: Optional[str]):
        """Set the swap STIV directory path."""
        if path != self._swap_stiv_directory:
            self._swap_stiv_directory = path
            self._emit_state_change("swap_stiv_directory", path)

    def reset(self):
        """Reset model to initial state (close project)."""
        self._project_filename = None
        self._is_project_loaded = False
        self._project_name = None
        self._swap_directory = None
        self._swap_image_directory = None
        self._swap_grids_directory = None
        self._swap_stiv_directory = None
        self.project_closed.emit()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model state to dictionary.

        Returns:
            Dictionary representation of project state
        """
        return {
            "project_filename": self._project_filename,
            "is_project_loaded": self._is_project_loaded,
            "project_name": self._project_name,
            "swap_directory": self._swap_directory,
            "swap_image_directory": self._swap_image_directory,
            "swap_grids_directory": self._swap_grids_directory,
            "swap_stiv_directory": self._swap_stiv_directory,
        }

    def from_dict(self, data: Dict[str, Any]):
        """Deserialize model state from dictionary.

        Args:
            data: Dictionary containing model state
        """
        if "project_filename" in data:
            self.project_filename = data["project_filename"]
        if "project_name" in data:
            self.project_name = data["project_name"]
        if "swap_directory" in data:
            self.swap_directory = data["swap_directory"]
        if "swap_image_directory" in data:
            self.swap_image_directory = data["swap_image_directory"]
        if "swap_grids_directory" in data:
            self.swap_grids_directory = data["swap_grids_directory"]
        if "swap_stiv_directory" in data:
            self.swap_stiv_directory = data["swap_stiv_directory"]
