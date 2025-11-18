"""Tests for ProjectController."""

import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest
from PyQt5.QtCore import QDir, QCoreApplication
from PyQt5.QtWidgets import QApplication

from image_velocimetry_tools.gui.controllers.project_controller import ProjectController
from image_velocimetry_tools.gui.models.project_model import ProjectModel
from image_velocimetry_tools.services.project_service import ProjectService


# Create QApplication instance for Qt testing
# This must be done before creating any QObject instances
@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    # Don't call app.quit() here as it can cause issues with other tests


@pytest.fixture
def mock_main_window(qapp):
    """Create a mock main window for testing."""
    mw = MagicMock()

    # Mock sticky settings
    mw.sticky_settings = MagicMock()
    mw.sticky_settings.get = MagicMock(return_value=None)
    mw.sticky_settings.set = MagicMock()
    mw.sticky_settings.new = MagicMock()

    # Mock directories
    mw.swap_directory = "/tmp/swap"
    mw.swap_image_directory = "/tmp/swap/1-Images"
    mw.swap_grids_directory = "/tmp/swap/2-Grids"
    mw.swap_stiv_directory = "/tmp/swap/3-STIV"

    # Mock icon path
    mw.__icon_path__ = "/icons"

    # Mock warning dialog
    mw.warning_dialog = MagicMock()

    # Mock window title
    mw.setWindowTitle = MagicMock()

    # Mock __dict__ for project data extraction
    mw.__dict__ = {
        "project_filename": None,
        "video_file_name": None,
        "is_video_loaded": False,
        "test_string": "test_value",
        "test_int": 42,
        "test_float": 3.14,
        "test_bool": True,
        "test_list": [1, 2, 3],
        "test_dict": {"key": "value"},
        "test_none": None,
    }

    return mw


@pytest.fixture(scope="function")
def project_model(qapp):
    """Create a fresh ProjectModel instance for each test."""
    model = ProjectModel()
    yield model
    # Cleanup: block all signals to prevent issues during teardown
    model.blockSignals(True)


@pytest.fixture(scope="function")
def project_service(qapp):
    """Create a fresh ProjectService instance for each test."""
    return ProjectService()


@pytest.fixture(scope="function")
def project_controller(qapp, mock_main_window, project_model, project_service):
    """Create a fresh ProjectController instance for each test."""
    controller = ProjectController(mock_main_window, project_model, project_service)
    yield controller
    # Cleanup: block all signals before destruction
    controller.blockSignals(True)
    project_model.blockSignals(True)


class TestProjectControllerInit:
    """Tests for ProjectController initialization."""

    def test_init_stores_references(self, project_controller, mock_main_window, project_model, project_service):
        """Test that controller stores references to main window, model, and service."""
        assert project_controller.main_window == mock_main_window
        assert project_controller.project_model == project_model
        assert project_controller.project_service == project_service

    def test_init_connects_signals(self, project_controller):
        """Test that signals are connected during initialization."""
        # Verify that signal connections were made by checking receiver count
        assert project_controller.project_model.receivers(
            project_controller.project_model.project_created
        ) > 0


class TestNewProject:
    """Tests for new_project method."""

    def test_new_project_sets_default_filename(self, project_controller, project_model):
        """Test that new project sets a default filename."""
        project_controller.new_project()

        assert project_model.project_filename is not None
        assert project_model.project_filename.endswith("New_IVy_Project.ivy")

    def test_new_project_emits_created_signal(self, project_controller, project_model):
        """Test that new project emits project_created signal."""
        signal_emitted = False

        def on_created():
            nonlocal signal_emitted
            signal_emitted = True

        project_model.project_created.connect(on_created)
        project_controller.new_project()

        assert signal_emitted

    def test_new_project_calls_clear_project(self, project_controller):
        """Test that new project calls clear_project."""
        with patch.object(project_controller, 'clear_project') as mock_clear:
            project_controller.new_project()
            mock_clear.assert_called_once()


class TestOpenProject:
    """Tests for open_project method."""

    @patch('image_velocimetry_tools.gui.controllers.project_controller.QtWidgets.QFileDialog.getOpenFileName')
    def test_open_project_user_cancels(self, mock_dialog, project_controller):
        """Test that open_project returns False when user cancels."""
        mock_dialog.return_value = ("", "")  # User cancelled

        result = project_controller.open_project()

        assert result is False

    @patch('image_velocimetry_tools.gui.controllers.project_controller.QtWidgets.QFileDialog.getOpenFileName')
    def test_open_project_extracts_archive(self, mock_dialog, project_controller, tmp_path):
        """Test that open_project extracts the project archive."""
        project_file = str(tmp_path / "test.ivy")
        mock_dialog.return_value = (project_file, "")

        with patch.object(project_controller.project_service, 'extract_project_archive') as mock_extract:
            with patch.object(project_controller.project_service, 'load_project_from_json') as mock_load:
                mock_load.return_value = {"test": "data"}

                result = project_controller.open_project()

                mock_extract.assert_called_once_with(
                    project_file,
                    project_controller.main_window.swap_directory
                )

    @patch('image_velocimetry_tools.gui.controllers.project_controller.QtWidgets.QFileDialog.getOpenFileName')
    def test_open_project_loads_json(self, mock_dialog, project_controller, tmp_path):
        """Test that open_project loads project data from JSON."""
        project_file = str(tmp_path / "test.ivy")
        mock_dialog.return_value = (project_file, "")

        with patch.object(project_controller.project_service, 'extract_project_archive'):
            with patch.object(project_controller.project_service, 'load_project_from_json') as mock_load:
                mock_load.return_value = {"test": "data"}

                result = project_controller.open_project()

                expected_json_path = os.path.join(
                    project_controller.main_window.swap_directory,
                    "project_data.json"
                )
                mock_load.assert_called_once_with(expected_json_path)

    @patch('image_velocimetry_tools.gui.controllers.project_controller.QtWidgets.QFileDialog.getOpenFileName')
    def test_open_project_updates_model(self, mock_dialog, project_controller, project_model, tmp_path):
        """Test that open_project updates the project model."""
        project_file = str(tmp_path / "my_project.ivy")
        mock_dialog.return_value = (project_file, "")

        with patch.object(project_controller.project_service, 'extract_project_archive'):
            with patch.object(project_controller.project_service, 'load_project_from_json') as mock_load:
                mock_load.return_value = {"test": "data"}

                project_controller.open_project()

                assert project_model.project_filename == project_file
                assert project_model.project_name == "my_project"

    @patch('image_velocimetry_tools.gui.controllers.project_controller.QtWidgets.QFileDialog.getOpenFileName')
    def test_open_project_emits_loaded_signal(self, mock_dialog, project_controller, project_model, tmp_path):
        """Test that open_project emits project_loaded signal."""
        project_file = str(tmp_path / "test.ivy")
        mock_dialog.return_value = (project_file, "")

        signal_emitted = False
        received_path = None

        def on_loaded(path):
            nonlocal signal_emitted, received_path
            signal_emitted = True
            received_path = path

        project_model.project_loaded.connect(on_loaded)

        with patch.object(project_controller.project_service, 'extract_project_archive'):
            with patch.object(project_controller.project_service, 'load_project_from_json') as mock_load:
                mock_load.return_value = {"test": "data"}

                project_controller.open_project()

        assert signal_emitted
        assert received_path == project_file

    @patch('image_velocimetry_tools.gui.controllers.project_controller.QtWidgets.QFileDialog.getOpenFileName')
    def test_open_project_handles_extract_error(self, mock_dialog, project_controller, mock_main_window, tmp_path):
        """Test that open_project handles extraction errors gracefully."""
        project_file = str(tmp_path / "test.ivy")
        mock_dialog.return_value = (project_file, "")

        with patch.object(project_controller.project_service, 'extract_project_archive') as mock_extract:
            mock_extract.side_effect = ValueError("Bad archive")

            result = project_controller.open_project()

            assert result is False
            mock_main_window.warning_dialog.assert_called_once()

    @patch('image_velocimetry_tools.gui.controllers.project_controller.QtWidgets.QFileDialog.getOpenFileName')
    def test_open_project_handles_json_load_error(self, mock_dialog, project_controller, mock_main_window, tmp_path):
        """Test that open_project handles JSON loading errors gracefully."""
        project_file = str(tmp_path / "test.ivy")
        mock_dialog.return_value = (project_file, "")

        with patch.object(project_controller.project_service, 'extract_project_archive'):
            with patch.object(project_controller.project_service, 'load_project_from_json') as mock_load:
                mock_load.side_effect = ValueError("Bad JSON")

                result = project_controller.open_project()

                assert result is False
                mock_main_window.warning_dialog.assert_called_once()


class TestSaveProject:
    """Tests for save_project method."""

    @patch('image_velocimetry_tools.gui.controllers.project_controller.QtWidgets.QFileDialog.getSaveFileName')
    def test_save_project_user_cancels(self, mock_dialog, project_controller):
        """Test that save_project returns False when user cancels."""
        mock_dialog.return_value = ("", "")  # User cancelled

        result = project_controller.save_project({})

        assert result is False

    @patch('image_velocimetry_tools.gui.controllers.project_controller.QtWidgets.QFileDialog.getSaveFileName')
    def test_save_project_adds_ivy_extension(self, mock_dialog, project_controller, tmp_path):
        """Test that save_project adds .ivy extension if missing."""
        save_file = str(tmp_path / "my_project")  # No extension
        mock_dialog.return_value = (save_file, "")

        with patch.object(project_controller.project_service, 'save_project_to_json'):
            with patch.object(project_controller.project_service, 'create_project_archive'):
                project_controller.save_project({"test": "data"})

                # Should have added .ivy extension
                assert project_controller.project_model.project_filename.endswith(".ivy")

    @patch('image_velocimetry_tools.gui.controllers.project_controller.QtWidgets.QFileDialog.getSaveFileName')
    def test_save_project_saves_to_json(self, mock_dialog, project_controller, tmp_path):
        """Test that save_project saves data to JSON."""
        save_file = str(tmp_path / "test.ivy")
        mock_dialog.return_value = (save_file, "")

        project_dict = {"key": "value", "number": 42}

        with patch.object(project_controller.project_service, 'save_project_to_json') as mock_save:
            with patch.object(project_controller.project_service, 'create_project_archive'):
                project_controller.save_project(project_dict)

                expected_json_path = os.path.join(
                    project_controller.main_window.swap_directory,
                    "project_data.json"
                )
                mock_save.assert_called_once_with(project_dict, expected_json_path)

    @patch('image_velocimetry_tools.gui.controllers.project_controller.QtWidgets.QFileDialog.getSaveFileName')
    def test_save_project_creates_archive(self, mock_dialog, project_controller, tmp_path):
        """Test that save_project creates project archive."""
        save_file = str(tmp_path / "test.ivy")
        mock_dialog.return_value = (save_file, "")

        with patch.object(project_controller.project_service, 'save_project_to_json'):
            with patch.object(project_controller.project_service, 'create_project_archive') as mock_archive:
                project_controller.save_project({"test": "data"})

                mock_archive.assert_called_once_with(
                    save_file,
                    project_controller.main_window.swap_directory
                )

    @patch('image_velocimetry_tools.gui.controllers.project_controller.QtWidgets.QFileDialog.getSaveFileName')
    def test_save_project_updates_model(self, mock_dialog, project_controller, project_model, tmp_path):
        """Test that save_project updates the project model."""
        save_file = str(tmp_path / "my_project.ivy")
        mock_dialog.return_value = (save_file, "")

        with patch.object(project_controller.project_service, 'save_project_to_json'):
            with patch.object(project_controller.project_service, 'create_project_archive'):
                project_controller.save_project({"test": "data"})

                assert project_model.project_filename == save_file
                assert project_model.project_name == "my_project"

    @patch('image_velocimetry_tools.gui.controllers.project_controller.QtWidgets.QFileDialog.getSaveFileName')
    def test_save_project_emits_saved_signal(self, mock_dialog, project_controller, project_model, tmp_path):
        """Test that save_project emits project_saved signal."""
        save_file = str(tmp_path / "test.ivy")
        mock_dialog.return_value = (save_file, "")

        signal_emitted = False
        received_path = None

        def on_saved(path):
            nonlocal signal_emitted, received_path
            signal_emitted = True
            received_path = path

        project_model.project_saved.connect(on_saved)

        with patch.object(project_controller.project_service, 'save_project_to_json'):
            with patch.object(project_controller.project_service, 'create_project_archive'):
                project_controller.save_project({"test": "data"})

        assert signal_emitted
        assert received_path == save_file

    @patch('image_velocimetry_tools.gui.controllers.project_controller.QtWidgets.QFileDialog.getSaveFileName')
    def test_save_project_handles_json_save_error(self, mock_dialog, project_controller, mock_main_window, tmp_path):
        """Test that save_project handles JSON saving errors gracefully."""
        save_file = str(tmp_path / "test.ivy")
        mock_dialog.return_value = (save_file, "")

        with patch.object(project_controller.project_service, 'save_project_to_json') as mock_save:
            mock_save.side_effect = ValueError("Cannot save JSON")

            result = project_controller.save_project({"test": "data"})

            assert result is False
            mock_main_window.warning_dialog.assert_called_once()

    @patch('image_velocimetry_tools.gui.controllers.project_controller.QtWidgets.QFileDialog.getSaveFileName')
    def test_save_project_handles_archive_error(self, mock_dialog, project_controller, mock_main_window, tmp_path):
        """Test that save_project handles archive creation errors gracefully."""
        save_file = str(tmp_path / "test.ivy")
        mock_dialog.return_value = (save_file, "")

        with patch.object(project_controller.project_service, 'save_project_to_json'):
            with patch.object(project_controller.project_service, 'create_project_archive') as mock_archive:
                mock_archive.side_effect = IOError("Cannot create archive")

                result = project_controller.save_project({"test": "data"})

                assert result is False
                mock_main_window.warning_dialog.assert_called_once()


class TestClearProject:
    """Tests for clear_project method."""

    def test_clear_project_resets_model(self, project_controller, project_model):
        """Test that clear_project resets the project model."""
        # Set some values
        project_model.project_filename = "/path/to/project.ivy"
        project_model.project_name = "test"

        project_controller.clear_project()

        assert project_model.project_filename is None
        assert project_model.project_name is None
        assert not project_model.is_project_loaded

    def test_clear_project_emits_closed_signal(self, project_controller, project_model):
        """Test that clear_project emits project_closed signal."""
        signal_emitted = False

        def on_closed():
            nonlocal signal_emitted
            signal_emitted = True

        project_model.project_closed.connect(on_closed)
        project_controller.clear_project()

        assert signal_emitted


class TestGetProjectDictFromMainWindow:
    """Tests for get_project_dict_from_main_window method."""

    def test_extracts_serializable_types(self, project_controller, mock_main_window):
        """Test that method extracts only serializable types."""
        project_dict = project_controller.get_project_dict_from_main_window()

        # Should include serializable types
        assert "test_string" in project_dict
        assert "test_int" in project_dict
        assert "test_float" in project_dict
        assert "test_bool" in project_dict
        assert "test_list" in project_dict
        assert "test_dict" in project_dict
        assert "test_none" in project_dict

        # Verify values
        assert project_dict["test_string"] == "test_value"
        assert project_dict["test_int"] == 42
        assert project_dict["test_float"] == 3.14
        assert project_dict["test_bool"] is True
        assert project_dict["test_list"] == [1, 2, 3]
        assert project_dict["test_dict"] == {"key": "value"}
        assert project_dict["test_none"] is None


class TestModelSignalHandlers:
    """Tests for model signal handlers."""

    def test_on_model_project_loaded_updates_window_title(self, project_controller, mock_main_window):
        """Test that project loaded handler updates window title."""
        project_controller.on_model_project_loaded("/path/to/my_project.ivy")

        mock_main_window.setWindowTitle.assert_called_once_with("IVyTools - my_project")

    def test_on_model_project_closed_resets_window_title(self, project_controller, mock_main_window):
        """Test that project closed handler resets window title."""
        project_controller.on_model_project_closed()

        mock_main_window.setWindowTitle.assert_called_once_with("IVyTools")
