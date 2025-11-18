"""Tests for VideoController.

Note: These tests require PyQt5 to be available. Many tests use mocks
to avoid needing a full Qt application running.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

# Skip all tests if PyQt5 is not available
pytest.importorskip("PyQt5")

from PyQt5.QtCore import QObject
from image_velocimetry_tools.gui.controllers.video_controller import VideoController
from image_velocimetry_tools.gui.models.video_model import VideoModel
from image_velocimetry_tools.services.video_service import VideoService


class TestVideoControllerInit:
    """Tests for VideoController initialization."""

    @pytest.fixture
    def mock_main_window(self):
        """Create a mock main window with required attributes."""
        mock_window = Mock()
        mock_window.video_player = Mock()
        mock_window.buttonPlay = Mock()
        mock_window.sliderVideoPlayHead = Mock()
        mock_window.buttonClipStart = Mock()
        mock_window.buttonClipEnd = Mock()
        mock_window.buttonClearClip = Mock()
        mock_window.labelVideoPlayheadTime = Mock()
        mock_window.labelVideoDuration = Mock()
        mock_window.labelVideoFramerateValue = Mock()
        mock_window.labelVideoTimestepValue = Mock()
        mock_window.labelNumOfFramesValue = Mock()
        mock_window.labelVideoResolutionValue = Mock()
        mock_window.labelStartFrameValue = Mock()
        mock_window.labelEndFrameValue = Mock()
        mock_window.labelNewFrameRateValue = Mock()
        mock_window.labelNewTimestepValue = Mock()
        mock_window.labelNewNumFramesValue = Mock()
        mock_window.labelLensCxValue = Mock()
        mock_window.labelLensCyValue = Mock()
        mock_window.labelLensK1Value = Mock()
        mock_window.labelLensK2Value = Mock()
        mock_window.labelVideoPreload = Mock()
        mock_window.groupboxFrameExtraction = Mock()
        mock_window.signal_opencv_updates = Mock()
        mock_window.sticky_settings = Mock()
        mock_window.update_statusbar = Mock()
        mock_window.wait_cursor = Mock()
        mock_window._IvyTools__program_name__ = "Test IVy"
        mock_window._IvyTools__version__ = "1.0.0"
        mock_window._IvyTools__icon_path__ = "icons"
        return mock_window

    @pytest.fixture
    def video_model(self):
        """Create a VideoModel instance."""
        return VideoModel()

    @pytest.fixture
    def video_service(self):
        """Create a VideoService instance."""
        return VideoService()

    def test_controller_init(self, mock_main_window, video_model, video_service):
        """Test controller initialization."""
        controller = VideoController(mock_main_window, video_model, video_service)

        assert controller.main_window == mock_main_window
        assert controller.video_model == video_model
        assert controller.video_service == video_service
        assert controller.model == video_model
        assert controller.service == video_service


class TestVideoControllerClipTimes:
    """Tests for clip time management."""

    @pytest.fixture
    def setup(self):
        """Set up controller with mocks."""
        mock_window = Mock()
        mock_window.sliderVideoPlayHead = Mock()
        mock_window.sliderVideoPlayHead.sliderPosition.return_value = 5000
        mock_window.buttonClipStart = Mock()
        mock_window.buttonClipEnd = Mock()
        mock_window.labelStartFrameValue = Mock()
        mock_window.labelEndFrameValue = Mock()
        mock_window.update_statusbar = Mock()

        # Minimal signal connections setup
        mock_window.video_player = Mock()
        mock_window.buttonPlay = Mock()
        mock_window.buttonClearClip = Mock()
        mock_window.labelVideoPlayheadTime = Mock()
        mock_window.labelVideoDuration = Mock()
        mock_window.labelVideoFramerateValue = Mock()
        mock_window.labelVideoTimestepValue = Mock()
        mock_window.labelNumOfFramesValue = Mock()
        mock_window.labelVideoResolutionValue = Mock()
        mock_window.labelNewFrameRateValue = Mock()
        mock_window.labelNewTimestepValue = Mock()
        mock_window.labelNewNumFramesValue = Mock()
        mock_window.labelLensCxValue = Mock()
        mock_window.labelLensCyValue = Mock()
        mock_window.labelLensK1Value = Mock()
        mock_window.labelLensK2Value = Mock()
        mock_window.labelVideoPreload = Mock()
        mock_window.groupboxFrameExtraction = Mock()

        video_model = VideoModel()
        video_model.video_file_name = "/test/video.mp4"
        video_model.video_metadata = {
            "duration": 30000,
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": 30.0,
            "frame_count": 900,
            "avg_timestep_ms": 33.33
        }

        video_service = VideoService()

        controller = VideoController(mock_window, video_model, video_service)

        return {
            "controller": controller,
            "window": mock_window,
            "model": video_model,
            "service": video_service
        }

    def test_set_clip_start_time(self, setup):
        """Test setting clip start time."""
        controller = setup["controller"]
        model = setup["model"]
        window = setup["window"]

        # Simulate setting clip start time
        controller.on_set_clip_start()

        # Clip start should be set to slider position
        assert model.video_clip_start_time == 5000

    def test_set_clip_end_time(self, setup):
        """Test setting clip end time."""
        controller = setup["controller"]
        model = setup["model"]

        # Simulate setting clip end time
        controller.on_set_clip_end()

        # Clip end should be set to slider position
        assert model.video_clip_end_time == 5000

    def test_clear_clip_times(self, setup):
        """Test clearing clip times."""
        controller = setup["controller"]
        model = setup["model"]

        # Set some clip times
        model.video_clip_start_time = 5000
        model.video_clip_end_time = 10000

        # Clear them
        controller.on_clear_clip_times()

        # Should be reset to 0
        assert model.video_clip_start_time == 0
        assert model.video_clip_end_time == 0

    def test_clip_start_cannot_exceed_end(self, setup):
        """Test that clip start time cannot exceed end time."""
        controller = setup["controller"]
        model = setup["model"]
        window = setup["window"]

        # Set end time to 3000ms
        model.video_clip_end_time = 3000

        # Try to set start time to 5000ms (slider position)
        controller.on_set_clip_start()

        # Start time should be clamped to end time
        assert model.video_clip_start_time == 3000


class TestVideoControllerModelSignals:
    """Tests for handling model signals."""

    @pytest.fixture
    def setup(self):
        """Set up controller with mocks."""
        mock_window = Mock()
        mock_window.video_player = Mock()
        mock_window.buttonPlay = Mock()
        mock_window.sliderVideoPlayHead = Mock()
        mock_window.buttonClipStart = Mock()
        mock_window.buttonClipEnd = Mock()
        mock_window.buttonClearClip = Mock()
        mock_window.labelVideoPlayheadTime = Mock()
        mock_window.labelVideoDuration = Mock()
        mock_window.labelVideoFramerateValue = Mock()
        mock_window.labelVideoTimestepValue = Mock()
        mock_window.labelNumOfFramesValue = Mock()
        mock_window.labelVideoResolutionValue = Mock()
        mock_window.labelStartFrameValue = Mock()
        mock_window.labelEndFrameValue = Mock()
        mock_window.labelNewFrameRateValue = Mock()
        mock_window.labelNewTimestepValue = Mock()
        mock_window.labelNewNumFramesValue = Mock()
        mock_window.labelLensCxValue = Mock()
        mock_window.labelLensCyValue = Mock()
        mock_window.labelLensK1Value = Mock()
        mock_window.labelLensK2Value = Mock()
        mock_window.labelVideoPreload = Mock()
        mock_window.groupboxFrameExtraction = Mock()
        mock_window.update_statusbar = Mock()
        mock_window.setWindowTitle = Mock()
        mock_window._IvyTools__program_name__ = "Test"
        mock_window._IvyTools__version__ = "1.0"

        video_model = VideoModel()
        video_service = VideoService()
        controller = VideoController(mock_window, video_model, video_service)

        return {
            "controller": controller,
            "window": mock_window,
            "model": video_model
        }

    def test_on_model_video_loaded(self, setup):
        """Test handling video loaded signal."""
        controller = setup["controller"]
        window = setup["window"]

        # Simulate video loaded signal
        controller.on_model_video_loaded("/test/video.mp4")

        # Video player should be updated
        window.video_player.setMedia.assert_called_once()
        window.setWindowTitle.assert_called_once()

    def test_on_model_metadata_changed(self, setup):
        """Test handling metadata changed signal."""
        controller = setup["controller"]
        window = setup["window"]
        model = setup["model"]

        # Set metadata
        metadata = {
            "duration": 30000,
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": 30.0,
            "frame_count": 900,
            "avg_timestep_ms": 33.33
        }
        model.video_metadata = metadata

        # Simulate metadata changed signal
        controller.on_model_metadata_changed(metadata)

        # Labels should be updated
        window.labelVideoFramerateValue.setText.assert_called()
        window.labelVideoTimestepValue.setText.assert_called()
        window.labelNumOfFramesValue.setText.assert_called()
        window.labelVideoResolutionValue.setText.assert_called()
        window.groupboxFrameExtraction.setEnabled.assert_called_with(True)
