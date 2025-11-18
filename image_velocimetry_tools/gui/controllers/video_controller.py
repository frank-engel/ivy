"""Controller for video playback and processing UI coordination."""

import logging
import os
from typing import Optional
from PyQt5.QtCore import pyqtSlot, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5 import QtWidgets, QtGui

from image_velocimetry_tools.gui.controllers.base_controller import BaseController
from image_velocimetry_tools.gui.models.video_model import VideoModel
from image_velocimetry_tools.services.video_service import VideoService
from image_velocimetry_tools.common_functions import (
    float_seconds_to_time_string,
    seconds_to_frame_number,
    seconds_to_hhmmss,
    resource_path,
)
from image_velocimetry_tools.opencv_tools import opencv_get_video_metadata
from image_velocimetry_tools.ffmpeg_tools import ffprobe_add_exif_metadata


class VideoController(BaseController):
    """Controller for video playback and processing UI.

    This controller coordinates between:
    - Video UI widgets (player, sliders, buttons)
    - VideoModel (state management)
    - VideoService (business logic)

    Responsibilities:
    - Video loading and metadata parsing
    - Playback control (play/pause/seek)
    - Clip time management
    - UI state updates based on model changes
    """

    def __init__(
        self,
        main_window,
        video_model: VideoModel,
        video_service: VideoService
    ):
        """Initialize the video controller.

        Args:
            main_window: Reference to main window for widget access
            video_model: Video state model
            video_service: Video business logic service
        """
        super().__init__(main_window, video_model, video_service)
        self.video_model = video_model
        self.video_service = video_service

        # Connect signals after initialization
        self._connect_signals()

    def _connect_signals(self):
        """Connect UI signals to controller methods and model signals to UI updates."""
        mw = self.main_window

        # Video player signals
        mw.video_player.positionChanged.connect(self.on_video_position_changed)
        mw.video_player.durationChanged.connect(self.on_video_duration_changed)
        mw.video_player.stateChanged.connect(self.on_media_state_changed)

        # UI control signals
        mw.buttonPlay.clicked.connect(self.on_play_clicked)
        mw.sliderVideoPlayHead.sliderMoved.connect(self.on_playhead_moved)
        mw.buttonClipStart.clicked.connect(self.on_set_clip_start)
        mw.buttonClipEnd.clicked.connect(self.on_set_clip_end)
        mw.buttonClearClip.clicked.connect(self.on_clear_clip_times)

        # Model signals
        self.video_model.video_loaded.connect(self.on_model_video_loaded)
        self.video_model.metadata_changed.connect(self.on_model_metadata_changed)
        self.video_model.clip_times_changed.connect(self.on_model_clip_times_changed)

        self.logger.debug("Video controller signals connected")

    @pyqtSlot()
    def on_play_clicked(self):
        """Handle play button click - toggle play/pause."""
        mw = self.main_window
        if mw.video_player.state() == QMediaPlayer.PlayingState:
            mw.video_player.pause()
            self.logger.debug("Video paused")
        else:
            mw.video_player.play()
            self.logger.debug("Video playing")

    @pyqtSlot()
    def on_video_position_changed(self, position: int):
        """Handle video position changes from player.

        Updates slider and time label to reflect current position.

        Args:
            position: Current position in milliseconds
        """
        mw = self.main_window
        mw.sliderVideoPlayHead.setValue(position)

        # Update time label with position and frame number
        time_str = float_seconds_to_time_string(position / 1000, precision="second")
        frame_num = seconds_to_frame_number(
            position / 1000,
            self.video_model.video_frame_rate
        )
        mw.labelVideoPlayheadTime.setText(f"{time_str} [{frame_num}]")

    @pyqtSlot()
    def on_video_duration_changed(self, duration: int):
        """Handle video duration changes when video is loaded.

        Updates slider range and duration label.

        Args:
            duration: Video duration in milliseconds
        """
        mw = self.main_window
        mw.sliderVideoPlayHead.setRange(0, duration)

        # Update duration label
        try:
            time_str = float_seconds_to_time_string(duration / 1000, precision="second")
            frame_num = seconds_to_frame_number(
                duration / 1000,
                self.video_model.video_frame_rate
            )
            mw.labelVideoDuration.setText(f"{time_str} [{frame_num}]")
        except Exception as e:
            self.logger.warning(f"Failed to update duration label: {e}")

        # Initialize clip times to full video
        self.video_model.video_clip_start_time = 0
        self.video_model.video_clip_end_time = 0

    @pyqtSlot()
    def on_media_state_changed(self):
        """Update play button icon based on player state."""
        mw = self.main_window
        icon_path = mw.__icon_path__  # Access private attribute for icon path

        if mw.video_player.state() == QMediaPlayer.PlayingState:
            # Show pause icon
            icon = QtGui.QIcon(
                resource_path(icon_path + os.sep + "pause-solid.svg")
            )
        else:
            # Show play icon
            icon = QtGui.QIcon(
                resource_path(icon_path + os.sep + "play-solid.svg")
            )

        mw.buttonPlay.setIcon(icon)

    @pyqtSlot(int)
    def on_playhead_moved(self, position: int):
        """Handle playhead slider movement - seek to new position.

        Args:
            position: New position in milliseconds
        """
        self.main_window.video_player.setPosition(position)

    @pyqtSlot()
    def on_set_clip_start(self):
        """Set clip start time to current playhead position."""
        if not self.video_model.is_video_loaded:
            return

        mw = self.main_window
        start_time = mw.sliderVideoPlayHead.sliderPosition()
        end_time = self.video_model.video_clip_end_time

        # Cannot have start time after end time
        if end_time > 0 and start_time > end_time:
            start_time = end_time

        self.video_model.video_clip_start_time = start_time
        self.logger.debug(f"Clip start time set to {start_time}ms")

    @pyqtSlot()
    def on_set_clip_end(self):
        """Set clip end time to current playhead position."""
        if not self.video_model.is_video_loaded:
            return

        mw = self.main_window
        end_time = mw.sliderVideoPlayHead.sliderPosition()
        start_time = self.video_model.video_clip_start_time

        # Cannot have start time after end time
        if start_time > end_time:
            start_time = end_time

        self.video_model.video_clip_start_time = start_time
        self.video_model.video_clip_end_time = end_time
        self.logger.debug(f"Clip end time set to {end_time}ms")

    @pyqtSlot()
    def on_clear_clip_times(self):
        """Clear clip start and end times (reset to full video)."""
        self.video_model.clear_clip_times()
        self.logger.debug("Clip times cleared")

    @pyqtSlot(str)
    def on_model_video_loaded(self, file_path: str):
        """Handle video loaded signal from model.

        Starts video playback and updates window title.

        Args:
            file_path: Path to loaded video file
        """
        mw = self.main_window

        # Load video into media player
        media_content = QMediaContent(QUrl.fromLocalFile(file_path))
        mw.video_player.setMedia(media_content)

        # Update window title
        mw.setWindowTitle(
            f"{mw._IvyTools__program_name__} v{mw._IvyTools__version__} -- {file_path}"
        )

        self.logger.info(f"Video loaded: {file_path}")

    @pyqtSlot(dict)
    def on_model_metadata_changed(self, metadata: dict):
        """Handle video metadata changes from model.

        Updates UI labels with video information.

        Args:
            metadata: Video metadata dictionary
        """
        mw = self.main_window

        # Update metadata labels
        mw.labelVideoFramerateValue.setText(
            f"{self.video_model.video_frame_rate:.3f} fps"
        )
        mw.labelVideoTimestepValue.setText(
            f"{self.video_model.video_timestep_ms:.4f} ms"
        )
        mw.labelNumOfFramesValue.setText(
            f"{self.video_model.video_num_frames:d}"
        )
        mw.labelVideoResolutionValue.setText(
            f"{self.video_model.video_resolution} px"
        )

        # Update frame range labels
        mw.labelStartFrameValue.setText("0")
        mw.labelEndFrameValue.setText(
            f"{seconds_to_frame_number(self.video_model.video_duration / 1000, self.video_model.video_frame_rate)}"
        )

        # Initialize extraction parameters
        mw.labelNewFrameRateValue.setText(
            f"{self.video_model.video_frame_rate:.3f} fps"
        )
        mw.labelNewTimestepValue.setText(
            f"{self.video_model.video_timestep_ms:.4f} ms"
        )
        mw.labelNewNumFramesValue.setText(
            f"{self.video_model.video_num_frames}"
        )

        # Initialize lens correction defaults
        if "width" in metadata and "height" in metadata:
            mw.labelLensCxValue.setText(f"{metadata['width'] / 2:.3f}")
            mw.labelLensCyValue.setText(f"{metadata['height'] / 2:.3f}")
        mw.labelLensK1Value.setText("0.000")
        mw.labelLensK2Value.setText("0.000")

        # Hide preload label
        mw.labelVideoPreload.setHidden(True)

        # Enable frame extraction controls
        mw.groupboxFrameExtraction.setEnabled(True)

        self.logger.debug("Video metadata UI updated")

    @pyqtSlot(int, int)
    def on_model_clip_times_changed(self, start_ms: int, end_ms: int):
        """Handle clip time changes from model.

        Updates clip button labels and frame labels.

        Args:
            start_ms: Clip start time in milliseconds
            end_ms: Clip end time in milliseconds
        """
        mw = self.main_window

        # Update clip start button
        if start_ms > 0:
            start_str = seconds_to_hhmmss(start_ms / 1000, precision='high')
            mw.buttonClipStart.setText(f"Clip Start [{start_str}]")
            start_frame = seconds_to_frame_number(
                start_ms / 1000,
                self.video_model.video_frame_rate
            )
            mw.labelStartFrameValue.setText(f"{start_frame}")
        else:
            mw.buttonClipStart.setText("Clip Start")
            mw.labelStartFrameValue.setText("0")

        # Update clip end button
        if end_ms > 0:
            end_str = seconds_to_hhmmss(end_ms / 1000, precision='high')
            mw.buttonClipEnd.setText(f"Clip End [{end_str}]")
            end_frame = seconds_to_frame_number(
                end_ms / 1000,
                self.video_model.video_frame_rate
            )
            mw.labelEndFrameValue.setText(f"{end_frame}")
        else:
            mw.buttonClipEnd.setText("Clip End")
            mw.labelEndFrameValue.setText(
                f"{self.video_model.video_num_frames}"
            )

        # Show clip information in status bar
        self._show_clip_information()

    def _show_clip_information(self):
        """Update the statusbar with clip information."""
        if not self.video_model.is_video_loaded:
            return

        start_ms = self.video_model.video_clip_start_time
        end_ms = self.video_model.video_clip_end_time

        if end_ms == 0 or end_ms is None:
            end_ms = self.video_model.video_duration

        if start_ms == 0 and end_ms == self.video_model.video_duration:
            # Full video
            message = "Clip: Full video selected"
        else:
            # Partial clip
            duration_s = (end_ms - start_ms) / 1000
            start_frame = seconds_to_frame_number(
                start_ms / 1000,
                self.video_model.video_frame_rate
            )
            end_frame = seconds_to_frame_number(
                end_ms / 1000,
                self.video_model.video_frame_rate
            )
            num_frames = end_frame - start_frame

            message = (
                f"Clip: {duration_s:.3f}s "
                f"({num_frames} frames from {start_frame} to {end_frame})"
            )

        self.main_window.update_statusbar(message)

    def load_video(self, file_path: str):
        """Load a video file and parse its metadata.

        Args:
            file_path: Path to video file
        """
        self.logger.info(f"Loading video: {file_path}")

        # Get video metadata using opencv
        metadata = opencv_get_video_metadata(
            file_path,
            status_callback=self.main_window.signal_opencv_updates.emit
        )

        # Add EXIF metadata from ffprobe
        metadata = ffprobe_add_exif_metadata(file_path, metadata)

        # Update model
        self.video_model.video_file_name = file_path
        self.video_model.video_metadata = metadata

        self.logger.debug(f"Video metadata loaded: {metadata}")

    def open_video_dialog(self):
        """Show file dialog to open a video file."""
        mw = self.main_window

        # Get last video path from settings
        try:
            last_path = mw.sticky_settings.get("last_video_file_name")
        except KeyError:
            from PyQt5.QtCore import QDir
            last_path = QDir.homePath()

        # Show file dialog
        filter_spec = "Videos (*.mp4 *.mov *.wmv *.avi *.mkv);;All files (*.*)"
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Open Video File",
            last_path,
            filter_spec
        )

        if file_path:
            # Save to settings
            try:
                mw.sticky_settings.set("last_video_file_name", file_path)
            except KeyError:
                mw.sticky_settings.new("last_video_file_name", file_path)

            # Load the video (with wait cursor)
            with mw.wait_cursor():
                self.load_video(file_path)
