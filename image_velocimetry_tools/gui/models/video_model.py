"""Model for video state management."""

from typing import Optional, Dict, Any
from pathlib import Path
from PyQt5.QtCore import pyqtSignal

from image_velocimetry_tools.gui.models.base_model import BaseModel


class VideoModel(BaseModel):
    """Model representing video state.

    This model holds all video-related state including:
    - Video file information (path, metadata)
    - Playback state (position, duration)
    - Clip times (start, end)
    - Processing parameters (rotation, flip, etc.)

    Signals:
        video_loaded: Emitted when a new video is loaded (file_path)
        video_unloaded: Emitted when video is unloaded
        metadata_changed: Emitted when video metadata is updated
        clip_times_changed: Emitted when clip start/end times change
        processing_params_changed: Emitted when processing parameters change
    """

    # Qt Signals
    video_loaded = pyqtSignal(str)  # file_path
    video_unloaded = pyqtSignal()
    metadata_changed = pyqtSignal(dict)  # metadata dict
    clip_times_changed = pyqtSignal(int, int)  # start_ms, end_ms
    processing_params_changed = pyqtSignal()

    def __init__(self):
        """Initialize the video model with default values."""
        super().__init__()

        # Video file information
        self._video_file_name: Optional[str] = None
        self._is_video_loaded: bool = False

        # Video metadata
        self._video_metadata: Dict[str, Any] = {}
        self._video_duration: int = 0  # milliseconds
        self._video_resolution: str = ""
        self._video_frame_rate: float = 0.0
        self._video_num_frames: int = 0
        self._video_timestep_ms: float = 0.0

        # Clip times
        self._video_clip_start_time: int = 0  # milliseconds
        self._video_clip_end_time: int = 0  # milliseconds
        self._video_clip_filename: Optional[str] = None

        # Processing parameters
        self._video_rotation: int = 0
        self._video_flip: str = "none"
        self._video_strip_audio: bool = False
        self._video_normalize_luma: bool = False
        self._video_curve_preset: str = "none"
        self._video_ffmpeg_stabilize: bool = False

    # Video file properties
    @property
    def video_file_name(self) -> Optional[str]:
        """Get the current video file path."""
        return self._video_file_name

    @video_file_name.setter
    def video_file_name(self, path: Optional[str]):
        """Set the video file path."""
        if path != self._video_file_name:
            self._video_file_name = path
            self._is_video_loaded = path is not None
            if path:
                self.video_loaded.emit(path)
            else:
                self.video_unloaded.emit()
            self._emit_state_change("video_file_name", path)

    @property
    def is_video_loaded(self) -> bool:
        """Check if a video is currently loaded."""
        return self._is_video_loaded

    # Metadata properties
    @property
    def video_metadata(self) -> Dict[str, Any]:
        """Get video metadata dictionary."""
        return self._video_metadata

    @video_metadata.setter
    def video_metadata(self, metadata: Dict[str, Any]):
        """Set video metadata and extract common fields."""
        self._video_metadata = metadata

        # Extract commonly-used fields
        if "duration" in metadata:
            self._video_duration = metadata["duration"]
        if "width" in metadata and "height" in metadata:
            self._video_resolution = f"{metadata['width']}x{metadata['height']}"
        if "avg_frame_rate" in metadata:
            self._video_frame_rate = metadata["avg_frame_rate"]
        if "frame_count" in metadata:
            self._video_num_frames = metadata["frame_count"]
        if "avg_timestep_ms" in metadata:
            self._video_timestep_ms = metadata["avg_timestep_ms"]

        self.metadata_changed.emit(metadata)
        self._emit_state_change("video_metadata", metadata)

    @property
    def video_duration(self) -> int:
        """Get video duration in milliseconds."""
        return self._video_duration

    @property
    def video_resolution(self) -> str:
        """Get video resolution as 'widthxheight' string."""
        return self._video_resolution

    @property
    def video_frame_rate(self) -> float:
        """Get video frame rate in fps."""
        return self._video_frame_rate

    @property
    def video_num_frames(self) -> int:
        """Get total number of frames in video."""
        return self._video_num_frames

    @property
    def video_timestep_ms(self) -> float:
        """Get average timestep between frames in milliseconds."""
        return self._video_timestep_ms

    # Clip time properties
    @property
    def video_clip_start_time(self) -> int:
        """Get clip start time in milliseconds."""
        return self._video_clip_start_time

    @video_clip_start_time.setter
    def video_clip_start_time(self, time_ms: int):
        """Set clip start time in milliseconds."""
        if time_ms != self._video_clip_start_time:
            self._video_clip_start_time = time_ms
            self.clip_times_changed.emit(self._video_clip_start_time, self._video_clip_end_time)
            self._emit_state_change("video_clip_start_time", time_ms)

    @property
    def video_clip_end_time(self) -> int:
        """Get clip end time in milliseconds."""
        return self._video_clip_end_time

    @video_clip_end_time.setter
    def video_clip_end_time(self, time_ms: int):
        """Set clip end time in milliseconds."""
        if time_ms != self._video_clip_end_time:
            self._video_clip_end_time = time_ms
            self.clip_times_changed.emit(self._video_clip_start_time, self._video_clip_end_time)
            self._emit_state_change("video_clip_end_time", time_ms)

    @property
    def video_clip_filename(self) -> Optional[str]:
        """Get output filename for video clip."""
        return self._video_clip_filename

    @video_clip_filename.setter
    def video_clip_filename(self, filename: Optional[str]):
        """Set output filename for video clip."""
        if filename != self._video_clip_filename:
            self._video_clip_filename = filename
            self._emit_state_change("video_clip_filename", filename)

    # Processing parameter properties
    @property
    def video_rotation(self) -> int:
        """Get video rotation angle in degrees."""
        return self._video_rotation

    @video_rotation.setter
    def video_rotation(self, angle: int):
        """Set video rotation angle in degrees."""
        if angle != self._video_rotation:
            self._video_rotation = angle
            self.processing_params_changed.emit()
            self._emit_state_change("video_rotation", angle)

    @property
    def video_flip(self) -> str:
        """Get video flip direction ('none', 'hflip', 'vflip')."""
        return self._video_flip

    @video_flip.setter
    def video_flip(self, direction: str):
        """Set video flip direction."""
        if direction != self._video_flip:
            self._video_flip = direction
            self.processing_params_changed.emit()
            self._emit_state_change("video_flip", direction)

    @property
    def video_strip_audio(self) -> bool:
        """Check if audio should be stripped."""
        return self._video_strip_audio

    @video_strip_audio.setter
    def video_strip_audio(self, strip: bool):
        """Set whether to strip audio."""
        if strip != self._video_strip_audio:
            self._video_strip_audio = strip
            self.processing_params_changed.emit()
            self._emit_state_change("video_strip_audio", strip)

    @property
    def video_normalize_luma(self) -> bool:
        """Check if luma normalization is enabled."""
        return self._video_normalize_luma

    @video_normalize_luma.setter
    def video_normalize_luma(self, normalize: bool):
        """Set whether to normalize luma."""
        if normalize != self._video_normalize_luma:
            self._video_normalize_luma = normalize
            self.processing_params_changed.emit()
            self._emit_state_change("video_normalize_luma", normalize)

    @property
    def video_curve_preset(self) -> str:
        """Get curve preset name."""
        return self._video_curve_preset

    @video_curve_preset.setter
    def video_curve_preset(self, preset: str):
        """Set curve preset name."""
        if preset != self._video_curve_preset:
            self._video_curve_preset = preset
            self.processing_params_changed.emit()
            self._emit_state_change("video_curve_preset", preset)

    @property
    def video_ffmpeg_stabilize(self) -> bool:
        """Check if video stabilization is enabled."""
        return self._video_ffmpeg_stabilize

    @video_ffmpeg_stabilize.setter
    def video_ffmpeg_stabilize(self, stabilize: bool):
        """Set whether to stabilize video."""
        if stabilize != self._video_ffmpeg_stabilize:
            self._video_ffmpeg_stabilize = stabilize
            self.processing_params_changed.emit()
            self._emit_state_change("video_ffmpeg_stabilize", stabilize)

    def set_clip_times(self, start_ms: int, end_ms: int):
        """Set both clip start and end times at once.

        Args:
            start_ms: Clip start time in milliseconds
            end_ms: Clip end time in milliseconds
        """
        changed = False
        if start_ms != self._video_clip_start_time:
            self._video_clip_start_time = start_ms
            changed = True
        if end_ms != self._video_clip_end_time:
            self._video_clip_end_time = end_ms
            changed = True

        if changed:
            self.clip_times_changed.emit(start_ms, end_ms)
            self._emit_state_change("clip_times", (start_ms, end_ms))

    def clear_clip_times(self):
        """Reset clip times to default (full video)."""
        self.set_clip_times(0, 0)

    def reset(self):
        """Reset model to initial state (unload video)."""
        self.video_file_name = None
        self._video_metadata = {}
        self._video_duration = 0
        self._video_resolution = ""
        self._video_frame_rate = 0.0
        self._video_num_frames = 0
        self._video_timestep_ms = 0.0
        self.clear_clip_times()
        self._video_clip_filename = None
        self._video_rotation = 0
        self._video_flip = "none"
        self._video_strip_audio = False
        self._video_normalize_luma = False
        self._video_curve_preset = "none"
        self._video_ffmpeg_stabilize = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model state to dictionary.

        Returns:
            Dictionary representation of video state
        """
        return {
            "video_file_name": self._video_file_name,
            "is_video_loaded": self._is_video_loaded,
            "video_metadata": self._video_metadata,
            "video_clip_start_time": self._video_clip_start_time,
            "video_clip_end_time": self._video_clip_end_time,
            "video_clip_filename": self._video_clip_filename,
            "video_rotation": self._video_rotation,
            "video_flip": self._video_flip,
            "video_strip_audio": self._video_strip_audio,
            "video_normalize_luma": self._video_normalize_luma,
            "video_curve_preset": self._video_curve_preset,
            "video_ffmpeg_stabilize": self._video_ffmpeg_stabilize,
        }

    def from_dict(self, data: Dict[str, Any]):
        """Deserialize model state from dictionary.

        Args:
            data: Dictionary containing model state
        """
        if "video_file_name" in data:
            self.video_file_name = data["video_file_name"]
        if "video_metadata" in data:
            self.video_metadata = data["video_metadata"]
        if "video_clip_start_time" in data and "video_clip_end_time" in data:
            self.set_clip_times(data["video_clip_start_time"], data["video_clip_end_time"])
        if "video_clip_filename" in data:
            self.video_clip_filename = data["video_clip_filename"]
        if "video_rotation" in data:
            self.video_rotation = data["video_rotation"]
        if "video_flip" in data:
            self.video_flip = data["video_flip"]
        if "video_strip_audio" in data:
            self.video_strip_audio = data["video_strip_audio"]
        if "video_normalize_luma" in data:
            self.video_normalize_luma = data["video_normalize_luma"]
        if "video_curve_preset" in data:
            self.video_curve_preset = data["video_curve_preset"]
        if "video_ffmpeg_stabilize" in data:
            self.video_ffmpeg_stabilize = data["video_ffmpeg_stabilize"]
