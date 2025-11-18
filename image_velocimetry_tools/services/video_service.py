"""Service for video processing business logic."""

import os
from pathlib import Path
from typing import Optional

from image_velocimetry_tools.services.base_service import BaseService
from image_velocimetry_tools.common_functions import seconds_to_hhmmss


class VideoService(BaseService):
    """Service for video processing operations.

    This service handles video-related business logic including:
    - Clip time validation
    - FFmpeg parameter building
    - Output filename generation

    This service has no Qt dependencies and can be used from CLI or GUI.
    """

    def __init__(self):
        """Initialize the video service."""
        super().__init__()

    def validate_clip_times(
        self,
        start_time_ms: float,
        end_time_ms: float,
        video_duration_ms: float
    ) -> None:
        """Validate clip start and end times.

        Args:
            start_time_ms: Clip start time in milliseconds
            end_time_ms: Clip end time in milliseconds
            video_duration_ms: Total video duration in milliseconds

        Raises:
            ValueError: If times are invalid
        """
        if video_duration_ms <= 0:
            raise ValueError("Video duration must be positive")

        if start_time_ms < 0:
            raise ValueError("Start time cannot be negative")

        if end_time_ms <= start_time_ms:
            raise ValueError("End time must be greater than start time")

        if end_time_ms > video_duration_ms:
            raise ValueError("End time cannot exceed video duration")

        self.logger.debug(
            f"Validated clip times: start={start_time_ms}ms, "
            f"end={end_time_ms}ms, duration={video_duration_ms}ms"
        )

    def generate_clip_filename(
        self,
        input_video_path: str,
        start_time_ms: float,
        end_time_ms: float,
        output_dir: str,
        rotation: int = 0,
        flip: str = "none",
        normalize_luma: bool = False,
        curve_preset: str = "none",
        stabilize: bool = False
    ) -> str:
        """Generate output filename for video clip based on processing parameters.

        The filename includes metadata about the processing applied:
        - Start and end times
        - Rotation angle (if > 0)
        - Flip direction (if not "none")
        - Luma normalization (if enabled)
        - Curve preset (if not "none")
        - Stabilization (if enabled)

        Args:
            input_video_path: Path to input video file
            start_time_ms: Clip start time in milliseconds
            end_time_ms: Clip end time in milliseconds
            output_dir: Directory for output file
            rotation: Rotation angle in degrees
            flip: Flip direction ("none", "hflip", "vflip")
            normalize_luma: Whether luma normalization is enabled
            curve_preset: Curve preset name
            stabilize: Whether stabilization is enabled

        Returns:
            Full path to output video file
        """
        # Build metadata components for filename
        file_meta = []

        # Add time range
        file_meta.append(
            f"s{start_time_ms / 1000:05.3f}s_e{end_time_ms / 1000:05.3f}s"
        )

        # Add rotation if specified
        if rotation > 0:
            file_meta.append(f"rot{rotation}")

        # Add flip if specified
        if flip is not None and flip != "none":
            file_meta.append(f"{flip}")

        # Add luma normalization if enabled
        if normalize_luma:
            file_meta.append("normluma")

        # Add curve preset if specified
        if curve_preset is not None and curve_preset != "none":
            file_meta.append(f"{curve_preset}")

        # Add stabilization if enabled
        if stabilize:
            file_meta.append("stab")

        # Join metadata components
        file_str_middle = "_".join(map(str, file_meta))

        # Extract basename and extension from input video
        basename = os.path.basename(input_video_path) if input_video_path else "video.mp4"

        # Ensure basename has a valid file extension
        if "." in basename:
            name, ext = basename.rsplit(".", 1)
        else:
            name, ext = basename, "mp4"  # Default to .mp4 if no extension

        # Construct output filename
        output_filename = f"{name}_{file_str_middle}.{ext}"

        # Construct full output path
        output_path = os.path.join(output_dir, output_filename)

        self.logger.debug(f"Generated clip filename: {output_path}")

        return output_path

    def build_ffmpeg_parameters(
        self,
        input_video: str,
        output_video: str,
        start_time_ms: float,
        end_time_ms: float,
        rotation: int = 0,
        flip: str = "none",
        strip_audio: bool = False,
        normalize_luma: bool = False,
        curve_preset: str = "none",
        stabilize: bool = False,
        extract_frames: bool = False,
        extract_frame_step: int = 1,
        extracted_frames_folder: str = "",
        extract_frame_pattern: str = "f%05d.jpg",
        calibrate_radial: bool = False,
        cx: float = 0.0,
        cy: float = 0.0,
        k1: float = 0.0,
        k2: float = 0.0
    ) -> dict:
        """Build FFmpeg parameters dictionary.

        This creates a parameter dictionary that can be passed to
        create_ffmpeg_command() from ffmpeg_tools module.

        Args:
            input_video: Path to input video file
            output_video: Path to output video file
            start_time_ms: Clip start time in milliseconds
            end_time_ms: Clip end time in milliseconds
            rotation: Rotation angle in degrees
            flip: Flip direction ("none", "hflip", "vflip")
            strip_audio: Whether to remove audio from output
            normalize_luma: Whether to normalize luma
            curve_preset: Curve preset name
            stabilize: Whether to stabilize video
            extract_frames: Whether to extract frames instead of video
            extract_frame_step: Frame step for extraction (extract every Nth frame)
            extracted_frames_folder: Output folder for extracted frames
            extract_frame_pattern: Filename pattern for extracted frames
            calibrate_radial: Whether to apply radial distortion correction
            cx: Lens center X coordinate (dimensionless)
            cy: Lens center Y coordinate (dimensionless)
            k1: Radial distortion coefficient k1
            k2: Radial distortion coefficient k2

        Returns:
            Dictionary of FFmpeg parameters
        """
        # Convert times from milliseconds to HH:MM:SS.mmm format
        start_time_str = seconds_to_hhmmss(start_time_ms / 1000, precision="high")
        end_time_str = seconds_to_hhmmss(end_time_ms / 1000, precision="high")

        # Build parameters dictionary
        params = {
            "input_video": input_video,
            "output_video": output_video,
            "start_time": start_time_str,
            "end_time": end_time_str,
            "video_rotation": rotation,
            "video_flip": flip,
            "strip_audio": strip_audio,
            "normalize_luma": normalize_luma,
            "curve_preset": curve_preset,
            "stabilize": stabilize,
            "extract_frames": extract_frames,
            "extract_frame_step": extract_frame_step,
            "extracted_frames_folder": extracted_frames_folder,
            "extract_frame_pattern": extract_frame_pattern,
            "calibrate_radial": calibrate_radial,
            "cx": cx,
            "cy": cy,
            "k1": k1,
            "k2": k2,
        }

        self.logger.debug(f"Built FFmpeg parameters: {params}")

        return params
