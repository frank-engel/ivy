"""Service for video processing business logic."""

import glob
import json
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable

from image_velocimetry_tools.services.base_service import BaseService
from image_velocimetry_tools.common_functions import seconds_to_hhmmss, hhmmss_to_seconds
from image_velocimetry_tools.ffmpeg_tools import create_ffmpeg_command


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

    def get_video_metadata(
        self,
        video_path: str,
        ffprobe_cmd: str = "ffprobe"
    ) -> Dict:
        """Get video metadata using ffprobe.

        This method extracts video properties such as dimensions, frame rate,
        duration, and frame count using ffprobe. Suitable for both GUI and
        batch/headless processing.

        Args:
            video_path: Path to video file
            ffprobe_cmd: Path to ffprobe executable (default: "ffprobe")

        Returns:
            Dictionary with video metadata:
                - width: Video width in pixels
                - height: Video height in pixels
                - avg_frame_rate: Average frame rate (fps)
                - duration: Video duration in seconds
                - frame_count: Total number of frames

        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If ffprobe execution fails
        """
        # Validate video exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cmd = [
            ffprobe_cmd,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,duration,nb_frames',
            '-of', 'json',
            video_path
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            output = json.loads(result.stdout)

            if 'streams' not in output or len(output['streams']) == 0:
                raise RuntimeError("No video stream found in file")

            stream = output['streams'][0]

            # Parse frame rate (e.g., "30000/1001" -> 29.97)
            frame_rate_str = stream.get('r_frame_rate', '30/1')
            num, denom = map(int, frame_rate_str.split('/'))
            avg_frame_rate = num / denom if denom != 0 else 30.0

            metadata = {
                "width": int(stream.get('width', 1920)),
                "height": int(stream.get('height', 1080)),
                "avg_frame_rate": avg_frame_rate,
                "duration": float(stream.get('duration', 0.0)),
                "frame_count": int(stream.get('nb_frames', 0)),
            }

            self.logger.debug(
                f"Video metadata: {metadata['width']}x{metadata['height']}, "
                f"{metadata['avg_frame_rate']:.2f} fps, "
                f"{metadata['duration']:.2f}s, "
                f"{metadata['frame_count']} frames"
            )

            return metadata

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else "Unknown error"
            raise RuntimeError(f"ffprobe failed: {error_msg}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse ffprobe output: {e}")
        except Exception as e:
            raise RuntimeError(f"Error getting video metadata: {e}")

    def extract_frames(
        self,
        video_path: str,
        output_directory: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        frame_step: int = 1,
        frame_pattern: str = "f%05d.jpg",
        max_frames: Optional[int] = None,
        ffmpeg_cmd: str = "ffmpeg",
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Tuple[List[str], Dict]:
        """Extract frames from video using ffmpeg.

        This method is designed for both GUI (with progress callback) and
        batch/headless processing. It extracts frames from a video file
        to individual image files.

        Args:
            video_path: Path to input video file
            output_directory: Directory to save extracted frames
            start_time: Optional start time in seconds (float)
            end_time: Optional end time in seconds (float)
            frame_step: Extract every Nth frame (1 = all frames)
            frame_pattern: Filename pattern for frames (default: "f%05d.jpg")
            max_frames: Maximum number of frames to extract (None = no limit)
            ffmpeg_cmd: Path to ffmpeg executable (default: "ffmpeg")
            progress_callback: Optional callback(percent, message) for progress updates

        Returns:
            Tuple of:
                - List of extracted frame file paths (sorted)
                - Dictionary with extraction metadata:
                    * num_frames: Number of frames extracted
                    * frame_rate: Video frame rate (fps)
                    * frame_step: Frame step used
                    * timestep_ms: Time between frames in milliseconds
                    * start_time: Start time used (HH:MM:SS)
                    * end_time: End time used (HH:MM:SS)
                    * output_directory: Output directory path

        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If ffmpeg execution fails
            ValueError: If parameters are invalid
        """
        # Validate inputs
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if frame_step < 1:
            raise ValueError("frame_step must be >= 1")

        # Create output directory
        os.makedirs(output_directory, exist_ok=True)

        # Report progress
        if progress_callback:
            progress_callback(5, "Getting video metadata...")

        # Get video metadata
        try:
            metadata = self.get_video_metadata(video_path, ffprobe_cmd=ffmpeg_cmd.replace('ffmpeg', 'ffprobe'))
            frame_rate = metadata.get("avg_frame_rate", 30.0)
        except Exception as e:
            self.logger.warning(f"Failed to get metadata, using defaults: {e}")
            frame_rate = 30.0

        # Report progress
        if progress_callback:
            progress_callback(10, "Building ffmpeg command...")

        # Build ffmpeg parameters
        params = {
            "input_video": video_path,
            "extract_frames": True,
            "extracted_frames_folder": output_directory,
            "extract_frame_pattern": frame_pattern,
            "extract_frame_step": frame_step,
        }

        # Convert times to HH:MM:SS format if provided
        start_time_str = None
        end_time_str = None

        if start_time is not None:
            start_time_str = seconds_to_hhmmss(start_time, precision="high")
            params["start_time"] = start_time_str

        if end_time is not None:
            end_time_str = seconds_to_hhmmss(end_time, precision="high")
            params["end_time"] = end_time_str

        # Build ffmpeg command
        ffmpeg_command = create_ffmpeg_command(params)
        self.logger.debug(f"FFmpeg command: {ffmpeg_command}")

        # Report progress
        if progress_callback:
            progress_callback(20, "Extracting frames...")

        # Execute ffmpeg
        try:
            result = subprocess.run(
                ffmpeg_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            self.logger.info("Frame extraction complete")

            # Report progress
            if progress_callback:
                progress_callback(80, "Frame extraction complete")

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else "Unknown error"
            self.logger.error(f"FFmpeg extraction failed: {error_msg}")
            raise RuntimeError(f"FFmpeg frame extraction failed: {error_msg}")

        # Report progress
        if progress_callback:
            progress_callback(90, "Collecting extracted frames...")

        # Get list of extracted frames
        frame_glob_pattern = frame_pattern.replace('%05d', '*').replace('%04d', '*').replace('%03d', '*')
        frame_search_path = os.path.join(output_directory, frame_glob_pattern)
        frame_files = sorted(glob.glob(frame_search_path))

        if len(frame_files) == 0:
            raise RuntimeError(f"No frames extracted. Check ffmpeg output and video file.")

        # Limit to max_frames if specified
        if max_frames is not None and len(frame_files) > max_frames:
            self.logger.warning(
                f"Extracted {len(frame_files)} frames, limiting to {max_frames}"
            )
            frame_files = frame_files[:max_frames]

        self.logger.info(
            f"Extracted {len(frame_files)} frames to {output_directory}"
        )

        # Calculate timestep
        timestep_ms = (1000.0 / frame_rate) * frame_step

        # Build extraction metadata
        extraction_metadata = {
            "num_frames": len(frame_files),
            "frame_rate": frame_rate,
            "frame_step": frame_step,
            "timestep_ms": timestep_ms,
            "start_time": start_time_str or "00:00:00",
            "end_time": end_time_str or "",
            "output_directory": output_directory,
        }

        # Report completion
        if progress_callback:
            progress_callback(100, f"Extracted {len(frame_files)} frames")

        return frame_files, extraction_metadata
