"""Tests for VideoService."""

import pytest
import os
from pathlib import Path
from image_velocimetry_tools.services.video_service import VideoService


class TestVideoService:
    """Tests for VideoService business logic."""

    @pytest.fixture
    def service(self):
        """Create a VideoService instance."""
        return VideoService()

    # Clip Time Validation Tests
    def test_validate_clip_times_valid(self, service):
        """Test validation with valid clip times."""
        # Should not raise
        service.validate_clip_times(
            start_time_ms=0,
            end_time_ms=10000,
            video_duration_ms=30000
        )

    def test_validate_clip_times_start_negative(self, service):
        """Test validation rejects negative start time."""
        with pytest.raises(ValueError, match="Start time cannot be negative"):
            service.validate_clip_times(
                start_time_ms=-1000,
                end_time_ms=10000,
                video_duration_ms=30000
            )

    def test_validate_clip_times_end_before_start(self, service):
        """Test validation rejects end time before start time."""
        with pytest.raises(ValueError, match="End time must be greater than start time"):
            service.validate_clip_times(
                start_time_ms=10000,
                end_time_ms=5000,
                video_duration_ms=30000
            )

    def test_validate_clip_times_end_exceeds_duration(self, service):
        """Test validation rejects end time exceeding video duration."""
        with pytest.raises(ValueError, match="End time cannot exceed video duration"):
            service.validate_clip_times(
                start_time_ms=5000,
                end_time_ms=40000,
                video_duration_ms=30000
            )

    def test_validate_clip_times_start_equals_end(self, service):
        """Test validation rejects start time equal to end time."""
        with pytest.raises(ValueError, match="End time must be greater than start time"):
            service.validate_clip_times(
                start_time_ms=10000,
                end_time_ms=10000,
                video_duration_ms=30000
            )

    def test_validate_clip_times_zero_duration(self, service):
        """Test validation with zero duration video."""
        with pytest.raises(ValueError, match="Video duration must be positive"):
            service.validate_clip_times(
                start_time_ms=0,
                end_time_ms=0,
                video_duration_ms=0
            )

    # Output Filename Generation Tests
    def test_generate_clip_filename_basic(self, service):
        """Test basic clip filename generation."""
        filename = service.generate_clip_filename(
            input_video_path="/path/to/video.mp4",
            start_time_ms=5000,
            end_time_ms=15000,
            output_dir="/output",
            rotation=0,
            flip="none",
            normalize_luma=False,
            curve_preset="none",
            stabilize=False
        )

        assert filename.startswith("/output")
        assert "video" in filename
        assert "s5.000s" in filename
        assert "e15.000s" in filename
        assert filename.endswith(".mp4")

    def test_generate_clip_filename_with_rotation(self, service):
        """Test clip filename includes rotation."""
        filename = service.generate_clip_filename(
            input_video_path="/path/to/video.mp4",
            start_time_ms=5000,
            end_time_ms=15000,
            output_dir="/output",
            rotation=90,
            flip="none",
            normalize_luma=False,
            curve_preset="none",
            stabilize=False
        )

        assert "rot90" in filename

    def test_generate_clip_filename_with_flip(self, service):
        """Test clip filename includes flip."""
        filename = service.generate_clip_filename(
            input_video_path="/path/to/video.mp4",
            start_time_ms=5000,
            end_time_ms=15000,
            output_dir="/output",
            rotation=0,
            flip="vflip",
            normalize_luma=False,
            curve_preset="none",
            stabilize=False
        )

        assert "vflip" in filename

    def test_generate_clip_filename_with_normalize_luma(self, service):
        """Test clip filename includes normalize luma indicator."""
        filename = service.generate_clip_filename(
            input_video_path="/path/to/video.mp4",
            start_time_ms=5000,
            end_time_ms=15000,
            output_dir="/output",
            rotation=0,
            flip="none",
            normalize_luma=True,
            curve_preset="none",
            stabilize=False
        )

        assert "normluma" in filename

    def test_generate_clip_filename_with_curve_preset(self, service):
        """Test clip filename includes curve preset."""
        filename = service.generate_clip_filename(
            input_video_path="/path/to/video.mp4",
            start_time_ms=5000,
            end_time_ms=15000,
            output_dir="/output",
            rotation=0,
            flip="none",
            normalize_luma=False,
            curve_preset="lighter",
            stabilize=False
        )

        assert "lighter" in filename

    def test_generate_clip_filename_with_stabilization(self, service):
        """Test clip filename includes stabilization indicator."""
        filename = service.generate_clip_filename(
            input_video_path="/path/to/video.mp4",
            start_time_ms=5000,
            end_time_ms=15000,
            output_dir="/output",
            rotation=0,
            flip="none",
            normalize_luma=False,
            curve_preset="none",
            stabilize=True
        )

        assert "stab" in filename

    def test_generate_clip_filename_all_options(self, service):
        """Test clip filename with all options enabled."""
        filename = service.generate_clip_filename(
            input_video_path="/path/to/video.mp4",
            start_time_ms=5000,
            end_time_ms=15000,
            output_dir="/output",
            rotation=180,
            flip="hflip",
            normalize_luma=True,
            curve_preset="darker",
            stabilize=True
        )

        assert "rot180" in filename
        assert "hflip" in filename
        assert "normluma" in filename
        assert "darker" in filename
        assert "stab" in filename

    def test_generate_clip_filename_preserves_extension(self, service):
        """Test that output filename preserves video extension."""
        filename = service.generate_clip_filename(
            input_video_path="/path/to/video.avi",
            start_time_ms=5000,
            end_time_ms=15000,
            output_dir="/output",
            rotation=0,
            flip="none",
            normalize_luma=False,
            curve_preset="none",
            stabilize=False
        )

        assert filename.endswith(".avi")

    def test_generate_clip_filename_no_extension(self, service):
        """Test filename generation when input has no extension."""
        filename = service.generate_clip_filename(
            input_video_path="/path/to/video",
            start_time_ms=5000,
            end_time_ms=15000,
            output_dir="/output",
            rotation=0,
            flip="none",
            normalize_luma=False,
            curve_preset="none",
            stabilize=False
        )

        # Should default to .mp4
        assert filename.endswith(".mp4")

    # FFmpeg Parameters Building Tests
    def test_build_ffmpeg_parameters_basic(self, service):
        """Test basic FFmpeg parameter dict building."""
        params = service.build_ffmpeg_parameters(
            input_video="/path/to/input.mp4",
            output_video="/path/to/output.mp4",
            start_time_ms=5000,
            end_time_ms=15000,
            rotation=0,
            flip="none",
            strip_audio=False,
            normalize_luma=False,
            curve_preset="none",
            stabilize=False,
            extract_frames=False,
            extract_frame_step=1,
            extracted_frames_folder="/frames",
            calibrate_radial=False,
            cx=960.0,
            cy=540.0,
            k1=0.0,
            k2=0.0
        )

        assert params["input_video"] == "/path/to/input.mp4"
        assert params["output_video"] == "/path/to/output.mp4"
        assert "start_time" in params
        assert "end_time" in params
        assert params["video_rotation"] == 0
        assert params["video_flip"] == "none"
        assert params["strip_audio"] is False
        assert params["normalize_luma"] is False
        assert params["curve_preset"] == "none"
        assert params["stabilize"] is False
        assert params["extract_frames"] is False

    def test_build_ffmpeg_parameters_time_conversion(self, service):
        """Test that times are converted to HH:MM:SS format."""
        params = service.build_ffmpeg_parameters(
            input_video="/path/to/input.mp4",
            output_video="/path/to/output.mp4",
            start_time_ms=5000,  # 5 seconds
            end_time_ms=65000,   # 1 minute 5 seconds
            rotation=0,
            flip="none",
            strip_audio=False,
            normalize_luma=False,
            curve_preset="none",
            stabilize=False,
            extract_frames=False,
            extract_frame_step=1,
            extracted_frames_folder="/frames",
            calibrate_radial=False,
            cx=960.0,
            cy=540.0,
            k1=0.0,
            k2=0.0
        )

        # Times should be in HH:MM:SS.mmm format
        assert "00:00:05" in params["start_time"]
        assert "00:01:05" in params["end_time"]

    def test_build_ffmpeg_parameters_with_all_options(self, service):
        """Test FFmpeg parameters with all options enabled."""
        params = service.build_ffmpeg_parameters(
            input_video="/path/to/input.mp4",
            output_video="/path/to/output.mp4",
            start_time_ms=5000,
            end_time_ms=15000,
            rotation=90,
            flip="vflip",
            strip_audio=True,
            normalize_luma=True,
            curve_preset="lighter",
            stabilize=True,
            extract_frames=True,
            extract_frame_step=5,
            extracted_frames_folder="/frames",
            calibrate_radial=True,
            cx=960.0,
            cy=540.0,
            k1=-0.1,
            k2=0.05
        )

        assert params["video_rotation"] == 90
        assert params["video_flip"] == "vflip"
        assert params["strip_audio"] is True
        assert params["normalize_luma"] is True
        assert params["curve_preset"] == "lighter"
        assert params["stabilize"] is True
        assert params["extract_frames"] is True
        assert params["extract_frame_step"] == 5
        assert params["calibrate_radial"] is True
        assert params["k1"] == -0.1
        assert params["k2"] == 0.05

    def test_build_ffmpeg_parameters_radial_correction(self, service):
        """Test FFmpeg parameters include lens correction values."""
        params = service.build_ffmpeg_parameters(
            input_video="/path/to/input.mp4",
            output_video="/path/to/output.mp4",
            start_time_ms=5000,
            end_time_ms=15000,
            rotation=0,
            flip="none",
            strip_audio=False,
            normalize_luma=False,
            curve_preset="none",
            stabilize=False,
            extract_frames=False,
            extract_frame_step=1,
            extracted_frames_folder="/frames",
            calibrate_radial=True,
            cx=1024.5,
            cy=768.3,
            k1=-0.15,
            k2=0.08
        )

        assert params["calibrate_radial"] is True
        assert params["cx"] == 1024.5
        assert params["cy"] == 768.3
        assert params["k1"] == -0.15
        assert params["k2"] == 0.08
