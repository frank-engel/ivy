"""Integration tests for VideoService batch processing methods.

These tests validate the batch-compatible methods added in Phase 2:
- get_video_metadata()
- extract_frames()

Tests use real video files from examples/videos/
"""

import os
import tempfile
import shutil
import pytest
from pathlib import Path

from image_velocimetry_tools.services.video_service import VideoService


# Find repository root (where examples/ directory is located)
def get_repo_root():
    """Find the repository root directory."""
    current_file = Path(__file__).resolve()
    # Go up from tests/test_integration/ to repo root
    repo_root = current_file.parent.parent.parent.parent
    return repo_root


REPO_ROOT = get_repo_root()
EXAMPLES_DIR = REPO_ROOT / "examples"
VIDEOS_DIR = EXAMPLES_DIR / "videos"


class TestVideoServiceBatch:
    """Integration tests for VideoService batch methods."""

    @pytest.fixture
    def video_service(self):
        """Create VideoService instance."""
        return VideoService()

    @pytest.fixture
    def test_video_path(self):
        """Path to test video file."""
        # Use the shortest video for faster tests
        return str(VIDEOS_DIR / "03337000_bullet_20170630-120000.mp4")

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp(prefix="ivy_test_video_")
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    def test_get_video_metadata(self, video_service, test_video_path):
        """Test getting video metadata with ffprobe."""
        # Skip if test video doesn't exist
        if not os.path.exists(test_video_path):
            pytest.skip(f"Test video not found: {test_video_path}")

        # Get metadata
        metadata = video_service.get_video_metadata(test_video_path)

        # Validate structure
        assert isinstance(metadata, dict)
        assert "width" in metadata
        assert "height" in metadata
        assert "avg_frame_rate" in metadata
        assert "duration" in metadata
        assert "frame_count" in metadata

        # Validate values are reasonable
        assert metadata["width"] > 0
        assert metadata["height"] > 0
        assert metadata["avg_frame_rate"] > 0
        assert metadata["duration"] > 0

        # Log for debugging
        print(f"\nVideo metadata: {metadata}")

    def test_get_video_metadata_nonexistent_file(self, video_service):
        """Test that get_video_metadata raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            video_service.get_video_metadata("/nonexistent/video.mp4")

    def test_extract_frames_basic(self, video_service, test_video_path, temp_output_dir):
        """Test basic frame extraction."""
        # Skip if test video doesn't exist
        if not os.path.exists(test_video_path):
            pytest.skip(f"Test video not found: {test_video_path}")

        # Extract frames from first 2 seconds
        frames, metadata = video_service.extract_frames(
            video_path=test_video_path,
            output_directory=temp_output_dir,
            start_time=0,
            end_time=2,
            frame_step=1,
            max_frames=60  # Limit to 60 frames for speed
        )

        # Validate frames were extracted
        assert len(frames) > 0
        assert all(os.path.exists(f) for f in frames)
        assert all(f.endswith('.jpg') for f in frames)

        # Validate metadata
        assert isinstance(metadata, dict)
        assert metadata["num_frames"] == len(frames)
        assert metadata["frame_rate"] > 0
        assert metadata["frame_step"] == 1
        assert metadata["timestep_ms"] > 0
        assert metadata["output_directory"] == temp_output_dir

        # Log for debugging
        print(f"\nExtracted {len(frames)} frames")
        print(f"Metadata: {metadata}")

    def test_extract_frames_with_time_window(self, video_service, test_video_path, temp_output_dir):
        """Test frame extraction with specific time window."""
        # Skip if test video doesn't exist
        if not os.path.exists(test_video_path):
            pytest.skip(f"Test video not found: {test_video_path}")

        # Extract frames from 15-20 seconds (same as batch config)
        frames, metadata = video_service.extract_frames(
            video_path=test_video_path,
            output_directory=temp_output_dir,
            start_time=15,
            end_time=20,
            frame_step=1,
            max_frames=200
        )

        # Validate frames were extracted
        assert len(frames) > 0
        assert len(frames) <= 200

        # Validate time window in metadata
        assert "15:00" in metadata["start_time"] or "00:15" in metadata["start_time"]
        assert "20:00" in metadata["end_time"] or "00:20" in metadata["end_time"]

        print(f"\nExtracted {len(frames)} frames from 15-20s window")

    def test_extract_frames_with_frame_step(self, video_service, test_video_path, temp_output_dir):
        """Test frame extraction with frame stepping."""
        # Skip if test video doesn't exist
        if not os.path.exists(test_video_path):
            pytest.skip(f"Test video not found: {test_video_path}")

        # Extract every 5th frame from first 2 seconds
        frames, metadata = video_service.extract_frames(
            video_path=test_video_path,
            output_directory=temp_output_dir,
            start_time=0,
            end_time=2,
            frame_step=5
        )

        # Validate frames were extracted with correct step
        assert len(frames) > 0
        assert metadata["frame_step"] == 5
        assert metadata["timestep_ms"] > 0

        # Timestep should be 5x the base frame interval
        expected_base_interval = 1000.0 / metadata["frame_rate"]
        expected_timestep = expected_base_interval * 5
        assert abs(metadata["timestep_ms"] - expected_timestep) < 0.1

        print(f"\nExtracted {len(frames)} frames with step=5")

    def test_extract_frames_with_progress_callback(
        self,
        video_service,
        test_video_path,
        temp_output_dir
    ):
        """Test frame extraction with progress callback."""
        # Skip if test video doesn't exist
        if not os.path.exists(test_video_path):
            pytest.skip(f"Test video not found: {test_video_path}")

        # Track progress updates
        progress_updates = []

        def progress_callback(percent, message):
            progress_updates.append((percent, message))
            print(f"  {percent}%: {message}")

        # Extract frames with progress tracking
        frames, metadata = video_service.extract_frames(
            video_path=test_video_path,
            output_directory=temp_output_dir,
            start_time=0,
            end_time=1,
            frame_step=1,
            max_frames=30,
            progress_callback=progress_callback
        )

        # Validate progress updates were received
        assert len(progress_updates) > 0

        # Validate progress values are in range 0-100
        assert all(0 <= p[0] <= 100 for p in progress_updates)

        # Validate final progress is 100%
        assert progress_updates[-1][0] == 100

        print(f"\nReceived {len(progress_updates)} progress updates")

    def test_extract_frames_invalid_inputs(self, video_service, temp_output_dir):
        """Test that extract_frames validates inputs correctly."""
        # Nonexistent video
        with pytest.raises(FileNotFoundError):
            video_service.extract_frames(
                "/nonexistent/video.mp4",
                temp_output_dir
            )

        # Invalid frame step
        test_video = str(VIDEOS_DIR / "03337000_bullet_20170630-120000.mp4")
        if os.path.exists(test_video):
            with pytest.raises(ValueError):
                video_service.extract_frames(
                    test_video,
                    temp_output_dir,
                    frame_step=0  # Invalid
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
