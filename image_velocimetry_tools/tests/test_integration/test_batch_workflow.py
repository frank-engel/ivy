"""Integration tests for complete batch processing workflow.

These tests validate that all Phase 2 services work together correctly:
- ProjectService loads scaffold configuration
- VideoService extracts frames from video
- Services can be chained together

This serves as a smoke test before creating the BatchOrchestrator.
"""

import os
import tempfile
import shutil
import pytest
from pathlib import Path

from image_velocimetry_tools.services.project_service import ProjectService
from image_velocimetry_tools.services.video_service import VideoService


# Find repository root (where examples/ directory is located)
def get_repo_root():
    """Find the repository root directory."""
    current_file = Path(__file__).resolve()
    # Go up from tests/test_integration/ to repo root
    repo_root = current_file.parent.parent.parent.parent
    return repo_root


def check_ffmpeg_available():
    """Check if ffmpeg and ffprobe are available."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


REPO_ROOT = get_repo_root()
EXAMPLES_DIR = REPO_ROOT / "examples"
VIDEOS_DIR = EXAMPLES_DIR / "videos"
FFMPEG_AVAILABLE = check_ffmpeg_available()


class TestBatchWorkflow:
    """Integration tests for batch processing workflow."""

    @pytest.fixture
    def project_service(self):
        """Create ProjectService instance."""
        return ProjectService()

    @pytest.fixture
    def video_service(self):
        """Create VideoService instance."""
        return VideoService()

    @pytest.fixture
    def scaffold_path(self):
        """Path to test scaffold project."""
        return str(EXAMPLES_DIR / "scaffold_project.ivy")

    @pytest.fixture
    def test_video_path(self):
        """Path to test video file."""
        return str(VIDEOS_DIR / "03337000_bullet_20170630-120000.mp4")

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp(prefix="ivy_test_workflow_")
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe not installed")
    def test_load_scaffold_and_extract_frames(
        self,
        project_service,
        video_service,
        scaffold_path,
        test_video_path,
        temp_dir
    ):
        """Test loading scaffold and extracting frames from video."""
        # Skip if files don't exist
        if not os.path.exists(scaffold_path):
            pytest.skip(f"Scaffold not found: {scaffold_path}")
        if not os.path.exists(test_video_path):
            pytest.skip(f"Test video not found: {test_video_path}")

        # Step 1: Load scaffold configuration
        print("\n=== Step 1: Loading scaffold configuration ===")
        scaffold_config = project_service.load_scaffold_configuration(scaffold_path)

        assert "stiv_params" in scaffold_config
        assert "rectification_method" in scaffold_config
        print(f"Scaffold loaded: {scaffold_config['rectification_method']} rectification")

        # Step 2: Get video metadata
        print("\n=== Step 2: Getting video metadata ===")
        video_metadata = video_service.get_video_metadata(test_video_path)

        assert video_metadata["width"] > 0
        assert video_metadata["avg_frame_rate"] > 0
        print(f"Video: {video_metadata['width']}x{video_metadata['height']}, "
              f"{video_metadata['avg_frame_rate']:.2f} fps")

        # Step 3: Extract frames from video (15-20 second window, matching batch config)
        print("\n=== Step 3: Extracting frames ===")
        frames_dir = os.path.join(temp_dir, "frames")
        frames, frame_metadata = video_service.extract_frames(
            video_path=test_video_path,
            output_directory=frames_dir,
            start_time=15,
            end_time=20,
            frame_step=1,
            max_frames=200
        )

        assert len(frames) > 0
        assert all(os.path.exists(f) for f in frames)
        print(f"Extracted {len(frames)} frames")

        # Step 4: Validate frame extraction metadata matches scaffold expectations
        print("\n=== Step 4: Validating compatibility ===")

        # Calculate expected timestep based on video frame rate and frame step
        expected_timestep_ms = (1000.0 / video_metadata["avg_frame_rate"]) * frame_metadata["frame_step"]
        actual_timestep_ms = frame_metadata["timestep_ms"]

        assert abs(actual_timestep_ms - expected_timestep_ms) < 0.1
        print(f"Timestep: {actual_timestep_ms:.3f} ms")

        # Cleanup scaffold temp directory
        if scaffold_config["temp_cleanup_required"]:
            scaffold_temp = os.path.dirname(scaffold_config["swap_directory"])
            if os.path.exists(scaffold_temp):
                shutil.rmtree(scaffold_temp)

        print("\n=== Workflow test complete ===")

    def test_scaffold_parameters_are_usable(self, project_service, scaffold_path):
        """Test that scaffold parameters are in the correct format for services."""
        # Skip if scaffold doesn't exist
        if not os.path.exists(scaffold_path):
            pytest.skip(f"Scaffold not found: {scaffold_path}")

        # Load scaffold
        config = project_service.load_scaffold_configuration(scaffold_path)

        # Validate STIV params are ready to use
        stiv_params = config["stiv_params"]
        required_stiv_keys = ["phi_origin", "phi_range", "dphi", "num_pixels"]
        for key in required_stiv_keys:
            assert key in stiv_params, f"Missing STIV parameter: {key}"
            assert isinstance(stiv_params[key], (int, float)), f"{key} must be numeric"

        # Validate rectification params exist
        assert config["rectification_params"] is not None
        assert isinstance(config["rectification_params"], dict)

        # Validate cross-section data
        xs_data = config["cross_section_data"]
        assert "line" in xs_data
        assert xs_data["line"] is not None

        # Cleanup
        if config["temp_cleanup_required"]:
            temp_dir = os.path.dirname(config["swap_directory"])
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        print("\nScaffold parameters validated - ready for batch processing")

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe not installed")
    def test_video_metadata_sufficient_for_stiv(
        self,
        video_service,
        test_video_path
    ):
        """Test that video metadata contains everything needed for STIV."""
        # Skip if video doesn't exist
        if not os.path.exists(test_video_path):
            pytest.skip(f"Test video not found: {test_video_path}")

        # Get metadata
        metadata = video_service.get_video_metadata(test_video_path)

        # STIV needs frame rate to calculate timestep
        assert "avg_frame_rate" in metadata
        assert metadata["avg_frame_rate"] > 0

        # Calculate timestep (needed for STIV velocity calculation)
        frame_rate = metadata["avg_frame_rate"]
        frame_step = 1  # Example
        timestep_s = frame_step / frame_rate

        assert timestep_s > 0
        print(f"\nTimestep for STIV: {timestep_s:.4f} seconds")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
