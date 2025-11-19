"""Integration tests for BatchOrchestrator service.

These tests validate the complete batch processing workflow:
- Loading scaffold configuration into ScaffoldConfig dataclass
- Creating VideoConfig from batch parameters
- Processing single video through complete workflow
- Processing multiple videos in batch mode
- Validating discharge results against expected values

Tests use real data from examples/ directory.
"""

import os
import tempfile
import shutil
import pytest
import csv
from pathlib import Path

from image_velocimetry_tools.batch import (
    BatchOrchestrator,
    ScaffoldConfig,
    VideoConfig,
    BatchVideoConfig,
)
from image_velocimetry_tools.services.project_service import ProjectService


# Find repository root (where examples/ directory is located)
def get_repo_root():
    """Find the repository root directory."""
    current_file = Path(__file__).resolve()
    # Go up from tests/test_integration/ to repo root
    repo_root = current_file.parent.parent.parent.parent
    return repo_root


def check_ffmpeg_available():
    """Check if ffmpeg and ffprobe are available.

    Checks for custom IVyTools environment variables first, then falls back
    to PATH lookup.
    """
    # Check for custom IVyTools environment variables
    ffmpeg_custom = os.environ.get("FFMPEG-IVyTools")
    ffprobe_custom = os.environ.get("FFPROBE-IVyTools")

    if ffmpeg_custom and ffprobe_custom:
        # Check if custom paths exist
        if os.path.exists(ffmpeg_custom) and os.path.exists(ffprobe_custom):
            return True

    # Fall back to PATH lookup
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


REPO_ROOT = get_repo_root()
EXAMPLES_DIR = REPO_ROOT / "examples"
VIDEOS_DIR = EXAMPLES_DIR / "videos"
BATCH_CSV_PATH = EXAMPLES_DIR / "batch_boneyard.csv"
SCAFFOLD_PATH = EXAMPLES_DIR / "scaffold_project.ivy"
FFMPEG_AVAILABLE = check_ffmpeg_available()


class TestBatchOrchestrator:
    """Integration tests for BatchOrchestrator."""

    @pytest.fixture
    def project_service(self):
        """Create ProjectService instance."""
        return ProjectService()

    @pytest.fixture
    def orchestrator(self):
        """Create BatchOrchestrator instance."""
        return BatchOrchestrator()

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp(prefix="ivy_test_batch_")
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def scaffold_config(self, project_service):
        """Load scaffold configuration."""
        if not os.path.exists(str(SCAFFOLD_PATH)):
            pytest.skip(f"Scaffold not found: {SCAFFOLD_PATH}")

        # Load using ProjectService
        config_dict = project_service.load_scaffold_configuration(str(SCAFFOLD_PATH))

        # Convert to ScaffoldConfig dataclass
        scaffold_config = ScaffoldConfig(
            scaffold_path=str(SCAFFOLD_PATH),
            project_dict=config_dict["project_dict"],
            swap_directory=config_dict["swap_directory"],
            rectification_method=config_dict["rectification_method"],
            rectification_params=config_dict["rectification_params"],
            stiv_params=config_dict["stiv_params"],
            cross_section_data=config_dict["cross_section_data"],
            grid_params=config_dict["grid_params"],
            display_units=config_dict["display_units"],
            temp_cleanup_required=config_dict["temp_cleanup_required"],
        )

        yield scaffold_config

        # Cleanup scaffold temp directory
        if scaffold_config.temp_cleanup_required:
            temp_dir = os.path.dirname(scaffold_config.swap_directory)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    @pytest.fixture
    def batch_video_configs(self):
        """Load video configurations from batch CSV."""
        if not os.path.exists(str(BATCH_CSV_PATH)):
            pytest.skip(f"Batch CSV not found: {BATCH_CSV_PATH}")

        video_configs = []

        with open(str(BATCH_CSV_PATH), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Resolve video path
                video_path = str(VIDEOS_DIR / os.path.basename(row['video_path']))

                if not os.path.exists(video_path):
                    continue  # Skip missing videos

                # Parse times (convert HH:MM:SS to seconds if needed)
                start_time = float(row.get('start_time', 0))
                end_time = float(row.get('end_time', 0))

                config = VideoConfig(
                    video_path=video_path,
                    water_surface_elevation=float(row['water_surface_elevation']),
                    measurement_date=row['measurement_date'],
                    alpha=float(row.get('alpha', 0.85)),
                    start_time=start_time if start_time > 0 else None,
                    end_time=end_time if end_time > 0 else None,
                    comments=row.get('comments', ''),
                )
                video_configs.append(config)

        return video_configs

    def test_scaffold_config_dataclass(self, scaffold_config):
        """Test that ScaffoldConfig dataclass is correctly populated."""
        # Validate dataclass structure
        assert isinstance(scaffold_config, ScaffoldConfig)
        assert scaffold_config.rectification_method in ["homography", "camera matrix", "scale"]

        # Validate STIV params
        assert "phi_origin" in scaffold_config.stiv_params
        assert "phi_range" in scaffold_config.stiv_params
        assert "dphi" in scaffold_config.stiv_params
        assert "num_pixels" in scaffold_config.stiv_params

        # Validate cross-section data
        assert "line" in scaffold_config.cross_section_data
        assert scaffold_config.cross_section_data["line"] is not None

        print(f"\nScaffold config validated:")
        print(f"  Method: {scaffold_config.rectification_method}")
        print(f"  STIV params: {scaffold_config.stiv_params}")

    def test_video_config_dataclass(self, batch_video_configs):
        """Test that VideoConfig dataclass validates correctly."""
        if not batch_video_configs:
            pytest.skip("No video configs available")

        config = batch_video_configs[0]

        # Validate dataclass
        assert isinstance(config, VideoConfig)
        assert os.path.exists(config.video_path)
        assert config.water_surface_elevation > 0
        assert 0 < config.alpha <= 1

        print(f"\nVideo config validated:")
        print(f"  Path: {config.video_path}")
        print(f"  WSE: {config.water_surface_elevation} m")
        print(f"  Alpha: {config.alpha}")

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe not installed")
    @pytest.mark.skip(reason="Requires complete implementation - grid points not yet stored in scaffold")
    def test_process_single_video(
        self,
        orchestrator,
        scaffold_config,
        batch_video_configs,
        temp_output_dir
    ):
        """Test processing a single video through complete workflow.

        Note: This test is skipped until grid points are properly stored in scaffold.
        Currently the orchestrator expects grid_points in grid_params but they're
        not saved in the scaffold .ivy file yet.
        """
        if not batch_video_configs:
            pytest.skip("No video configs available")

        # Use first video
        video_config = batch_video_configs[0]
        video_name = Path(video_config.video_path).stem

        print(f"\n=== Processing {video_name} ===")

        # Create combined config
        combined_config = BatchVideoConfig(
            scaffold=scaffold_config,
            video=video_config,
        )

        # Track progress
        progress_updates = []

        def progress_callback(percent, message):
            progress_updates.append((percent, message))
            print(f"  {percent}%: {message}")

        # Process video
        result = orchestrator.process_video(
            config=combined_config,
            output_directory=temp_output_dir,
            progress_callback=progress_callback,
            cleanup_temp_files=False,  # Keep files for inspection
        )

        # Validate result
        print(f"\n=== Result ===")
        print(result)

        if not result.success:
            pytest.fail(f"Processing failed: {result.error_stage} - {result.error_message}")

        # Validate outputs exist
        assert result.num_frames_extracted > 0
        assert len(result.frame_files) > 0
        assert os.path.exists(result.frames_directory)

        assert len(result.rectified_frame_files) > 0
        assert os.path.exists(result.rectified_frames_directory)

        assert result.total_discharge > 0
        assert result.total_area > 0
        assert result.mean_velocity > 0

        assert os.path.exists(result.output_csv_path)

        # Validate against expected values from BATCH_VALIDATION.md
        # Expected for 120000 video: Q=3.1995 m³/s, A=4.6755 m², V=0.6843 m/s
        if "120000" in video_name:
            expected_q = 3.1995
            tolerance = 0.01  # 0.5% tolerance

            assert abs(result.total_discharge - expected_q) < tolerance, \
                f"Discharge mismatch: got {result.total_discharge:.4f}, expected {expected_q:.4f}"

        # Validate progress updates
        assert len(progress_updates) > 0
        assert progress_updates[-1][0] == 100  # Final progress should be 100%

        print(f"\n=== Validation complete ===")

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe not installed")
    @pytest.mark.skip(reason="Requires complete implementation - grid points not yet stored in scaffold")
    def test_process_batch(
        self,
        orchestrator,
        scaffold_config,
        batch_video_configs,
        temp_output_dir
    ):
        """Test processing multiple videos in batch mode.

        Note: This test is skipped until grid points are properly stored in scaffold.
        """
        if not batch_video_configs:
            pytest.skip("No video configs available")

        print(f"\n=== Processing batch of {len(batch_video_configs)} videos ===")

        # Track progress
        progress_updates = []

        def progress_callback(percent, message):
            progress_updates.append((percent, message))
            print(f"  {percent}%: {message}")

        # Process batch
        batch_result = orchestrator.process_batch(
            scaffold_config=scaffold_config,
            video_configs=batch_video_configs,
            output_directory=temp_output_dir,
            progress_callback=progress_callback,
            cleanup_temp_files=False,  # Keep files for inspection
        )

        # Validate batch result
        print(f"\n=== Batch Result ===")
        print(batch_result)

        assert batch_result.total_videos == len(batch_video_configs)
        assert batch_result.successful > 0
        assert len(batch_result.video_results) == batch_result.total_videos

        # Validate batch summary CSV
        assert os.path.exists(batch_result.batch_csv_path)

        # Validate discharge summary
        discharge_summary = batch_result.get_discharge_summary()
        print(f"\n=== Discharge Summary ===")
        print(f"  Count: {discharge_summary['count']}")
        print(f"  Mean: {discharge_summary['mean']:.4f} m³/s")
        print(f"  Min: {discharge_summary['min']:.4f} m³/s")
        print(f"  Max: {discharge_summary['max']:.4f} m³/s")
        print(f"  Std: {discharge_summary['std']:.4f} m³/s")

        assert discharge_summary['count'] > 0
        assert discharge_summary['mean'] > 0

        # Validate progress updates
        assert len(progress_updates) > 0
        assert progress_updates[-1][0] == 100

        print(f"\n=== Batch validation complete ===")

    def test_batch_result_methods(self):
        """Test BatchResult helper methods."""
        from image_velocimetry_tools.batch.config import BatchResult, ProcessingResult

        # Create mock results
        results = [
            ProcessingResult(video_path="video1.mp4", success=True, total_discharge=2.0),
            ProcessingResult(video_path="video2.mp4", success=True, total_discharge=3.0),
            ProcessingResult(video_path="video3.mp4", success=False, error_message="Test error"),
        ]

        batch_result = BatchResult(
            total_videos=3,
            successful=2,
            failed=1,
            video_results=results,
        )

        # Test get_successful_results
        successful = batch_result.get_successful_results()
        assert len(successful) == 2
        assert all(r.success for r in successful)

        # Test get_failed_results
        failed = batch_result.get_failed_results()
        assert len(failed) == 1
        assert not failed[0].success

        # Test get_discharge_summary
        summary = batch_result.get_discharge_summary()
        assert summary['count'] == 2
        assert summary['mean'] == 2.5
        assert summary['min'] == 2.0
        assert summary['max'] == 3.0

        print("\nBatchResult methods validated")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
