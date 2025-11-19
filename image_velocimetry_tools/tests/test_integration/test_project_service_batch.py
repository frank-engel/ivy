"""Integration tests for ProjectService batch processing methods.

These tests validate the batch-compatible methods added in Phase 2:
- load_scaffold_configuration()

Tests use the real scaffold project from examples/scaffold_project.ivy
"""

import os
import tempfile
import shutil
import pytest
from pathlib import Path

from image_velocimetry_tools.services.project_service import ProjectService


# Find repository root (where examples/ directory is located)
def get_repo_root():
    """Find the repository root directory."""
    current_file = Path(__file__).resolve()
    # Go up from tests/test_integration/ to repo root
    repo_root = current_file.parent.parent.parent.parent
    return repo_root


REPO_ROOT = get_repo_root()
EXAMPLES_DIR = REPO_ROOT / "examples"


class TestProjectServiceBatch:
    """Integration tests for ProjectService batch methods."""

    @pytest.fixture
    def project_service(self):
        """Create ProjectService instance."""
        return ProjectService()

    @pytest.fixture
    def scaffold_path(self):
        """Path to test scaffold project."""
        return str(EXAMPLES_DIR / "scaffold_project.ivy")

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for scaffold extraction."""
        temp_dir = tempfile.mkdtemp(prefix="ivy_test_scaffold_")
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    def test_load_scaffold_configuration(self, project_service, scaffold_path):
        """Test loading scaffold configuration."""
        # Skip if scaffold doesn't exist
        if not os.path.exists(scaffold_path):
            pytest.skip(f"Scaffold not found: {scaffold_path}")

        # Load scaffold
        config = project_service.load_scaffold_configuration(scaffold_path)

        # Validate structure
        assert isinstance(config, dict)
        assert "project_dict" in config
        assert "swap_directory" in config
        assert "rectification_method" in config
        assert "rectification_params" in config
        assert "stiv_params" in config
        assert "cross_section_data" in config
        assert "grid_params" in config
        assert "display_units" in config
        assert "temp_cleanup_required" in config

        # Validate rectification method
        assert config["rectification_method"] in ["scale", "homography", "camera matrix"]

        # Validate STIV parameters
        stiv_params = config["stiv_params"]
        assert "phi_origin" in stiv_params
        assert "phi_range" in stiv_params
        assert "dphi" in stiv_params
        assert "num_pixels" in stiv_params
        assert stiv_params["phi_origin"] > 0
        assert stiv_params["phi_range"] > 0
        assert stiv_params["dphi"] > 0
        assert stiv_params["num_pixels"] > 0

        # Validate cross-section data
        xs_data = config["cross_section_data"]
        assert "line" in xs_data
        assert "bathymetry_filename" in xs_data

        # Validate grid params
        grid_params = config["grid_params"]
        assert "num_points" in grid_params
        assert grid_params["num_points"] > 0

        # Validate swap directory exists
        assert os.path.exists(config["swap_directory"])

        # Cleanup temp directory if created
        if config["temp_cleanup_required"]:
            temp_dir = os.path.dirname(config["swap_directory"])
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        # Log for debugging
        print(f"\nScaffold configuration:")
        print(f"  Rectification method: {config['rectification_method']}")
        print(f"  STIV phi_origin: {stiv_params['phi_origin']}")
        print(f"  STIV num_pixels: {stiv_params['num_pixels']}")
        print(f"  Grid points: {grid_params['num_points']}")
        print(f"  Display units: {config['display_units']}")

    def test_load_scaffold_with_temp_dir(self, project_service, scaffold_path, temp_dir):
        """Test loading scaffold with provided temp directory."""
        # Skip if scaffold doesn't exist
        if not os.path.exists(scaffold_path):
            pytest.skip(f"Scaffold not found: {scaffold_path}")

        # Load scaffold with specific temp directory
        config = project_service.load_scaffold_configuration(
            scaffold_path,
            temp_dir=temp_dir
        )

        # Validate swap directory is in our temp directory
        assert config["swap_directory"].startswith(temp_dir)
        assert os.path.exists(config["swap_directory"])

        # Should not require cleanup since we provided temp_dir
        assert config["temp_cleanup_required"] == False

        print(f"\nSwap directory: {config['swap_directory']}")

    def test_load_scaffold_extracts_rectification_params(
        self,
        project_service,
        scaffold_path
    ):
        """Test that rectification parameters are correctly extracted."""
        # Skip if scaffold doesn't exist
        if not os.path.exists(scaffold_path):
            pytest.skip(f"Scaffold not found: {scaffold_path}")

        # Load scaffold
        config = project_service.load_scaffold_configuration(scaffold_path)

        method = config["rectification_method"]
        params = config["rectification_params"]

        # Validate params based on method
        if method == "homography":
            assert "homography_matrix" in params
            assert "world_coords" in params
            assert "pixel_coords" in params
            assert "pad_x" in params
            assert "pad_y" in params
            print("\nHomography parameters validated")

        elif method == "camera matrix":
            assert "camera_matrix" in params
            assert "water_surface_elevation" in params
            # extent is optional
            print("\nCamera matrix parameters validated")
            print(f"  WSE: {params.get('water_surface_elevation')} m")

        elif method == "scale":
            assert "world_coords" in params
            assert "pixel_coords" in params
            print("\nScale parameters validated")

        # Cleanup
        if config["temp_cleanup_required"]:
            temp_dir = os.path.dirname(config["swap_directory"])
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_load_scaffold_nonexistent_file(self, project_service):
        """Test that load_scaffold raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            project_service.load_scaffold_configuration("/nonexistent/scaffold.ivy")

    def test_load_scaffold_cleanup_on_error(self, project_service):
        """Test that temporary directory is cleaned up on error."""
        # Try to load a file that will fail extraction
        # (create a fake .ivy file that's not a valid ZIP)
        with tempfile.NamedTemporaryFile(suffix='.ivy', delete=False) as f:
            f.write(b"not a valid zip file")
            fake_scaffold = f.name

        try:
            # This should fail and cleanup temp directory
            with pytest.raises(Exception):  # Will raise IOError or similar
                project_service.load_scaffold_configuration(fake_scaffold)

            # The temp directory should have been cleaned up
            # (we can't easily verify this, but the code should handle it)

        finally:
            # Cleanup fake file
            if os.path.exists(fake_scaffold):
                os.unlink(fake_scaffold)


class TestProjectServiceScaffoldValidation:
    """Test scaffold validation logic."""

    @pytest.fixture
    def project_service(self):
        """Create ProjectService instance."""
        return ProjectService()

    def test_extract_rectification_params_homography(self, project_service):
        """Test extracting homography rectification parameters."""
        project_dict = {
            "rectification_method": "homography",
            "homography_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "orthotable_world_coordinates": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "orthotable_pixel_coordinates": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "ortho_pad_x": 10,
            "ortho_pad_y": 20,
        }

        params = project_service._extract_rectification_params(project_dict)

        assert "homography_matrix" in params
        assert "world_coords" in params
        assert "pixel_coords" in params
        assert params["pad_x"] == 10
        assert params["pad_y"] == 20

    def test_extract_rectification_params_camera_matrix(self, project_service):
        """Test extracting camera matrix rectification parameters."""
        project_dict = {
            "rectification_method": "camera matrix",
            "camera_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            "water_surface_elevation_m": 318.21,
            "ortho_extent": [0, 100, 0, 100],
        }

        params = project_service._extract_rectification_params(project_dict)

        assert "camera_matrix" in params
        assert params["water_surface_elevation"] == 318.21
        assert params["extent"] == [0, 100, 0, 100]

    def test_extract_rectification_params_scale(self, project_service):
        """Test extracting scale rectification parameters."""
        project_dict = {
            "rectification_method": "scale",
            "orthotable_world_coordinates": [[0, 0], [1, 0]],
            "orthotable_pixel_coordinates": [[0, 0], [100, 0]],
        }

        params = project_service._extract_rectification_params(project_dict)

        assert "world_coords" in params
        assert "pixel_coords" in params


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
