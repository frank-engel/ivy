"""Tests for ScaffoldLoader service."""

import pytest
import tempfile
import shutil
import json
import zipfile
from pathlib import Path

from image_velocimetry_tools.services.scaffold_loader import ScaffoldLoader
from image_velocimetry_tools.batch.exceptions import InvalidScaffoldError


@pytest.fixture
def scaffold_loader():
    """Fixture providing a ScaffoldLoader instance."""
    return ScaffoldLoader()


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory that's cleaned up after use."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_project_data():
    """Fixture providing sample project_data.json content."""
    return {
        "rectification_parameters": {
            "method": "camera matrix",
            "ground_control_points": [
                [0, 0, 100],
                [1, 0, 100],
                [0, 1, 100],
                [1, 1, 100],
                [0.5, 0.5, 100],
                [1, 0.5, 100],
            ],
            "image_control_points": [
                [100, 100],
                [200, 100],
                [100, 200],
                [200, 200],
                [150, 150],
                [200, 150],
            ],
            "pixel_gsd": 0.01,
        },
        "cross_section_geometry_path": "5-discharge/cross_section.mat",
        "grid_parameters": {
            "use_cross_section_line": True,
            "cross_section_line_start": [100, 100],
            "cross_section_line_end": [500, 100],
            "num_points": 50,
            "mask_polygons": [],
        },
        "stiv_parameters": {
            "num_pixels": 20,
            "phi_origin": 90,
            "d_phi": 1.0,
            "phi_range": 90,
            "max_vel_threshold_mps": 10.0,
            "gaussian_blur_sigma": 0.5,
        },
        "ffmpeg_parameters": {
            "frame_rate": 10,
            "frame_step": 1,
        },
        "extraction_parameters": {
            "timestep_ms": 100,
        },
    }


@pytest.fixture
def sample_scaffold_zip(temp_dir, sample_project_data):
    """Fixture providing a valid scaffold .ivy file."""
    scaffold_path = Path(temp_dir) / "scaffold_test.ivy"

    # Create a temporary directory structure
    scaffold_content_dir = Path(temp_dir) / "scaffold_content"
    scaffold_content_dir.mkdir()

    # Create project_data.json
    project_json_path = scaffold_content_dir / "project_data.json"
    with open(project_json_path, "w") as f:
        json.dump(sample_project_data, f)

    # Create directory structure
    (scaffold_content_dir / "1-images").mkdir()
    (scaffold_content_dir / "2-orthorectification").mkdir()
    (scaffold_content_dir / "5-discharge").mkdir()

    # Create cross-section file
    xs_file = scaffold_content_dir / "5-discharge" / "cross_section.mat"
    xs_file.write_text("dummy mat file content")

    # Create calibration image
    calib_image = (
        scaffold_content_dir / "2-orthorectification" / "calibration.jpg"
    )
    calib_image.write_text("dummy image content")

    # Create ZIP archive
    with zipfile.ZipFile(scaffold_path, "w") as zipf:
        for file_path in scaffold_content_dir.rglob("*"):
            if file_path.is_file():
                arcname = str(file_path.relative_to(scaffold_content_dir))
                zipf.write(file_path, arcname=arcname)

    return str(scaffold_path)


class TestScaffoldLoaderLoadScaffold:
    """Tests for load_scaffold method."""

    def test_load_valid_scaffold(
        self, scaffold_loader, sample_scaffold_zip, temp_dir
    ):
        """Test loading a valid scaffold file."""
        extract_dir = Path(temp_dir) / "extract"

        result = scaffold_loader.load_scaffold(
            sample_scaffold_zip, temp_extract_dir=str(extract_dir)
        )

        assert "project_data" in result
        assert "extract_dir" in result
        assert "cross_section_path" in result
        assert "calibration_image_path" in result

        # Verify project_data loaded correctly
        assert (
            result["project_data"]["rectification_parameters"]["method"]
            == "camera matrix"
        )

        # Verify files exist
        assert Path(result["extract_dir"]).exists()
        assert Path(result["cross_section_path"]).exists()
        assert Path(result["calibration_image_path"]).exists()

    def test_load_scaffold_with_temp_dir_creation(
        self, scaffold_loader, sample_scaffold_zip
    ):
        """Test that scaffold loader creates temp directory if not specified."""
        result = scaffold_loader.load_scaffold(sample_scaffold_zip)

        assert Path(result["extract_dir"]).exists()
        assert (
            "scaffold_" in result["extract_dir"]
        )  # Check temp directory naming

        # Cleanup
        scaffold_loader.cleanup_scaffold(result["extract_dir"])

    def test_load_nonexistent_scaffold_raises_error(
        self, scaffold_loader, temp_dir
    ):
        """Test that loading non-existent scaffold raises InvalidScaffoldError."""
        scaffold_path = Path(temp_dir) / "does_not_exist.ivy"

        with pytest.raises(InvalidScaffoldError, match="does not exist"):
            scaffold_loader.load_scaffold(str(scaffold_path))

    def test_load_wrong_extension_raises_error(
        self, scaffold_loader, temp_dir
    ):
        """Test that wrong file extension raises InvalidScaffoldError."""
        wrong_file = Path(temp_dir) / "scaffold.txt"
        wrong_file.write_text("not a zip")

        with pytest.raises(InvalidScaffoldError, match=".ivy extension"):
            scaffold_loader.load_scaffold(str(wrong_file))

    def test_load_corrupted_zip_raises_error(self, scaffold_loader, temp_dir):
        """Test that corrupted ZIP file raises InvalidScaffoldError."""
        corrupted_file = Path(temp_dir) / "corrupted.ivy"
        corrupted_file.write_text("not a valid zip file")

        with pytest.raises(InvalidScaffoldError, match="Failed to extract"):
            scaffold_loader.load_scaffold(str(corrupted_file))

    def test_load_scaffold_missing_project_data_raises_error(
        self, scaffold_loader, temp_dir
    ):
        """Test that scaffold without project_data.json raises error."""
        scaffold_path = Path(temp_dir) / "no_project_data.ivy"

        # Create ZIP without project_data.json
        with zipfile.ZipFile(scaffold_path, "w") as zipf:
            zipf.writestr("dummy.txt", "dummy content")

        with pytest.raises(
            InvalidScaffoldError, match="missing project_data.json"
        ):
            scaffold_loader.load_scaffold(str(scaffold_path))

    def test_load_scaffold_invalid_json_raises_error(
        self, scaffold_loader, temp_dir
    ):
        """Test that invalid JSON in project_data.json raises error."""
        scaffold_path = Path(temp_dir) / "invalid_json.ivy"

        # Create ZIP with invalid JSON
        with zipfile.ZipFile(scaffold_path, "w") as zipf:
            zipf.writestr("project_data.json", "{invalid json")

        with pytest.raises(
            InvalidScaffoldError, match="Failed to load project_data.json"
        ):
            scaffold_loader.load_scaffold(str(scaffold_path))


class TestScaffoldLoaderValidateScaffold:
    """Tests for validate_scaffold_data method."""

    def test_validate_valid_scaffold(
        self, scaffold_loader, sample_project_data, temp_dir
    ):
        """Test validation of valid scaffold data."""
        # Create cross-section file that validation expects
        xs_dir = Path(temp_dir) / "5-discharge"
        xs_dir.mkdir(parents=True, exist_ok=True)
        xs_file = xs_dir / "cross_section.mat"
        xs_file.write_text("dummy cross-section data")

        errors = scaffold_loader.validate_scaffold_data(
            sample_project_data, temp_dir
        )

        assert errors == []

    def test_validate_missing_required_keys(self, scaffold_loader, temp_dir):
        """Test validation detects missing required keys."""
        incomplete_data = {
            "rectification_parameters": {
                "method": "camera matrix",
            }
            # Missing other required keys
        }

        errors = scaffold_loader.validate_scaffold_data(
            incomplete_data, temp_dir
        )

        assert len(errors) > 0
        assert any("Missing required key" in err for err in errors)

    def test_validate_wrong_rectification_method(
        self, scaffold_loader, sample_project_data, temp_dir
    ):
        """Test validation rejects non-camera-matrix methods."""
        sample_project_data["rectification_parameters"][
            "method"
        ] = "homography"

        errors = scaffold_loader.validate_scaffold_data(
            sample_project_data, temp_dir
        )

        assert len(errors) > 0
        assert any("camera matrix" in err for err in errors)

    def test_validate_insufficient_gcps(
        self, scaffold_loader, sample_project_data, temp_dir
    ):
        """Test validation detects insufficient GCPs for camera matrix."""
        # Camera matrix needs at least 6 GCPs
        sample_project_data["rectification_parameters"][
            "ground_control_points"
        ] = [
            [0, 0, 100],
            [1, 0, 100],
            [0, 1, 100],  # Only 3 GCPs
        ]
        sample_project_data["rectification_parameters"][
            "image_control_points"
        ] = [[100, 100], [200, 100], [100, 200]]

        errors = scaffold_loader.validate_scaffold_data(
            sample_project_data, temp_dir
        )

        assert len(errors) > 0
        assert any("at least 6 GCPs" in err for err in errors)

    def test_validate_mismatched_gcp_icp_count(
        self, scaffold_loader, sample_project_data, temp_dir
    ):
        """Test validation detects GCP/ICP count mismatch."""
        # Different number of GCPs and ICPs
        sample_project_data["rectification_parameters"][
            "image_control_points"
        ] = [
            [100, 100],
            [200, 100],  # Only 2 ICPs
        ]

        errors = scaffold_loader.validate_scaffold_data(
            sample_project_data, temp_dir
        )

        assert len(errors) > 0
        assert any("must match" in err for err in errors)

    def test_validate_missing_stiv_parameters(
        self, scaffold_loader, sample_project_data, temp_dir
    ):
        """Test validation detects missing STIV parameters."""
        del sample_project_data["stiv_parameters"]["num_pixels"]

        errors = scaffold_loader.validate_scaffold_data(
            sample_project_data, temp_dir
        )

        assert len(errors) > 0
        assert any("STIV parameter" in err for err in errors)

    def test_validate_grid_not_on_cross_section(
        self, scaffold_loader, sample_project_data, temp_dir
    ):
        """Test validation requires grid along cross-section."""
        sample_project_data["grid_parameters"][
            "use_cross_section_line"
        ] = False

        errors = scaffold_loader.validate_scaffold_data(
            sample_project_data, temp_dir
        )

        assert len(errors) > 0
        assert any("along cross-section" in err for err in errors)


class TestScaffoldLoaderFindFiles:
    """Tests for file finding methods."""

    def test_find_cross_section_file(self, scaffold_loader, temp_dir):
        """Test finding cross-section file in standard location."""
        extract_dir = Path(temp_dir) / "extract"
        extract_dir.mkdir()
        discharge_dir = extract_dir / "5-discharge"
        discharge_dir.mkdir()

        # Create cross-section file
        xs_file = discharge_dir / "Cross-Section_AC3.mat"
        xs_file.write_text("dummy")

        result = scaffold_loader._find_cross_section_file(str(extract_dir))

        assert result is not None
        assert "Cross-Section" in result
        assert result.endswith(".mat")

    def test_find_cross_section_file_not_found(
        self, scaffold_loader, temp_dir
    ):
        """Test that _find_cross_section_file returns None when not found."""
        extract_dir = Path(temp_dir) / "extract"
        extract_dir.mkdir()

        result = scaffold_loader._find_cross_section_file(str(extract_dir))

        assert result is None

    def test_find_calibration_image(self, scaffold_loader, temp_dir):
        """Test finding calibration image."""
        extract_dir = Path(temp_dir) / "extract"
        extract_dir.mkdir()
        ortho_dir = extract_dir / "2-orthorectification"
        ortho_dir.mkdir()

        # Create calibration image
        calib_image = ortho_dir / "calibration.jpg"
        calib_image.write_text("dummy")

        result = scaffold_loader._find_calibration_image(str(extract_dir))

        assert result is not None
        assert "calibration" in result
        assert result.endswith(".jpg")

    def test_find_calibration_image_not_found(self, scaffold_loader, temp_dir):
        """Test that _find_calibration_image returns None when not found."""
        extract_dir = Path(temp_dir) / "extract"
        extract_dir.mkdir()

        result = scaffold_loader._find_calibration_image(str(extract_dir))

        assert result is None


class TestScaffoldLoaderCleanup:
    """Tests for cleanup_scaffold method."""

    def test_cleanup_removes_directory(self, scaffold_loader, temp_dir):
        """Test that cleanup removes the extraction directory."""
        extract_dir = Path(temp_dir) / "to_cleanup"
        extract_dir.mkdir()
        (extract_dir / "file.txt").write_text("test")

        assert extract_dir.exists()

        scaffold_loader.cleanup_scaffold(str(extract_dir))

        assert not extract_dir.exists()

    def test_cleanup_nonexistent_directory_no_error(
        self, scaffold_loader, temp_dir
    ):
        """Test that cleanup of non-existent directory doesn't raise error."""
        extract_dir = Path(temp_dir) / "does_not_exist"

        # Should not raise error
        scaffold_loader.cleanup_scaffold(str(extract_dir))


class TestScaffoldLoaderGetScaffoldInfo:
    """Tests for get_scaffold_info method."""

    def test_get_info_valid_scaffold(
        self, scaffold_loader, sample_scaffold_zip
    ):
        """Test getting info from valid scaffold."""
        info = scaffold_loader.get_scaffold_info(sample_scaffold_zip)

        assert info["is_valid"] is True
        assert info["has_project_data"] is True
        assert info["has_cross_section"] is True
        assert info["file_size_mb"] > 0

    def test_get_info_incomplete_scaffold(self, scaffold_loader, temp_dir):
        """Test get_scaffold_info detects incomplete scaffold."""
        scaffold_path = Path(temp_dir) / "incomplete.ivy"

        # Create ZIP with only project_data.json (missing cross-section)
        with zipfile.ZipFile(scaffold_path, "w") as zipf:
            zipf.writestr("project_data.json", "{}")

        info = scaffold_loader.get_scaffold_info(str(scaffold_path))

        assert info["has_project_data"] is True
        assert info["has_cross_section"] is False
        assert info["is_valid"] is False

    def test_get_info_nonexistent_file_raises_error(
        self, scaffold_loader, temp_dir
    ):
        """Test get_scaffold_info raises error for non-existent file."""
        scaffold_path = Path(temp_dir) / "does_not_exist.ivy"

        with pytest.raises(InvalidScaffoldError, match="does not exist"):
            scaffold_loader.get_scaffold_info(str(scaffold_path))

    def test_get_info_corrupted_zip_raises_error(
        self, scaffold_loader, temp_dir
    ):
        """Test get_scaffold_info raises error for corrupted ZIP."""
        scaffold_path = Path(temp_dir) / "corrupted.ivy"
        scaffold_path.write_text("not a zip")

        with pytest.raises(InvalidScaffoldError, match="corrupted"):
            scaffold_loader.get_scaffold_info(str(scaffold_path))


class TestScaffoldLoaderConstants:
    """Tests for ScaffoldLoader constants."""

    def test_required_keys_constant(self, scaffold_loader):
        """Test that REQUIRED_KEYS constant is defined correctly."""
        assert "rectification_parameters" in scaffold_loader.REQUIRED_KEYS
        assert "cross_section_geometry_path" in scaffold_loader.REQUIRED_KEYS
        assert "grid_parameters" in scaffold_loader.REQUIRED_KEYS
        assert "stiv_parameters" in scaffold_loader.REQUIRED_KEYS
        assert "ffmpeg_parameters" in scaffold_loader.REQUIRED_KEYS

    def test_required_stiv_keys_constant(self, scaffold_loader):
        """Test that REQUIRED_STIV_KEYS constant is defined correctly."""
        assert "num_pixels" in scaffold_loader.REQUIRED_STIV_KEYS
        assert "phi_origin" in scaffold_loader.REQUIRED_STIV_KEYS
        assert "d_phi" in scaffold_loader.REQUIRED_STIV_KEYS
        assert "phi_range" in scaffold_loader.REQUIRED_STIV_KEYS
        assert "max_vel_threshold_mps" in scaffold_loader.REQUIRED_STIV_KEYS
