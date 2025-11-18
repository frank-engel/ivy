"""Tests for ProjectService."""

import os
import json
import zipfile
import tempfile
import shutil
import pytest
from pathlib import Path
import platform

from image_velocimetry_tools.services.project_service import ProjectService


@pytest.fixture
def project_service():
    """Fixture providing a ProjectService instance."""
    return ProjectService()


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory that's cleaned up after use."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_project_dict():
    """Fixture providing a sample project dictionary."""
    return {
        "project_name": "Test Project",
        "version": "1.0.0",
        "video_file_name": "/path/to/video.mp4",
        "video_clip_start_time": 0,
        "video_clip_end_time": 5000,
        "is_video_loaded": True,
        "ffmpeg_parameters": {
            "rotation": 0,
            "flip": "none",
            "stabilize": False
        }
    }


class TestProjectServiceSaveJSON:
    """Tests for save_project_to_json method."""

    def test_save_valid_project_dict(self, project_service, temp_dir, sample_project_dict):
        """Test saving a valid project dictionary to JSON."""
        json_path = os.path.join(temp_dir, "test_project.json")

        result = project_service.save_project_to_json(sample_project_dict, json_path)

        assert result is True
        assert os.path.exists(json_path)

        # Verify content
        with open(json_path, "r") as f:
            loaded_data = json.load(f)

        assert loaded_data == sample_project_dict

    def test_save_creates_parent_directory(self, project_service, temp_dir, sample_project_dict):
        """Test that save creates parent directory if it doesn't exist."""
        json_path = os.path.join(temp_dir, "subdir", "nested", "project.json")

        result = project_service.save_project_to_json(sample_project_dict, json_path)

        assert result is True
        assert os.path.exists(json_path)

    def test_save_with_invalid_dict_raises_error(self, project_service, temp_dir):
        """Test that saving non-dict raises ValueError."""
        json_path = os.path.join(temp_dir, "test.json")

        with pytest.raises(ValueError, match="project_dict must be a dictionary"):
            project_service.save_project_to_json("not a dict", json_path)

    def test_save_with_empty_path_raises_error(self, project_service, sample_project_dict):
        """Test that empty path raises ValueError."""
        with pytest.raises(ValueError, match="json_path cannot be empty"):
            project_service.save_project_to_json(sample_project_dict, "")

    def test_save_with_invalid_path_raises_error(self, project_service, sample_project_dict):
        """Test that invalid path raises IOError on all platforms."""

        # Windows: use a protected directory
        if platform.system() == "Windows":
            invalid_path = r"C:\Windows\System32\__cannot_write_here.json"

        # Unix: use a root-protected directory -- left in here as one day we may want to run IVyTools in linux
        else:
            # skip if running as root
            if hasattr(os, "getuid") and os.getuid() == 0:
                pytest.skip("Skipping permission test when running as root")
            invalid_path = "/root/cannot_write_here/project.json"

        with pytest.raises(IOError):
            project_service.save_project_to_json(sample_project_dict, invalid_path)


class TestProjectServiceLoadJSON:
    """Tests for load_project_from_json method."""

    def test_load_valid_project(self, project_service, temp_dir, sample_project_dict):
        """Test loading a valid project JSON file."""
        json_path = os.path.join(temp_dir, "test_project.json")

        # Create a JSON file first
        with open(json_path, "w") as f:
            json.dump(sample_project_dict, f)

        loaded_dict = project_service.load_project_from_json(json_path)

        assert loaded_dict == sample_project_dict

    def test_load_nonexistent_file_raises_error(self, project_service, temp_dir):
        """Test that loading non-existent file raises FileNotFoundError."""
        json_path = os.path.join(temp_dir, "does_not_exist.json")

        with pytest.raises(FileNotFoundError):
            project_service.load_project_from_json(json_path)

    def test_load_invalid_json_raises_error(self, project_service, temp_dir):
        """Test that loading invalid JSON raises ValueError."""
        json_path = os.path.join(temp_dir, "invalid.json")

        # Create invalid JSON file
        with open(json_path, "w") as f:
            f.write("{invalid json content")

        with pytest.raises(ValueError, match="invalid JSON"):
            project_service.load_project_from_json(json_path)

    def test_load_non_dict_json_raises_error(self, project_service, temp_dir):
        """Test that loading JSON array raises ValueError."""
        json_path = os.path.join(temp_dir, "array.json")

        # Create JSON array instead of object
        with open(json_path, "w") as f:
            json.dump([1, 2, 3], f)

        with pytest.raises(ValueError, match="must contain a dictionary"):
            project_service.load_project_from_json(json_path)


class TestProjectServiceCreateArchive:
    """Tests for create_project_archive method."""

    def test_create_archive_basic(self, project_service, temp_dir):
        """Test creating a basic project archive."""
        # Create source directory with some files
        source_dir = os.path.join(temp_dir, "source")
        os.makedirs(source_dir)

        # Create test files
        test_files = ["file1.txt", "file2.json", "subdir/file3.txt"]
        for file_path in test_files:
            full_path = os.path.join(source_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(f"Content of {file_path}")

        zip_path = os.path.join(temp_dir, "test_project.zip")

        result = project_service.create_project_archive(source_dir, zip_path)

        assert result is True
        assert os.path.exists(zip_path)

        # Verify ZIP contents
        with zipfile.ZipFile(zip_path, "r") as zipf:
            names = zipf.namelist()
            assert "file1.txt" in names
            assert "file2.json" in names
            assert "subdir/file3.txt" in names or "subdir\\file3.txt" in names

    def test_create_archive_with_exclusions(self, project_service, temp_dir):
        """Test creating archive with excluded file extensions."""
        source_dir = os.path.join(temp_dir, "source")
        os.makedirs(source_dir)

        # Create files with different extensions
        with open(os.path.join(source_dir, "keep.txt"), "w") as f:
            f.write("keep this")
        with open(os.path.join(source_dir, "exclude.dat"), "w") as f:
            f.write("exclude this")

        zip_path = os.path.join(temp_dir, "test.zip")

        result = project_service.create_project_archive(
            source_dir, zip_path, exclude_extensions=[".dat"]
        )

        assert result is True

        # Verify only .txt file is in archive
        with zipfile.ZipFile(zip_path, "r") as zipf:
            names = zipf.namelist()
            assert "keep.txt" in names
            assert "exclude.dat" not in names

    def test_create_archive_with_progress_callback(self, project_service, temp_dir):
        """Test that progress callback is called during archive creation."""
        source_dir = os.path.join(temp_dir, "source")
        os.makedirs(source_dir)

        # Create a few files
        for i in range(5):
            with open(os.path.join(source_dir, f"file{i}.txt"), "w") as f:
                f.write(f"File {i}")

        zip_path = os.path.join(temp_dir, "test.zip")

        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        project_service.create_project_archive(
            source_dir, zip_path, progress_callback=progress_callback
        )

        # Verify progress callback was called
        assert len(progress_calls) > 0
        # Last call should be (5, 5)
        assert progress_calls[-1] == (5, 5)

    def test_create_archive_nonexistent_source_raises_error(self, project_service, temp_dir):
        """Test that non-existent source directory raises FileNotFoundError."""
        source_dir = os.path.join(temp_dir, "does_not_exist")
        zip_path = os.path.join(temp_dir, "test.zip")

        with pytest.raises(FileNotFoundError):
            project_service.create_project_archive(source_dir, zip_path)

    def test_create_archive_empty_path_raises_error(self, project_service, temp_dir):
        """Test that empty output path raises ValueError."""
        source_dir = temp_dir

        with pytest.raises(ValueError, match="output_zip_path cannot be empty"):
            project_service.create_project_archive(source_dir, "")


class TestProjectServiceExtractArchive:
    """Tests for extract_project_archive method."""

    def test_extract_valid_archive(self, project_service, temp_dir):
        """Test extracting a valid project archive."""
        # Create a test archive
        source_dir = os.path.join(temp_dir, "source")
        os.makedirs(source_dir)

        test_content = "test content"
        with open(os.path.join(source_dir, "test.txt"), "w") as f:
            f.write(test_content)

        zip_path = os.path.join(temp_dir, "test.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(
                os.path.join(source_dir, "test.txt"),
                arcname="test.txt"
            )

        # Extract to new directory
        extract_dir = os.path.join(temp_dir, "extracted")

        result = project_service.extract_project_archive(zip_path, extract_dir)

        assert result is True
        assert os.path.exists(os.path.join(extract_dir, "test.txt"))

        # Verify content
        with open(os.path.join(extract_dir, "test.txt"), "r") as f:
            assert f.read() == test_content

    def test_extract_nonexistent_archive_raises_error(self, project_service, temp_dir):
        """Test that non-existent archive raises FileNotFoundError."""
        zip_path = os.path.join(temp_dir, "does_not_exist.zip")
        extract_dir = os.path.join(temp_dir, "extracted")

        with pytest.raises(FileNotFoundError):
            project_service.extract_project_archive(zip_path, extract_dir)

    def test_extract_corrupted_archive_raises_error(self, project_service, temp_dir):
        """Test that corrupted archive raises BadZipFile."""
        # Create a corrupted ZIP file
        zip_path = os.path.join(temp_dir, "corrupted.zip")
        with open(zip_path, "w") as f:
            f.write("This is not a valid ZIP file")

        extract_dir = os.path.join(temp_dir, "extracted")

        with pytest.raises(zipfile.BadZipFile):
            project_service.extract_project_archive(zip_path, extract_dir)

    def test_extract_empty_directory_raises_error(self, project_service, temp_dir):
        """Test that empty extraction directory raises ValueError."""
        zip_path = os.path.join(temp_dir, "test.zip")

        with pytest.raises(ValueError, match="extract_to_directory cannot be empty"):
            project_service.extract_project_archive(zip_path, "")


class TestProjectServiceValidation:
    """Tests for validate_project_dict method."""

    def test_validate_valid_project_dict(self, project_service, sample_project_dict):
        """Test validating a valid project dictionary."""
        errors = project_service.validate_project_dict(sample_project_dict)

        assert errors == []

    def test_validate_non_dict_fails(self, project_service):
        """Test that non-dictionary input fails validation."""
        errors = project_service.validate_project_dict("not a dict")

        assert len(errors) > 0
        assert any("must be a dictionary" in error for error in errors)

    def test_validate_empty_dict_succeeds(self, project_service):
        """Test that empty dictionary passes basic validation."""
        # Note: Empty dict is technically valid, specific fields validated elsewhere
        errors = project_service.validate_project_dict({})

        assert errors == []
