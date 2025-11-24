"""Tests for filesystem utilities."""

import tempfile
from pathlib import Path

import pytest

from image_velocimetry_tools.utils.filesystem import (
    make_safe_filename,
    ensure_directory,
    get_file_size,
)


class TestMakeSafeFilename:
    """Test make_safe_filename function."""

    def test_simple_filename_unchanged(self):
        """Test that simple valid filename is unchanged."""
        assert make_safe_filename("file.txt") == "file.txt"
        assert make_safe_filename("my_file_123.csv") == "my_file_123.csv"

    def test_removes_invalid_characters(self):
        """Test removal of characters invalid on Windows/Linux/macOS."""
        # Windows invalid: / \ : * ? " < > |
        assert make_safe_filename("file/name") == "file_name"
        assert make_safe_filename("file\\name") == "file_name"
        assert make_safe_filename("file:name") == "file_name"
        assert make_safe_filename("file*name") == "file_name"
        assert make_safe_filename("file?name") == "file_name"
        assert make_safe_filename('file"name') == "file_name"
        assert make_safe_filename("file<name") == "file_name"
        assert make_safe_filename("file>name") == "file_name"
        assert make_safe_filename("file|name") == "file_name"

    def test_replaces_spaces(self):
        """Test that spaces are replaced with underscores."""
        assert make_safe_filename("my file name.txt") == "my_file_name.txt"
        assert make_safe_filename("file   with   spaces") == "file_with_spaces"

    def test_custom_replacement(self):
        """Test custom replacement character."""
        assert make_safe_filename("file/name", replacement="-") == "file-name"
        assert make_safe_filename("my file", replacement="-") == "my-file"

    def test_removes_leading_trailing_dots_spaces(self):
        """Test removal of leading/trailing dots, spaces, underscores."""
        assert make_safe_filename("   file.txt   ") == "file.txt"
        assert make_safe_filename("...file.txt...") == "file.txt"
        assert make_safe_filename("___file___") == "file"
        assert make_safe_filename("  .  file  .  ") == "file"

    def test_collapses_multiple_replacements(self):
        """Test that multiple replacement characters are collapsed."""
        assert make_safe_filename("file///name") == "file_name"
        assert make_safe_filename("file:::name") == "file_name"
        assert make_safe_filename("file   name") == "file_name"

    def test_preserves_extension_when_truncating(self):
        """Test that extension is preserved when truncating long names."""
        long_name = "a" * 300 + ".txt"
        result = make_safe_filename(long_name, max_length=255)

        assert len(result) == 255
        assert result.endswith(".txt")
        assert result.startswith("a")

    def test_truncates_without_extension(self):
        """Test truncation of long names without extension."""
        long_name = "a" * 300
        result = make_safe_filename(long_name, max_length=255)

        assert len(result) == 255
        assert result == "a" * 255

    def test_handles_windows_reserved_names(self):
        """Test handling of Windows reserved names."""
        # Reserved names should be prefixed with underscore
        assert make_safe_filename("CON") == "_CON"
        assert make_safe_filename("PRN") == "_PRN"
        assert make_safe_filename("AUX") == "_AUX"
        assert make_safe_filename("NUL") == "_NUL"
        assert make_safe_filename("COM1") == "_COM1"
        assert make_safe_filename("LPT1") == "_LPT1"

        # Case insensitive
        assert make_safe_filename("con") == "_con"
        assert make_safe_filename("Con.txt") == "_Con.txt"

    def test_handles_windows_reserved_names_with_extension(self):
        """Test handling of Windows reserved names with extensions."""
        assert make_safe_filename("CON.txt") == "_CON.txt"
        assert make_safe_filename("prn.log") == "_prn.log"

    def test_empty_string_returns_default(self):
        """Test that empty string returns 'unnamed'."""
        assert make_safe_filename("") == "unnamed"
        assert make_safe_filename("   ") == "unnamed"
        assert make_safe_filename("...") == "unnamed"
        assert make_safe_filename("///") == "unnamed"

    def test_removes_control_characters(self):
        """Test removal of control characters."""
        # Test null byte
        assert make_safe_filename("file\x00name") == "file_name"

        # Test various control characters
        assert make_safe_filename("file\x01\x02\x03name") == "file_name"
        assert make_safe_filename("file\x1fname") == "file_name"
        assert make_safe_filename("file\x7fname") == "file_name"

    def test_complex_example(self):
        """Test complex real-world example."""
        input_str = "My:File?Name/with Spaces<and|bars>"
        expected = "My_File_Name_with_Spaces_and_bars"
        assert make_safe_filename(input_str) == expected

    def test_preserves_dots_in_filename(self):
        """Test that dots within filename are preserved."""
        assert make_safe_filename("file.name.with.dots.txt") == "file.name.with.dots.txt"

    def test_unicode_characters_preserved(self):
        """Test that valid Unicode characters are preserved."""
        # Most Unicode should be fine on modern filesystems
        assert make_safe_filename("文件名.txt") == "文件名.txt"
        assert make_safe_filename("файл.txt") == "файл.txt"


class TestEnsureDirectory:
    """Test ensure_directory function."""

    def test_creates_directory(self):
        """Test that directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "test_subdir"

            result = ensure_directory(test_dir)

            assert result == test_dir
            assert test_dir.exists()
            assert test_dir.is_dir()

    def test_creates_nested_directories(self):
        """Test that nested directories are created (like mkdir -p)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "level1" / "level2" / "level3"

            result = ensure_directory(test_dir)

            assert result == test_dir
            assert test_dir.exists()
            assert test_dir.is_dir()

    def test_does_not_error_if_exists(self):
        """Test that no error is raised if directory exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "existing"
            test_dir.mkdir()

            # Should not raise error
            result = ensure_directory(test_dir)

            assert result == test_dir
            assert test_dir.exists()

    def test_accepts_string_path(self):
        """Test that function accepts string paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir_str = str(Path(tmpdir) / "string_path")

            result = ensure_directory(test_dir_str)

            assert isinstance(result, Path)
            assert result.exists()
            assert result.is_dir()

    def test_returns_absolute_path(self):
        """Test that function returns absolute path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to temp directory
            import os
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                # Create with relative path
                result = ensure_directory("relative/path")

                # Should return absolute path
                assert result.is_absolute()
                assert result.exists()
            finally:
                os.chdir(original_cwd)


class TestGetFileSize:
    """Test get_file_size function."""

    def test_returns_file_size(self):
        """Test that file size is returned correctly."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # Write known amount of data
            data = b"A" * 1024  # 1 KB
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name

        try:
            size = get_file_size(tmp_path)
            assert size == 1024
        finally:
            Path(tmp_path).unlink()

    def test_accepts_string_path(self):
        """Test that function accepts string paths."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            data = b"test"
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name

        try:
            size = get_file_size(tmp_path)
            assert size == 4
        finally:
            Path(tmp_path).unlink()

    def test_accepts_path_object(self):
        """Test that function accepts Path objects."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            data = b"test"
            tmp.write(data)
            tmp.flush()
            tmp_path = Path(tmp.name)

        try:
            size = get_file_size(tmp_path)
            assert size == 4
        finally:
            tmp_path.unlink()

    def test_raises_error_for_nonexistent_file(self):
        """Test that FileNotFoundError is raised for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            get_file_size("/nonexistent/file.txt")

    def test_empty_file_returns_zero(self):
        """Test that empty file returns size 0."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            size = get_file_size(tmp_path)
            assert size == 0
        finally:
            Path(tmp_path).unlink()
