"""Tests for platform detection and platform-specific utilities."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from image_velocimetry_tools.utils.platform import (
    get_platform,
    is_windows,
    is_linux,
    is_mac,
    is_raspberry_pi,
    get_config_dir,
    get_cache_dir,
    get_data_dir,
    get_executable_extension,
)


class TestPlatformDetection:
    """Test platform detection functions."""

    def test_get_platform_returns_string(self):
        """Test that get_platform returns a string."""
        platform = get_platform()
        assert isinstance(platform, str)
        assert platform in ("win32", "linux", "darwin", "linux2")

    def test_platform_checks_are_exclusive(self):
        """Test that only one platform check returns True."""
        checks = [is_windows(), is_linux(), is_mac()]
        assert sum(checks) == 1, "Exactly one platform should be detected"

    def test_is_windows_on_win32(self):
        """Test is_windows returns True on Windows."""
        with patch.object(sys, "platform", "win32"):
            assert is_windows() is True
            assert is_linux() is False
            assert is_mac() is False

    def test_is_linux_on_linux(self):
        """Test is_linux returns True on Linux."""
        with patch.object(sys, "platform", "linux"):
            assert is_linux() is True
            assert is_windows() is False
            assert is_mac() is False

    def test_is_linux_on_linux2(self):
        """Test is_linux returns True on linux2."""
        with patch.object(sys, "platform", "linux2"):
            assert is_linux() is True

    def test_is_mac_on_darwin(self):
        """Test is_mac returns True on macOS."""
        with patch.object(sys, "platform", "darwin"):
            assert is_mac() is True
            assert is_windows() is False
            assert is_linux() is False


class TestRaspberryPiDetection:
    """Test Raspberry Pi hardware detection."""

    def test_is_raspberry_pi_with_model_file(self):
        """Test Raspberry Pi detection with valid model file."""
        model_content = "Raspberry Pi 4 Model B Rev 1.4\x00"

        with patch("builtins.open", mock_open(read_data=model_content)):
            assert is_raspberry_pi() is True

    def test_is_raspberry_pi_without_model_file(self):
        """Test Raspberry Pi detection when file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert is_raspberry_pi() is False

    def test_is_raspberry_pi_with_permission_error(self):
        """Test Raspberry Pi detection with permission error."""
        with patch("builtins.open", side_effect=PermissionError):
            assert is_raspberry_pi() is False

    def test_is_raspberry_pi_non_rpi_model(self):
        """Test Raspberry Pi detection with non-RPi hardware."""
        model_content = "Generic x86_64 Computer\x00"

        with patch("builtins.open", mock_open(read_data=model_content)):
            assert is_raspberry_pi() is False


class TestPlatformDirectories:
    """Test platform-specific directory functions."""

    def test_get_config_dir_on_windows(self):
        """Test config directory on Windows."""
        with patch.object(sys, "platform", "win32"):
            with patch.dict(os.environ, {"APPDATA": "C:\\Users\\Test\\AppData\\Roaming"}):
                config_dir = get_config_dir()
                # Compare path parts to avoid separator issues when testing on different platforms
                assert str(config_dir).replace("\\", "/") == "C:/Users/Test/AppData/Roaming/IVyTools"

    def test_get_config_dir_on_linux(self):
        """Test config directory on Linux (XDG spec)."""
        with patch.object(sys, "platform", "linux"):
            with patch.dict(os.environ, {}, clear=True):
                # Mock Path.home() to return a predictable value
                with patch("pathlib.Path.home", return_value=Path("/home/testuser")):
                    config_dir = get_config_dir()
                    assert config_dir == Path("/home/testuser/.config/ivytools")

    def test_get_config_dir_on_linux_with_xdg(self):
        """Test config directory on Linux with XDG_CONFIG_HOME set."""
        with patch.object(sys, "platform", "linux"):
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": "/custom/config"}):
                config_dir = get_config_dir()
                assert config_dir == Path("/custom/config/ivytools")

    def test_get_config_dir_on_mac(self):
        """Test config directory on macOS."""
        with patch.object(sys, "platform", "darwin"):
            with patch("pathlib.Path.home", return_value=Path("/Users/testuser")):
                config_dir = get_config_dir()
                assert config_dir == Path("/Users/testuser/Library/Application Support/ivytools")

    def test_get_config_dir_with_env_override(self):
        """Test config directory with environment variable override."""
        with patch.dict(os.environ, {"IVYTOOLS_CONFIG_DIR": "/custom/path"}):
            config_dir = get_config_dir()
            assert config_dir == Path("/custom/path")

    def test_get_cache_dir_on_windows(self):
        """Test cache directory on Windows."""
        with patch.object(sys, "platform", "win32"):
            with patch.dict(os.environ, {"LOCALAPPDATA": "C:\\Users\\Test\\AppData\\Local"}):
                cache_dir = get_cache_dir()
                # Compare path parts to avoid separator issues when testing on different platforms
                assert str(cache_dir).replace("\\", "/") == "C:/Users/Test/AppData/Local/IVyTools/cache"

    def test_get_cache_dir_on_linux(self):
        """Test cache directory on Linux (XDG spec)."""
        with patch.object(sys, "platform", "linux"):
            with patch.dict(os.environ, {}, clear=True):
                with patch("pathlib.Path.home", return_value=Path("/home/testuser")):
                    cache_dir = get_cache_dir()
                    assert cache_dir == Path("/home/testuser/.cache/ivytools")

    def test_get_cache_dir_with_env_override(self):
        """Test cache directory with environment variable override."""
        with patch.dict(os.environ, {"IVYTOOLS_CACHE_DIR": "/tmp/cache"}):
            cache_dir = get_cache_dir()
            assert cache_dir == Path("/tmp/cache")

    def test_get_data_dir_on_windows(self):
        """Test data directory on Windows."""
        with patch.object(sys, "platform", "win32"):
            with patch.dict(os.environ, {"APPDATA": "C:\\Users\\Test\\AppData\\Roaming"}):
                data_dir = get_data_dir()
                # Compare path parts to avoid separator issues when testing on different platforms
                assert str(data_dir).replace("\\", "/") == "C:/Users/Test/AppData/Roaming/IVyTools/data"

    def test_get_data_dir_on_linux(self):
        """Test data directory on Linux (XDG spec)."""
        with patch.object(sys, "platform", "linux"):
            with patch.dict(os.environ, {}, clear=True):
                with patch("pathlib.Path.home", return_value=Path("/home/testuser")):
                    data_dir = get_data_dir()
                    assert data_dir == Path("/home/testuser/.local/share/ivytools")

    def test_get_data_dir_with_env_override(self):
        """Test data directory with environment variable override."""
        with patch.dict(os.environ, {"IVYTOOLS_DATA_DIR": "/data/ivytools"}):
            data_dir = get_data_dir()
            assert data_dir == Path("/data/ivytools")

    def test_custom_app_name(self):
        """Test directory functions with custom app name."""
        with patch.object(sys, "platform", "linux"):
            with patch.dict(os.environ, {}, clear=True):
                with patch("pathlib.Path.home", return_value=Path("/home/testuser")):
                    config_dir = get_config_dir("custom_app")
                    assert config_dir == Path("/home/testuser/.config/custom_app")


class TestExecutableExtension:
    """Test executable extension function."""

    def test_get_executable_extension_on_windows(self):
        """Test executable extension on Windows."""
        with patch.object(sys, "platform", "win32"):
            assert get_executable_extension() == ".exe"

    def test_get_executable_extension_on_linux(self):
        """Test executable extension on Linux."""
        with patch.object(sys, "platform", "linux"):
            assert get_executable_extension() == ""

    def test_get_executable_extension_on_mac(self):
        """Test executable extension on macOS."""
        with patch.object(sys, "platform", "darwin"):
            assert get_executable_extension() == ""


class TestRealPlatform:
    """Test functions on actual platform (integration-style tests)."""

    def test_get_platform_is_valid(self):
        """Test that get_platform returns valid platform on real system."""
        platform = get_platform()
        assert platform in ("win32", "linux", "linux2", "darwin")

    def test_directory_functions_return_paths(self):
        """Test that directory functions return Path objects."""
        config_dir = get_config_dir()
        cache_dir = get_cache_dir()
        data_dir = get_data_dir()

        assert isinstance(config_dir, Path)
        assert isinstance(cache_dir, Path)
        assert isinstance(data_dir, Path)

    def test_directories_are_absolute(self):
        """Test that directory functions return absolute paths."""
        config_dir = get_config_dir()
        cache_dir = get_cache_dir()
        data_dir = get_data_dir()

        assert config_dir.is_absolute()
        assert cache_dir.is_absolute()
        assert data_dir.is_absolute()
