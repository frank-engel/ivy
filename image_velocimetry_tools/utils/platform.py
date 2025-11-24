"""Platform detection and platform-specific utilities.

This module provides functions to detect the current platform and return
platform-appropriate paths for configuration, cache, and data directories.

Example
-------
>>> from image_velocimetry_tools.utils.platform import is_linux, get_config_dir
>>> if is_linux():
...     config_dir = get_config_dir()
...     print(f"Config directory: {config_dir}")
Config directory: /home/user/.config/ivytools
"""

import os
import sys
import platform
from pathlib import Path
from typing import Optional


def get_platform() -> str:
    """Return current platform identifier.

    Returns
    -------
    str
        Platform identifier: 'windows', 'linux', or 'darwin' (macOS)

    Example
    -------
    >>> platform = get_platform()
    >>> print(f"Running on: {platform}")
    Running on: linux
    """
    return sys.platform


def is_windows() -> bool:
    """Check if running on Windows.

    Returns
    -------
    bool
        True if running on Windows, False otherwise

    Example
    -------
    >>> if is_windows():
    ...     print("Running on Windows")
    """
    return sys.platform == "win32"


def is_linux() -> bool:
    """Check if running on Linux.

    Returns
    -------
    bool
        True if running on Linux, False otherwise

    Example
    -------
    >>> if is_linux():
    ...     print("Running on Linux")
    """
    return sys.platform.startswith("linux")


def is_mac() -> bool:
    """Check if running on macOS.

    Returns
    -------
    bool
        True if running on macOS, False otherwise

    Example
    -------
    >>> if is_mac():
    ...     print("Running on macOS")
    """
    return sys.platform == "darwin"


def is_raspberry_pi() -> bool:
    """Detect if running on Raspberry Pi hardware.

    This function checks for the Raspberry Pi device tree model file
    that is present on Raspberry Pi OS.

    Returns
    -------
    bool
        True if running on Raspberry Pi, False otherwise

    Example
    -------
    >>> if is_raspberry_pi():
    ...     print("Running on Raspberry Pi - using optimized settings")

    Notes
    -----
    This detection method works on Raspberry Pi OS (formerly Raspbian)
    but may not work on other Linux distributions installed on Raspberry Pi.
    """
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().lower()
            return "raspberry pi" in model
    except (FileNotFoundError, PermissionError, OSError):
        return False


def get_config_dir(app_name: str = "ivytools") -> Path:
    """Get platform-appropriate configuration directory.

    This follows platform conventions:
    - Windows: %APPDATA%/IVyTools
    - Linux: ~/.config/ivytools (XDG Base Directory spec)
    - macOS: ~/Library/Application Support/ivytools

    Parameters
    ----------
    app_name : str, default="ivytools"
        Application name for directory naming

    Returns
    -------
    Path
        Absolute path to configuration directory

    Example
    -------
    >>> config_dir = get_config_dir()
    >>> config_file = config_dir / "settings.json"
    >>> print(config_file)
    /home/user/.config/ivytools/settings.json

    Notes
    -----
    The directory is not created by this function. Use `ensure_directory()`
    to create it if needed.

    Environment variable IVYTOOLS_CONFIG_DIR can override the default location.
    """
    # Check for environment variable override
    env_override = os.environ.get("IVYTOOLS_CONFIG_DIR")
    if env_override:
        return Path(env_override)

    if is_windows():
        # Windows: %APPDATA%/IVyTools
        base = Path(os.environ.get("APPDATA", Path.home()))
        return base / "IVyTools"
    elif is_mac():
        # macOS: ~/Library/Application Support/ivytools
        return Path.home() / "Library" / "Application Support" / app_name
    else:
        # Linux: ~/.config/ivytools (XDG Base Directory spec)
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config:
            base = Path(xdg_config)
        else:
            base = Path.home() / ".config"
        return base / app_name


def get_cache_dir(app_name: str = "ivytools") -> Path:
    """Get platform-appropriate cache directory.

    This follows platform conventions:
    - Windows: %LOCALAPPDATA%/IVyTools/cache
    - Linux: ~/.cache/ivytools (XDG Base Directory spec)
    - macOS: ~/Library/Caches/ivytools

    Parameters
    ----------
    app_name : str, default="ivytools"
        Application name for directory naming

    Returns
    -------
    Path
        Absolute path to cache directory

    Example
    -------
    >>> cache_dir = get_cache_dir()
    >>> temp_file = cache_dir / "processed_frame.jpg"

    Notes
    -----
    Cache directory is for temporary files that can be safely deleted.
    The directory is not created by this function.

    Environment variable IVYTOOLS_CACHE_DIR can override the default location.
    """
    # Check for environment variable override
    env_override = os.environ.get("IVYTOOLS_CACHE_DIR")
    if env_override:
        return Path(env_override)

    if is_windows():
        # Windows: %LOCALAPPDATA%/IVyTools/cache
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        return base / "IVyTools" / "cache"
    elif is_mac():
        # macOS: ~/Library/Caches/ivytools
        return Path.home() / "Library" / "Caches" / app_name
    else:
        # Linux: ~/.cache/ivytools (XDG Base Directory spec)
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            base = Path(xdg_cache)
        else:
            base = Path.home() / ".cache"
        return base / app_name


def get_data_dir(app_name: str = "ivytools") -> Path:
    """Get platform-appropriate data directory.

    This follows platform conventions:
    - Windows: %APPDATA%/IVyTools/data
    - Linux: ~/.local/share/ivytools (XDG Base Directory spec)
    - macOS: ~/Library/Application Support/ivytools/data

    Parameters
    ----------
    app_name : str, default="ivytools"
        Application name for directory naming

    Returns
    -------
    Path
        Absolute path to data directory

    Example
    -------
    >>> data_dir = get_data_dir()
    >>> results_file = data_dir / "results.csv"

    Notes
    -----
    Data directory is for persistent application data.
    The directory is not created by this function.

    Environment variable IVYTOOLS_DATA_DIR can override the default location.
    """
    # Check for environment variable override
    env_override = os.environ.get("IVYTOOLS_DATA_DIR")
    if env_override:
        return Path(env_override)

    if is_windows():
        # Windows: %APPDATA%/IVyTools/data
        base = Path(os.environ.get("APPDATA", Path.home()))
        return base / "IVyTools" / "data"
    elif is_mac():
        # macOS: ~/Library/Application Support/ivytools/data
        return Path.home() / "Library" / "Application Support" / app_name / "data"
    else:
        # Linux: ~/.local/share/ivytools (XDG Base Directory spec)
        xdg_data = os.environ.get("XDG_DATA_HOME")
        if xdg_data:
            base = Path(xdg_data)
        else:
            base = Path.home() / ".local" / "share"
        return base / app_name


def get_executable_extension() -> str:
    """Get platform-specific executable extension.

    Returns
    -------
    str
        '.exe' on Windows, empty string on other platforms

    Example
    -------
    >>> exe_ext = get_executable_extension()
    >>> ffmpeg_name = f"ffmpeg{exe_ext}"
    >>> print(ffmpeg_name)
    ffmpeg.exe  # on Windows
    ffmpeg      # on Linux/macOS
    """
    return ".exe" if is_windows() else ""
