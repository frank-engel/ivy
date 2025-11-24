"""Cross-platform utilities for IVyTools.

This module provides platform-independent utilities for file operations,
path handling, and platform detection to ensure IVyTools runs correctly
on Windows, Linux, and macOS.
"""

from image_velocimetry_tools.utils.platform import (
    get_platform,
    is_windows,
    is_linux,
    is_mac,
    is_raspberry_pi,
    get_config_dir,
    get_cache_dir,
    get_data_dir,
)

from image_velocimetry_tools.utils.filesystem import (
    make_safe_filename,
    ensure_directory,
)

__all__ = [
    # Platform detection
    "get_platform",
    "is_windows",
    "is_linux",
    "is_mac",
    "is_raspberry_pi",
    # Platform-specific directories
    "get_config_dir",
    "get_cache_dir",
    "get_data_dir",
    # Filesystem utilities
    "make_safe_filename",
    "ensure_directory",
]
