"""Cross-platform filesystem utilities.

This module provides utilities for file operations that work consistently
across Windows, Linux, and macOS.
"""

import re
import os
from pathlib import Path
from typing import Union


def make_safe_filename(
    input_string: str,
    replacement: str = "_",
    max_length: int = 255
) -> str:
    """Make a string safe for use as a filename on any platform.

    This function removes or replaces characters that are not allowed
    in filenames on Windows, Linux, or macOS. It handles:
    - Windows: / \\ : * ? " < > |
    - Linux: / (null byte)
    - macOS: / : (null byte)
    - Control characters (0x00-0x1F, 0x7F)
    - Leading/trailing dots and spaces

    Parameters
    ----------
    input_string : str
        The input string to sanitize
    replacement : str, default="_"
        Character to use as replacement for invalid characters
    max_length : int, default=255
        Maximum filename length (most filesystems support 255)

    Returns
    -------
    str
        Sanitized filename safe for use on any platform

    Example
    -------
    >>> make_safe_filename("My:File?Name/with Spaces<and|bars>")
    'My_File_Name_with_Spaces_and_bars_'

    >>> make_safe_filename("file.txt")
    'file.txt'

    >>> make_safe_filename("   .hidden   ")
    'hidden'

    Notes
    -----
    - Replaces spaces with underscores by default
    - Removes leading/trailing dots, spaces, and underscores
    - Truncates to max_length if necessary
    - Empty results are replaced with "unnamed"
    - Reserved Windows names (CON, PRN, AUX, etc.) are prefixed with underscore
    """
    if not input_string:
        return "unnamed"

    # Define characters not allowed on any major platform
    # Windows: / \ : * ? " < > |
    # Linux/macOS: / (and null byte)
    # We'll be conservative and disallow all of these
    disallowed_chars = r'[/\\:*?"<>|\x00-\x1F\x7F]'

    # Replace disallowed characters
    safe_string = re.sub(disallowed_chars, replacement, input_string)

    # Replace spaces with replacement character
    safe_string = safe_string.replace(" ", replacement)

    # Remove leading/trailing dots, spaces, and underscores
    safe_string = safe_string.strip(". _")

    # Collapse multiple replacement characters into one
    if replacement:
        pattern = re.escape(replacement) + "+"
        safe_string = re.sub(pattern, replacement, safe_string)

    # Truncate to max length
    if len(safe_string) > max_length:
        # Try to preserve extension if present
        parts = safe_string.rsplit(".", 1)
        if len(parts) == 2 and len(parts[1]) <= 10:  # Reasonable extension length
            ext = "." + parts[1]
            name = parts[0][:max_length - len(ext)]
            safe_string = name + ext
        else:
            safe_string = safe_string[:max_length]

    # Handle Windows reserved names (CON, PRN, AUX, NUL, COM1-9, LPT1-9)
    # Case-insensitive check
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    }

    # Check both with and without extension
    name_upper = safe_string.upper()
    name_without_ext = safe_string.rsplit(".", 1)[0].upper()

    if name_upper in reserved_names or name_without_ext in reserved_names:
        safe_string = "_" + safe_string

    # Final check - if empty after all sanitization, return default
    if not safe_string:
        return "unnamed"

    return safe_string


def ensure_directory(path: Union[str, Path], mode: int = 0o755) -> Path:
    """Create directory if it doesn't exist, with proper permissions.

    This function creates a directory and all necessary parent directories,
    similar to `mkdir -p` in Unix systems.

    Parameters
    ----------
    path : str or Path
        Path to directory to create
    mode : int, default=0o755
        Permission mode for created directories (Unix only)
        On Windows, this parameter is ignored

    Returns
    -------
    Path
        Absolute path to the created/existing directory

    Raises
    ------
    OSError
        If directory cannot be created due to permissions or other errors
    PermissionError
        If lacking write permission to create directory

    Example
    -------
    >>> from pathlib import Path
    >>> ensure_directory("/tmp/my_app/data")
    PosixPath('/tmp/my_app/data')

    >>> # Directory and all parents now exist
    >>> Path("/tmp/my_app/data").exists()
    True

    Notes
    -----
    - If directory already exists, no error is raised
    - On Windows, the mode parameter is ignored (Windows uses ACLs)
    - Parent directories are created with the same mode
    """
    path_obj = Path(path).resolve()

    # Create directory and parents if they don't exist
    path_obj.mkdir(parents=True, exist_ok=True, mode=mode)

    return path_obj


def get_file_size(path: Union[str, Path]) -> int:
    """Get file size in bytes.

    Parameters
    ----------
    path : str or Path
        Path to file

    Returns
    -------
    int
        File size in bytes

    Raises
    ------
    FileNotFoundError
        If file does not exist
    OSError
        If file cannot be accessed

    Example
    -------
    >>> size = get_file_size("video.mp4")
    >>> print(f"File size: {size / 1024 / 1024:.2f} MB")
    File size: 125.43 MB
    """
    path_obj = Path(path)
    return path_obj.stat().st_size
