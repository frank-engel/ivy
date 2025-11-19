"""Shared pytest fixtures for IVyTools tests."""

import pytest
from pathlib import Path


@pytest.fixture
def test_data_dir():
    """Return path to test data directory.

    Returns:
        Path to the img_seq test data directory
    """
    return Path(__file__).parent / "img_seq"


@pytest.fixture
def test_welton_data_dir():
    """Return path to Welton test data directory.

    Returns:
        Path to the img_seq_welton_main_drain test data directory
    """
    return Path(__file__).parent / "img_seq_welton_main_drain"


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs.

    Args:
        tmp_path: pytest's tmp_path fixture

    Returns:
        Path to temporary output directory
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
