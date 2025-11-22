"""Public API for IVyTools batch processing.

This module provides a simple, user-friendly API for running batch
image velocimetry analysis on multiple videos.
"""

from .batch_api import (
    run_batch_processing,
    BatchResults,
    JobResult,
)

__all__ = [
    'run_batch_processing',
    'BatchResults',
    'JobResult',
]
