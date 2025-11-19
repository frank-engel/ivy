"""Batch processing module for IVyTools.

This module provides batch processing capabilities for processing multiple
videos using a scaffold template configuration.

Main components:
- config: Dataclasses for batch configuration and results
- orchestrator: BatchOrchestrator service for workflow management
"""

from image_velocimetry_tools.batch.config import (
    ScaffoldConfig,
    VideoConfig,
    BatchVideoConfig,
    ProcessingResult,
    BatchResult,
)
from image_velocimetry_tools.batch.orchestrator import BatchOrchestrator

__all__ = [
    "ScaffoldConfig",
    "VideoConfig",
    "BatchVideoConfig",
    "ProcessingResult",
    "BatchResult",
    "BatchOrchestrator",
]
