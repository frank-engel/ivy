# Batch Processing Module

This module provides batch processing capabilities for IVyTools, enabling automated processing of multiple river videos with shared configuration.

## Quick Start

### Python API

```python
from image_velocimetry_tools.api import process_video

result = process_video(
    scaffold_path="scaffold.ivy",
    video_path="river.mp4",
    water_surface_elevation=318.5,
    output_directory="results/",
)

print(f"Discharge: {result.total_discharge:.2f} m³/s")
```

### Command-Line Interface

```bash
ivytools-batch process scaffold.ivy river.mp4 --wse 318.5 --output results/
```

## Module Structure

- **`config.py`**: Configuration and result dataclasses
  - `ScaffoldConfig`: Template configuration from .ivy scaffold
  - `VideoConfig`: Per-video parameters
  - `ProcessingResult`: Single video results
  - `BatchResult`: Batch processing results

- **`orchestrator.py`**: Workflow orchestration service
  - `BatchOrchestrator`: Coordinates all services for complete workflow
  - `process_video()`: Process single video
  - `process_batch()`: Process multiple videos

## Documentation

For complete documentation, see:
- **[Batch Processing Guide](../../docs/BATCH_PROCESSING.md)**: Comprehensive guide with examples
- **[Architecture](../../ARCHITECTURE.md)**: System architecture and design principles
- **[Integration Tests](../tests/test_integration/)**: Example usage and validation

## Architecture

The batch module follows MVP principles with clear separation:

```
User Code
    ↓
API Layer (api.py, cli.py)
    ↓
Batch Layer (BatchOrchestrator)
    ↓
Service Layer (VideoService, STIVService, etc.)
```

## Features

- Template-based processing with scaffold .ivy projects
- Progress reporting via callbacks
- Detailed error tracking with stage information
- Structured results with dataclasses
- Type-safe configuration
- Dependency injection for testability

## Example Usage

### Process Batch from CSV

```python
from image_velocimetry_tools.api import process_batch_csv

batch_result = process_batch_csv(
    scaffold_path="scaffold.ivy",
    batch_csv_path="batch_config.csv",
    output_directory="batch_results/",
)

# Get statistics
summary = batch_result.get_discharge_summary()
print(f"Mean discharge: {summary['mean']:.2f} m³/s")

# Check individual results
for result in batch_result.video_results:
    if result.success:
        print(f"✓ {result.video_path}: Q={result.total_discharge:.2f} m³/s")
    else:
        print(f"✗ {result.video_path}: {result.error_message}")
```

### Custom Workflow

```python
from image_velocimetry_tools.batch import BatchOrchestrator, BatchVideoConfig
from image_velocimetry_tools.services.project_service import ProjectService

# Load scaffold
project_service = ProjectService()
scaffold_dict = project_service.load_scaffold_configuration("scaffold.ivy")

# Convert to ScaffoldConfig dataclass
from image_velocimetry_tools.batch.config import ScaffoldConfig, VideoConfig

scaffold = ScaffoldConfig(
    scaffold_path="scaffold.ivy",
    **scaffold_dict
)

# Create video configuration
video = VideoConfig(
    video_path="river.mp4",
    water_surface_elevation=318.5,
    alpha=0.85,
)

# Process
orchestrator = BatchOrchestrator()
result = orchestrator.process_video(
    config=BatchVideoConfig(scaffold=scaffold, video=video),
    output_directory="results/",
)
```

## Development

### Running Tests

```bash
# Run batch orchestrator tests
pytest image_velocimetry_tools/tests/test_integration/test_batch_orchestrator.py -v

# Run all integration tests
pytest image_velocimetry_tools/tests/test_integration/ -v -s
```

### Adding New Features

1. Add methods to service layer first
2. Update BatchOrchestrator to use new methods
3. Update API functions if needed
4. Update CLI commands if needed
5. Add tests and documentation

## Known Limitations

- Grid points not yet automatically stored in scaffold (TODO)
- .ivy project saving not yet complete (TODO)
- Time format in CSV only supports seconds (not HH:MM:SS)

## See Also

- [VideoService](../services/video_service.py): Video operations
- [STIVService](../services/stiv_service.py): STIV velocimetry
- [DischargeService](../services/discharge_service.py): Discharge calculations
- [ProjectService](../services/project_service.py): Project save/load
