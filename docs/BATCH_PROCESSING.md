  # Batch Video Processing

IVyTools provides powerful batch processing capabilities for analyzing multiple river videos in an automated workflow. This guide explains how to use the batch processing system via Python API or command-line interface (CLI).

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Getting Started](#getting-started)
4. [Python API](#python-api)
5. [Command-Line Interface](#command-line-interface)
6. [Batch CSV Format](#batch-csv-format)
7. [Workflow Details](#workflow-details)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

## Overview

The batch processing system automates the complete IVyTools workflow:

1. **Extract frames** from video using ffmpeg
2. **Rectify frames** to world coordinates (homography, camera matrix, or scale)
3. **Compute velocities** using STIV (Space-Time Image Velocimetry)
4. **Calculate discharge** using velocity-area method
5. **Save results** to CSV and .ivy project files

### Key Features

- **Template-based processing**: Use a scaffold .ivy project as template for multiple videos
- **Flexible configuration**: Process via Python API, CLI, or CSV batch files
- **Progress reporting**: Real-time progress updates via callbacks
- **Error handling**: Detailed error tracking with stage-specific information
- **Results validation**: Compare outputs against expected discharge values

### Use Cases

- **Continuous monitoring**: Process videos from long-term camera deployments
- **Rating curve development**: Analyze videos across range of flows
- **Batch reprocessing**: Reanalyze historical videos with updated parameters
- **Automated workflows**: Integrate into data pipelines and automation scripts

## Architecture

The batch processing system follows the MVP (Model-View-Presenter) architecture with clear separation of concerns:

### Service Layer

Five core services handle business logic:

- **VideoService**: Video operations (metadata, frame extraction)
- **OrthorectificationService**: Image rectification (homography, camera matrix, scale)
- **STIVService**: STIV velocimetry analysis
- **DischargeService**: Discharge calculation and uncertainty
- **ProjectService**: Project save/load operations

### Batch Layer

The batch layer orchestrates workflows:

- **BatchOrchestrator**: Coordinates services for complete workflow
- **Configuration dataclasses**: Type-safe configuration (`ScaffoldConfig`, `VideoConfig`)
- **Result dataclasses**: Structured results (`ProcessingResult`, `BatchResult`)

### API Layer

High-level interfaces for programmatic and CLI access:

- **Python API** (`api.py`): Functions for Python scripts and notebooks
- **CLI** (`cli.py`): Command-line tool for terminal usage

```
┌─────────────────────────────────────────┐
│         User Interfaces                 │
│  ┌──────────────┐  ┌──────────────┐    │
│  │  Python API  │  │     CLI      │    │
│  └──────────────┘  └──────────────┘    │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│        Batch Orchestrator               │
│  ┌──────────────────────────────────┐  │
│  │  BatchOrchestrator.process_video │  │
│  │  BatchOrchestrator.process_batch │  │
│  └──────────────────────────────────┘  │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│           Service Layer                 │
│  ┌──────┐ ┌───────┐ ┌──────┐ ┌──────┐ │
│  │Video │ │ Ortho │ │ STIV │ │Disch.│ │
│  │Svc   │ │  Svc  │ │ Svc  │ │ Svc  │ │
│  └──────┘ └───────┘ └──────┘ └──────┘ │
└─────────────────────────────────────────┘
```

## Getting Started

### Installation

Install IVyTools with batch processing support:

```bash
pip install -e .
```

This installs:
- The `image_velocimetry_tools` Python package
- The `ivytools-batch` command-line tool

### Prerequisites

1. **ffmpeg and ffprobe**: Required for video processing
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg

   # macOS
   brew install ffmpeg

   # Windows: Download from https://ffmpeg.org/
   # Set environment variables FFMPEG-IVyTools and FFPROBE-IVyTools
   ```

2. **Scaffold project**: A template .ivy project containing:
   - Camera calibration and rectification parameters
   - STIV search parameters (phi_origin, phi_range, dphi, num_pixels)
   - Cross-section bathymetry and grid configuration
   - Example: `examples/scaffold_project.ivy`

3. **Video files**: Input videos in formats supported by ffmpeg (mp4, avi, mov, etc.)

### Quick Start

Process a single video using the CLI:

```bash
ivytools-batch process \
  examples/scaffold_project.ivy \
  examples/videos/03337000_bullet_20170630-120000.mp4 \
  --wse 318.5 \
  --output results/
```

Or using Python:

```python
from image_velocimetry_tools.api import process_video

result = process_video(
    scaffold_path="examples/scaffold_project.ivy",
    video_path="examples/videos/03337000_bullet_20170630-120000.mp4",
    water_surface_elevation=318.5,
    output_directory="results/",
)

print(f"Discharge: {result.total_discharge:.2f} m³/s")
```

## Python API

The Python API provides clean, high-level functions for batch processing.

### Basic Usage

#### Process Single Video

```python
from image_velocimetry_tools.api import process_video

result = process_video(
    scaffold_path="scaffold.ivy",
    video_path="river.mp4",
    water_surface_elevation=318.5,
    output_directory="results/",
    alpha=0.85,
    start_time=15.0,
    end_time=20.0,
)

if result.success:
    print(f"✓ Discharge: {result.total_discharge:.4f} m³/s")
    print(f"  Area:      {result.total_area:.4f} m²")
    print(f"  Velocity:  {result.mean_velocity:.4f} m/s")
else:
    print(f"✗ Failed: {result.error_message}")
```

#### Process Batch from CSV

```python
from image_velocimetry_tools.api import process_batch_csv

batch_result = process_batch_csv(
    scaffold_path="scaffold.ivy",
    batch_csv_path="batch_config.csv",
    output_directory="batch_results/",
)

print(f"Processed: {batch_result.successful}/{batch_result.total_videos}")

# Discharge statistics
summary = batch_result.get_discharge_summary()
print(f"Mean discharge: {summary['mean']:.2f} m³/s")
print(f"Range: {summary['min']:.2f} - {summary['max']:.2f} m³/s")
```

#### Process Batch Programmatically

```python
from image_velocimetry_tools.api import process_batch

video_configs = [
    {
        "video_path": "video1.mp4",
        "water_surface_elevation": 318.5,
        "alpha": 0.85,
        "start_time": 15.0,
        "end_time": 20.0,
    },
    {
        "video_path": "video2.mp4",
        "water_surface_elevation": 318.7,
        "alpha": 0.85,
    },
]

batch_result = process_batch(
    scaffold_path="scaffold.ivy",
    video_configs=video_configs,
    output_directory="batch_results/",
)
```

#### Load and Inspect Scaffold

```python
from image_velocimetry_tools.api import load_scaffold

scaffold = load_scaffold("scaffold.ivy")

print(f"Rectification: {scaffold.rectification_method}")
print(f"STIV params: {scaffold.stiv_params}")
print(f"Grid points: {scaffold.grid_params['num_points']}")
```

### Progress Reporting

Add a progress callback for real-time updates:

```python
def progress_callback(percent, message):
    print(f"[{percent:3d}%] {message}")

result = process_video(
    scaffold_path="scaffold.ivy",
    video_path="river.mp4",
    water_surface_elevation=318.5,
    output_directory="results/",
    progress_callback=progress_callback,
)
```

### API Reference

#### `process_video()`

Process a single video through complete workflow.

**Parameters:**
- `scaffold_path` (str): Path to scaffold .ivy template project
- `video_path` (str): Path to input video file
- `water_surface_elevation` (float): Water surface elevation in meters
- `output_directory` (str): Directory for output files
- `measurement_date` (str, optional): Date in YYYY-MM-DD format
- `alpha` (float, optional): Alpha coefficient (default: 0.85)
- `start_time` (float, optional): Start time in seconds
- `end_time` (float, optional): End time in seconds
- `frame_step` (int, optional): Extract every Nth frame (default: 1)
- `max_frames` (int, optional): Maximum frames to extract
- `comments` (str, optional): Comments about measurement
- `progress_callback` (callable, optional): Progress callback function
- `cleanup_temp_files` (bool, optional): Delete temp files (default: True)

**Returns:** `ProcessingResult` object with:
- `success` (bool): Whether processing succeeded
- `total_discharge` (float): Discharge in m³/s
- `total_area` (float): Cross-sectional area in m²
- `mean_velocity` (float): Mean velocity in m/s
- `num_frames_extracted` (int): Number of frames
- `output_csv_path` (str): Path to discharge CSV
- `output_project_path` (str): Path to .ivy project
- `error_message` (str): Error description if failed
- `error_stage` (str): Stage where error occurred

#### `process_batch()`

Process multiple videos with shared scaffold.

**Parameters:**
- `scaffold_path` (str): Path to scaffold .ivy template
- `video_configs` (list): List of video configuration dicts
- `output_directory` (str): Root output directory
- `progress_callback` (callable, optional): Progress callback
- `cleanup_temp_files` (bool, optional): Delete temp files

**Returns:** `BatchResult` object with:
- `total_videos` (int): Total number of videos
- `successful` (int): Number of successful processes
- `failed` (int): Number of failures
- `video_results` (list): List of `ProcessingResult` objects
- `batch_csv_path` (str): Path to batch summary CSV

#### `process_batch_csv()`

Process batch from CSV configuration file.

**Parameters:**
- `scaffold_path` (str): Path to scaffold .ivy template
- `batch_csv_path` (str): Path to batch CSV file
- `output_directory` (str): Root output directory
- `progress_callback` (callable, optional): Progress callback
- `cleanup_temp_files` (bool, optional): Delete temp files

**Returns:** `BatchResult` object

#### `load_scaffold()`

Load scaffold configuration from .ivy file.

**Parameters:**
- `scaffold_path` (str): Path to scaffold .ivy file
- `temp_dir` (str, optional): Temporary directory for extraction

**Returns:** `ScaffoldConfig` object with:
- `rectification_method` (str): Method name
- `rectification_params` (dict): Method-specific parameters
- `stiv_params` (dict): STIV search parameters
- `cross_section_data` (dict): Cross-section bathymetry
- `grid_params` (dict): Grid configuration

## Command-Line Interface

The CLI provides terminal access to batch processing functionality.

### Installation

The CLI is installed automatically with the package:

```bash
pip install -e .
ivytools-batch --help
```

### Commands

#### `process` - Process Single Video

Process one video through the complete workflow.

```bash
ivytools-batch process SCAFFOLD VIDEO --wse WSE [OPTIONS]
```

**Required arguments:**
- `SCAFFOLD`: Path to scaffold .ivy template
- `VIDEO`: Path to video file
- `--wse WSE`: Water surface elevation in meters

**Optional arguments:**
- `--output DIR`, `-o DIR`: Output directory (default: output/)
- `--alpha ALPHA`: Alpha coefficient (default: 0.85)
- `--start TIME`: Start time in seconds
- `--end TIME`: End time in seconds
- `--frame-step N`: Extract every Nth frame (default: 1)
- `--max-frames N`: Maximum frames to extract
- `--date DATE`: Measurement date (YYYY-MM-DD)
- `--comments TEXT`: Comments about measurement
- `--keep-temp`: Keep temporary files
- `--quiet`, `-q`: Suppress progress output

**Example:**

```bash
ivytools-batch process \
  scaffold.ivy \
  river_20230615.mp4 \
  --wse 318.5 \
  --output results/ \
  --start 15 \
  --end 20 \
  --alpha 0.85
```

#### `batch` - Process Batch from CSV

Process multiple videos from CSV configuration.

```bash
ivytools-batch batch SCAFFOLD CSV [OPTIONS]
```

**Required arguments:**
- `SCAFFOLD`: Path to scaffold .ivy template
- `CSV`: Path to batch configuration CSV

**Optional arguments:**
- `--output DIR`, `-o DIR`: Output directory (default: batch_output/)
- `--keep-temp`: Keep temporary files
- `--quiet`, `-q`: Suppress progress output

**Example:**

```bash
ivytools-batch batch scaffold.ivy batch_config.csv --output batch_results/
```

#### `validate` - Validate Configuration

Validate scaffold and batch CSV files.

```bash
ivytools-batch validate SCAFFOLD [CSV]
```

**Arguments:**
- `SCAFFOLD`: Path to scaffold .ivy template
- `CSV` (optional): Path to batch CSV file

**Example:**

```bash
ivytools-batch validate scaffold.ivy batch_config.csv
```

#### `info` - Display Scaffold Information

Display scaffold configuration details.

```bash
ivytools-batch info SCAFFOLD
```

**Example:**

```bash
ivytools-batch info scaffold.ivy
```

### CLI Output

The CLI provides formatted output with:

- **Progress reporting**: Real-time updates during processing
- **Status indicators**: ✓ for success, ✗ for failure
- **Summary statistics**: Discharge, area, velocity, processing time
- **Error details**: Stage and error message for failures

Example output:

```
Processing: river_20230615.mp4
Output: results/

[  5%] Extracting frames...
[ 20%] Getting video metadata...
[ 25%] Rectifying frames...
[ 50%] Computing velocities (STIV)...
[ 85%] Calculating discharge...
[ 90%] Saving results...
[100%] Complete!

============================================================
✓ Processing completed successfully!

Results:
  Discharge:       3.1995 m³/s
  Area:            4.6755 m²
  Mean velocity:   0.6843 m/s
  Frames:          150
  Processing time: 45.2s

Output files:
  CSV:     results/river_20230615_discharge.csv
  Project: results/river_20230615_results.ivy
```

## Batch CSV Format

The batch CSV file defines multiple videos to process with a shared scaffold.

### Required Columns

- `video_path`: Path to video file
- `water_surface_elevation`: Water surface elevation in meters

### Optional Columns

- `measurement_date`: Date in YYYY-MM-DD format
- `alpha`: Alpha coefficient (default: 0.85)
- `start_time`: Start time in seconds
- `end_time`: End time in seconds
- `frame_step`: Extract every Nth frame (default: 1)
- `max_frames`: Maximum frames to extract
- `comments`: Comments about measurement

### Example CSV

```csv
video_path,water_surface_elevation,measurement_date,alpha,start_time,end_time,comments
videos/river_115500.mp4,318.45,2017-06-30,0.85,15,20,Morning measurement
videos/river_120000.mp4,318.50,2017-06-30,0.85,15,20,Midday measurement
videos/river_103500.mp4,318.72,2017-07-01,0.85,15,20,Evening measurement
```

### Tips

- **Relative paths**: Video paths can be relative to CSV location
- **Time format**: Times are in seconds (not HH:MM:SS)
- **Missing values**: Optional columns can be empty
- **Comments**: Use comments for metadata and notes

## Workflow Details

The batch processing workflow consists of 8 stages:

### Stage 1: Frame Extraction

Extract frames from video using ffmpeg:
- Respects start/end time window
- Applies frame step (every Nth frame)
- Limits to max_frames if specified
- Saves frames as JPG images

**Output**: `frames/` directory with frame files

### Stage 2: Video Metadata

Extract video properties using ffprobe:
- Resolution (width x height)
- Frame rate (fps)
- Duration (seconds)
- Frame count

**Output**: Video metadata dictionary

### Stage 3: Frame Rectification

Rectify frames to world coordinates:
- Applies rectification method (homography, camera matrix, scale)
- Uses parameters from scaffold
- Produces georeferenced images

**Output**: `rectified/` directory with rectified frames

### Stage 4: Grid Preparation

Extract grid points and GSD from scaffold:
- Loads grid points (velocity measurement locations)
- Gets pixel GSD (ground sample distance)

**Output**: Grid configuration

### Stage 5: STIV Processing

Compute velocities using STIV:
- Creates image stack from rectified frames
- Performs exhaustive search over orientation angles
- Extracts magnitude and direction at each grid point

**Output**: Velocity fields (magnitude, direction)

### Stage 6: Discharge Calculation

Calculate discharge using velocity-area method:
- Extracts cross-section stations and depths
- Interpolates velocities to cross-section
- Computes discharge using velocity-area integration
- Calculates uncertainty

**Output**: Discharge dataframe with Q, A, V

### Stage 7: Results Saving

Save results to files:
- Discharge CSV with station-by-station data
- .ivy project with complete configuration
- Summary statistics

**Output**: CSV and .ivy files

### Stage 8: Cleanup

Remove temporary files:
- Deletes raw frames (keeps rectified)
- Cleans up scaffold temp directory
- Optional: keep all files with `--keep-temp`

## Examples

### Example 1: Rating Curve Development

Process videos across a range of flows to develop a rating curve:

```python
from image_velocimetry_tools.api import process_batch

# Define videos at different water levels
video_configs = [
    {"video_path": "low_flow.mp4", "water_surface_elevation": 318.2, "alpha": 0.85},
    {"video_path": "med_flow.mp4", "water_surface_elevation": 318.5, "alpha": 0.85},
    {"video_path": "high_flow.mp4", "water_surface_elevation": 318.9, "alpha": 0.85},
]

# Process batch
batch_result = process_batch(
    scaffold_path="scaffold.ivy",
    video_configs=video_configs,
    output_directory="rating_curve/",
)

# Extract discharge vs. stage
for result in batch_result.get_successful_results():
    wse = result.video_metadata.get("wse", 0)  # Would need to store this
    q = result.total_discharge
    print(f"WSE={wse:.2f}m, Q={q:.2f}m³/s")
```

### Example 2: Time Series Analysis

Process videos from continuous monitoring:

```python
import glob
from image_velocimetry_tools.api import process_batch_csv

# Assume videos named with timestamps: river_YYYYMMDD-HHMMSS.mp4
# CSV has water levels from gage data

batch_result = process_batch_csv(
    scaffold_path="scaffold.ivy",
    batch_csv_path="time_series.csv",
    output_directory="time_series_results/",
)

# Plot time series
import matplotlib.pyplot as plt

times = []
discharges = []

for result in batch_result.get_successful_results():
    # Parse timestamp from filename
    # Add to time series
    pass

plt.plot(times, discharges)
plt.xlabel("Time")
plt.ylabel("Discharge (m³/s)")
plt.title("River Discharge Time Series")
plt.show()
```

### Example 3: Automated Pipeline

Integrate batch processing into an automated pipeline:

```python
import schedule
import time
from pathlib import Path
from image_velocimetry_tools.api import process_video

def process_latest_video():
    """Process the most recent video from camera."""

    # Find latest video
    videos = sorted(Path("incoming/").glob("*.mp4"))
    if not videos:
        return

    latest = videos[-1]

    # Get water level from gage API
    wse = get_water_level_from_api()  # Your function

    # Process video
    result = process_video(
        scaffold_path="scaffold.ivy",
        video_path=str(latest),
        water_surface_elevation=wse,
        output_directory=f"results/{latest.stem}/",
    )

    if result.success:
        # Upload results to database
        upload_to_database(result)  # Your function

        # Move processed video to archive
        latest.rename(f"archive/{latest.name}")
    else:
        # Alert on failure
        send_alert(f"Processing failed: {result.error_message}")

# Schedule processing every hour
schedule.every().hour.at(":05").do(process_latest_video)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Example 4: Sensitivity Analysis

Test sensitivity to alpha coefficient:

```python
from image_velocimetry_tools.api import process_video
import numpy as np

alphas = np.linspace(0.75, 0.95, 5)
results = []

for alpha in alphas:
    result = process_video(
        scaffold_path="scaffold.ivy",
        video_path="test_video.mp4",
        water_surface_elevation=318.5,
        output_directory=f"sensitivity_alpha_{alpha:.2f}/",
        alpha=alpha,
    )

    if result.success:
        results.append({
            "alpha": alpha,
            "discharge": result.total_discharge,
            "velocity": result.mean_velocity,
        })

# Plot sensitivity
import matplotlib.pyplot as plt

alphas = [r["alpha"] for r in results]
discharges = [r["discharge"] for r in results]

plt.plot(alphas, discharges, 'o-')
plt.xlabel("Alpha coefficient")
plt.ylabel("Discharge (m³/s)")
plt.title("Sensitivity to Alpha")
plt.grid(True)
plt.show()
```

## Troubleshooting

### Common Issues

#### 1. ffmpeg/ffprobe Not Found

**Error:**
```
FileNotFoundError: ffprobe not found
```

**Solution:**
- Install ffmpeg: `sudo apt-get install ffmpeg` (Linux) or `brew install ffmpeg` (Mac)
- Windows: Download from https://ffmpeg.org/ and set environment variables:
  ```
  FFMPEG-IVyTools=C:\path\to\ffmpeg.exe
  FFPROBE-IVyTools=C:\path\to\ffprobe.exe
  ```

#### 2. Grid Points Not Found

**Error:**
```
ValueError: Grid points not found in scaffold
```

**Solution:**
- Grid points are not yet automatically stored in scaffold .ivy files
- TODO: Implement grid point generation from cross-section line
- Temporary workaround: Manually add grid_points to scaffold

#### 3. Video File Not Found

**Error:**
```
FileNotFoundError: Video file not found: /path/to/video.mp4
```

**Solution:**
- Verify video path is correct
- Use absolute paths or paths relative to working directory
- Check file permissions

#### 4. Scaffold Invalid

**Error:**
```
ValueError: Missing required STIV parameter: phi_origin
```

**Solution:**
- Validate scaffold with `ivytools-batch validate scaffold.ivy`
- Ensure scaffold was created with complete configuration
- Check that all required parameters are present

#### 5. Memory Issues

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
- Reduce `max_frames` to limit memory usage
- Use `frame_step > 1` to extract fewer frames
- Process shorter time windows
- Close other applications

#### 6. Processing Slow

**Issue:** Processing takes too long

**Solutions:**
- Use shorter time windows (`start_time`, `end_time`)
- Increase `frame_step` to extract fewer frames
- Reduce `max_frames`
- Check ffmpeg is using hardware acceleration
- Use SSD for temporary files

### Getting Help

If you encounter issues:

1. Check this documentation
2. Run validation: `ivytools-batch validate scaffold.ivy batch.csv`
3. Check logs for detailed error messages
4. Report issues at: https://github.com/frank-engel-usgs/IVyTools/issues

### Best Practices

1. **Test with one video first**: Validate workflow before batch processing
2. **Use validation command**: Run `ivytools-batch validate` before large batches
3. **Monitor disk space**: Frame extraction can use significant space
4. **Keep scaffolds organized**: Use descriptive names for different sites/configurations
5. **Document parameters**: Use comments in batch CSV for metadata
6. **Backup results**: Save discharge CSVs and summary statistics
7. **Version control scaffolds**: Track changes to calibration and parameters

## Advanced Topics

### Custom Workflows

For custom workflows, use the service layer directly:

```python
from image_velocimetry_tools.batch import BatchOrchestrator
from image_velocimetry_tools.services.video_service import VideoService

# Create services with custom configuration
video_service = VideoService()
orchestrator = BatchOrchestrator(video_service=video_service)

# Custom processing logic
# ...
```

### Parallel Processing

Process multiple videos in parallel using multiprocessing:

```python
from multiprocessing import Pool
from image_velocimetry_tools.api import process_video

def process_one(config):
    return process_video(**config)

# Define video configs
configs = [...]  # List of config dicts

# Process in parallel
with Pool(processes=4) as pool:
    results = pool.map(process_one, configs)
```

### Result Validation

Compare results against expected values:

```python
from image_velocimetry_tools.api import process_video

result = process_video(...)

# Validate against expected discharge
expected_q = 3.1995  # m³/s
tolerance = 0.01  # 0.5%

if abs(result.total_discharge - expected_q) < tolerance:
    print("✓ Discharge within tolerance")
else:
    print(f"⚠ Discharge mismatch: {result.total_discharge:.4f} vs {expected_q:.4f}")
```

## References

- IVyTools GitHub: https://github.com/frank-engel-usgs/IVyTools
- Architecture documentation: `ARCHITECTURE.md`
- STIV methodology: Fujita et al. (2007)
- Discharge calculation: USGS techniques and methods

## Changelog

- **v1.0.0** (2024): Initial batch processing release
  - Python API with `process_video()`, `process_batch()`, `process_batch_csv()`
  - CLI with `process`, `batch`, `validate`, `info` commands
  - BatchOrchestrator service with workflow orchestration
  - Configuration and result dataclasses
  - Integration tests with real data

---

**For more information, see the main IVyTools documentation or contact the development team.**
