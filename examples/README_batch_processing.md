# IVyTools Batch Processing Guide

## Overview

The batch processing feature allows you to process multiple videos from the same fixed camera using a **scaffold .ivy project** as a template. This is ideal for:

- Processing time-lapse videos of a river at different flow stages
- Analyzing multiple discharge measurements from a permanent monitoring station
- Automating velocity and discharge calculations for routine measurements

## Key Concepts

### Scaffold Project

A **scaffold .ivy project** is a template that contains:
- Camera calibration (camera matrix or homography)
- Ground control points (GCPs)
- Cross-section geometry (AreaComp .mat file)
- STIV processing parameters
- Measurement metadata template

The scaffold represents the **fixed** components (camera position, cross-section) while each video has a **variable** water surface elevation (WSE).

### Batch Configuration CSV

A CSV file that lists the videos to process with their metadata:

```csv
video_path,water_surface_elevation,measurement_date,measurement_number,gage_height,start_time,end_time,comments,alpha
/videos/flow1.mp4,2.20,2025-04-04,1,2.15,00:00:00,00:00:10,High flow,0.85
/videos/flow2.mp4,1.85,2025-04-05,2,1.80,,,Recession,0.82
/videos/flow3.mp4,1.50,2025-04-06,3,1.45,5.0,15.0,Base flow,
```

**Required columns:**
- `video_path`: Full path to video file
- `water_surface_elevation`: WSE in meters (critical!)

**Optional columns:**
- `measurement_date`: Date of measurement (YYYY-MM-DD)
- `measurement_number`: Sequential measurement number
- `gage_height`: Gage height in meters
- `start_time`: Start time for frame extraction (HH:MM:SS or seconds)
- `end_time`: End time for frame extraction (HH:MM:SS or seconds)
- `comments`: Any notes about this measurement
- `alpha`: Surface-to-average velocity coefficient (defaults to 0.85 if not specified)

## Workflow

### Step 1: Create a Scaffold Project

1. Open IVyTools GUI
2. Load a representative video from your camera
3. Complete the full processing workflow:
   - Extract frames
   - Set up GCPs and orthorectification
   - Load cross-section geometry (AreaComp .mat file)
   - Configure STIV parameters (search line length, phi range, etc.)
   - Calculate discharge for this reference measurement
4. Save the project as `scaffold_template.ivy`

### Step 2: Prepare Batch Configuration CSV

Create a CSV file listing all videos with their WSE values:

```python
from image_velocimetry_tools.batch_processor import create_batch_config_template

# Create a template
create_batch_config_template("my_batch_config.csv")

# Edit the CSV file with your actual video paths and WSE values
```

### Step 3: Run Batch Processing

```python
from image_velocimetry_tools.batch_processor import BatchProcessor

# Initialize processor
processor = BatchProcessor(
    scaffold_ivy_path="scaffold_template.ivy",
    batch_config_csv="my_batch_config.csv"
)

# Process all videos
results = processor.process_batch(output_directory="./batch_results")

# Export summary
processor.export_results_csv(results, "batch_discharge_summary.csv")
```

Or use the example script:

```bash
python examples/run_batch_processing.py
```

### Step 4: Review Results

The batch processor creates:

1. **Individual .ivy files** (one per video)
   - Location: `{output_directory}/{video_basename}.ivy`
   - Can be opened in IVyTools GUI for manual review
   - Contains all processing results, velocities, discharge, etc.

2. **Summary CSV** with discharge results for all videos
   - Columns: filename, date, WSE, discharge, area, uncertainty, etc.
   - Easy to import into Excel/R/Python for analysis

3. **Log file** with detailed processing information
   - Useful for debugging failed videos

## Processing Steps (Automated)

For each video, the batch processor:

1. **Extracts frames** from video (with optional start/end time)
2. **Creates grayscale image stack** for STIV processing
3. **Applies orthorectification** using scaffold camera matrix + new WSE
4. **Generates cross-section grid** points
5. **Runs STIV** to calculate surface velocities
6. **Calculates discharge** using cross-section geometry
7. **Computes uncertainty** (ISO and IVE methods)
8. **Saves .ivy project** for manual review

## Error Handling

The batch processor uses **graceful error handling**:
- If one video fails, processing continues with the next
- Failed videos are logged with error messages
- Summary CSV includes processing status for each video

Common failure reasons:
- Video file not found or corrupted
- FFmpeg extraction failed
- STIV processing error (not enough features)
- Cross-section file missing

## Performance Notes

- Processing time depends on: video length, number of frames, grid points, STIV parameters
- Typical processing time: 2-5 minutes per video (200 frames, 25 grid points)
- Runs headless (no GUI) - suitable for server/cluster deployment
- Temporary frame files are automatically cleaned up

## Example Use Case

**Scenario:** You have a fixed camera monitoring a river. You collected 10 videos over a week as the flow changed.

**Setup:**
1. Process the first video in IVyTools GUI to create the scaffold
2. Note the WSE for each of the 10 videos (from pressure transducer or manual measurement)
3. Create CSV with video paths and WSE values
4. Run batch processor

**Result:**
- 10 .ivy files (one per flow condition)
- Summary CSV with discharge rating curve data
- Can plot Q vs. WSE, analyze uncertainties, etc.

## Tips

1. **Verify scaffold quality** - Make sure GCPs, cross-section, and STIV parameters are good
2. **Consistent video settings** - All videos should be from the same camera position/settings
3. **Accurate WSE** - This is the most critical input! Double-check WSE values
4. **Start small** - Test with 2-3 videos first before running a large batch
5. **Review individual .ivy files** - Open in GUI to verify results look reasonable

## API Reference

### BatchProcessor Class

```python
processor = BatchProcessor(
    scaffold_ivy_path: str,        # Path to scaffold .ivy
    batch_config_csv: str = None,  # Path to batch CSV (optional)
    batch_configs: List[BatchVideoConfig] = None  # Or provide list directly
)

# Process all videos
results = processor.process_batch(
    output_directory: str,         # Where to save outputs
    progress_callback = None       # Optional callback function
)

# Export summary
processor.export_results_csv(
    results: List[BatchResult],
    output_csv_path: str
)

# Cleanup
processor.cleanup()
```

### Programmatic Usage (without CSV)

```python
from image_velocimetry_tools.batch_processor import (
    BatchProcessor,
    BatchVideoConfig
)

# Create configurations manually
configs = [
    BatchVideoConfig(
        video_path="/videos/flow1.mp4",
        water_surface_elevation=2.20,
        measurement_date="2025-04-04",
        measurement_number=1,
        gage_height=2.15,
        comments="High flow"
    ),
    BatchVideoConfig(
        video_path="/videos/flow2.mp4",
        water_surface_elevation=1.85,
        measurement_date="2025-04-05",
        measurement_number=2,
        gage_height=1.80,
    )
]

# Process
processor = BatchProcessor(
    scaffold_ivy_path="scaffold.ivy",
    batch_configs=configs
)
results = processor.process_batch("./output")
```

## Troubleshooting

**Problem:** "Scaffold project failed to load"
- **Solution:** Verify the .ivy file exists and is a valid IVy project

**Problem:** "No frames extracted from video"
- **Solution:** Check video path, verify FFmpeg is installed, check start/end times

**Problem:** "Cross-section bathymetry file not found"
- **Solution:** Ensure the scaffold .ivy includes the AreaComp .mat file

**Problem:** "STIV processing failed"
- **Solution:** Video may have insufficient features. Try adjusting STIV parameters in scaffold.

## Advanced: Server Deployment

The batch processor can run headless on a server:

```python
# run_batch_server.py
import logging
from image_velocimetry_tools.batch_processor import BatchProcessor

logging.basicConfig(filename='server_batch.log', level=logging.INFO)

processor = BatchProcessor(
    scaffold_ivy_path="/data/scaffold.ivy",
    batch_config_csv="/data/batch_config.csv"
)

results = processor.process_batch("/data/output")
processor.export_results_csv(results, "/data/summary.csv")
```

Run via command line:
```bash
python run_batch_server.py
```

## Support

For questions or issues with batch processing:
- Open an issue on GitHub
- Check the batch processing log file for detailed error messages
- Verify individual .ivy files in GUI to diagnose problems
