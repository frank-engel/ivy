# IVyTools Batch Processing API

Simple Python API for automated batch processing of multiple videos.

## Quick Start

```python
from image_velocimetry_tools.api import run_batch_processing

# Process your videos
results = run_batch_processing(
    scaffold_project='my_template.ivy',
    batch_csv='my_videos.csv',
    output_folder='results'
)

# Print summary
results.print_summary()
```

## Documentation

- **Full Guide**: See [docs/batch_processing_guide.md](../../docs/source/batch_processing_guide.md)
- **Examples**: See [examples/batch_processing_example.py](../../examples/batch_processing_example.py)

## What You Need

1. **Template Project** (.ivy file)
   - Camera calibration complete
   - Cross-section loaded
   - STIV settings configured

2. **Video List** (CSV file)
   - Required columns: `video_path`, `water_surface_elevation`
   - Optional columns: `start_time`, `end_time`, `alpha`, etc.

3. **Output Folder**
   - Where results will be saved

## Example CSV File

```csv
video_path,water_surface_elevation,start_time,end_time
videos/video1.mp4,525.4,10,30
videos/video2.mp4,526.1,15,35
videos/video3.mp4,523.8,10,25
```

## API Reference

### `run_batch_processing()`

Main function for batch processing.

**Parameters:**
- `scaffold_project` (str): Path to template .ivy file
- `batch_csv` (str): Path to CSV file with video list
- `output_folder` (str): Path to output folder
- `stop_on_error` (bool): Stop if a job fails (default: False)
- `save_projects` (bool): Save .ivy project for each video (default: False)
- `progress_callback` (callable): Function to call for progress updates (optional)

**Returns:**
- `BatchResults`: Object with results for all jobs

### `BatchResults`

Contains results from batch processing.

**Attributes:**
- `jobs` (list): List of JobResult objects
- `total_jobs` (int): Total number of videos
- `successful_jobs` (int): Number successful
- `failed_jobs` (int): Number failed
- `total_time` (float): Processing time in seconds
- `output_folder` (str): Path to output
- `summary_csv` (str): Path to summary CSV

**Methods:**
- `print_summary()`: Print human-readable summary
- `get_successful_jobs()`: Get list of successful jobs
- `get_failed_jobs()`: Get list of failed jobs
- `get_job_by_video(video_name)`: Find job by video filename

### `JobResult`

Results for a single video.

**Attributes:**
- `job_id` (str): Job identifier
- `video_name` (str): Video filename
- `status` (str): "completed" or "failed"
- `discharge` (float): Discharge in display units
- `area` (float): Area in display units
- `discharge_units` (str): "cfs" or "m³/s"
- `area_units` (str): "ft²" or "m²"
- `processing_time` (float): Time in seconds
- `error_message` (str): Error if failed
- `details` (dict): Additional statistics

**Properties:**
- `successful` (bool): True if completed
- `failed` (bool): True if failed

## More Information

See the complete guide at [docs/batch_processing_guide.md](../../docs/source/batch_processing_guide.md)
