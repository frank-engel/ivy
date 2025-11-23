# IVyTools Batch Processing Guide

**Process Multiple Videos Automatically**

This guide explains how to use IVyTools to automatically analyze multiple videos at once. This is useful when you have many videos from the same location and want to process them all with the same settings.

---

## Table of Contents

1. [What is Batch Processing?](#what-is-batch-processing)
2. [Before You Start](#before-you-start)
3. [Step 1: Prepare Your Template Project](#step-1-prepare-your-template-project)
4. [Step 2: Create Your Video List (CSV File)](#step-2-create-your-video-list-csv-file)
5. [Step 3: Run Batch Processing](#step-3-run-batch-processing)
6. [Understanding the Results](#understanding-the-results)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Options](#advanced-options)

---

## What is Batch Processing?

Batch processing lets you analyze many videos automatically instead of processing each one manually through the IVyTools interface. You provide:

1. **A template project** (.ivy file) - Contains your camera setup and cross-section
2. **A list of videos** (CSV file) - Specifies which videos to process and their water levels

IVyTools will then:
- Process each video using the same camera calibration and settings
- Calculate discharge (flow rate) for each video
- Create a summary report with all results

This saves hours of manual work when you have multiple videos from the same camera location.

---

## Before You Start

### What You Need

1. **Python installed** on your computer (version 3.8 or newer)
2. **IVyTools installed** (see installation guide)
3. **A template project** - One .ivy project file with:
   - Camera calibration (completed in IVyTools)
   - Cross-section geometry (your .ac3 file loaded)
   - STIV analysis settings configured
4. **Your videos** - Multiple video files from the same camera location
5. **Water level information** - The water surface elevation for each video

### Important Notes

- All videos must be from the **same camera position** as your template project
- Your camera calibration only works for videos from that specific camera location
- Videos can be different lengths and taken at different times
- Water levels can be different for each video (you'll specify this in the CSV file)

---

## Step 1: Prepare Your Template Project

Your template project (.ivy file) serves as the master configuration for all videos. It should contain:

### 1. Camera Calibration
Complete the camera calibration in IVyTools:
- Set up your Ground Control Points (GCPs)
- Complete the camera matrix calibration
- Verify the rectification looks correct

### 2. Cross-Section Geometry
- Load your .ac3 cross-section file
- Make sure it's positioned correctly
- Verify the measurement grid looks good

### 3. STIV Settings
- Configure STIV parameters (pixel size, angle range, etc.)
- Test on one video to make sure settings work well
- Save the project

**Tip:** Process one video completely through the IVyTools interface first. Once you're happy with the results, save that project and use it as your template.

---

## Step 2: Create Your Video List (CSV File)

The CSV file tells IVyTools which videos to process and what water level to use for each one.

### Required Information

Create a spreadsheet (Excel, Google Sheets, etc.) with these columns:

| Column Name | Description | Example |
|-------------|-------------|---------|
| `video_path` | Path to your video file | `videos/stream_06_30_2017.mp4` |
| `water_surface_elevation` | Water level in feet or meters | `525.4` |

### Optional Information

You can also include these columns:

| Column Name | Description | Example |
|-------------|-------------|---------|
| `start_time` | Start analyzing at this time in the video | `10.5` (10.5 seconds) |
| `end_time` | Stop analyzing at this time | `30` (30 seconds) |
| `alpha` | Velocity correction factor | `0.85` (default) |
| `measurement_number` | Your reference number | `12345` |
| `measurement_date` | Date of measurement | `2017-06-30` |
| `measurement_time` | Time of measurement | `11:55:00` |
| `gage_height` | USGS gage reading | `8.45` |
| `comments` | Notes about this measurement | `High water event` |

### Example CSV File

```csv
video_path,water_surface_elevation,start_time,end_time,comments
videos/june_30_morning.mp4,525.4,10,30,Clear conditions
videos/june_30_noon.mp4,526.1,15,35,Slightly higher water
videos/july_11_morning.mp4,523.8,10,25,Normal flow
```

### Important Tips

- **Video paths** can be:
  - Relative to the CSV file location: `videos/myfile.mp4`
  - Or full paths: `C:/Users/YourName/Videos/myfile.mp4`
- **Water elevation units** should match your template project (English = feet, Metric = meters)
- **Time format** can be:
  - Seconds: `15.5`
  - Minutes:seconds: `1:30`
  - Hours:minutes:seconds: `0:01:30.5`
- Save as **CSV format** (not Excel .xlsx)

---

## Step 3: Run Batch Processing

### Option A: Use the Example Script (Easiest)

1. Open the example script: `examples/batch_processing_example.py`

2. Modify it to use your files:

```python
from image_velocimetry_tools.api import run_batch_processing

# Update these paths to your files
results = run_batch_processing(
    scaffold_project='path/to/your/template.ivy',
    batch_csv='path/to/your/videos.csv',
    output_folder='path/to/results'
)

# Print results
results.print_summary()
```

3. Run the script:
   - On Windows: Double-click the script or run `python batch_processing_example.py`
   - On Mac/Linux: Run `python3 batch_processing_example.py`

### Option B: Write Your Own Script

Create a new Python file (e.g., `my_batch_process.py`):

```python
from image_velocimetry_tools.api import run_batch_processing

# Process your videos
results = run_batch_processing(
    scaffold_project='my_template.ivy',
    batch_csv='my_videos.csv',
    output_folder='results'
)

# Show summary
print(f"Processed {results.total_jobs} videos")
print(f"Successful: {results.successful_jobs}")
print(f"Failed: {results.failed_jobs}")

# Show details for each video
for job in results.jobs:
    if job.successful:
        print(f"{job.video_name}: {job.discharge:.2f} {job.discharge_units}")
    else:
        print(f"{job.video_name}: FAILED - {job.error_message}")
```

### Option C: Track Progress While Processing

If you're processing many videos, you might want to see progress:

```python
from image_velocimetry_tools.api import run_batch_processing

# Define a function to show progress
def show_progress(job_num, total, video_name, status):
    print(f"Processing {job_num}/{total}: {video_name} - {status}")

# Run with progress updates
results = run_batch_processing(
    scaffold_project='my_template.ivy',
    batch_csv='my_videos.csv',
    output_folder='results',
    progress_callback=show_progress
)
```

---

## Understanding the Results

### Output Folder Contents

After processing, your output folder will contain:

```
results/
├── batch_summary.csv          # Summary of all videos
├── job_001/                   # Results for first video
│   ├── frames/               # Extracted video frames
│   ├── orthorectified/       # Rectified images
│   ├── stiv_results/         # STIV analysis results
│   └── ...
├── job_002/                   # Results for second video
│   └── ...
└── job_003/                   # Results for third video
    └── ...
```

### Reading the Summary CSV

Open `batch_summary.csv` in Excel or any spreadsheet program. Key columns:

| Column | What it Means |
|--------|---------------|
| `status` | "completed" or "failed" |
| `discharge` | Calculated flow rate (cfs or m³/s) |
| `area` | Cross-sectional area (ft² or m²) |
| `Q/A` | Discharge divided by area (mean velocity) |
| `avg_velocity` | Average velocity across section |
| `max_depth` | Maximum water depth in section |
| `processing_time_seconds` | How long it took |
| `error_message` | Why it failed (if applicable) |

### Understanding Your Results

**Discharge (Q):**
- In English units: cubic feet per second (cfs)
- In Metric units: cubic meters per second (m³/s)
- This is the total flow rate through the cross-section

**Area (A):**
- In English units: square feet (ft²)
- In Metric units: square meters (m²)
- This is the wetted cross-sectional area

**Q/A:**
- Discharge divided by area
- Gives you a sense of average velocity
- Higher values = faster flow

**Uncertainty:**
- `uncertainty_iso_95pct`: Standard uncertainty (95% confidence)
- `uncertainty_ive_95pct`: IVE uncertainty (95% confidence)
- Lower values = more confident in the measurement

### Accessing Results in Python

```python
# After running batch processing
results = run_batch_processing(...)

# Print summary
results.print_summary()

# Get successful jobs
successful = results.get_successful_jobs()
for job in successful:
    print(f"{job.video_name}: {job.discharge:.2f} {job.discharge_units}")

# Get failed jobs
failed = results.get_failed_jobs()
for job in failed:
    print(f"{job.video_name} failed: {job.error_message}")

# Find specific video
job = results.get_job_by_video("my_video.mp4")
if job:
    print(f"Discharge: {job.discharge} {job.discharge_units}")
    print(f"Area: {job.area} {job.area_units}")
    print(f"Processing time: {job.processing_time} seconds")
```

---

## Troubleshooting

### "Scaffold project not found"
- Check the file path in your script
- Make sure the .ivy file exists
- Use the full path if needed: `C:/Users/YourName/Documents/template.ivy`

### "Batch CSV not found"
- Check the CSV file path
- Make sure you saved the file as CSV format (not .xlsx)

### "Required field 'video_path' is missing"
- Open your CSV file
- Make sure the first row has column headers
- Make sure `video_path` and `water_surface_elevation` columns exist

### "Video file does not exist"
- Check the video paths in your CSV
- If using relative paths, they should be relative to the CSV file location
- Try using full paths: `C:/Users/YourName/Videos/video.mp4`

### "Failed to parse row"
- Check for empty rows in your CSV (delete them)
- Make sure `water_surface_elevation` contains numbers (not text)
- Make sure time values are in correct format (e.g., "10" or "1:30")

### Job Failed: "FFmpeg not found"
- Make sure FFmpeg is installed on your system
- On Windows, FFmpeg should be in your PATH or in the IVyTools bin folder
- See IVyTools installation guide for FFmpeg setup

### Job Failed: "Insufficient GCPs" or "Rectification failed"
- Your template project's camera calibration may not be complete
- Open the template in IVyTools and verify calibration works
- Make sure you have enough Ground Control Points

### Results Look Wrong
- Verify your template project works correctly on one video first
- Check that water_surface_elevation values in CSV are correct
- Make sure you're using the right units (feet vs meters)
- Verify your cross-section .ac3 file is correct

### Processing is Very Slow
- Processing time depends on video length and resolution
- Typical: 2-10 minutes per video
- Longer videos take longer
- If using `save_projects=True`, this creates large files and is much slower

---

## Advanced Options

### Stop on First Error

By default, if one video fails, processing continues with the remaining videos. To stop immediately when something goes wrong:

```python
results = run_batch_processing(
    scaffold_project='template.ivy',
    batch_csv='videos.csv',
    output_folder='results',
    stop_on_error=True  # Stop if any job fails
)
```

### Save Complete .ivy Projects

You can save a complete .ivy project file for each video. This lets you open individual results in the IVyTools interface for detailed review.

```python
results = run_batch_processing(
    scaffold_project='template.ivy',
    batch_csv='videos.csv',
    output_folder='results',
    save_projects=True  # Save .ivy file for each video
)
```

**Warning:** This creates large files (100-500 MB per video) and significantly slows down processing. Only use if you need to review individual videos in the IVyTools interface.

### Using Different Alpha Values

The alpha coefficient corrects surface velocity to mean channel velocity (typically 0.85). You can specify different values:

**In your CSV:**
```csv
video_path,water_surface_elevation,alpha
video1.mp4,525.4,0.85
video2.mp4,526.1,0.90
video3.mp4,523.8,0.82
```

### Clipping Videos

You can process only part of each video:

```csv
video_path,water_surface_elevation,start_time,end_time
video1.mp4,525.4,10,30
```

This processes from 10 seconds to 30 seconds of the video (skipping the first 10 seconds and everything after 30 seconds).

**Why clip videos?**
- Skip poor quality footage at start/end
- Process only the stable part of a measurement
- Reduce processing time for long videos

---

## Tips for Success

### 1. Test First
Process one video completely through the IVyTools interface before running batch processing. This helps you:
- Verify your camera calibration works
- Fine-tune your STIV settings
- Make sure your cross-section is correct

### 2. Start Small
When first using batch processing:
- Start with 2-3 videos
- Verify results look correct
- Then process your full batch

### 3. Organize Your Files
Keep your files organized:
```
MyProject/
├── template.ivy           # Your template project
├── videos.csv            # Your video list
├── videos/               # Your video files
│   ├── video1.mp4
│   ├── video2.mp4
│   └── video3.mp4
└── results/              # Output folder
```

### 4. Check Water Elevations
- Double-check water surface elevation values in your CSV
- Wrong water levels = wrong results
- Units matter! (feet vs meters)

### 5. Verify Units
Make sure units are consistent:
- Template project display units (English or Metric)
- Water elevations in CSV (feet or meters)
- Cross-section .ac3 file units

### 6. Monitor Progress
For large batches, use the progress callback to track processing:
```python
def show_progress(job_num, total, video, status):
    print(f"[{job_num}/{total}] {video}: {status}")

results = run_batch_processing(
    ...,
    progress_callback=show_progress
)
```

---

## Getting Help

If you encounter issues:

1. **Check this guide** - Review the Troubleshooting section
2. **Check the example** - Look at `examples/batch_processing_example.py`
3. **Verify your template** - Make sure it works on one video first
4. **Check error messages** - Error messages usually indicate what's wrong
5. **Check the summary CSV** - The `error_message` column shows why jobs failed

---

## Quick Reference

### Minimal Working Example

```python
from image_velocimetry_tools.api import run_batch_processing

results = run_batch_processing(
    scaffold_project='my_template.ivy',
    batch_csv='my_videos.csv',
    output_folder='results'
)

results.print_summary()
```

### Required CSV Columns

```csv
video_path,water_surface_elevation
videos/video1.mp4,525.4
videos/video2.mp4,526.1
```

### Output Units

- **English**: cfs (discharge), ft² (area), ft (elevation/depth)
- **Metric**: m³/s (discharge), m² (area), m (elevation/depth)

### Processing Time

- Typical: 2-10 minutes per video
- Depends on video length and resolution
- Add 2-5 minutes per video if `save_projects=True`

---

## Summary

Batch processing lets you analyze many videos automatically:

1. **Prepare** a template .ivy project with calibration and settings
2. **Create** a CSV file listing videos and water levels
3. **Run** the batch processing script
4. **Review** results in the summary CSV and output folders

This saves hours of manual work and ensures consistent analysis across all your videos.

Good luck with your analysis!
