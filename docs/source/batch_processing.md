# Batch Processing

**IVyTools** includes a headless batch processing capability that enables users to process multiple videos from the same fixed camera setup without using the graphical interface. This is particularly useful when measuring discharge at different water surface elevations using a permanent or semi-permanent camera installation.

## Overview

Batch processing allows users to:
- Process multiple videos from the same camera location
- Apply the same camera calibration, ground control points, and cross-section geometry to all videos
- Automatically compute velocities and discharge for each video with varying water surface elevations
- Generate individual `.ivy` project files and a comprehensive summary CSV for all measurements

The batch processor operates in "headless" mode, meaning it runs without a graphical user interface and can process videos automatically using command-line scripts.

## Prerequisites

Before running batch processing, you need:

1. **A scaffold `.ivy` project**: This is a template project containing all the fixed parameters for your camera installation:
   - Camera calibration (projection matrix or homography)
   - Ground control points (GCPs)
   - Cross-section geometry and bathymetry
   - STIV processing parameters (grid, search line length, angle range, etc.)
   - Measurement station information

2. **Video files**: Multiple videos from the same camera location, recorded at different water surface elevations

3. **Water surface elevation data**: Known WSE values for each video (from gage height or survey data)

### Creating a Scaffold Project

The scaffold project is created using the standard **IVyTools** GUI workflow:

1. Open **IVyTools** and process a single video through the complete workflow
2. Complete all calibration steps:
   - Extract frames from your video
   - Perform orthorectification (digitize GCPs, compute camera matrix)
   - Load cross-section geometry
   - Create the STIV processing grid
   - Configure STIV parameters
3. Save the project as a `.ivy` file
4. This saved project becomes your scaffold

**Important**: The scaffold contains all the parameters that remain constant across multiple measurements. Each video in the batch will inherit these settings and only update the water surface elevation and video-specific metadata.

## Batch Configuration CSV

The batch processor reads video information from a CSV file. This file specifies which videos to process and their associated metadata.

### CSV Format

The CSV file must contain the following columns (header row required):

| Column | Required | Description |
|--------|----------|-------------|
| `video_path` | Yes | Path to the video file (relative or absolute) |
| `water_surface_elevation` | Yes | Water surface elevation in the same units as the scaffold project |
| `measurement_date` | Yes | Date of measurement (YYYY-MM-DD format) |
| `measurement_number` | Yes | Unique measurement identifier number |
| `gage_height` | No | Gage height at time of measurement (typically same as WSE) |
| `start_time` | No | Video clip start time (HH:MM:SS format, e.g., 00:00:15) |
| `end_time` | No | Video clip end time (HH:MM:SS format, e.g., 00:00:20) |
| `comments` | No | User comments for this measurement |
| `alpha` | No | Alpha coefficient (surface to depth-averaged velocity conversion factor). Default: 0.85 |

### Example CSV File

```csv
video_path,water_surface_elevation,measurement_date,measurement_number,gage_height,start_time,end_time,comments,alpha
videos/measurement_001.mp4,318.21,2017-06-30,1,318.21,00:00:15,00:00:20,Low flow conditions,0.88
videos/measurement_002.mp4,318.21,2017-06-30,2,318.21,00:00:15,00:00:20,Repeat measurement,0.88
videos/measurement_003.mp4,318.60,2017-07-11,3,318.60,00:00:15,00:00:20,Medium flow,0.93
videos/measurement_004.mp4,320.15,2017-07-15,4,320.15,00:00:10,00:00:25,High flow conditions,0.90
```

### Notes on CSV Configuration

- **File paths**: Video paths can be relative to the script location or absolute paths
- **Time windows**: If `start_time` and `end_time` are specified, only frames between these times will be extracted and processed
- **Units**: The `water_surface_elevation` must use the same coordinate system and units as the GCPs in your scaffold project
  - If your scaffold uses English units (feet), provide WSE in feet
  - If your scaffold uses metric units (meters), provide WSE in meters
- **Alpha coefficient**: This parameter converts surface velocities to depth-averaged velocities. Common values range from 0.80 to 0.95 depending on flow conditions. The default is 0.85.

## Running Batch Processing

Batch processing is executed using a Python script. **IVyTools** includes an example script in the `examples` directory that demonstrates the workflow.

### Basic Usage

```python
from image_velocimetry_tools.batch_processor import BatchProcessor

# Initialize the batch processor with your scaffold project
processor = BatchProcessor(scaffold_path="path/to/scaffold.ivy")

# Load batch configuration from CSV
processor.load_batch_csv("path/to/batch_config.csv")

# Run batch processing
results = processor.run_batch(output_directory="path/to/output")

# Export summary to CSV
processor.export_results_csv(results, "path/to/output/batch_summary.csv")
```

### Example Script

The following example shows a complete batch processing script:

```python
import os
import logging
from pathlib import Path
from image_velocimetry_tools.batch_processor import BatchProcessor

def main():
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scaffold_path = os.path.join(script_dir, "scaffold_project.ivy")
    batch_csv = os.path.join(script_dir, "batch_config.csv")
    output_directory = os.path.join(script_dir, "output")

    # Create output directory
    os.makedirs(output_directory, exist_ok=True)

    # Configure logging
    log_file = os.path.join(output_directory, "batch_processing.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

    # Initialize batch processor
    logging.info("Initializing batch processor...")
    processor = BatchProcessor(scaffold_path=scaffold_path)

    # Load batch configuration
    logging.info("Loading batch configuration from CSV...")
    processor.load_batch_csv(batch_csv)

    # Run batch processing
    logging.info("Starting batch processing...")
    results = processor.run_batch(output_directory=output_directory)

    # Export summary CSV
    summary_csv = os.path.join(output_directory, "batch_summary.csv")
    processor.export_results_csv(results, summary_csv)
    logging.info(f"Batch processing complete. Summary saved to: {summary_csv}")

if __name__ == "__main__":
    main()
```

### Running from Command Line

To execute the batch processing script:

```bash
python run_batch_processing.py
```

The batch processor will:
1. Load the scaffold project
2. Read the batch configuration CSV
3. Process each video sequentially
4. Display progress and status messages
5. Generate output files

## Batch Processing Workflow

For each video in the batch configuration, the processor performs the following steps:

1. **Frame Extraction**: Extract frames from the video according to the specified time window and frame step
2. **Orthorectification**: Apply the camera transformation to rectify frames using the video's specific water surface elevation
3. **STIV Processing**: Compute velocity vectors using Space-Time Image Velocimetry on the rectified frames
4. **Discharge Calculation**: Compute discharge using the mid-section method and cross-section geometry
5. **Uncertainty Analysis**: Calculate measurement uncertainty using ISO 748 and IVE methods
6. **Project Creation**: Save all results as a complete `.ivy` project file

## Understanding the Outputs

Batch processing generates several output files:

### Individual `.ivy` Project Files

For each video processed, a complete `.ivy` project file is created in the output directory. Each project contains:

- **1-images folder**: Extracted frames (`f*.jpg`), orthorectified frames (`t*.jpg`), and space-time images (`sti_*.jpg`)
- **2-orthorectification folder**: Water surface elevation, pixel GSD, camera matrix, and GCP data (CSV files)
- **3-grids folder**: Reserved for future features
- **4-velocities folder**: STIV results including velocity components, magnitudes, and directions (`stiv_results.csv`)
- **5-discharge folder**: Cross-section data and discharge table (`discharge_table.csv`, `cross_section_ac3.mat`)
- **6-qaqc folder**: Reserved for future quality assurance features
- **project_data.json**: Complete project metadata and results

Each `.ivy` file can be opened in **IVyTools** for review, quality control, or further analysis.

**System Comment**: Every batch-created project includes an automatic System comment indicating:
- That the project was created by the batch processor
- The creation timestamp
- Source video filename
- Measurement date and number

### Batch Summary CSV

A comprehensive summary CSV file contains one row per video with the following columns:

| Column | Description |
|--------|-------------|
| `video_filename` | Name of the processed video file |
| `measurement_date` | Date of measurement |
| `measurement_number` | Measurement identifier |
| `wse_m` | Water surface elevation (in meters) |
| `gage_height` | Gage height value |
| `total_discharge` | Computed total discharge |
| `total_area` | Total wetted cross-sectional area |
| `iso_uncertainty` | ISO 748 uncertainty estimate (as decimal, e.g., 0.15 = 15%) |
| `ive_uncertainty` | IVE uncertainty estimate (as decimal) |
| `cross_section_width` | Width of the cross-section |
| `hydraulic_radius` | Hydraulic radius |
| `mean_velocity` | Mean depth-averaged velocity |
| `mean_surface_velocity` | Mean surface velocity across the section |
| `max_surface_velocity` | Maximum surface velocity measured |
| `max_depth` | Maximum depth in the cross-section |
| `alpha` | Alpha coefficient used for this measurement |
| `num_stations_used` | Number of stations used in discharge calculation |
| `num_stations_total` | Total number of stations in cross-section |
| `processing_status` | "success" or "failed" |
| `processing_duration_sec` | Time taken to process this video (seconds) |
| `error_message` | Error description if processing failed |
| `output_ivy_path` | Path to the generated `.ivy` project file |

This summary file can be imported into spreadsheet software or other data analysis tools for further review and analysis.

### Log File

A detailed log file (`batch_processing.log`) is created in the output directory containing:
- Processing steps for each video
- Water surface elevation conversions
- Frame extraction information
- STIV processing details
- Discharge calculation results
- Any warnings or errors encountered

This log is useful for troubleshooting and understanding exactly what the batch processor did.

## Tips and Best Practices

### Scaffold Project Preparation

- **Test your scaffold**: Before running a batch, manually process one video using the GUI to verify all parameters are correct
- **Use consistent units**: Ensure all GCP coordinates, water surface elevations, and bathymetry data use the same coordinate system and units
- **Document your setup**: Save notes about your camera location, GCP survey methods, and any assumptions made

### Batch Configuration

- **Start small**: Test with 2-3 videos before processing a large batch
- **Organize videos**: Keep all videos in a dedicated folder with clear, descriptive filenames
- **Check WSE values**: Double-check water surface elevation values before running the batch - errors here will invalidate all results
- **Use appropriate time windows**: Select video segments with stable flow conditions and good surface features

### Alpha Coefficient Selection

The alpha coefficient (α) converts surface velocities to depth-averaged velocities. Consider these guidelines:

- **Typical range**: 0.80 to 0.95
- **Low flow / rough bed**: Use higher values (0.90-0.95)
- **High flow / smooth bed**: Use lower values (0.80-0.85)
- **Default**: 0.85 is a reasonable default for most natural channels
- **Site-specific calibration**: If possible, calibrate α using direct discharge measurements (e.g., ADCP data)

### Quality Control

After batch processing:

1. **Review the batch summary CSV**: Check for any failed videos or unusual discharge values
2. **Examine the log file**: Look for warnings or errors that might indicate issues
3. **Spot-check individual projects**: Open several `.ivy` files in **IVyTools** to verify:
   - Orthorectification quality
   - Velocity vector fields
   - Discharge calculation results
4. **Compare uncertainty estimates**: High uncertainty (>20%) may indicate poor data quality or processing issues

### Processing Performance

- **Processing time**: Each video may take several minutes depending on video length, frame count, and computer performance
- **Disk space**: Each `.ivy` project can be 50-500 MB depending on the number of frames
- **Memory usage**: Processing large videos or many frames may require substantial RAM
- **Serial processing**: Videos are processed one at a time to avoid memory issues

## Troubleshooting

### Common Issues

**Issue**: "Module not found" errors when running the script
- **Solution**: Ensure **IVyTools** and all dependencies are installed in your Python environment. See [Setting up the development environment](./setup.html) for installation instructions.

**Issue**: Water surface elevations appear incorrect in output
- **Solution**: Check that WSE values in the CSV match the units of your scaffold project (feet vs. meters). The batch processor will automatically convert between English and SI units based on the scaffold settings.

**Issue**: Discharge calculation fails with "station mismatch" error
- **Solution**: Verify that the cross-section geometry in the scaffold project is complete and valid. The number of stations should match between the velocity grid and bathymetry data.

**Issue**: STIV processing produces poor velocity results
- **Solution**:
  - Ensure frames have good surface features (ripples, foam, debris)
  - Check that orthorectification is working correctly
  - Verify STIV parameters (search line length, angle range) in the scaffold are appropriate
  - Consider adjusting the time window in the batch CSV to select better video segments

**Issue**: Batch processing stops or crashes partway through
- **Solution**:
  - Check the log file for specific error messages
  - Reduce the number of frames being processed (increase frame step or shorten time windows)
  - Ensure adequate disk space is available
  - Try processing the problematic video individually to identify the issue

### Getting Help

If you encounter issues not covered here:

1. Check the log file for detailed error messages
2. Try processing the video manually through the **IVyTools** GUI to identify where the problem occurs
3. Verify that your scaffold project works correctly with at least one video before running the batch
4. Consult the other sections of this documentation for details on specific processing steps

## Advanced Usage

### Custom Processing Scripts

Advanced users can modify the batch processing script to:

- Add progress callbacks for custom status updates
- Implement email notifications when processing completes
- Filter or validate results programmatically
- Integrate with other data processing pipelines
- Customize output file naming and organization

### Parallel Processing

The current batch processor runs videos sequentially. For large batches, consider:

- Running multiple batch scripts simultaneously with different CSV files
- Splitting your batch CSV into smaller chunks
- Processing on multiple computers if available

Note: Each video processing job can use significant memory, so parallel processing should be done carefully to avoid system resource issues.

## Example Workflow

Here's a complete example workflow for processing a rating curve measurement campaign:

1. **Field work**:
   - Set up a permanent camera installation at your stream gage
   - Survey ground control points in the field of view
   - Survey the channel cross-section bathymetry
   - Record videos at various flow stages over several months

2. **Scaffold creation**:
   - Process one video through **IVyTools** GUI
   - Calibrate camera using surveyed GCPs
   - Load cross-section geometry
   - Create STIV grid and configure parameters
   - Test processing to verify good results
   - Save as `rating_curve_scaffold.ivy`

3. **Batch preparation**:
   - Organize all videos in a `videos` folder
   - Create a CSV file with WSE for each video
   - Record water surface elevations from gage data
   - Include appropriate alpha coefficients

4. **Batch processing**:
   - Run the batch processing script
   - Monitor progress via console output
   - Review log file for any issues

5. **Quality control**:
   - Open batch summary CSV in spreadsheet software
   - Plot discharge vs. stage to visualize rating curve
   - Identify and investigate any outliers
   - Open selected `.ivy` files for detailed review

6. **Analysis**:
   - Fit a rating curve to discharge and stage data
   - Use uncertainty estimates to compute confidence bands
   - Compare to historical ratings or other discharge measurements
   - Generate final report and archive data

This workflow enables rapid development of stage-discharge relationships using image velocimetry, particularly useful for sites where traditional rating curves are difficult or expensive to develop.
