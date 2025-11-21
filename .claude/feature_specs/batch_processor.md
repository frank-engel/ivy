# Batch Processor Feature Specification

## Goal
Enable users to process multiple river discharge analysis jobs with a single command, using a scaffold project template and a CSV defining per-job variations.

## User Story
As a researcher, I want to analyze 100 videos with similar parameters but different video files and ROIs, so I can automate my workflow instead of clicking through the UI 100 times.

## Input Files

### 1. Scaffold Project File (scaffold_project.ivy)
Contains all common configuration:
- Video processing parameters (frame extractions specs, ffmpeg effects, camera distortion params)
- rectification method and information (for batch, always will be "camera_matrix"), ground control points and their point correspondences
- the cross-section geometry file loaded internally by AreaComp, and its pixel location in the rectified image
- Processing grid (for batch, this will always be points along the cross-section line), including masks that exclude portions of the line
- Space-time Image Velocimetry parameters (see stiv module, two_dimensional_stiv_exhaustive)
- Comments about the measurement that are in common for all measurements (site name, id, etc.)
- **Excludes**: for batch, even though these may be included, they are overriden by the batch CSV data file. 
  - water surface elevation: this will change for each measurement
  - STIV results
  - discharge point data and summary data

    
### 2. Batch CSV File
Each row = one job. See `/batch_test_data/inputs/boneyard_batch.csv`. Required columns:
- `video_path`: full path the video for this particular job
- `water_surface_elevation`: water surface elevation in the same units (and coords) as the ground control points (specified in the scaffold as `display_units`)
- `start_time`, `end_time`: Time in ss or in hh:mm:ss.s clipping the video to look at for frame extraction. No values means take the entire video

Optional columns (override scaffold):
- `alpha`: Coeffecient that corrects surface velocity to mean channel velocity. If missing, 0.85 should be the default
- `measurement_number`: reference number for the measurement
- `measurement_date`: date of the measurement
- `measurement_time`: time of the measurement
- `gage_height`: reference to the USGS gage height, which may or may not correspond to the water surface elevation
- `comments`: job specific comments
- Any other parameter that can vary per job

## Output
- One result file per job (named using `job_id`)
- Batch summary CSV with success/failure status, processing results
- Error log for failed jobs, stored with the other output

## API Requirements

### Programmatic API
```python
from image_velocimetry_tools.batch import BatchProcessor

processor = BatchProcessor(
    scaffold_path="scaffold.json",
    batch_csv="jobs.csv",
    output_dir="./results",
    # ... others as needed
)

results = processor.run()
for result in results:
    print(f"{result.job_id}: {result.status}")
```

## Architecture Components

### New Classes Needed
1. **Model**: `BatchJob` - Represents a single job in the batch
2. **Model**: `BatchConfig` - Configuration for the batch run
3. **Service**: `BatchProcessor` - Orchestrates batch execution
4. **Service**: `ScaffoldLoader` - Loads and validates scaffold
5. **Service**: `BatchCSVParser` - Parses and validates CSV
6. **Service**: `JobExecutor` - Executes individual jobs
7. **Controller**: `BatchController` - CLI entry point (potentially future feature)

### Integration with Existing Code
- Reuse existing services as much as possible, expanding as needed, without breaking normal single video USer Workflow (via UI client app)

## Error Handling
- Validate scaffold and CSV before starting any jobs
- Continue processing on individual job failures
- Comprehensive error reporting
- Option to stop on first failure

## Success Criteria
- [ ] Clear error messages for malformed inputs
- [ ] All job results captured even if some fail
- [ ] Existing UI workflow unaffected
- [ ] API usable from external scripts

## Non-Goals (for v1)
- GUI for batch creation
- Resume partially completed batches
- Distributed processing across machines
- Specific parallel processing (beyond what existing modules already do)
```