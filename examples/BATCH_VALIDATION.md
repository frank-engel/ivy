# Batch Processing Validation Guide

This document describes the expected behavior for the batch processing system based on test data from the original implementation.

## Test Data

### Input Files
- **Scaffold Project**: `scaffold_project.ivy` (87 MB)
  - Contains camera calibration, GCPs, cross-section geometry
  - Fixed camera setup for Boneyard Creek
  - Uses homography rectification method

- **Batch Configuration**: `batch_boneyard.csv`
  - 3 test videos with varying water surface elevations
  - Different alpha coefficients per video
  - Time windowing (15-20 second clips)

- **Videos**:
  1. `03337000_bullet_20170630-115500.mp4` (44 MB)
  2. `03337000_bullet_20170630-120000.mp4` (39 MB)
  3. `03337000_bullet_20170711-103500.mp4` (67 MB)

### Expected Output

The batch processor should generate:

1. **Individual .ivy project files** (one per video)
   - Complete project with frames, STIV results, discharge calculations
   - Preserved metadata (date, measurement number, comments)
   - Video-specific water surface elevation applied

2. **Batch summary CSV** with the following results:

| Video | WSE (m) | Discharge (m³/s) | Area (m²) | Mean Vel (m/s) | Alpha | Stations |
|-------|---------|------------------|-----------|----------------|-------|----------|
| 115500 | 318.211 | 2.1704 | 4.6755 | 0.4642 | 0.880 | 27/27 |
| 120000 | 318.211 | 3.1995 | 4.6755 | 0.6843 | 0.880 | 27/27 |
| 103500 | 318.600 | 3.4487 | 5.3869 | 0.6402 | 0.930 | 27/27 |

### Validation Criteria

#### Accuracy
- Discharge values should match within **±0.01 m³/s** (0.5% tolerance)
- Area values should match within **±0.01 m²**
- Velocity values should match within **±0.01 m/s**

#### Functional Requirements
- Process all 3 videos without errors
- Handle different WSE values correctly (affects area and discharge)
- Apply video-specific alpha coefficients
- Extract correct time windows (15-20 seconds)
- Generate 27 cross-section stations for each video
- Calculate ISO 748 uncertainty

#### Output Requirements
- Create valid .ivy project files that can be opened in GUI
- Generate CSV summary with all expected columns
- Preserve all metadata (dates, measurement numbers, comments)

## Key Differences from Original Implementation

### Architecture
- **Original**: Direct calls to core modules from batch_processing_helpers.py
- **New**: Uses MVP service layer (DischargeService, STIVService, etc.)
- **Benefit**: Shared logic between GUI and batch, better testability

### Scaffold Concept
The current implementation uses a "scaffold" .ivy project containing:
- Camera calibration (homography or camera matrix)
- GCPs for georeferencing
- Cross-section bathymetry (AC3 .mat file)
- STIV processing parameters

**Future Enhancement**: This could be replaced with a simpler JSON configuration file:
```json
{
  "camera_calibration": {...},
  "gcps": [...],
  "cross_section": "path/to/bathymetry.mat",
  "stiv_params": {...}
}
```

Benefits of JSON config:
- Smaller file size (KB vs 87 MB)
- Human-readable and editable
- Version control friendly
- No need to create a full .ivy project as template

## Testing Workflow

### Phase 1 Testing (Post-Service Enhancement)
Run integration test after each service is enhanced:
```bash
# Test single service in isolation
python -m pytest tests/integration/test_video_service_batch.py
python -m pytest tests/integration/test_discharge_service_batch.py
```

### Phase 3 Testing (Post-BatchOrchestrator)
Test full workflow:
```bash
# Test single video processing
python -m pytest tests/integration/test_batch_orchestrator.py::test_single_video
```

### Phase 4 Testing (Post-Python API)
Test API interface:
```python
from ivy.batch import process_video

result = process_video(
    video_path='examples/videos/03337000_bullet_20170630-115500.mp4',
    scaffold_path='examples/scaffold_project.ivy',
    water_surface_elevation=318.211,
    alpha=0.88,
    start_time=15,
    end_time=20,
    output_dir='test_output/'
)

# Validate result
assert result.success
assert abs(result.total_discharge - 2.1704) < 0.01
assert abs(result.total_area - 4.6755) < 0.01
```

### Phase 5 Testing (Post-CLI)
Test command-line interface:
```bash
# Single video
ivy-batch process examples/videos/03337000_bullet_20170630-115500.mp4 \
    --scaffold examples/scaffold_project.ivy \
    --wse 318.211 \
    --alpha 0.88 \
    --start-time 15 \
    --end-time 20 \
    --output test_output/

# Full batch
ivy-batch batch examples/batch_boneyard.csv \
    --scaffold examples/scaffold_project.ivy \
    --output test_output/
```

### Phase 6 Testing (Final Integration)
Run complete batch and compare output:
```bash
# Process all 3 videos
ivy-batch batch examples/batch_boneyard.csv \
    --scaffold examples/scaffold_project.ivy \
    --output validation_output/

# Compare results
diff validation_output/batch_discharge_summary.csv examples/output/batch_discharge_summary.csv
```

## Notes for Manual QA

When testing each phase, verify:

1. **No crashes or unhandled exceptions**
2. **Progress reporting works** (if implemented)
3. **Error messages are clear** when things go wrong
4. **Output files are valid** (can open .ivy files in GUI)
5. **Performance is reasonable** (each video ~15-20 seconds)
6. **Memory usage is stable** (no leaks during batch processing)

## Known Limitations

- Requires fixed camera setup (scaffold concept)
- Only supports homography or camera matrix rectification
- Assumes cross-section data is available in AC3 format
- No parallel processing (processes videos sequentially)
- No resume capability if batch is interrupted

These limitations could be addressed in future enhancements.
