# Integration Tests for Batch Processing

This directory contains integration tests for the batch processing functionality added in Phase 2.

## Overview

These tests validate that the batch-compatible service methods work correctly with real data:
- **test_video_service_batch.py** - Tests VideoService.get_video_metadata() and extract_frames()
- **test_project_service_batch.py** - Tests ProjectService.load_scaffold_configuration()
- **test_batch_workflow.py** - Tests complete workflow combining multiple services

## Test Data

Tests use real data from the `examples/` directory:
- `examples/scaffold_project.ivy` (87 MB) - Template project with camera calibration, GCPs, cross-section
- `examples/videos/03337000_bullet_20170630-120000.mp4` (39 MB) - Test video
- Other videos in `examples/videos/` for additional coverage

## Running Tests

### Run all integration tests:
```bash
pytest image_velocimetry_tools/tests/test_integration/ -v
```

### Run specific test file:
```bash
pytest image_velocimetry_tools/tests/test_integration/test_video_service_batch.py -v
```

### Run specific test:
```bash
pytest image_velocimetry_tools/tests/test_integration/test_video_service_batch.py::TestVideoServiceBatch::test_get_video_metadata -v
```

### Run with output (see print statements):
```bash
pytest image_velocimetry_tools/tests/test_integration/ -v -s
```

### Run with coverage:
```bash
pytest image_velocimetry_tools/tests/test_integration/ --cov=image_velocimetry_tools.services --cov-report=html
```

## Dependencies

Tests require:
- **pytest** - Testing framework
- **ffmpeg** and **ffprobe** - Must be installed and in PATH
- **Test data** - Files in examples/ directory
- **Standard dependencies** - NumPy, PIL, skimage, etc.

### Skipped Tests

Tests automatically skip if required files are missing:
```python
if not os.path.exists(test_video_path):
    pytest.skip(f"Test video not found: {test_video_path}")
```

This allows tests to pass in environments without the full test dataset.

## What's Tested

### VideoService
✅ `get_video_metadata()` - Extract video properties with ffprobe
✅ `extract_frames()` - Frame extraction with time windows
✅ Frame extraction with frame stepping
✅ Progress callback functionality
✅ Input validation and error handling

### ProjectService
✅ `load_scaffold_configuration()` - Load scaffold .ivy project
✅ Scaffold validation (required fields)
✅ Rectification parameter extraction (homography, camera matrix, scale)
✅ Temporary directory management
✅ Cleanup on errors

### Workflow Integration
✅ Load scaffold → Extract frames (complete chain)
✅ Validate scaffold parameters are usable
✅ Validate video metadata sufficient for STIV
✅ Compatibility between services

## Expected Results

All tests should pass if:
1. Test data files exist in `examples/` directory
2. ffmpeg and ffprobe are installed
3. All Python dependencies are available

### Example Output
```
test_video_service_batch.py::TestVideoServiceBatch::test_get_video_metadata PASSED
test_video_service_batch.py::TestVideoServiceBatch::test_extract_frames_basic PASSED
test_project_service_batch.py::TestProjectServiceBatch::test_load_scaffold_configuration PASSED
test_batch_workflow.py::TestBatchWorkflow::test_load_scaffold_and_extract_frames PASSED
```

## Troubleshooting

### Tests are skipped
- Verify test data exists in `examples/` directory
- Check that `examples/scaffold_project.ivy` and test videos are present

### ffmpeg/ffprobe errors
- Install ffmpeg: `sudo apt-get install ffmpeg` (Linux) or `brew install ffmpeg` (Mac)
- Verify installation: `ffmpeg -version` and `ffprobe -version`

### Import errors
- Install dependencies: `pip install -r requirements.txt`
- Verify image_velocimetry_tools package is in PYTHONPATH

### Memory errors
- Some tests extract frames which can use significant memory
- Tests use `max_frames` limits to keep memory usage reasonable
- Temporary files are cleaned up automatically via fixtures

## Test Fixtures

Tests use pytest fixtures for resource management:
- `video_service()` - Create VideoService instance
- `project_service()` - Create ProjectService instance
- `temp_output_dir()` - Temporary directory with automatic cleanup
- `test_video_path()` - Path to test video
- `scaffold_path()` - Path to scaffold project

Fixtures with cleanup ensure temporary files are removed after tests complete.

## Future Tests

Additional tests to add:
- **STIVService.process_stiv()** - Full STIV workflow test (requires more setup)
- **DischargeService.process_discharge_workflow()** - Discharge calculation test (requires cross-section data)
- **OrthorectificationService.rectify_frames_batch()** - Frame rectification test (requires rectification params)
- **Performance tests** - Measure processing time for benchmarking
- **Error recovery tests** - Test handling of corrupted files, missing data, etc.

## Notes

- Tests are designed for CI/CD environments
- Tests clean up after themselves (no leftover files)
- Tests provide detailed output for debugging
- Tests validate both success and error cases
- Tests use realistic data matching production use cases
