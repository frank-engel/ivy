# IVyTools Release Notes - Batch Processing Feature

## Version 1.0.0.2

**Release Date**: November 2023

This release introduces a major new feature: **batch processing support** for automated image velocimetry analysis across multiple videos. The batch processing system enables researchers to process large datasets efficiently with minimal manual intervention.

---

## 🎉 Major Features

### Batch Processing System

A complete batch processing framework for analyzing multiple videos in a single operation:

- **Service Layer Architecture**: Robust backend with `BatchProcessor`, `ScaffoldManager`, and `BatchJobRunner`
- **Comprehensive Error Handling**: Graceful failure handling with detailed error reporting and optional stop-on-error mode
- **Progress Tracking**: Real-time progress callbacks for monitoring batch operations
- **Result Aggregation**: Automatic collection and summarization of all job results
- **Project Archiving**: Optional `.ivy` project file generation for each processed video

**Test Coverage**: 108 passing unit tests ensuring reliability and correctness

### Python API for Batch Processing

A clean, user-friendly Python API for programmatic batch processing:

```python
from image_velocimetry_tools.api import run_batch_processing

results = run_batch_processing(
    scaffold_project="scaffold.ivy",
    batch_csv="videos.csv",
    output_folder="results/",
    stop_on_error=False,
    save_projects=True
)

print(f"Processed {results.total_jobs} videos")
print(f"Success rate: {results.success_rate:.1f}%")
```

**Key Components**:
- `run_batch_processing()` - Main entry point function
- `BatchResults` - Container with summary statistics and job access
- `JobResult` - Individual job result with metadata and outputs

**Features**:
- Simple function-based interface
- Optional progress callbacks for monitoring
- Comprehensive result objects with statistics
- Clean imports from `image_velocimetry_tools.api`

### Comprehensive Documentation

New user-facing documentation designed for non-programmers:

- **`docs/batch_processing_guide.md`** (400+ lines)
  - Step-by-step setup instructions
  - CSV format specification with examples
  - Running batch jobs from Python
  - Understanding and interpreting results
  - Troubleshooting common issues
  - Best practices and tips

- **`examples/batch_processing_example.py`**
  - Three usage patterns: basic, with progress, advanced analysis
  - Real examples using `batch_test_data`
  - Ready-to-run demonstration code

- **`image_velocimetry_tools/api/README.md`**
  - Quick reference for API usage
  - Parameter descriptions
  - Return value documentation

---

## 🔧 Installation & Packaging Improvements

### Python 3.11 Support

- **Requirement**: Added `python_requires='>=3.11'` to setup.py
- **Compatibility**: Full support for Python 3.11+ (required by areacomp3 and qrev dependencies)
- **Migration**: Users on Python 3.8 must upgrade to Python 3.11+

### Dependency Management Overhaul

Created dual requirements system for better dependency management:

- **`requirements.txt`**: Exact frozen versions for reproducible development environment
- **`install_requires.txt`**: Relaxed version constraints for distribution and end-user installation

**Key dependency fixes**:
- `matplotlib>=3.6.3` (was >=3.3.3) - Matches areacomp3 requirement
- `numba>=0.58.0,<0.59.0` - Python 3.11 compatible, works with areacomp3's JIT decorators
- Removed `ffmpeg-python` - Not actually used (only ffmpeg binary required)
- Removed unused packages: `scikit-learn`, `openpyxl`, `PyYAML`, `xmltodict`, `utm`, `simplekml`, `tabulate`, `networkx`, `PyWavelets`

### Installation Testing Tools

New tools for validating package installation:

- **`test_installation.py`**: Quick validation script testing 16 import checks
- **`INSTALL_TESTING.md`**: Comprehensive installation testing procedures
- **`PACKAGING.md`**: Complete guide for version testing and PyPI publishing
- **`PIP_INSTALL_ANALYSIS.md`**: Analysis of pip installation issues and solutions

### Setup.py Enhancements

- Added `image_velocimetry_tools.api` package to distribution
- Added Python 3.11 classifier for PyPI metadata
- Switched to reading `install_requires.txt` instead of `requirements.txt`
- Clear separation between dev and runtime dependencies

---

## 🐛 Bug Fixes

### Numba/AreaComp3 Compatibility

**Issue**: AreaComp3's `@numba.jit` decorators incompatible with modern numba versions (0.59+)

**Impact**: GUI failed to load cross-section data with:
```
TypingError: Cannot determine Numba type of <class 'areacomp.gui.areasurvey.AreaSurvey'>
```

**Fix**:
- Pinned numba to 0.58.x (supports Python 3.11, compatible with areacomp3)
- Added deprecation warning suppression for cleaner console output
- Documented issue in `AREACOMP_NUMBA_BUG.md` for upstream fix

**Related Commits**:
- `3a42622` - Pin numba to 0.58.x for compatibility
- `2c75157` - Suppress numba deprecation warnings
- `1df8f46` - Add detailed error logging for diagnosis
- `e0b2cdf` - Document bug for AreaComp developers

### PyQt5 Type Compatibility

**Issue**: `painter.drawLine()` failed with numpy.float64 coordinate values

**Error**:
```
TypeError: arguments did not match any overloaded call:
  drawLine(self, x1: int, y1: int, x2: int, y2: int): argument 1 has unexpected type 'float'
```

**Fix**: Convert numpy coordinates to Python integers for PyQt5's `drawLine(int, int, int, int)` overload

**Location**: `image_velocimetry_tools/gui/stiv_processor.py:926`

**Related Commits**:
- `69c3d60` - Convert coordinates to int for PyQt5 compatibility
- `2c75157` - Initial float conversion attempt

### NumPy Typing Import Error

**Issue**: Import error on package initialization:
```
cannot import name '_8Bit' from 'numpy.typing'
```

**Cause**: `_8Bit` and `_64Bit` types don't exist in numpy 1.26.2, were imported but never used

**Fix**: Removed unused imports from `image_velocimetry_tools/stiv.py`

**Related Commit**: `9d25585`

### Dependency Resolution Failures

**Issue**: `pip install .` failed with `ResolutionTooDeep` error after 200,000 rounds

**Cause**: 127 exact version pins (`==`) in requirements.txt created impossible constraint solving

**Fix**:
- Created `install_requires.txt` with relaxed constraints
- Modified setup.py to read from new file
- Resolved matplotlib and statsmodels version conflicts

**Related Commits**:
- `d296d25`, `84924ed` - Resolve dependency resolution
- `d98e5ac` - Update matplotlib for areacomp3
- `26bae73` - Relax statsmodels constraint
- `20dff2e` - Remove ffmpeg-python

### GUI Error Handling Improvements

**Issue**: Generic error messages hid root cause of failures

**Fix**: Added detailed exception logging to AreaComp load failures with exception details in error dialogs

**Related Commit**: `1df8f46`

---

## 📊 Integration Testing

Comprehensive integration testing completed using `batch_test_data`:

- ✅ Batch processor processes multiple videos successfully
- ✅ Cross-section geometry loaded from AreaComp3 files
- ✅ Discharge calculations match expected USGS values
- ✅ Uncertainty calculations implemented and validated
- ✅ Results output in correct display units (SI/English)
- ✅ Optional `.ivy` project archives generated correctly
- ✅ Batch summary statistics computed accurately

**Test Data**: Real USGS cross-section and video data
**Test Scenarios**: Multi-video batch processing with various configurations

---

## 🔄 Breaking Changes

### Python Version Requirement

- **Previous**: Python 3.8 supported
- **New**: Python 3.11+ required

**Reason**: areacomp3 and qrev dependencies require Python ~3.11

**Migration**: Users must upgrade to Python 3.11+ and recreate virtual environments

### Dependency Changes

- **matplotlib**: Now requires >=3.6.3 (was >=3.3.3)
- **numba**: Explicitly pinned to 0.58.x (was transitive dependency)
- **Removed packages**: Several unused packages removed from install_requires

**Migration**: Run `pip install -e .` to update dependencies

---

## 📁 New Files

### API & Examples
- `image_velocimetry_tools/api/__init__.py`
- `image_velocimetry_tools/api/batch_api.py`
- `image_velocimetry_tools/api/README.md`
- `examples/batch_processing_example.py`

### Documentation
- `docs/batch_processing_guide.md`
- `AREACOMP_NUMBA_BUG.md`
- `PACKAGING.md`
- `INSTALL_TESTING.md`
- `PIP_INSTALL_ANALYSIS.md`

### Installation & Testing
- `install_requires.txt`
- `test_installation.py`

---

## 📝 Modified Files

### Core Application
- `setup.py` - Python 3.11 requirement, new API package, install_requires.txt
- `image_velocimetry_tools/stiv.py` - Removed unused numpy.typing imports
- `image_velocimetry_tools/gui/ivy.py` - Numba warning suppression
- `image_velocimetry_tools/gui/stiv_processor.py` - PyQt5 type fixes
- `image_velocimetry_tools/gui/xsgeometry.py` - Enhanced error logging
- `.gitignore` - Enabled examples/ folder

### Dependencies
- `requirements.txt` - Updated for Python 3.11 compatibility
- `install_requires.txt` - Created and refined for distribution

---

## 🚀 Usage Examples

### Basic Batch Processing

```python
from image_velocimetry_tools.api import run_batch_processing

results = run_batch_processing(
    scaffold_project="my_scaffold.ivy",
    batch_csv="videos.csv",
    output_folder="batch_output/"
)

print(f"Total: {results.total_jobs}")
print(f"Successful: {results.successful_jobs}")
print(f"Failed: {results.failed_jobs}")
```

### With Progress Monitoring

```python
def progress_callback(current, total, video_name, status):
    print(f"[{current}/{total}] {video_name}: {status}")

results = run_batch_processing(
    scaffold_project="scaffold.ivy",
    batch_csv="batch.csv",
    output_folder="output/",
    progress_callback=progress_callback
)
```

### Advanced Result Analysis

```python
results = run_batch_processing(...)

for job in results.successful_jobs_list:
    print(f"{job.video_file}: Q={job.discharge:.2f} m³/s")
    print(f"  Uncertainty: ±{job.uncertainty:.1f}%")
    print(f"  Duration: {job.duration:.1f}s")

if results.failed_jobs_list:
    print("\nFailed jobs:")
    for job in results.failed_jobs_list:
        print(f"  {job.video_file}: {job.error_message}")
```

---

## 🔍 Testing

### Unit Tests
- **Total**: 108 tests passing
- **Coverage**: Batch processor, scaffold manager, job runner, models
- **Validation**: Input validation, error handling, result aggregation

### Integration Tests
- Real USGS video and cross-section data
- Full end-to-end batch processing workflow
- Discharge calculation validation
- Project archiving verification

### Installation Tests
- 16 import checks covering all modules
- API functionality verification
- Dependency compatibility validation

---

## 🙏 Acknowledgments

- **AreaComp3 Team** (USGS) - Cross-section analysis integration
- **QRev Team** (USGS) - ADCP data processing support

---

## 📞 Support & Feedback

- **Repository**: https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy
- **Issues**: Report bugs via the repository issue tracker
- **Contact**: Frank L. Engel (fengel@usgs.gov)

---

## 🔜 Future Enhancements

Potential future improvements being considered:

- GUI interface for batch processing
- Resume capability for interrupted batch jobs
- Parallel processing for multiple videos
- Enhanced progress visualization
- Batch job templates and presets
- Cloud storage integration for large datasets

---

## 📜 License

GNU General Public License v3.0

---

**Full Changelog**: All commits included in branch `claude/implement-batch-processor-01HcpyFffPYUL4fwoiip7eSd`
