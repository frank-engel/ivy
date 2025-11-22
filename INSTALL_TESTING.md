# Installation Testing Guide

Quick guide for testing the package installation after fixing dependency issues.

## What Changed

### 1. Created `install_requires.txt`
- Relaxed version constraints (using `>=` and `<` instead of `==`)
- Includes all runtime dependencies
- UTF-8 encoded (compatible with all platforms)
- Suitable for PyPI distribution

### 2. Updated `setup.py`
- Now reads from `install_requires.txt` instead of `requirements.txt`
- Added new package modules (batch, services, api)
- Fixed encoding to UTF-8

### 3. Kept `requirements.txt` unchanged
- Still contains exact frozen versions for reproducible dev environment
- Use this with `pip install -r requirements.txt` for exact dev setup
- UTF-16 encoded (your current format)

## Quick Test

### Test 1: Fresh Install
```bash
# Create fresh virtual environment
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate

# Install package
pip install .

# Run test script
python test_installation.py

# Cleanup
deactivate
rm -rf test_env  # Windows: rmdir /s test_env
```

### Test 2: Build Distribution
```bash
# Install build tools
pip install build

# Build package
python -m build

# You should see:
# dist/image_velocimetry_tools-1.0.0.2-py3-none-any.whl
# dist/image_velocimetry_tools-1.0.0.2.tar.gz
```

### Test 3: Install from Wheel
```bash
# Create fresh environment
python -m venv wheel_test
source wheel_test/bin/activate

# Install from built wheel
pip install dist/image_velocimetry_tools-1.0.0.2-py3-none-any.whl

# Test import
python -c "from image_velocimetry_tools.api import run_batch_processing; print('Success!')"

# Cleanup
deactivate
```

## Expected Results

### Success
```
Testing image_velocimetry_tools installation
============================================================

Import Tests:
------------------------------------------------------------
✓ Core package
✓ GUI components
✓ Batch processing
✓ Batch models
✓ Services layer
✓ Public API
✓ Batch API
✓ NumPy
✓ Pandas
✓ SciPy
✓ Matplotlib
✓ OpenCV
✓ scikit-image
✓ PyQt5
✓ HDF5 support

API Tests:
------------------------------------------------------------
✓ API imports successful
  - run_batch_processing: True
  - BatchResults: True
  - JobResult: True

============================================================
Results: 16/16 tests passed
✓ All tests passed! Installation successful.
```

### If You See Errors

**Missing dependency:**
```
✗ OpenCV
  Error: No module named 'cv2'
```
→ Add the package to `install_requires.txt`

**Version conflict:**
```
ERROR: Cannot install because these package versions have conflicting dependencies:
  package-a requires package-b>=2.0
  package-c requires package-b<2.0
```
→ Adjust version bounds in `install_requires.txt` to find compatible range

**Build failure:**
```
ERROR: Could not build wheels for some-package
```
→ May need system dependencies (e.g., compilers for native code)

## Troubleshooting

### areacomp3 Installation Issues

The `areacomp3` package is a git dependency. If it causes problems:

**Option 1: Skip it for testing**
```bash
# Temporarily comment out in install_requires.txt
# areacomp3 @ git+https://code.usgs.gov/RIVRS/areacomp.git@master

pip install .
```

**Option 2: Install manually first**
```bash
pip install git+https://code.usgs.gov/RIVRS/areacomp.git@master
pip install .
```

**Option 3: Make it optional** (for PyPI)
Edit `setup.py`:
```python
extras_require={
    'usgs': ['areacomp3 @ git+https://code.usgs.gov/RIVRS/areacomp.git@master'],
}
```

Then users install with: `pip install image_velocimetry_tools[usgs]`

## Next Steps

Once installation works:

1. **Run unit tests**
   ```bash
   pytest tests/
   ```

2. **Test batch processing**
   ```bash
   python examples/batch_processing_example.py
   ```

3. **Test on different Python versions**
   ```bash
   # Python 3.8
   python3.8 -m venv test38
   source test38/bin/activate
   pip install .
   python test_installation.py
   deactivate

   # Python 3.9
   python3.9 -m venv test39
   source test39/bin/activate
   pip install .
   python test_installation.py
   deactivate
   ```

4. **Prepare for PyPI** (see PACKAGING.md)
   ```bash
   python -m build
   twine check dist/*
   twine upload --repository testpypi dist/*
   ```

## Files Created

- `install_requires.txt` - Distribution dependencies (relaxed constraints)
- `test_installation.py` - Quick installation validation script
- `PACKAGING.md` - Complete guide for version testing and PyPI publishing
- `INSTALL_TESTING.md` - This file

## Files Modified

- `setup.py` - Now uses `install_requires.txt` and includes new packages

## Files Unchanged

- `requirements.txt` - Still has exact frozen versions for dev environment
