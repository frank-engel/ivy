# IVyTools Installation Guide

## Requirements

- **Python 3.11+** (Required by areacomp3 and qrev dependencies)
- FFmpeg binary (must be in PATH or `./bin/` directory)

## Installation Steps

### 1. Create Virtual Environment

```bash
# Create Python 3.11 environment
python3.11 -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 2. Install IVyTools

```bash
pip install -e .
```

## Expected Dependency Warnings

During installation, you will see dependency conflict warnings. **These are expected and can be safely ignored**:

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed...
areacomp3 1.4.0 requires numba~=0.61.0, but you have numba 0.58.1 which is incompatible.
```

**Why?**
- IVyTools explicitly pins `numba>=0.58.0,<0.59.0`
- This overrides areacomp3's requirement of `numba~=0.61.0`
- Numba 0.61+ has stricter JIT compilation rules that cause runtime errors with areacomp3's code
- Numba 0.58.x works correctly with both IVyTools and areacomp3 despite the version mismatch

## Verification

After installation, verify everything works:

```bash
# Test imports
python -c "from image_velocimetry_tools.api import run_batch_processing; print('✓ API import successful')"

# Check numba version (should be 0.58.x)
python -c "import numba; print(f'numba version: {numba.__version__}')"

# Run GUI (if applicable)
python -m image_velocimetry_tools.gui.ivy
```

## Troubleshooting

### "Cannot determine Numba type" Error

If you see this error when using the GUI:
```
numba.core.errors.TypingError: Cannot determine Numba type of <class 'areacomp.gui.areasurvey.AreaSurvey'>
```

Your numba version is too new. Downgrade:
```bash
pip install "numba>=0.58.0,<0.59.0" --force-reinstall
```

### NumPy Typing Import Error

If you see:
```
cannot import name '_8Bit' from 'numpy.typing'
```

This should be fixed in the current version. If you still see it, please report an issue.

### PyQt5 DrawLine Type Error

If you see:
```
TypeError: arguments did not match any overloaded call: drawLine(...)
```

This should be fixed in the current version. If you still see it, please report an issue.

## Development Installation

For development with testing and linting tools:

```bash
# Install in editable mode
pip install -e .

# Install development dependencies (includes pytest, black, flake8, etc.)
pip install -r requirements-dev.txt
```

**Note**: The `requirements-dev.txt` file overrides qrev's ancient `py==1.8.0` requirement with a modern version compatible with Python 3.11 and pytest. You'll see dependency warnings - these are expected and safe to ignore.

### Running Tests

After installing dev dependencies:

```bash
# Run all tests
pytest

# Run specific test directory
pytest image_velocimetry_tools/tests/test_batch

# Run integration tests (requires batch_test_data)
pytest image_velocimetry_tools/tests/test_integration

# Run with coverage
pytest --cov=image_velocimetry_tools --cov-report=html

# Run with verbose output
pytest -v
```

## Rebuilding Dependencies

To regenerate `requirements.txt` with exact versions:

```bash
pip uninstall image_velocimetry_tools
python setup.py sdist
pip install image_velocimetry_tools
pip freeze > requirements.txt
```

## Platform-Specific Notes

### Windows
- Use PowerShell or Command Prompt
- FFmpeg must be in PATH or `.\bin\` directory
- Visual C++ redistributables may be required

### Linux
- Install FFmpeg via package manager: `sudo apt install ffmpeg`
- May need `python3.11-dev` for building some packages

### macOS
- Install FFmpeg via Homebrew: `brew install ffmpeg`
- May need Xcode Command Line Tools
