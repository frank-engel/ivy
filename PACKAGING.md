# Packaging and PyPI Publishing Guide

This document explains the dual-requirements approach and how to test/publish to PyPI.

## Dual Requirements System

### `requirements.txt` (Development Environment)
- **Purpose**: Exact reproducible development environment
- **Format**: Frozen with `==` version pins (from `pip freeze`)
- **Encoding**: UTF-16
- **Usage**: `pip install -r requirements.txt`
- **When to update**: After testing a working environment
- **Includes**: All packages (runtime + dev tools + transitive dependencies)

### `install_requires.txt` (Distribution)
- **Purpose**: Flexible installation for end users
- **Format**: Relaxed constraints with `>=` and `<` bounds
- **Encoding**: UTF-8
- **Usage**: Automatically used by `setup.py`
- **When to update**: When adding/removing runtime dependencies
- **Includes**: Only direct runtime dependencies

## Testing Version Compatibility

### 1. Quick Test in Fresh Virtual Environment

```bash
# Create fresh environment
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate

# Test installation from source
pip install .

# Test import
python -c "from image_velocimetry_tools.api import run_batch_processing; print('OK')"

# Run tests
pytest tests/

# Cleanup
deactivate
rm -rf test_env
```

### 2. Using pip-tools for Dependency Management

Install pip-tools:
```bash
pip install pip-tools
```

Generate lockfile from install_requires.txt:
```bash
pip-compile install_requires.txt --output-file install_requires.lock
```

This will resolve all versions and show conflicts.

### 3. Using tox for Multi-Version Testing

Create `tox.ini`:
```ini
[tox]
envlist = py38,py39,py310,py311

[testenv]
deps =
    pytest
    -e .
commands =
    pytest tests/
```

Run tests across Python versions:
```bash
pip install tox
tox
```

### 4. Test with Different Dependency Versions

Test with minimum versions:
```bash
python -m venv min_env
source min_env/bin/activate

# Install with minimum versions
pip install numpy==1.22.1 pandas==1.4.0 # ... etc

# Test
pytest tests/
```

Test with latest versions:
```bash
python -m venv latest_env
source latest_env/bin/activate

# Install with latest compatible versions
pip install --upgrade numpy pandas matplotlib # ... etc

# Test
pytest tests/
```

### 5. Check for Dependency Conflicts

```bash
pip install pipdeptree
pipdeptree --warn conflict
```

### 6. Validate Package Metadata

```bash
pip install twine check-manifest

# Check package can be built
python -m build

# Validate built package
twine check dist/*

# Check MANIFEST includes all needed files
check-manifest
```

## Finding Minimum Compatible Versions

### Strategy 1: Start Conservative
Use versions from your current working environment as minimums:
```
numpy>=1.22.1  # Your current version that works
```

### Strategy 2: Check Package Release Dates
Match minimum Python version support:
```
# If targeting Python 3.8+, check what versions support 3.8
numpy>=1.20.0  # First version supporting Python 3.8
```

### Strategy 3: Test Breaking Changes
For each major dependency, check changelog for breaking changes:
```bash
# Example: Find when API changed
pip install numpy==1.20.0
python -c "from image_velocimetry_tools import ..." # Test
pip install numpy==1.21.0
python -c "from image_velocimetry_tools import ..." # Test again
```

## Handling Special Dependencies

### areacomp3 (Git Dependency)

For PyPI publishing, git dependencies are problematic. Options:

**Option 1: Make it optional**
```python
# In setup.py
extras_require={
    'usgs': ['areacomp3 @ git+https://...'],
}

# Users install with:
# pip install image_velocimetry_tools[usgs]
```

**Option 2: Document manual installation**
```python
# Remove from install_requires
# Add to README:
# "USGS users must install areacomp3 separately:
#  pip install git+https://code.usgs.gov/RIVRS/areacomp.git@master"
```

**Option 3: Vendor it** (if license permits)
Include the code directly in your package.

## Pre-Publishing Checklist

### 1. Version Number
Update version in `setup.py`:
```python
version='1.0.1',  # Increment appropriately
```

### 2. Test Installation Flow

```bash
# Build distribution
python -m pip install --upgrade build
python -m build

# This creates:
# dist/image_velocimetry_tools-1.0.1.tar.gz
# dist/image_velocimetry_tools-1.0.1-py3-none-any.whl

# Test wheel installation
pip install dist/image_velocimetry_tools-1.0.1-py3-none-any.whl

# Test that everything works
python -c "from image_velocimetry_tools.api import run_batch_processing"
```

### 3. Check Package Contents

```bash
# Extract and inspect wheel
unzip -l dist/image_velocimetry_tools-1.0.1-py3-none-any.whl

# Verify includes:
# - All .py files
# - README.md
# - LICENSE
# - No unnecessary files (__pycache__, .pyc, etc.)
```

### 4. Test with TestPyPI

```bash
# Upload to TestPyPI first
pip install twine
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ image_velocimetry_tools

# If it works, upload to real PyPI:
twine upload dist/*
```

## Continuous Dependency Testing

### GitHub Actions Example

Create `.github/workflows/test.yml`:
```yaml
name: Test Installation

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install package
      run: |
        pip install .

    - name: Test import
      run: |
        python -c "from image_velocimetry_tools.api import run_batch_processing"

    - name: Run tests
      run: |
        pip install pytest
        pytest tests/
```

## Version Constraint Philosophy

### Upper Bounds
**Use upper bounds** to prevent breaking changes:
```
numpy>=1.22.1,<2.0  # Prevent numpy 2.0 breaking changes
```

### Lower Bounds
**Use tested minimums**:
```
numpy>=1.22.1  # We know this version works
```

### Exceptions
**No upper bound** for very stable APIs:
```
python-dateutil>=2.9.0  # Very stable API
```

**Tighter bounds** for unstable packages:
```
somepackage>=0.5.0,<0.6  # Pre-1.0 packages change often
```

## Troubleshooting

### "ResolutionTooDeep" Error
**Cause**: Over-constrained dependencies (too many `==`)
**Fix**: Relax constraints in `install_requires.txt`

### "No matching distribution"
**Cause**: Package doesn't exist for your Python version/OS
**Fix**: Check package compatibility, consider alternatives

### "Incompatible versions"
**Cause**: Two packages need conflicting versions of same dependency
**Fix**: Find overlapping compatible range or update packages

### Slow installation
**Cause**: Building packages from source (no wheel available)
**Fix**: Add binary package or wait for wheels

## Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [pip-tools documentation](https://pip-tools.readthedocs.io/)
- [PyPI Help](https://pypi.org/help/)
- [Semantic Versioning](https://semver.org/)

## Quick Commands Reference

```bash
# Test current setup
pip install -e .                              # Editable install
python -c "import image_velocimetry_tools"    # Test import

# Build package
python -m build                               # Build wheel and sdist

# Check package
twine check dist/*                            # Validate metadata
check-manifest                                # Check MANIFEST.in

# Upload to PyPI
twine upload --repository testpypi dist/*     # Test first!
twine upload dist/*                           # Real PyPI

# Dependency analysis
pipdeptree                                    # Show dependency tree
pip-compile install_requires.txt              # Generate lockfile
```
