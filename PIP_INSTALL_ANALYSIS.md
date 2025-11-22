# Pip Install Analysis and Fixes

## Issues Found from Terminal Output

### 1. **Matplotlib Version Conflict** (FIXED)
```
ERROR: Cannot install because these package versions have conflicting dependencies:
    image-velocimetry-tools 1.0.0.2 depends on matplotlib==3.3.3
    areacomp3 1.4.0 depends on matplotlib==3.6.3
```

**Fix Applied**: Updated `install_requires.txt` to use `matplotlib>=3.6.3,<4.0` (matching areacomp3's requirement)

### 2. **Dev Packages in .venv**
Your `.venv` has ALL 127 packages from `requirements.txt` installed, including dev-only tools:
- pytest, coverage (testing)
- sphinx, alabaster, sphinx-book-theme (docs)
- black, pylint, isort (linting/formatting)
- pyinstaller (packaging)
- And many more...

These aren't needed at runtime and are causing dependency conflicts.

### 3. **Excessive Backtracking**
Pip spent several minutes trying version combinations because:
- areacomp3 and qrev have strict requirements
- Many packages need to be compiled from source
- Resolver is checking 200,000+ combinations

### 4. **Build from Source Issues**
In fresh environment, some packages tried to build from source:
- numpy (15.8 MB download, compile time)
- scipy (compiling)
- scikit-learn (FAILED - build error)
- matplotlib (35.9 MB, compile time)

## Recommendations

### Option 1: Use Existing .venv with Upgraded Matplotlib (FASTEST)

Your .venv already has everything. Just upgrade matplotlib:

```bash
# Activate your .venv
.venv\Scripts\activate

# Upgrade matplotlib to satisfy areacomp3
pip install --upgrade "matplotlib>=3.6.3,<4.0"

# Now install your package
pip install -e .

# Test it worked
python -c "from image_velocimetry_tools.api import run_batch_processing; print('Success!')"
```

**Pros**: Fast, no compilation needed
**Cons**: Keeps unnecessary dev packages

### Option 2: Fresh Install with Pre-built Wheels (RECOMMENDED)

```bash
# Create fresh environment
python -m venv fresh_env
fresh_env\Scripts\activate

# Upgrade pip first (important!)
python -m pip install --upgrade pip

# Install build tools
pip install wheel setuptools

# Try installing - should use pre-built wheels
pip install .

# If that fails, install problematic packages first:
pip install numpy scipy scikit-learn matplotlib pandas
pip install .
```

**Pros**: Clean environment, only runtime deps
**Cons**: May need compilation if wheels unavailable

### Option 3: Install Without areacomp3 First

Since areacomp3 is causing the conflict, try:

```bash
# Edit install_requires.txt temporarily
# Comment out the areacomp3 line:
# # areacomp3 @ git+https://code.usgs.gov/RIVRS/areacomp.git@master

# Install without it
pip install .

# Then install areacomp3 manually
pip install git+https://code.usgs.gov/RIVRS/areacomp.git@master

# Test
python -c "from image_velocimetry_tools.api import run_batch_processing; print('Success!')"
```

### Option 4: Make areacomp3 Optional (FOR PYPI)

For PyPI distribution, make areacomp3 optional since it's a git dependency:

Edit `setup.py` to add:
```python
extras_require={
    'usgs': ['areacomp3 @ git+https://code.usgs.gov/RIVRS/areacomp.git@master'],
},
```

Then users can:
```bash
pip install image_velocimetry_tools  # Without areacomp3
pip install image_velocimetry_tools[usgs]  # With areacomp3
```

## Updated install_requires.txt Changes

✅ **Fixed**:
- matplotlib minimum updated to 3.6.3 (matches areacomp3)
- Removed ffmpeg-python (not used)
- Only runtime dependencies (no dev tools)

## Next Steps

I recommend **Option 1** for immediate testing (fastest), then **Option 4** for PyPI preparation.

Try this now:
```powershell
# In PowerShell
.\.venv\Scripts\activate
pip install --upgrade "matplotlib>=3.6.3,<4.0"
pip install -e .
python test_installation.py
```

If that works, you're done! If not, try Option 2 or 3.

## For PyPI Publishing

Before publishing to PyPI, you'll need to decide on the areacomp3 dependency:
1. Make it optional (extras_require) - RECOMMENDED
2. Vendor it (include source code) - if license permits
3. Document manual installation - requires users to install separately

PyPI doesn't allow git dependencies in install_requires, only in extras_require.
