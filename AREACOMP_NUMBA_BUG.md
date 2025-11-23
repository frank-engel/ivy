# AreaComp3 Numba Compatibility Issue

## Summary

AreaComp3's `@numba.jit` decorators are incompatible with modern numba versions (0.59+) due to missing `nopython` parameter. This causes runtime errors when loading cross-section data in applications using Python 3.11+.

## Environment

- **Python Version**: 3.11.9
- **AreaComp3**: master branch (as of November 2023)
- **Numba**: 0.58.x (workaround), 0.62+ (fails)
- **Application**: IVyTools (image velocimetry analysis)

## Symptoms

When calling `xs_survey.compute_channel_char()` after loading an AreaComp .mat file:

```python
from areacomp.gui.areasurvey import AreaSurvey

xs_survey = AreaSurvey()
xs_survey.load_areacomp("cross_section.mat", units="SI")
xs_survey.compute_channel_char(xs_survey.stage)  # ← Fails here
```

### Error with Numba 0.62+

```
numba.core.errors.TypingError: Failed in nopython mode pipeline (step: nopython frontend)
non-precise type pyobject
During: typing of argument at areacomp/gui/areasurvey.py (954)

File "areacomp/gui/areasurvey.py", line 954:
    def remove_arbitrary(station_stage):
        <source elided>

    @numba.jit
    ^

This error may have been caused by the following argument(s):
- argument 0: Cannot determine Numba type of <class 'areacomp.gui.areasurvey.AreaSurvey'>
```

### Deprecation Warnings with Numba 0.58.x

```
NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the
'numba.jit' decorator. The implicit default value for this argument is currently
False, but it will be changed to True in Numba 0.59.0.

  @numba.jit
```

## Root Cause

The issue is in `areacomp/gui/areasurvey.py` where several methods use `@numba.jit` decorator without specifying the `nopython` parameter:

### Problem Locations

1. **Line 954** - `remove_arbitrary()` function
2. **Line 979** - Another decorated function
3. **Line 1003** - Another decorated function

### The Bug

```python
@numba.jit  # ❌ Missing nopython parameter
def remove_arbitrary(station_stage):
    # ... function body
```

**Why this fails:**
- Numba 0.59+ defaults `nopython=True` for `@numba.jit`
- When `nopython=True`, numba cannot compile methods that receive `self` (the AreaSurvey instance)
- The error occurs because numba can't determine the type of the `self` parameter

## Impact

- **Python 3.11+**: Required by AreaComp3 and QRev
- **Numba 0.56.x**: Doesn't support Python 3.11 (requires Python <3.11)
- **Numba 0.57-0.61**: May work but approaching deprecation
- **Numba 0.62+**: Hard failure due to stricter JIT compilation

This creates an impossible dependency triangle:
- Python 3.11 required → Can't use numba 0.56.x
- Numba 0.62+ available → Incompatible with current `@numba.jit` usage
- Numba 0.58.x works → But will break in numba 0.59+

## Workaround (Temporary)

Pin numba to 0.58.x in application dependencies:

```txt
# install_requires.txt
numba>=0.58.0,<0.59.0
```

This avoids both the Python 3.11 incompatibility (0.56.x) and the strict nopython mode (0.62+).

Also suppress the deprecation warnings:

```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='numba')
```

## Recommended Fix

### Option 1: Explicitly Set `nopython=False` (Quick Fix)

If the decorated functions genuinely need object mode (e.g., they manipulate class instances):

```python
@numba.jit(nopython=False)  # ✓ Explicit object mode
def remove_arbitrary(station_stage):
    # ... function body
```

This will work with all numba versions but won't get the performance benefits of nopython mode.

### Option 2: Refactor to Use `@numba.njit` (Recommended)

If the functions can be refactored to pure functions (no `self` parameter):

```python
@numba.njit  # ✓ nopython mode, better performance
def remove_arbitrary(station_stage_array):
    """Pure function operating on arrays, no class instance needed"""
    # ... refactored function body
```

Then call it from the class method:

```python
class AreaSurvey:
    def calculate_wetted_perimeter(self, clean_ss):
        # Extract data from self
        station_stage = self._prepare_station_stage(clean_ss)

        # Call pure numba function
        result = remove_arbitrary(station_stage)

        return result
```

### Option 3: Remove JIT Compilation

If performance testing shows JIT compilation isn't providing significant speedup:

```python
# @numba.jit  # Remove decorator
def remove_arbitrary(station_stage):
    # ... function body works as pure Python
```

## Testing Recommendations

1. **Test with numba 0.58.x** - Current working version
2. **Test with numba 0.62+** - Future-proof the fix
3. **Profile performance** - Ensure JIT is actually beneficial
4. **Test all code paths** - Ensure channel characteristic computations work across different cross-section types

## Additional Context

This issue was discovered when integrating AreaComp3 into IVyTools for batch processing of image velocimetry discharge measurements. The batch processing API works fine because it only calls `load_areacomp()` and doesn't trigger the channel characteristic computations. However, the GUI workflow calls `compute_channel_char()` when loading projects, which triggers the numba JIT compilation error.

## Files Affected in AreaComp3

- `areacomp/gui/areasurvey.py` - Lines 954, 979, 1003 (and potentially others)
- Any other files using `@numba.jit` without explicit `nopython` parameter

## Contact

For questions about this issue:
- **IVyTools Repository**: https://code.usgs.gov/hydrologic-remote-sensing-branch/ivy
- **Issue discovered by**: Frank L. Engel (sandcountyfrank@gmail.com)
- **Date**: November 2025

---

## Appendix: Numba Version Compatibility Matrix

| Numba Version | Python 3.11 | AreaComp3 Current | Notes |
|---------------|-------------|-------------------|-------|
| 0.56.x | ❌ | ✅ | Python 3.11 not supported |
| 0.57.x | ✅ | ⚠️ | Works but approaching deprecation |
| 0.58.x | ✅ | ⚠️ | **Current workaround**, deprecation warnings |
| 0.59.x | ✅ | ❌ | Will default nopython=True |
| 0.62+ | ✅ | ❌ | Hard failure with current code |

**Legend**: ✅ Works | ⚠️ Works with warnings | ❌ Broken
