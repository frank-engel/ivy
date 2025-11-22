# GUI Integration Plan: Wind Correction Feature

## Overview

This document outlines the integration plan for adding wind-induced surface velocity correction to the IVyTools GUI. The feature is **optional** and will be disabled by default to maintain backward compatibility.

## Design Principles

1. **Optional Feature**: Wind correction is off by default
2. **Per-Measurement**: Wind parameters stored in individual *.ivy files
3. **Non-Intrusive**: Does not affect existing workflows when disabled
4. **Simple Interface**: Minimal required inputs (wind speed, direction, sensor height)
5. **Clear Feedback**: User can see correction magnitude before applying

## File Format Changes

### *.ivy File Structure

Add optional `wind_correction` section to project metadata:

```python
{
    "project_metadata": {
        # ... existing metadata ...
        "wind_correction": {
            "enabled": false,  # boolean flag
            "wind_speed_mps": null,
            "wind_direction_deg": null,  # geographic, "coming from"
            "flow_direction_deg": null,  # geographic, "going to"
            "sensor_height_m": 10.0,
            "terrain": "open_terrain",
            "drift_fraction_f": 0.03,
            "alpha_override": null  # optional custom exponent
        }
    }
}
```

## GUI Components

### 1. Discharge Tab - Wind Correction Panel

Location: `image_velocimetry_tools/gui/discharge.py`

Add collapsible panel below existing discharge settings:

```
┌─ Wind Correction (Optional) ─────────────────┐
│ ☐ Enable wind correction                     │
│                                               │
│ Wind Speed (m/s):      [     5.0     ]       │
│ Wind Direction (°):    [    270      ]       │
│ Flow Direction (°):    [     90      ]       │
│ Sensor Height (m):     [    10.0     ]       │
│                                               │
│ Advanced ▼                                    │
│   Terrain:  [Open Terrain ▼]                 │
│   Drift f:  [     0.03     ]                 │
│                                               │
│ [Preview Correction]                          │
└───────────────────────────────────────────────┘

Correction Preview:
  U10 = 5.47 m/s
  Effective drift = +0.08 m/s (with flow)

  Measured velocity: 1.50 m/s
  → Corrected:       1.42 m/s  (-5.3%)
```

### 2. Field Definitions

**Wind Speed** (required)
- Units: m/s (meters per second)
- Range: 0-30 m/s
- Tooltip: "Measured wind speed at sensor height"

**Wind Direction** (required)
- Units: degrees (0-360°)
- Convention: Geographic (0° = North, "coming from")
- Tooltip: "Direction wind is coming FROM (e.g., 270° = west wind)"

**Flow Direction** (required)
- Units: degrees (0-360°)
- Convention: Geographic (0° = North, "going to")
- Tooltip: "Direction river is flowing TO (e.g., 90° = east flow)"
- Suggestion: Auto-populate from cross-section orientation if available

**Sensor Height** (optional, default 10.0)
- Units: meters
- Range: 0.1-10.0 m
- Tooltip: "Height of wind sensor above water surface"

**Terrain** (advanced, optional)
- Dropdown options:
  - Open Water (α=0.11)
  - Open Terrain (α=0.14) [default]
  - Suburban (α=0.20)
  - Forest/Rough (α=0.22)
  - Urban (α=0.30)
- Tooltip: "Affects wind profile conversion to 10-m equivalent"

**Drift Fraction f** (advanced, optional, default 0.03)
- Range: 0.01-0.05
- Tooltip: "Fraction of wind speed contributing to surface drift (0.03 typical for rivers)"

### 3. Validation Rules

- Wind direction and flow direction must both be provided
- If enabled, all required fields must be filled
- Show warning if wind speed > 10 m/s (high correction uncertainty)
- Show warning if |wind_dir - flow_dir| < 30° (strong effect expected)

### 4. Preview Calculation

When user clicks "Preview Correction":
1. Calculate U10 using power-law
2. Calculate effective drift component
3. Show before/after comparison for typical velocity (e.g., 1.5 m/s)
4. Display percentage change
5. Highlight if correction > 10% (significant)

## Implementation Steps

### Phase 1: Backend Support (Complete)
- ✅ `wind_correction.py` module created
- ✅ `discharge_tools.py` updated with `wind_params` support

### Phase 2: GUI Elements

**File: `image_velocimetry_tools/gui/discharge.py`**

Add to discharge processing class:

```python
class DischargeProcessor:
    def __init__(self):
        # ... existing init ...
        self.wind_correction_enabled = False
        self.wind_params = None

    def setup_wind_panel(self):
        """Create wind correction UI panel"""
        # QGroupBox with collapsible behavior
        # Input fields for required params
        # Advanced options collapsible
        # Preview button and display

    def validate_wind_params(self):
        """Validate wind correction inputs"""
        if not self.wind_correction_enabled:
            return True
        # Check required fields
        # Validate ranges
        # Return True/False

    def preview_wind_correction(self):
        """Calculate and display correction preview"""
        from ..wind_correction import wind_correction_metadata

        meta = wind_correction_metadata(
            self.wind_speed,
            self.wind_dir,
            self.flow_dir,
            sensor_height=self.sensor_height,
            terrain=self.terrain,
            f=self.drift_f
        )
        # Display results in preview pane

    def get_wind_params_dict(self):
        """Package wind params for discharge calculation"""
        if not self.wind_correction_enabled:
            return None

        return {
            'wind_speed': self.wind_speed,
            'wind_dir': self.wind_dir,
            'flow_dir': self.flow_dir,
            'sensor_height': self.sensor_height,
            'terrain': self.terrain,
            'f': self.drift_f
        }
```

**File: `image_velocimetry_tools/gui/dialogs/settings.py`**

Add wind correction defaults to global settings:

```python
DEFAULT_SETTINGS = {
    # ... existing settings ...
    'wind_correction': {
        'default_sensor_height_m': 10.0,
        'default_terrain': 'open_terrain',
        'default_drift_fraction': 0.03,
        'show_high_wind_warning': True,
        'warning_threshold_mps': 10.0
    }
}
```

### Phase 3: File I/O

**File: `image_velocimetry_tools/file_management.py`**

Update project save/load to handle wind correction metadata:

```python
def save_project(project_data, filepath):
    # ... existing save logic ...

    # Add wind correction metadata if present
    if hasattr(project_data, 'wind_correction'):
        data['wind_correction'] = {
            'enabled': project_data.wind_correction.enabled,
            'wind_speed_mps': project_data.wind_correction.wind_speed,
            # ... etc
        }

def load_project(filepath):
    # ... existing load logic ...

    # Load wind correction if present
    if 'wind_correction' in data:
        project.wind_correction = WindCorrectionParams(
            **data['wind_correction']
        )
    else:
        project.wind_correction = WindCorrectionParams(enabled=False)
```

## User Workflow

### Typical Use Case

1. User opens IVy project
2. Navigate to Discharge tab
3. Click "Enable wind correction" checkbox
4. Enter wind data:
   - Speed: 5.0 m/s
   - Direction: 270° (west wind)
   - Flow: 90° (east flow)
   - Height: 10.0 m (default)
5. Click "Preview Correction"
6. Review preview:
   - U10 = 5.47 m/s
   - Drift = +0.08 m/s
   - Effect: -5.3% on velocity
7. If acceptable, proceed with discharge calculation
8. Wind parameters saved with project

### Edge Cases

**No wind data available**
- Feature disabled
- Proceeds with standard calculation

**Wind direction unknown**
- Cannot enable correction
- Show error message

**Very high winds (>15 m/s)**
- Show warning: "High wind speeds may have increased uncertainty"
- Allow user to proceed or disable

**Wind perpendicular to flow**
- Correction ≈ 0
- Inform user minimal effect expected

## Testing Checklist

- [ ] Enable/disable toggles correctly
- [ ] Required fields enforced when enabled
- [ ] Preview calculation matches `wind_correction.py` output
- [ ] Wind params saved to *.ivy file
- [ ] Wind params loaded from *.ivy file
- [ ] Disabled projects still work (backward compatibility)
- [ ] Discharge calculation uses wind_params when enabled
- [ ] Validation catches invalid inputs
- [ ] Terrain dropdown populates correctly
- [ ] Advanced options collapse/expand
- [ ] Tooltips display correctly

## Documentation Updates Needed

1. User manual section on wind correction
2. Example with/without correction comparison
3. When to use wind correction (decision tree)
4. Uncertainty considerations
5. API documentation for programmatic use

## Future Enhancements (Optional)

1. **Auto-populate from weather station**
   - Import wind data from NOAA/weather APIs
   - Match timestamp to measurement

2. **Batch correction**
   - Apply same wind params to multiple projects
   - Batch import from CSV

3. **Sensitivity analysis**
   - Show how discharge changes with ±20% wind speed
   - Uncertainty propagation

4. **Visualization**
   - Plot wind vector vs flow direction
   - Show drift component graphically

## References

User manual style guide: `/docs/source/` (markdown format)
