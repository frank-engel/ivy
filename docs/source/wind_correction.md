# Wind-Induced Surface Velocity Correction

Wind-induced surface drift can significantly affect surface velocity measurements in exposed river reaches and open water bodies. **IVyTools** provides an optional wind correction feature to account for this effect when calculating discharge.

**Important:** Wind correction is an **optional** feature. Many measurements will not require wind correction, particularly those conducted in sheltered locations or during calm conditions. This feature should only be enabled when:

1. Wind data is available for the measurement period
2. The measurement site is exposed to wind
3. Wind speeds exceed approximately 3 m/s (7 mph)

## Background

Wind blowing across a water surface induces a drift current in the upper layers of the water column. This drift can either increase or decrease the measured surface velocity, depending on the relative directions of wind and flow.

The correction methodology is based on established hydrodynamic relationships from the peer-reviewed literature, specifically:

- **Wüest, A., and Lorke, A., 2003**. Small-scale hydrodynamics in lakes: Annual Review of Fluid Mechanics, v. 35, p. 373-412. [https://doi.org/10.1146/annurev.fluid.35.101101.161220](https://doi.org/10.1146/annurev.fluid.35.101101.161220)

Additional references for wind profile parameters:
- Panofsky, H.A., and Dutton, J.A., 1984. Atmospheric Turbulence
- Wieringa, J., 1993. Representative roughness parameters for homogeneous terrain
- Smith, S.D., 1980. Wind stress and heat flux over the ocean in gale force winds

## When to Apply Wind Correction

Consider enabling wind correction when:

- **Wind speed > 3 m/s (7 mph)**: Below this threshold, wind effects are typically negligible
- **Exposed measurement site**: Open water bodies, wide rivers, or reaches with minimal riparian vegetation
- **Wind data available**: Measured wind speed, direction, and sensor height must be known
- **Flow direction known**: River flow direction must be determined (can be estimated from cross-section orientation)

Wind correction is **NOT recommended** when:

- Wind speed < 3 m/s
- Measurement site is sheltered (narrow channels, heavy vegetation)
- Wind data is unavailable or unreliable
- Flow direction is uncertain

## Correction Methodology

The wind correction process involves three steps:

### Step 1: Convert Wind Speed to 10-m Equivalent (U₁₀)

Wind sensors are often not located exactly 10 m above the water surface. **IVyTools** uses a power-law wind profile to convert the measured wind speed at sensor height *h* to an equivalent 10-m wind speed:

$$U_{10} = U_h \times \left(\frac{10}{h}\right)^\alpha$$

where:
- *U*<sub>h</sub> = measured wind speed at height *h* [m/s]
- *h* = sensor height above water surface [m]
- *α* = power-law exponent (depends on terrain)

Typical *α* values:
- **Open Water**: 0.11
- **Open Terrain**: 0.14 (default)
- **Suburban**: 0.20
- **Forest/Rough**: 0.22
- **Urban**: 0.30

### Step 2: Calculate Wind-Induced Drift

The wind-induced surface drift velocity is computed as:

$$u_w = f \times U_{10}$$

where:
- *f* = drift fraction (typically 0.03 for rivers, meaning 3% of wind speed)
- *U*<sub>10</sub> = 10-m wind speed [m/s]

### Step 3: Project Drift Along Flow Direction

Only the component of wind drift aligned with (or against) the flow direction affects the measurement:

$$u_{eff} = u_w \times \cos(\theta_w - \theta_f)$$

where:
- *θ*<sub>w</sub> = wind direction (geographic, degrees)
- *θ*<sub>f</sub> = flow direction (geographic, degrees)

**Corrected velocity** = Measured velocity - *u*<sub>eff</sub>

**Sign convention**:
- Positive *u*<sub>eff</sub>: Wind aids flow (measured velocity is **higher** than actual)
- Negative *u*<sub>eff</sub>: Wind opposes flow (measured velocity is **lower** than actual)

## Using Wind Correction in IVyTools

### Enabling Wind Correction

1. Navigate to the **Discharge** tab

2. Locate the **Wind Correction (Optional)** panel (collapsed by default)

3. Check the **Enable wind correction** checkbox

### Required Inputs

When wind correction is enabled, the following fields must be completed:

**Wind Speed** (m/s)
- Measured wind speed at sensor height
- Valid range: 0-30 m/s
- Example: 5.0

**Wind Direction** (degrees)
- Direction wind is **coming FROM** (geographic convention)
- Valid range: 0-360°
- 0° = North, 90° = East, 180° = South, 270° = West
- Example: 270 (west wind)

**Flow Direction** (degrees)
- Direction river is **flowing TO** (geographic convention)
- Valid range: 0-360°
- Same convention as wind direction
- Example: 90 (eastward flow)
- **Tip**: Can often be estimated from cross-section orientation

**Sensor Height** (m)
- Height of wind sensor above water surface
- Default: 10.0 m
- Valid range: 0.1-10.0 m
- Example: 9.144 (30 feet)

### Optional Advanced Parameters

Click **Advanced** to access additional settings:

**Terrain Type**
- Select terrain classification for wind profile
- Default: Open Terrain
- Options: Open Water, Open Terrain, Suburban, Forest/Rough, Urban

**Drift Fraction (f)**
- Fraction of wind speed contributing to surface drift
- Default: 0.03 (3%)
- Valid range: 0.01-0.05
- Adjust if site-specific calibration data available

### Preview Correction

Before applying wind correction to the discharge calculation:

1. Complete all required input fields

2. Click **Preview Correction** button

3. Review the correction preview:
   ```
   U₁₀ = 5.47 m/s
   C₁₀ = 0.00105 (drag coefficient)
   Effective drift = +0.08 m/s (with flow)

   Example at 1.50 m/s measured:
   → Corrected: 1.42 m/s (-5.3%)
   ```

4. Evaluate whether correction magnitude is reasonable

5. Proceed with discharge calculation if acceptable

### Interpreting Preview Results

**U₁₀**: 10-m equivalent wind speed. Should be similar to measured wind if sensor is near 10 m height.

**Effective drift**: Component of wind drift along flow direction.
- Positive values: Wind aids flow (will decrease corrected velocity)
- Negative values: Wind opposes flow (will increase corrected velocity)
- Near zero: Wind perpendicular to flow (minimal correction)

**Percentage change**: Relative impact on velocity. If > 10%, wind effect is significant.

## Example Application

### Scenario
- Measured surface velocity: 1.50 m/s
- Wind speed (at 30 ft sensor): 5.0 m/s
- Wind direction: 270° (west wind)
- Flow direction: 90° (east flow)
- Sensor height: 9.144 m (30 ft)

### Calculation

1. **Convert to U₁₀**:
   - *α* = 0.14 (open terrain)
   - U₁₀ = 5.0 × (10/9.144)^0.14 = 5.47 m/s

2. **Calculate drift**:
   - *u*<sub>w</sub> = 0.03 × 5.47 = 0.164 m/s

3. **Project along flow**:
   - *θ*<sub>w</sub> = 270°, *θ*<sub>f</sub> = 90°
   - *u*<sub>eff</sub> = 0.164 × cos(270° - 90°) = 0.164 × cos(180°) = -0.164 m/s
   - Wind opposes flow!

4. **Correct velocity**:
   - *v*<sub>corrected</sub> = 1.50 - (-0.164) = 1.66 m/s
   - Correction: +11% (wind was slowing measured velocity)

### Result
Without correction, discharge would be **underestimated by ~11%** in this scenario.

## Uncertainty Considerations

Wind correction introduces additional uncertainty into the discharge estimate. Sources of uncertainty include:

- **Wind measurement error**: Instrument accuracy, temporal variability
- **Wind profile assumptions**: Power-law may not perfectly represent actual profile
- **Drift fraction uncertainty**: *f* = 0.03 is a typical value but varies with conditions
- **Direction uncertainty**: Errors in wind or flow direction estimation

**IVyTools** does not currently propagate wind correction uncertainty into the total discharge uncertainty estimate. Users should be aware that enabling wind correction may increase overall uncertainty, particularly when:

- Wind speeds are very high (> 10 m/s)
- Wind direction is highly variable
- Sensor height is significantly different from 10 m
- Site-specific drift calibration is unavailable

As a general guideline:
- Wind correction uncertainty: approximately ±2% of corrected velocity
- Total uncertainty may increase by 1-3 percentage points when correction is applied

## Data Storage

Wind correction parameters are saved with the project in the `*.ivy` file. This ensures:

- Reproducibility of discharge calculations
- Documentation of correction methodology
- Ability to review or modify parameters later

When a project is saved with wind correction enabled, the following metadata is stored:

```json
{
  "wind_correction": {
    "enabled": true,
    "wind_speed_mps": 5.0,
    "wind_direction_deg": 270,
    "flow_direction_deg": 90,
    "sensor_height_m": 9.144,
    "terrain": "open_terrain",
    "drift_fraction_f": 0.03
  }
}
```

## Validation and Quality Control

To ensure wind correction is applied correctly:

1. **Verify sign**: After correction, consider whether the change makes physical sense
   - Wind with flow → measured velocity should decrease after correction
   - Wind against flow → measured velocity should increase after correction

2. **Check magnitude**: Typical corrections are 2-15% of measured velocity
   - Very large corrections (> 20%) may indicate input errors

3. **Review time alignment**: Ensure wind data corresponds to measurement time period

4. **Compare with historical data**: If previous measurements exist, verify consistency

5. **Document assumptions**: Note any uncertainties in wind or flow direction

## Frequently Asked Questions

**Q: Should I always enable wind correction?**

A: No. Only enable wind correction when wind speeds exceed ~3 m/s AND reliable wind data is available. Many measurements will not require correction.

**Q: What if I only know approximate flow direction?**

A: Small errors in flow direction (±10-15°) have minimal impact on the correction. Use your best estimate based on cross-section orientation or site observations.

**Q: Can I use weather station data from several miles away?**

A: Use caution. Local topography can significantly affect wind speed and direction. On-site measurements are strongly preferred.

**Q: What if wind direction is perpendicular to flow?**

A: The effective drift component will be near zero, and correction will have minimal impact. You may choose to disable correction in this case.

**Q: How do I determine *f* for my site?**

A: The default value of 0.03 is appropriate for most river applications. Site-specific calibration requires paired velocity and wind measurements, which is beyond the scope of typical **IVyTools** applications.

**Q: Does wind correction affect uncertainty?**

A: Currently, **IVyTools** does not include wind correction uncertainty in the total uncertainty budget. Users should be aware that correction adds some additional uncertainty (typically 1-3 percentage points).

## References

Rantz, S.E., and others, 1982. Measurement and computation of streamflow:
U.S. Geological Survey Water-Supply Paper 2175, v. 1, 284 p.
[https://doi.org/10.3133/wsp2175](https://doi.org/10.3133/wsp2175)

Turnipseed, D.P., and Sauer, V.B., 2010. Discharge measurements at gaging
stations: U.S. Geological Survey Techniques and Methods book 3, chap. A8,
87 p. [https://doi.org/10.3133/tm3A8](https://doi.org/10.3133/tm3A8)

Wüest, A., and Lorke, A., 2003. Small-scale hydrodynamics in lakes:
Annual Review of Fluid Mechanics, v. 35, p. 373-412.
[https://doi.org/10.1146/annurev.fluid.35.101101.161220](https://doi.org/10.1146/annurev.fluid.35.101101.161220)
