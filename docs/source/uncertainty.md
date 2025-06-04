# Uncertainty Estimation in IVyTools

## Overview

**IVyTools** estimates total discharge uncertainty using two methods:

1.  **ISO 748:2007** – This method applies standard techniques for
    estimating uncertainty in midsection (velocity-area) discharge
    measurements.
2.  **Interpolated Variance Estimator (IVE)** – A method described by
    Cohn and others (2013), adapted here to suit image-based velocity
    measurements.

Both methods output a **2-sigma (95% confidence interval)** estimate of
uncertainty, although the original IVE method, as published, is based on
a **1-sigma (68% confidence interval)**. This intentional modification
ensures consistency in reported uncertainty intervals across the
software.

The estimated uncertainty is displayed in the **IVyTools** [Discharge
Tab](discharge_tab.md) alongside a "User Rating" dropdown, which offers
qualitative quality categories (e.g., “Good (3-5%)”). **IVyTools**
automatically selects a rating based on the ISO 748 result, but the user
may override this selection based on expert judgment or additional data.
Refer to the [Discharge
Tab](file:///C:\REPOS\ivy\docs\word\discharge_tab.md) for more
information.

## ISO 748-Based Uncertainty

The ISO method incorporates various systematic and random sources of
uncertainty, including:

-   Systematic uncertainty from instrumentation and unaccounted sources.
-   Random uncertainty due to:
    -   Number of verticals
    -   Cross-sectional width estimation
    -   Depth measurement error
    -   Velocity exposure time
    -   Surface velocity to depth-averaged velocity conversion
    -   Image orthorectification error
    -   Instrument repeatability
    -   Point velocity method error

These components are combined using the ISO’s recommended propagation
formulas. Defaults are informed by ISO tables and the **IVyTools**
configuration.

## Interpolated Variance Estimator (IVE)

The IVE method, initially proposed by Cohn and others (2013), is used to
estimate uncertainty based on the spatial structure of the flow data. In
**IVyTools**, this method has been adapted for image velocimetry data
and expanded to include additional sources of uncertainty:

-   Uncertainties related to alpha arise from errors in estimating the
    coefficient when converting surface data to mean velocity.
-   Rectification uncertainty stems from inaccuracies in camera
    calibration and spatial referencing.

**Important Note**: The IVE implementation in **IVyTools** reports
uncertainty as a 2-sigma value, which deviates from the 1-sigma
convention used in the original paper. Users interpreting or comparing
results should be aware of this distinction.

## Additional Considerations

Both the ISO and IVE methods in **IVyTools** are adapted for image 
velocimetry-based flow measurement, resulting in some components being 
treated differently compared to traditional ADCP or current meter measurements.

-   Rectification uncertainty is computed from the RMSE of the image
    orthorectification and normalized by the scene width, enabling
    uncertainty propagation tied to geometric accuracy.
-   The confidence in the alpha coefficient used to correct surface
    velocity is context-dependent, with a default value of 3% (1σ).
-   All uncertainties are assumed to follow normal distributions, which
    allows for propagation through standard error combination formulas.
-   When specific metadata is missing, such as scene geometry or
    equipment specifications, **IVyTools** utilizes conservative default
    values derived from literature and internal validation.

## References

Cohn, T.A., Kiang, J.E., and Mason, R.R., 2013, Estimating Discharge
Measurement Uncertainty Using the Interpolated Variance Estimator:
Journal of Hydraulic Engineering, v. 139, no. 5, p. 502–510, at
<https://doi.org/10.1061/(ASCE)HY.1943-7900.0000695>.

ISO, 2007, Hydrometry — Measurement of liquid flow in open channels
using current-meters or floats.
