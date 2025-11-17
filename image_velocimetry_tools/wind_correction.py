"""IVy module for wind-induced surface velocity correction

This module provides functions to correct surface velocity measurements for
wind-induced drift using established hydrodynamic relationships. The correction
is particularly important for exposed water bodies where wind can significantly
affect surface velocity measurements.

The methodology is based on:
- Wüest, A., and Lorke, A., 2003. Small-scale hydrodynamics in lakes:
  Annual Review of Fluid Mechanics, v. 35, p. 373-412.
  https://doi.org/10.1146/annurev.fluid.35.101101.161220

References for terrain-based roughness parameters:
- Panofsky, H.A., and Dutton, J.A., 1984. Atmospheric Turbulence
- Wieringa, J., 1993. Representative roughness parameters for homogeneous terrain
- Smith, S.D., 1980. Wind stress and heat flux over the ocean in gale force winds

Author: Frank L. Engel, Ph.D.
"""

import numpy as np
from scipy.optimize import fsolve


# Terrain-based power-law exponents (α) with citations
TERRAIN_ALPHA = {
    "open_water": 0.11,       # Panofsky & Dutton, 1984
    "open_terrain": 0.14,     # Wieringa, 1993; Smith, 1980
    "suburban": 0.20,         # Wieringa, 1993
    "forest_or_rough": 0.22,  # Wieringa, 1993
    "urban": 0.30             # Wieringa, 1993
}


def alpha_for_terrain(terrain="open_terrain"):
    """Get power-law exponent for specified terrain type.

    Parameters
    ----------
    terrain : str
        Terrain classification. Options: 'open_water', 'open_terrain',
        'suburban', 'forest_or_rough', 'urban'

    Returns
    -------
    float
        Power-law exponent α for wind profile extrapolation

    Raises
    ------
    ValueError
        If terrain type is not recognized
    """
    try:
        return TERRAIN_ALPHA[terrain]
    except KeyError:
        raise ValueError(
            f"Unknown terrain kind '{terrain}'. "
            f"Options: {list(TERRAIN_ALPHA.keys())}"
        )


def low_wind_c10(U10, eps=0.1):
    """Compute drag coefficient for low wind speeds.

    Uses empirical fit from Wüest and Lorke (2003) Figure 3.
    C10 = 0.0044 × (U10 + ε)^(-1.15)

    Parameters
    ----------
    U10 : float or array-like
        10-m wind speed [m/s]. Must be >= 0.
    eps : float, optional
        Small offset [m/s] to avoid singularity at U10=0. Default = 0.1.

    Returns
    -------
    float or ndarray
        Surface drag coefficient C10 (dimensionless)

    Raises
    ------
    ValueError
        If U10 < 0
    """
    U10 = np.asarray(U10, dtype=float)

    if np.any(U10 < 0):
        raise ValueError("U10 must be non-negative.")

    return 0.0044 * (U10 + eps) ** -1.15


def charnock_c10(U10, kappa=0.41, K=11.3, g=9.81, max_iter=20, tol=1e-6):
    """Compute drag coefficient using Charnock's law.

    Solves the implicit relation iteratively:
    C10 = [1 / (κ^(-1) × ln(gz0 / (C10 × U10^2)) + K)]^2

    where κ is von Kármán constant and K is empirical constant.

    Parameters
    ----------
    U10 : float or array-like
        10-m wind speed [m/s]. Must be > 0.
    kappa : float, optional
        von Kármán constant. Default = 0.41.
    K : float, optional
        Empirical constant. Default = 11.3.
    g : float, optional
        Gravitational acceleration [m/s^2]. Default = 9.81.
    max_iter : int, optional
        Maximum iterations for convergence. Default = 20.
    tol : float, optional
        Convergence tolerance. Default = 1e-6.

    Returns
    -------
    float or ndarray
        Surface drag coefficient C10 (dimensionless)

    Raises
    ------
    ValueError
        If U10 <= 0
    """
    scalar_input = np.isscalar(U10)
    U_arr = np.array([U10] if scalar_input else U10, dtype=float)

    if np.any(U_arr <= 0):
        raise ValueError("U10 must be positive for Charnock's law.")

    C_arr = np.zeros_like(U_arr)

    for idx, u in enumerate(U_arr):
        c = 0.001  # initial guess
        for _ in range(max_iter):
            c_new = (1.0 / (kappa**-1 * np.log(g*10/(c*u**2)) + K))**2
            if abs(c_new - c) < tol:
                c = c_new
                break
            c = c_new
        C_arr[idx] = c

    return C_arr[0] if scalar_input else C_arr


def composite_c10(U10, transition=(2.5, 5.0)):
    """Compute drag coefficient using composite model.

    Blends low-wind empirical fit with Charnock's law:
    - U10 <= transition[0]: low_wind_c10
    - U10 >= transition[1]: charnock_c10
    - Between: linear interpolation

    Parameters
    ----------
    U10 : float or array-like
        10-m wind speed [m/s]
    transition : tuple of float, optional
        (low, high) wind speed bounds for blending [m/s].
        Default = (2.5, 5.0).

    Returns
    -------
    float or ndarray
        Surface drag coefficient C10 (dimensionless)
    """
    U10 = np.atleast_1d(U10)
    out = np.zeros_like(U10, dtype=float)

    for i, u in enumerate(U10):
        if u <= transition[0]:
            out[i] = low_wind_c10(u)
        elif u >= transition[1]:
            out[i] = charnock_c10(u)
        else:
            # Linear blend
            w = (u - transition[0]) / (transition[1] - transition[0])
            c_low = low_wind_c10(u)
            c_ch = charnock_c10(u)
            out[i] = (1 - w) * c_low + w * c_ch

    return out if len(out) > 1 else out[0]


def U10_from_Uh_powerlaw(Uh, h, alpha=None, terrain="open_terrain"):
    """Convert wind speed at height h to equivalent 10-m wind speed.

    Uses power-law wind profile:
    U10 = Uh × (10 / h)^α

    Parameters
    ----------
    Uh : float
        Measured wind speed at height h [m/s]
    h : float
        Measurement height [m], must be in (0, 10]
    alpha : float, optional
        Power-law exponent. If None, determined from terrain.
    terrain : str, optional
        Terrain type (used if alpha is None). Default = 'open_terrain'.

    Returns
    -------
    float
        Equivalent 10-m wind speed U10 [m/s]

    Raises
    ------
    ValueError
        If Uh < 0 or h not in (0, 10]
    """
    Uh = float(Uh)
    h = float(h)

    if Uh < 0:
        raise ValueError("Uh must be non-negative.")
    if not (0 < h <= 10):
        raise ValueError("Height h must be in range (0, 10] meters.")

    if alpha is None:
        alpha = alpha_for_terrain(terrain)

    return Uh * (10.0 / h) ** alpha


def geographic_to_arithmetic(geographic_angle, signed180=False):
    """Convert geographic angle (compass bearing) to arithmetic angle.

    Geographic: 0° = North, clockwise
    Arithmetic: 0° = East, counterclockwise

    Parameters
    ----------
    geographic_angle : float or ndarray
        Compass bearing in degrees [0, 360) or [-180, 180]
    signed180 : bool, optional
        If True, returns angles in [-180, 180]. Default = False.

    Returns
    -------
    float or ndarray
        Arithmetic angle in degrees
    """
    geographic_angle = np.asarray(geographic_angle)
    geographic_angle = np.where(geographic_angle < 0,
                                360 + geographic_angle,
                                geographic_angle)
    arithmetic_angle = (450 - geographic_angle) % 360

    if signed180:
        arithmetic_angle = np.where(arithmetic_angle >= 180,
                                   arithmetic_angle - 360,
                                   arithmetic_angle)

    return arithmetic_angle


def arithmetic_to_geographic(arithmetic_angle):
    """Convert arithmetic angle to geographic angle (compass bearing).

    Arithmetic: 0° = East, counterclockwise
    Geographic: 0° = North, clockwise

    Parameters
    ----------
    arithmetic_angle : float or ndarray
        Arithmetic angle in degrees

    Returns
    -------
    float or ndarray
        Compass bearing in degrees [0, 360)
    """
    return (90 - np.asarray(arithmetic_angle)) % 360


def wind_induced_drift(U10, wind_dir, flow_dir, f=0.03):
    """Calculate wind-induced surface drift and effective component.

    Computes the component of wind drift aligned with flow direction:
    u_eff = f × U10 × cos(θ_wind - θ_flow)

    Parameters
    ----------
    U10 : float
        10-m wind speed [m/s]
    wind_dir : float
        Wind direction in degrees (geographic, "coming from")
    flow_dir : float
        Flow direction in degrees (geographic, "going to")
    f : float, optional
        Wind drift fraction (0.03 typical for rivers). Default = 0.03.

    Returns
    -------
    dict
        {
            'u_w': total wind drift magnitude [m/s],
            'u_eff': effective drift along flow direction [m/s],
            'theta_w': wind direction [radians],
            'theta_f': flow direction [radians]
        }
    """
    u_w = f * U10

    theta_w = np.radians(wind_dir)
    theta_f = np.radians(flow_dir)

    u_eff = u_w * np.cos(theta_w - theta_f)

    return {
        'u_w': u_w,
        'u_eff': u_eff,
        'theta_w': theta_w,
        'theta_f': theta_f
    }


def apply_wind_correction(u_measured, wind_speed, wind_dir, flow_dir,
                         sensor_height=10.0, terrain="open_terrain",
                         alpha=None, f=0.03):
    """Apply wind correction to measured surface velocity.

    Complete pipeline:
    1. Convert wind at sensor height to U10
    2. Calculate wind-induced drift
    3. Compute effective drift component along flow
    4. Subtract drift from measured velocity

    Parameters
    ----------
    u_measured : float or array-like
        Measured surface velocity [m/s]
    wind_speed : float or array-like
        Measured wind speed [m/s]
    wind_dir : float or array-like
        Wind direction in degrees (geographic, "coming from")
    flow_dir : float
        River flow direction in degrees (geographic, "going to")
    sensor_height : float, optional
        Wind sensor height [m]. Default = 10.0.
    terrain : str, optional
        Terrain type. Default = 'open_terrain'.
    alpha : float, optional
        Custom power-law exponent (overrides terrain).
    f : float, optional
        Wind drift fraction. Default = 0.03.

    Returns
    -------
    float or ndarray
        Wind-corrected surface velocity [m/s]

    Notes
    -----
    Positive u_eff means wind aids flow (measured velocity higher than actual).
    Negative u_eff means wind opposes flow (measured velocity lower than actual).
    """
    # Convert to numpy arrays for vectorization
    u_measured = np.asarray(u_measured)
    wind_speed = np.asarray(wind_speed)
    wind_dir = np.asarray(wind_dir)

    # Convert wind speed to U10
    U10 = U10_from_Uh_powerlaw(wind_speed, sensor_height, alpha, terrain)

    # Calculate effective drift
    drift_result = wind_induced_drift(U10, wind_dir, flow_dir, f)
    u_eff = drift_result['u_eff']

    # Correct measured velocity
    u_corrected = u_measured - u_eff

    return u_corrected


def wind_correction_metadata(wind_speed, wind_dir, flow_dir,
                             sensor_height=10.0, terrain="open_terrain",
                             alpha=None, f=0.03):
    """Generate metadata dictionary for wind correction.

    Computes all intermediate values for documentation/validation.

    Parameters
    ----------
    wind_speed : float
        Measured wind speed [m/s]
    wind_dir : float
        Wind direction in degrees (geographic)
    flow_dir : float
        Flow direction in degrees (geographic)
    sensor_height : float, optional
        Wind sensor height [m]. Default = 10.0.
    terrain : str, optional
        Terrain type. Default = 'open_terrain'.
    alpha : float, optional
        Custom power-law exponent.
    f : float, optional
        Wind drift fraction. Default = 0.03.

    Returns
    -------
    dict
        Complete metadata including U10, C10, drift components, etc.
    """
    # Wind profile
    if alpha is None:
        alpha = alpha_for_terrain(terrain)
    U10 = U10_from_Uh_powerlaw(wind_speed, sensor_height, alpha, terrain)

    # Drag coefficient
    C10 = composite_c10(U10)

    # Wind drift
    drift = wind_induced_drift(U10, wind_dir, flow_dir, f)

    return {
        'sensor_height_m': sensor_height,
        'terrain': terrain,
        'alpha': alpha,
        'wind_speed_measured_mps': wind_speed,
        'U10_mps': U10,
        'C10': C10,
        'drift_fraction_f': f,
        'wind_direction_deg': wind_dir,
        'flow_direction_deg': flow_dir,
        'u_w_total_mps': drift['u_w'],
        'u_eff_alongflow_mps': drift['u_eff'],
        'theta_w_rad': drift['theta_w'],
        'theta_f_rad': drift['theta_f']
    }
