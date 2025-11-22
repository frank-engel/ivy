"""IVy module containing discharge computation functions"""

import numpy as np

def compute_discharge_midsection(cumulative_distances,
                                 average_velocities,
                                 vertical_depths,
                                 return_details=False):
    """
    Compute discharge using the mid-section method as described in ISO 748.

    Parameters:
    - cumulative_distances (list or np.array): List or NumPy array of
      cumulative distances along the measurement transect.
    - average_velocities (list or np.array): List or NumPy array of average
      velocities at different vertical depths.
    - vertical_depths (list or np.array): List or NumPy array of vertical
      depths corresponding to the average velocities.
    - return_details (bool): If True, returns widths, areas, and unit
         discharges.

    Returns:
    - float: Total discharge computed using the mid-section method.
    - float: Total area computed using the mid-section method.
    - (Optional) np.array: Widths, Areas, and Unit Discharges if
         return_details=True.
    """

    # Convert input to NumPy arrays for uniform handling
    cumulative_distances = np.asarray(cumulative_distances, dtype=float)
    average_velocities = np.asarray(average_velocities, dtype=float)
    vertical_depths = np.asarray(vertical_depths, dtype=float)

    # Ensure input arrays have the same length
    if not (len(cumulative_distances) == len(average_velocities) == len(vertical_depths)):
        raise ValueError("Input arrays must have the same length.")

    n = len(average_velocities)

    # Compute widths (vectorized)
    widths = np.empty(n)
    # First station
    widths[0] = np.abs(
        cumulative_distances[1] - cumulative_distances[0]) / 2
    # Last station
    widths[-1] = np.abs(cumulative_distances[-1] - cumulative_distances[
        -2]) / 2
    # Interior stations
    widths[1:-1] = np.abs(cumulative_distances[2:] - cumulative_distances[
                                                     :-2]) / 2

    # Compute areas and unit discharges
    areas = widths * vertical_depths
    unit_discharges = average_velocities * areas

    # Compute total discharge and total area
    total_discharge = np.nansum(unit_discharges)
    total_area = np.nansum(areas)

    if return_details:
        return total_discharge, total_area, widths, areas, unit_discharges
    return total_discharge, total_area


def convert_surface_velocity_rantz(surface_velocity, alpha=0.85, wind_params=None):
    """Convert surface velocity to mean velocity using Rantz (1986) method.

    Parameters
    ----------
    surface_velocity : float or np.ndarray
        Measured surface velocity [m/s]
    alpha : float, optional
        Rantz velocity coefficient. Default = 0.85.
    wind_params : dict, optional
        Wind correction parameters. If provided, surface velocity is
        corrected for wind-induced drift before applying alpha.

        Required keys:
            'wind_speed': float, measured wind speed [m/s]
            'wind_dir': float, wind direction [degrees, geographic]
            'flow_dir': float, flow direction [degrees, geographic]

        Optional keys:
            'sensor_height': float, wind sensor height [m], default 10.0
            'terrain': str, terrain type, default 'open_terrain'
            'alpha_wind': float, power-law exponent (overrides terrain)
            'f': float, wind drift fraction, default 0.03

    Returns
    -------
    float or np.ndarray
        Mean velocity [m/s]

    Examples
    --------
    Basic usage without wind correction:
    >>> v_mean = convert_surface_velocity_rantz(1.5)

    With wind correction:
    >>> wind_params = {
    ...     'wind_speed': 5.0,
    ...     'wind_dir': 270,
    ...     'flow_dir': 90,
    ...     'sensor_height': 9.144
    ... }
    >>> v_mean = convert_surface_velocity_rantz(1.5, wind_params=wind_params)
    """
    # Apply wind correction if parameters provided
    if wind_params is not None:
        try:
            from . import wind_correction
        except ImportError:
            raise ImportError(
                "wind_correction module required for wind correction. "
                "Ensure wind_correction.py is in image_velocimetry_tools/"
            )

        # Validate required parameters
        required = ['wind_speed', 'wind_dir', 'flow_dir']
        missing = [k for k in required if k not in wind_params]
        if missing:
            raise ValueError(
                f"wind_params missing required keys: {missing}. "
                f"Required: {required}"
            )

        # Extract parameters with defaults
        wind_speed = wind_params['wind_speed']
        wind_dir = wind_params['wind_dir']
        flow_dir = wind_params['flow_dir']
        sensor_height = wind_params.get('sensor_height', 10.0)
        terrain = wind_params.get('terrain', 'open_terrain')
        alpha_wind = wind_params.get('alpha_wind', None)
        f = wind_params.get('f', 0.03)

        # Apply correction
        surface_velocity = wind_correction.apply_wind_correction(
            surface_velocity,
            wind_speed,
            wind_dir,
            flow_dir,
            sensor_height=sensor_height,
            terrain=terrain,
            alpha=alpha_wind,
            f=f
        )

    # Apply Rantz coefficient
    if isinstance(surface_velocity, (float, int, np.float64)):
        return alpha * surface_velocity
    elif isinstance(surface_velocity, np.ndarray):
        return alpha * surface_velocity
    else:
        raise ValueError("Input must be a float or a numpy array with dtype float.")
