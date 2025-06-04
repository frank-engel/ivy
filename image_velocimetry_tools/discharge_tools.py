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


def convert_surface_velocity_rantz(surface_velocity, alpha=0.85):
    """Convert a surface velocity into a mean velocity using the Rantz (1986) method."""
    if isinstance(surface_velocity, (float, int, np.float64)):
        return alpha * surface_velocity
    elif isinstance(surface_velocity, np.ndarray):
        return alpha * surface_velocity
    else:
        raise ValueError("Input must be a float or a numpy array with dtype float.")
