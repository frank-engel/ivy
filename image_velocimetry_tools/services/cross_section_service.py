"""
Service layer for cross-section geometry business logic.

This service extracts geometric calculations and data processing operations
from the CrossSectionGeometry GUI component, providing testable methods for
cross-section analysis.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional


class CrossSectionService:
    """
    Service for cross-section geometry calculations.

    This service provides methods for:
    - Geometric calculations (distance, conversions)
    - Station analysis (crossings, interpolation)
    - Data validation (duplicate detection)
    - Wetted width calculations
    - Station transformations
    """

    @staticmethod
    def compute_pixel_distance(points: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance between consecutive x/y points.

        This calculates the distance between consecutive points in a series,
        useful for converting pixel coordinates to real-world distances.

        Parameters
        ----------
        points : np.ndarray
            Array of shape (N, 2) containing [x, y] coordinates

        Returns
        -------
        np.ndarray
            Array of distances between consecutive points, length N-1
        """
        distance = np.sqrt(
            (np.diff(points[:, 0]) ** 2) + (np.diff(points[:, 1]) ** 2)
        )
        return distance

    @staticmethod
    def compute_pixel_to_real_world_conversion(
        pixel_distance: Union[float, np.ndarray],
        real_world_width: float
    ) -> Union[float, np.ndarray]:
        """
        Compute conversion factor from pixel distance to real-world distance.

        Parameters
        ----------
        pixel_distance : float or np.ndarray
            Distance in pixels
        real_world_width : float
            Corresponding real-world distance (meters or feet)

        Returns
        -------
        float or np.ndarray
            Conversion factor (real_world_units per pixel)
        """
        # Handle array input
        if isinstance(pixel_distance, np.ndarray):
            pixel_distance = pixel_distance[0] if len(pixel_distance) > 0 else pixel_distance

        conversion_factor = real_world_width / pixel_distance
        return conversion_factor

    @staticmethod
    def find_station_crossings(
        stations: Union[List, np.ndarray],
        elevations: Union[List, np.ndarray],
        target_elevation: float,
        mode: str = 'all',
        epsilon: float = 1e-1
    ) -> List[float]:
        """
        Find station values where elevations cross a target elevation.

        Uses linear interpolation to find crossing points between the
        elevation profile and a target water surface elevation.

        Parameters
        ----------
        stations : list or np.ndarray
            Station distance values
        elevations : list or np.ndarray
            Elevation values corresponding to stations
        target_elevation : float
            Target elevation to find crossings for
        mode : str, optional
            'all' returns all crossings, 'firstlast' returns only first and last.
            Defaults to 'all'.
        epsilon : float, optional
            Tolerance for numerical precision. Defaults to 1e-1.

        Returns
        -------
        list of float
            Station values where elevations cross the target elevation
        """
        stations = np.array(stations)
        elevations = np.array(elevations)

        # Find indices where interpolation is needed with tolerance
        indices = np.where(
            (
                (elevations[:-1] > target_elevation + epsilon)
                & (elevations[1:] < target_elevation - epsilon)
            )
            | (
                (elevations[:-1] < target_elevation - epsilon)
                & (elevations[1:] > target_elevation + epsilon)
            )
            | (
                (np.abs(elevations[:-1] - target_elevation) <= epsilon)
                & (np.abs(elevations[1:] - target_elevation) > epsilon)
            )
            | (
                (np.abs(elevations[:-1] - target_elevation) > epsilon)
                & (np.abs(elevations[1:] - target_elevation) <= epsilon)
            )
        )[0]

        result_stations = []
        for index in indices:
            x1, x2 = stations[index], stations[index + 1]
            y1, y2 = elevations[index], elevations[index + 1]

            # Linear interpolation formula
            interpolated_station = x1 + (target_elevation - y1) * (x2 - x1) / (y2 - y1)
            result_stations.append(interpolated_station)

        # Filter to first and last if requested
        if (
            mode.lower() == 'firstlast'
            and len(result_stations) > 2
            and len(result_stations) % 2 == 0
        ):
            result_stations = [result_stations[0], result_stations[-1]]

        return result_stations

    @staticmethod
    def interpolate_elevations(
        stations: Union[List, np.ndarray],
        elevations: Union[List, np.ndarray],
        target_stations: Union[List, np.ndarray]
    ) -> np.ndarray:
        """
        Interpolate elevations at target station locations.

        Uses linear interpolation to estimate elevations at specified
        station locations based on known station-elevation pairs.

        Parameters
        ----------
        stations : list or np.ndarray
            Known station values
        elevations : list or np.ndarray
            Known elevation values at stations
        target_stations : list or np.ndarray
            Station values where elevations should be interpolated

        Returns
        -------
        np.ndarray
            Interpolated elevation values at target stations
        """
        stations = np.array(stations)
        elevations = np.array(elevations)
        target_stations = np.array(target_stations)

        # Use numpy's linear interpolation
        interpolated = np.interp(target_stations, stations, elevations)

        return interpolated

    @staticmethod
    def check_duplicate_stations(
        stations: Union[List, np.ndarray],
        tolerance: float = 1e-6
    ) -> List[int]:
        """
        Check for duplicate station values within a tolerance.

        Identifies indices of stations that are duplicates or nearly
        identical within the specified tolerance.

        Parameters
        ----------
        stations : list or np.ndarray
            Station values to check
        tolerance : float, optional
            Tolerance for considering stations as duplicates.
            Defaults to 1e-6.

        Returns
        -------
        list of int
            Indices of duplicate stations
        """
        if len(stations) == 0:
            return []

        stations = np.array(stations)

        # Create DataFrame for easier duplicate detection
        df = pd.DataFrame({'station': stations, 'index': range(len(stations))})

        # Round to tolerance for comparison
        # Convert tolerance to number of decimal places
        if tolerance > 0:
            decimals = int(-np.log10(tolerance))
            df['rounded'] = df['station'].round(decimals)
        else:
            df['rounded'] = df['station']

        # Find duplicates
        duplicates = df[df.duplicated(subset=['rounded'], keep=False)]

        # Return indices of duplicates
        return duplicates['index'].tolist()

    @staticmethod
    def compute_wetted_width(
        stations: Union[List, np.ndarray],
        elevations: Union[List, np.ndarray],
        water_surface_elevation: float,
        mode: str = 'firstlast'
    ) -> float:
        """
        Compute wetted width at a given water surface elevation.

        Calculates the horizontal distance between the first and last
        crossing of the water surface elevation with the channel bed.

        Parameters
        ----------
        stations : list or np.ndarray
            Station distance values
        elevations : list or np.ndarray
            Elevation values at stations
        water_surface_elevation : float
            Water surface elevation
        mode : str, optional
            Mode for finding crossings. Defaults to 'firstlast'.

        Returns
        -------
        float
            Wetted width (distance between first and last crossing)
        """
        crossings = CrossSectionService.find_station_crossings(
            stations, elevations, water_surface_elevation, mode=mode
        )

        if len(crossings) >= 2:
            wetted_width = crossings[-1] - crossings[0]
        else:
            # No crossings or insufficient crossings
            wetted_width = 0.0

        return wetted_width

    @staticmethod
    def flip_stations(stations: Union[List, np.ndarray]) -> np.ndarray:
        """
        Flip station orientation (reverse bank direction).

        Reverses the station values so that the start becomes the end
        and vice versa. Useful when switching between left and right
        bank references.

        Parameters
        ----------
        stations : list or np.ndarray
            Original station values

        Returns
        -------
        np.ndarray
            Flipped station values
        """
        stations = np.array(stations)

        if len(stations) == 0:
            return stations

        # Reverse: max becomes 0, 0 becomes max
        max_station = np.nanmax(stations)
        flipped = max_station - stations - (0 - np.nanmin(stations))

        # Reverse the order
        flipped = flipped[::-1]

        return flipped

    @staticmethod
    def convert_stations_to_metric(
        stations: Union[List, np.ndarray],
        current_units: str
    ) -> np.ndarray:
        """
        Convert station values to metric units.

        Parameters
        ----------
        stations : list or np.ndarray
            Station values
        current_units : str
            Current units: 'English' or 'Metric'

        Returns
        -------
        np.ndarray
            Station values in metric units
        """
        stations = np.array(stations)

        if current_units == 'English':
            # Convert feet to meters
            stations = stations * 0.3048

        return stations

    @staticmethod
    def convert_elevations_to_metric(
        elevations: Union[List, np.ndarray],
        current_units: str
    ) -> np.ndarray:
        """
        Convert elevation values to metric units.

        Parameters
        ----------
        elevations : list or np.ndarray
            Elevation values
        current_units : str
            Current units: 'English' or 'Metric'

        Returns
        -------
        np.ndarray
            Elevation values in metric units
        """
        elevations = np.array(elevations)

        if current_units == 'English':
            # Convert feet to meters
            elevations = elevations * 0.3048

        return elevations

    @staticmethod
    def validate_station_range(
        station: float,
        min_station: float,
        max_station: float
    ) -> bool:
        """
        Validate that a station value is within acceptable range.

        Parameters
        ----------
        station : float
            Station value to validate
        min_station : float
            Minimum acceptable station
        max_station : float
            Maximum acceptable station

        Returns
        -------
        bool
            True if station is within range, False otherwise
        """
        return min_station <= station <= max_station

    @staticmethod
    def compute_channel_area(
        stations: Union[List, np.ndarray],
        elevations: Union[List, np.ndarray],
        water_surface_elevation: float
    ) -> float:
        """
        Compute cross-sectional area at a given water surface elevation.

        Uses trapezoidal integration to calculate the area between
        the channel bed and water surface.

        Parameters
        ----------
        stations : list or np.ndarray
            Station distance values
        elevations : list or np.ndarray
            Channel bed elevations
        water_surface_elevation : float
            Water surface elevation

        Returns
        -------
        float
            Cross-sectional area
        """
        stations = np.array(stations)
        elevations = np.array(elevations)

        # Compute depths (water surface - bed elevation)
        depths = water_surface_elevation - elevations

        # Only include wetted portions (positive depths)
        depths = np.maximum(depths, 0.0)

        # Use trapezoidal integration
        area = np.trapz(depths, stations)

        return area
