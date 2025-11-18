"""
Service layer for STIV (Space-Time Image Velocimetry) business logic.

This service extracts business logic from the STIV and STI Review tabs,
providing testable methods for velocity/angle conversions, manual velocity
processing, data loading, and STIV optimization calculations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

from image_velocimetry_tools.common_functions import (
    load_csv_with_numpy,
    component_in_direction,
    geographic_to_arithmetic,
)
from image_velocimetry_tools.stiv import optimum_stiv_sample_time


class STIVService:
    """
    Service for STIV-related business logic operations.

    This service provides methods for:
    - Velocity and angle conversions
    - Manual velocity processing
    - STIV results data loading
    - Manual corrections application
    - STIV optimization calculations
    """

    @staticmethod
    def compute_sti_velocity(
        theta: Union[float, np.ndarray],
        gsd: float,
        dt: float
    ) -> Union[float, np.ndarray]:
        """
        Calculate velocity from STI angle using Fujita et al. (2007) equation 16.

        The relationship between streak angle (theta) and velocity is:
        velocity = tan(theta) * gsd / dt

        Parameters
        ----------
        theta : float or np.ndarray
            STI streak angle in degrees
        gsd : float
            Ground scale distance (pixel size) in meters
        dt : float
            Frame interval (time step) in seconds

        Returns
        -------
        float or np.ndarray
            Velocity in m/s
        """
        return np.tan(np.deg2rad(theta)) * gsd / dt

    @staticmethod
    def compute_sti_angle(
        velocity: Union[float, np.ndarray],
        gsd: float,
        dt: float
    ) -> Union[float, np.ndarray]:
        """
        Calculate STI angle from velocity (inverse of compute_sti_velocity).

        This implements the inverse of Fujita et al. (2007) equation 16:
        theta = arctan(-velocity * dt / gsd)

        Parameters
        ----------
        velocity : float or np.ndarray
            Velocity in m/s
        gsd : float
            Ground scale distance (pixel size) in meters
        dt : float
            Frame interval (time step) in seconds

        Returns
        -------
        float or np.ndarray
            STI streak angle in degrees
        """
        return np.degrees(np.arctan((-velocity * dt) / gsd))

    @staticmethod
    def compute_velocity_from_manual_angle(
        average_direction: float,
        gsd: float,
        dt: float,
        is_upstream: bool = False
    ) -> float:
        """
        Convert manual angle measurement to velocity magnitude.

        When a user manually draws an angle on an STI image, this method
        converts that angle to a velocity magnitude. The velocity sign
        can be inverted to indicate upstream flow.

        Parameters
        ----------
        average_direction : float
            Average direction angle in degrees from manual measurements.
            Values <= -900 indicate canceled/no manual edit.
        gsd : float
            Ground scale distance (pixel size) in meters
        dt : float
            Frame interval (time step) in seconds
        is_upstream : bool, optional
            If True, velocity is negated to indicate upstream flow.
            Defaults to False.

        Returns
        -------
        float
            Manual velocity in m/s. Returns NaN if manual edit was canceled.
        """
        # Check if user canceled the manual edit (sentinel value)
        if average_direction <= -900:
            return np.nan

        # Convert angle to velocity, take absolute magnitude
        manual_velocity_mps = np.abs(
            np.tan(np.deg2rad(average_direction)) * gsd / dt
        )

        # Apply upstream flow correction if needed
        if is_upstream:
            manual_velocity_mps = -manual_velocity_mps

        return manual_velocity_mps

    @staticmethod
    def load_stiv_results_from_csv(csv_file_path: str) -> Dict[str, np.ndarray]:
        """
        Load STIV results from a CSV file.

        The CSV file is expected to contain columns for X, Y, U, V, Magnitude,
        Scalar_Projection, Direction, Tagline_Direction, and Normal_Direction.

        Parameters
        ----------
        csv_file_path : str
            Path to the STIV results CSV file

        Returns
        -------
        dict
            Dictionary containing STIV result arrays with keys:
            - 'X', 'Y': Grid point coordinates
            - 'U', 'V': Velocity components
            - 'Magnitude': Total velocity magnitude
            - 'Scalar_Projection': Velocity projected along tagline normal
            - 'Direction': Flow direction in geographic degrees
            - 'Tagline_Direction': Tagline orientation
            - 'Normal_Direction': Normal to tagline direction

        Raises
        ------
        FileNotFoundError
            If the CSV file does not exist
        """
        headers, data = load_csv_with_numpy(csv_file_path)

        # Extract columns from the CSV data
        result = {
            'X': data[:, 0].astype(float),
            'Y': data[:, 1].astype(float),
            'U': data[:, 2].astype(float),
            'V': data[:, 3].astype(float),
            'Magnitude': data[:, 4].astype(float),
            'Scalar_Projection': data[:, 5].astype(float),
            'Direction': data[:, 6].astype(float),
            'Tagline_Direction': data[:, 7].astype(float),
            'Normal_Direction': data[:, 8].astype(float),
        }

        return result

    @staticmethod
    def prepare_table_data(
        magnitudes_mps: np.ndarray,
        directions: np.ndarray,
        thetas: Optional[np.ndarray],
        survey_units: Dict[str, float]
    ) -> List[Dict]:
        """
        Prepare STIV results data for display in the STI Review table.

        Parameters
        ----------
        magnitudes_mps : np.ndarray
            Velocity magnitudes in m/s
        directions : np.ndarray
            Flow directions in geographic degrees
        thetas : np.ndarray or None
            STI streak angles in degrees. If None, NaN values are used.
        survey_units : dict
            Units conversion dictionary with keys 'V' (conversion factor)
            and 'label_V' (unit label string)

        Returns
        -------
        list of dict
            List of dictionaries, one per row, containing:
            - 'id': Row ID (1-indexed)
            - 'direction': Flow direction
            - 'theta': STI streak angle
            - 'original_velocity': Original velocity magnitude (converted to display units)
            - 'manual_velocity': Manual velocity (initialized to original, converted to display units)
        """
        num_rows = len(magnitudes_mps)
        table_data = []

        # Handle case where thetas is None
        if thetas is None:
            thetas = np.empty(num_rows)
            thetas[:] = np.nan

        for row, (magnitude, direction, theta) in enumerate(
            zip(magnitudes_mps, directions, thetas)
        ):
            row_data = {
                'id': row + 1,
                'direction': direction,
                'theta': theta,
                'original_velocity': magnitude * survey_units['V'],
                'manual_velocity': magnitude * survey_units['V'],  # Initialize to original
            }
            table_data.append(row_data)

        return table_data

    @staticmethod
    def apply_manual_corrections(
        stiv_data: Dict[str, np.ndarray],
        manual_velocities: np.ndarray,
        manual_indices: List[int],
        tagline_direction: float
    ) -> Dict[str, np.ndarray]:
        """
        Apply manual velocity corrections to STIV results.

        This method recomputes scalar projections for manually edited velocities,
        projecting the new velocity magnitudes along the tagline normal direction.

        Parameters
        ----------
        stiv_data : dict
            Dictionary containing STIV results with keys 'Direction',
            'Scalar_Projection', and 'Tagline_Direction'
        manual_velocities : np.ndarray
            Array of velocity magnitudes (some manually edited) in m/s
        manual_indices : list of int
            Indices of rows that were manually edited
        tagline_direction : float
            Tagline direction in geographic degrees

        Returns
        -------
        dict
            Dictionary containing:
            - 'scalar_projections': Updated scalar projections (m/s)
            - 'manual_indices': List of manually edited indices
        """
        # Start with original scalar projections
        scalar_projections = stiv_data['Scalar_Projection'].copy()
        directions = stiv_data['Direction']

        # Recompute scalar projections for manually modified results
        for i in manual_indices:
            scalar_projections[i] = component_in_direction(
                magnitudes=manual_velocities[i],
                directions_deg=geographic_to_arithmetic(directions[i]),
                tagline_angle_deg=geographic_to_arithmetic(tagline_direction),
            )

        return {
            'scalar_projections': scalar_projections,
            'manual_indices': manual_indices
        }

    @staticmethod
    def compute_optimum_sample_time(gsd: float, velocity: float) -> float:
        """
        Compute the optimum STIV sample time for given conditions.

        This method calls the STIV optimization algorithm to determine
        the best temporal sampling interval for STIV analysis based on
        the pixel resolution and expected flow velocity.

        Parameters
        ----------
        gsd : float
            Ground scale distance (pixel size) in meters
        velocity : float
            Expected flow velocity in m/s

        Returns
        -------
        float
            Optimum sample time in milliseconds
        """
        return optimum_stiv_sample_time(gsd, velocity)

    @staticmethod
    def compute_frame_step(
        sample_time_ms: float,
        video_frame_rate: float
    ) -> int:
        """
        Compute the frame step from sample time and video frame rate.

        The frame step determines how many video frames to skip between
        successive STIV analysis frames.

        Parameters
        ----------
        sample_time_ms : float
            Desired sample time in milliseconds
        video_frame_rate : float
            Video frame rate in frames per second

        Returns
        -------
        int
            Frame step (number of frames to skip). Minimum value is 1.
        """
        # Calculate video frame interval in milliseconds
        video_frame_interval_ms = 1000.0 / video_frame_rate

        # Calculate frame step
        frame_step = round(sample_time_ms / video_frame_interval_ms)

        # Ensure frame step is at least 1
        if frame_step < 1:
            frame_step = 1

        return frame_step

    @staticmethod
    def compute_sample_time_seconds(
        frame_step: int,
        video_frame_rate: float
    ) -> float:
        """
        Compute actual sample time in seconds from frame step and frame rate.

        This is the inverse of compute_frame_step, calculating the actual
        temporal interval based on the integer frame step.

        Parameters
        ----------
        frame_step : int
            Number of frames to skip between STIV analysis frames
        video_frame_rate : float
            Video frame rate in frames per second

        Returns
        -------
        float
            Actual sample time in seconds
        """
        # Calculate video frame interval in milliseconds
        video_frame_interval_ms = 1000.0 / video_frame_rate

        # Calculate sample time in seconds
        sample_time_s = frame_step * video_frame_interval_ms / 1000.0

        return sample_time_s
