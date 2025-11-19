"""
Service layer for STIV (Space-Time Image Velocimetry) business logic.

This service extracts business logic from the STIV and STI Review tabs,
providing testable methods for velocity/angle conversions, manual velocity
processing, data loading, and STIV optimization calculations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging

from image_velocimetry_tools.common_functions import (
    load_csv_with_numpy,
    component_in_direction,
    geographic_to_arithmetic,
)
from image_velocimetry_tools.stiv import optimum_stiv_sample_time, two_dimensional_stiv_exhaustive
from image_velocimetry_tools.image_processing_tools import create_grayscale_image_stack


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

    @staticmethod
    def process_stiv(
        frame_files: List[str],
        grid_points: np.ndarray,
        phi_origin: float,
        phi_range: float,
        dphi: float,
        num_pixels: int,
        pixel_gsd: float,
        timestep_seconds: float,
        gaussian_blur_sigma: float = 0.0,
        max_vel_threshold_mps: float = 10.0,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, np.ndarray]:
        """Process STIV velocimetry on a set of frames.

        This method performs the complete STIV workflow:
        1. Creates grayscale image stack from frame files
        2. Runs exhaustive STIV search at each grid point
        3. Returns velocity magnitudes, directions, and STI images

        Suitable for both GUI (with progress callback) and batch/headless
        processing.

        Parameters
        ----------
        frame_files : List[str]
            List of frame file paths (sorted in temporal order)
        grid_points : np.ndarray
            Grid points in pixel coordinates, shape (N, 2) where N is
            number of grid points. Each row is [x, y].
        phi_origin : float
            Search origin angle in degrees (geographic convention)
        phi_range : float
            Search range in degrees (+/- from origin)
        dphi : float
            Search angle step in degrees
        num_pixels : int
            Number of pixels along search line
        pixel_gsd : float
            Pixel ground scale distance (pixel size) in meters
        timestep_seconds : float
            Time step between frames in seconds
        gaussian_blur_sigma : float, optional
            Gaussian blur sigma for STI preprocessing (default: 0.0 = no blur)
        max_vel_threshold_mps : float, optional
            Maximum velocity threshold in m/s (default: 10.0)
        progress_callback : Optional[Callable[[int, str], None]], optional
            Optional callback(percent, message) for progress updates

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing STIV results:
            - 'magnitudes_mps': Velocity magnitudes in m/s, shape (N,)
            - 'directions_deg': Flow directions in degrees (geographic), shape (N,)
            - 'st_images': Space-time images, shape (N, num_pixels, num_frames)
            - 'theta_angles': STI streak angles in degrees, shape (N,)

        Raises
        ------
        ValueError
            If inputs are invalid (empty frame list, invalid grid, etc.)
        RuntimeError
            If STIV processing fails

        Notes
        -----
        This method uses the exhaustive STIV search algorithm which evaluates
        all angles in the search range. For large datasets, consider using
        parallel processing or optimized algorithms.

        The progress_callback signature is: callback(percent: int, message: str)
        where percent is 0-100.
        """
        logger = logging.getLogger(__name__)

        # Validate inputs
        if len(frame_files) == 0:
            raise ValueError("frame_files cannot be empty")

        if grid_points.shape[0] == 0:
            raise ValueError("grid_points cannot be empty")

        if grid_points.shape[1] != 2:
            raise ValueError("grid_points must have shape (N, 2)")

        if num_pixels < 1:
            raise ValueError("num_pixels must be >= 1")

        if pixel_gsd <= 0:
            raise ValueError("pixel_gsd must be positive")

        if timestep_seconds <= 0:
            raise ValueError("timestep_seconds must be positive")

        logger.info(
            f"Running STIV on {len(frame_files)} frames with "
            f"{len(grid_points)} grid points"
        )

        # Report progress
        if progress_callback:
            progress_callback(5, "Creating grayscale image stack...")

        # Create grayscale image stack
        try:
            logger.debug("Creating grayscale image stack...")
            image_stack = create_grayscale_image_stack(frame_files)
            logger.debug(f"Image stack shape: {image_stack.shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to create image stack: {e}")

        # Report progress
        if progress_callback:
            progress_callback(15, f"Processing {len(grid_points)} grid points...")

        # Create simple progress wrapper for core STIV function
        # The core STIV function expects an object with an emit() method
        class ProgressWrapper:
            def __init__(self, callback):
                self.callback = callback
                self.logger = logging.getLogger(__name__)

            def emit(self, value):
                # Map STIV progress (0-100) to overall progress (15-95)
                overall_percent = int(15 + (value * 0.8))
                if self.callback:
                    self.callback(overall_percent, f"STIV processing... {value}%")
                if value % 20 == 0:
                    self.logger.debug(f"STIV progress: {value}%")

        progress_wrapper = ProgressWrapper(progress_callback)

        # Run STIV exhaustive search
        logger.debug(f"Running STIV exhaustive search...")
        logger.debug(f"  Grid points: {len(grid_points)}")
        logger.debug(
            f"  Phi origin: {phi_origin}°, range: {phi_range}°, dphi: {dphi}°"
        )
        logger.debug(f"  Search line pixels: {num_pixels}")
        logger.debug(f"  Pixel GSD: {pixel_gsd} m")
        logger.debug(f"  Timestep: {timestep_seconds} s")

        try:
            magnitudes_mps, directions_deg, st_images, theta_angles = \
                two_dimensional_stiv_exhaustive(
                    x_origin=grid_points[:, 0].astype(float),
                    y_origin=grid_points[:, 1].astype(float),
                    image_stack=image_stack,
                    num_pixels=num_pixels,
                    phi_origin=phi_origin,
                    d_phi=dphi,
                    phi_range=phi_range,
                    pixel_gsd=pixel_gsd,
                    d_t=timestep_seconds,
                    sigma=gaussian_blur_sigma,
                    max_vel_threshold=max_vel_threshold_mps,
                    progress_signal=progress_wrapper,
                )
        except Exception as e:
            raise RuntimeError(f"STIV processing failed: {e}")

        # Report statistics
        mean_vel = np.nanmean(magnitudes_mps)
        max_vel = np.nanmax(magnitudes_mps)
        num_valid = np.sum(~np.isnan(magnitudes_mps))

        logger.info(
            f"STIV complete. Mean velocity: {mean_vel:.3f} m/s, "
            f"Max velocity: {max_vel:.3f} m/s, "
            f"Valid points: {num_valid}/{len(magnitudes_mps)}"
        )

        # Report completion
        if progress_callback:
            progress_callback(
                100,
                f"STIV complete: {num_valid} valid velocities"
            )

        # Return structured results
        return {
            'magnitudes_mps': magnitudes_mps,
            'directions_deg': directions_deg,
            'st_images': st_images,
            'theta_angles': theta_angles,
        }
