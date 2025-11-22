"""
DischargeService - Business logic for discharge computations

This service handles discharge calculations, uncertainty analysis, and
cross-section data processing.
"""

import copy
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from image_velocimetry_tools.discharge_tools import (
    compute_discharge_midsection,
    convert_surface_velocity_rantz,
)
from image_velocimetry_tools.uncertainty import Uncertainty


class DischargeService:
    """Service for discharge computation business logic."""

    def __init__(self):
        """Initialize the DischargeService."""
        self.logger = logging.getLogger(self.__class__.__name__)

    # ==================== Cross-Section Data ====================

    def get_station_and_depth_from_grid(
        self,
        xs_survey,
        grid_points: np.ndarray,
        water_surface_elevation: float,
        xs_line_endpoints: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get station distances and depths from cross-section survey for batch processing.

        This method converts grid points (world coordinates) to station distances
        along the cross-section and interpolates elevations from the AC3 data.

        Args:
            xs_survey: AreaSurvey instance with loaded AC3 data
            grid_points: Grid points in world coordinates (N x 2) [X, Y]
            water_surface_elevation: Water surface elevation in meters (SI)
            xs_line_endpoints: Cross-section line endpoints (2 x 2) [[x1, y1], [x2, y2]]

        Returns:
            Tuple of (stations, depths) as numpy arrays in SI units (meters)
        """
        # Compute distances of grid points along the cross-section line
        # The cross-section line defines the direction, grid points should lie along it

        # Extract endpoints
        start_point = xs_line_endpoints[0]  # [x1, y1]
        end_point = xs_line_endpoints[1]    # [x2, y2]

        # Compute pixel distances along the line
        # For each grid point, compute distance from start point
        pixel_distances = np.sqrt(
            (grid_points[:, 0] - start_point[0])**2 +
            (grid_points[:, 1] - start_point[1])**2
        )

        # Total pixel distance (length of cross-section line in pixels)
        total_pixel_distance = np.sqrt(
            (end_point[0] - start_point[0])**2 +
            (end_point[1] - start_point[1])**2
        )

        # Get AC3 cross-section data
        # IMPORTANT: xs_survey.survey DataFrame may be in English or Metric units
        # Check xs_survey.units and convert to SI if needed
        stations_ac3 = xs_survey.survey["Stations"].to_numpy()
        elevations_ac3 = xs_survey.survey["AdjustedStage"].to_numpy()

        # Check units and convert to SI (meters) if needed
        if xs_survey.units == "English":
            self.logger.info("AC3 cross-section is in English units, converting to SI")
            # Convert feet to meters (1 ft = 0.3048 m)
            stations_ac3 = stations_ac3 * 0.3048
            elevations_ac3 = elevations_ac3 * 0.3048
            # Also convert water_surface_elevation context for crossings calculation
            wse_for_crossings = water_surface_elevation  # Already in SI
        else:
            wse_for_crossings = water_surface_elevation

        # Find crossings at water surface elevation to get wetted width
        from image_velocimetry_tools.services.cross_section_service import CrossSectionService
        xs_service = CrossSectionService()

        crossings = xs_service.find_station_crossings(
            stations_ac3,
            elevations_ac3,
            wse_for_crossings,
            mode='firstlast'
        )

        if len(crossings) < 2:
            # No valid wetted width, use full cross-section width
            wetted_width = np.max(stations_ac3) - np.min(stations_ac3)
            left_edge = np.min(stations_ac3)
        else:
            wetted_width = crossings[-1] - crossings[0]
            left_edge = crossings[0]

        # Convert pixel distances to real-world station distances
        conversion_factor = wetted_width / total_pixel_distance
        real_world_distances = pixel_distances * conversion_factor

        # Add offset to match AC3 station coordinate system
        stations = real_world_distances + left_edge

        # Interpolate elevations at these stations from AC3 data
        elevations = xs_service.interpolate_elevations(
            stations_ac3,
            elevations_ac3,
            stations
        )

        # Convert elevations to depths
        depths = water_surface_elevation - elevations

        self.logger.debug(
            f"Converted {len(stations)} grid points to stations and depths. "
            f"Wetted width: {wetted_width:.2f}m, Station range: {stations[0]:.2f} to {stations[-1]:.2f}m"
        )

        return stations, depths

    def get_station_and_depth(
        self,
        xs_survey,
        grid_points: np.ndarray,
        water_surface_elevation: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get station distances and depths from cross-section survey.

        Args:
            xs_survey: CrossSectionGeometry instance with AC3 backend (GUI mode)
            grid_points: Grid points in pixel coordinates (N x 2)
            water_surface_elevation: Water surface elevation in meters

        Returns:
            Tuple of (stations, depths) as numpy arrays
        """
        # Get station and elevation from AC3 (GUI mode with get_pixel_xs method)
        stations, elevations = xs_survey.get_pixel_xs(grid_points)

        # AreaComp returns elevations (stage), convert to depths
        depths = water_surface_elevation - elevations

        self.logger.debug(
            f"Extracted {len(stations)} stations with depths from cross-section"
        )

        return stations, depths

    # ==================== Velocity Extraction ====================

    def extract_velocity_from_stiv(
        self,
        stiv_results,
        add_edge_zeros: bool = True
    ) -> np.ndarray:
        """
        Extract surface velocities from STIV results.

        Args:
            stiv_results: STIV results object with magnitudes and directions
            add_edge_zeros: If True, add zero velocity at edges

        Returns:
            Surface velocities as numpy array
        """
        # Get velocity components
        D = np.radians(stiv_results.directions)  # Convert to radians
        U = stiv_results.magnitudes_mps * np.cos(D)
        V = stiv_results.magnitudes_mps * np.sin(D)

        # Use normal magnitudes if available, otherwise compute from components
        if (stiv_results.magnitude_normals_mps is not None and
                np.any(stiv_results.magnitude_normals_mps)):
            M = stiv_results.magnitude_normals_mps
        else:
            M = np.sqrt(U**2 + V**2)

        surface_velocities = M

        # Add zero velocities at edges
        if add_edge_zeros:
            surface_velocities = np.insert(surface_velocities, 0, 0)
            surface_velocities = np.append(surface_velocities, 0)

        self.logger.debug(
            f"Extracted {len(surface_velocities)} surface velocities from STIV"
        )

        return surface_velocities

    # ==================== Discharge Data Creation ====================

    def create_discharge_dataframe(
        self,
        stations: np.ndarray,
        depths: np.ndarray,
        surface_velocities: np.ndarray,
        alpha: float = 0.85,
        existing_status: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Create discharge stations dataframe with computed widths, areas, and unit discharges.

        All values are in metric units (meters, m/s, m^2, m^3/s).

        Args:
            stations: Station distances along cross-section (m)
            depths: Depths at each station (m)
            surface_velocities: Surface velocities at each station (m/s)
            alpha: Alpha coefficient (default 0.85 per Rantz)
            existing_status: Optional array of "Used"/"Not Used" status

        Returns:
            DataFrame with discharge data (in metric units)
        """
        num_stations = len(stations)

        # Determine status for each station
        if existing_status is not None:
            status = existing_status
        else:
            # All stations are "Used" initially
            status = np.full(num_stations, "Used", dtype=object)

        # Initialize data dictionary
        data = {
            "ID": np.arange(0, num_stations),
            "Status": status,
            "Station Distance": stations,
            "Width": np.zeros(num_stations),
            "Depth": depths,
            "Area": np.zeros(num_stations),
            "Surface Velocity": surface_velocities,
            "α (alpha)": alpha,
            "Unit Discharge": np.zeros(num_stations),
        }

        # Compute widths and areas for middle stations
        for i in range(1, num_stations - 1):
            data["Width"][i] = (
                data["Station Distance"][i + 1] -
                data["Station Distance"][i - 1]
            ) / 2
            data["Area"][i] = data["Width"][i] * data["Depth"][i]

        # Compute width and area for first station (left edge)
        data["Width"][0] = (
            data["Station Distance"][1] - data["Station Distance"][0]
        ) / 2
        data["Area"][0] = (data["Width"][0] / 2) * data["Depth"][0]

        # Compute width and area for last station (right edge)
        data["Width"][-1] = (
            data["Station Distance"][-1] - data["Station Distance"][-2]
        ) / 2
        data["Area"][-1] = (data["Width"][-1] / 2) * data["Depth"][-1]

        # Compute unit discharge (handle NaN velocities)
        surface_vel_clean = np.nan_to_num(data["Surface Velocity"], nan=0)
        data["Unit Discharge"] = data["Area"] * surface_vel_clean * data["α (alpha)"]

        # Create DataFrame
        df = pd.DataFrame(data)

        # Reorder columns
        df = df[
            [
                "ID",
                "Status",
                "Station Distance",
                "Width",
                "Depth",
                "Area",
                "Surface Velocity",
                "α (alpha)",
                "Unit Discharge",
            ]
        ]

        self.logger.info(
            f"Created discharge dataframe with {num_stations} stations"
        )

        return df

    # ==================== Discharge Computation ====================

    def compute_discharge(
        self,
        discharge_dataframe: pd.DataFrame
    ) -> Dict:
        """
        Compute total discharge and area from discharge dataframe.

        Uses the mid-section method. Only processes stations marked as "Used".

        Args:
            discharge_dataframe: DataFrame with discharge data (metric units)

        Returns:
            Dictionary with:
                - total_discharge: Total discharge (m^3/s)
                - total_area: Total cross-sectional area (m^2)
                - discharge_results: Per-station results dict
        """
        # Filter to only "Used" stations
        used_df = discharge_dataframe[discharge_dataframe["Status"] == "Used"]

        if used_df.empty:
            self.logger.warning("No stations marked as 'Used' for discharge computation")
            return {
                "total_discharge": 0.0,
                "total_area": 0.0,
                "discharge_results": {}
            }

        # Extract required columns
        cumulative_distances = used_df["Station Distance"].astype(float).values
        surface_velocities = used_df["Surface Velocity"].astype(float).values
        alpha_values = used_df["α (alpha)"].astype(float).values

        # Convert surface velocities to average velocities using Rantz method
        average_velocities = convert_surface_velocity_rantz(
            surface_velocities, alpha=alpha_values
        )

        vertical_depths = used_df["Depth"].astype(float).values

        # Compute discharge and area using mid-section method
        total_discharge, total_area = compute_discharge_midsection(
            cumulative_distances, average_velocities, vertical_depths
        )

        self.logger.info(
            f"Computed total discharge: {total_discharge:.3f} m^3/s, "
            f"total area: {total_area:.2f} m^2"
        )

        return {
            "total_discharge": total_discharge,
            "total_area": total_area,
            "discharge_results": discharge_dataframe.to_dict(orient="index")
        }

    # ==================== Uncertainty Computation ====================

    def compute_uncertainty(
        self,
        discharge_results: Dict,
        total_discharge: float,
        rectification_rmse: float,
        scene_width: float
    ) -> Dict:
        """
        Compute discharge measurement uncertainty using ISO and IVE methods.

        Args:
            discharge_results: Per-station discharge results
            total_discharge: Total discharge (m^3/s)
            rectification_rmse: Rectification RMSE (m)
            scene_width: Scene width (m)

        Returns:
            Dictionary with:
                - u_iso: ISO uncertainty components and total
                - u_ive: IVE uncertainty components and total
                - u_iso_contribution: ISO uncertainty contributions
                - u_ive_contribution: IVE uncertainty contributions
        """
        q_dict = copy.deepcopy(discharge_results)
        ortho_info = {
            "rmse_m": rectification_rmse,
            "scene_width_m": scene_width
        }

        uncertainty = Uncertainty()
        uncertainty.compute_uncertainty(
            q_dict,
            total_discharge=total_discharge,
            ortho_info=ortho_info
        )

        self.logger.info(
            f"Computed uncertainty - ISO 95% CI: {uncertainty.u_iso['u95_q'] * 100:.2f}%, "
            f"IVE 95% CI: {uncertainty.u_ive['u95_q'] * 100:.2f}%"
        )

        return {
            "u_iso": uncertainty.u_iso,
            "u_ive": uncertainty.u_ive,
            "u_iso_contribution": uncertainty.u_iso_contribution,
            "u_ive_contribution": uncertainty.u_ive_contribution,
        }

    # ==================== Summary Statistics ====================

    def compute_summary_statistics(
        self,
        discharge_results: Dict,
        total_discharge: float,
        total_area: float
    ) -> Dict:
        """
        Compute summary statistics from discharge results.

        Args:
            discharge_results: Per-station discharge results
            total_discharge: Total discharge (m^3/s)
            total_area: Total area (m^2)

        Returns:
            Dictionary with summary statistics (metric units)
        """
        # Extract values
        surface_velocities = [
            float(result["Surface Velocity"]) for result in discharge_results.values()
        ]
        alphas = [
            float(result["α (alpha)"]) for result in discharge_results.values()
        ]

        # Calculate statistics
        average_velocity = total_discharge / total_area if total_area > 0 else 0
        average_alpha = np.nansum(alphas) / len(alphas) if alphas else 0
        average_surface_velocity = (
            np.nansum(surface_velocities) / len(surface_velocities)
            if surface_velocities else 0
        )
        max_surface_velocity = np.nanmax(surface_velocities) if surface_velocities else 0

        return {
            "average_velocity": average_velocity,
            "average_alpha": average_alpha,
            "average_surface_velocity": average_surface_velocity,
            "max_surface_velocity": max_surface_velocity,
        }
