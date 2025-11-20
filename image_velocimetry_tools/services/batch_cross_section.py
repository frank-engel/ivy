"""Batch Cross-Section Service - Headless wrapper for cross-section operations.

This module provides a lightweight cross-section interface for batch processing
that wraps AreaSurvey without GUI dependencies.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

try:
    from areacomp.gui.projectdata import ProjectData
    PROJECTDATA_AVAILABLE = True
except ImportError:
    PROJECTDATA_AVAILABLE = False
    ProjectData = None


class BatchCrossSectionWrapper:
    """Lightweight cross-section wrapper for batch processing.

    This class wraps an AreaSurvey object and provides the interface needed
    by DischargeService without requiring a full GUI parent. It implements
    the minimal set of methods needed for discharge calculations.

    Attributes:
        xs_survey: AreaSurvey object with loaded bathymetry
        cross_section_line: Cross-section line geometry (2D points)
        start_bank: Starting bank ("left" or "right")
        display_units: Display units ("English" or "Metric")
        water_surface_elevation_m: Water surface elevation in meters
    """

    def __init__(
        self,
        xs_survey,
        cross_section_line: np.ndarray,
        start_bank: str = "left",
        display_units: str = "English",
        water_surface_elevation_m: Optional[float] = None
    ):
        """Initialize batch cross-section wrapper.

        Args:
            xs_survey: AreaSurvey object with loaded bathymetry
            cross_section_line: Cross-section line geometry (Nx2 array of [x, y] points)
            start_bank: Starting bank ("left" or "right")
            display_units: Display units ("English" or "Metric")
            water_surface_elevation_m: Water surface elevation in meters (optional)

        Raises:
            ValueError: If xs_survey is None or invalid
        """
        if xs_survey is None:
            raise ValueError("xs_survey cannot be None")

        self.xs_survey = xs_survey
        self.cross_section_line = cross_section_line
        self.start_bank = start_bank
        self.display_units = display_units
        self.water_surface_elevation_m = water_surface_elevation_m
        self.logger = logging.getLogger(__name__)

        # Set water surface elevation on xs_survey if provided
        if water_surface_elevation_m is not None:
            self._set_water_surface_elevation(water_surface_elevation_m)

    def _set_water_surface_elevation(self, wse_m: float):
        """Set water surface elevation on the xs_survey object.

        Args:
            wse_m: Water surface elevation in meters
        """
        # AreaComp uses feet for stage, convert from meters
        wse_ft = wse_m * 3.28084
        self.xs_survey.stage = wse_ft
        self.xs_survey.max_stage = wse_ft
        self.logger.debug(f"Set water surface elevation: {wse_m:.3f} m ({wse_ft:.3f} ft)")

    def get_pixel_xs(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get station distances and elevations for given world coordinate points.

        This method converts world coordinate points (from grid) to:
        1. Station distances along the cross-section
        2. Interpolated elevations from the bathymetry

        Args:
            points: Array of world coordinate points (Nx2, columns: [x, y])

        Returns:
            Tuple of (stations, elevations):
                - stations: Station distances in meters
                - elevations: Interpolated elevations in meters

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If computation fails
        """
        if points is None or len(points) == 0:
            raise ValueError("points cannot be None or empty")

        try:
            # Get the cross-section line extents (first and last points)
            if self.cross_section_line is None or len(self.cross_section_line) < 2:
                raise ValueError("cross_section_line must have at least 2 points")

            point_extents = self.cross_section_line[[0, -1], :]  # First and last points

            self.logger.debug(
                f"Cross-section line extents: "
                f"({point_extents[0,0]:.2f}, {point_extents[0,1]:.2f}) to "
                f"({point_extents[1,0]:.2f}, {point_extents[1,1]:.2f})"
            )
            self.logger.debug(f"Processing {len(points)} grid points")

            # Combine extents with grid points, then compute projections
            new_arr = np.insert(point_extents, 1, points, axis=0)
            df = pd.DataFrame(new_arr, columns=["x", "y"])

            if not PROJECTDATA_AVAILABLE:
                raise RuntimeError(
                    "areacomp ProjectData not available. Cannot compute station projections."
                )

            # Use ProjectData to compute station positions along the cross-section
            proj = ProjectData()
            proj.compute_data(df, rtn=True)

            self.logger.debug(f"ProjectData stations range: [{proj.stations.min():.2f}, {proj.stations.max():.2f}]")

            # Compute pixel-to-distance conversion factor
            pixel_dist = self._compute_pixel_distance(point_extents)
            self.logger.debug(f"Pixel distance (line length): {pixel_dist:.2f} pixels")

            # Get wetted width from cross-section
            wetted_width, station_offset = self._compute_wetted_width()
            self.logger.debug(
                f"Wetted width: {wetted_width:.2f} {self.display_units}, "
                f"station offset: {station_offset:.2f}"
            )

            # Convert pixel distances to real-world stations
            p_conversion = pixel_dist / wetted_width if wetted_width > 0 else 1.0
            self.logger.debug(f"Pixel conversion factor: {p_conversion:.4f}")

            pixel_stations = proj.stations * p_conversion + station_offset

            self.logger.debug(
                f"Pixel stations range before flipping: "
                f"[{pixel_stations.min():.2f}, {pixel_stations.max():.2f}]"
            )

            # Handle right-bank start (flip stations)
            if self.start_bank == "right":
                pixel_stations = (
                    np.nanmax(pixel_stations)
                    - pixel_stations
                    - (0 - np.nanmin(pixel_stations))
                )
                self.logger.debug("Flipped stations for right-bank start")

            # Interpolate elevations from bathymetry
            elevations = self._interpolate_elevations(pixel_stations)

            # Convert to metric if needed
            if self.display_units == "English":
                elevations_ft = elevations.copy()
                elevations = elevations * 0.3048  # Convert feet to meters
                self.logger.debug(
                    f"Converted elevations from feet to meters: "
                    f"range [{elevations_ft.min():.2f}, {elevations_ft.max():.2f}] ft -> "
                    f"[{elevations.min():.2f}, {elevations.max():.2f}] m"
                )

            self.logger.info(
                f"Computed cross-section: "
                f"{len(pixel_stations)} stations from {pixel_stations.min():.2f} to {pixel_stations.max():.2f} m, "
                f"elevations from {elevations.min():.2f} to {elevations.max():.2f} m"
            )

            return pixel_stations, elevations

        except Exception as e:
            self.logger.error(f"Failed to compute pixel cross-section: {e}")
            raise RuntimeError(f"get_pixel_xs failed: {e}")

    def _compute_pixel_distance(self, points: np.ndarray) -> float:
        """Compute Euclidean distance between two points.

        Args:
            points: Array of 2 points (2x2: [[x1, y1], [x2, y2]])

        Returns:
            Distance in pixels
        """
        if len(points) != 2:
            raise ValueError("points must contain exactly 2 points")

        dx = points[1, 0] - points[0, 0]
        dy = points[1, 1] - points[0, 1]
        return np.sqrt(dx**2 + dy**2)

    def _compute_wetted_width(self) -> Tuple[float, float]:
        """Compute wetted width from cross-section survey.

        Returns:
            Tuple of (wetted_width, station_offset):
                - wetted_width: Wetted width at current stage
                - station_offset: Station of left edge of wetted width

        Raises:
            ValueError: If survey data is invalid
        """
        try:
            # Get current stage (water surface elevation)
            stage = self.xs_survey.stage

            # Find stations where cross-section intersects water surface
            stations = self.xs_survey.survey["Stations"].to_numpy()
            adjusted_stage = self.xs_survey.survey["AdjustedStage"].to_numpy()

            self.logger.debug(
                f"Computing wetted width: WSE={stage:.2f} {self.display_units}, "
                f"{len(stations)} survey points from {stations.min():.2f} to {stations.max():.2f}"
            )
            self.logger.debug(
                f"Survey elevations: range [{adjusted_stage.min():.2f}, {adjusted_stage.max():.2f}] {self.display_units}"
            )

            # Find first and last crossing of water surface
            crossings = self._find_station_crossings(
                stations, adjusted_stage, stage
            )

            if len(crossings) < 2:
                self.logger.error(
                    f"Only found {len(crossings)} water surface crossings (need 2). "
                    f"WSE={stage:.2f}, survey elevation range=[{adjusted_stage.min():.2f}, {adjusted_stage.max():.2f}]"
                )
                raise ValueError(
                    f"Cannot compute wetted width: found {len(crossings)} crossings, need at least 2"
                )

            wetted_width = crossings[-1] - crossings[0]
            station_offset = crossings[0]

            self.logger.debug(
                f"Found {len(crossings)} crossings: {crossings}, "
                f"wetted width: {wetted_width:.2f}, offset: {station_offset:.2f}"
            )

            return wetted_width, station_offset

        except Exception as e:
            self.logger.error(f"Failed to compute wetted width: {e}")
            raise ValueError(f"Wetted width computation failed: {e}")

    def _find_station_crossings(
        self,
        stations: np.ndarray,
        elevations: np.ndarray,
        target_elevation: float
    ) -> np.ndarray:
        """Find station positions where elevation crosses target elevation.

        For a typical river channel cross-section:
        - Left bank: high elevation (dry, above water)
        - Channel: low elevation (wet, below water)
        - Right bank: high elevation (dry, above water)

        We need to find where elevation crosses WSE going from:
        - Left crossing: dry→wet (above→below water surface)
        - Right crossing: wet→dry (below→above water surface)

        Args:
            stations: Station distances
            elevations: Elevations at each station
            target_elevation: Target elevation (water surface)

        Returns:
            Array of station positions where elevation crosses target
        """
        # Find where elevation is above water (dry)
        above = elevations >= target_elevation
        crossings = []

        # Find LEFT crossing (dry to wet: above→below water)
        # This is where we ENTER the water from the left bank
        for i in range(len(above) - 1):
            if above[i] and not above[i + 1]:  # above→below (DRY to WET)
                # Linear interpolation to find exact crossing
                t = (target_elevation - elevations[i]) / (elevations[i + 1] - elevations[i])
                crossing = stations[i] + t * (stations[i + 1] - stations[i])
                crossings.append(crossing)
                self.logger.debug(f"Found left crossing (dry→wet) at station {crossing:.2f}")
                break

        # Find RIGHT crossing (wet to dry: below→above water)
        # This is where we EXIT the water to the right bank
        for i in range(len(above) - 1, 0, -1):
            if not above[i - 1] and above[i]:  # below→above (WET to DRY)
                # Linear interpolation
                t = (target_elevation - elevations[i - 1]) / (elevations[i] - elevations[i - 1])
                crossing = stations[i - 1] + t * (stations[i] - stations[i - 1])
                crossings.append(crossing)
                self.logger.debug(f"Found right crossing (wet→dry) at station {crossing:.2f}")
                break

        return np.array(crossings)

    def _interpolate_elevations(self, target_stations: np.ndarray) -> np.ndarray:
        """Interpolate elevations at target stations.

        Args:
            target_stations: Target station distances

        Returns:
            Interpolated elevations
        """
        try:
            # Get survey data
            survey_stations = self.xs_survey.survey["Stations"].to_numpy()
            survey_elevations = self.xs_survey.survey["AdjustedStage"].to_numpy()

            # Interpolate
            elevations = np.interp(
                target_stations,
                survey_stations,
                survey_elevations,
                left=np.nan,
                right=np.nan
            )

            return elevations

        except Exception as e:
            self.logger.error(f"Failed to interpolate elevations: {e}")
            raise ValueError(f"Elevation interpolation failed: {e}")
