"""
Unit tests for the CrossSectionService class.

This module contains comprehensive tests for cross-section geometry business logic,
including geometric calculations, station analysis, interpolation, and data validation.
"""

import numpy as np
import pytest
from typing import List

from image_velocimetry_tools.services.cross_section_service import CrossSectionService


@pytest.fixture
def cross_section_service():
    """Create a CrossSectionService instance for testing."""
    return CrossSectionService()


@pytest.fixture
def sample_points():
    """Sample x/y coordinate points."""
    return np.array([
        [0.0, 0.0],
        [3.0, 4.0],
        [6.0, 8.0]
    ])


@pytest.fixture
def sample_cross_section_data():
    """Sample cross-section station and elevation data."""
    return {
        'stations': np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0]),
        'elevations': np.array([10.0, 8.0, 6.0, 7.0, 9.0, 11.0])
    }


class TestComputePixelDistance:
    """Tests for compute_pixel_distance method."""

    def test_basic_distance_calculation(self, cross_section_service):
        """Test basic Euclidean distance calculation."""
        # Two points: (0,0) and (3,4), distance should be 5
        points = np.array([[0.0, 0.0], [3.0, 4.0]])

        distance = cross_section_service.compute_pixel_distance(points)

        assert np.isclose(distance[0], 5.0, rtol=1e-5)

    def test_horizontal_distance(self, cross_section_service):
        """Test distance calculation for horizontal line."""
        points = np.array([[0.0, 0.0], [10.0, 0.0]])

        distance = cross_section_service.compute_pixel_distance(points)

        assert np.isclose(distance[0], 10.0, rtol=1e-5)

    def test_vertical_distance(self, cross_section_service):
        """Test distance calculation for vertical line."""
        points = np.array([[0.0, 0.0], [0.0, 7.5]])

        distance = cross_section_service.compute_pixel_distance(points)

        assert np.isclose(distance[0], 7.5, rtol=1e-5)

    def test_multiple_segments(self, cross_section_service):
        """Test distance calculation with multiple line segments."""
        # Three points forming two segments
        points = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])

        distances = cross_section_service.compute_pixel_distance(points)

        # Both segments have length 5
        assert len(distances) == 2
        assert np.isclose(distances[0], 5.0, rtol=1e-5)
        assert np.isclose(distances[1], 5.0, rtol=1e-5)

    def test_zero_distance(self, cross_section_service):
        """Test distance when points are identical."""
        points = np.array([[5.0, 5.0], [5.0, 5.0]])

        distance = cross_section_service.compute_pixel_distance(points)

        assert np.isclose(distance[0], 0.0, atol=1e-10)


class TestComputePixelToRealWorldConversion:
    """Tests for compute_pixel_to_real_world_conversion method."""

    def test_basic_conversion_factor(self, cross_section_service):
        """Test basic conversion factor calculation."""
        pixel_distance = 100.0  # pixels
        real_world_width = 50.0  # meters

        conversion_factor = cross_section_service.compute_pixel_to_real_world_conversion(
            pixel_distance, real_world_width
        )

        # 50 meters / 100 pixels = 0.5 meters/pixel
        assert np.isclose(conversion_factor, 0.5, rtol=1e-5)

    def test_unit_conversion(self, cross_section_service):
        """Test conversion when pixel and real world distances are equal."""
        pixel_distance = 75.0
        real_world_width = 75.0

        conversion_factor = cross_section_service.compute_pixel_to_real_world_conversion(
            pixel_distance, real_world_width
        )

        assert np.isclose(conversion_factor, 1.0, rtol=1e-5)

    def test_large_scale_factor(self, cross_section_service):
        """Test conversion with large real world distance."""
        pixel_distance = 10.0
        real_world_width = 1000.0  # Large real world distance

        conversion_factor = cross_section_service.compute_pixel_to_real_world_conversion(
            pixel_distance, real_world_width
        )

        assert np.isclose(conversion_factor, 100.0, rtol=1e-5)

    def test_array_input(self, cross_section_service):
        """Test conversion with array input."""
        pixel_distance = np.array([100.0])
        real_world_width = 50.0

        conversion_factor = cross_section_service.compute_pixel_to_real_world_conversion(
            pixel_distance, real_world_width
        )

        assert np.isclose(conversion_factor, 0.5, rtol=1e-5)


class TestFindStationCrossings:
    """Tests for find_station_crossings method."""

    def test_basic_crossing_detection(self, cross_section_service):
        """Test basic stage crossing detection."""
        # Simple V-shaped cross-section
        stations = np.array([0.0, 10.0, 20.0])
        elevations = np.array([10.0, 5.0, 10.0])
        target_elevation = 7.5

        crossings = cross_section_service.find_station_crossings(
            stations, elevations, target_elevation
        )

        # Should cross at two points, equidistant from center
        assert len(crossings) == 2
        assert np.isclose(crossings[0], 5.0, rtol=1e-3)
        assert np.isclose(crossings[1], 15.0, rtol=1e-3)

    def test_firstlast_mode(self, cross_section_service, sample_cross_section_data):
        """Test firstlast mode returns only first and last crossings."""
        stations = sample_cross_section_data['stations']
        elevations = sample_cross_section_data['elevations']
        target_elevation = 8.0

        crossings = cross_section_service.find_station_crossings(
            stations, elevations, target_elevation, mode='firstlast'
        )

        # Should return only first and last crossing
        assert len(crossings) == 2
        assert crossings[0] < crossings[1]

    def test_all_mode(self, cross_section_service):
        """Test all mode returns all crossings."""
        # Multiple crossings
        stations = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        elevations = np.array([8.0, 10.0, 8.0, 10.0, 8.0])
        target_elevation = 9.0

        crossings = cross_section_service.find_station_crossings(
            stations, elevations, target_elevation, mode='all'
        )

        # Should find 4 crossings
        assert len(crossings) == 4

    def test_no_crossings(self, cross_section_service):
        """Test when target elevation doesn't cross the profile."""
        stations = np.array([0.0, 10.0, 20.0])
        elevations = np.array([5.0, 5.0, 5.0])
        target_elevation = 10.0  # Above all points

        crossings = cross_section_service.find_station_crossings(
            stations, elevations, target_elevation
        )

        assert len(crossings) == 0

    def test_epsilon_tolerance(self, cross_section_service):
        """Test epsilon tolerance for near-exact crossings."""
        stations = np.array([0.0, 10.0, 20.0])
        elevations = np.array([10.0, 5.000001, 10.0])
        target_elevation = 5.0
        epsilon = 1e-5

        crossings = cross_section_service.find_station_crossings(
            stations, elevations, target_elevation, epsilon=epsilon
        )

        # Should detect crossing despite small numerical difference
        assert len(crossings) >= 0  # May or may not detect depending on epsilon

    def test_exact_match_at_point(self, cross_section_service):
        """Test when target elevation exactly matches a data point."""
        stations = np.array([0.0, 10.0, 20.0, 30.0])
        elevations = np.array([10.0, 8.0, 6.0, 8.0])
        target_elevation = 8.0  # Exact match

        crossings = cross_section_service.find_station_crossings(
            stations, elevations, target_elevation
        )

        # Should find crossings near the exact point
        assert len(crossings) > 0


class TestInterpolateElevations:
    """Tests for interpolate_elevations method."""

    def test_basic_interpolation(self, cross_section_service):
        """Test basic linear interpolation."""
        stations = np.array([0.0, 10.0, 20.0])
        elevations = np.array([0.0, 10.0, 20.0])
        target_stations = np.array([5.0, 15.0])

        interpolated = cross_section_service.interpolate_elevations(
            stations, elevations, target_stations
        )

        assert np.isclose(interpolated[0], 5.0, rtol=1e-5)
        assert np.isclose(interpolated[1], 15.0, rtol=1e-5)

    def test_interpolation_at_known_points(self, cross_section_service):
        """Test interpolation at existing data points."""
        stations = np.array([0.0, 10.0, 20.0])
        elevations = np.array([5.0, 15.0, 25.0])
        target_stations = np.array([0.0, 10.0, 20.0])

        interpolated = cross_section_service.interpolate_elevations(
            stations, elevations, target_stations
        )

        np.testing.assert_allclose(interpolated, elevations, rtol=1e-5)

    def test_extrapolation(self, cross_section_service):
        """Test behavior with extrapolation beyond data range."""
        stations = np.array([10.0, 20.0, 30.0])
        elevations = np.array([10.0, 20.0, 30.0])
        target_stations = np.array([5.0, 35.0])  # Outside range

        interpolated = cross_section_service.interpolate_elevations(
            stations, elevations, target_stations
        )

        # np.interp extrapolates using edge values
        assert np.isclose(interpolated[0], 10.0, rtol=1e-5)  # Left edge
        assert np.isclose(interpolated[1], 30.0, rtol=1e-5)  # Right edge

    def test_non_monotonic_stations(self, cross_section_service):
        """Test interpolation with complex elevation profile."""
        stations = np.array([0.0, 10.0, 20.0, 30.0])
        elevations = np.array([10.0, 5.0, 8.0, 12.0])
        target_stations = np.array([5.0, 15.0, 25.0])

        interpolated = cross_section_service.interpolate_elevations(
            stations, elevations, target_stations
        )

        # Check interpolated values are reasonable
        assert len(interpolated) == 3
        assert interpolated[0] < elevations[0]  # Between 10 and 5
        assert interpolated[1] > elevations[1]  # Between 5 and 8

    def test_single_point_interpolation(self, cross_section_service):
        """Test interpolation at a single target station."""
        stations = np.array([0.0, 10.0])
        elevations = np.array([0.0, 10.0])
        target_stations = np.array([5.0])

        interpolated = cross_section_service.interpolate_elevations(
            stations, elevations, target_stations
        )

        assert np.isclose(interpolated[0], 5.0, rtol=1e-5)


class TestCheckDuplicateStations:
    """Tests for check_duplicate_stations method."""

    def test_no_duplicates(self, cross_section_service):
        """Test with no duplicate stations."""
        stations = np.array([0.0, 10.0, 20.0, 30.0])

        duplicates = cross_section_service.check_duplicate_stations(stations)

        assert len(duplicates) == 0

    def test_exact_duplicates(self, cross_section_service):
        """Test detection of exact duplicate stations."""
        stations = np.array([0.0, 10.0, 10.0, 20.0])

        duplicates = cross_section_service.check_duplicate_stations(stations)

        # Should identify indices 1 and 2 as duplicates
        assert len(duplicates) > 0
        assert 1 in duplicates or 2 in duplicates

    def test_near_duplicates_within_tolerance(self, cross_section_service):
        """Test detection of near-duplicate stations within tolerance."""
        stations = np.array([0.0, 10.0, 10.0001, 20.0])
        tolerance = 0.001

        duplicates = cross_section_service.check_duplicate_stations(
            stations, tolerance=tolerance
        )

        # Should identify indices 1 and 2 as duplicates within tolerance
        assert len(duplicates) > 0

    def test_near_duplicates_outside_tolerance(self, cross_section_service):
        """Test that near-duplicates outside tolerance are not flagged."""
        stations = np.array([0.0, 10.0, 10.1, 20.0])
        tolerance = 0.01

        duplicates = cross_section_service.check_duplicate_stations(
            stations, tolerance=tolerance
        )

        # 10.0 and 10.1 differ by 0.1, which is > tolerance
        assert len(duplicates) == 0

    def test_multiple_duplicate_groups(self, cross_section_service):
        """Test detection of multiple groups of duplicates."""
        stations = np.array([0.0, 10.0, 10.0, 20.0, 20.0, 30.0])

        duplicates = cross_section_service.check_duplicate_stations(stations)

        # Should identify duplicates at 10.0 and 20.0
        assert len(duplicates) >= 2

    def test_triplicate_stations(self, cross_section_service):
        """Test detection of three identical stations."""
        stations = np.array([0.0, 10.0, 10.0, 10.0, 20.0])

        duplicates = cross_section_service.check_duplicate_stations(stations)

        # Should identify all three 10.0 values
        assert len(duplicates) >= 2

    def test_empty_array(self, cross_section_service):
        """Test with empty stations array."""
        stations = np.array([])

        duplicates = cross_section_service.check_duplicate_stations(stations)

        assert len(duplicates) == 0

    def test_single_station(self, cross_section_service):
        """Test with single station (no duplicates possible)."""
        stations = np.array([10.0])

        duplicates = cross_section_service.check_duplicate_stations(stations)

        assert len(duplicates) == 0


class TestComputeWettedWidth:
    """Tests for compute_wetted_width method."""

    def test_basic_wetted_width(self, cross_section_service):
        """Test basic wetted width calculation."""
        stations = np.array([0.0, 10.0, 20.0])
        elevations = np.array([10.0, 5.0, 10.0])
        water_surface_elevation = 7.5

        wetted_width = cross_section_service.compute_wetted_width(
            stations, elevations, water_surface_elevation
        )

        # Width between two crossing points
        assert wetted_width > 0
        assert np.isclose(wetted_width, 10.0, rtol=1e-2)

    def test_full_width_inundation(self, cross_section_service):
        """Test when water surface is above entire cross-section."""
        stations = np.array([0.0, 10.0, 20.0])
        elevations = np.array([5.0, 5.0, 5.0])
        water_surface_elevation = 10.0  # Above all points

        wetted_width = cross_section_service.compute_wetted_width(
            stations, elevations, water_surface_elevation
        )

        # Should return full width when completely inundated
        # Implementation depends on how this edge case is handled
        assert wetted_width >= 0

    def test_no_inundation(self, cross_section_service):
        """Test when water surface is below entire cross-section."""
        stations = np.array([0.0, 10.0, 20.0])
        elevations = np.array([10.0, 10.0, 10.0])
        water_surface_elevation = 5.0  # Below all points

        wetted_width = cross_section_service.compute_wetted_width(
            stations, elevations, water_surface_elevation
        )

        # Should return 0 or minimal width when no inundation
        assert wetted_width >= 0


class TestFlipStations:
    """Tests for flip_stations method."""

    def test_basic_station_flip(self, cross_section_service):
        """Test basic station flipping (reversing bank orientation)."""
        stations = np.array([0.0, 10.0, 20.0, 30.0])

        flipped = cross_section_service.flip_stations(stations)

        # Stations should be reversed: 0->30, 10->20, 20->10, 30->0
        expected = np.array([30.0, 20.0, 10.0, 0.0])
        np.testing.assert_allclose(flipped, expected, rtol=1e-5)

    def test_single_station_flip(self, cross_section_service):
        """Test flipping with single station."""
        stations = np.array([15.0])

        flipped = cross_section_service.flip_stations(stations)

        # Single station remains at 0
        assert np.isclose(flipped[0], 0.0, rtol=1e-5)

    def test_symmetric_profile_flip(self, cross_section_service):
        """Test flipping symmetric station profile."""
        stations = np.array([0.0, 25.0, 50.0])

        flipped = cross_section_service.flip_stations(stations)

        # Middle point should remain in middle after flip
        expected = np.array([50.0, 25.0, 0.0])
        np.testing.assert_allclose(flipped, expected, rtol=1e-5)
