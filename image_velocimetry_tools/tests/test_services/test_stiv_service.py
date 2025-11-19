"""
Unit tests for the STIVService class.

This module contains comprehensive tests for the STIV service business logic,
including velocity/angle conversions, manual velocity processing, data loading,
and STIV optimization calculations.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch, mock_open
import os
import tempfile

from image_velocimetry_tools.services.stiv_service import STIVService


@pytest.fixture
def stiv_service():
    """Create a STIVService instance for testing."""
    return STIVService()


@pytest.fixture
def sample_stiv_csv_content():
    """Sample CSV content for STIV results."""
    return """X,Y,U,V,Magnitude,Scalar_Projection,Direction,Tagline_Direction,Normal_Direction
10.0,20.0,1.5,0.5,1.58,1.4,18.43,90.0,0.0
15.0,25.0,2.0,1.0,2.24,1.8,26.57,90.0,0.0
20.0,30.0,1.2,0.8,1.44,1.0,33.69,90.0,0.0"""


@pytest.fixture
def sample_stiv_data():
    """Sample STIV data dictionary."""
    return {
        'X': np.array([10.0, 15.0, 20.0]),
        'Y': np.array([20.0, 25.0, 30.0]),
        'U': np.array([1.5, 2.0, 1.2]),
        'V': np.array([0.5, 1.0, 0.8]),
        'Magnitude': np.array([1.58, 2.24, 1.44]),
        'Scalar_Projection': np.array([1.4, 1.8, 1.0]),
        'Direction': np.array([18.43, 26.57, 33.69]),
        'Tagline_Direction': np.array([90.0, 90.0, 90.0]),
        'Normal_Direction': np.array([0.0, 0.0, 0.0])
    }


@pytest.fixture
def survey_units_english():
    """English survey units conversion factors."""
    return {
        'V': 3.28084,  # m/s to ft/s
        'label_V': 'ft/s'
    }


@pytest.fixture
def survey_units_metric():
    """Metric survey units conversion factors."""
    return {
        'V': 1.0,  # m/s to m/s
        'label_V': 'm/s'
    }


class TestComputeSTIVelocity:
    """Tests for compute_sti_velocity method."""

    def test_basic_velocity_calculation(self, stiv_service):
        """Test basic STI velocity calculation using Fujita et al. (2007) equation 16."""
        # Given theta = 45 degrees, gsd = 0.15 m, dt = 0.1 s
        # velocity = tan(45°) * 0.15 / 0.1 = 1.0 * 1.5 = 1.5 m/s
        theta = 45.0
        gsd = 0.15
        dt = 0.1

        velocity = stiv_service.compute_sti_velocity(theta, gsd, dt)

        assert np.isclose(velocity, 1.5, rtol=1e-5)

    def test_zero_angle(self, stiv_service):
        """Test velocity calculation with zero angle."""
        theta = 0.0
        gsd = 0.15
        dt = 0.1

        velocity = stiv_service.compute_sti_velocity(theta, gsd, dt)

        assert np.isclose(velocity, 0.0, rtol=1e-5)

    def test_negative_angle(self, stiv_service):
        """Test velocity calculation with negative angle."""
        theta = -30.0
        gsd = 0.15
        dt = 0.1

        velocity = stiv_service.compute_sti_velocity(theta, gsd, dt)

        expected = np.tan(np.deg2rad(-30.0)) * 0.15 / 0.1
        assert np.isclose(velocity, expected, rtol=1e-5)

    def test_array_input(self, stiv_service):
        """Test velocity calculation with array input."""
        thetas = np.array([0.0, 45.0, -45.0])
        gsd = 0.15
        dt = 0.1

        velocities = stiv_service.compute_sti_velocity(thetas, gsd, dt)

        expected = np.tan(np.deg2rad(thetas)) * gsd / dt
        np.testing.assert_allclose(velocities, expected, rtol=1e-5)


class TestComputeSTIAngle:
    """Tests for compute_sti_angle method."""

    def test_basic_angle_calculation(self, stiv_service):
        """Test basic STI angle calculation (inverse of velocity calculation)."""
        # Given velocity = 1.5 m/s, gsd = 0.15 m, dt = 0.1 s
        # angle = arctan(-1.5 * 0.1 / 0.15) = arctan(-1.0) = -45 degrees
        velocity = 1.5
        gsd = 0.15
        dt = 0.1

        angle = stiv_service.compute_sti_angle(velocity, gsd, dt)

        expected = np.degrees(np.arctan((-velocity * dt) / gsd))
        assert np.isclose(angle, expected, rtol=1e-5)

    def test_zero_velocity(self, stiv_service):
        """Test angle calculation with zero velocity."""
        velocity = 0.0
        gsd = 0.15
        dt = 0.1

        angle = stiv_service.compute_sti_angle(velocity, gsd, dt)

        assert np.isclose(angle, 0.0, rtol=1e-5)

    def test_array_input(self, stiv_service):
        """Test angle calculation with array input."""
        velocities = np.array([0.0, 1.5, 3.0])
        gsd = 0.15
        dt = 0.1

        angles = stiv_service.compute_sti_angle(velocities, gsd, dt)

        expected = np.degrees(np.arctan((-velocities * dt) / gsd))
        np.testing.assert_allclose(angles, expected, rtol=1e-5)

    def test_velocity_angle_roundtrip(self, stiv_service):
        """Test that velocity -> angle -> velocity is consistent."""
        original_velocity = 2.5
        gsd = 0.15
        dt = 0.1

        angle = stiv_service.compute_sti_angle(original_velocity, gsd, dt)
        recovered_velocity = stiv_service.compute_sti_velocity(angle, gsd, dt)

        assert np.isclose(recovered_velocity, -original_velocity, rtol=1e-5)


class TestComputeVelocityFromManualAngle:
    """Tests for compute_velocity_from_manual_angle method."""

    def test_basic_manual_velocity_calculation(self, stiv_service):
        """Test manual velocity calculation from angle."""
        average_direction = 45.0
        gsd = 0.15
        dt = 0.1
        is_upstream = False

        velocity = stiv_service.compute_velocity_from_manual_angle(
            average_direction, gsd, dt, is_upstream
        )

        # velocity = abs(tan(45°) * 0.15 / 0.1) = 1.5 m/s
        expected = np.abs(np.tan(np.deg2rad(45.0)) * gsd / dt)
        assert np.isclose(velocity, expected, rtol=1e-5)
        assert velocity > 0  # Not upstream, so positive

    def test_upstream_velocity(self, stiv_service):
        """Test manual velocity with upstream flag."""
        average_direction = 45.0
        gsd = 0.15
        dt = 0.1
        is_upstream = True

        velocity = stiv_service.compute_velocity_from_manual_angle(
            average_direction, gsd, dt, is_upstream
        )

        expected = -np.abs(np.tan(np.deg2rad(45.0)) * gsd / dt)
        assert np.isclose(velocity, expected, rtol=1e-5)
        assert velocity < 0  # Upstream, so negative

    def test_canceled_manual_edit(self, stiv_service):
        """Test that canceled manual edit returns NaN."""
        average_direction = -999.0  # Indicates canceled/no manual edit
        gsd = 0.15
        dt = 0.1
        is_upstream = False

        velocity = stiv_service.compute_velocity_from_manual_angle(
            average_direction, gsd, dt, is_upstream
        )

        assert np.isnan(velocity)

    def test_zero_angle(self, stiv_service):
        """Test manual velocity with zero angle."""
        average_direction = 0.0
        gsd = 0.15
        dt = 0.1
        is_upstream = False

        velocity = stiv_service.compute_velocity_from_manual_angle(
            average_direction, gsd, dt, is_upstream
        )

        assert np.isclose(velocity, 0.0, atol=1e-10)


class TestLoadSTIVResultsFromCSV:
    """Tests for load_stiv_results_from_csv method."""

    def test_load_valid_csv(self, stiv_service, sample_stiv_csv_content):
        """Test loading valid STIV results CSV."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(sample_stiv_csv_content)
            temp_path = f.name

        try:
            data = stiv_service.load_stiv_results_from_csv(temp_path)

            assert 'Magnitude' in data
            assert 'Scalar_Projection' in data
            assert 'Direction' in data
            assert len(data['Magnitude']) == 3
            assert np.isclose(data['Magnitude'][0], 1.58, rtol=1e-2)
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self, stiv_service):
        """Test loading nonexistent CSV file."""
        with pytest.raises(FileNotFoundError):
            stiv_service.load_stiv_results_from_csv('/nonexistent/path.csv')

    def test_extract_specific_columns(self, stiv_service, sample_stiv_csv_content):
        """Test that specific columns are correctly extracted."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(sample_stiv_csv_content)
            temp_path = f.name

        try:
            data = stiv_service.load_stiv_results_from_csv(temp_path)

            # Check magnitudes
            np.testing.assert_allclose(
                data['Magnitude'],
                np.array([1.58, 2.24, 1.44]),
                rtol=1e-2
            )

            # Check directions
            np.testing.assert_allclose(
                data['Direction'],
                np.array([18.43, 26.57, 33.69]),
                rtol=1e-2
            )
        finally:
            os.unlink(temp_path)


class TestPrepareTableData:
    """Tests for prepare_table_data method."""

    def test_basic_table_preparation(self, stiv_service, survey_units_metric):
        """Test basic table data preparation with metric units."""
        magnitudes_mps = np.array([1.5, 2.0, 1.2])
        directions = np.array([18.0, 27.0, 34.0])
        thetas = np.array([45.0, 50.0, 40.0])

        table_data = stiv_service.prepare_table_data(
            magnitudes_mps, directions, thetas, survey_units_metric
        )

        assert len(table_data) == 3
        assert table_data[0]['id'] == 1
        assert np.isclose(table_data[0]['original_velocity'], 1.5, rtol=1e-5)
        assert np.isclose(table_data[0]['direction'], 18.0, rtol=1e-5)
        assert np.isclose(table_data[0]['theta'], 45.0, rtol=1e-5)

    def test_english_units_conversion(self, stiv_service, survey_units_english):
        """Test table data preparation with English units."""
        magnitudes_mps = np.array([1.0])
        directions = np.array([0.0])
        thetas = np.array([0.0])

        table_data = stiv_service.prepare_table_data(
            magnitudes_mps, directions, thetas, survey_units_english
        )

        # 1.0 m/s * 3.28084 = 3.28084 ft/s
        assert np.isclose(
            table_data[0]['original_velocity'],
            1.0 * survey_units_english['V'],
            rtol=1e-5
        )

    def test_nan_theta_handling(self, stiv_service, survey_units_metric):
        """Test handling of NaN theta values."""
        magnitudes_mps = np.array([1.5, 2.0])
        directions = np.array([18.0, 27.0])
        thetas = None  # No theta data available

        table_data = stiv_service.prepare_table_data(
            magnitudes_mps, directions, thetas, survey_units_metric
        )

        assert np.isnan(table_data[0]['theta'])
        assert np.isnan(table_data[1]['theta'])

    def test_manual_velocity_initialization(self, stiv_service, survey_units_metric):
        """Test that manual velocity is initialized to original velocity."""
        magnitudes_mps = np.array([2.5])
        directions = np.array([30.0])
        thetas = np.array([45.0])

        table_data = stiv_service.prepare_table_data(
            magnitudes_mps, directions, thetas, survey_units_metric
        )

        assert np.isclose(
            table_data[0]['manual_velocity'],
            table_data[0]['original_velocity'],
            rtol=1e-5
        )


class TestApplyManualCorrections:
    """Tests for apply_manual_corrections method."""

    def test_basic_manual_correction(self, stiv_service, sample_stiv_data):
        """Test applying manual velocity corrections."""
        manual_velocities = np.array([1.5, 2.5, 1.0])
        manual_indices = [0, 2]  # Only indices 0 and 2 were manually edited
        tagline_direction = 90.0

        result = stiv_service.apply_manual_corrections(
            sample_stiv_data, manual_velocities, manual_indices, tagline_direction
        )

        assert 'scalar_projections' in result
        assert len(result['scalar_projections']) == 3
        # Only indices 0 and 2 should be updated
        assert not np.isclose(
            result['scalar_projections'][0],
            sample_stiv_data['Scalar_Projection'][0]
        )

    def test_no_manual_corrections(self, stiv_service, sample_stiv_data):
        """Test with no manual corrections applied."""
        manual_velocities = sample_stiv_data['Magnitude'].copy()
        manual_indices = []  # No manual edits
        tagline_direction = 90.0

        result = stiv_service.apply_manual_corrections(
            sample_stiv_data, manual_velocities, manual_indices, tagline_direction
        )

        # Scalar projections should remain unchanged
        np.testing.assert_allclose(
            result['scalar_projections'],
            sample_stiv_data['Scalar_Projection'],
            rtol=1e-5
        )

    def test_all_manual_corrections(self, stiv_service, sample_stiv_data):
        """Test applying manual corrections to all rows."""
        manual_velocities = np.array([2.0, 3.0, 1.5])
        manual_indices = [0, 1, 2]  # All manually edited
        tagline_direction = 90.0

        result = stiv_service.apply_manual_corrections(
            sample_stiv_data, manual_velocities, manual_indices, tagline_direction
        )

        assert len(result['scalar_projections']) == 3
        # All should be recalculated
        for i in range(3):
            assert result['scalar_projections'][i] is not None


class TestComputeOptimumSampleTime:
    """Tests for compute_optimum_sample_time method."""

    def test_basic_sample_time_calculation(self, stiv_service):
        """Test basic optimum sample time calculation."""
        gsd = 0.15  # meters
        velocity = 2.0  # m/s

        sample_time_ms = stiv_service.compute_optimum_sample_time(gsd, velocity)

        # This should call the optimum_stiv_sample_time function
        assert sample_time_ms > 0
        assert isinstance(sample_time_ms, (int, float))

    def test_high_velocity(self, stiv_service):
        """Test sample time with high velocity."""
        gsd = 0.15
        velocity = 6.0  # High velocity

        sample_time_ms = stiv_service.compute_optimum_sample_time(gsd, velocity)

        # Higher velocity should result in shorter sample time
        assert sample_time_ms > 0

    def test_low_velocity(self, stiv_service):
        """Test sample time with low velocity."""
        gsd = 0.15
        velocity = 0.5  # Low velocity

        sample_time_ms = stiv_service.compute_optimum_sample_time(gsd, velocity)

        # Lower velocity should result in longer sample time
        assert sample_time_ms > 0


class TestComputeFrameStep:
    """Tests for compute_frame_step method."""

    def test_basic_frame_step_calculation(self, stiv_service):
        """Test basic frame step calculation."""
        sample_time_ms = 100.0  # 100 ms
        video_frame_rate = 30.0  # 30 fps

        frame_step = stiv_service.compute_frame_step(sample_time_ms, video_frame_rate)

        # Video frame time = 1000/30 = 33.33 ms
        # Frame step = round(100 / 33.33) = round(3) = 3
        assert frame_step == 3

    def test_minimum_frame_step(self, stiv_service):
        """Test that frame step is at least 1."""
        sample_time_ms = 1.0  # Very short sample time
        video_frame_rate = 30.0

        frame_step = stiv_service.compute_frame_step(sample_time_ms, video_frame_rate)

        # Even with very short sample time, frame step should be at least 1
        assert frame_step >= 1

    def test_high_frame_rate(self, stiv_service):
        """Test frame step calculation with high frame rate."""
        sample_time_ms = 100.0
        video_frame_rate = 60.0  # High frame rate

        frame_step = stiv_service.compute_frame_step(sample_time_ms, video_frame_rate)

        # Video frame time = 1000/60 = 16.67 ms
        # Frame step = round(100 / 16.67) = round(6) = 6
        assert frame_step == 6

    def test_frame_step_is_integer(self, stiv_service):
        """Test that frame step is always an integer."""
        sample_time_ms = 87.5
        video_frame_rate = 29.97

        frame_step = stiv_service.compute_frame_step(sample_time_ms, video_frame_rate)

        assert isinstance(frame_step, int)


class TestComputeSampleTimeSeconds:
    """Tests for compute_sample_time_seconds method."""

    def test_basic_sample_time_seconds(self, stiv_service):
        """Test conversion from frame step to sample time in seconds."""
        frame_step = 3
        video_frame_rate = 30.0

        sample_time_s = stiv_service.compute_sample_time_seconds(
            frame_step, video_frame_rate
        )

        # Sample time = 3 * (1000/30) / 1000 = 3/30 = 0.1 seconds
        expected = frame_step * (1000 / video_frame_rate) / 1000
        assert np.isclose(sample_time_s, expected, rtol=1e-5)

    def test_single_frame_step(self, stiv_service):
        """Test sample time with frame step of 1."""
        frame_step = 1
        video_frame_rate = 30.0

        sample_time_s = stiv_service.compute_sample_time_seconds(
            frame_step, video_frame_rate
        )

        # Sample time = 1/30 = 0.0333... seconds
        expected = 1 / video_frame_rate
        assert np.isclose(sample_time_s, expected, rtol=1e-5)
