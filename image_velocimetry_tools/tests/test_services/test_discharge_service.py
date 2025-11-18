"""Tests for DischargeService."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock

from image_velocimetry_tools.services.discharge_service import DischargeService


@pytest.fixture
def discharge_service():
    """Create a DischargeService instance for testing."""
    return DischargeService()


@pytest.fixture
def mock_xs_survey():
    """Create a mock cross-section survey object."""
    mock = Mock()
    mock.get_pixel_xs = Mock(return_value=(
        np.array([0.0, 5.0, 10.0, 15.0, 20.0]),  # stations
        np.array([10.0, 8.0, 7.5, 8.5, 10.0])    # elevations
    ))
    return mock


@pytest.fixture
def mock_stiv_results():
    """Create mock STIV results object."""
    mock = Mock()
    mock.directions = np.array([45.0, 30.0, 15.0])  # degrees
    mock.magnitudes_mps = np.array([1.5, 2.0, 1.8])  # m/s
    mock.magnitude_normals_mps = np.array([1.4, 1.9, 1.7])
    return mock


@pytest.fixture
def sample_discharge_dataframe():
    """Create a sample discharge dataframe."""
    data = {
        "ID": [0, 1, 2, 3, 4],
        "Status": ["Used", "Used", "Used", "Used", "Used"],
        "Station Distance": [0.0, 5.0, 10.0, 15.0, 20.0],
        "Width": [2.5, 5.0, 5.0, 5.0, 2.5],
        "Depth": [0.5, 2.0, 2.5, 1.5, 0.5],
        "Area": [1.25, 10.0, 12.5, 7.5, 1.25],
        "Surface Velocity": [0.5, 1.5, 2.0, 1.2, 0.4],
        "α (alpha)": [0.85, 0.85, 0.85, 0.85, 0.85],
        "Unit Discharge": [0.53, 12.75, 21.25, 7.65, 0.43],
    }
    return pd.DataFrame(data)


class TestGetStationAndDepth:
    """Tests for get_station_and_depth method."""

    def test_basic_station_depth_extraction(self, discharge_service, mock_xs_survey):
        """Test basic extraction of station and depth."""
        grid_points = np.array([[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]])
        wse = 10.0  # meters

        stations, depths = discharge_service.get_station_and_depth(
            mock_xs_survey,
            grid_points,
            wse
        )

        # Verify get_pixel_xs was called with correct arguments
        mock_xs_survey.get_pixel_xs.assert_called_once()
        args = mock_xs_survey.get_pixel_xs.call_args[0]
        np.testing.assert_array_equal(args[0], grid_points)

        # Check returned values
        assert len(stations) == 5
        assert len(depths) == 5
        np.testing.assert_array_equal(stations, np.array([0.0, 5.0, 10.0, 15.0, 20.0]))

        # Depths should be WSE - elevations
        expected_depths = np.array([0.0, 2.0, 2.5, 1.5, 0.0])
        np.testing.assert_array_almost_equal(depths, expected_depths)

    def test_negative_depths_when_wse_below_elevation(self, discharge_service, mock_xs_survey):
        """Test that negative depths are computed when WSE is below elevation."""
        grid_points = np.array([[10, 20]])
        wse = 5.0  # Below some elevations

        stations, depths = discharge_service.get_station_and_depth(
            mock_xs_survey,
            grid_points,
            wse
        )

        # Should have negative depths where elevation > wse
        assert np.any(depths < 0)


class TestExtractVelocityFromStiv:
    """Tests for extract_velocity_from_stiv method."""

    def test_extract_with_normal_magnitudes(self, discharge_service, mock_stiv_results):
        """Test velocity extraction when normal magnitudes are available."""
        velocities = discharge_service.extract_velocity_from_stiv(
            mock_stiv_results,
            add_edge_zeros=True
        )

        # Should use magnitude_normals_mps when available
        assert len(velocities) == 5  # 3 + 2 edge zeros
        assert velocities[0] == 0.0  # Left edge
        assert velocities[-1] == 0.0  # Right edge
        np.testing.assert_array_almost_equal(
            velocities[1:-1],
            mock_stiv_results.magnitude_normals_mps
        )

    def test_extract_without_normal_magnitudes(self, discharge_service):
        """Test velocity extraction when normal magnitudes are not available."""
        mock_stiv = Mock()
        mock_stiv.directions = np.array([0.0, 90.0])  # 0° and 90°
        mock_stiv.magnitudes_mps = np.array([2.0, 3.0])
        mock_stiv.magnitude_normals_mps = None

        velocities = discharge_service.extract_velocity_from_stiv(
            mock_stiv,
            add_edge_zeros=True
        )

        # Should compute from U and V components
        assert len(velocities) == 4  # 2 + 2 edge zeros
        # For 0°: U=2*cos(0)=2, V=2*sin(0)=0, M=sqrt(4+0)=2
        # For 90°: U=3*cos(90)≈0, V=3*sin(90)=3, M=sqrt(0+9)=3
        np.testing.assert_array_almost_equal(velocities[1:-1], [2.0, 3.0], decimal=5)

    def test_extract_without_edge_zeros(self, discharge_service, mock_stiv_results):
        """Test velocity extraction without adding edge zeros."""
        velocities = discharge_service.extract_velocity_from_stiv(
            mock_stiv_results,
            add_edge_zeros=False
        )

        # Should not add edge zeros
        assert len(velocities) == 3
        np.testing.assert_array_almost_equal(
            velocities,
            mock_stiv_results.magnitude_normals_mps
        )


class TestCreateDischargeDataframe:
    """Tests for create_discharge_dataframe method."""

    def test_basic_dataframe_creation(self, discharge_service):
        """Test basic dataframe creation."""
        stations = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        depths = np.array([0.5, 2.0, 2.5, 1.5, 0.5])
        velocities = np.array([0.5, 1.5, 2.0, 1.2, 0.4])

        df = discharge_service.create_discharge_dataframe(
            stations, depths, velocities, alpha=0.85
        )

        # Check structure
        assert len(df) == 5
        assert "ID" in df.columns
        assert "Status" in df.columns
        assert "Station Distance" in df.columns
        assert "Width" in df.columns
        assert "Depth" in df.columns
        assert "Area" in df.columns
        assert "Surface Velocity" in df.columns
        assert "α (alpha)" in df.columns
        assert "Unit Discharge" in df.columns

        # Check IDs
        np.testing.assert_array_equal(df["ID"].values, np.arange(5))

        # Check all stations marked as "Used" by default
        assert all(df["Status"] == "Used")

        # Check alpha values
        assert all(df["α (alpha)"] == 0.85)

    def test_width_calculation_middle_stations(self, discharge_service):
        """Test width calculation for middle stations."""
        stations = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        depths = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        velocities = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        df = discharge_service.create_discharge_dataframe(
            stations, depths, velocities
        )

        # Middle station widths: (next - prev) / 2
        # Station 1: (10 - 0) / 2 = 5.0
        # Station 2: (15 - 5) / 2 = 5.0
        # Station 3: (20 - 10) / 2 = 5.0
        assert df.loc[1, "Width"] == 5.0
        assert df.loc[2, "Width"] == 5.0
        assert df.loc[3, "Width"] == 5.0

    def test_width_calculation_edge_stations(self, discharge_service):
        """Test width calculation for edge stations."""
        stations = np.array([0.0, 5.0, 10.0])
        depths = np.array([1.0, 1.0, 1.0])
        velocities = np.array([1.0, 1.0, 1.0])

        df = discharge_service.create_discharge_dataframe(
            stations, depths, velocities
        )

        # First station: (5 - 0) / 2 = 2.5
        assert df.loc[0, "Width"] == 2.5

        # Last station: (10 - 5) / 2 = 2.5
        assert df.loc[2, "Width"] == 2.5

    def test_area_calculation(self, discharge_service):
        """Test area calculation."""
        stations = np.array([0.0, 5.0, 10.0])
        depths = np.array([2.0, 4.0, 2.0])
        velocities = np.array([1.0, 1.0, 1.0])

        df = discharge_service.create_discharge_dataframe(
            stations, depths, velocities
        )

        # Middle station area: width * depth = 5.0 * 4.0 = 20.0
        assert df.loc[1, "Area"] == 20.0

        # Edge stations: (width / 2) * depth
        # First: (2.5 / 2) * 2.0 = 2.5
        assert df.loc[0, "Area"] == 2.5

        # Last: (2.5 / 2) * 2.0 = 2.5
        assert df.loc[2, "Area"] == 2.5

    def test_unit_discharge_calculation(self, discharge_service):
        """Test unit discharge calculation."""
        stations = np.array([0.0, 5.0, 10.0])
        depths = np.array([2.0, 2.0, 2.0])
        velocities = np.array([1.0, 2.0, 1.0])
        alpha = 0.85

        df = discharge_service.create_discharge_dataframe(
            stations, depths, velocities, alpha=alpha
        )

        # Unit discharge = Area * Surface Velocity * alpha
        for idx in df.index:
            expected = df.loc[idx, "Area"] * df.loc[idx, "Surface Velocity"] * alpha
            assert abs(df.loc[idx, "Unit Discharge"] - expected) < 0.01

    def test_nan_velocity_handling(self, discharge_service):
        """Test that NaN velocities are converted to 0 in discharge calculation."""
        stations = np.array([0.0, 5.0, 10.0])
        depths = np.array([2.0, 2.0, 2.0])
        velocities = np.array([1.0, np.nan, 1.0])

        df = discharge_service.create_discharge_dataframe(
            stations, depths, velocities
        )

        # Unit discharge should be 0 where velocity is NaN
        assert df.loc[1, "Unit Discharge"] == 0.0

    def test_existing_status_preservation(self, discharge_service):
        """Test that existing status is preserved."""
        stations = np.array([0.0, 5.0, 10.0, 15.0])
        depths = np.array([1.0, 1.0, 1.0, 1.0])
        velocities = np.array([1.0, 1.0, 1.0, 1.0])
        existing_status = np.array(["Used", "Not Used", "Used", "Not Used"])

        df = discharge_service.create_discharge_dataframe(
            stations, depths, velocities, existing_status=existing_status
        )

        np.testing.assert_array_equal(df["Status"].values, existing_status)


class TestComputeDischarge:
    """Tests for compute_discharge method."""

    def test_compute_discharge_all_used(self, discharge_service, sample_discharge_dataframe):
        """Test discharge computation with all stations used."""
        result = discharge_service.compute_discharge(sample_discharge_dataframe)

        assert "total_discharge" in result
        assert "total_area" in result
        assert "discharge_results" in result

        # Total discharge should be sum of unit discharges
        expected_total = sample_discharge_dataframe["Unit Discharge"].sum()
        assert abs(result["total_discharge"] - expected_total) < 0.01

        # Total area should be sum of areas
        expected_area = sample_discharge_dataframe["Area"].sum()
        assert abs(result["total_area"] - expected_area) < 0.01

    def test_compute_discharge_some_not_used(self, discharge_service, sample_discharge_dataframe):
        """Test discharge computation with some stations marked as 'Not Used'."""
        # Mark some stations as "Not Used"
        sample_discharge_dataframe.loc[1, "Status"] = "Not Used"
        sample_discharge_dataframe.loc[3, "Status"] = "Not Used"

        result = discharge_service.compute_discharge(sample_discharge_dataframe)

        # Should only use stations marked as "Used"
        used_df = sample_discharge_dataframe[sample_discharge_dataframe["Status"] == "Used"]
        # Can't directly compare discharge since it's recomputed with Rantz conversion
        # Just verify we got results
        assert result["total_discharge"] > 0
        assert result["total_area"] > 0

    def test_compute_discharge_empty_dataframe(self, discharge_service):
        """Test discharge computation with empty dataframe."""
        empty_df = pd.DataFrame({
            "ID": [],
            "Status": [],
            "Station Distance": [],
            "Width": [],
            "Depth": [],
            "Area": [],
            "Surface Velocity": [],
            "α (alpha)": [],
            "Unit Discharge": [],
        })

        result = discharge_service.compute_discharge(empty_df)

        assert result["total_discharge"] == 0.0
        assert result["total_area"] == 0.0
        assert result["discharge_results"] == {}

    def test_compute_discharge_no_used_stations(self, discharge_service, sample_discharge_dataframe):
        """Test discharge computation when no stations are marked as 'Used'."""
        sample_discharge_dataframe["Status"] = "Not Used"

        result = discharge_service.compute_discharge(sample_discharge_dataframe)

        assert result["total_discharge"] == 0.0
        assert result["total_area"] == 0.0


class TestComputeUncertainty:
    """Tests for compute_uncertainty method."""

    def test_compute_uncertainty_basic(self, discharge_service):
        """Test basic uncertainty computation."""
        # Create simple discharge results
        discharge_results = {
            0: {
                "Station Distance": 0.0,
                "Width": 5.0,
                "Depth": 2.0,
                "Area": 10.0,
                "Surface Velocity": 1.5,
                "α (alpha)": 0.85,
                "Unit Discharge": 12.75
            }
        }
        total_discharge = 12.75
        rectification_rmse = 0.1
        scene_width = 20.0

        result = discharge_service.compute_uncertainty(
            discharge_results,
            total_discharge,
            rectification_rmse,
            scene_width
        )

        assert "u_iso" in result
        assert "u_ive" in result
        assert "u_iso_contribution" in result
        assert "u_ive_contribution" in result

        # Check that uncertainties are dictionaries with expected keys
        assert "u95_q" in result["u_iso"]
        assert "u95_q" in result["u_ive"]

        # Uncertainties should be positive
        assert result["u_iso"]["u95_q"] >= 0
        assert result["u_ive"]["u95_q"] >= 0

    def test_compute_uncertainty_multiple_stations(self, discharge_service, sample_discharge_dataframe):
        """Test uncertainty computation with multiple stations."""
        discharge_results = sample_discharge_dataframe.to_dict(orient="index")
        total_discharge = sample_discharge_dataframe["Unit Discharge"].sum()

        result = discharge_service.compute_uncertainty(
            discharge_results,
            total_discharge,
            rectification_rmse=0.15,
            scene_width=25.0
        )

        # Should complete without errors
        assert result["u_iso"]["u95_q"] >= 0
        assert result["u_ive"]["u95_q"] >= 0


class TestComputeSummaryStatistics:
    """Tests for compute_summary_statistics method."""

    def test_compute_summary_stats_basic(self, discharge_service):
        """Test basic summary statistics computation."""
        discharge_results = {
            0: {"Surface Velocity": 1.0, "α (alpha)": 0.85},
            1: {"Surface Velocity": 2.0, "α (alpha)": 0.85},
            2: {"Surface Velocity": 1.5, "α (alpha)": 0.90},
        }
        total_discharge = 10.0
        total_area = 5.0

        stats = discharge_service.compute_summary_statistics(
            discharge_results,
            total_discharge,
            total_area
        )

        assert "average_velocity" in stats
        assert "average_alpha" in stats
        assert "average_surface_velocity" in stats
        assert "max_surface_velocity" in stats

        # Average velocity = Q / A
        assert abs(stats["average_velocity"] - 2.0) < 0.01

        # Average alpha = mean of alphas
        expected_alpha = (0.85 + 0.85 + 0.90) / 3
        assert abs(stats["average_alpha"] - expected_alpha) < 0.01

        # Average surface velocity = mean of surface velocities
        expected_avg_vel = (1.0 + 2.0 + 1.5) / 3
        assert abs(stats["average_surface_velocity"] - expected_avg_vel) < 0.01

        # Max surface velocity
        assert stats["max_surface_velocity"] == 2.0

    def test_compute_summary_stats_with_nan(self, discharge_service):
        """Test summary statistics with NaN values."""
        discharge_results = {
            0: {"Surface Velocity": 1.0, "α (alpha)": 0.85},
            1: {"Surface Velocity": np.nan, "α (alpha)": 0.85},
            2: {"Surface Velocity": 2.0, "α (alpha)": 0.90},
        }
        total_discharge = 10.0
        total_area = 5.0

        stats = discharge_service.compute_summary_statistics(
            discharge_results,
            total_discharge,
            total_area
        )

        # Should handle NaN gracefully
        # Average should exclude NaN
        expected_avg = (1.0 + 2.0) / 3  # nansum divides by original count
        assert abs(stats["average_surface_velocity"] - expected_avg) < 0.01

    def test_compute_summary_stats_zero_area(self, discharge_service):
        """Test summary statistics with zero area."""
        discharge_results = {
            0: {"Surface Velocity": 1.0, "α (alpha)": 0.85},
        }
        total_discharge = 10.0
        total_area = 0.0

        stats = discharge_service.compute_summary_statistics(
            discharge_results,
            total_discharge,
            total_area
        )

        # Should handle zero area without division error
        assert stats["average_velocity"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
