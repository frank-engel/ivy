import unittest
from image_velocimetry_tools.discharge_tools import *


class TestComputeDischarge(unittest.TestCase):

    def assertAlmostEqualWithinPercentage(self, first, second, percentage):
        delta = max(first, second) * (percentage / 100.0)
        self.assertAlmostEqual(first, second, delta=delta)

    def test_compute_discharge(self):
        # TODO: This test is failing, and I am not sure why. FLE 12/14/2024
        # Test case (Turnipseed & Sauer, 2010, Fig. 2)
        # Note that in the field notes of Fig 2., the hydrographer truncates to
        # 2 decimal places but compute_discharge does not.
        dist_from_start = [
            1,
            4,
            7,
            10,
            13,
            16,
            19,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            47,
            50,
            53,
            56,
            59,
            62,
            65,
            68,
            71,
        ]
        vertical_depths = [
            0.0,
            0.95,
            1.4,
            2.0,
            2.1,
            2.3,
            2.25,
            2.21,
            2.52,
            2.81,
            3.01,
            2.95,
            3.11,
            3.21,
            3.03,
            3.11,
            2.92,
            2.54,
            2.20,
            2.05,
            2.22,
            2.14,
            2.23,
            2.04,
            1.43,
            1.07,
            0.62,
            0.0,
        ]
        avg_velocities = [
            0.0,
            0.29,
            0.38,
            0.32,
            0.4,
            0.42,
            0.43,
            0.48,
            0.57,
            0.62,
            0.64,
            0.72,
            0.72,
            0.74,
            0.72,
            0.65,
            0.58,
            0.52,
            0.51,
            0.47,
            0.46,
            0.43,
            0.42,
            0.4,
            0.36,
            0.4,
            0.28,
            0.0,
        ]
        expected_area = 143.6  # From fig. 2 in TM 3-A8
        expected_discharge = 73.39
        result, _ = compute_discharge_midsection(
            dist_from_start, avg_velocities, vertical_depths
        )
        self.assertAlmostEqualWithinPercentage(
            result, expected_discharge, percentage=1
        )

        # Test case 2, vertical wall scenario. edge depths are 1.0 and 5.0
        dist_from_start = [0, 3, 7, 9, 10]
        avg_velocities = [2.0, 3.0, 4.0, 5.0, 6.0]
        vertical_depths = [1.0, 2.0, 3.0, 4.0, 5.0]
        expected_discharge = 105.0
        expected_area = 26.0
        discharge, area = compute_discharge_midsection(
            dist_from_start, avg_velocities, vertical_depths
        )
        self.assertAlmostEqual(discharge, expected_discharge, places=2)
        self.assertAlmostEqual(area, expected_area, places=2)

        # Test case 3: Unequal lengths of velocities and depths
        dist_from_start = [0, 3, 7, 9, 10]
        avg_velocities = [1.0, 2.0, 3.0]
        vertical_depths = [1.0, 2.0, 3.0, 4.0]
        with self.assertRaises(ValueError):
            compute_discharge_midsection(
                dist_from_start, avg_velocities, vertical_depths
            )

        # Test 4: with only three points in the section.
        # The minimum for Mid-section is 3 points but the way this computes
        # it uses the end points as the edges; this only works as
        # mid-section method if the end point ar not at the edge of water.
        # If the edges were masked points would be lost.CJM
        cumulative_distances = [0, 1, 2]
        average_velocities = [0.0, 0.0, 0.0]
        vertical_depths = [0.0, 0.0, 0.0]
        discharge, area = compute_discharge_midsection(
            cumulative_distances, average_velocities, vertical_depths
        )

        # Expected results are zero since there is no depth or velocity
        self.assertEqual(discharge, 0)
        self.assertEqual(area, 0)

        # Test 5: handling of NaN values in velocity input.
        # This calculates the segment with the v=NAN as zero I think there
        # should be some interpolation done between the segments; This may
        # be done somewhere else
        cumulative_distances = [0, 1, 2, 3]
        average_velocities = [0.0, np.nan, 2.0, 0.0]
        vertical_depths = [0.0, 1.2, 1.5, 0.0]
        discharge, area = compute_discharge_midsection(
            cumulative_distances, average_velocities, vertical_depths
        )

        # NaN velocities are treated as zero -- calculated manually -- CJM
        expected_discharge = 3.0
        expected_area = 2.70
        self.assertAlmostEqual(discharge, expected_discharge, places=2)
        self.assertAlmostEqual(area, expected_area, places=2)

    def test_compute_discharge_with_details(self):
        """Test compute_discharge_midsection with return_details=True"""

        cumulative_distances = [0, 3, 7, 9, 10]
        average_velocities = [2.0, 3.0, 4.0, 5.0, 6.0]
        vertical_depths = [1.0, 2.0, 3.0, 4.0, 5.0]

        expected_discharge = 105.0
        expected_area = 26.0
        expected_widths = [1.5, 3.5, 3.0, 1.5, 0.5]
        expected_areas = [1.5, 7.0, 9.0, 6.0, 2.5]
        expected_unit_discharges = [3.0, 21.0, 36.0, 30.0, 15.0]

        discharge, area, widths, areas, unit_discharges = (
            compute_discharge_midsection(
                cumulative_distances,
                average_velocities,
                vertical_depths,
                return_details=True,
            )
        )

        self.assertAlmostEqual(discharge, expected_discharge, places=2)
        self.assertAlmostEqual(area, expected_area, places=2)
        np.testing.assert_almost_equal(widths, expected_widths, decimal=2)
        np.testing.assert_almost_equal(areas, expected_areas, decimal=2)
        np.testing.assert_almost_equal(
            unit_discharges, expected_unit_discharges, decimal=2
        )

    def test_convert_surface_velocity_rantz(self):
        surface_velocity = 2.0
        mean_velocity = convert_surface_velocity_rantz(surface_velocity)

        # Verify the converted velocity
        self.assertAlmostEqual(mean_velocity, 1.70, places=2)

        # test case 2
        surface_velocity = np.array([2.0, 3.0, 4.0])
        alpha_val = 0.80
        mean_velocity = convert_surface_velocity_rantz(
            surface_velocity, alpha_val
        )

        # Verify the converted velocities
        expected = np.array(surface_velocity * alpha_val)
        # print(surface_velocity * alpha_val)
        np.testing.assert_almost_equal(mean_velocity, expected, decimal=2)

        # test case 3
        surface_velocity = "1"
        with self.assertRaises(ValueError) as context:
            convert_surface_velocity_rantz(surface_velocity)

        self.assertIn(
            "Input must be a float or a numpy array", str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
