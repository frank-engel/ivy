import unittest
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from image_velocimetry_tools.stiv import *
from image_velocimetry_tools.image_processing_tools import (
    create_grayscale_image_stack,
)
from image_velocimetry_tools.common_functions import load_mat_file
import pytest


class TestSTIV(unittest.TestCase):
    """These functions test the STIV processing tools.

    NOTE: these are long-running tests (on the order of minutes).
    """

    def setUp(self):
        # Create a temporary directory to store test images
        self.temp_dir = "test_images"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.map_file_path = f"{self.temp_dir}{os.sep}image_stack.dat"

        # Test input data for Welton Main Drain. SOURCE:
        # Engel, F. L., Cadogan, A., Duan, J. D., 2022. Small Unoccupied
        # Aircraft System Imagery and Associated Data used for Discharge
        # Measurement at Eight Locations Across the United States in 2019 and
        # 2020: U.S. Geological Survey data release.
        # https://doi.org/10.5066/P9H2MM1M.
        self.image_paths = glob.glob(f"./img_seq_welton_main_drain/*.jpg")
        self.image_stack = create_grayscale_image_stack(self.image_paths)

        # Load the Matlab results
        self.matlab_results = load_mat_file(
            f"./img_seq_welton_main_drain/welton_main_drain_matlab_stiv_exuasitve_resutls.mat"
        )
        self.matlab_results["phi0geo"] = np.array([[180]])

        # Extract x, y, and magMax data
        x_data = self.matlab_results["xGrid"]
        y_data = self.matlab_results["yGrid"]
        magMax_data = self.matlab_results["magMax"]
        # Direction in geo angle
        phiMax_data = arithmetic_to_geographic(
            -1 * self.matlab_results["phiMax"]
        )

        # Ensure they have the expected shape (575, 1)
        assert x_data.shape == (575, 1)
        assert y_data.shape == (575, 1)
        assert magMax_data.shape == (575, 1)
        assert phiMax_data.shape == (575, 1)

        # Set random seed for reproducibility
        np.random.seed(42)

        # Randomly sample ~10% (around 57 points)
        sample_size = int(0.1 * x_data.shape[0])  # 10% of 575
        self.sample_indices = np.random.choice(
            x_data.shape[0], sample_size, replace=False
        )

        # Store sampled values for easy access
        self.sample_x = x_data[self.sample_indices]
        self.sample_y = y_data[self.sample_indices]
        self.sample_magMax = magMax_data[self.sample_indices]
        self.sample_phiMax = phiMax_data[self.sample_indices]

    def assertArrayAlmostEqual(self, expected, actual, tol=0.05):
        """
        Assert that two numpy arrays are nearly equal within a specified
        tolerance.

        Parameters
        ----------
        expected : numpy.ndarray
            The expected numpy array.
        actual : numpy.ndarray
            The actual numpy array to be compared with the expected.
        tol : float, optional
            The tolerance level for element-wise comparison (default is 0.05
            for 5% tolerance).

        Raises
        ------
        AssertionError
            If the input arrays have different types or sizes.
        """
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise AssertionError(
                "Input arrays must have the same shape and data type."
            )

        diff = np.abs(expected - actual)
        max_diff = np.max(diff)
        max_value = np.max(np.abs(expected))

        self.assertLessEqual(max_diff, max_value * tol)

    def test_extract_pixels_along_line(self):
        # Test input data
        image_paths = glob.glob(f"./img_seq/*.jpg")
        # image_paths = glob.glob(f"C:/Training/Image_Velocimetry/Exercises/1_worked/boneyard_creek_3D/20160824100501/20160824100501s1.660s_e4.654s_none_normluma_none/t*.jpg")

        # Corresponds to a line from upstream to downstream through the middle of the channel in the resized boneyard
        # creek images in /img_seq (transformed files)
        point1 = np.array([112.7, 59.5])
        point2 = np.array([176.6, 242.4])

        # point1 = np.array([305, 201])
        # point2 = np.array([481, 945])

        def plot_result(result):
            plt.imshow(result, aspect="auto", cmap="gray")
            plt.xlabel("Length")
            plt.ylabel("Time")
            plt.gca().xaxis.tick_top()
            plt.gca().xaxis.set_label_position("top")
            plt.show()

        # Call the function being tested
        result = extract_pixels_along_line(image_paths, point1, point2)

        # Apply the standardization filter
        filtered_image = apply_standardization_filter(result)

        # plot_result(result)
        # plot_result(filtered_image)

        # Define expected result at a length of 100 pixels down the line
        expected_result = np.array(
            [
                [209, 209, 221],
                [214, 218, 227],
                [204, 207, 216],
                [205, 208, 217],
                [181, 184, 193],
                [208, 211, 220],
                [184, 187, 196],
                [190, 193, 202],
                [243, 246, 253],
                [225, 228, 237],
                [178, 180, 192],
                [227, 230, 239],
                [180, 183, 192],
                [216, 216, 226],
                [202, 204, 216],
                [226, 226, 236],
                [174, 174, 186],
                [206, 208, 220],
                [218, 220, 232],
                [159, 161, 173],
                [171, 173, 186],
                [162, 164, 176],
                [195, 197, 209],
                [173, 175, 187],
                [178, 180, 192],
            ]
        )

        # Perform assertions
        self.assertTrue(
            np.array_equal(result[:, 100, :], expected_result),
            "Test failed: Incorrect pixel extraction.",
        )

    @pytest.mark.skip(reason="This test is temporarily disabled")
    def test_two_dimensional_stiv_exhaustive(self):
        magnitudes, directions, stis, thetas = two_dimensional_stiv_exhaustive(
            x_origin=self.sample_x.astype(float),
            y_origin=self.sample_y.astype(float),
            image_stack=self.image_stack,
            num_pixels=self.matlab_results["nPix"].item(),
            phi_origin=self.matlab_results["phi0geo"].item(),
            d_phi=self.matlab_results["dPhi"].item(),
            phi_range=self.matlab_results["phiRange"].item(),
            pixel_gsd=self.matlab_results["pixSize"].item(),
            d_t=self.matlab_results["dt"].item(),
            sigma=0.0,
        )

        # Optionally, make a plot
        if False:
            dir_all = -1 * self.matlab_results["phiMax"]
            dir_all_rad = np.radians(dir_all)
            directions_rad = np.radians(geographic_to_arithmetic(directions))
            image = plt.imread(self.image_paths[0])
            scale_factor = max(
                np.max(self.matlab_results["magMax"]), np.max(magnitudes)
            )  # Normalize based on max magnitude
            plt.imshow(image)
            plt.quiver(
                self.matlab_results["xGrid"].astype(float),
                self.matlab_results["yGrid"].astype(float),
                self.matlab_results["magMax"]
                * np.cos(dir_all_rad)
                / scale_factor,
                -1
                * (self.matlab_results["magMax"] * -np.sin(dir_all_rad))
                / scale_factor,
                # negated b/c imshow origin is upper left, not lower left
                color="k",
                scale=1e-2,
                scale_units="xy",
                width=2e-3,  # Ensure consistent arrow width
                headwidth=4,
                headlength=6,
            )
            plt.quiver(
                self.sample_x.astype(float),
                self.sample_y.astype(float),
                magnitudes * np.cos(directions_rad) / scale_factor,
                -1 * (magnitudes * -np.sin(directions_rad)) / scale_factor,
                # negated b/c imshow origin is upper left, not lower left
                color="y",
                scale=1e-2,
                scale_units="xy",
                width=2e-3,  # Ensure consistent arrow width
                headwidth=4,
                headlength=6,
            )
            plt.show()

        # Perform assertions
        absolute_mag_error = np.abs(self.sample_magMax - magnitudes)
        percent_error = (absolute_mag_error / self.sample_magMax) * 100
        percent_error = np.nan_to_num(percent_error)
        mean_mag_percent_error = np.mean(percent_error)
        self.assertTrue(mean_mag_percent_error < 8)

        absolute_dir_error = np.abs(self.sample_phiMax - directions)
        percent_error = (absolute_dir_error / self.sample_phiMax) * 100
        percent_error = np.nan_to_num(percent_error)
        mean_dir_percent_error = np.nanmean(percent_error)
        self.assertTrue(mean_dir_percent_error < 5)

    @pytest.mark.skip(reason="This test is temporarily disabled")
    def test_two_dimensional_stiv_optimized(self):
        x0 = self.sample_x.astype(float).flatten()
        y0 = self.sample_y.astype(float).flatten()
        magnitudes, directions = two_dimensional_stiv_optimized(
            x_origin=x0,
            y_origin=y0,
            image_stack=self.image_stack,
            num_pixels=self.matlab_results["nPix"].item(),
            phi_origin=self.matlab_results["phi0geo"].repeat(x0.shape),
            # should be same shape as x0
            pixel_gsd=self.matlab_results["pixSize"].item(),
            d_t=self.matlab_results["dt"].item(),
            tolerance=0.5,
            max_vel_threshold=2.0,
        )

        # Optionally, make a plot
        if False:
            dir_all = -1 * self.matlab_results["phiMax"]
            dir_all_rad = np.radians(dir_all)
            directions_rad = np.radians(geographic_to_arithmetic(directions))
            image = plt.imread(self.image_paths[0])
            scale_factor = max(
                np.max(self.matlab_results["magMax"]), np.max(magnitudes)
            )  # Normalize based on max magnitude
            plt.imshow(image)
            plt.quiver(
                self.matlab_results["xGrid"].astype(float),
                self.matlab_results["yGrid"].astype(float),
                self.matlab_results["magMax"]
                * np.cos(dir_all_rad)
                / scale_factor,
                -1
                * (self.matlab_results["magMax"] * -np.sin(dir_all_rad))
                / scale_factor,
                # negated b/c imshow origin is upper left, not lower left
                color="k",
                scale=1e-2,
                scale_units="xy",
                width=2e-3,  # Ensure consistent arrow width
                headwidth=4,
                headlength=6,
            )
            plt.quiver(
                self.sample_x.astype(float),
                self.sample_y.astype(float),
                magnitudes * np.cos(directions_rad) / scale_factor,
                -1 * (magnitudes * -np.sin(directions_rad)) / scale_factor,
                # negated b/c imshow origin is upper left, not lower left
                color="y",
                scale=1e-2,
                scale_units="xy",
                width=2e-3,  # Ensure consistent arrow width
                headwidth=4,
                headlength=6,
            )
            plt.show()

        # Perform assertions
        absolute_mag_error = np.abs(self.sample_magMax - magnitudes)
        percent_error = (absolute_mag_error / self.sample_magMax) * 100
        percent_error = np.nan_to_num(percent_error)
        mean_mag_percent_error = np.mean(percent_error)
        self.assertTrue(mean_mag_percent_error < 25)

        absolute_dir_error = np.abs(self.sample_phiMax - directions)
        percent_error = (absolute_dir_error / self.sample_phiMax) * 100
        percent_error = np.nan_to_num(percent_error)
        mean_dir_percent_error = np.nanmean(percent_error)
        self.assertTrue(mean_dir_percent_error < 5)

    @pytest.mark.skip(reason="This test is temporarily disabled")
    def test_two_dimensional_stiv_exhaustive_with_memory_map(self):
        magnitudes, directions, stis, thetas = two_dimensional_stiv_exhaustive(
            x_origin=self.sample_x.astype(float),
            y_origin=self.sample_y.astype(float),
            image_stack=self.image_stack,
            num_pixels=self.matlab_results["nPix"].item(),
            phi_origin=self.matlab_results["phi0geo"].item(),
            d_phi=self.matlab_results["dPhi"].item(),
            phi_range=self.matlab_results["phiRange"].item(),
            pixel_gsd=self.matlab_results["pixSize"].item(),
            d_t=self.matlab_results["dt"].item(),
            sigma=0.0,
            # map_file_path=self.map_file_path,
        )

        # Perform assertions
        absolute_mag_error = np.abs(self.sample_magMax - magnitudes)
        percent_error = (absolute_mag_error / self.sample_magMax) * 100
        percent_error = np.nan_to_num(percent_error)
        mean_mag_percent_error = np.mean(percent_error)
        self.assertTrue(mean_mag_percent_error < 8)

        absolute_dir_error = np.abs(self.sample_phiMax - directions)
        percent_error = (absolute_dir_error / self.sample_phiMax) * 100
        percent_error = np.nan_to_num(percent_error)
        mean_dir_percent_error = np.nanmean(percent_error)
        self.assertTrue(mean_dir_percent_error < 5)

    @pytest.mark.skip(reason="This test is temporarily disabled")
    def test_two_dimensional_stiv_optimized_with_memory_map(self):
        x0 = self.sample_x.astype(float).flatten()
        y0 = self.sample_y.astype(float).flatten()
        magnitudes, directions = two_dimensional_stiv_optimized(
            x_origin=x0,
            y_origin=y0,
            image_stack=self.image_stack,
            num_pixels=self.matlab_results["nPix"].item(),
            phi_origin=self.matlab_results["phi0geo"].repeat(x0.shape),
            # should be same shape as x0
            pixel_gsd=self.matlab_results["pixSize"].item(),
            d_t=self.matlab_results["dt"].item(),
            tolerance=0.5,
            max_vel_threshold=2.0,
            map_file_path=self.map_file_path,
        )

        # Perform assertions
        absolute_mag_error = np.abs(self.sample_magMax - magnitudes)
        percent_error = (absolute_mag_error / self.sample_magMax) * 100
        percent_error = np.nan_to_num(percent_error)
        mean_mag_percent_error = np.mean(percent_error)
        self.assertTrue(mean_mag_percent_error < 25)

        absolute_dir_error = np.abs(self.sample_phiMax - directions)
        percent_error = (absolute_dir_error / self.sample_phiMax) * 100
        percent_error = np.nan_to_num(percent_error)
        mean_dir_percent_error = np.nanmean(percent_error)
        self.assertTrue(mean_dir_percent_error < 5)


if __name__ == "__main__":
    pass
