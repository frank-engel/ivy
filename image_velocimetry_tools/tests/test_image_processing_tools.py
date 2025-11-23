import unittest
import os
import tempfile
import shutil
import numpy as np
import cv2
import glob

# Import the functions you want to test
from image_velocimetry_tools.image_processing_tools import *


class TestCreateBinaryMask(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to store test images
        self.temp_dir = "test_images"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.map_file_path = f"{self.temp_dir}{os.sep}image_stack.dat"

        # Load some images that should take up sufficient memory
        self.image_paths = glob.glob("./img_seq_welton_main_drain/*.jpg")

    def test_create_mask_with_single_polygon(self):
        image_width = 400
        image_height = 400
        polygon = np.array([(100, 100), (200, 100), (200, 200), (100, 200)])
        polygons = [polygon]
        binary_mask = create_binary_mask(polygons, image_width, image_height)
        # Check if pixels inside the polygon are set to 1
        for point in polygon:
            x, y = point
            self.assertEqual(binary_mask[y, x], 0)

        # Check if pixels outside the polygon are set to 0
        for x in range(image_width):
            for y in range(image_height):
                in_any_polygon = any(
                    cv2.pointPolygonTest(polygon, (x, y), False) >= 0
                    for polygon in polygons
                )
                if not in_any_polygon:
                    self.assertEqual(binary_mask[y, x], 1)

        self.assertEqual(binary_mask.shape, (image_height, image_width))

    def test_create_mask_with_multiple_polygons(self):
        image_width = 500
        image_height = 500
        polygon1 = np.array([(100, 100), (200, 100), (200, 200), (100, 200)])
        polygon2 = np.array([(300, 300), (400, 300), (400, 400), (300, 400)])
        polygons = [polygon1, polygon2]
        binary_mask = create_binary_mask(polygons, image_width, image_height)

        # Check if pixels inside the polygons are set to 1
        for polygon in polygons:
            for point in polygon:
                x, y = point
                self.assertEqual(binary_mask[y, x], 0)

        # Check if pixels outside the polygons are set to 0
        for x in range(image_width):
            for y in range(image_height):
                in_any_polygon = any(
                    cv2.pointPolygonTest(polygon, (x, y), False) >= 0
                    for polygon in polygons
                )
                if not in_any_polygon:
                    self.assertEqual(binary_mask[y, x], 1)

        self.assertEqual(binary_mask.shape, (image_height, image_width))

    def test_create_mask_with_multiple_polygons_real_image(self):
        first_image = cv2.imread(self.image_paths[0], cv2.IMREAD_GRAYSCALE)
        height, width = first_image.shape

        polygons = [
            np.array(
                [
                    [558.0, 1.0],
                    [565.0, 1076.0],
                    [-1.0, 1080.0],
                    [0.0, -1.0],
                    [0.0, -1.0],
                ]
            ),
            np.array(
                [
                    [1465.0, 0.0],
                    [1472.0, 514.0],
                    [1481.0, 1080.0],
                    [1927.0, 1082.0],
                    [1921.0, 1.0],
                    [1921.0, 1.0],
                ]
            ),
        ]
        binary_mask = create_binary_mask(polygons, width, height)

        # Check if pixels inside the polygons are set to 1
        for polygon in polygons:
            for point in polygon:
                x, y = point
                x, y = int(x), int(y)

                # It is possible and valid that the user digitized masks
                # outside the actual image. Don't test those.
                if 0 <= x < width and 0 <= y < height:
                    self.assertEqual(binary_mask[y, x], 0)

        # Check if pixels outside the polygons are set to 0
        for x in range(width):
            for y in range(height):
                # Note, convert polygons to int32
                in_any_polygon = any(
                    cv2.pointPolygonTest(
                        polygon.astype(np.int32), (x, y), False
                    )
                    <= 1
                    for polygon in polygons
                )
                if not in_any_polygon:
                    self.assertEqual(binary_mask[y, x], 1)

        self.assertEqual(binary_mask.shape, (height, width))

    def test_create_mask_with_no_polygons(self):
        image_width = 300
        image_height = 300
        polygons = []
        binary_mask = create_binary_mask(polygons, image_width, image_height)
        self.assertEqual(binary_mask.shape, (image_height, image_width))
        self.assertEqual(np.sum(binary_mask), image_height * image_width)


class TestGenerateGrid(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to store test images
        self.temp_dir = "test_images"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.map_file_path = f"{self.temp_dir}{os.sep}image_stack.dat"

        # Load some images that should take up sufficient memory
        self.image_paths = glob.glob("./img_seq_welton_main_drain/*.jpg")
        self.first_image = cv2.imread(
            self.image_paths[0], cv2.IMREAD_GRAYSCALE
        )

    def test_basic_grid(self):
        # Test with a basic case of a 5x5 image with 1 unmasked region
        image_width = 5
        image_height = 5
        vertical_spacing = 2
        horizontal_spacing = 2
        mask = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ]
        )

        grid = generate_grid(
            image_width,
            image_height,
            vertical_spacing,
            horizontal_spacing,
            mask,
        )

        expected_grid = np.array(
            [[0, 0], [2, 0], [4, 0], [0, 2], [4, 2], [0, 4], [2, 4], [4, 4]]
        )

        np.testing.assert_array_equal(grid, expected_grid)

    def test_empty_mask(self):
        # Test with an empty mask (no unmasked region)
        image_width = 5
        image_height = 5
        vertical_spacing = 2
        horizontal_spacing = 2
        mask = np.zeros((image_height, image_width))

        grid = generate_grid(
            image_width,
            image_height,
            vertical_spacing,
            horizontal_spacing,
            mask,
        )

        expected_grid = np.array([]).reshape(0, 2)

        np.testing.assert_array_equal(grid, expected_grid)

    def test_irregular_spacing(self):
        # Test with irregular vertical and horizontal spacing
        image_width = 6
        image_height = 6
        vertical_spacing = 3
        horizontal_spacing = 2
        mask = np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 1],
                [1, 1, 0, 0, 1, 1],
                [1, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        )

        grid = generate_grid(
            image_width,
            image_height,
            vertical_spacing,
            horizontal_spacing,
            mask,
        )

        expected_grid = np.array(
            [[0, 0], [2, 0], [4, 0], [0, 3], [2, 3], [4, 3]]
        )

        np.testing.assert_array_equal(grid, expected_grid)

    def test_generate_grid_and_close_small_gaps_real_image(self):
        height, width = self.first_image.shape
        polygons = [
            np.array(
                [
                    [558.0, 1.0],
                    [565.0, 1076.0],
                    [-1.0, 1080.0],
                    [0.0, -1.0],
                    [0.0, -1.0],
                ]
            ),
            np.array(
                [
                    [1465.0, 0.0],
                    [1472.0, 514.0],
                    [1481.0, 1080.0],
                    [1927.0, 1082.0],
                    [1921.0, 1.0],
                    [1921.0, 1.0],
                ]
            ),
        ]
        binary_mask = create_binary_mask(polygons, width, height)
        finished_mask = close_small_gaps(binary_mask)

        self.assertEqual(
            978727, np.sum(finished_mask)
        )  # This is the expected mask, assuming all the 1s add up as desired.


class TestCloseSmallGaps(unittest.TestCase):
    def test_close_small_gaps_default_parameters(self):
        # Create a binary mask with small gaps
        binary_mask = np.zeros((100, 100), dtype=np.uint8)
        binary_mask[20:40, 30:50] = 1

        # Apply close_small_gaps with default parameters
        result_mask = close_small_gaps(binary_mask)

        # Check if the result mask is binary (0 or 1)
        unique_values = np.unique(result_mask)
        self.assertListEqual(list(unique_values), [0, 1])

    def test_close_small_gaps_custom_parameters(self):
        # Create a binary mask with small gaps and noise
        binary_mask = np.zeros((100, 100), dtype=np.uint8)
        binary_mask[20:40, 30:50] = 1
        binary_mask[70:75, 60:65] = 1

        # Apply close_small_gaps with custom parameters
        result_mask = close_small_gaps(
            binary_mask, kernel_size=3, area_threshold=0.02, blur_sigma=2.0
        )

        # Check if the result mask is binary (0 or 1)
        unique_values = np.unique(result_mask)
        self.assertListEqual(list(unique_values), [0, 1])

    def test_close_small_gaps_sliver(self):
        # Create a binary mask with small gaps
        binary_mask = np.zeros((100, 100), dtype=np.uint8)
        binary_mask[20:40, 30:50] = 1
        binary_mask[40:70, 30] = 1

        # Apply close_small_gaps with default parameters
        result_mask = close_small_gaps(binary_mask)

        # Check if the result mask is binary (0 or 1)
        unique_values = np.unique(result_mask)
        self.assertListEqual(list(unique_values), [0, 1])


class TestGeneratePointsForMidSection(unittest.TestCase):
    def setUp(self):
        """Set up test parameters and masks."""
        self.image_width = 100
        self.image_height = 100
        self.binary_mask = np.ones(
            (self.image_height, self.image_width), dtype=np.uint8
        )  # Fully unmasked

    def test_point_spacing(self):
        """Test points are evenly spaced."""
        line_start = np.array([0.0, 0.0])
        line_end = np.array([10, 10])

        number_points = 4

        # Call the function
        points = generate_points_along_line(
            self.image_width,
            self.image_height,
            line_start,
            line_end,
            number_points,
            self.binary_mask,
        )

        # Expected points for 4 evenly spaced points
        expected_points = np.array([[0, 0], [3, 3], [7, 7], [10, 10]])

        # Validate the generated points match the expected output
        np.testing.assert_array_equal(points, expected_points)

    def test_points_with_masked_regions(self):
        """Test points are excluded if they fall in masked regions. **The points along the line (number_points) should be
        recomputed based off the new edge of the masked regions,
        this may cause an issue if masked regions are not on the edge? CJM"""
        # Create a mask with only the middle region unmasked
        self.binary_mask = np.zeros(
            (self.image_height, self.image_width), dtype=np.uint8
        )
        cv2.rectangle(
            self.binary_mask, (20, 20), (80, 80), 1, -1
        )  # Unmasked rectangle

        line_start = np.array([10, 10])
        line_end = np.array([90, 90])
        number_points = 5

        # Call the function
        points = generate_points_along_line(
            self.image_width,
            self.image_height,
            line_start,
            line_end,
            number_points,
            self.binary_mask,
        )

        # Validate all points are within the unmasked region
        for point in points:
            self.assertEqual(self.binary_mask[point[1], point[0]], 1)
