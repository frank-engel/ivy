import unittest
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem

from image_velocimetry_tools.common_functions import *
from image_velocimetry_tools.image_processing_tools import (
    create_grayscale_image_stack,
    estimate_image_stack_memory_usage,
)


class TestCommonFunctions(unittest.TestCase):
    def setUp(self):
        pass

    def test_quotify_a_string_no_spaces(self):
        string = "C:/Path/No/Spaces/filename.ext"
        self.assertEqual(
            quotify_a_string(string), "C:/Path/No/Spaces/filename.ext"
        )

    def test_quotify_a_string_spaces(self):
        string = "C:/Path/With/Spaces Contained/In the file name.ext"
        self.assertEqual(
            quotify_a_string(string),
            '"C:/Path/With/Spaces Contained/In the file name.ext"',
        )

    def test_quotify_a_string_no_spaces_single_quotes(self):
        string = "'C:/Path/No/Spaces/withSingleQuotes.ext'"
        self.assertEqual(
            quotify_a_string(string),
            '"C:/Path/No/Spaces/withSingleQuotes.ext"',
        )

    def test_quotify_a_string_no_spaces_double_quotes(self):
        string = '"C:/Path/No/Spaces/withDoubleQuotes.ext"'
        self.assertEqual(
            quotify_a_string(string),
            '"C:/Path/No/Spaces/withDoubleQuotes.ext"',
        )

    def test_quotify_a_string_with_spaces(self):
        self.assertEqual(quotify_a_string("hello world"), '"hello world"')

    def test_quotify_a_string_without_spaces(self):
        self.assertEqual(quotify_a_string("hello"), "hello")

    def test_seconds_to_hhmmss_low(self):
        self.assertEqual(seconds_to_hhmmss(10.14), "00:00:10")
        self.assertEqual(seconds_to_hhmmss(7200), "02:00:00")
        self.assertEqual(seconds_to_hhmmss(3661), "01:01:01")
        self.assertEqual(seconds_to_hhmmss(3661.25), "01:01:01")
        self.assertEqual(seconds_to_hhmmss(7200), "02:00:00")
        self.assertEqual(seconds_to_hhmmss(43200.5), "12:00:00")

    def test_seconds_to_hhmmss_high(self):
        self.assertEqual(
            seconds_to_hhmmss(10.61, "high"), "00:00:10.61"
        )  # The {:d} format spec truncates, not rounds
        self.assertEqual(seconds_to_hhmmss(3600, "high"), "01:00:00.00")
        self.assertEqual(seconds_to_hhmmss(3661.4, "high"), "01:01:01.40")
        self.assertEqual(
            seconds_to_hhmmss(3661.25, precision="high"), "01:01:01.25"
        )
        self.assertEqual(
            seconds_to_hhmmss(7200, precision="high"), "02:00:00.00"
        )
        self.assertEqual(
            seconds_to_hhmmss(43200.5, precision="high"), "12:00:00.50"
        )

    def test_invalid_precision_str_to_float(self):
        with self.assertRaises(ValueError):
            seconds_to_hhmmss(3600, "invalid")

    def test_hhmmss_to_seconds(self):
        # Test valid input with low precision
        self.assertEqual(hhmmss_to_seconds("00:00:01"), 1.0)
        self.assertEqual(hhmmss_to_seconds("00:01:00"), 60.0)
        self.assertEqual(hhmmss_to_seconds("01:00:00"), 3600.0)
        self.assertEqual(hhmmss_to_seconds("23:59:59"), 86399.0)

        # Test valid input with high precision
        self.assertEqual(hhmmss_to_seconds("00:00:01.50"), 1.5)
        self.assertEqual(hhmmss_to_seconds("00:01:00.25"), 60.25)
        self.assertEqual(hhmmss_to_seconds("01:00:00.75"), 3600.75)
        self.assertEqual(hhmmss_to_seconds("23:59:59.99"), 86399.99)
        self.assertEqual(hhmmss_to_seconds("00:00:30.03"), 30.03)
        self.assertEqual(hhmmss_to_seconds("00:05:30.03"), 330.03)
        self.assertEqual(hhmmss_to_seconds("00:00:32"), 32)
        self.assertEqual(hhmmss_to_seconds("01:10:32"), 4232)
        self.assertEqual(hhmmss_to_seconds("01:10:32.64"), 4232.64)

        # Test invalid input
        # self.assertEqual(hhmmss_to_seconds("24:00:00"), None)
        # self.assertEqual(hhmmss_to_seconds("00:60:00"), None)
        # self.assertEqual(hhmmss_to_seconds("00:00:60"), None)

    def test_string_to_boolean_case1(self):
        self.assertEqual(string_to_boolean("Yes"), True)
        self.assertEqual(string_to_boolean("y"), True)
        self.assertEqual(string_to_boolean("True"), True)
        self.assertEqual(string_to_boolean("No"), False)
        self.assertEqual(string_to_boolean("n"), False)
        self.assertEqual(string_to_boolean("f"), False)
        self.assertEqual(string_to_boolean("f"), False)
        self.assertEqual(string_to_boolean(None), False)

    def test_hundredth_precision(self):
        self.assertEqual(
            float_seconds_to_time_string(3661.23456, "hundredth"),
            "01:01:01.23",
        )
        self.assertEqual(
            float_seconds_to_time_string(123.456789, "hundredth"),
            "00:02:03.46",
        )

    def test_second_precision(self):
        self.assertEqual(
            float_seconds_to_time_string(3661.23456, "second"), "01:01:01"
        )
        self.assertEqual(
            float_seconds_to_time_string(123.456789, "second"), "00:02:03"
        )

    def test_only_seconds_precision(self):
        self.assertEqual(
            float_seconds_to_time_string(3661.23456, "only_seconds"), "3661.23"
        )
        self.assertEqual(
            float_seconds_to_time_string(123.456789, "only_seconds"), "123.46"
        )

    def test_invalid_precision_float_to_str(self):
        with self.assertRaises(ValueError):
            float_seconds_to_time_string(123, "invalid")

    # def test_valid_timestamp_with_microseconds(self):
    #     ts = "2022-05-14T20:24:55.123456Z"
    #     expected = datetime(2022, 5, 14, 20, 24, 55, 123456,
    #                         tzinfo=timezone.utc)
    #     self.assertEqual(parse_creation_time(ts), expected)
    #
    # def test_valid_timestamp_without_microseconds(self):
    #     ts = "2022-05-14T20:24:55Z"
    #     expected = datetime(2022, 5, 14, 20, 24, 55, tzinfo=timezone.utc)
    #     self.assertEqual(parse_creation_time(ts), expected)
    #
    # def test_invalid_timestamp_format(self):
    #     ts = "2022/05/14 20:24:55"
    #     self.assertIsNone(parse_creation_time(ts))
    #
    # def test_empty_string(self):
    #     ts = ""
    #     self.assertIsNone(parse_creation_time(ts))
    #
    # def test_none_input(self):
    #     ts = None
    #     self.assertIsNone(parse_creation_time(ts))
    #
    # def test_non_string_input(self):
    #     ts = 123456
    #     self.assertIsNone(parse_creation_time(ts))

    def test_get_normal_vectors(self):
        # Test case 1: Two vertices on the x-axis
        v1 = np.array([0, 0])
        v2 = np.array([4, 0])
        expected_output = np.array(
            [
                [-0.0, 1.0],
                [-0.0, 1.0],
                [-0.0, 1.0],
                [-0.0, 1.0],
                [-0.0, 1.0],
                [-0.0, 1.0],
                [-0.0, 1.0],
                [-0.0, 1.0],
                [-0.0, 1.0],
                [-0.0, 1.0],
            ]
        )
        normals, vector_locations, distance = get_normal_vectors(v1, v2)
        # plt.quiver(vector_locations[:, 0], vector_locations[:, 1], normals[:, 0], normals[:, 1])
        # plt.show()
        assert np.allclose(normals, expected_output)

        # Test case 2: Two vertices at arbitrary positions
        v1 = np.array([1, 2])
        v2 = np.array([5, 6])
        expected_output = np.array(
            [
                [-0.70710678, 0.70710678],
                [-0.70710678, 0.70710678],
                [-0.70710678, 0.70710678],
                [-0.70710678, 0.70710678],
                [-0.70710678, 0.70710678],
                [-0.70710678, 0.70710678],
                [-0.70710678, 0.70710678],
                [-0.70710678, 0.70710678],
                [-0.70710678, 0.70710678],
                [-0.70710678, 0.70710678],
            ]
        )
        normals, vector_locations, distance = get_normal_vectors(v1, v2)
        # plt.quiver(vector_locations[:, 0], vector_locations[:, 1], normals[:, 0] * 2, normals[:, 1] * 2, angles="xy", scale_units='xy', scale=1)
        # plt.axis("equal")
        # plt.show()
        assert np.allclose(normals, expected_output)

        # Test case 3: Three vertices on the y-axis with a different number of vectors
        v1 = np.array([0, 0])
        v2 = np.array([0, 4])
        expected_output = np.array(
            [
                [-1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
            ]
        )

        normals, vector_locations, distance = get_normal_vectors(
            v1, v2, num_vectors=15
        )
        # plt.quiver(vector_locations[:, 0], vector_locations[:, 1], normals[:, 0], normals[:, 1])
        # plt.show()
        assert np.allclose(normals, expected_output)

    def test_geographic_to_arithmetic_single_value(self):
        # Test a single geographic angle conversion
        geographic_angle = 45
        expected_arithmetic_angle = 45
        result = geographic_to_arithmetic(geographic_angle)
        self.assertAlmostEqual(result, expected_arithmetic_angle)

    def test_geographic_to_arithmetic_array(self):
        # Test geographic angle conversion with an array of values
        geographic_angles = np.array([30, 90, 180, 270, 360])
        expected_arithmetic_angles = np.array([60, 0, 270, 180, 90])
        result = geographic_to_arithmetic(geographic_angles)
        np.testing.assert_almost_equal(result, expected_arithmetic_angles)

    def test_arithmetic_to_geographic_single_value(self):
        # Test a single arithmetic angle conversion
        arithmetic_angle = 30
        expected_geographic_angle = 60
        result = arithmetic_to_geographic(arithmetic_angle)
        self.assertAlmostEqual(result, expected_geographic_angle)

    def test_arithmetic_to_geographic_array(self):
        # Test arithmetic angle conversion with an array of values
        arithmetic_angles = np.array([60, 0, 90, 180, 270])
        expected_geographic_angles = np.array([30, 90, 0, 270, 180])
        result = arithmetic_to_geographic(arithmetic_angles)
        np.testing.assert_almost_equal(result, expected_geographic_angles)

    def test_geographic_to_arithmetic_edge_cases(self):
        # Test geographic angles including negative and > 360 cases
        geographic_angles = np.array([-90, -45, 0, 45, 360, 450, 720])
        expected_arithmetic_angles = np.array([180, 135, 90, 45, 90, 0, 90])
        result = geographic_to_arithmetic(geographic_angles)
        np.testing.assert_almost_equal(result, expected_arithmetic_angles)

    def test_arithmetic_to_geographic_edge_cases(self):
        # Test arithmetic angles including negative and > 360 cases
        arithmetic_angles = np.array([-90, -45, 0, 45, 360, 450, 720])
        expected_geographic_angles = np.array([180, 135, 90, 45, 90, 0, 90])
        result = arithmetic_to_geographic(arithmetic_angles)
        np.testing.assert_almost_equal(result, expected_geographic_angles)

    def test_geographic_to_arithmetic_signed180_single_value(self):
        # Test conversion with signed180=True for a single value
        geographic_angle = 270
        expected_arithmetic_angle = -180
        result = geographic_to_arithmetic(geographic_angle, signed180=True)
        self.assertAlmostEqual(result, expected_arithmetic_angle)

    def test_geographic_to_arithmetic_signed180_array(self):
        # Test conversion with signed180=True for an array of values
        geographic_angles = np.array([30, 90, 180, 270, 360])
        expected_arithmetic_angles = np.array([60, 0, -90, -180, 90])
        result = geographic_to_arithmetic(geographic_angles, signed180=True)
        np.testing.assert_almost_equal(result, expected_arithmetic_angles)

    def test_random_angle_conversion(self):
        # Generate random angles in range [-720, 720] to cover all edge cases
        np.random.seed(42)  # Ensure reproducibility
        random_geographic_angles = np.random.uniform(-720, 720, 1000)
        random_arithmetic_angles = np.random.uniform(-720, 720, 1000)

        # Convert and back-convert to check consistency
        converted_arithmetic = geographic_to_arithmetic(
            random_geographic_angles
        )
        round_trip_geographic = arithmetic_to_geographic(converted_arithmetic)
        np.testing.assert_almost_equal(
            round_trip_geographic % 360, random_geographic_angles % 360
        )

        converted_geographic = arithmetic_to_geographic(
            random_arithmetic_angles
        )
        round_trip_arithmetic = geographic_to_arithmetic(converted_geographic)
        np.testing.assert_almost_equal(
            round_trip_arithmetic % 360, random_arithmetic_angles % 360
        )

    def test_load_mat_file(self):
        mat_file = "./img_seq_welton_main_drain/welton_main_drain_matlab_stiv_exuasitve_resutls.mat"
        loaded_variables = load_mat_file(mat_file)

        # Assert that the loaded variables match the expected values
        np.testing.assert_array_equal(
            loaded_variables["pixSize"], np.array(0.0061)
        )
        self.assertEqual(loaded_variables["__version__"], "1.0")

    #
    # def test_new_coordinates(self):
    #     vector_Z = np.array([3, 4, 5, 6])  # Example vector Z: (x, y, u, v)
    #     angle_m = np.radians(30)  # Example angle in radians
    #     _, u_new, v_new = calculate_projection(vector_Z, angle_m)
    #     self.assertAlmostEqual(u_new, 1.76776695)
    #     self.assertAlmostEqual(v_new, 5.23223305)


class TestParseCreationTime(unittest.TestCase):

    # === Updated versions of your existing tests ===
    def test_valid_timestamp_with_microseconds(self):
        ts = "2022-05-14T20:24:55.123456Z"
        expected = datetime(2022, 5, 14, 20, 24, 55, 123456)
        self.assertEqual(parse_creation_time(ts), expected)

    def test_valid_timestamp_without_microseconds(self):
        ts = "2022-05-14T20:24:55Z"
        expected = datetime(2022, 5, 14, 20, 24, 55)
        self.assertEqual(parse_creation_time(ts), expected)

    def test_invalid_timestamp_format(self):
        ts = "2022/05/14 20:24:55"
        expected = datetime(2022, 5, 14, 20, 24, 55)
        self.assertEqual(parse_creation_time(ts), expected)

    def test_empty_string(self):
        self.assertIsNone(parse_creation_time(""))

    # === New fuzzy/extended parsing test cases ===
    def test_compact_timestamp_with_fractional_seconds(self):
        ts = "Remote+Record_20250406T113118.732-05_00"
        expected = datetime(2025, 4, 6, 11, 31, 18, 732000)
        self.assertEqual(parse_creation_time(ts), expected)

    def test_aug_day_format(self):
        ts = "NJ_Ramapo_Pompton_Dam_Spillway_full-Aug31"
        parsed = parse_creation_time(ts)
        self.assertEqual(parsed.month, 8)
        self.assertEqual(parsed.day, 31)

    def test_compact_datetime_with_dash(self):
        ts = "09085000-20250327-114207"
        expected = datetime(2025, 3, 27, 11, 42, 7)
        self.assertEqual(parse_creation_time(ts), expected)

    def test_14_digit_compact_datetime(self):
        ts = "20160824100501"
        expected = datetime(2016, 8, 24, 10, 5, 1)
        self.assertEqual(parse_creation_time(ts), expected)

    def test_random_string_with_embedded_date(self):
        ts = "logfile_run3_20230928_extra"
        expected = datetime(2023, 9, 28)
        self.assertEqual(parse_creation_time(ts), expected)

    def test_string_with_multiple_dates_prefers_first(self):
        ts = "cam20230101_other20230201T120000.000"
        expected = datetime(2023, 2, 1, 12, 0)
        self.assertEqual(parse_creation_time(ts), expected)


class TestCreateGrayscaleImageStack(unittest.TestCase):
    # Define a setup method to prepare any necessary resources or data for the tests
    def setUp(self):
        # Create a temporary directory to store test images
        self.temp_dir = "test_images"
        os.makedirs(self.temp_dir, exist_ok=True)

        # Create some test images
        for i in range(3):
            image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
            file_path = os.path.join(self.temp_dir, f"test_image_{i}.jpg")
            cv2.imwrite(file_path, image)

    # Define a teardown method to clean up after the tests
    def tearDown(self):
        # Remove the temporary directory and test images
        for file in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, file)
            os.remove(file_path)
        os.rmdir(self.temp_dir)

    def test_image_stack_shape(self):
        image_paths = [
            os.path.join(self.temp_dir, f"test_image_{i}.jpg")
            for i in range(3)
        ]
        image_stack = create_grayscale_image_stack(image_paths)
        self.assertEqual(
            image_stack.shape, (100, 100, 3)
        )  # Ensure the shape is as expected

    def test_image_stack_dtype(self):
        image_paths = [
            os.path.join(self.temp_dir, f"test_image_{i}.jpg")
            for i in range(3)
        ]
        image_stack = create_grayscale_image_stack(image_paths)
        self.assertEqual(
            image_stack.dtype, np.uint8
        )  # Ensure the data type is as expected

    def test_image_stack_values(self):
        image_paths = [
            os.path.join(self.temp_dir, f"test_image_{i}.jpg")
            for i in range(3)
        ]
        image_stack = create_grayscale_image_stack(image_paths)
        # Ensure that all values are within the valid grayscale range [0, 255]
        self.assertTrue(np.all(image_stack >= 0))
        self.assertTrue(np.all(image_stack <= 255))

    # 12/14/2024: Removed test b/c this is not the expected behaviour now.
    # Instead, the function skips frames that don't have the same resolution
    # as the same frame, rather than throw ValueError.
    # def test_resolution_mismatch_error(self):
    #     # Create two images with different resolutions
    #     image1 = np.zeros((100, 100), dtype=np.uint8)
    #     image2 = np.zeros((200, 200), dtype=np.uint8)
    #
    #     # Create a temporary directory to store the test images
    #     cv2.imwrite(os.path.join(self.temp_dir, "image1.jpg"), image1)
    #     cv2.imwrite(os.path.join(self.temp_dir, "image2.jpg"), image2)
    #
    #     # Define image_paths with the paths to the test images
    #     image_paths = [
    #         os.path.join(self.temp_dir, "image1.jpg"),
    #         os.path.join(self.temp_dir, "image2.jpg"),
    #     ]
    #
    #     # Verify that create_grayscale_image_stack raises an error
    #     with self.assertRaises(ValueError):
    #         create_grayscale_image_stack(image_paths)


# Ensure a QApplication instance exists before running tests
app = QApplication([])


class TestGetColumnContents(unittest.TestCase):
    def setUp(self):
        """Set up a QTableWidget for testing."""
        self.table = QTableWidget(3, 3)  # Create a 3x3 table

        # Fill table with sample data
        self.table.setItem(0, 0, QTableWidgetItem("Row0-Col0"))
        self.table.setItem(0, 1, QTableWidgetItem("Row0-Col1"))
        self.table.setItem(0, 2, QTableWidgetItem("Row0-Col2"))

        self.table.setItem(1, 0, QTableWidgetItem("Row1-Col0"))
        self.table.setItem(1, 1, QTableWidgetItem("Row1-Col1"))
        # Leave self.table[1][2] empty (None)

        self.table.setItem(2, 0, QTableWidgetItem("Row2-Col0"))
        # Leave self.table[2][1] and self.table[2][2] empty (None)

    def test_full_column(self):
        """Test a column with all cells filled."""
        result = get_column_contents(self.table, 0)
        expected = {0: "Row0-Col0", 1: "Row1-Col0", 2: "Row2-Col0"}
        self.assertEqual(result, expected)

    def test_partial_column(self):
        """Test a column with some empty cells."""
        result = get_column_contents(self.table, 1)
        expected = {0: "Row0-Col1", 1: "Row1-Col1", 2: ""}
        self.assertEqual(result, expected)

    def test_empty_column(self):
        """Test a column with all cells empty."""
        result = get_column_contents(self.table, 2)
        expected = {0: "Row0-Col2", 1: "", 2: ""}
        self.assertEqual(result, expected)

    def test_invalid_column_index(self):
        """Test behavior with an invalid column index."""
        with self.assertRaises(IndexError):
            get_column_contents(self.table, 3)  # Out of bounds

    def test_empty_table(self):
        """Test behavior with an empty table."""
        empty_table = QTableWidget(0, 0)
        with self.assertRaises(IndexError):
            get_column_contents(empty_table, 0)


class TestSetColumnContents(unittest.TestCase):
    def setUp(self):
        """Set up a QTableWidget for testing."""
        self.table = QTableWidget(3, 3)  # Create a 3x3 table
        self.table.setItem(0, 0, QTableWidgetItem("Row0-Col0"))
        self.table.setItem(1, 0, QTableWidgetItem("Row1-Col0"))
        self.table.setItem(2, 0, QTableWidgetItem("Row2-Col0"))

    def test_valid_input(self):
        """Test setting a column with valid data."""
        data = {0: "Row0-Col1", 1: "Row1-Col1", 2: "Row2-Col1"}
        set_column_contents(self.table, 1, data)  # Set data in column 1

        for row, value in data.items():
            self.assertEqual(self.table.item(row, 1).text(), value)

    def test_mismatched_row_count(self):
        """Test behavior when the row count in the dictionary doesn't match the table."""
        data = {0: "Row0-Col1", 1: "Row1-Col1"}  # Only 2 rows
        with self.assertRaises(ValueError) as context:
            set_column_contents(self.table, 1, data)
        self.assertIn("Row count mismatch", str(context.exception))

    def test_invalid_column_index(self):
        """Test behavior when the column index is out of range."""
        data = {0: "Row0-Col2", 1: "Row1-Col2", 2: "Row2-Col2"}
        with self.assertRaises(IndexError) as context:
            set_column_contents(self.table, 5, data)  # Invalid column index
        self.assertIn("Column index 5 is out of range", str(context.exception))

    def test_invalid_row_index(self):
        """Test behavior when the dictionary contains invalid row indices."""
        data = {0: "Row0-Col1", 3: "Row3-Col1"}  # Row 3 doesn't exist
        with self.assertRaises(ValueError):
            set_column_contents(self.table, 1, data)

    def test_empty_table(self):
        """Test behavior with an empty table."""
        empty_table = QTableWidget(0, 0)
        data = {}
        with self.assertRaises(IndexError) as context:
            set_column_contents(empty_table, 0, data)
        self.assertIn("Table has no columns", str(context.exception))

    def test_empty_data_dict(self):
        """Test setting a column with an empty dictionary."""
        data = {}
        with self.assertRaises(ValueError) as context:
            set_column_contents(self.table, 1, data)
        self.assertIn("Row count mismatch", str(context.exception))

    def test_column_with_empty_values(self):
        """Test setting a column with some empty string values."""
        data = {0: "Row0-Col1", 1: "", 2: "Row2-Col1"}
        set_column_contents(self.table, 1, data)

        for row, value in data.items():
            self.assertEqual(self.table.item(row, 1).text(), value)


# I separated this to expand on the testing. CJM
class TestComputeVectorsAngles(unittest.TestCase):
    def test_tagline_and_flow_direction(self):
        """
        Tests if the tagline_dir_geog_deg and mean_flow_dir_geog_deg
        match expected values for a vertical tagline and a known flow.
        """

        # Imagine a vertical tagline in image coordinates:
        #   X: all 0, Y goes from 0 to 10
        # Internally, the function flips dy => "arithmetic" angle is -90 deg =>
        # tagline_dir_geog_deg should end up being 180 deg (due to how arithmetic_to_geographic is defined).

        X = np.array([0, 0, 0], dtype=float)
        Y = np.array([0, 5, 10], dtype=float)

        # Set flow in the +X direction (U>0, V=0).
        # For mean flow: arithmetic angle is arctan2(Vavg, Uavg) = arctan2(0, positive) = 0 deg
        # => in geographic: (90 - 0) % 360 = 90 deg
        U = np.array([2, 2, 2], dtype=float)  # purely in +X
        V = np.array([0, 0, 0], dtype=float)

        (
            vectors_image,
            norm_vectors_image,
            normal_unit_vector,
            scalar_projections,
            tagline_dir_geog_deg,
            mean_flow_dir_geog_deg,
        ) = compute_vectors_with_projections(X, Y, U, V)

        # Notes: so the tagline direction in geographic coords. should be 180 deg.
        # for arithmetic, the tagline is at -90 deg (pointing down, makes sense),
        # and arithmetic_to_geographic(angle) = (90 - angle) % 360 => (90 - (-90)) % 360 => 180

        # tagline_dir_geog_deg and mean_flow_dir_geog_deg
        # should be arrays of identical values. I'm not sure if I am checking this correctly >:(
        self.assertAlmostEqual(
            tagline_dir_geog_deg[0],
            180.0,
            delta=1e-6,
            msg="Expected tagline_dir_geog_deg to be ~180° for a vertical tagline in this coordinate setup.",
        )

        # Mean flow direction: U>0, V=0 => arithmetic angle is 0 deg
        # geographic is (90 - 0) % 360 => 90°.
        self.assertAlmostEqual(
            mean_flow_dir_geog_deg[0],
            90.0,
            delta=1e-6,
            msg="Expected mean_flow_dir_geog_deg to be ~90° for purely +X flow under this definition.",
        )

        # to confirm the entire arrays are the same repeating values
        self.assertTrue(
            np.allclose(tagline_dir_geog_deg, 180.0, atol=1e-6),
            "All tagline_dir_geog_deg entries should be 180.0",
        )
        self.assertTrue(
            np.allclose(mean_flow_dir_geog_deg, 90.0, atol=1e-6),
            "All mean_flow_dir_geog_deg entries should be 90.0",
        )
