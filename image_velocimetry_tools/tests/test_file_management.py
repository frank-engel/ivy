import unittest
import io
import os
import tempfile
import shutil
import cv2

# Import the functions you want to test
from image_velocimetry_tools.file_management import *


class TestTemporaryDirectoryFunctions(unittest.TestCase):
    def test_create_temp_directory(self):
        # Test if the function returns a valid directory path
        temp_dir = create_temp_directory()
        self.assertTrue(os.path.isdir(temp_dir))
        # Clean up the directory after testing
        clean_up_temp_directory(temp_dir)

    def test_clean_up_temp_directory(self):
        # Create a temporary directory and add a file
        temp_dir = create_temp_directory()
        temp_file = os.path.join(temp_dir, "test_file.txt")
        with open(temp_file, "w") as f:
            f.write("Test data")
        self.assertTrue(os.path.exists(temp_file))
        # Test if the function removes the directory and its contents
        clean_up_temp_directory(temp_dir)
        self.assertFalse(os.path.exists(temp_dir))
        self.assertFalse(os.path.exists(temp_file))


class TestMakeWindowsSafeFilename(unittest.TestCase):
    def test_no_disallowed_characters(self):
        input_string = "SafeFilename123"
        expected_result = "SafeFilename123"
        self.assertEqual(make_windows_safe_filename(input_string), expected_result)

    def test_with_disallowed_characters(self):
        input_string = "File/Name:with?disallowed*chars<>"
        expected_result = "File_Name_with_disallowed_chars__"
        self.assertEqual(make_windows_safe_filename(input_string), expected_result)

    def test_with_spaces(self):
        input_string = "File Name With Spaces"
        expected_result = "File_Name_With_Spaces"
        self.assertEqual(make_windows_safe_filename(input_string), expected_result)

    def test_mix_of_disallowed_characters_and_spaces(self):
        input_string = "My:File?Name/with Spaces<and|bars>"
        expected_result = "My_File_Name_with_Spaces_and_bars_"
        self.assertEqual(make_windows_safe_filename(input_string), expected_result)

    def test_empty_string(self):
        input_string = ""
        expected_result = ""
        self.assertEqual(make_windows_safe_filename(input_string), expected_result)


class TestFormatWindowsPath(unittest.TestCase):
    def test_with_spaces(self):
        path_with_spaces = "C:\\Program Files (x86)\\WinRar\\Rar.exe"
        formatted_path = format_windows_path(path_with_spaces)
        self.assertEqual(formatted_path, '"C:/Program Files (x86)/WinRar/Rar.exe"')

    def test_without_spaces(self):
        path_without_spaces = "D:/ProgramFiles/SomeApp/App.exe"
        formatted_path = format_windows_path(path_without_spaces)
        self.assertEqual(formatted_path, path_without_spaces)


class TestCompareVersions(unittest.TestCase):
    def test_versions_up_to_date(self):
        self.assertEqual(
            compare_versions_core("0.8.2.2", "0.8.2.2"), "IVy is up to date."
        )

    def test_minor_version_behind(self):
        self.assertEqual(
            compare_versions_core("0.8.2.2", "0.8.3.0"),
            "IVy is on the same major version, but minor versions or patches behind.",
        )

    def test_patch_version_behind(self):
        self.assertEqual(
            compare_versions_core("0.8.2.2", "0.8.2.3"),
            "IVy is on the same major version, but minor versions or patches behind.",
        )

    def test_major_version_behind(self):
        self.assertEqual(
            compare_versions_core("0.8.2.2", "1.0.0.0"),
            "IVy is behind by a major version.",
        )

    def test_major_version_ahead(self):
        self.assertEqual(
            compare_versions_core("1.0.0.0", "0.8.2.2"),
            "IVy is behind by a major version.",
        )

    def test_different_major_versions(self):
        self.assertEqual(
            compare_versions_core("2.3.1.0", "1.9.8.7"),
            "IVy is behind by a major version.",
        )

    def test_pre_release_versions(self):
        self.assertEqual(
            compare_versions_core("0.8.2.2", "0.8.2.2-beta"),
            "IVy is on the same major version, but minor versions or patches behind.",
        )

    def test_non_semantic_versions(self):
        with self.assertRaises(ValueError):
            compare_versions_core("0.8.2", "0.8.2.2")

    def test_invalid_version_format(self):
        with self.assertRaises(ValueError):
            compare_versions_core("invalid_version", "0.8.2.2")


class TestLoadAndParseGCPCSV(unittest.TestCase):
    def setUp(self):
        self.csv_english = io.StringIO("""GCP Name,X (ft),Y (ft),Z (ft),X (pixel),Y (pixel)
GCP1,1000,2000,30,400,300
GCP2,1010,2010,32,410,310
""")

        self.csv_bad = io.StringIO("""1000,2000,30
1010,2010,32
""")

        self.csv_minimal = io.StringIO("""GCP Name,X (ft),Y (ft)
GCP1,1000,2000
GCP2,1010,2010
""")

        self.unit_prompt = lambda: "English"

    def test_parse_english_units(self):
        df, units = load_and_parse_gcp_csv(
            self.csv_english, swap_ortho_path="", unit_prompt_callback=self.unit_prompt
        )
        self.assertEqual(units, "English")
        self.assertIn("X", df.columns)
        self.assertAlmostEqual(df["X"].iloc[0], 304.8, delta=0.1)
        self.assertAlmostEqual(df["Y"].iloc[1], 612.65, delta=0.1)
        self.assertAlmostEqual(df["Z"].iloc[0], 9.144, delta=0.01)

    def test_missing_optional_columns(self):
        df, _ = load_and_parse_gcp_csv(
            self.csv_minimal, swap_ortho_path="", unit_prompt_callback=self.unit_prompt
        )
        expected_cols = [
            "X", "Y", "Z",
            "X (pixel)", "Y (pixel)",
            "Error X (pixel)", "Error Y (pixel)",
            "Tot. Error (pixel)",
            "Use in Rectification", "Use in Validation"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)
        self.assertEqual(df["Z"].iloc[0], 0.0)
        self.assertTrue(df["Use in Rectification"].iloc[0])

    def test_numeric_conversion(self):
        df, _ = load_and_parse_gcp_csv(
            self.csv_english, swap_ortho_path="", unit_prompt_callback=self.unit_prompt
        )
        self.assertTrue(pd.to_numeric(df["X"], errors="coerce").notnull().all())
        self.assertTrue(pd.to_numeric(df["Z"], errors="coerce").notnull().all())


if __name__ == "__main__":
    unittest.main()
