"""IVy modules containing file management functions"""

import datetime
import json
import logging
import os
import re
import shutil
import tempfile
from glob import glob

import numpy as np
import pandas as pd
import requests
from PyQt5 import QtCore
from packaging.version import Version
from image_velocimetry_tools.common_functions import units_conversion


def create_temp_directory():
    """
    Create a temporary directory for storing interim files and return its path.

    Returns:
        str: The path to the created temporary directory.

    Example:
    >>> temp_dir = create_temp_directory()
    >>> print(temp_dir)
    '/tmp/tmpexample12345'  # The actual path will vary

    Note:
    The temporary directory is automatically created using the `tempfile` module and can be used
    for storing temporary files, memory maps, and other short-lived data. It's important to ensure
    that you clean up this directory when your application is done with it using the
    `clean_up_temp_directory` function.
    """
    temp_dir = tempfile.mkdtemp()
    return temp_dir


def clean_up_temp_directory(temp_dir_path):
    """
    Clean up a temporary directory and its contents.

    Args:
        temp_dir_path (str): The path to the temporary directory to be removed.

    Example:
    >>> temp_dir = create_temp_directory()
    >>> # ... Use the temporary directory for interim files ...
    >>> clean_up_temp_directory(temp_dir)

    Note:
    This function removes the specified temporary directory and all its contents, including any
    interim files and data. It's essential to call this function when your application is done
    using the temporary directory to prevent clutter and save disk space.

    Args:
        temp_dir_path (str): The path to the temporary directory to be cleaned up.

    """
    shutil.rmtree(temp_dir_path)


def make_windows_safe_filename(input_string):
    """
    Make a string safe for use as a Windows filename by replacing disallowed characters and spaces.

    Parameters:
    input_string (str): The input string containing the desired filename.

    Returns:
    str: A sanitized version of the input string with disallowed characters and spaces replaced by underscores.

    The function replaces the following disallowed characters with underscores:
    - '/' (forward slash)
    - '\' (backslash)
    - ':' (colon)
    - '*' (asterisk)
    - '?' (question mark)
    - '"' (double quotation marks)
    - '<' (less than)
    - '>' (greater than)
    - '|' (pipe)

    It also replaces spaces with underscores. Additionally, it removes leading and trailing underscores if present.

    Example:
    >>> make_windows_safe_filename("My:File?Name/with Spaces<and|bars>")
    'My_File_Name_with_Spaces_and_bars_'
    """
    # Define a regex pattern to match disallowed characters in Windows filenames
    disallowed_chars = r'[\/:*?"<>|]'

    # Replace disallowed characters and spaces with underscores
    safe_filename = re.sub(disallowed_chars, "_", input_string)

    # Replace spaces with underscores
    safe_filename = safe_filename.replace(" ", "_")
    return safe_filename


def format_windows_path(path):
    """
    Format a Windows file path by replacing backslashes with forward slashes and wrapping it in double quotes if it contains spaces.

    Parameters:
    path (str): The input file path to be formatted.

    Returns:
    str: The formatted file path with double quotes if spaces are present, or the original path with backslashes replaced by forward slashes if no spaces are found.

    Examples:
    >>> format_windows_path("C:\\Program Files (x86)\\WinRar\\Rar.exe")
    '"C:/Program Files (x86)/WinRar/Rar.exe"'

    >>> format_windows_path("D:\\ProgramFiles\\SomeApp\\App.exe")
    'D:/ProgramFiles/SomeApp/App.exe'
    """
    # Replace backslashes with forward slashes
    path = path.replace("\\", "/")

    # Check if the path contains spaces
    if " " in path:
        # Wrap the path in double quotes
        formatted_path = f'"{path}"'
    else:
        formatted_path = path

    return formatted_path


def safe_make_directory(path, overwrite=False):
    """Create a directory in the specified path. Overwrite existing directed if prompted."""
    os.makedirs(path, exist_ok=True)
    if overwrite:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))
    return format_windows_path(path)


def serialize_numpy_array(numpy_array):
    """Convert a numpy array to a serialized JSON string

    Args:
        numpy_array (ndarray): the input numpy array

    Returns:
        str: serialized JSON string
    """
    if isinstance(numpy_array, np.ndarray):
        serialized_array = (
            numpy_array.tolist()
        )  # Convert the NumPy array to a nested list
        return json.dumps(serialized_array)
    else:
        raise ValueError("Input is not a NumPy array")


def deserialize_numpy_array(serialized_json):
    """Convert a JSON seriealized string into a numpy array

    Args:
        serialized_json (str): the JSON array string

    Returns:
        ndarray: a numpy array
    """
    serialized_list = json.loads(serialized_json)
    if isinstance(serialized_list, list):
        return np.array(serialized_list)
    else:
        raise ValueError("Input is not a serialized NumPy array")


def locate_video_file(project_dict):
    """
    Validates and locates a video file for the project.

    Args:
        project_dict (dict): The project data containing a possible video file path.

    Returns:
        str: Path to the located video file or None if not found.
    """
    # Extract video file path from the project dictionary
    video_file_path = project_dict.get("video_file_name", None)

    if video_file_path:
        # Check if the file exists
        if os.path.isfile(video_file_path):
            logging.info(f"Video file found: {video_file_path}")
            return video_file_path
        else:
            logging.warning(
                f"Video file not found at specified path: {video_file_path}"
            )

    # Try finding the video base name in the project directory
    project_file_path = project_dict.get("project_file_path", None)
    if not project_file_path:
        logging.error(
            "Project file path is not specified in the project dictionary."
        )
        return None

    project_dir = os.path.dirname(project_file_path)
    video_base_name = (
        os.path.basename(video_file_path) if video_file_path else None
    )

    if video_base_name:
        potential_match = os.path.join(project_dir, video_base_name)
        if os.path.isfile(potential_match):
            logging.info(
                f"Video file located in project directory: {potential_match}"
            )
            return potential_match
        else:
            logging.warning(
                f"Video base name '{video_base_name}' not found in project directory."
            )

    # # No specific video file found, look for any supported formats
    # supported_formats = (
    #     "*.mp4",
    #     "*.avi",
    #     "*.mov",
    #     "*.mkv",
    #     "*.wmv",
    # )  # Add other formats as needed
    #
    # for ext in supported_formats:
    #     video_files = glob(os.path.join(project_dir, ext))
    #     if video_files:
    #         logging.info(f"Video file located in project directory: {video_files[0]}")
    #         return video_files[0]

    # No video file found
    logging.error("No video file found in the project directory.")
    return None


def set_date(date_str, date_edit):
    """
    Set the date for a QDateEdit widget.

    Parameters
    ----------
    date_str : str or None
        The date string in the format "MM/DD/YYYY". If None, no action is taken.
        If an empty string, defaults to "10/1/2023".
    date_edit : QDateEdit
        The QDateEdit widget to set the date on.

    Notes
    -----
    This function parses the provided date string and converts it to a QtCore.QDate
    object to set the date on the given QDateEdit widget.
    """
    if date_str is not None:
        if date_str == "":
            date_object = datetime.datetime.strptime("10/1/2023", "%m/%d/%Y")
            q_date = QtCore.QDate(
                date_object.year, date_object.month, date_object.day
            )
            date_edit.setDate(q_date)
        else:
            date_object = datetime.datetime.strptime(date_str, "%m/%d/%Y")
            q_date = QtCore.QDate(
                date_object.year, date_object.month, date_object.day
            )
            date_edit.setDate(q_date)


def set_time(time_str, time_edit):
    """
    Set the time for a QTimeEdit widget.

    Parameters
    ----------
    time_str : str or None
        The time string in the format "HH:MM:SS". If None, no action is taken.
        If an empty string, defaults to "12:00:00".
    time_edit : QTimeEdit
        The QTimeEdit widget to set the time on.

    Notes
    -----
    This function parses the provided time string and converts it to a QtCore.QTime
    object to set the time on the given QTimeEdit widget.
    """
    if time_str is not None:
        if time_str == "":
            time_object = datetime.datetime.strptime("12:00:00", "%H:%M:%S")
            q_time = QtCore.QTime(
                time_object.hour, time_object.minute, time_object.second
            )
            time_edit.setTime(q_time)
        else:
            time_object = datetime.datetime.strptime(time_str, "%H:%M:%S")
            q_time = QtCore.QTime(
                time_object.hour, time_object.minute, time_object.second
            )
            time_edit.setTime(q_time)


def set_value_if_not_none(value, widget, value_type=float):
    """
    Set a value on a widget if the value is not None.

    Parameters
    ----------
    value : any
        The value to set. Must be compatible with `value_type`.
    widget : QWidget
        The widget to set the value on. Must have a `setValue` method.
    value_type : type, optional
        The type to cast `value` to before setting, by default float.

    Notes
    -----
    This function checks if the value is not None before casting it to the specified
    type and setting it on the widget.
    """
    if value is not None:
        widget.setValue(value_type(value))


def set_text_if_not_none(value, widget):
    """
    Set text on a widget if the value is not None.

    Parameters
    ----------
    value : str or None
        The text to set. If None, no action is taken.
    widget : QWidget
        The widget to set the text on. Must have a `setText` method.

    Notes
    -----
    This function checks if the value is not None before setting it as the text
    on the widget.
    """
    if value is not None:
        widget.setText(value)


def validate_version_format(version: str):
    """Ensure the version is a valid 4-part semantic version, including optional pre-release identifiers."""
    if not re.fullmatch(r"v?\d+\.\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?", version):
        raise ValueError(f"Invalid version format: {version}")


def compare_versions_core(app_version: str, webpage_version: str) -> str:
    """Core function to Compare the application version against the version supplied by a webpage

    Args:
        app_version (str): the version of the app, such as ivy.__version__
        webpage_version (str): a string read from a website with the current version

    Returns:
        str: _description_
    """
    # Validate version formats
    validate_version_format(app_version)
    validate_version_format(webpage_version)

    # Strip "v" prefix if present and create Version objects
    app_ver = Version(app_version.lstrip("v"))
    web_ver = Version(webpage_version.lstrip("v"))

    if app_ver == web_ver:
        return "IVy is up to date."
    elif app_ver.major == web_ver.major:
        return "IVy is on the same major version, but minor versions or patches behind."
    else:
        return "IVy is behind by a major version."


def compare_versions(app_version: str, url: str):
    """Compare the application version against the version supplied by a webpage

    Args:
        app_version (str): the version of the app, such as ivy.__version__
        url (str): the URL associated with the current version tag

    Returns:
        str: result of the version comparison
    """
    try:
        # Fetch the webpage content
        response = requests.get(url, verify=False)
        response.raise_for_status()
        html_content = response.text

        # Use regex to extract the version from the `ivy-version` div
        match = re.search(
            r'<div\s+id="ivy-version">\s*([vV]?\d+\.\d+\.\d+\.\d+)\s*</div>',
            html_content,
        )
        if not match:
            print(
                "Error: Couldn't find the version div or version text on the webpage."
            )
            return None

        webpage_version = match.group(1).strip()

        # Compare versions
        app_ver = Version(app_version)
        web_ver = Version(webpage_version)

        result = compare_versions_core(app_version, webpage_version)

        # print(result)
        return result
        # if app_ver == web_ver:
        #     print("IVy is up to date.")
        # elif app_ver.major == web_ver.major:
        #     print(
        #         "IVy is on the same major version, but minor versions or patches behind.")
        # else:
        #     print("IVy is behind by a major version.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def load_and_parse_gcp_csv(file_name, swap_ortho_path, unit_prompt_callback):
    """This function loads the GCP CSV file and does necessary units conversions
    The resulting dataframe will always be in metric units.

    """
    df = pd.read_csv(
        file_name,
        header=None,
        delimiter=",",
        keep_default_na=False,
        on_bad_lines="skip",
    )
    header = df.iloc[0]
    df.columns = header.values.tolist()

    unit_str = ""
    survey_units = "Metric"

    # Check for units in header
    units_in_header = any(
        label in header.values.tolist()
        for label in ["X (m)", "Y (m)", "Z (m)", "X (ft)", "Y (ft)", "Z (ft)"]
    )

    if not units_in_header:
        if os.path.normpath(file_name) == os.path.normpath(
            os.path.join(swap_ortho_path, "ground_control_points.csv")
        ):
            survey_units = "Metric"
        else:
            user_choice = (
                unit_prompt_callback()
            )  # returns "Metric" or "English"
            survey_units = user_choice
    else:
        for label in ["X (m)", "Y (m)", "Z (m)", "X (ft)", "Y (ft)", "Z (ft)"]:
            if label in header.values.tolist():
                if "m" in label:
                    survey_units = "Metric"
                    unit_str = " (m)"
                elif "ft" in label:
                    survey_units = "English"
                    unit_str = " (ft)"
                break

    # Trim header
    df = df[1:]

    # Add missing columns as needed
    col_count = len(df.columns)
    if col_count == 6:
        df["Error X (pixel)"] = "N/A"
        df["Error Y (pixel)"] = "N/A"
        df["Tot. Error (pixel)"] = "N/A"
        df["Use in Rectification"] = "Yes"
        df["Use in Validation"] = "No"
        columns_to_convert = [
            f"X{unit_str}",
            f"Y{unit_str}",
            f"Z{unit_str}",
            "X (pixel)",
            "Y (pixel)",
        ]
    elif col_count == 4:
        df["X (pixel)"] = "N/A"
        df["Y (pixel)"] = "N/A"
        df["Error X (pixel)"] = "N/A"
        df["Error Y (pixel)"] = "N/A"
        df["Tot. Error (pixel)"] = "N/A"
        df["Use in Rectification"] = "Yes"
        df["Use in Validation"] = "No"
        columns_to_convert = [f"X{unit_str}", f"Y{unit_str}", f"Z{unit_str}"]
    elif col_count == 3:
        df[f"Z{unit_str}"] = np.zeros(len(df))
        df["X (pixel)"] = "N/A"
        df["Y (pixel)"] = "N/A"
        df["Error X (pixel)"] = "N/A"
        df["Error Y (pixel)"] = "N/A"
        df["Tot. Error (pixel)"] = "N/A"
        df["Use in Rectification"] = "Yes"
        df["Use in Validation"] = "No"
        columns_to_convert = [f"X{unit_str}", f"Y{unit_str}", f"Z{unit_str}"]
    elif col_count == 11:
        columns_to_convert = [
            f"X{unit_str}",
            f"Y{unit_str}",
            f"Z{unit_str}",
            "X (pixel)",
            "Y (pixel)",
            "Error X (pixel)",
            "Error Y (pixel)",
            "Tot. Error (pixel)",
        ]
    else:
        raise ValueError("Unsupported column configuration in GCP CSV.")

    # Convert numeric values
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna("N/A")

    # Convert units if necessary
    if survey_units != "Metric":
        factor = 1 / units_conversion("English")["L"]
        for col in [f"X{unit_str}", f"Y{unit_str}", f"Z{unit_str}"]:
            df[col] = pd.to_numeric(df[col], errors="coerce") * factor

    # Rename columns
    df.rename(
        columns={
            f"X{unit_str}": "X",
            f"Y{unit_str}": "Y",
            f"Z{unit_str}": "Z",
        },
        inplace=True,
    )

    return df, survey_units
