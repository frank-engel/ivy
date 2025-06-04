"""IVy module containing common functions used by other modules"""

import os
import sys
from datetime import datetime, timezone
from dateutil import parser as dateutil_parser
import re
import numpy as np
import scipy.io
from PIL import Image
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem


def resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def string_to_boolean(string: str) -> bool:
    """Converts a string with a truthy value to its corresponding boolean."""
    if isinstance(string, bool):
        return string
    elif string is None:
        return False
    elif not isinstance(string, str):
        raise TypeError("Expected a string input.")

    string_lower = string.lower()
    if string_lower in ("yes", "true", "t", "y", "1"):
        return True
    elif string_lower in ("no", "false", "f", "n", "0", ""):
        return False
    else:
        raise ValueError(f"Invalid string value: '{string}'")


def find_key_from_first_value(dictionary: dict, value: object) -> object:
    """Returns the first key in the dictionary that maps to the specified value,
    or None if no such key exists.
    """
    return next((k for k, v in dictionary.items() if v == value), None)


def find_matches_between_two_lists(list1: list, list2: list) -> list:
    """For each item in list1, return a list of offsets to its occurrences in list2"""
    return [[pos for pos, j in enumerate(list2) if i == j] for i in list1]


def float_seconds_to_time_string(seconds: float, precision: str = "hundredth") -> str:
    """Converts seconds expressed as a float to a string of the given format (e.g. SS.ss -> MM:SS.ss)."""
    only_seconds = seconds
    hours, frac_hours = divmod(seconds / 3600, 1)
    minutes, frac_minutes = divmod(frac_hours * 60, 1)
    seconds, frac_seconds = divmod(frac_minutes * 60, 1)

    if precision == "hundredth":
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds + frac_seconds:05.2f}"
    elif precision == "second":
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    elif precision == "only_seconds":
        return f"{only_seconds:.2f}"
    else:
        raise ValueError(f"Invalid precision value: {precision}")


def seconds_to_hhmmss(seconds: float, precision="low") -> str:
    """Converts seconds expressed as a float into a hh:mm:ss format."""
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if precision.lower() == "low":
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    elif precision.lower() == "high":
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
    else:
        raise ValueError("Invalid precision value. Valid values are 'low' and 'high'.")


def hhmmss_to_seconds(timestr: str) -> float:
    """Convert a hhmmss string to seconds

    Args:
        timestr (str): the input str in hhmmss format

    Returns:
        float: the number of sections in the string as a float
    """
    timestr = timestr.strip().replace(',',
                                      '.')  # Support comma as decimal separator
    parts = timestr.split(':')
    if len(parts) != 3:
        raise ValueError(
            f"Invalid time format: '{timestr}'. Expected hh:mm:ss[.sss]")

    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    except ValueError as e:
        raise ValueError(
            f"Unable to parse time string '{timestr}' to seconds: {e}")


def seconds_to_frame_number(seconds, fps):
    """Return the nearest frame number of a time in seconds given fps"""
    return int(seconds * fps)


def framenum_to_seconds(frame_number, fps):
    """Return the seconds timestamp of a frame given fps"""
    return frame_number / fps

def parse_creation_time(timestamp_str):
    if not isinstance(timestamp_str, str) or not timestamp_str.strip():
        return None

    # Complex patterns need to be listed first
    patterns = [
        ("%Y%m%dT%H%M%S.%f", r"\d{8}T\d{6}\.\d+"),
        ("%Y%m%dT%H%M%S", r"\d{8}T\d{6}"),
        (
        "%Y-%m-%dT%H:%M:%S.%fZ", r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z"),
        ("%Y-%m-%dT%H:%M:%SZ", r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"),
        ("%Y%m%d-%H%M%S", r"20\d{6}-\d{6}"),
        ("%Y%m%d%H%M%S", r"\d{14}"),
        ("%Y%m%d", r"\d{8}"),
        ("%b%d", r"[A-Za-z]{3}\d{1,2}")
    ]

    for fmt, regex in patterns:
        match = re.search(regex, timestamp_str)
        if match:
            try:
                return datetime.strptime(match.group(), fmt)
            except ValueError:
                continue

    # fallback to fuzzy parsing
    try:
        from dateutil import parser
        return parser.parse(timestamp_str, fuzzy=True)
    except Exception:
        return None


def quotify_a_string(string: str) -> str:
    """If needed, adds quotes to strings with spaces.

    This function can be used to check and modify an input string (e.g., a file path with spaces) to make it "safe"
    for use as a command line argument. If the input string has spaces, the function will return the string in
    quotes. Otherwise, it returns the original string.
    """
    string = string.replace("'", '"')
    if " " in string:
        string = '"' + string + '"'
    return string


def distance(point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance between two cartesian points."""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def midpoint(point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
    """Compute Euclidean midpoint between two cartesian points."""
    return np.array([(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2])


def centroid(points: np.ndarray) -> np.ndarray:
    """Find the centroid of all points in an array."""
    length = points.shape[0]
    sum_x = np.sum(points[:, 0])
    sum_y = np.sum(points[:, 1])
    return np.array([sum_x / length, sum_y / length])


def scale_coordinates(coordinates: np.ndarray, scaleFactor: float = 1.0) -> list:
    """Scale input coordinates symmetrically by specified factor."""
    return [
        (scaleFactor * (point[0]), scaleFactor * (point[1])) for point in coordinates
    ]


def translate_coordinates(
    coordinates: np.ndarray, translateX: float = 0, translateY: float = 0
) -> list:
    """Translate input coordinates by specified X and Y units."""
    return [((point[0]) + translateX, (point[1]) + translateY) for point in coordinates]


def bounding_box_naive(points: np.ndarray) -> list:
    """Return two points as tuples with lower left and upper right point bounding box coordinates."""
    bottomLeftX = min(point[0] for point in points)
    bottomLeftY = min(point[1] for point in points)
    topRightX = max(point[0] for point in points)
    topRightY = max(point[1] for point in points)
    return [(bottomLeftX, bottomLeftY), (topRightX, topRightY)]


def pillow_image_to_numpy_array(im: object) -> np.ndarray:
    """Convert PIL image to ndarray with single copy method

    Parameters
    ----------
    im : PIL image

    Returns
    -------
    data : ndarray representation of im

    Notes
    -----
    Authored by Alex Karpisnky, see https://uploadcare.com/blog/fast-import-of-pillow-images-to-numpy-opencv-arrays/
    for more information.
    """

    im.load()
    # unpack data
    e = Image._getencoder(im.mode, "raw", im.mode)
    e.setimage(im.im)

    # NumPy buffer for the result
    shape, typestr = Image._conv_type_shape(im)
    data = np.empty(shape, dtype=np.dtype(typestr))
    mem = data.data.cast("B", (data.data.nbytes,))

    bufsize, s, offset = 65536, 0, 0
    while not s:
        l, s, d = e.encode(bufsize)
        mem[offset : offset + len(d)] = d
        offset += len(d)
    if s < 0:
        raise RuntimeError("encoder error %d in tobytes" % s)
    return data


def get_normal_vectors(v1, v2, num_vectors=10):
    """
    Returns a list of evenly spaced normal vectors along a line defined by two vertices.

    Parameters
    ----------
    v1 : numpy.ndarray
        The first vertex of the line.
    v2 : numpy.ndarray
        The second vertex of the line.
    num_vectors : int, optional
        The number of evenly spaced normal vectors to generate, by default 10.

    Returns
    -------
    list of numpy.ndarray
        A list of normal vectors, each represented as a numpy array.
    list of numpy.ndarray
        A list of vector tail locations, where each location is evenly spaced along the supplied v1, v2
    float
        The distance between each vector tail along the line defined by v1, v2

    Raises
    ------
    ValueError
        If the two vertices are identical.

    Examples
    --------
    >>> v1 = np.array([0, 0])
    >>> v2 = np.array([4, 0])
    >>> get_normal_vectors(v1, v2)
    [array([0, 0.2]), array([0, 0.4]), array([0, 0.6]), array([0, 0.8]), array([0, 1.0]),
     array([0, 1.2]), array([0, 1.4]), array([0, 1.6]), array([0, 1.8]), array([0, 2.0])]

    >>> v1 = np.array([1, 2])
    >>> v2 = np.array([5, 6])
    >>> get_normal_vectors(v1, v2, num_vectors=5)
    [array([1.14142136, 1.85857864]), array([2.27279221, 3.55563593]), array([3.40416307, 5.25269423]),
     array([4.53553393, 6.94975253]), array([5.66690479, 8.64681083])]

    >>> v1 = np.array([0, 0])
    >>> v2 = np.array([0, 4])
    >>> get_normal_vectors(v1, v2, num_vectors=15)
    [array([0.2, 0. ]), array([0.4, 0. ]), array([0.6, 0. ]), array([0.8, 0. ]), array([1., 0.]),
     array([1.2, 0. ]), array([1.4, 0. ]), array([1.6, 0. ]), array([1.8, 0. ]), array([2., 0.]),
     array([2.2, 0. ]), array([2.4, 0. ]), array([2.6, 0. ]), array([2.8, 0. ]), array([3., 0.])]

    """
    # Calculate the direction vector and length of the line
    direction = v2 - v1
    length = np.linalg.norm(direction)

    # Calculate the unit direction vector
    unit_direction = direction / length

    # Calculate the normal vector
    normal = np.array([-unit_direction[1], unit_direction[0]])

    # Calculate the step size for evenly spacing the normal vectors
    step_size = length / (num_vectors + 1)

    # Create the vector locations. evenly spaced points along the line
    vector_tails = np.vstack(
        [np.linspace(v1[0], v2[0], num_vectors), np.linspace(v1[1], v2[1], num_vectors)]
    ).T

    # Calculate the normal vectors (vector heads) for each point
    normal_vectors = np.tile(normal, num_vectors).reshape(vector_tails.shape)

    # normal_vectors = []
    # for i in range(1, num_vectors + 1):
    #     offset = i * step_size
    #     normal_vectors.append(v1 + unit_direction * offset + normal * (length / (2 * (num_vectors + 1))))

    return normal_vectors, vector_tails, step_size


def get_vector_heads(normal_vectors, vector_tails, distance=1):
    """
    Calculates the vector head coordinates given the normal vectors, vector
    tail locations, and a user-specified distance.

    Parameters
    ----------
    normal_vectors : list of numpy.ndarray
        List of normal vectors, each represented as a numpy array.
    vector_tails : list of numpy.ndarray
        List of vector tail locations, where each location is evenly spaced
        along the line.
    distance : float
        The distance between each vector tail along the line defined by v1,
        v2. The default is 1.

    Returns
    -------
    list of numpy.ndarray
        A list of vector head coordinates, each represented as a numpy array.
    """

    vector_heads = []
    for i in range(len(normal_vectors)):
        vector_head = vector_tails[i] + distance * normal_vectors[i]
        vector_heads.append(vector_head)

    return vector_heads


def geographic_to_arithmetic(geographic_angle, signed180=False):
    """
    Convert a geographic angle to an arithmetic angle.

    Parameters:
    geographic_angle (float or ndarray): The geographic angle(s) to convert.
    signed180 (bool): If True, returns angles in the range [-180, 180].

    Returns:
    float or ndarray: The corresponding arithmetic angle(s) in degrees.
    """
    geographic_angle = np.where(geographic_angle < 0, 360 + geographic_angle,
                                geographic_angle)
    arithmetic_angle = (450 - np.array(geographic_angle)) % 360

    if signed180:
        arithmetic_angle = np.where(arithmetic_angle >= 180,
                                    arithmetic_angle - 360, arithmetic_angle)

    return arithmetic_angle


def arithmetic_to_geographic(arithmetic_angle):
    """
    Convert an arithmetic angle to a geographic angle.

    Parameters:
    arithmetic_angle (float or ndarray): The arithmetic angle(s) to convert.

    Returns:
    float or ndarray: The corresponding geographic angle(s) in degrees.
    """

    geographic_angle = (90 - np.array(arithmetic_angle)) % 360
    return geographic_angle


def load_mat_file(file_path):
    """
    Load a MATLAB .mat file and convert numeric variables to NumPy arrays.

    Args:
        file_path (str): The path to the .mat file.

    Returns:
        variables (dict): A dictionary containing variable names as keys and
        their corresponding values.
    """
    # Load the .mat file
    mat_contents = scipy.io.loadmat(file_path)

    # Create a dictionary to store the variables
    variables = {}

    # Iterate through the loaded variables
    for var_name in mat_contents:
        if isinstance(mat_contents[var_name], (int, float, np.ndarray)):
            # If the variable is numeric (int, float, or ndarray), convert
            # to NumPy array
            variables[var_name] = np.array(mat_contents[var_name])
        elif isinstance(mat_contents[var_name], str):
            # If the variable is a string, keep it as is
            variables[var_name] = mat_contents[var_name]

    return variables


def cartesian_to_polar(x, y):
    """Convert Cartesian coordinates to polar coordinates

    Parameters
    ----------
    x : float
        The x-coordinate in Cartesian coordinates.
    y : float
        The y-coordinate in Cartesian coordinates.

    Returns
    -------
    Tuple of (theta, rho)
        theta : float
            The angle in radians in polar coordinates.
        rho : float
            The radius in polar coordinates.

    """
    z = x + 1j * y
    return (np.angle(z), np.abs(z))


def polar_to_cartesian(theta, rho, isImage=False):
    """
    Convert polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    theta : float
        The angle in radians (polar coordinate).
    rho : float
        The radius (polar coordinate).
    isImage : bool, optional
        If True, adjusts the y-axis to match image coordinate systems
        (where the origin is in the upper-left and y increases downward).
        Default is False.

    Returns
    -------
    tuple of float
        The Cartesian coordinates (x, y).
    """
    x = rho * np.cos(theta)
    if isImage:
        y = -rho * np.sin(theta)
    else:
        y = rho * np.sin(theta)
    return (x, y)


def units_conversion(units_id="Metric"):
    """Computes the survey_units conversion from SI survey_units used internally to the
    desired display survey_units.

    Parameters
    ----------
    units_id: str
        String variable identifying survey_units (English, Metric) Metric is the default.

    Returns
    -------
    survey_units: dict
        dictionary of unit conversion and labels
    """

    if units_id == "Metric":
        units = {
            "L": 1,
            "Q": 1,
            "A": 1,
            "V": 1,
            "label_L": "(m)",
            "label_Q": "(m³/s)",
            "label_A": "(m²)",
            "label_V": "(m/s)",
            "ID": "SI",
        }

    else:
        units = {
            "L": 1.0 / 0.3048,
            "Q": (1.0 / 0.3048) ** 3,
            "A": (1.0 / 0.3048) ** 2,
            "V": 1.0 / 0.3048,
            "label_L": "(ft)",
            "label_Q": "(ft³/s)",
            "label_A": "(ft²)",
            "label_V": "(ft/s)",
            "ID": "English",
        }

    return units


def compute_vectors_with_projections(X, Y, U, V):
    """Compute vectors and their projections onto the normal unit vector.

    Parameters:
    -----------
    X : numpy.ndarray
        X-coordinates of the vectors.
    Y : numpy.ndarray
        Y-coordinates of the vectors.
    U : numpy.ndarray
        U-components of the vectors.
    V : numpy.ndarray
        V-components of the vectors.

    Returns:
    --------
    vectors : numpy.ndarray
        Array containing original vectors with columns [X, Y, U, V].
    norm_vectors : numpy.ndarray
        Array containing vectors with rotated components projected onto the
        normal unit vector.
    normal_unit_vector : numpy.ndarray
        Unit vector in the normal direction.
    scalar_projections : numpy.ndarray
        Array containing scalar projections of vectors onto the normal unit
        vector.
    tagline_dir_geog_deg : numpy.ndarray
        Array containing the direction of the tagline in geographic degrees
        tiled to be of the same size as scalar_projections.
    mean_flow_dir_geog_deg : numpy.ndarray
        Array containing the mean flow direction in geographic degrees
        tiled to be of the same size as scalar_projections.

    Notes:
    ------
    - The function computes the differences, distance between mean
      cross-section endpoints, and angle of the mean cross-section.
    - It finds the unit vector in the normal direction and compares against
      the mean flow direction.
    - Vectors are projected onto the normal unit vector and rotated
      accordingly.
    - Returns the original vectors, vectors with rotated components, unit
      vector in the normal direction, and scalar projections of vectors
      onto the normal unit vector.

    """
    # Compute the differences (slope vector)
    dx = X[-1] - X[0]  # Difference in X
    dy = Y[-1] - Y[0]  # Difference in Y

    # Adjust dy for image coordinate system
    dy = -dy  # Negate dy to map to Cartesian coordinates

    # Compute the distance between the mean cross-section endpoints
    dl = np.sqrt(dx**2 + dy**2)

    # Compute the angle of the mean cross-section
    theta_arithmetic_rad = np.arctan2(dy, dx)
    tagline_dir_ari_deg = np.degrees(theta_arithmetic_rad)
    tagline_dir_geog_deg = arithmetic_to_geographic(tagline_dir_ari_deg)

    # Compute the mean flow direction
    a, b = 1, 1  # Not used anymore, consider removing
    mean_flow_dir_ari_rad = np.arctan2(np.nanmean(a * V), np.nanmean(b * U))
    mean_flow_dir_ari_deg = np.degrees(mean_flow_dir_ari_rad)
    mean_flow_dir_geog_deg = arithmetic_to_geographic(mean_flow_dir_ari_deg)

    # Compute the direction normal to the cross-section
    normal_direction_geog_deg = tagline_dir_geog_deg - 90
    normal_direction_geog_deg %= 360
    if np.abs(normal_direction_geog_deg - mean_flow_dir_geog_deg) > 180:
        normal_direction_geog_deg -= 180


    # Find the unit vector in the normal direction
    # THIS IS ARITHMETIC ANGLE
    rotated_angle_ari_deg = geographic_to_arithmetic(normal_direction_geog_deg)
    normal_unit_vector = np.array(
        [
            np.cos(np.radians(rotated_angle_ari_deg)),
            np.sin(np.radians(rotated_angle_ari_deg)),
        ]
    )

    # Project the vectors onto the normal unit vector
    # THESE ARE ARITHMETIC ANGLES
    # TODO: sort out WHERE I need to apply image coord conversions. Is it
    #  here, or is it best done in the plotting call?
    vectors = np.column_stack([X, Y, U, V])
    scalar_projections = np.zeros_like(X)
    norm_vectors = vectors.copy()
    for i, vector in enumerate(norm_vectors):
        # Extract components of the vector
        x, y, u, v = vector

        # Check if u or v is NaN, and skip computation if either
        # is NaN
        if np.isnan(u) or np.isnan(v):
            scalar_projections[i] = np.nan
            continue
        else:
            # Compute the scalar projection of the vector onto the
            # normal unit vector
            scalar_projections[i] = np.dot([u, v], normal_unit_vector)

            # Compute the rotated vector components using scalar
            # projection
            rotated_u, rotated_v = scalar_projections[i] * normal_unit_vector

            # Update the original vector with rotated components
            norm_vectors[i, 2] = rotated_u  # Image coords
            norm_vectors[i, 3] = rotated_v

    tagline_dir_geog_deg = np.tile(tagline_dir_geog_deg, scalar_projections.shape)
    mean_flow_dir_geog_deg = np.tile(mean_flow_dir_geog_deg, scalar_projections.shape)

    return (
        convert_to_image_frame(vectors),
        convert_to_image_frame(norm_vectors),
        normal_unit_vector,
        scalar_projections,
        tagline_dir_geog_deg,
        mean_flow_dir_geog_deg,
    )


def convert_to_image_frame(vectors):
    """
    Converts velocity vectors from an arithmetic coordinate frame
    to an image coordinate frame.

    Parameters:
    vectors (numpy.ndarray): Input array with columns x, y, u, v.

    Returns:
    numpy.ndarray: Output array with adjusted u, v components.
    """
    # Make a copy to avoid modifying the original array
    vectors_image = vectors.copy()

    # Adjust the v component (invert its sign to account for image y-axis)
    # Only modify rows where u and v are not NaN
    valid_rows = ~np.isnan(vectors[:, 2]) & ~np.isnan(vectors[:, 3])
    vectors_image[valid_rows, 3] = -vectors[valid_rows, 3]

    return vectors_image


def load_csv_with_numpy(csv_file_path):
    """
    Load data and headers from a CSV file using NumPy.

    Parameters
    ----------
    csv_file_path : str
        The path to the CSV file to be loaded.

    Returns
    -------
    headers : list of str
        A list containing the headers from the CSV file.
    data : np.ndarray
        A NumPy array containing the data from the CSV file. Each row represents
        a data point, and each column represents a variable.

    Notes
    -----
    The function expects the CSV file to have a header row which will be read and returned.
    The data in the CSV file should be numerical, and the delimiter used is a comma.

    Examples
    --------
    Load data and headers from a CSV file:

    >>> csv_file_path = 'path/to/your/csvfile.csv'
    >>> headers, data = load_csv_with_numpy(csv_file_path)
    >>> print(headers)
    ['X (pixel)', 'Y (pixel', 'U (m/s)', 'V (m/s)', 'Magnitude (m/s)',
     'Normal Magnitude (m/s)', 'Vector Direction (deg)', 'Tagline Direction (deg)']
    >>> print(data)
    [[1.0, 2.0, 0.5, 0.3, 0.6, 0.4, 45.0, 30.0],
     [3.0, 4.0, 0.7, 0.2, 0.8, 0.5, 60.0, 45.0],
     ...]

    """
    # Read the headers
    with open(csv_file_path, "r") as file:
        headers = file.readline().strip().split(",")

    # Read the data
    data = np.loadtxt(csv_file_path, delimiter=",", skiprows=1)

    return headers, data

def difference_between_angles_radians(a, b):
    """Returns the smallest difference between two angles in radians."""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def component_in_direction(magnitudes, directions_deg, tagline_angle_deg):
    """
    Calculate the component of vectors acting in the direction 90 degrees from
    the tagline angle.

    Parameters:
    magnitudes_mps (np.array): Array of vector magnitudes_mps.
    directions_deg (np.array): Array of vector directions in degrees.
    tagline_angle_deg (float): The tagline angle in degrees.

    Returns:
    np.array: Array of components of the vectors in the direction 90 degrees
        from the tagline angle.
    """
    # Convert angles from degrees to radians
    directions_rad = np.deg2rad(directions_deg)
    tagline_angle_rad = np.deg2rad(tagline_angle_deg)

    # Calculate both perpendicular direction candidates
    perp1 = tagline_angle_rad + np.pi / 2  # 90° counterclockwise
    perp2 = tagline_angle_rad - np.pi / 2  # 90° clockwise

    # Choose the perpendicular direction closer to directions_rad
    if np.abs(difference_between_angles_radians(perp1, directions_rad)) < np.abs(
            difference_between_angles_radians(perp2, directions_rad)):
        perpendicular_direction_rad = perp1
    else:
        perpendicular_direction_rad = perp2

    # Calculate the components of the vectors in the desired direction
    components = magnitudes * np.cos(directions_rad -
                                     perpendicular_direction_rad)
    return components


def get_column_contents(table: QTableWidget, column_index: int) -> dict:
    """
    Extracts the contents of a specified column from a QTableWidget into a dictionary.

    Args:
        table (QTableWidget): The table widget to extract data from.
        column_index (int): The index of the column to extract.

    Returns:
        dict: A dictionary with row indices as keys and cell contents (as strings) as values.

    Raises:
        IndexError: If the column index is out of range or the table has no columns.
    """
    if table.columnCount() == 0:
        raise IndexError("Table has no columns.")
    if column_index < 0 or column_index >= table.columnCount():
        raise IndexError(f"Column index {column_index} is out of range.")

    column_contents = {}
    for row in range(table.rowCount()):
        item = table.item(row, column_index)
        column_contents[row] = item.text() if item is not None else ""
    return column_contents


def set_column_contents(table: QTableWidget, column_index: int, data: dict):
    """
    Sets the contents of a specified column in a QTableWidget using data from a dictionary.

    Args:
        table (QTableWidget): The table widget to populate.
        column_index (int): The index of the column to populate.
        data (dict): A dictionary where keys are row indices and values are strings for the column content.

    Raises:
        ValueError: If the number of rows in the dictionary does not match the table's row count.
        IndexError: If the column index is out of range.
    """
    if table.columnCount() == 0:
        raise IndexError("Table has no columns.")
    if column_index < 0 or column_index >= table.columnCount():
        raise IndexError(f"Column index {column_index} is out of range.")
    if len(data) != table.rowCount():
        raise ValueError(
            f"Row count mismatch: table has {table.rowCount()} rows, but data has {len(data)} items."
        )

    for row, value in data.items():
        try:
            row_index = int(row)  # Convert row to an integer
        except ValueError:
            raise ValueError(f"Row key {row} cannot be converted to an integer.")

        if row_index < 0 or row_index >= table.rowCount():
            raise IndexError(f"Row index {row_index} is out of range.")

        table.setItem(row_index, column_index, QTableWidgetItem(value))


def calculate_uv_components(magnitudes, directions):
    """
    Calculate U and V components based on the geographic direction and magnitude.

    Parameters:
        mfd_geog (float): Geographic direction in degrees (0-360).
        magnitudes (float): Magnitude of the vector.
        directions (float): Direction in radians in the GEO coordinate
        system.

    Returns:
        tuple: U and V components.
    """
    directions_rad_ari = np.deg2rad(
        geographic_to_arithmetic(directions)
    )

    # Calculate U and V components
    U = magnitudes * np.cos(directions_rad_ari)
    V = magnitudes * np.sin(directions_rad_ari)

    return U, V


def calculate_endpoint(phi_origin, num_pixels, x_start, y_start):
    """Calculate the endpoint of a line given X,Y start, angle, and number
     of pixels along the line

    Args:
        phi_origin (float): the line angle
        num_pixels (int): how many pixels long the line is
        x_start (float): the X pixel coordinate of the start of the line
        y_start (float): the Y pixel coordinate of the start of the line

    Returns:
        ndarray: the X,Y endpoint of the line
    """

    # Convert angle to arithmetic angle
    arithmetic_angle = geographic_to_arithmetic(phi_origin)

    # Convert angle from degrees to radians
    angle_rad = np.deg2rad(arithmetic_angle)

    # Calculate endpoint coordinates
    x_end = x_start + num_pixels * np.cos(angle_rad)
    y_end = y_start + num_pixels * -np.sin(angle_rad)

    return np.array([[x_start, y_start], [x_end, y_end]])


def dict_arrays_to_list(dictionary):
    """Convert dict arrays to a list

    This function is handy for preparing a complex dict for JSON serialization

    Args:
        dictionary (dict): the dict to convert

    Returns:
        list: a list
    """
    for k, v in zip(dictionary.keys(), dictionary.values()):
        if type(dictionary[k]) == np.ndarray:
            dictionary[k] = np.array(v).tolist()
    return dictionary

