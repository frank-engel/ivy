"""IVy module containing the STIV functions"""

import logging
import time
from typing import Tuple, Union, Any

import numpy as np
from PIL import Image
from PyQt5.QtCore import pyqtSignal, QObject
from numpy import (
    ndarray,
    dtype,
    signedinteger,
    unsignedinteger,
    floating,
    timedelta64,
)
from scipy.interpolate import RegularGridInterpolator, interp2d
from scipy.ndimage import zoom
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve
from scipy.stats import zscore
from tqdm import tqdm

from image_velocimetry_tools.common_functions import (
    geographic_to_arithmetic,
    arithmetic_to_geographic,
    cartesian_to_polar,
    polar_to_cartesian,
)
from image_velocimetry_tools.image_processing_tools import (
    create_grayscale_image_stack,
    create_image_interpolator,
)

# Globals
global_theta_max = None


class ProgressEmitter(QObject):
    # Create a signal for progress
    progress_signal = pyqtSignal(int)


def bresenham_line(x1, y1, x2, y2):
    """
    Generate the coordinates of a line using Bresenham's algorithm.

    This function generates the integer coordinates of a line between two points (x1, y1) and (x2, y2)
    using Bresenham's line algorithm.

    Parameters:
        x1 (int): The x-coordinate of the first point.
        y1 (int): The y-coordinate of the first point.
        x2 (int): The x-coordinate of the second point.
        y2 (int): The y-coordinate of the second point.

    Returns:
        List[Tuple[int, int]]: A list of integer coordinates (x, y) along the line.

    """
    line_coordinates = []

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    steep = abs(dy) > abs(dx)

    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    error = 0
    derror = dy / dx if dx != 0 else 0

    y = y1

    if y1 < y2:
        ystep = 1
    else:
        ystep = -1

    for x in range(x1, x2 + 1):
        if steep:
            line_coordinates.append((y, x))
        else:
            line_coordinates.append((x, y))

        error += derror
        if error >= 0.5:
            y += ystep
            error -= 1.0

    if swapped:
        line_coordinates.reverse()

    return line_coordinates


def extract_pixels_along_line(image_paths, point1, point2):
    """
    Extract the pixels that touch a line defined by two points in each image.

    This function takes a list of image file paths and extracts the pixels that intersect
    or touch the line defined by point1 and point2 for each image.

    Parameters:
        image_paths (List[str]): A list of file paths to the input images.
        point1 (np.ndarray): The (x, y) coordinates of the first point defining the line.
        point2 (np.ndarray): The (x, y) coordinates of the second point defining the line.

    Returns:
        np.ndarray: A NumPy array where each row represents the RGB values of the pixels that touch
        the line for each image. The shape of the array is (num_images, num_pixels, 3), where
        num_images is the number of input images, num_pixels is the number of pixels that touch
        the line, and 3 represents the RGB channels.

    """
    line_pixels = []

    for path in image_paths:
        # Open the image
        image = Image.open(path)

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Extract the coordinates of the line
        x1, y1 = map(int, point1)
        x2, y2 = map(int, point2)

        # Traverse along the line using Bresenham's algorithm
        line_coordinates = bresenham_line(x1, y1, x2, y2)

        # Extract the pixels that touch the line
        line_pixels_image = []
        for coord in line_coordinates:
            x, y = coord
            line_pixels_image.append(image_array[y, x])

        line_pixels.append(line_pixels_image)

    # Convert the list of lists to a NumPy array
    line_pixels_array = np.array(line_pixels)

    return line_pixels_array


def normalize_image_array(image_array, target_size):
    """Normalize the input image

    Args:
        image_array (ndarray): the image
        target_size (ndarray): the target normalized image size

    Returns:
        tuple: tuple containing the normalized array, minimum value, and range value
    """
    # Compute the normalization factors
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    range_val = max_val - min_val

    # Normalize the image array
    normalized_array = (image_array - min_val) / range_val

    # Resize the normalized array to the target size
    scale_factor = target_size / max(normalized_array.shape[:2])
    new_size = tuple(
        int(round(dim * scale_factor)) for dim in normalized_array.shape[:2]
    )
    normalized_array = zoom(
        normalized_array, (scale_factor, scale_factor, 1), order=1
    )

    # Create a square array of the target size
    square_array = np.zeros((target_size, target_size, 3))
    x_offset = (target_size - new_size[1]) // 2
    y_offset = (target_size - new_size[0]) // 2
    square_array[
        y_offset : y_offset + new_size[0], x_offset : x_offset + new_size[1]
    ] = normalized_array

    # Return the square array and normalization factors
    return square_array, min_val, range_val


def apply_standardization_filter(space_time_image):
    """
    Apply the standardization (STD) filter to the space-time image.

    Parameters:
        space_time_image (np.ndarray): The space-time image obtained from `extract_pixels_along_line`.

    Returns:
        np.ndarray: The filtered space-time image.
    """
    averaged_space_time_image = np.mean(
        space_time_image, axis=-1, keepdims=True
    )

    # Calculate the mean and standard deviation along the time axis
    mean_t = np.mean(averaged_space_time_image, axis=0)
    std_t = np.std(averaged_space_time_image, axis=0)

    # Apply the standardization filter
    filtered_image = (averaged_space_time_image - mean_t) / std_t

    # Scale the filtered results to the range 0-255
    filtered_image = np.clip(filtered_image * 255.0, 0, 255).astype(np.uint8)

    return filtered_image


def compute_angle_to_yaxis(point1, point2):
    """
    Calculate the angle between the y-axis and a line segment formed by two points.

    Parameters:
        point1 (tuple): The coordinates of the first point (x1, y1).
        point2 (tuple): The coordinates of the second point (x2, y2).

    Returns:
        float: The angle in degrees between the y-axis and the line segment formed by the two points.

    Raises:
        ValueError: If either point1 or point2 is not a tuple with two elements.
        ValueError: If the line segment is vertical (x1 = x2), making the angle calculation impossible.

    Examples:
        >>> compute_angle_to_yaxis((0, 0), (3, 4))
        36.86989764584401

    Note:
        This function assumes the points are in 2D space. The y-axis is considered to be the vertical axis.

    """
    if len(point1) != 2 or len(point2) != 2:
        raise ValueError(
            "Invalid input. Points should be tuples with two elements."
        )

    x1, y1 = point1
    x2, y2 = point2

    if np.isclose(x1, x2):
        raise ValueError("Vertical line segment. Angle cannot be calculated.")

    yy1 = np.max([y1, y2])
    yy2 = np.min([y1, y2])

    angle_radians = np.arccos(
        (yy1 - yy2) / np.sqrt((x1 - x2) ** 2 + (yy1 - yy2) ** 2)
    )
    return np.degrees(angle_radians)


def compute_velocity_along_search_line(pixel_gsd, sti_tscale, phi):
    """
    Compute velocity along a search line in a space-time image (STI).

    Parameters:
        pixel_gsd (float): Pixel ground scale in units of distance per pixel.
        sti_tscale (float): Seconds per pixel along the STI's time axis.
        phi (float): Angle computed from the STI at the specified search line.

    Returns:
        float: Velocity along the search line.

    Notes:
        - The angle phi should be specified in degrees and will be internally converted to radians.
        - The computed velocity represents the magnitude of the velocity along the search line.

    """
    phi = np.radians(phi)
    velocity = pixel_gsd / sti_tscale * np.tan(phi)
    return velocity


def extract_space_time_image(
    image_interpolator,
    x_origin,
    y_origin,
    search_angle_arithmetic,
    num_pixels,
    num_frames,
    normalize='none',
):
    """
    Extract space-time image along a search line.

    This function extracts pixel values along a search line from a sequence of
    images to create a space-time image (STI). The search line is defined by an
    origin point and an angle, and the STI shows how pixel intensity varies
    along the spatial dimension (search line) and temporal dimension (frames).

    Parameters
    ----------
    image_interpolator : RegularGridInterpolator
        Interpolator for the image stack, created with create_image_interpolator()
    x_origin : float
        X coordinate of search line origin in pixel coordinates
    y_origin : float
        Y coordinate of search line origin in pixel coordinates
    search_angle_arithmetic : float
        Search line angle in arithmetic degrees (0° = East, 90° = North,
        counter-clockwise from positive x-axis)
    num_pixels : int
        Desired number of pixels along search line
    num_frames : int
        Number of frames in the image sequence
    normalize : str, optional
        Normalization method. Options:
        - 'none': Return raw grayscale uint8 values (default)
        - 'zscore': Apply z-score normalization (Fujita's standardization filter)

    Returns
    -------
    sti : ndarray
        Space-time image with shape (num_pixels_actual, num_frames) where
        num_pixels_actual is num_pixels or num_pixels-1 (adjusted to be even).
        - If normalize='none': dtype is uint8 (grayscale values 0-255)
        - If normalize='zscore': dtype is float64 (normalized values)

    Notes
    -----
    The function ensures an even number of points along the search line for
    proper autocorrelation function computation. If the calculated line length
    results in an odd number of points, the last point is dropped.

    Examples
    --------
    >>> image_stack = create_grayscale_image_stack(image_paths)
    >>> interpolator = create_image_interpolator(image_stack)
    >>> sti = extract_space_time_image(
    ...     interpolator, 100.0, 200.0, 45.0, 50, 25, normalize='zscore'
    ... )

    See Also
    --------
    create_image_interpolator : Creates the interpolator for an image stack
    compute_velocity_from_sti : Computes velocity from an STI
    two_dimensional_stiv_exhaustive : Main STIV processing function

    References
    ----------
    Fujita, I., Notoya, Y., Tani, K., & Tateguchi, S. (2019). Efficient and
    accurate estimation of water surface velocity in STIV. Environmental Fluid
    Mechanics, 19(5), 1363-1378.
    """
    # Define search line endpoints based on angle
    x1, y1 = polar_to_cartesian(
        np.deg2rad(search_angle_arithmetic), num_pixels, isImage=True
    )
    x_end_points = np.array([x_origin, x_origin + x1])
    y_end_points = np.array([y_origin, y_origin + y1])

    # Get length of this line in pixel units
    dist = np.hypot(
        np.abs(np.diff(y_end_points, axis=0)),
        np.abs(np.diff(x_end_points, axis=0)),
    ).astype(int)

    # Create regularly spaced points along this line
    xi = np.linspace(x_end_points[0], x_end_points[1], dist.item() + 1)
    yi = np.linspace(y_end_points[0], y_end_points[1], dist.item() + 1)

    # Ensure even number of points for proper ACF computation
    if len(xi) % 2 != 0:
        xi = xi[:-1]
        yi = yi[:-1]

    # Extract pixel values along search line for each frame to build STI
    # This is a vectorized approach that extracts all pixels at once
    sti = np.zeros([len(xi), num_frames])
    sti_w, sti_h = sti.shape
    frame_indices = np.arange(num_frames)
    frame_indices_repeat = np.tile(frame_indices, xi.shape[0])
    yi_repeat = np.repeat(yi, num_frames)
    xi_repeat = np.repeat(xi, num_frames)

    coordinates = np.column_stack(
        (yi_repeat, xi_repeat, frame_indices_repeat)
    )
    sti = image_interpolator(coordinates).reshape(sti_w, sti_h)

    # Apply normalization if requested
    if normalize == 'zscore':
        # Apply z-score normalization (Fujita's standardization filter)
        # This subtracts the time average and divides by standard deviation
        normalized_sti = zscore(sti, axis=1, ddof=1)
        return normalized_sti
    elif normalize == 'none':
        # Return as uint8 grayscale
        return sti.astype(np.uint8)
    else:
        raise ValueError(
            f"Unknown normalization method: {normalize}. "
            f"Use 'none' or 'zscore'."
        )


def compute_velocity_from_sti(
    sti,
    pixel_gsd,
    d_t,
    d_rho=0.5,
    d_theta=0.5,
):
    """
    Compute velocity from a space-time image using autocorrelation method.

    This function implements the velocity extraction algorithm from
    Fujita et al. (2019) and Han et al. (2021), using the autocorrelation
    function (ACF) of the STI to determine the predominant streak angle
    and corresponding velocity magnitude.

    Parameters
    ----------
    sti : ndarray
        Space-time image with shape (num_pixels, num_frames).
        Can be normalized (z-score) or raw grayscale. If not normalized,
        will be automatically normalized before processing.
    pixel_gsd : float
        Pixel ground sampling distance (spatial resolution) in physical units
        (e.g., meters per pixel)
    d_t : float
        Time interval between frames in seconds
    d_rho : float, optional
        Radial resolution for ACF integration in polar coordinates.
        Default is 0.5.
    d_theta : float, optional
        Angular resolution for ACF integration in degrees.
        Default is 0.5.

    Returns
    -------
    velocity : float
        Velocity magnitude computed from the STI in physical units per second
    theta_max : float
        Predominant streak angle in degrees (0-180)
    p_value : float
        Peak value of the integrated ACF (quality/strength metric)

    Notes
    -----
    The algorithm performs the following steps:
    1. Normalizes the STI using z-score if not already normalized
    2. Computes the 2D autocorrelation function (ACF) using FFT convolution
    3. Crops to central portion of ACF around zero lag
    4. Converts ACF to polar coordinates centered at zero lag
    5. Integrates ACF along radial rays at different angles (0-180°)
    6. Finds the angle (theta_max) with maximum integrated ACF
    7. Computes velocity from theta_max using tan(theta) * (gsd / dt)

    The p_value represents the strength of the detected signal and can be
    used as a quality metric for the velocity estimate.

    References
    ----------
    Fujita, I., Notoya, Y., Tani, K., & Tateguchi, S. (2019). Efficient and
    accurate estimation of water surface velocity in STIV. Environmental Fluid
    Mechanics, 19(5), 1363-1378. https://doi.org/10.1007/s10652-018-9651-3

    Han, X., Chen, K., Zhong, Q., Chen, Q., Wang, F., & Li, D. (2021).
    Two-Dimensional Space-Time Image Velocimetry for Surface Flow Field of
    Mountain Rivers Based on UAV Video. Frontiers in Earth Science, 9, 686636.
    https://doi.org/10.3389/feart.2021.686636

    Examples
    --------
    >>> sti = extract_space_time_image(interpolator, 100, 200, 45, 50, 25)
    >>> velocity, theta, p_val = compute_velocity_from_sti(
    ...     sti, pixel_gsd=0.05, d_t=0.1
    ... )
    """
    num_pixels, num_frames = sti.shape

    # Normalize the STI if not already normalized
    # Check if values are in typical z-score range vs uint8 range
    if sti.dtype == np.uint8 or np.max(np.abs(sti)) > 10:
        normalized_sti = zscore(sti.astype(float), axis=1, ddof=1)
    else:
        normalized_sti = sti

    # Compute autocorrelation function using FFT convolution
    reversed_normalized_sti = np.flip(normalized_sti, axis=(0, 1))
    R = fftconvolve(normalized_sti, reversed_normalized_sti, mode="full")
    R = R / np.max(R)

    # Set up lag coordinates
    pixel_lag = np.arange(-num_pixels, num_pixels + 1)
    frame_lag = np.arange(-num_frames, num_frames + 1)

    # Crop to central portion of ACF
    # The center is at (num_pixels, num_frames) with lag (0,0) and ACF value 1
    pixel_lag_sub = pixel_lag[num_pixels // 2 : -num_pixels // 2]
    frame_lag_sub = frame_lag[num_frames // 2 : -num_frames // 2]

    R_sub = R[
        num_pixels - num_pixels // 2 : num_pixels + num_pixels // 2 + 1,
        num_frames - num_frames // 2 : num_frames + num_frames // 2 + 1,
    ]

    # Prepare for integration in polar coordinates
    rho_max = np.min([np.max(pixel_lag_sub), np.max(frame_lag_sub)])
    rho = np.arange(0, rho_max + d_rho, d_rho)
    theta = np.arange(0, 180 + d_theta, d_theta)

    # Get indices for zero lag
    x_lag_0 = num_pixels // 2 - 1
    t_lag_0 = num_frames // 2 - 1

    # Radiate rays on the ACF from origin at lag (0,0)
    theta_rad = np.deg2rad(theta)
    theta_expanded = np.tile(theta_rad, (len(rho), 1)).T
    t_ray, x_ray = polar_to_cartesian(theta_expanded, rho)

    # Set up interpolator for extracting ACF values along rays
    acf_interpolator = RegularGridInterpolator(
        (np.arange(R_sub.shape[0]), np.arange(R_sub.shape[1])),
        R_sub,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    # Interpolate ACF values along all rays (vectorized)
    x_coords = x_ray + x_lag_0
    t_coords = t_ray + t_lag_0
    coordinates = np.column_stack((x_coords.flatten(), t_coords.flatten()))
    acfRay = acf_interpolator(coordinates).reshape(len(theta), -1)

    # Integrate along each ray using trapezoidal rule
    F_theta = np.trapz(acfRay, axis=1) * d_rho

    # Find peak and corresponding angle
    p_value = np.max(F_theta)
    i_max = np.argmax(F_theta)
    theta_max = theta[i_max]

    # Calculate velocity using equation 16 of Fujita et al. (2007)
    velocity = np.tan(np.deg2rad(theta_max)) * (pixel_gsd / d_t)

    return velocity, theta_max, p_value


def two_dimensional_stiv_exhaustive(
    x_origin: np.ndarray,
    y_origin: np.ndarray,
    image_stack: np.ndarray,
    num_pixels: int,
    phi_origin: int,
    d_phi: float,
    phi_range: int,
    pixel_gsd: float,
    d_t: float,
    d_rho: Union[float, None] = 0.5,
    d_theta: Union[float, None] = 0.5,
    sigma: Union[float, None] = 0.5,
    max_vel_threshold: Union[float, None] = 10.0,
    # map_file_path: Union[str, None] = None,
    progress_signal: Union[pyqtSignal, None] = None,
):
    """Perform 2D Space-Time Image Velocimetry (STIV) using exhaustive orientation search

    Perform 2D Space-Time Image Velocimetry (STIV) for a set of search line
    origins specified as vectors of input coordinates by looping over these points
    (i.e., grid nodes) to produce a velocity field. Takes as input the coordinates
    of the search line origins, the image sequence, the number of pixels to be
    used along the search line, parameters describing the range of search line
    orientations to be considered in inferring flow directions, and the pixel size
    and frame interval to scale the velocities. The approach is based on using the
    autocorrelation function of the ACF for a given STI to estimate the velocity
    magnitude and then considering a range of search line orientations to
    establish the flow direction, based on work by Han et al. (2021) and Fujita et
    al. (2019).


    Parameters
    ----------
    x_origin : np.ndarray
        x coordinates of search line origins in pixel coordinates
    y_origin : np.ndarray
        y coordinates of search line origins in pixel coordinates
    image_stack : np.ndarray
        An numpy array of images as built with create_grayscale_image_stack
    num_pixels : int
        Number of pixels to extract along search lines
    phi_origin : int
        Initial guess used for search line orientations, specified in
        geographic angle in degrees
    d_phi : float
        Angular resolution of search line angles to consider (degrees)
    phi_range : int
        Range of search line angles to consider (degrees) relative to
        initial guesses; use 90 to consider the full forward hemisphere
    pixel_gsd : float
        Pixel size of the image sequence
    d_t : float
        Frame interval of the image sequence in seconds
    d_rho : float
        Vector of rho (magnitude) values onto which the ACF will be
        interpolated for a given streak angle theta. Default is 0.5.
    d_theta : float
        Increment for the streak angles to be used to integrate the ACF over
        rho in polar coordinates. Default is 0.5.
    sigma : float
        Sigma used in a Gaussian blur applied to the image stack. This may
        provide better results if the input images have a dynamic texture (
        e.g. surface waves). If less than 0.01, no blur will be applied.
        Default value is 0.5.
    max_vel_threshold : float
        A threshold of the maximum expected velocity magnitude. Resulting
        absolute velocity magnitudes_mps larger than this threshold are replaced
        by NaNs. The default is 10.
    map_file_path (str, optional):
        If specified, the path to a memory-mapped file to store
        the image stack. The default is to not use a map (=None)
    progress_signal (pyqtSignal, optional):
        If provided, this will enable emitting a signal of [int, int] out of
        the function to track progress. The default is to not use progress
        signal (=None)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the magnitude, directions, STIs and the thetas
        for the i_max node as an array. Effectively, the returned tuple
        gives the results of the STIV analysis corresponding to the peak
        magnitude and direction found given the input search parameters.

        Magnitudes are returned as an absolute value.
        Directions are returned in geographic degrees.

    Notes
    -----
    Based on work by Han et al. (2021) and Fujita et al. (2019).
    Han, X., Chen, K., Zhong, Q., Chen, Q., Wang, F., & Li, D. (2021).
    Two-Dimensional Space-Time Image Velocimetry for Surface Flow Field of
    Mountain Rivers Based on UAV Video. Frontiers in Earth Science, 9, 686636.
    https://doi.org/10.3389/feart.2021.686636
    Fujita, I., Notoya, Y., Tani, K., & Tateguchi, S. (2019). Efficient and
    accurate estimation of water surface velocity in STIV. Environmental Fluid
    Mechanics, 19(5), 1363â€“1378. https://doi.org/10.1007/s10652-018-9651-3

    Author
    ------
    Dr. Carl J. Legleiter, cjl@usgs.gov
    Observing Systems Division, Hydrologic Remote Sensing Branch
    United States Geological Survey

    Ported to Python and modified by:
    Dr. Frank L. Engel, fengel@usgs.gov
    Observing Systems Division, Hydrologic Remote Sensing Branch
    United States Geological Survey


    """
    # Initiate a progress tracker
    progress_emitter = ProgressEmitter()
    if progress_signal is not None:
        # If a signal is provided, connect the emitter's signal to it
        progress_emitter.progress_signal.connect(progress_signal.emit)

    # Provide more progress to the user
    if progress_signal is not None:
        # The image stack process takes time, and there is no signal. To provide
        # some user feedback, go ahead and emit 4% done
        progress_emitter.progress_signal.emit(int(5))

    # Set up a sequence of search line orientation angles phi, specified
    # relative to our initial search line orientation phi0geo, which we
    # first need to convert from a geographical angle to a mathematical angle.
    phi_origin_ari = geographic_to_arithmetic(phi_origin, signed180=False)

    # This angle will become the center of the +/-phiRange range relative to
    # the initial guess that we will consider as alternative search line
    # orientations, so subtract/add phiRange to get the lower/upper limit
    phi = np.arange(
        phi_origin_ari - phi_range, phi_origin_ari + phi_range + 1, d_phi
    )

    # Set up an array to hold values of the integral of the ACF, denoted by P
    # in Han et al. (2021)
    p = np.zeros(phi.shape)

    # Need to compute the time and space lags such that the center of the
    # ACF array is at a lag of (0,0), which has a correlation of 1 by
    # definition. The following should confirm this: R(nPix,nFrame). Get
    # lags in native units of pixels and frames
    pixel_lag = np.arange(-num_pixels, num_pixels + 1)
    num_frames = image_stack.shape[2]
    frame_lag = np.arange(-num_frames, num_frames + 1)

    # Crop to central portion of the ACF
    # We know that the center of the surface is at nPix,nFrame where the lag
    # vector is (0,0) and the ACF has a value of 1, so go out by half the
    # maximum lag in each direction from this center point
    pixel_lag_sub = pixel_lag[num_pixels // 2 : -num_pixels // 2]
    frame_lag_sub = frame_lag[num_frames // 2 : -num_frames // 2]

    # Convert to polar coordinates using native units of frames and pixels
    # Use a vectorized approach that doesn't need to double loop
    pixel_lag_expanded = pixel_lag_sub[:, np.newaxis]
    frame_lag_expanded = frame_lag_sub[np.newaxis, :]
    theta_grid, rho_grid = cartesian_to_polar(
        frame_lag_expanded, pixel_lag_expanded
    )

    # Prepare for numerical integration of the ACF over rho in polar
    # coordinates
    rho_max = np.min([np.max(pixel_lag_sub), np.max(frame_lag_sub)])

    # Create a vector of rho (magnitude) values we can use to interpolate the
    # ACF onto for a given angle theta.
    rho = np.arange(0, rho_max + d_rho, d_rho)

    # Set up the range of theta values for which we will be integrating over
    # rho
    theta = np.arange(0, 180 + d_theta, d_theta)

    # Get the index of lag (0, 0)
    x_lag_0 = num_pixels // 2 - 1
    t_lag_0 = num_frames // 2 - 1

    # Create the interpolator object
    image_interpolator = create_image_interpolator(image_stack, sigma=sigma)
    progress_emitter.progress_signal.emit(int(10))

    # Begin the outer loop
    magnitudes = np.zeros(x_origin.shape)
    directions = np.zeros(x_origin.shape)
    thetas = np.zeros(x_origin.shape)
    # stis = np.zeros([x_origin.shape[0],num_pixels, num_frames])
    stis = np.zeros([x_origin.shape[0], len(phi), num_pixels, num_frames])
    sti_at_i_max = np.zeros([x_origin.shape[0], num_pixels, num_frames])

    # Begin the inner loop
    for i_node in tqdm(range(len(x_origin)), desc="Processing node"):
        mag = np.empty(phi.shape)
        theta_max_i = np.empty(phi.shape)
        mag[:] = np.nan
        theta_max_i[:] = np.nan
        for i_phi in range(len(phi)):
            # Given the origin and current angle, define search line end points
            x1, y1 = polar_to_cartesian(
                np.deg2rad(phi[i_phi]), num_pixels, isImage=True
            )
            x_end_points = np.array([x_origin[i_node], x_origin[i_node] + x1])
            y_end_points = np.array([y_origin[i_node], y_origin[i_node] + y1])

            # Get length of this line in pixel units, which should be the
            # same as our nPix input, but recalculate here just to be safe
            dist = np.hypot(
                np.abs(np.diff(y_end_points, axis=0)),
                np.abs(np.diff(x_end_points, axis=0)),
            ).astype(int)

            # Create regularly spaced points along this line
            xi = np.linspace(x_end_points[0], x_end_points[1], dist.item() + 1)
            yi = np.linspace(y_end_points[0], y_end_points[1], dist.item() + 1)

            # To ensure that we can set up the STI, and hence the ACF,
            # such that the center of the ACF array at a lag of (0,0) has a
            # value of 1, we need to make the profile lines have an even
            # number of points, so if the length of xi and yi is not even,
            # drop the last point to make it even in length
            if len(xi) % 2 != 0:
                xi = xi[:-1]
                yi = yi[:-1]

            # Use the gridded interpolant for the image stack that we set up
            # outside the loop to extract pixel values along search line for
            # each frame to build the STI. This is a vectorized approach
            # that removes the extra loop.
            sti = np.zeros([len(xi), num_frames])
            sti_w, sti_h = sti.shape
            frame_indices = np.arange(num_frames)
            frame_indices_repeat = np.tile(frame_indices, xi.shape[0])
            yi = np.repeat(yi, num_frames)
            xi = np.repeat(xi, num_frames)

            coordinates = np.column_stack((yi, xi, frame_indices_repeat))
            sti = image_interpolator(coordinates).reshape(sti_w, sti_h)

            # Apply Fujita's standardization filter, basically subtracting
            # the time average (mean of each row) and then dividing by the
            # standard deviation of that row, where the rows are time series
            # for a fixed spatial position. Note that this is essentially a
            # z-score, so we can use a standard normalization approach for
            # this.
            normalized_sti = zscore(sti, axis=1, ddof=1)

            # Remap the STIs into a square array of length num_pixels. This
            # can be saved and plotted to verify STI accuracy.
            # stis[i_node,:,:] = remap_stiv_to_square_array(normalized_sti,
            #                                         num_pixels, num_frames)
            # stis[i_node,:,:] = normalized_sti
            # Store the normalized STI in the 4D stis array
            stis[i_node, i_phi, :, :] = normalized_sti

            # Calculate the ACF of the STI and
            reversed_normalized_sti = np.flip(normalized_sti, axis=(0, 1))
            R = fftconvolve(
                normalized_sti, reversed_normalized_sti, mode="full"
            )
            R = R / np.max(R)

            # Crop to central portion of the ACF We know that the center of
            # the surface is at nPix,nFrame where the lag vector is (0,
            # 0) and the ACF has a value of 1, so go out by half the maximum
            # lag in each direction from this center point
            R_sub = R[
                num_pixels
                - num_pixels // 2 : num_pixels
                + num_pixels // 2
                + 1,
                num_frames
                - num_frames // 2 : num_frames
                + num_frames // 2
                + 1,
            ]

            # Radiate rays on the ACF from the origin at lag (0,0)
            theta_rad = np.deg2rad(theta)
            theta_expanded = np.tile(theta_rad, (len(rho), 1)).T
            t_ray, x_ray = polar_to_cartesian(theta_expanded, rho)

            # Set up gridded interpolant for extracting ACF values along each
            # ray.
            acf_interpolator = RegularGridInterpolator(
                (np.arange(R_sub.shape[0]), np.arange(R_sub.shape[1])),
                R_sub,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )

            # We want all the rho values for each theta and then use those
            # as the integrand. This is a vectorized approach to do it.
            F_theta = np.zeros(len(theta))

            # Create arrays for all theta values
            x_coords = x_ray + x_lag_0
            t_coords = t_ray + t_lag_0

            # Stack the x and t coordinates along with the corresponding
            # theta values
            coordinates = np.column_stack(
                (x_coords.flatten(), t_coords.flatten())
            )

            # Interpolate the ACF values along all rays
            acfRay = acf_interpolator(coordinates).reshape(len(theta), -1)
            F_theta = np.trapz(acfRay, axis=1) * d_rho

            # Now get the peak value of the integrals and the angle with the
            # maximum autocorrelation
            p[i_phi] = np.max(F_theta)
            i_max = np.argmax(F_theta)
            theta_max = theta[i_max]

            # Now calculate velocity by bringing in the pixel size and frame
            # interval See equation 16 of Fujita et al. (2007)
            mag[i_phi] = np.tan(np.deg2rad(theta_max)) * (pixel_gsd / d_t)
            theta_max_i[i_phi] = theta_max

        # Get search line orientation angle for which radial integral of the
        # ACF in polar coordinates was maximized
        i_max = np.argmax(p)
        directions[i_node] = phi[i_max]
        magnitudes[i_node] = mag[i_max]
        thetas[i_node] = theta_max_i[i_max]
        sti_at_i_max[i_node] = stis[i_node, i_max, :, :]

        # If a signal is provided, emit progress information
        if progress_signal is not None:
            progress_percentage = int(((i_node / len(x_origin)) * 90) + 10)
            progress_emitter.progress_signal.emit(progress_percentage)

    # Apply the velocity threshold
    magnitudes[
        (magnitudes < -max_vel_threshold) | (magnitudes > max_vel_threshold)
    ] = np.nan

    # Magnitudes should only be returned as positive
    magnitudes = np.abs(magnitudes)

    # Directions should be returned as geographic angle (b/c phi was
    # supplied as geographic angle
    directions = arithmetic_to_geographic(directions)

    return magnitudes, directions, sti_at_i_max, thetas


def two_dimensional_stiv_optimized(
    x_origin: np.ndarray,
    y_origin: np.ndarray,
    image_stack: np.ndarray,
    num_pixels: int,
    phi_origin: np.ndarray,
    pixel_gsd: float,
    d_t: float,
    d_rho: Union[float, None] = 0.5,
    d_theta: Union[float, None] = 0.5,
    tolerance: Union[float, None] = 0.5,
    max_vel_threshold: Union[float, None] = 10.0,
    map_file_path: Union[str, None] = None,
    progress_signal: Union[any, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform 2D Space-Time Image Velocimetry (STIV) using optimized orientation search

    Perform 2D Space-Time Image Velocimetry (STIV) for a set of search line
    origins specified as vectors of input coordinates by looping over these points
    (i.e., grid nodes) to produce a velocity field. Takes as input the coordinates
    of the search line origins, the image sequence, the number of pixels to be
    used along the search line, parameters describing the range of search line
    orientations to be considered in inferring flow directions, and the pixel size
    and frame interval to scale the velocities. The approach is based on using the
    autocorrelation function of the ACF for a given STI to estimate the velocity
    magnitude and then considering a range of search line orientations to
    establish the flow direction, based on work by Han et al. (2021) and Fujita et
    al. (2019).


    Parameters
    ----------
    max_vel_threshold :
    tolerance :
    x_origin : np.ndarray
        x coordinates of search line origins in pixel coordinates
    y_origin : np.ndarray
        y coordinates of search line origins in pixel coordinates
    image_stack : np.ndarray
        An numpy array of images as built with create_grayscale_image_stack
    num_pixels : int
        Number of pixels to extract along search lines
    phi_origin : int
        Initial guess used for search line orientations, specified in
        geographic angle in degrees
    pixel_gsd : float
        Pixel size of the image sequence
    d_t : float
        Frame interval of the image sequence in seconds
    d_rho : float
        Vector of rho (magnitude) values onto which the ACF will be
        interpolated for a given streak angle theta. Default is 0.5.
    d_theta : float
        Increment for the streak angels to be used to integrate the ACF over
        rho in polar coordinates. Default is 0.5.
    max_vel_threshold : float
        A threshold of the maximum expected velocity magnitude. Resulting
        absolute velocity magnitudes_mps larger than this threshold are replaced
        by NaNs. The default is 10.
    tolerance : float
        The tolerance criterial for termination of the minimization function.
        The default is 0.5.
    map_file_path (str, optional):
        If specified, the path to a memory-mapped file to store
        the image stack. The default is to not use a map (=None)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the magnitudes_mps and directions for each node
        of the input array. Effectively, the returned tuple gives the results
        of the STIV analysis corresponding to the peak magnitude and
        direction found given the input search parameters.

        Magnitudes are returned as an absolute value.
        Directions are returned in geographic degrees.

    Notes
    -----
    Based on work by Han et al. (2021) and Fujita et al. (2019).
    Han, X., Chen, K., Zhong, Q., Chen, Q., Wang, F., & Li, D. (2021).
    Two-Dimensional Space-Time Image Velocimetry for Surface Flow Field of
    Mountain Rivers Based on UAV Video. Frontiers in Earth Science, 9, 686636.
    https://doi.org/10.3389/feart.2021.686636
    Fujita, I., Notoya, Y., Tani, K., & Tateguchi, S. (2019). Efficient and
    accurate estimation of water surface velocity in STIV. Environmental Fluid
    Mechanics, 19(5), 1363â€“1378. https://doi.org/10.1007/s10652-018-9651-3

    Author
    ------
    Dr. Carl J. Legleiter, cjl@usgs.gov
    Observing Systems Division, Hydrologic Remote Sensing Branch
    United States Geological Survey

    Ported to Python and modified by:
    Dr. Frank L. Engel, fengel@usgs.gov
    Observing Systems Division, Hydrologic Remote Sensing Branch
    United States Geological Survey
    """

    # Initiate a progress tracker
    progress_emitter = ProgressEmitter()
    if progress_signal is not None:
        # If a signal is provided, connect the emitter's signal to it
        progress_emitter.progress_signal.connect(progress_signal.emit)

    # Set up a sequence of search line orientation angles phi, specified
    # relative to our initial search line orientation phi_origin_ari, which we
    # first need to convert from a geographical angle to a mathematical angle.
    if isinstance(phi_origin, int):
        phi_origin = np.repeat(phi_origin, x_origin.shape[0])
    phi_origin_ari = geographic_to_arithmetic(phi_origin, signed180=True)

    # Need to compute the time and space lags such that the center of the
    # ACF array is at a lag of (0,0), which has a correlation of 1 by
    # definition. The following should confirm this: R(num_pixels,num_frames).
    # Get lags in native units of pixels and frames
    pixel_lag = np.arange(-num_pixels, num_pixels + 1)
    num_frames = image_stack.shape[2]
    frame_lag = np.arange(-num_frames, num_frames + 1)

    # Crop to central portion of the ACF
    # We know that the center of the surface is at nPix,nFrame where the lag
    # vector is (0,0) and the ACF has a value of 1, so go out by half the
    # maximum lag in each direction from this center point
    pixel_lag_sub = pixel_lag[num_pixels // 2 : -num_pixels // 2]
    frame_lag_sub = frame_lag[num_frames // 2 : -num_frames // 2]

    # Prepare for numerical integration of the ACF over rho in polar
    # coordinates
    rho_max = np.min([np.max(pixel_lag_sub), np.max(frame_lag_sub)])

    # Create a vector of rho (magnitude) values we can use to interpolate the
    # ACF onto for a given angle theta.
    rho = np.arange(0, rho_max + d_rho, d_rho)

    # Set up the range of theta values for which we will be integrating over
    # rho
    theta = np.arange(0, 180 + d_theta, d_theta)

    # Get the index of lag (0, 0)
    x_lag_0 = num_pixels // 2 - 1
    t_lag_0 = num_frames // 2 - 1

    # Create the interpolator object
    image_interpolator = create_image_interpolator(image_stack)

    # Define the objective function
    def stiv_objective_func(phi, theta_max=None):
        global global_theta_max  # stores resulting theta_max values

        # Given the origin and current angle, define search line end points
        x1, y1 = polar_to_cartesian(np.deg2rad(phi), num_pixels, isImage=True)
        x_end_points = np.array([x0, x0 + x1])
        y_end_points = np.array([y0, y0 + y1])

        # Get length of this line in pixel units, which should be the
        # same as our nPix input, but recalculate here just to be safe
        # dist = int(num_pixels)
        dist = np.hypot(
            np.abs(np.diff(y_end_points, axis=0)),
            np.abs(np.diff(x_end_points, axis=0)),
        ).astype(int)

        # Create regularly spaced points along this line
        xi = np.linspace(x_end_points[0], x_end_points[1], dist.item() + 1)
        yi = np.linspace(y_end_points[0], y_end_points[1], dist.item() + 1)

        # To ensure that we can set up the STI, and hence the ACF,
        # such that the center of the ACF array at a lag of (0,0) has a
        # value of 1, we need to make the profile lines have an even
        # number of points, so if the length of xi and yi is not even,
        # drop the last point to make it even in length
        if len(xi) % 2 != 0:
            xi = xi[:-1]
            yi = yi[:-1]

        # Use the gridded interpolant for the image stack that we set up
        # outside the loop to extract pixel values along search line for
        # each frame to build the STI. This is a vectorized approach
        # that removes the extra loop.
        sti = np.zeros([len(xi), num_frames])
        sti_w, sti_h = sti.shape
        frame_indices = np.arange(num_frames)
        frame_indices_repeat = np.tile(frame_indices, xi.shape[0])
        yi = np.repeat(yi, num_frames)
        xi = np.repeat(xi, num_frames)
        coordinates = np.column_stack((yi, xi, frame_indices_repeat))
        sti = image_interpolator(coordinates).reshape(sti_w, sti_h)

        # Apply Fujita's standardization filter, basically subtracting
        # the time average (mean of each row) and then dividing by the
        # standard deviation of that row, where the rows are time series
        # for a fixed spatial position. Note that this is essentially a
        # z-score, so we can use a standard normalization approach for
        # this.
        normalized_sti = zscore(sti, axis=1, ddof=1)

        # Calculate the ACF of the STI and
        reversed_normalized_sti = np.flip(normalized_sti, axis=(0, 1))
        R = fftconvolve(normalized_sti, reversed_normalized_sti, mode="full")
        R = R / np.max(R)

        # Crop to central portion of the ACF We know that the center of
        # the surface is at nPix,nFrame where the lag vector is (0,
        # 0) and the ACF has a value of 1, so go out by half the maximum
        # lag in each direction from this center point
        R_sub = R[
            num_pixels - num_pixels // 2 : num_pixels + num_pixels // 2 + 1,
            num_frames - num_frames // 2 : num_frames + num_frames // 2 + 1,
        ]

        # Radiate rays on the ACF from the origin at lag (0,0)
        theta_rad = np.deg2rad(theta)
        theta_expanded = np.tile(theta_rad, (len(rho), 1)).T
        t_ray, x_ray = polar_to_cartesian(theta_expanded, rho)

        # Set up gridded interpolant for extracting ACF values along each
        # ray.
        acf_interpolator = RegularGridInterpolator(
            (np.arange(R_sub.shape[0]), np.arange(R_sub.shape[1])),
            R_sub,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        # We want all the rho values for each theta and then use those
        # as the integrand. This is a vectorized approach to do it.
        F_theta = np.zeros(len(theta))

        # Create arrays for all theta values
        x_coords = x_ray + x_lag_0
        t_coords = t_ray + t_lag_0

        # Stack the x and t coordinates along with the corresponding
        # theta values
        coordinates = np.column_stack((x_coords.flatten(), t_coords.flatten()))

        # Interpolate the ACF values along all rays
        acfRay = acf_interpolator(coordinates).reshape(len(theta), -1)
        F_theta = np.trapz(acfRay, axis=1) * d_rho

        # Now get the peak value of the integrals and the angle with the
        # maximum autocorrelation
        p = np.max(F_theta)
        i_max = np.argmax(F_theta)
        global_theta_max = theta[i_max]  # Assign theta_max to global variable

        # Take the negative of the objective function to actually find the
        # maximum
        p = -1 * p
        return p

    # Start the timer
    start_time = time.time()

    # Init the resultant arrays
    magnitudes = np.zeros(x_origin.shape)
    directions = np.zeros(x_origin.shape)

    for ii in tqdm(range(len(x_origin)), desc="Processing node"):
        x0 = x_origin[ii]
        y0 = y_origin[ii]

        # Define the bounds for optimization (e.g., -90 to 90 degrees)
        bounds = (phi_origin_ari[0] - 90, phi_origin_ari[0] + 90)

        # Your initial guess for the angle
        initial_phi = phi_origin_ari[ii]

        # Call minimize_scalar to optimize phi
        result = minimize_scalar(
            stiv_objective_func,
            bounds=bounds,
            method="bounded",
            tol=tolerance,
        )
        directions[ii] = result.x  # this is 'p'
        magnitudes[ii] = np.tan(np.deg2rad(global_theta_max)) * pixel_gsd / d_t

    # Stop the timer
    end_time = time.time()

    # Apply the velocity threshold
    # This method can be noisier and produce obvious junk
    magnitudes[
        (magnitudes < -max_vel_threshold) | (magnitudes > max_vel_threshold)
    ] = np.nan

    # Magnitudes should only be returned as positive
    magnitudes = np.abs(magnitudes)

    # Directions should be returned as geographic angle (b/c phi was
    # supplied as geographic angle
    directions = arithmetic_to_geographic(directions)

    # If a signal is provided, emit progress information
    if progress_signal is not None:
        progress_percentage = int((ii / len(x_origin)) * 100)
        progress_emitter.progress_signal.emit(progress_percentage)

    return magnitudes, directions


def remap_stiv_to_square_array(normalized_sti, num_pixels, num_frames):
    """Remap the STIV Space-Time Image result to a square (normalized) array

    Args:
        normalized_sti (_type_): the normalized STI image
        num_pixels (float): the new square STI number of pixels (defines the size of the remapped STI)
        num_frames (int): the number of frames

    Returns:
        _type_: _description_
    """
    # Define the x and y coordinates for the original image
    x = np.arange(num_pixels)
    y = np.arange(num_frames)

    # Create a meshgrid for the original image
    X, Y = np.meshgrid(x, y)

    # Define the new x and y coordinates for the square image
    new_x = np.linspace(0, num_pixels - 1, num=num_frames)
    new_y = np.linspace(0, num_frames - 1, num=num_frames)

    # Create a meshgrid for the new square image
    new_X, new_Y = np.meshgrid(new_x, new_y)

    # Transpose the normalized_sti array
    normalized_sti_transposed = normalized_sti.T

    # Interpolate the transposed original image to the new size using nearest neighbor interpolation
    f = interp2d(x, y, normalized_sti_transposed)

    # Evaluate the interpolated function on the new grid
    square_sti = f(new_x, new_y)

    return square_sti


def process_node(args):
    (
        x_origin,
        y_origin,
        x_lag,
        t_lag,
        phi,
        theta,
        rho,
        d_rho,
        num_frames,
        num_pixels,
        d_t,
        pixel_gsd,
        image_interpolator,
    ) = args
    """Processing node used by the STIV processor

    Returns:
        tuple: tuple of the magnitude and peak for the node
    """

    mag = np.empty(phi.shape)
    peak = np.empty(phi.shape)
    mag[:] = np.nan
    peak[:] = np.nan
    for i_phi in range(len(phi)):
        # Given the origin and current angle, define search line end points
        x1, y1 = polar_to_cartesian(
            np.deg2rad(phi[i_phi]), num_pixels, isImage=True
        )
        x_end_points = np.array([x_origin, x_origin + x1])
        y_end_points = np.array([y_origin, y_origin + y1])

        # Get length of this line in pixel units, which should be the
        # same as our nPix input, but recalculate here just to be safe
        dist = np.hypot(
            np.abs(np.diff(y_end_points, axis=0)),
            np.abs(np.diff(x_end_points, axis=0)),
        ).astype(int)

        # Create regularly spaced points along this line
        xi = np.linspace(x_end_points[0], x_end_points[1], dist.item() + 1)
        yi = np.linspace(y_end_points[0], y_end_points[1], dist.item() + 1)

        # To ensure that we can set up the STI, and hence the ACF,
        # such that the center of the ACF array at a lag of (0,0) has a
        # value of 1, we need to make the profile lines have an even
        # number of points, so if the length of xi and yi is not even,
        # drop the last point to make it even in length
        if len(xi) % 2 != 0:
            xi = xi[:-1]
            yi = yi[:-1]

        # Use the gridded interpolant for the image stack that we set up
        # outside the loop to extract pixel values along search line for
        # each frame to build the STI. This is a vectorized approach
        # that removes the extra loop.
        sti = np.zeros([len(xi), num_frames])
        sti_w, sti_h = sti.shape
        frame_indices = np.arange(num_frames)
        frame_indices_repeat = np.tile(frame_indices, xi.shape[0])
        yi = np.repeat(yi, num_frames)
        xi = np.repeat(xi, num_frames)
        coordinates = np.column_stack((yi, xi, frame_indices_repeat))
        sti = image_interpolator(coordinates).reshape(sti_w, sti_h)

        # Apply Fujita's standardization filter, basically subtracting
        # the time average (mean of each row) and then dividing by the
        # standard deviation of that row, where the rows are time series
        # for a fixed spatial position. Note that this is essentially a
        # z-score, so we can use a standard normalization approach for
        # this.
        normalized_sti = zscore(sti, axis=1, ddof=1)

        # Calculate the ACF of the STI and
        reversed_normalized_sti = np.flip(normalized_sti, axis=(0, 1))
        R = fftconvolve(normalized_sti, reversed_normalized_sti, mode="full")
        R = R / np.max(R)

        # Crop to central portion of the ACF We know that the center of
        # the surface is at nPix,nFrame where the lag vector is (0,
        # 0) and the ACF has a value of 1, so go out by half the maximum
        # lag in each direction from this center point
        R_sub = R[
            num_pixels - num_pixels // 2 : num_pixels + num_pixels // 2 + 1,
            num_frames - num_frames // 2 : num_frames + num_frames // 2 + 1,
        ]

        # Radiate rays on the ACF from the origin at lag (0,0)
        theta_rad = np.deg2rad(theta)
        theta_expanded = np.tile(theta_rad, (len(rho), 1)).T
        t_ray, x_ray = polar_to_cartesian(theta_expanded, rho)

        # Set up gridded interpolant for extracting ACF values along each
        # ray.
        acf_interpolator = RegularGridInterpolator(
            (np.arange(R_sub.shape[0]), np.arange(R_sub.shape[1])),
            R_sub,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        # We want all the rho values for each theta and then use those
        # as the integrand. This is a vectorized approach to do it.
        F_theta = np.zeros(len(theta))

        # Create arrays for all theta values
        x_coords = x_ray + x_lag
        t_coords = t_ray + t_lag

        # Stack the x and t coordinates along with the corresponding
        # theta values
        coordinates = np.column_stack((x_coords.flatten(), t_coords.flatten()))

        # Interpolate the ACF values along all rays
        acfRay = acf_interpolator(coordinates).reshape(len(theta), -1)
        F_theta = np.trapz(acfRay, axis=1) * d_rho

        # Now get the peak value of the integrals and the angle with the
        # maximum autocorrelation
        peak[i_phi] = np.max(F_theta)
        i_max = np.argmax(F_theta)
        theta_max = theta[i_max]

        # Now calculate velocity by bringing in the pixel size and frame
        # interval See equation 16 of Fujita et al. (2007)
        mag[i_phi] = np.tan(np.deg2rad(theta_max)) * pixel_gsd / d_t
        # print(f"Node: {x_origin}, Mag: {mag[i_phi]}")
    return (mag, peak)


def two_dimensional_stiv_parallel(
    x_origin,
    y_origin,
    image_glob,
    num_pixels,
    phi_origin,
    d_phi,
    phi_range,
    pixel_gsd,
    d_t,
    d_rho=0.5,
    d_theta=0.5,
    num_processes=2,
):
    """Two-dimensional STIV parallel processing method

    CURRENTLY NOT IMPLEMENTED
    """
    raise NotImplementedError()  # This is not complete yet
    # This step takes awhile
    image_stack = create_grayscale_image_stack(image_glob)

    phi_origin_ari = geographic_to_arithmetic(phi_origin)
    phi = np.arange(
        phi_origin_ari - phi_range, phi_origin_ari + phi_range + 1, d_phi
    )
    phi_rad = np.deg2rad(phi)
    pixel_lag = np.arange(-num_pixels, num_pixels + 1)
    num_frames = image_stack.shape[2]
    frame_lag = np.arange(-image_stack.shape[2], image_stack.shape[2] + 1)

    pixel_lag_sub = pixel_lag[num_pixels // 2 : -num_pixels // 2]
    frame_lag_sub = frame_lag[num_frames // 2 : -num_frames // 2]
    pixel_lag_expanded = pixel_lag_sub[:, np.newaxis]
    frame_lag_expanded = frame_lag_sub[np.newaxis, :]
    theta_grid, rho_grid = cartesian_to_polar(
        frame_lag_expanded, pixel_lag_expanded
    )

    rho_max = np.min([np.max(pixel_lag_sub), np.max(frame_lag_sub)])
    rho = np.arange(0, rho_max + d_rho, d_rho)
    theta = np.arange(0, 180 + d_theta, d_theta)

    x_lag_0 = num_pixels // 2 - 1
    t_lag_0 = num_frames // 2 - 1

    # Create the interpolator object
    image_interpolator = create_image_interpolator(image_stack)
    node_args = [
        (
            x_origin[node],
            y_origin[node],
            x_lag_0,
            t_lag_0,
            phi,
            theta,
            rho,
            d_rho,
            num_frames,
            num_pixels,
            d_t,
            pixel_gsd,
            image_interpolator,
        )
        for node in range(len(x_origin))
    ]
    results = []

    for args in node_args:
        result = process_node(args)
        results.append(result)

    # with Pool(num_processes) as pool:
    #     node_args = [(x_origin[node], y_origin[node], x_lag_0, t_lag_0, phi,
    #                   theta, rho, d_rho, num_frames, num_pixels, d_t,
    #                   pixel_gsd, image_interpolator) for node in
    #                  range(len(x_origin))]
    #     results = list(tqdm(pool.imap(process_node, node_args, chunksize=1),
    #                         total=len(x_origin)))

    magnitudes = np.zeros(x_origin.shape)
    directions = np.zeros(x_origin.shape)

    # Results for each node
    for node, (mag, peak) in enumerate(results):
        # Process the results for each node as needed
        print(f"Node {node}: Magnitudes: {mag}, Peaks: {peak}")

    return magnitudes, directions


def save_sti(sti, filepath, format='jpg', auto_scale=True):
    """
    Save a space-time image to disk.

    This function saves an STI to disk in a standard image format, with
    automatic scaling for normalized STIs and direct saving for grayscale STIs.

    Parameters
    ----------
    sti : ndarray
        Space-time image to save. Can be:
        - uint8 grayscale (0-255): saved directly
        - float64 normalized (z-score): automatically scaled to 0-255
    filepath : str
        Output file path. Extension will be added if not present.
    format : str, optional
        Output format: 'jpg' (default), 'png', or 'tiff'
        PNG and TIFF preserve more detail but create larger files.
    auto_scale : bool, optional
        If True (default), automatically scales normalized STIs to 0-255.
        If False, clips values outside 0-255 range.

    Returns
    -------
    str
        Path to the saved file

    Examples
    --------
    >>> sti = extract_space_time_image(interpolator, 100, 200, 45, 50, 25)
    >>> save_sti(sti, 'output/sti_node_001.jpg')
    'output/sti_node_001.jpg'

    >>> # Save normalized STI with custom scaling
    >>> sti_norm = extract_space_time_image(..., normalize='zscore')
    >>> save_sti(sti_norm, 'output/sti_normalized', format='png')
    'output/sti_normalized.png'

    Notes
    -----
    For ML workflows, grayscale STIs (uint8) are recommended as they don't
    require scaling and preserve the original pixel intensity values.
    """
    import os

    # Ensure filepath has the correct extension
    base, ext = os.path.splitext(filepath)
    if not ext or ext[1:].lower() not in ['jpg', 'jpeg', 'png', 'tiff', 'tif']:
        filepath = f"{filepath}.{format}"

    # Handle different STI formats
    if sti.dtype == np.uint8:
        # Already uint8, save directly
        image_to_save = sti
    else:
        # Normalized or float data - need to convert to uint8
        if auto_scale:
            # Scale to 0-255 range
            sti_min = np.min(sti)
            sti_max = np.max(sti)
            if sti_max - sti_min > 0:
                scaled = (sti - sti_min) / (sti_max - sti_min) * 255
            else:
                scaled = np.zeros_like(sti)
            image_to_save = scaled.astype(np.uint8)
        else:
            # Clip to 0-255
            image_to_save = np.clip(sti, 0, 255).astype(np.uint8)

    # Save using PIL
    img = Image.fromarray(image_to_save)
    img.save(filepath)

    return filepath


def load_sti(filepath):
    """
    Load a space-time image from disk.

    Parameters
    ----------
    filepath : str
        Path to the STI image file (JPG, PNG, TIFF, etc.)

    Returns
    -------
    sti : ndarray
        Loaded STI as uint8 grayscale array with shape (height, width)

    Examples
    --------
    >>> sti = load_sti('output/sti_node_001.jpg')
    >>> velocity, theta, p = compute_velocity_from_sti(sti, 0.05, 0.1)

    Notes
    -----
    Loaded STIs are always returned as uint8 grayscale arrays, which can be
    directly passed to compute_velocity_from_sti() or used for visualization.
    """
    img = Image.open(filepath).convert('L')  # Convert to grayscale
    return np.array(img, dtype=np.uint8)


def generate_sti_dataset(
    image_stack,
    node_coordinates,
    search_angles,
    num_pixels,
    output_dir=None,
    sigma=0.5,
    normalize='none',
    progress_callback=None,
):
    """
    Generate a dataset of STIs for ML training or batch processing.

    This function extracts STIs for multiple nodes and search angles,
    optionally saving them to disk in an organized directory structure.
    Useful for creating training datasets for ML-based velocity extraction.

    Parameters
    ----------
    image_stack : ndarray
        Image stack created with create_grayscale_image_stack()
    node_coordinates : ndarray or list of tuples
        Array of (x, y) coordinates for STI extraction.
        Shape: (num_nodes, 2) or list of (x, y) tuples
    search_angles : ndarray or list
        Search line angles in arithmetic degrees.
        Can be single value (applied to all nodes) or array of angles.
    num_pixels : int
        Number of pixels along each search line
    output_dir : str, optional
        If provided, saves STIs to this directory with naming:
        'sti_node{i:04d}_angle{angle:03.0f}.jpg'
        If None, returns STIs in memory only.
    sigma : float, optional
        Gaussian blur sigma for image interpolator. Default is 0.5.
    normalize : str, optional
        'none' (default) for grayscale uint8 or 'zscore' for normalized.
    progress_callback : callable, optional
        Function to call with progress updates: callback(current, total, message)

    Returns
    -------
    stis : list of dict
        List of dictionaries containing:
        - 'sti': the STI array
        - 'node_idx': node index
        - 'x': x coordinate
        - 'y': y coordinate
        - 'angle': search angle
        - 'filepath': path to saved file (if output_dir provided)

    Examples
    --------
    >>> # Generate STIs for ML training
    >>> image_stack = create_grayscale_image_stack(image_paths)
    >>> nodes = [(100, 200), (150, 200), (200, 200)]
    >>> angles = np.arange(0, 180, 15)  # Every 15 degrees
    >>> dataset = generate_sti_dataset(
    ...     image_stack, nodes, angles, num_pixels=50,
    ...     output_dir='training_data/stis'
    ... )
    >>> print(f"Generated {len(dataset)} STIs")

    >>> # Generate for a single angle per node
    >>> nodes = grid_coordinates  # Nx2 array
    >>> angle = 45.0  # Single angle
    >>> dataset = generate_sti_dataset(
    ...     image_stack, nodes, angle, num_pixels=50
    ... )

    Notes
    -----
    For large datasets, providing output_dir is recommended to avoid
    memory issues. The function will save STIs incrementally.
    """
    import os
    from pathlib import Path

    # Create output directory if needed
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Convert inputs to arrays
    if isinstance(node_coordinates, list):
        node_coordinates = np.array(node_coordinates)

    # Handle single angle vs array of angles
    if np.isscalar(search_angles):
        search_angles = [search_angles]
    elif isinstance(search_angles, np.ndarray):
        search_angles = search_angles.tolist()

    # Create image interpolator
    image_interpolator = create_image_interpolator(image_stack, sigma=sigma)

    num_frames = image_stack.shape[2]
    num_nodes = len(node_coordinates)
    num_angles = len(search_angles)
    total_stis = num_nodes * num_angles

    # Generate STIs
    dataset = []
    count = 0

    for i_node, (x, y) in enumerate(node_coordinates):
        for angle in search_angles:
            # Extract STI
            sti = extract_space_time_image(
                image_interpolator=image_interpolator,
                x_origin=float(x),
                y_origin=float(y),
                search_angle_arithmetic=float(angle),
                num_pixels=num_pixels,
                num_frames=num_frames,
                normalize=normalize,
            )

            # Prepare result dictionary
            result = {
                'sti': sti,
                'node_idx': i_node,
                'x': float(x),
                'y': float(y),
                'angle': float(angle),
            }

            # Save to disk if requested
            if output_dir is not None:
                filename = f"sti_node{i_node:04d}_angle{int(angle):03d}.jpg"
                filepath = os.path.join(output_dir, filename)
                save_sti(sti, filepath)
                result['filepath'] = filepath

            dataset.append(result)
            count += 1

            # Progress callback
            if progress_callback is not None:
                progress_callback(
                    count,
                    total_stis,
                    f"Node {i_node+1}/{num_nodes}, Angle {angle:.1f}°",
                )

    return dataset


def optimum_stiv_sample_time(gsd, velocity):
    """Optimum STIV sample time in milliseconds.

    The optimum sample time would produce a theoretical STI angle (theta) of 45
    degrees.
    """
    sample_time_seconds = gsd / velocity
    milliseconds = sample_time_seconds * 1000
    return milliseconds
