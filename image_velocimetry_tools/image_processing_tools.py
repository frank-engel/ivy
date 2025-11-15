"""IVy module containing functions used for image processing"""

import concurrent.futures
import logging
import os
import time
import zlib
from tqdm import tqdm
import cv2
import h5py
import imutils
import numpy as np
from PIL import Image, ImageOps
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


class ImageProcessor(QObject):
    """Main class for the Image Processor

    Args:
        QObject (Qobject): the processing object
    """

    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(
        self,
        image_paths=None,
        map_file_path=None,
        map_file_size_thres=1e9,
        clahe_parameters=(2.0, 8, 8),
        auto_contrast_percent=None,
        do_clahe=False,
        do_auto_contrast=False,
        do_unsharp_mask=False,
        unsharp_parameters=(5, 1.0, 1.0),
        do_edge_enhance=False,
        edge_enhance_alpha=1.5,
        do_dog=False,
        dog_parameters=(1.0, 2.0),
        do_bilateral=False,
        bilateral_parameters=(9, 75, 75),
        do_local_std=False,
        local_std_kernel=15,
    ):
        """Class init

        Args:
            image_paths (list, optional): a list of paths to images, or None. Defaults to None.
            map_file_path (str, optional): path to the memory map file. Defaults to None.
            map_file_size_thres (float, optional): size threshold for the memory map in bytes. Defaults to 1e9.
            clahe_parameters (tuple, optional): CLAHE window size in the horizontal and vertical directions. Defaults to (2.0, 8, 8).
            auto_contrast_percent (float, optional): autocontrast parameter. Defaults to None.
            do_clahe (bool, optional): True if CLAHE is requested. Defaults to False.
            do_auto_contrast (bool, optional): True is autocontrast requested. Defaults to False.
            do_unsharp_mask (bool, optional): True if unsharp masking is requested. Defaults to False.
            unsharp_parameters (tuple, optional): Unsharp mask parameters (kernel_size, sigma, amount). Defaults to (5, 1.0, 1.0).
            do_edge_enhance (bool, optional): True if edge enhancement is requested. Defaults to False.
            edge_enhance_alpha (float, optional): Edge enhancement strength. Defaults to 1.5.
            do_dog (bool, optional): True if Difference of Gaussians is requested. Defaults to False.
            dog_parameters (tuple, optional): DoG parameters (sigma1, sigma2). Defaults to (1.0, 2.0).
            do_bilateral (bool, optional): True if bilateral filtering is requested. Defaults to False.
            bilateral_parameters (tuple, optional): Bilateral filter parameters (d, sigma_color, sigma_space). Defaults to (9, 75, 75).
            do_local_std (bool, optional): True if local std dev is requested. Defaults to False.
            local_std_kernel (int, optional): Local std dev kernel size. Defaults to 15.
        """
        super().__init__()
        self.image_paths = image_paths
        if image_paths is not None:
            self.image_root_path = os.path.dirname(image_paths)
        else:
            self.image_root_path = None
        self.map_file_path = map_file_path
        self.map_file_size_thres = map_file_size_thres
        self.image_stack = None
        self.clahe_parameters = clahe_parameters
        self.auto_contrast_percent = auto_contrast_percent
        self.do_clahe = do_clahe
        self.do_auto_contrast = do_auto_contrast
        self.do_unsharp_mask = do_unsharp_mask
        self.unsharp_parameters = unsharp_parameters
        self.do_edge_enhance = do_edge_enhance
        self.edge_enhance_alpha = edge_enhance_alpha
        self.do_dog = do_dog
        self.dog_parameters = dog_parameters
        self.do_bilateral = do_bilateral
        self.bilateral_parameters = bilateral_parameters
        self.do_local_std = do_local_std
        self.local_std_kernel = local_std_kernel

    def image_stack_creator(self, image_paths, map_file_path, map_file_size_thres):
        """Create an image stack for processing multiple images

        Args:
            image_paths (list): list of paths to images to process, produced by glob
            map_file_path (str): path to the map file
            map_file_size_thres (float): size threshold of the map file in bytes
        """
        if image_paths is None or map_file_path is None or map_file_size_thres is None:
            # Handle the case where arguments are missing
            return
        logging.debug(f" IMAGE STACK: Starting to create the image stack")
        # Go ahead and create the stack
        self.image_stack = create_grayscale_image_stack(
            image_paths, map_file_path, map_file_size_thres
        )
        logging.debug(f" IMAGE STACK: Finished the image stack")
        self.finished.emit()

    def preprocess_images(
        self,
        image_paths,
        clahe_parameters,
        auto_contrast_percent,
        do_clahe,
        do_auto_contrast,
        do_unsharp_mask=False,
        unsharp_parameters=(5, 1.0, 1.0),
        do_edge_enhance=False,
        edge_enhance_alpha=1.5,
        do_dog=False,
        dog_parameters=(1.0, 2.0),
        do_bilateral=False,
        bilateral_parameters=(9, 75, 75),
        do_local_std=False,
        local_std_kernel=15,
    ):
        """Main function for handling the preprocessing thread

        Args:
            image_paths (list): paths to images
            clahe_parameters (tuple): CLAHE paramters
            auto_contrast_percent (float): autocontract parameter
            do_clahe (bool): True if clahe is requested
            do_auto_contrast (bool): True if autocontrast is requested
            do_unsharp_mask (bool): True if unsharp masking is requested
            unsharp_parameters (tuple): Unsharp mask parameters (kernel_size, sigma, amount)
            do_edge_enhance (bool): True if edge enhancement is requested
            edge_enhance_alpha (float): Edge enhancement strength
            do_dog (bool): True if Difference of Gaussians is requested
            dog_parameters (tuple): DoG parameters (sigma1, sigma2)
            do_bilateral (bool): True if bilateral filtering is requested
            bilateral_parameters (tuple): Bilateral filter parameters (d, sigma_color, sigma_space)
            do_local_std (bool): True if local std dev is requested
            local_std_kernel (int): Local std dev kernel size
        """
        if not image_paths:
            return
        image_root_path = os.path.dirname(image_paths[0])
        total_frames = len(image_paths)

        # Define a function to process a single image
        def process_single_image(image_index):
            current_image = image_paths[image_index]
            cv_image = image_file_to_opencv_image(current_image)
            logging.debug(f"Process single image task: {current_image}")

            if do_clahe:
                clip = float(clahe_parameters[0])
                horz_tile_size = int(clahe_parameters[1])
                vert_tile_size = int(clahe_parameters[2])
                cv_image = apply_clahe_to_image(
                    cv_image,
                    clip_size=clip,
                    horz_tile_size=horz_tile_size,
                    vert_tile_size=vert_tile_size,
                )

            if do_auto_contrast:
                cv_image, alpha, beta = automatic_brightness_and_contrast_adjustment(
                    cv_image, clip_histogram_percentage=auto_contrast_percent
                )

            if do_unsharp_mask:
                kernel_size = int(unsharp_parameters[0])
                sigma = float(unsharp_parameters[1])
                amount = float(unsharp_parameters[2])
                cv_image = apply_unsharp_mask(
                    cv_image, kernel_size=kernel_size, sigma=sigma, amount=amount
                )

            if do_edge_enhance:
                cv_image = apply_edge_enhancement(cv_image, alpha=edge_enhance_alpha)

            if do_dog:
                sigma1 = float(dog_parameters[0])
                sigma2 = float(dog_parameters[1])
                cv_image = apply_difference_of_gaussians(
                    cv_image, sigma1=sigma1, sigma2=sigma2
                )

            if do_bilateral:
                d = int(bilateral_parameters[0])
                sigma_color = float(bilateral_parameters[1])
                sigma_space = float(bilateral_parameters[2])
                cv_image = apply_bilateral_filter_exposed(
                    cv_image, d=d, sigma_color=sigma_color, sigma_space=sigma_space
                )

            if do_local_std:
                cv_image = apply_local_std_dev(cv_image, kernel_size=local_std_kernel)

            # Save the processed image as a JPG file
            output_file = os.path.join(image_root_path, f"f{image_index:05d}.jpg")
            cv2.imwrite(output_file, cv_image)

            # Introduce a delay
            # TODO: There are still some resource related issues here.
            time.sleep(0.5)

            # Calculate and update the progress
            progress = int((image_index + 1) / total_frames * 100)
            self.progress.emit(progress)

        # Use concurrent.futures.ThreadPoolExecutor for concurrency
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Map the processing function over the image indices
            executor.map(process_single_image, range(total_frames))

        # Emit finished signal when all images are processed
        self.finished.emit()


def image_file_to_opencv_image(in_file, flag=cv2.IMREAD_COLOR):
    """Return image file contents as opencv image"""
    image = cv2.imread(in_file, flag)
    return image


def image_file_to_numpy_array(in_file):
    """Return image file contents as a NumPy array."""
    image = image_file_to_opencv_image(in_file)
    if image is None:
        raise ValueError(f"Could not read image from file: {in_file}")
    return np.array(image)


def convert_opencv_image_to_qt_pixmap(opencv_image, display_width, display_height):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    converted = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    p = converted.scaled(display_width, display_height, Qt.KeepAspectRatio)
    return QPixmap.fromImage(p)


def apply_clahe_to_image(
    input_image, clip_size=2.0, horz_tile_size=8, vert_tile_size=8
):
    """Return an opencv image with CLAHE applied"""
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=clip_size, tileGridSize=(horz_tile_size, vert_tile_size)
    )
    return clahe.apply(gray)


def automatic_brightness_and_contrast_adjustment(image, clip_histogram_percentage=1.0):
    """Apply an automated histogram adjustment to an opencv image. Returns the gain and bias parameters.

    This function computes an optimize histogram strech by figuring out what the best gain and bias should
    be based on a clipped-tail histogram of the grayscale image (f). It then returns the adjusted image (g),
    gain (alpha), and bias (beta) in the form:

        g(i,j) = alpha * f(i,j) + beta

    Notes
    -----
    Adapted from this approach:
    https://stackoverflow.com/a/56909036/9541277
    """
    # Compute histogram of the grayscale image to get at "strongest" pixels
    if is_image_grayscale(image):
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
    histogram_size = len(histogram)

    # Compute a CDF from the histogram using floating point precision
    points = []
    points.append(float(histogram[0]))
    for index in range(1, histogram_size):
        points.append(points[index - 1] + float(histogram[index]))

    # Clip the histogram as directed, 2-tailed
    max_point = points[-1]
    clip_histogram_percentage *= max_point / 100.0
    clip_histogram_percentage /= 2.0
    min_gray = 0
    max_gray = histogram_size - 1
    while points[min_gray] < clip_histogram_percentage:
        min_gray += 1
    while points[max_gray] >= (max_point - clip_histogram_percentage):
        max_gray -= 1

    # Compute gain and bias
    alpha = 255 / (max_gray - min_gray)
    beta = -1 * min_gray * alpha

    if is_image_grayscale(image):  # Convert to a RGB first
        scaled_image = cv2.convertScaleAbs(
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), alpha=alpha, beta=beta
        )
        scaled_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
    else:
        scaled_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return scaled_image, alpha, beta


def is_image_grayscale(cv_image):
    """Check if the supplied OpenCV image is grayscale

    Args:
        cv_image (ndarray): image in numpy format as supplied by openCV

    Returns:
        bool: True if the image is grayscale
    """
    if len(cv_image.shape) < 3:
        return True
    if cv_image.shape[2] == 1:
        return True
    b, g, r = cv_image[:, :, 0], cv_image[:, :, 1], cv_image[:, :, 2]
    if (b == g).all() and (b == r).all():
        return True
    return False


def create_change_overlay_image(image_paths):
    """Create the change overlay image.

    This imaage shows the overall movement correction made by the
    stabilization process.

    Args:
        image_paths (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if len(image_paths) == 2:
        imgA = Image.open(image_paths[0])
        imgB = Image.open(image_paths[1])

        # Create separate red, green, and blue images
        red_image = Image.new("L", imgA.size)
        green_image = Image.new("L", imgA.size)
        blue_image = Image.new("L", imgA.size)

        # Paste the grayscale images into their respective channels
        red_image.paste(imgB, (0, 0))
        green_image.paste(imgA, (0, 0))
        blue_image.paste(imgB, (0, 0))

        # Merge the separate channels into an RGB image
        composite = Image.merge("RGB", (red_image, green_image, blue_image))

        # Determine the base folder from the first image path
        base_folder = os.path.dirname(image_paths[0])

        # Generate a new file name for the modified image
        file_name, file_extension = os.path.splitext(os.path.basename(image_paths[0]))
        modified_image_path = os.path.join(
            base_folder, f"!{file_name}_change_overlay{file_extension}"
        )

        # Save the modified image as JPEG
        change_overlay_image = composite
        change_overlay_image.save(modified_image_path)

        return modified_image_path

    else:
        raise ValueError(
            "This function requires exactly two image paths to create a red-cyan composite."
        )


def apply_color_tint(image, tint_strength=1.0, color="red"):
    """Tint a supplied image

    Args:
        image (ndarray): the image
        tint_strength (float, optional): the strength of the tint. Defaults to 1.0.
        color (str, optional): color to tint (red or blue). Defaults to "red".

    Returns:
        _type_: _description_
    """
    # Define the color ramp based on the tint strength
    if color == "red":
        color_ramp = [
            (
                int(255 * (1 - tint_strength)),
                int(255 * (1 - tint_strength)),
                255,
            ),  # Blue
            (
                255,
                int(255 * (1 - tint_strength)),
                int(255 * (1 - tint_strength)),
            ),  # Red
        ]
    elif color == "blue":
        color_ramp = [
            (
                int(255 * (1 - tint_strength)),
                int(255 * (1 - tint_strength)),
                255,
            ),  # Blue
            (
                255,
                int(255 * (1 - tint_strength)),
                int(255 * (1 - tint_strength)),
            ),  # Red
        ]

    # Calculate the tint color based on the position in the list
    width, height = image.size
    num_images = 3  # We're always using 3 images

    tint_color = (
        int(
            color_ramp[0][0]
            + (color_ramp[1][0] - color_ramp[0][0]) * (1 / (num_images - 1))
        ),
        int(
            color_ramp[0][1]
            + (color_ramp[1][1] - color_ramp[0][1]) * (1 / (num_images - 1))
        ),
        int(
            color_ramp[0][2]
            + (color_ramp[1][2] - color_ramp[0][2]) * (1 / (num_images - 1))
        ),
    )

    # Create a tinted image by multiplying the grayscale image with the tint color
    tinted_image = ImageOps.colorize(
        ImageOps.grayscale(image), "#000000", f"rgb{tint_color}"
    )

    # Convert the tinted image back to RGB
    tinted_image = tinted_image.convert("RGB")

    return tinted_image


def convert_image_file_to_grayscale(path, height, width, dtype):
    """Convert input image file to grayscale using OpenCv."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img.shape == (height, width):
        return img.astype(dtype)
    else:
        print(f"Skipping image {path} due to resolution mismatch.")
        return None


def create_grayscale_image_stack(
    image_paths, progress_callback=None, map_file_path=None, map_file_size_thres=1e9
):
    """
    Load a list of image files, convert them to grayscale, and create an image stack.

    Parameters:
    image_paths (list of str): A list of file paths to image files.

    progress_callback (function, optional): A callback function to report progress.
        The function should accept a single argument indicating the progress percentage.

    map_file_path (str, optional): If specified, the path to a memory-mapped file to store
        the image stack. If the total estimated memory usage exceeds map_file_size_thres,
        this memory-mapped file will be used to optimize memory consumption.

    map_file_size_thres (int, optional): A threshold (in bytes) for the total estimated
        memory usage. If the estimated memory usage is larger than this threshold and
        map_file_path is specified, a memory-mapped file will be used. Default is 1e9 bytes
         (2 gigabytes).

    Returns:
    ndarray: A 3D NumPy array representing the image stack where each 2D slice
             corresponds to a grayscale image, and the third dimension represents
             different images. The data type of the array is float.

    Example:
    >>> image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    >>> image_stack = create_grayscale_image_stack(image_paths)
    >>> print(image_stack.shape)
    (height, width, num_images)
    >>> print(image_stack.dtype)
    float64

    Notes:
    - This function loads and converts images to grayscale using OpenCV (cv2.IMREAD_GRAYSCALE).
    - The input images should have the same dimensions (height and width).
    - Grayscale images are represented as float arrays with values in the range [0, 255],
      where 0 corresponds to black and 255 corresponds to white.
    - The order of images in the stack corresponds to the order of image paths in the input list.
    - Memory usage optimization: If map_file_path is specified and the estimated memory
      usage exceeds map_file_size_thres, a memory-mapped file is used to reduce memory consumption.

    """
    first_image = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    height, width = first_image.shape
    num_images = len(image_paths)

    # Estimate the needed memory for the stack
    memory_estimate_bytes = estimate_image_stack_memory_usage(
        height, width, num_images, num_bands=1
    )  # uint8 grayscale

    # Use uint8 dtype for grayscale images
    dtype = np.uint8

    grayscale_images = []
    for i, path in enumerate(image_paths):
        # Read image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Check dimensions
        if img.shape == (height, width):
            grayscale_images.append(img.astype(dtype))
        else:
            print(f"Skipping image {path} due to resolution mismatch.")

        # Report progress for the first loop (1-30%)
        if progress_callback and i < num_images * 0.3:
            progress_callback(int((i + 1) / (num_images * 0.3) * 30))

    # Create a memory-mapped array if map_file_path is specified and the total estimated size is larger than the threshold
    if map_file_path and memory_estimate_bytes > map_file_size_thres:
        # Ensure the directory for the memory-mapped file exists
        map_dir = os.path.dirname(map_file_path)
        os.makedirs(map_dir, exist_ok=True)

        # Create a memory-mapped array with the same shape as the image stack
        image_stack = np.memmap(
            map_file_path, dtype=dtype, mode="w+", shape=(height, width, num_images)
        )

        # Directly assign the data to the memory-mapped array
        for i, img in enumerate(grayscale_images):
            image_stack[:, :, i] = img

            # Report progress for the second loop (31-100%) for each image
            progress = int((i / num_images * 70) + 30)
            if progress_callback:
                progress_callback(progress)

    else:
        # Create a regular NumPy array
        image_stack = np.dstack(grayscale_images)
        if progress_callback:
            progress_callback(100)

    return image_stack


# def create_grayscale_image_stack(image_paths, progress_callback=None,
#                                  map_file_path=None, map_file_size_thres=1e9):
#     """
#     Load a list of image files, convert them to grayscale, and create an
#     image stack in parallel.
#
#         Parameters:
#         image_paths (list of str): A list of file paths to image files.
#
#         progress_callback (function, optional): A callback function to report
#             progress. The function should accept a siqngle argument indicating
#             the progress percentage.
#
#         map_file_path (str, optional): If specified, the path to a
#             memory-mapped file to store the image stack. If the total estimated
#             memory usage exceeds map_file_size_thres, this memory-mapped file
#             will be used to optimize memory consumption.
#
#         map_file_size_thres (int, optional): A threshold (in bytes) for the
#             total estimated memory usage. If the estimated memory usage is
#             larger than this threshold and map_file_path is specified,
#             a memory-mapped file will be used. Default is 1e9 bytes (2
#             gigabytes).
#
#         Returns:
#             ndarray: A 3D NumPy array representing the image stack
#                 where each 2D slice corresponds to a grayscale image, and the third
#                 dimension represents different images. The data type of the
#                 array is uint8.
#
#         Notes:
#         - This function loads and converts images to grayscale using OpenCV
#           (cv2.IMREAD_GRAYSCALE).
#         - The input images should have the same dimensions (height and width).
#         - Grayscale images are represented as uint8 arrays with values in the
#           range [0, 255],
#           where 0 corresponds to black and 255 corresponds to white.
#         - The order of images in the stack corresponds to the order of image
#           paths in the input list.
#         - Memory usage optimization: If map_file_path is specified and the
#           estimated memory
#           usage exceeds map_file_size_thres, a memory-mapped file is used to
#           reduce memory consumption.
#
#         Example:
#         >>> image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
#         >>> image_stack = create_grayscale_image_stack(image_paths)
#         >>> print(image_stack.shape)
#         (height, width, num_images)
#         >>> print(image_stack.dtype)
#         uint8
#
#         """
#     first_image = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
#     height, width = first_image.shape
#     num_images = len(image_paths)
#     dtype = np.uint8
#
#     memory_estimate_bytes = estimate_image_stack_memory_usage(
#         height, width, num_images, num_bands=1
#     )  # uint8 grayscale
#
#     grayscale_images = []
#     lock = threading.Lock()
#
#     def append_to_list(img):
#         nonlocal grayscale_images
#         with lock:
#             grayscale_images.append(img.astype(dtype))
#             if progress_callback:
#                 progress_callback(int(len(grayscale_images) / num_images * 70))
#
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = [executor.submit(convert_image_file_to_grayscale, path,
#                                    height, width, dtype) for path in
#                    image_paths]
#         for future in concurrent.futures.as_completed(futures):
#             img = future.result()
#             if img is not None:
#                 append_to_list(img)
#
#     if map_file_path and memory_estimate_bytes > map_file_size_thres:
#         map_dir = os.path.dirname(map_file_path)
#         os.makedirs(map_dir, exist_ok=True)
#         image_stack = np.memmap(
#             map_file_path, dtype=dtype, mode="w+", shape=(height, width, num_images)
#         )
#         with lock:
#             for i, img in enumerate(grayscale_images):
#                 image_stack[:, :, i] = img
#                 if progress_callback:
#                     progress_callback(70 + int((i + 1) / num_images * 30))
#     else:
#         with lock:
#             if progress_callback:
#                 progress_callback(85)
#             image_stack = np.dstack(grayscale_images)
#
#     if progress_callback:
#         progress_callback(100)
#
#     return image_stack


def compress_image_stack(image_stack):
    """
    Compress a 3D NumPy array representing an image stack.

    Parameters:
    image_stack (ndarray): A 3D NumPy array representing the image stack.

    Returns:
    ndarray: A compressed 1D NumPy array.
    """
    # Flatten the 3D array to a 1D array
    flattened_array = image_stack.flatten()

    # Compress the flattened array using zlib
    compressed_array = zlib.compress(flattened_array)

    return compressed_array


def decompress_image_stack(compressed_array, shape):
    """
    Decompress a compressed 1D NumPy array to a 3D array.

    Parameters:
    compressed_array (ndarray): A compressed 1D NumPy array.
    shape (tuple): The shape of the original 3D array.

    Returns:
    ndarray: A 3D NumPy array representing the decompressed image stack.
    """
    # Decompress the 1D array
    flattened_array = zlib.decompress(compressed_array)

    # Reshape the flattened array to the original 3D shape
    decompressed_array = np.reshape(flattened_array, shape)

    return decompressed_array


def create_grayscale_image_stack_hdf5(image_paths, hdf5_file_path):
    """
    Load a list of image files, convert them to grayscale, and create an image stack.
    Save the image stack as an HDF5 file.

    Parameters:
    image_paths (list of str): A list of file paths to image files.

    hdf5_file_path (str): The path to the HDF5 file to save the image stack.

    Returns:
    None
    """

    def load_and_convert_image(index):
        grayscale_image = cv2.imread(image_paths[index], cv2.IMREAD_GRAYSCALE)
        return grayscale_image.astype(float)

    first_image = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    height, width = first_image.shape
    num_images = len(image_paths)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        grayscale_images = list(executor.map(load_and_convert_image, range(num_images)))

    # Convert the list of grayscale images to a NumPy array
    image_stack = np.dstack(grayscale_images)

    # Save the image stack to an HDF5 file
    with h5py.File(hdf5_file_path, "w") as hf:
        hf.create_dataset("image_stack", data=image_stack)


def load_grayscale_image_stack_hdf5(hdf5_file_path):
    """
    Load a grayscale image stack from an HDF5 file.

    Parameters:
    hdf5_file_path (str): The path to the HDF5 file containing the image stack.

    Returns:
    ndarray: A 3D NumPy array representing the loaded image stack where each 2D slice
             corresponds to a grayscale image, and the third dimension represents
             different images. The data type of the array is float.
    """
    with h5py.File(hdf5_file_path, "r") as hf:
        image_stack = hf["image_stack"][:]

    return image_stack


def bilateral_filter(image, sigma_spatial=0.5, sigma_range=0.1):
    """
    Apply bilateral filtering to the input image.

    Parameters:
        image (ndarray): Input image.
        sigma_spatial (float): Standard deviation for spatial filtering.
        sigma_range (float): Standard deviation for range filtering.

    Returns:
        filtered_image (ndarray): Filtered image.
    """
    # Perform spatial filtering
    spatial_filtered = gaussian_filter(image, sigma=sigma_spatial)

    # Intensity-based filter (range filtering)
    intensity_filtered = np.zeros_like(image)
    for channel in range(image.shape[-1]):
        intensity_filtered[..., channel] = gaussian_filter(
            image[..., channel], sigma=sigma_range
        )

    # Combine spatial and intensity-based filtering
    filtered_image = (
        spatial_filtered * intensity_filtered / (spatial_filtered.max() + 1e-8)
    )

    return filtered_image


def create_image_interpolator(
    image_stack, sigma=1.0, blur_sigma_spatial=0.5, blur_sigma_range=0.1
):
    """
    Create an interpolator for the input image stack with bilateral filtering preprocessing.

    Parameters:
        image_stack (ndarray): Input image stack with shape (height, width, num_images).
        blur_sigma_spatial (float): Standard deviation for spatial filtering in bilateral filtering.
        blur_sigma_range (float): Standard deviation for range filtering in bilateral filtering.

    Returns:
        interpolator: Interpolator for the preprocessed image stack.
    """
    # Apply bilateral filtering to the image stack
    # Note this is slower, and seems to perform with STIV poorly. I am
    # leaving it in as a possible method, but not using it at this time
    # --FLE March 2024
    # blurred_stack = bilateral_filter(image_stack, sigma_spatial=blur_sigma_spatial, sigma_range=blur_sigma_range)
    if sigma > 0.01:
        blurred_stack = gaussian_filter(image_stack, sigma=1.0)
    else:
        blurred_stack = image_stack

    height, width, num_images = blurred_stack.shape

    # Create a set of coordinates for the grid along axis=2
    indices = np.arange(num_images)

    # Define the interpolator using RegularGridInterpolator
    interpolator = RegularGridInterpolator(
        (np.arange(height), np.arange(width), indices),  # Grid coordinates
        blurred_stack,  # Values to interpolate (the blurred image stack)
        method="nearest",
        bounds_error=False,  # Disable bounds error checking
        fill_value=0.0,  # Fill value for out-of-bounds points
    )
    return interpolator


def estimate_image_stack_memory_usage(height, width, num_images, num_bands=1):
    """
    Estimate the memory usage in bytes for an image stack based on its dimensions.

    Parameters:
    height (int): Height (number of pixels) of each image in the stack.
    width (int): Width (number of pixels) of each image in the stack.
    num_images (int): Number of images in the stack.
    num_bands (int, optional): Number of color bands (1 for grayscale, 3 for RGB). Default is 1.

    Returns:
    int: Estimated memory usage in bytes.

    Note:
    The estimation assumes a specified number of color bands per pixel (e.g., 1 for grayscale).
    """
    # Calculate memory usage per image
    bytes_per_pixel = num_bands  # Each pixel is represented by num_bands bytes
    memory_per_image = height * width * bytes_per_pixel

    # Calculate total memory usage for the image stack
    total_memory_usage = memory_per_image * num_images

    return total_memory_usage


def create_binary_mask(polygons, image_width, image_height):
    """
    Create a binary mask from a list of polygons.

    This function takes a list of polygons represented as NumPy arrays of points
    and produces a binary mask. The binary mask is used to mark regions outside the
    the polygons with 1 and regions inside the polygons with 0.

    Parameters:
    - polygons (list of numpy.ndarray): A list of polygons, where each polygon is
      represented as a NumPy array of points with shape (N, 2), and N is the number
      of vertices in the polygon.
    - image_width (int): The width of the target binary mask image.
    - image_height (int): The height of the target binary mask image.

    Returns:
    - binary_mask (numpy.ndarray): A binary mask with the same dimensions as the
      specified image_width and image_height. The mask has values of 1 for regions
      outside the polygons and 0 for regions inside the polygons.

    Example:
    >>> polygons = [np.array([(100, 100), (200, 100), (200, 200), (100, 200)]),
    ...             np.array([(300, 300), (400, 300), (400, 400), (300, 400])]
    ... binary_mask = create_binary_mask(polygons, image_width, image_height)

    Notes:
    - The function assumes that the polygons' points are in integer coordinates. It will
      convert the supplied polygons into int32 before processing.

    """
    # Ensure that polygons are of the correct data type (int32)
    polygons = [polygon.astype(np.int32) for polygon in polygons]

    # Create an empty binary mask.
    binary_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Create a blank image (all black) with the same dimensions as the mask.
    image = np.zeros((image_height, image_width), dtype=np.uint8)

    # Draw the polygons on the blank image.
    for polygon in polygons:
        cv2.fillPoly(image, [polygon], 1)

    # Copy the inverted image to the binary mask.
    binary_mask = 1 - image
    return binary_mask


def close_small_gaps(binary_mask, kernel_size=5, area_threshold=0.03, blur_sigma=1.0):
    """
    Close small gaps and smooth a binary mask.

    This function takes a binary mask and performs operations to close small gaps and smooth the mask.
    It uses dilation to close gaps, removes small connected components based on an area threshold,
    and applies Gaussian blur to achieve smoothing.

    Parameters:
    - binary_mask (numpy.ndarray): The input binary mask to be processed.
    - kernel_size (int, optional): The size of the dilation kernel. Default is 5.
    - area_threshold (float, optional): The area threshold as a percentage of the total pixels
      in the image for removing small regions. Default is 0.03 (3%).
    - blur_sigma (float, optional): The standard deviation of the Gaussian blur filter for
      smoothing. Default is 1.0.

    Returns:
    - smoothed_mask (numpy.ndarray): The processed binary mask with small gaps closed,
      small regions removed, and smoothed transitions.

    Example:
    >>> result_mask = close_small_gaps(binary_mask, kernel_size=7, area_threshold=0.02, blur_sigma=2.0)

    Notes:
    - The function uses dilation for closing small gaps and Gaussian blur for smoothing.
    - Connected components with an area below the area threshold are removed from the mask.

    """

    # Create a kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Dilate the binary mask to close small gaps
    closed_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # Find connected components and their areas
    _, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)

    # Get the total number of pixels in the image
    total_pixels = closed_mask.size

    # Set labels to 0 for regions with an area below the threshold
    for label, stat in enumerate(stats):
        if stat[4] / total_pixels < area_threshold:
            closed_mask[labels == label] = 0

    # Apply Gaussian blur to smooth the binary mask
    smoothed_mask = cv2.GaussianBlur(closed_mask.astype(np.float32), (0, 0), blur_sigma)

    # Threshold the smoothed mask to set values less than 1 to 0
    thresholded_mask = np.where(smoothed_mask < 1, 0, smoothed_mask)

    return thresholded_mask


def generate_points_along_line(
    image_width, image_height, line_start, line_end, number_points, mask
):
    """
    Generate evenly spaced points along a line in unmasked regions of an image.

    Parameters:
    - image_width (int): Width of the image in pixels.
    - image_height (int): Height of the image in pixels.
    - line_start (numpy.ndarray): Pixel coordinates of the start point on the line.
    - line_end (numpy.ndarray): Pixel coordinates of the end point on the line.
    - number_points (int): Number of points to generate along the line.
    - mask (numpy.ndarray): Binary mask of the same size as the image, where
      1 represents unmasked regions, and 0 represents masked regions.

    Returns:
    - numpy.ndarray: An array of pixel locations corresponding to points along the line.

    The function identifies unmasked regions in the image using the provided mask
    and generates evenly spaced points along the line connecting the start and end points.
    The generated points are returned as a NumPy array, and they are located in unmasked
    regions of the image.

    Example:
    >>> image_width = 400
    >>> image_height = 300
    >>> line_start = np.array([50, 50])
    >>> line_end = np.array([350, 250])
    >>> number_points = 10
    >>> mask = np.ones((image_height, image_width))
    >>> points_along_line = generate_points_along_line(image_width, image_height, line_start, line_end, number_points, mask)
    >>> print(points_along_line)
    array([[ 50.,  50.],
           [ 90.,  90.],
           [130., 130.],
           ...
           [310., 210.],
           [350., 250.]])

    """
    # Ensure the line points are within the image bounds
    line_start = np.clip(line_start, [0, 0], [image_width - 1, image_height - 1])
    line_end = np.clip(line_end, [0, 0], [image_width - 1, image_height - 1])

    # Generate evenly spaced points along the line
    t_values = np.linspace(0, 1, number_points)
    line_points = np.outer(1 - t_values, line_start) + np.outer(t_values, line_end)
    line_points = np.round(line_points).astype(int)

    # Filter points to keep only those in unmasked regions
    valid_points = line_points[
        (line_points[:, 0] < image_width)
        & (line_points[:, 1] < image_height)
        & (mask[line_points[:, 1], line_points[:, 0]] == 1)
    ]

    return valid_points


def generate_grid(
    image_width, image_height, vertical_spacing, horizontal_spacing, mask
):
    """
    Generate a regular grid of pixel locations in unmasked regions of an image.

    Parameters:
    - image_width (int): Width of the image in pixels.
    - image_height (int): Height of the image in pixels.
    - vertical_spacing (int): Vertical spacing between grid nodes.
    - horizontal_spacing (int): Horizontal spacing between grid nodes.
    - mask (numpy.ndarray): Binary mask of the same size as the image, where
      1 represents unmasked regions, and 0 represents masked regions.

    Returns:
    - numpy.ndarray: An array of pixel locations corresponding to each grid node.

    The function identifies unmasked regions in the image using the provided mask
    and generates a regular grid of nodes with the specified vertical and horizontal
    spacing. The generated grid is returned as a NumPy array, and it only includes
    grid nodes located in unmasked regions of the image.

    Example:
    >>> image_width = 400
    >>> image_height = 300
    >>> vertical_spacing = 50
    >>> horizontal_spacing = 50
    >>> mask = np.ones((image_height, image_width))
    >>> grid = generate_grid(image_width, image_height, vertical_spacing, horizontal_spacing, mask)
    >>> print(grid)
    array([[  0,   0],
           [ 50,   0],
           [100,   0],
           ...
           [350, 250],
           [400, 250],
           [  0, 300],
           [ 50, 300],
           ...
           [400, 300]])

    """

    # Find the coordinates of unmasked pixels
    unmasked_coords = np.argwhere(mask == 1)

    # Filter the coordinates to include only those within the image bounds
    valid_coords = unmasked_coords[
        (unmasked_coords[:, 0] < image_height) & (unmasked_coords[:, 1] < image_width)
    ]

    # Generate grid nodes based on the specified spacing
    x_range = np.arange(0, image_width, horizontal_spacing)
    y_range = np.arange(0, image_height, vertical_spacing)

    # Create a mesh grid for the grid nodes
    grid_x, grid_y = np.meshgrid(x_range, y_range)

    # Flatten the grid and stack the x and y coordinates
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # Filter grid points to keep only those in unmasked regions
    grid_points = grid_points[
        (grid_points[:, 0] < image_width)
        & (grid_points[:, 1] < image_height)
        & (mask[grid_points[:, 1].astype(int), grid_points[:, 0].astype(int)] == 1)
    ]

    return grid_points


def resize_and_save_image(
    input_filename, output_filename, new_size=(300, 300), overwrite=True
):
    """Resize and save the supplied image file

    Args:
        input_filename (str): path to the image
        output_filename (str): path to the new resized image
        new_size (tuple, optional): new size as a tuple of (width, height). Defaults to (300, 300).
        overwrite (bool, optional): True if enabling overwriting of the image. Defaults to True.
    """
    # Step 1: Load the image
    image = cv2.imread(input_filename)

    if image is None:
        print(f"Error: Unable to load the image from {input_filename}")
        return

    # Step 2: Resize the image
    resized_image = imutils.resize(image, width=new_size[0], height=new_size[1])

    # Step 3: Save the resized image
    if overwrite or not cv2.os.path.exists(output_filename):
        cv2.imwrite(output_filename, resized_image)
    else:
        pass


def flip_image_array(image: np.ndarray, flip_x: bool = False, flip_y: bool =
False) -> np.ndarray:
    """
    Flip a NumPy image array along horizontal and/or vertical axes.

    Parameters:
    - image (np.ndarray): The input image array (H, W, C).
    - flip_x (bool): If True, flip the image horizontally (left-right).
    - flip_y (bool): If True, flip the image vertically (top-bottom).

    Returns:
    - np.ndarray: The flipped image (or original if no flips are applied).
    """
    if not flip_x and not flip_y:
        return image

    # Apply flips as needed
    if flip_y:
        image = image[::-1, :, :]
    if flip_x:
        image = image[:, ::-1, :]
    return image


def flip_and_save_images(image_folder, flip_x=False, flip_y=False, progress_callback=None):
    images_to_process = os.listdir(image_folder)
    num_images = len(images_to_process)
    idx = 0

    for filename in tqdm(images_to_process, total=num_images):
        if filename.startswith("f") and filename.lower().endswith(".jpg"):
            src_path = os.path.join(image_folder, filename)
            dest_filename = "t" + filename[1:]  # Replace 'f' with 't'
            dest_path = os.path.join(image_folder, dest_filename)

            # Load, flip, and save using Pillow
            with Image.open(src_path) as img:
                if flip_x and flip_y:
                    img = img.transpose(Image.ROTATE_180)
                elif flip_x:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                elif flip_y:
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                # Save to destination
                img.save(dest_path, format="JPEG")

            if progress_callback:
                progress_callback.emit(int((idx / num_images) * 100))
            idx += 1


# ============================================================================
# Water ROI Extraction Functions
# ============================================================================


def compute_temporal_variance(image_paths, sample_rate=5, progress_callback=None):
    """
    Compute temporal variance across frames to identify moving regions (water).

    This is a lightweight motion detection method that computes pixel-wise variance
    across a subset of frames. Higher variance indicates motion (water), while
    lower variance indicates static regions (banks, vegetation).

    Args:
        image_paths (list): List of paths to image files
        sample_rate (int): Use every Nth frame for speed (default: 5)
        progress_callback (function): Optional callback for progress reporting

    Returns:
        ndarray: Variance map (grayscale image) where bright pixels indicate motion
    """
    if not image_paths:
        raise ValueError("No image paths provided")

    # Sample frames for efficiency
    sampled_paths = image_paths[::sample_rate]

    # Read first image to get dimensions
    first_img = cv2.imread(sampled_paths[0], cv2.IMREAD_GRAYSCALE)
    if first_img is None:
        raise ValueError(f"Could not read image: {sampled_paths[0]}")

    height, width = first_img.shape
    num_samples = len(sampled_paths)

    # Accumulate images for variance computation
    frames = np.zeros((height, width, num_samples), dtype=np.float32)
    frames[:, :, 0] = first_img.astype(np.float32)

    # Load all sampled frames
    for idx, path in enumerate(sampled_paths[1:], start=1):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None and img.shape == (height, width):
            frames[:, :, idx] = img.astype(np.float32)

        if progress_callback and idx % 10 == 0:
            progress = int((idx / num_samples) * 100)
            progress_callback(progress)

    # Compute variance across time dimension
    variance_map = np.var(frames, axis=2)

    # Normalize to 0-255 range for display
    variance_map = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX)
    variance_map = variance_map.astype(np.uint8)

    if progress_callback:
        progress_callback(100)

    return variance_map


def extract_water_roi_from_variance(variance_map, threshold_percentile=50,
                                    min_area_percent=5.0, morph_kernel_size=5):
    """
    Extract water ROI from temporal variance map using adaptive thresholding.

    Args:
        variance_map (ndarray): Temporal variance map from compute_temporal_variance
        threshold_percentile (float): Percentile threshold (0-100). Higher = more selective
        min_area_percent (float): Minimum connected component area as % of image
        morph_kernel_size (int): Kernel size for morphological operations

    Returns:
        ndarray: Binary mask where 1 = water, 0 = non-water (uint8)
    """
    # Compute adaptive threshold based on percentile
    threshold_value = np.percentile(variance_map, threshold_percentile)

    # Threshold to create binary mask
    binary_mask = (variance_map > threshold_value).astype(np.uint8)

    # Apply morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (morph_kernel_size, morph_kernel_size))

    # Close small gaps
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Remove small regions
    total_pixels = binary_mask.size
    min_area = int((min_area_percent / 100.0) * total_pixels)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    # Keep only large components
    cleaned_mask = np.zeros_like(binary_mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned_mask[labels == label] = 1

    return cleaned_mask


def extract_water_roi_by_color(image, color_space='HSV',
                                hue_range=(90, 140), sat_range=(20, 255),
                                val_range=(20, 255)):
    """
    Extract water ROI using color-based segmentation in HSV space.

    Water typically appears in blue-green-cyan range in HSV. This function
    provides a starting point that users can adjust based on their specific
    water conditions (clear, turbid, muddy, etc.).

    Args:
        image (ndarray): Input BGR image from OpenCV
        color_space (str): Color space to use ('HSV' or 'LAB')
        hue_range (tuple): Min/max hue values (0-180 in OpenCV)
        sat_range (tuple): Min/max saturation values (0-255)
        val_range (tuple): Min/max value/brightness (0-255)

    Returns:
        ndarray: Binary mask where 1 = water, 0 = non-water
    """
    if color_space == 'HSV':
        # Convert to HSV
        if is_image_grayscale(image):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color range
        lower_bound = np.array([hue_range[0], sat_range[0], val_range[0]])
        upper_bound = np.array([hue_range[1], sat_range[1], val_range[1]])

        # Create mask
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

    elif color_space == 'LAB':
        # Convert to LAB (useful for turbid/muddy water)
        if is_image_grayscale(image):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # For LAB, use hue_range as L range, sat_range as A, val_range as B
        lower_bound = np.array([hue_range[0], sat_range[0], val_range[0]])
        upper_bound = np.array([hue_range[1], sat_range[1], val_range[1]])

        mask = cv2.inRange(lab, lower_bound, upper_bound)

    # Normalize to 0-1
    mask = (mask > 0).astype(np.uint8)

    return mask


def combine_roi_masks(masks, method='union'):
    """
    Combine multiple ROI masks using different strategies.

    Args:
        masks (list): List of binary masks (each is ndarray)
        method (str): Combination method - 'union', 'intersection', or 'majority'

    Returns:
        ndarray: Combined binary mask
    """
    if not masks:
        raise ValueError("No masks provided")

    if len(masks) == 1:
        return masks[0]

    # Stack masks
    stacked = np.stack(masks, axis=2)

    if method == 'union':
        # Any mask has water
        combined = np.any(stacked, axis=2).astype(np.uint8)
    elif method == 'intersection':
        # All masks agree it's water
        combined = np.all(stacked, axis=2).astype(np.uint8)
    elif method == 'majority':
        # Majority vote
        combined = (np.mean(stacked, axis=2) > 0.5).astype(np.uint8)
    else:
        raise ValueError(f"Unknown method: {method}")

    return combined


# ============================================================================
# Image Enhancement Functions
# ============================================================================


def apply_unsharp_mask(image, kernel_size=5, sigma=1.0, amount=1.0, threshold=0):
    """
    Apply unsharp masking to enhance edges and fine details.

    This is excellent for enhancing water surface texture. The algorithm
    subtracts a blurred version from the original to enhance high frequencies.

    Args:
        image (ndarray): Input image (grayscale or color)
        kernel_size (int): Gaussian kernel size (must be odd)
        sigma (float): Gaussian kernel standard deviation
        amount (float): Strength of sharpening (1.0 = 100%)
        threshold (int): Minimum brightness change to sharpen (0-255)

    Returns:
        ndarray: Sharpened image
    """
    # Work with grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Color image - convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        is_color_input = True
    else:
        gray = image.copy()
        is_color_input = False

    # Create blurred version
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)

    # Calculate sharpening mask
    sharpened = cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)

    # Apply threshold if specified
    if threshold > 0:
        low_contrast_mask = np.abs(gray.astype(np.int16) - blurred.astype(np.int16)) < threshold
        sharpened = np.where(low_contrast_mask, gray, sharpened)

    # Clip values
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened


def apply_edge_enhancement(image, alpha=1.5):
    """
    Enhance edges using Laplacian edge detection.

    This highlights edges and boundaries in the image, useful for
    emphasizing water surface patterns.

    Args:
        image (ndarray): Input image (grayscale or color)
        alpha (float): Enhancement strength (1.0 = original, >1.0 = enhanced)

    Returns:
        ndarray: Edge-enhanced image
    """
    # Work with grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Compute Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Add laplacian back to original (edge enhancement)
    enhanced = gray.astype(np.float64) + alpha * laplacian

    # Normalize and convert back
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return enhanced


def apply_difference_of_gaussians(image, sigma1=1.0, sigma2=2.0, normalize=True):
    """
    Apply Difference of Gaussians (DoG) filter for band-pass filtering.

    This enhances features at specific scales, excellent for highlighting
    water surface texture patterns.

    Args:
        image (ndarray): Input image (grayscale or color)
        sigma1 (float): Smaller Gaussian sigma (captures finer details)
        sigma2 (float): Larger Gaussian sigma (captures coarser details)
        normalize (bool): If True, normalize output to 0-255 range

    Returns:
        ndarray: DoG filtered image
    """
    # Work with grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply two Gaussian blurs with different sigmas
    gaussian1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
    gaussian2 = cv2.GaussianBlur(gray, (0, 0), sigma2)

    # Compute difference
    dog = gaussian1.astype(np.float32) - gaussian2.astype(np.float32)

    if normalize:
        # Normalize to 0-255
        dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
        dog = dog.astype(np.uint8)

    return dog


def apply_local_std_dev(image, kernel_size=15):
    """
    Compute local standard deviation to highlight texture variation.

    This creates a texture map showing areas with high local variation
    (like moving water) vs smooth areas (like calm water or banks).

    Args:
        image (ndarray): Input image (grayscale or color)
        kernel_size (int): Size of local neighborhood

    Returns:
        ndarray: Local standard deviation map (0-255)
    """
    # Work with grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Convert to float
    gray_float = gray.astype(np.float32)

    # Compute local mean
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    local_mean = cv2.filter2D(gray_float, -1, kernel)

    # Compute local mean of squares
    local_mean_sq = cv2.filter2D(gray_float ** 2, -1, kernel)

    # Local variance = E[X^2] - E[X]^2
    local_var = local_mean_sq - local_mean ** 2
    local_var = np.maximum(local_var, 0)  # Handle numerical errors

    # Local standard deviation
    local_std = np.sqrt(local_var)

    # Normalize to 0-255
    local_std = cv2.normalize(local_std, None, 0, 255, cv2.NORM_MINMAX)
    local_std = local_std.astype(np.uint8)

    return local_std


def apply_bilateral_filter_exposed(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter for edge-preserving smoothing.

    This reduces noise while preserving important edges, useful as a
    pre-processing step before velocity analysis.

    Args:
        image (ndarray): Input image (grayscale or color)
        d (int): Diameter of pixel neighborhood
        sigma_color (float): Filter sigma in color space
        sigma_space (float): Filter sigma in coordinate space

    Returns:
        ndarray: Filtered image
    """
    # Work with grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply bilateral filter
    filtered = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)

    return filtered


# ============================================================================
# Frame Quality Assessment Functions
# ============================================================================


def detect_blur(image, threshold=100.0):
    """
    Detect if an image is blurry using Laplacian variance method.

    Args:
        image (ndarray): Input image
        threshold (float): Blur threshold (lower = more blurry)
                          Typical values: <100 = blurry, >100 = sharp

    Returns:
        tuple: (is_blurry (bool), blur_score (float))
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Compute Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    is_blurry = laplacian_var < threshold

    return is_blurry, laplacian_var


def analyze_exposure(image):
    """
    Analyze image exposure quality.

    Args:
        image (ndarray): Input image

    Returns:
        dict: Exposure metrics including:
            - mean_brightness: Average brightness (0-255)
            - is_underexposed: Boolean
            - is_overexposed: Boolean
            - histogram_spread: Measure of contrast
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Compute histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Compute metrics
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)

    # Check for under/over exposure
    # Underexposed if mean is low and most pixels in dark range
    dark_pixels = np.sum(hist[:50]) / gray.size
    is_underexposed = (mean_brightness < 60) or (dark_pixels > 0.5)

    # Overexposed if mean is high and most pixels in bright range
    bright_pixels = np.sum(hist[200:]) / gray.size
    is_overexposed = (mean_brightness > 200) or (bright_pixels > 0.5)

    return {
        'mean_brightness': mean_brightness,
        'std_brightness': std_brightness,
        'is_underexposed': is_underexposed,
        'is_overexposed': is_overexposed,
        'histogram_spread': std_brightness
    }


# ============================================================================
# Visualization Functions
# ============================================================================


def create_motion_heatmap(variance_map, colormap=cv2.COLORMAP_JET):
    """
    Create a color-coded motion heatmap from variance map.

    Args:
        variance_map (ndarray): Temporal variance map (grayscale)
        colormap (int): OpenCV colormap constant

    Returns:
        ndarray: Color-coded heatmap (BGR)
    """
    # Apply colormap
    heatmap = cv2.applyColorMap(variance_map, colormap)

    return heatmap


def create_texture_visualization(image, method='local_std', **kwargs):
    """
    Create texture visualization overlay.

    Args:
        image (ndarray): Input image
        method (str): Method to use ('local_std', 'dog', 'edges')
        **kwargs: Additional arguments for the chosen method

    Returns:
        ndarray: Texture visualization
    """
    if method == 'local_std':
        kernel_size = kwargs.get('kernel_size', 15)
        texture_map = apply_local_std_dev(image, kernel_size=kernel_size)
    elif method == 'dog':
        sigma1 = kwargs.get('sigma1', 1.0)
        sigma2 = kwargs.get('sigma2', 2.0)
        texture_map = apply_difference_of_gaussians(image, sigma1=sigma1, sigma2=sigma2)
    elif method == 'edges':
        alpha = kwargs.get('alpha', 1.5)
        texture_map = apply_edge_enhancement(image, alpha=alpha)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply colormap for visualization
    colored = cv2.applyColorMap(texture_map, cv2.COLORMAP_HOT)

    return colored


def overlay_roi_on_image(image, roi_mask, color=(0, 255, 255), alpha=0.3):
    """
    Overlay ROI mask on image with transparency.

    Args:
        image (ndarray): Input image (BGR or grayscale)
        roi_mask (ndarray): Binary ROI mask (0 or 1)
        color (tuple): BGR color for overlay (default: cyan)
        alpha (float): Transparency (0=transparent, 1=opaque)

    Returns:
        ndarray: Image with ROI overlay (BGR)
    """
    # Ensure image is BGR
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image.copy()

    # Create colored overlay
    overlay = image_bgr.copy()
    overlay[roi_mask == 1] = color

    # Blend with original
    result = cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)

    return result