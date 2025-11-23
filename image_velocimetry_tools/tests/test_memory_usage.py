import unittest
import cv2
import time
import glob
from pympler import asizeof
from image_velocimetry_tools.common_functions import *
from image_velocimetry_tools.image_processing_tools import (
    create_grayscale_image_stack,
)


class TestCreateGrayscaleImageStack(unittest.TestCase):
    # Define a setup method to prepare any necessary resources or data for the tests
    def setUp(self):
        # Create a temporary directory to store test images
        self.temp_dir = "test_images"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.map_file_path = f"{self.temp_dir}{os.sep}image_stack.dat"

        # Load some images that should take up sufficient memory
        self.image_paths = glob.glob("./img_seq_welton_main_drain/*.jpg")

    # Define a teardown method to clean up after the tests
    def tearDown(self):
        # Remove the temporary directory and test images
        for file in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, file)
            os.remove(file_path)
        os.rmdir(self.temp_dir)

    def test_memory_usage(self):
        image_paths = self.image_paths
        print(
            f"Total number of images that will be stacked: {len(image_paths)}"
        )

        first_image = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
        height, width = first_image.shape
        print(f"The image resolution is {width}x{height} pixels per image")

        # Measure the memory usage before calling the function
        initial_memory_usage = asizeof.asizeof(image_paths)

        # Call the function and measure the memory usage after
        image_stack = create_grayscale_image_stack(image_paths)
        final_memory_usage = asizeof.asizeof(image_stack)
        print(
            f"Memory usage of image_stack: "
            f"{final_memory_usage / (1024 ** 3):.2f} GB"
        )

        # Ensure that the memory usage increased significantly due to the image_stack
        # Adjust the threshold as needed based on your specific test case
        memory_increase = final_memory_usage - initial_memory_usage
        self.assertTrue(
            memory_increase > 1000000
        )  # For example, check if memory increased by at least 1 MB

    def test_image_stack_creation_time(self):
        # Start measuring time
        start_time = time.time()

        # Call the function to create the image_stack
        image_stack = create_grayscale_image_stack(self.image_paths)
        memory_usage = asizeof.asizeof(image_stack)
        print(
            f"    Size of image_stack in memory: "
            f"{memory_usage / (1024 ** 3):.2f} GB"
        )

        # End measuring time
        end_time = time.time()

        # Calculate the elapsed time in seconds
        elapsed_time = end_time - start_time

        # Print the time it took to create the image_stack
        print(f"Time to create image_stack: {elapsed_time:.2f} seconds")

        # Optionally, you can assert that the creation time is within an acceptable range
        # For example, check if it took less than 10 seconds
        self.assertLess(elapsed_time, 15.0)

    def test_image_stack_creation_time_with_memory_map(self):
        # Start measuring time
        start_time = time.time()

        map_file_path = self.map_file_path

        # Call the function to create the image_stack
        image_stack = create_grayscale_image_stack(
            self.image_paths, map_file_path=map_file_path
        )
        memory_usage = asizeof.asizeof(image_stack)
        print(
            f"    Size of image_stack in memory (mapped version): "
            f"{memory_usage / (1024 ** 3):.2f} GB"
        )

        # End measuring time
        end_time = time.time()

        # Calculate the elapsed time in seconds
        elapsed_time = end_time - start_time

        # Print the time it took to create the image_stack
        print(
            f"Time to create image_stack with memory_map: {elapsed_time:.2f} seconds"
        )

        # Optionally, you can assert that the creation time is within an acceptable range
        # For example, check if it took less than 10 seconds
        self.assertLess(elapsed_time, 15.0)
