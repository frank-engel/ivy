"""IVy module containing ImageMagick related functions

Note: IVy is no longer using ImageMagick, but this module is retained for now.
"""

import logging
import os

from image_velocimetry_tools.common_functions import (
    resource_path,
    quotify_a_string,
)

# Locate the ImageMagic exes and ensure they are reachable
imagemagick_loc = "bin"
im_convert_cmd = resource_path(imagemagick_loc + os.sep + "convert")
# im_convert_cmd = os.path.join(os.path.dirname(__file__), imagemagick_loc) + os.sep + 'convert'


def create_median_image_command(images, out_folder):
    """Create a median image creation command for ImageMagick

    Args:
        images (list): list of file paths to images
        out_folder (str): output folder path

    Returns:
        _type_: _description_
    """
    images_str = " ".join(map(str, images))
    s = os.path.basename(images[0])
    e = os.path.basename(images[-1])
    name = f"!median_{os.path.splitext(s)[0]}_{os.path.splitext(e)[0]}"
    output_location = quotify_a_string(f"{out_folder}{os.sep}{name}.jpg")
    cmd = (
        im_convert_cmd
        + f" {images_str} -evaluate-sequence median {output_location}"
    )
    logging.info(cmd)
    return cmd
