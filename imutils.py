"""Simple shim for imutils package (only includes what we need)."""
import cv2


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize an image while maintaining aspect ratio.

    Args:
        image: Input image
        width: Desired width (optional)
        height: Desired height (optional)
        inter: Interpolation method

    Returns:
        Resized image
    """
    if width is None and height is None:
        return image

    (h, w) = image.shape[:2]

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
