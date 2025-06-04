"""IVy module containing the image processor threading"""

import logging

from PyQt5.QtCore import QThread, pyqtSignal


class ImageProcessorThread(QThread):
    """Image Processing Thread class

    Args:
        QThread (QThread): the QThread
    """

    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, image_processor, method_name, *args, **kwargs):
        """Class init

        Args:
            image_processor: the image processor object
            method_name (str): the method to apply
        """
        super().__init__()
        self.image_processor = image_processor
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """Run the thread using the method specified"""
        try:
            logging.debug("ImageProcessorThread: Thread is starting its execution")
            # Check if the method exists in the ImageProcessor class
            if hasattr(self.image_processor, self.method_name):
                logging.debug(f"Executing method '{self.method_name}'")
                method = getattr(self.image_processor, self.method_name)
                method(*self.args, **self.kwargs)
            else:
                logging.debug(
                    f"ImageProcessorThread: Method '{self.method_name}' does not exist in ImageProcessor."
                )
        except Exception as e:
            logging.debug(f"ImageProcessorThread: Exception in thread: {e}")
            import traceback

            traceback.print_exc()  # Print the traceback for more details
        finally:
            logging.debug("ImageProcessorThread: Thread is about to finish")
            self.finished.emit()
