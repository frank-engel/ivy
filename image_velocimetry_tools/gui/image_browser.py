"""IVy module for the image browser"""

import glob
import logging
import os

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel, QFileDialog, QMessageBox
from colormap import rgb2hex, rgb2hsv

from image_velocimetry_tools.graphics import AnnotationView, Instructions
from image_velocimetry_tools.image_processing_tools import (
    image_file_to_opencv_image,
    apply_clahe_to_image,
    automatic_brightness_and_contrast_adjustment,
    convert_opencv_image_to_qt_pixmap,
    ImageProcessor,
)

global icons_path
icon_path = "icons"


class ImageBrowserWidget(QLabel):
    """Subclass of QLabel for displaying image"""

    # Class attributes
    imageLoadedSignal = pyqtSignal(bool)

    def __init__(self, parent, image=None):
        """Class init

        Args:
            parent (IVyTools): The main IVyTools object
            image (QImage, optional): a supplied QImage. Defaults to None.
        """
        super().__init__(parent)
        self.image_sequence_index = 0
        self.image_sequence = []
        self.digitized_points = []
        self.parent = parent
        self.image = QImage()
        self.image_file_path = None
        self.image_folder_path = None
        self.original_image = self.image

        # self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        # self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        # self.setScaledContents(True)

        # Load image
        self.setPixmap(QPixmap().fromImage(self.image))
        self.setAlignment(Qt.AlignCenter)

        # Connect to the previous/next image signals
        parent.signal_previous_image.connect(self.loadPreviousImage)
        parent.signal_next_image.connect(self.loadNextImage)

    def openImage(self, image=None):
        """Load a new image into the"""
        if image is not None:
            image_folder = os.path.splitext(image)[0]
            self.image_folder_path = image_folder
            self.image_file_path = image
        else:
            filter = "Images (*.png *.jpg *.tif);;All files (*.*)"
            self.image_file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Image", "", filter  # path
            )

        if self.image_file_path:
            # Reset values when opening an image
            self.parent.imagebrowser_zoom_factor = 1

            # Get image format
            image_format = self.image.format()
            self.image = QImage(self.image_file_path)
            self.original_image = self.image.copy()

            self.setPixmap(QPixmap().fromImage(self.image))
            self.resize(self.pixmap().size())

            self.imageLoadedSignal.emit(True)

        elif self.image_file_path == "":
            # User selected Cancel
            pass
        else:
            QMessageBox.information(
                self, "Error", "Unable to open image.", QMessageBox.Ok
            )

    def openImageFolder(self, image_folder=None, glob_pattern=""):
        """Open a folder containing images

        Args:
            image_folder (str): path to a folder containing images
            glob_pattern (str): file filtering expression used by the glob package
        """
        self.image_sequence_index = 0
        self.image_sequence = []
        """Load a folder and create image glob"""
        if image_folder is not None:
            self.image_folder_path = image_folder
        else:
            self.image_folder_path = QFileDialog.getExistingDirectory(
                self,
                "Select a folder containing a sequence of images",
                "",
                # path
                QFileDialog.ShowDirsOnly,
            )

        if self.image_folder_path:
            # Create sorted glob of images matches filter
            types = ["*.png", "*.jpg", "*.tif"]
            if glob_pattern:
                self.image_sequence.extend(
                    sorted(
                        glob.glob(
                            self.image_folder_path + os.sep + glob_pattern
                        )
                    )
                )
            else:
                for images in types:
                    self.image_sequence.extend(
                        sorted(
                            glob.glob(self.image_folder_path + os.sep + images)
                        )
                    )

            # Select first image
            if self.image_sequence:  # no images match search
                self.image_file_path = self.image_sequence[
                    self.image_sequence_index
                ]  # path to first result of glob

                # Reset values when opening an image
                self.parent.imagebrowser_zoom_factor = 1
                self.parent.scrollareaImageBrowser.setVisible(True)

                # Get image format
                self.image = QImage(self.image_file_path)
                self.original_image = self.image.copy()

                # pixmap = QPixmap(image_file)
                self.setPixmap(QPixmap().fromImage(self.image))
                self.resize(self.pixmap().size())

                self.imageLoadedSignal.emit(True)

        elif self.image_file_path == "":
            # User selected Cancel
            pass
        else:
            QMessageBox.information(
                self, "Error", "Unable to open image.", QMessageBox.Ok
            )

    def replaceImage(self, pixmap):
        """Load a supplied QPixmap image into the browser"""

        # Reset values when opening an image
        self.parent.imagebrowser_zoom_factor = 1

        # Get image format
        self.image = pixmap.toImage()
        self.original_image = self.image.copy()
        self.setPixmap(pixmap)
        self.resize(self.pixmap().size())

        self.imageLoadedSignal.emit(True)

    def loadPreviousImage(self):
        """Load the previous image in the sequence of images"""
        idx = self.image_sequence_index - 1
        if idx >= 0:  # we still have images we can advance
            self.image_file_path = self.image_sequence[idx]
            self.image_sequence_index = idx

        self.image = QImage(self.image_file_path)

        image_format = self.image.format()

        self.original_image = self.image.copy()
        self.setPixmap(QPixmap().fromImage(self.image))
        self.repaint()

    def loadNextImage(self):
        """Load the next image in the sequence of images"""
        idx = self.image_sequence_index + 1
        if idx < len(
            self.image_sequence
        ):  # we still have images we can advance
            self.image_file_path = self.image_sequence[idx]
            self.image_sequence_index = idx

        self.image = QImage(self.image_file_path)

        image_format = self.image.format()

        self.original_image = self.image.copy()
        self.setPixmap(QPixmap().fromImage(self.image))
        self.repaint()


class ImageBrowserTab:
    """Class for managing the Image Browser tab"""

    signal_previous_image = pyqtSignal(
        int
    )  # When user clicks the previous image button
    signal_next_image = pyqtSignal(
        int
    )  # When user clicks the next image button

    def __init__(self, ivy_framework):
        """Class init

        Args:
            ivy_framework (IVyTools object): The main IVyTools object
        """
        self.ivy_framework = ivy_framework
        self.sequence_index = 0
        self.sequence = []
        self.folder_path = None
        self.image_path = ""
        self.zoom_factor = 1
        self.image = None
        self.original_image = None
        self.imageBrowser = AnnotationView()
        self.cross_section_line_exists = False
        self.cross_section_start_bank = None
        self.signal_cross_section_exists = None
        self.reload = False
        self.glob_pattern = "f*.jpg"
        self.preprocess_steps_number = 0
        self.preprocess_steps = []
        self.region_of_interest_pixels = None
        self.current_pixel = []
        self.selected_color_hex = ""

    def open_image_folder(self):
        """Open a folder of images for the image browser"""
        logging.debug(
            f"Opening folder with images with pattern: '"
            f"{self.glob_pattern}'"
        )
        self.sequence_index = 0
        self.sequence = []

        try:
            last_imagebrowser_folder_path = (
                self.ivy_framework.sticky_settings.get(
                    "last_imagebrowser_folder_path"
                )
            )
        except KeyError:
            last_imagebrowser_folder_path = None

        if not self.reload or not self.folder_path:
            self.folder_path = QtWidgets.QFileDialog.getExistingDirectory(
                self.ivy_framework,
                "Select a folder containing a sequence of images",
                last_imagebrowser_folder_path,
                QtWidgets.QFileDialog.ShowDirsOnly,
            )

        if self.folder_path:
            try:
                self.ivy_framework.sticky_settings.set(
                    "last_imagebrowser_folder_path", self.folder_path
                )
            except KeyError:
                self.ivy_framework.sticky_settings.new(
                    "last_imagebrowser_folder_path", self.folder_path
                )

            types = ["*.png", "*.jpg", "*.tif"]
            if self.glob_pattern:
                self.sequence.extend(
                    sorted(
                        glob.glob(
                            os.path.join(self.folder_path, self.glob_pattern)
                        )
                    )
                )
            else:
                for image_type in types:
                    self.sequence.extend(
                        sorted(
                            glob.glob(
                                os.path.join(self.folder_path, image_type)
                            )
                        )
                    )

            if self.sequence:
                self.image_path = self.sequence[self.sequence_index]
                self.zoom_factor = 1
                self.imageBrowser.setEnabled(True)
                self.image = QtGui.QImage(self.image_path)
                self.original_image = self.image.copy()
                self.imageBrowser.scene.setImage(self.image)
        elif not self.image_path:
            pass
        else:
            QtWidgets.QMessageBox.information(
                self.ivy_framework,
                "Error",
                "Unable to open image.",
                QtWidgets.QMessageBox.Ok,
            )
        current_image = self.image_path
        if current_image is None:
            message = (
                f"IMAGE BROWSER: No images found matching Frame Filtering "
                f"criteria. Try adjusting "
                f"Frame Filtering or loading a different folder."
            )
            self.ivy_framework.toolbuttonPreviousImage.setEnabled(False)
            self.ivy_framework.toolbuttonNextImage.setEnabled(False)
        else:
            message = f"IMAGE BROWSER: Current image: {current_image}"
            self.ivy_framework.toolbuttonPreviousImage.setEnabled(True)
            self.ivy_framework.toolbuttonNextImage.setEnabled(True)
        self.ivy_framework.update_statusbar(message)
        self.reload = False

    def apply_file_filter(self):
        """Apply a file filter to the browser"""
        self.glob_pattern = self.ivy_framework.lineeditFrameFiltering.text()

    def apply_to_this_frame(self):
        """Applies the CLAHE and/or auto-contrast filtering to the current frame."""
        if self.sequence:
            current_image = self.sequence[self.sequence_index]
            if (
                self.ivy_framework.checkboxApplyClahe.isChecked()
                or self.ivy_framework.checkboxAutoContrast.isChecked()
            ):
                cv_image = image_file_to_opencv_image(current_image)
                height, width, _ = cv_image.shape
                if self.ivy_framework.checkboxApplyClahe.isChecked():
                    clip = float(
                        self.ivy_framework.lineeditClaheClipLimit.text()
                    )
                    horz_tile_size = int(
                        self.ivy_framework.lineeditClaheHorzTileSize.text()
                    )
                    vert_tile_size = int(
                        self.ivy_framework.lineeditClaheVertTileSize.text()
                    )
                    cv_image = apply_clahe_to_image(
                        cv_image,
                        clip_size=clip,
                        horz_tile_size=horz_tile_size,
                        vert_tile_size=vert_tile_size,
                    )
                if self.ivy_framework.checkboxAutoContrast.isChecked():
                    percent = float(
                        self.ivy_framework.lineeditAutoContrastPercentClip.text()
                    )
                    cv_image, alpha, beta = (
                        automatic_brightness_and_contrast_adjustment(
                            cv_image, clip_histogram_percentage=percent
                        )
                    )

                pixmap = convert_opencv_image_to_qt_pixmap(
                    cv_image, display_width=width, display_height=height
                )
                self.imageBrowser.scene.setImage(pixmap)

            else:
                # Reset values when opening an image
                self.zoom_factor = 1
                self.imageBrowser.setEnabled(True)

                # Get image format
                current_QImage = QtGui.QImage(self.image_path)
                self.original_image = self.image.copy()

                # Set the image
                self.imageBrowser.scene.setImage(current_QImage)
                message = f"IMAGE BROWSER: Current image: {current_image}"
                self.ivy_framework.update_statusbar(message)

    def apply_to_all_frames(self):
        """Apply the image processing to all frames in the image sequence."""
        message = (
            f"IMAGE BROWSER: Applying image preprocessing to "
            f"all frames and saving outputs, please be patient."
        )
        self.ivy_framework.update_statusbar(message)
        self.ivy_framework.progressBar.show()

        # Pull parameters from the gui
        do_clahe = self.ivy_framework.checkboxApplyClahe.isChecked()
        do_auto_contrast = self.ivy_framework.checkboxAutoContrast.isChecked()
        clip = float(self.ivy_framework.lineeditClaheClipLimit.text())
        horz_tile_size = int(
            self.ivy_framework.lineeditClaheHorzTileSize.text()
        )
        vert_tile_size = int(
            self.ivy_framework.lineeditClaheVertTileSize.text()
        )
        clahe_parameters = (clip, horz_tile_size, vert_tile_size)
        auto_contrast_percent = float(
            self.ivy_framework.lineeditAutoContrastPercentClip.text()
        )

        # Keep track of the processing steps taken
        self.preprocess_steps_number += 1
        self.preprocess_steps.append(
            (
                self.preprocess_steps_number,
                do_clahe,
                clahe_parameters,
                do_auto_contrast,
                auto_contrast_percent,
            )
        )

        # Connect the progress signal
        processing_thread = ImageProcessor()
        processing_thread.progress.connect(self.preprocessor_process_progress)

        # Connect the finished signal to a slot for handling the result
        processing_thread.finished.connect(self.preprocessor_process_finished)

        # TODO: this is the actual connected call for Apply to All!
        self.ivy_framework.progressBar.setValue(0)
        self.ivy_framework.progressBar.show()
        processing_thread.preprocess_images(
            image_paths=self.sequence,
            clahe_parameters=clahe_parameters,
            auto_contrast_percent=auto_contrast_percent,
            do_clahe=do_clahe,
            do_auto_contrast=do_auto_contrast,
        )

        # # Create an instance of the ImageProcessorThread
        # processing_thread = ImageProcessorThread(
        #     self.ivy_tools.image_processor,
        #     method_name="preprocess_images",
        #     image_paths=self.sequence,
        #     clahe_parameters=clahe_parameters,
        #     auto_contrast_percent=auto_contrast_percent,
        #     do_clahe=do_clahe,
        #     do_auto_contrast=do_auto_contrast,
        # )

    def preprocessor_process_progress(self, progress):
        """Helper function to track preprocessor progress

        Args:
            progress (QProgressBar): the progress bar object
        """
        self.ivy_framework.progressBar.setValue(progress)

    def preprocessor_process_finished(self):
        """Executes when the preprocessor has finished."""
        # self.progressBar.setValue(100)
        self.ivy_framework.progressBar.hide()
        message = f"IMAGE BROWSER: Successfully preprocessed all frames."
        self.ivy_framework.update_statusbar(message)
        self.ivy_framework.progressBar.setValue(0)
        self.reload_image_folder()

    def reload_image_folder(self):
        """Reload the image folder"""
        self.apply_file_filter()
        self.reload = True
        self.open_image_folder()

    def add_mask(self):
        """Add mask to the current image"""
        if self.ivy_framework.toolbuttonCreateMask.isChecked():
            self.imageBrowser.scene.set_current_instruction(
                Instructions.POLYGON_INSTRUCTION
            )
        else:
            self.imageBrowser.scene.set_current_instruction(
                Instructions.NO_INSTRUCTION
            )
            self.ivy_framework.toolbuttonCreateMask.setChecked(False)
            self.ivy_framework.toolbuttonCreateMask.repaint()

    def clear_mask(self):
        """Clear all masks"""
        self.imageBrowser.clearPolygons()
        self.region_of_interest_pixels = None

    def save_roi(self, polygon_points):
        """Save the roi to the class attribute

        Args:
            polygon_points (ndarray): the polygon as an array of points
        """
        self.region_of_interest_pixels = polygon_points

    def editing_complete(self):
        """Triggered when editing has finsished."""
        if self.imageBrowser.drawROI == "Polygon":
            self.ivy_framework.toolbuttonCreateMask.setChecked(False)
            self.ivy_framework.toolbuttonCreateMask.repaint()
            self.imageBrowser.drawROI = None
        if self.cross_section_line_exists:
            self.toolbuttonDrawCrossSection.setChecked(False)
            self.toolbuttonDrawCrossSection.repaint()
            if self.radioButtonLeft.isChecked():
                self.cross_section_start_bank = "left"
            if self.radioButtonRight.isChecked():
                self.cross_section_start_bank = "right"
            self.signal_cross_section_exists.emit(True)

    def button_state_checker(self, calling_button: str):
        """Watch the pan and eydropper button and manage state."""
        if calling_button.lower() == "pan":
            self.ivy_framework.toolbuttonEyeDropper.setChecked(False)
            self.ivy_framework.toolbuttonEyeDropper.repaint()
            self.ivy_framework.toolbuttonEyeDropper.repaint()
        if calling_button.lower() == "eyedropper":
            self.ivy_framework.toolbuttonEyeDropper.setChecked(False)
            self.ivy_framework.toolbuttonCreatePoint.setChecked(False)

    def get_pixel(self, x, y):
        """Extract pixel information from the current frame."""
        row = int(y)
        column = int(x)
        logging.debug(
            "Clicked on image pixel (row="
            + str(row)
            + ", column="
            + str(column)
            + ")"
        )
        # x = event.pos().x() / self.ortho_original_image_zoom_factor
        # y = event.pos().y() / self.ortho_original_image_zoom_factor
        # c = self.ortho_original_image.image.pixel(x, y)
        self.current_pixel = [x, y]
        logging.debug(
            f"##### Pixel Info: x: {self.current_pixel[0]}, "
            f"y: {self.current_pixel[1]}."
        )

        image = self.imageBrowser.scene.image()
        c = image.pixelColor(x, y)
        c_rgb = QtGui.QColor(c).getRgb()  # 8-bit RGBA
        c_hex = rgb2hex(c_rgb[0], c_rgb[1], c_rgb[2])
        c_hsv = rgb2hsv(c_rgb[0], c_rgb[1], c_rgb[2], normalised=False)
        c_hsv = (
            c_hsv[0] * 360,
            c_hsv[1] * 100,
            c_hsv[2] * 100,
        )  # rgb2hsl returns normalized HSL despite docs saying otherwise
        # Set the color swatch
        self.ivy_framework.labelColorPatch.setStyleSheet(
            f"background-color: {c_hex}"
        )

        self.current_pixel = [x, y, c_rgb, c_hex, c_hsv]
        self.selected_color_hex = c_hex
        self.ivy_framework.lineeditHexColorSelection.setText(c_hex)
        self.ivy_framework.labelHSV.setText(
            f"HSV: ({c_hsv[0]:.2f}°, {c_hsv[1]:.2f}, {c_hsv[2]:.2f})"
        )
        logging.debug(
            f"##### Pixel Info: x: {self.current_pixel[0]}, "
            f"y: {self.current_pixel[1]}. "
            f"color: {self.current_pixel[2]}"
        )

    def eyedropper(self):
        """Connects the point digitizer to the eyedropper button."""
        pixmap = QtGui.QPixmap(icon_path + os.sep + "eye-dropper-solid.svg")
        pixmap = pixmap.scaledToWidth(32)
        cursor = QtGui.QCursor(pixmap, hotX=0, hotY=32)
        print(f"cursor hotspot: {cursor.hotSpot()}")
        if self.ivy_framework.toolbuttonEyeDropper.isChecked():
            # self.imagebrowser_button_state_checker("eyedropper")
            self.imageBrowser.setCursor(cursor)

            # Create the mouse event
            self.imageBrowser.leftMouseButtonReleased.connect(self.get_pixel)

        else:
            self.imageBrowser.setCursor(Qt.ArrowCursor)

    def zoom_image(self, zoom_value):
        """Zoom in and zoom out."""
        self.zoom_factor = zoom_value
        self.imageBrowser.zoomEvent(self.imagebrowser_zoom_factor)
        # self.toolbuttonZoomIn.setEnabled(self.imagebrowser_zoom_factor < 4.0)
        # self.toolbuttonZoomOut.setEnabled(self.imagebrowser_zoom_factor > 0.333)

    def normal_size(self):
        """View image with its normal dimensions."""
        self.imageBrowser.clearZoom()
        self.zoom_factor = 1.0

    def set_previous_image(self):
        """Set the previous image matching filter spec in the image display."""
        num_images = len(self.sequence)
        if num_images > 0:
            self.sequence_index -= 1
            if self.sequence_index > num_images - 1:
                self.sequence_index = num_images - 1
                self.ivy_framework.toolbuttonNextImage.setEnabled(False)
            if self.sequence_index < 0:
                self.sequence_index = 0
                self.ivy_framework.toolbuttonPreviousImage.setEnabled(False)
            if 0 < self.sequence_index < num_images - 1:
                self.ivy_framework.toolbuttonPreviousImage.setEnabled(True)
                self.ivy_framework.toolbuttonNextImage.setEnabled(True)
            # self.signal_next_image.emit(self.sequence_index)
            self.image_path = self.sequence[
                self.sequence_index
            ]  # path to first result of glob

            # Reset values when opening an image
            self.zoom_factor = 1
            self.imageBrowser.setEnabled(True)

            # Get image format
            self.image = QtGui.QImage(self.image_path)
            self.original_image = self.image.copy()

            # Set the image
            self.imageBrowser.scene.setImage(self.image)
            current_image = self.image_path
            message = f"IMAGE BROWSER: Current image: {current_image}"
            self.ivy_framework.update_statusbar(message)

    def set_next_image(self):
        """Set the next image matching filter spec in the image display."""
        num_images = len(self.sequence)
        if num_images > 0:
            self.sequence_index += 1
            if self.sequence_index > num_images - 1:
                self.sequence_index = num_images - 1
                self.ivy_framework.toolbuttonNextImage.setEnabled(False)
            if self.sequence_index < 0:
                self.sequence_index = 0
                self.ivy_framework.toolbuttonPreviousImage.setEnabled(False)
            if 0 < self.sequence_index < num_images - 1:
                self.ivy_framework.toolbuttonPreviousImage.setEnabled(True)
                self.ivy_framework.toolbuttonNextImage.setEnabled(True)

            # self.signal_next_image.emit(self.sequence_index)
            self.image_path = self.sequence[
                self.sequence_index
            ]  # path to first result of glob

            # Reset values when opening an image
            self.zoom_factor = 1
            self.imageBrowser.setEnabled(True)

            # Get image format
            self.image = QtGui.QImage(self.image_path)
            self.original_image = self.image.copy()

            # Set the image
            self.imageBrowser.scene.setImage(self.image)
            current_image = self.image_path
            message = f"IMAGE BROWSER: Current image: {current_image}"
            self.ivy_framework.update_statusbar(message)
