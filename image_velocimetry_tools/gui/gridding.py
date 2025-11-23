"""IVy module for handling gridding in the application."""

import logging

import numpy as np
from PyQt5 import QtGui
from PyQt5.QtGui import QImage

from image_velocimetry_tools.graphics import AnnotationView, Instructions
from image_velocimetry_tools.services import GridService

global icons_path
icon_path = "icons"


class GridPreparationTab:
    """Class for managing the Grid Preparation Tab"""

    def __init__(self, ivy_framework):
        """Class init

        Args:
            ivy_framework (IVyTools Object): The main IVy object
        """
        self.ivy_framework = ivy_framework
        self.image_path = ""
        self.zoom_factor = 1
        self.image = None
        self.original_image = None
        self.imageBrowser = AnnotationView()
        self.reload = False
        self.glob_pattern = ""
        self.region_of_interest_pixels = None
        self.current_pixel = []
        self.selected_color_hex = ""

    def add_point(self):
        """Add a single point."""
        if self.ivy_framework.toolbuttonCreatePoint.isChecked():
            # Recreate any lines/points, respecting existing masks
            self.clear_point()
            self.clear_line()
            self.imageBrowser.scene.set_current_instruction(
                Instructions.POINT_INSTRUCTION
            )
        else:
            self.imageBrowser.scene.set_current_instruction(
                Instructions.NO_INSTRUCTION
            )
            self.ivy_framework.toolbuttonCreatePoint.setChecked(False)
            self.ivy_framework.toolbuttonCreatePoint.repaint()

    def add_line(self):
        """Add a line comprising two vertices"""
        if self.ivy_framework.toolbuttonCreateLine.isChecked():
            # Recreate any lines/points, respecting existing masks
            self.clear_point()
            self.clear_line()
            self.imageBrowser.scene.set_current_instruction(
                Instructions.SIMPLE_LINE_INSTRUCTION
            )
        else:
            self.imageBrowser.scene.set_current_instruction(
                Instructions.NO_INSTRUCTION
            )
            self.ivy_framework.toolbuttonCreatePoint.setChecked(False)
            self.ivy_framework.toolbuttonCreatePoint.repaint()

    def add_line_of_given_length(self):
        """Add a lane of a given length."""
        if self.ivy_framework.toolbuttonCreateXsLine.isChecked():
            # Recreate any lines/points, respecting existing masks
            self.clear_point()
            self.clear_line()
            if self.ivy_framework.rectified_xs_image.scene.line_item:
                xs_line = (
                    self.ivy_framework.rectified_xs_image.scene.line_item[-1]
                )
            else:
                return
            line_eps = np.array(
                [
                    [xs_line.m_points[0].x(), xs_line.m_points[0].y()],
                    [xs_line.m_points[1].x(), xs_line.m_points[1].y()],
                ]
            )
            self.imageBrowser.scene.set_current_instruction(
                Instructions.ADD_LINE_BY_POINTS,
                length=self.ivy_framework.cross_section_length_pixels,
                points=line_eps,
            )
            self.imageBrowser.scene.line_item[-1].setPen(
                QtGui.QPen(QtGui.QColor("yellow"), 5)
            )
        else:
            self.imageBrowser.scene.set_current_instruction(
                Instructions.NO_INSTRUCTION
            )
            self.ivy_framework.toolbuttonCreateMask.setChecked(False)
            self.ivy_framework.toolbuttonCreateMask.repaint()

    def add_mask(self):
        """Add a mask polygon."""
        if self.ivy_framework.toolbuttonCreateMask.isChecked():
            self.imageBrowser.scene.set_current_instruction(
                Instructions.POLYGON_INSTRUCTION
            )
        else:
            self.imageBrowser.scene.set_current_instruction(
                Instructions.NO_INSTRUCTION
            )
            self.ivy_framework.toolbuttonCreateMask.setChecked(False)
            self.ivy_framework.toolbuttonCreatePoint.repaint()

    def clear_point(self):
        """Clear all points from the object."""
        self.imageBrowser.clearPoints()

    def clear_line(self):
        """Clear all lines from the object."""
        self.imageBrowser.clearLines()
        self.clear_point()

    def clear_mask(self):
        """Clear all masks from the object."""
        self.imageBrowser.clearPolygons()
        self.region_of_interest_pixels = None

    def save_roi(self, polygon_points):
        """Save the polygon region to the object.

        Args:
            polygon_points (ndarray): _description_
        """
        self.region_of_interest_pixels = polygon_points

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


class GridGenerator:
    """Class for generating a grid"""

    def __init__(self, ivy_framework):
        """Class init

        Args:
            ivy_framework (IVyTools object): The main IVyTools objects
        """
        self.ivy_framework = ivy_framework
        self.grid_service = GridService()
        self.image = None
        self.height = None
        self.width = None
        self.horz_grid_size = self.ivy_framework.horz_grid_size
        self.vert_grid_size = self.ivy_framework.vert_grid_size
        self.number_grid_points_along_line = (
            self.ivy_framework.number_grid_points_along_line
        )
        self.number_grid_points_along_xs_line = (
            self.ivy_framework.number_grid_points_along_xs_line
        )
        self.line_mode = self.ivy_framework.line_mode
        self.grid = None
        self.binary_mask = None
        self.mask_polygons = []

    def set_image(self, image):
        """Set the image in the object

        Args:
            image (QImage): the image
        """
        self.image = image
        self.height, self.width, _ = self.image.shape

    def make_grid(self, image, mask_polygons):
        """Make the computational grid.

        Args:
            image (QImage): the image
            mask_polygons (ndarray): polygons that will be used to screen points from the grid or line.

        """
        self.set_image(image)
        self.mask_polygons = mask_polygons

        # Use GridService to generate grid with mask
        self.grid, self.binary_mask = self.grid_service.generate_regular_grid(
            self.width,
            self.height,
            self.vert_grid_size,
            self.horz_grid_size,
            mask_polygons=mask_polygons,
            clean_mask=True,
        )
        logging.debug(f"Grid generated using GridService")
        self.enable_image_velocimetry_tabs()
        return self.grid

    def make_line(
        self, image, line_start, line_end, num_points, mask_polygons
    ):
        """
        Generate evenly spaced points along a line in unmasked regions of the image.

        Parameters:
        - image (numpy.ndarray): The image for which to generate points along the line.
        - line_start (numpy.ndarray): Pixel coordinates of the start point on the line.
        - line_end (numpy.ndarray): Pixel coordinates of the end point on the line.
        - mask_polygons (list): List of polygons to create a binary mask.

        Returns:
        - numpy.ndarray: An array of pixel locations corresponding to points along the line.
        """
        self.set_image(image)
        self.mask_polygons = mask_polygons

        # Use GridService to generate line grid with mask
        line_points, self.binary_mask = self.grid_service.generate_line_grid(
            self.width,
            self.height,
            line_start,
            line_end,
            num_points,
            mask_polygons=mask_polygons,
            clean_mask=True,
        )
        logging.debug(f"Line Grid generated using GridService")
        self.enable_image_velocimetry_tabs()
        self.grid = line_points
        self.ivy_framework.set_tab_icon("tabGridPreparation", "good")
        self.ivy_framework.is_cross_section_grid = True
        self.ivy_framework.enable_disable_tabs(
            self.ivy_framework.tabWidget, "tabImageVelocimetry", True
        )
        return line_points

    def enable_image_velocimetry_tabs(self):
        """After a grid is made, enable the image velocimetry tabs"""
        self.ivy_framework.tabWidget_ImageVelocimetryMethods.setEnabled(True)
        self.ivy_framework.tabSpaceTimeImageReview.setEnabled(True)
        # Enable the Image Velocimetry Tabs
        to_enable = [
            "groupboxSpaceTimeParameters",
            "groupboxSpaceTimeOptParameters",
            "pushbuttonExportProjectedFrames",
            "buttonSTIVProcessVelocities",
            "buttonSTIVOptProcessVelocities",
        ]
        self.ivy_framework.set_qwidget_state_by_name(to_enable, True)
