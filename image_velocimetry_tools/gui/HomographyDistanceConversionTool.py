"""Ivy module for the Homography Distance Conversion Tool"""

import logging
import os

import numpy as np
from PyQt5 import QtWidgets, QtGui

from image_velocimetry_tools.common_functions import resource_path
from image_velocimetry_tools.gui.dialogs import (
    wHomographyDistanceConversionTool,
)
from image_velocimetry_tools.orthorectification import FourPointSolution


class HomographyDistanceConversionTool(
    QtWidgets.QDialog, wHomographyDistanceConversionTool.Ui_Dialog
):
    """Homography distance computation tool

    Args:
        QtWidgets (QWidget): the Dialog widget
        wHomographyDistanceConversionTool (Ui_Dialog): UI
    """

    def __init__(self, parent=None):
        """Class init

        Args:
            parent (IVyTools object, optional): The main IVyTools object. Defaults to None.
        """
        super(HomographyDistanceConversionTool, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Homography Distance Conversion Tool")

        if os.environ.get("IVY_ENV") == "development":
            icon_path = "image_velocimetry_tools\\gui\\icons"
        else:
            icon_path = "icons"
        self.setWindowIcon(
            QtGui.QIcon(resource_path(icon_path + os.sep + "IVy_logo.svg"))
        )
        self.backgroundImage.setPixmap(
            QtGui.QPixmap(
                resource_path(icon_path + os.sep + "4-point-diagram.png")
            )
        )

        self.p = None
        self.q = None
        self.r = None
        self.s = None
        self.t = None
        self.u = None
        self.coordinates = None

        self.lineeditP.editingFinished.connect(self.set_p)
        self.lineeditQ.editingFinished.connect(self.set_q)
        self.lineeditR.editingFinished.connect(self.set_r)
        self.lineeditS.editingFinished.connect(self.set_s)
        self.lineeditT.editingFinished.connect(self.set_t)
        self.lineeditU.editingFinished.connect(self.set_u)
        self.buttonComputeCoordinates.clicked.connect(self.compute_coordinates)
        self.buttonWriteCSVFile.clicked.connect(self.write_csv_file)

    def set_p(self):
        """Set p"""
        self.p = float(self.lineeditP.text())

    def set_q(self):
        """Set q"""
        self.q = float(self.lineeditQ.text())

    def set_r(self):
        """Set r"""
        self.r = float(self.lineeditR.text())

    def set_s(self):
        """Set s"""
        self.s = float(self.lineeditS.text())

    def set_t(self):
        """Set t"""
        self.t = float(self.lineeditT.text())

    def set_u(self):
        """Set u"""
        self.u = float(self.lineeditU.text())

    def compute_coordinates(self):
        """Solve the linear system of equations describing the homography coordinates."""
        distances = [self.p, self.q, self.r, self.s, self.t, self.u]
        if all(i is not None for i in distances):
            self.coordinates = FourPointSolution(
                distances
            ).get_world_coordinates()
            logging.debug(f"Coordinates computed:\n{self.coordinates}")
            i = 0
            self.labelX1Y1value.setText(
                f"({self.coordinates[i, 0]:.2f}, {self.coordinates[i, 1]:.2f}, {self.coordinates[i, 2]:.2f})"
            )
            i += 1
            self.labelX2Y2value.setText(
                f"({self.coordinates[i, 0]:.2f}, {self.coordinates[i, 1]:.2f}, {self.coordinates[i, 2]:.2f})"
            )
            i += 1
            self.labelX3Y3value.setText(
                f"({self.coordinates[i, 0]:.2f}, {self.coordinates[i, 1]:.2f}, {self.coordinates[i, 2]:.2f})"
            )
            i += 1
            self.labelX4Y4value.setText(
                f"({self.coordinates[i, 0]:.2f}, {self.coordinates[i, 1]:.2f}, {self.coordinates[i, 2]:.2f})"
            )

    def write_csv_file(self):
        """Write the homography results to a CSV file"""
        if self.coordinates is not None:
            labels = np.array([["X1Y1", "X2Y2", "X3Y3", "X4Y4"]]).T
            filter = "Comma-Separated Values (*.csv);;All files (*.*)"
            points_file, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Transformed Coordinates", "", filter  # path
            )
            if points_file:
                # Save file
                points = np.hstack((labels, self.coordinates))
                np.savetxt(
                    points_file,
                    points,
                    delimiter=",",
                    header="ID,X,Y,Z",
                    fmt="%s",
                )
                logging.info(
                    f"CSV file with point coordinates saved here: {points_file}"
                )
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    "Error",
                    "Unable to save digitized points.",
                    QtWidgets.QMessageBox.Ok,
                )
