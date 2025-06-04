"""IVy module for handling the Lens Characteristics dialog"""

import os

from PyQt5 import QtWidgets, QtGui

from image_velocimetry_tools.common_functions import resource_path
from image_velocimetry_tools.graphics import AnnotationView
from image_velocimetry_tools.gui.dialogs import wLensCharacteristics



class LensCharacteristics(QtWidgets.QDialog, wLensCharacteristics.Ui_Dialog):
    """Lens Characteristics Dialog class

    This Lens Characteristics dialog collects the needed parameters for applying a
    simplified Brown Len's model.

    Args:
        QtWidgets (QtWidget): the Dialog object
        wLensCharacteristics: the UI
    """

    global IVY_ENV
    IVY_ENV = os.environ.get("IVY_ENV")

    def __init__(self, parent=None, width=1, height=1, cx=0, cy=0, k1=0, k2=0):
        """Class init

        Args:
            parent (IVyTools, optional): The main IVyTools object. Defaults to None.
            width (int, optional): image width. Defaults to 1.
            height (int, optional): image height. Defaults to 1.
            cx (int, optional): principal point x coordinate. Defaults to 0.
            cy (int, optional): principal point y coordinate. Defaults to 0.
            k1 (int, optional): k1 coefficient. Defaults to 0.
            k2 (int, optional): k2 coefficient. Defaults to 0.
        """
        super(LensCharacteristics, self).__init__(parent)
        self.setupUi(self)

        if IVY_ENV == "development":
            self.__icon_path__ = "image_velocimetry_tools/gui/icons"
        else:
            self.__icon_path__ = "icons"

        self.setWindowTitle("Lens Characteristics")
        self.setWindowIcon(
            QtGui.QIcon(resource_path(self.__icon_path__ + os.sep + "IVy_logo.svg"))
        )
        self.lensDistortionModel = AnnotationView()
        self.lensDistortionModel.setEnabled(True)
        self.gridlayoutDistortionModel.addWidget(self.lensDistortionModel)
        # self.backgroundImage.setPixmap(QtGui.QPixmap(resource_path(icon_path + os.sep + "4-point-diagram.png")))
        # Get image format
        # self.imagebrowser_image = QtGui.QImage(self.imagebrowser_image_path)
        # self.imagebrowser_original_image = self.imagebrowser_image.copy()
        #
        # # Set the image
        # self.imageBrowser.scene.setImage(self.imagebrowser_image)
        self.lensDistortionModel_image = QtGui.QImage(
            resource_path(self.__icon_path__ + os.sep + "barrel-distortion.png")
        )
        self.lensDistortionModel.scene.setImage(self.lensDistortionModel_image)
        self._cx_raw = cx
        self._cy_raw = cy
        self._k1_raw = k1
        self._k2_raw = k2
        if width > 0:
            self.width = width
        else:
            self.width = 1
        if height > 0:
            self.height = height
        else:
            self.height = 1
        self.cx = 0.5
        self.cy = 0.5
        self.cx_dim = self.cx / width
        self.cy_dim = self.cy / height
        self.k1 = 0.0
        self.k2 = 0.0

        self.lineeditCx.editingFinished.connect(self.set_cx)
        self.lineeditCy.editingFinished.connect(self.set_cy)
        self.lineeditK1.editingFinished.connect(self.set_k1)
        self.lineeditK2.editingFinished.connect(self.set_k2)
        self.get_cx()
        self.get_cy()
        self.get_k1()
        self.get_k2()

    def get_cx(self):
        """Get cx"""
        self.lineeditCx.setText(f"{self._cx_raw}")
        self.cx = self._cx_raw
        self.cx_dim = self.cx / self.width

    def set_cx(self):
        """Set cx"""
        self.cx = float(self.lineeditCx.text())

    def get_cy(self):
        """Get cy"""
        self.lineeditCy.setText(f"{self._cy_raw}")
        self.cy = self._cy_raw
        self.cy_dim = self.cy / self.height

    def set_cy(self):
        """Set cy"""
        self.cy = float(self.lineeditCy.text())

    def get_k1(self):
        """Get k1"""
        self.lineeditK1.setText(f"{self._k1_raw}")
        self.k1 = self._k1_raw

    def set_k1(self):
        """Set k1"""
        self.k1 = float(self.lineeditK1.text())

    def get_k2(self):
        """Get k2"""
        self.lineeditK2.setText(f"{self._k2_raw}")
        self.k2 = self._k2_raw

    def set_k2(self):
        """Set k2"""
        self.k2 = float(self.lineeditK2.text())
