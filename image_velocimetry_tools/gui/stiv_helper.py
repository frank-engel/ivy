"""IVy module used to manage the STIV helper dialog

The STIV Helper tool applies the logic described in the 
discussion of our paper (Legleiter et al., 2024) and attempts
to recommend the best STIV parameterization that optimizes the
algorithm. 
"""

from PyQt5 import QtGui, QtWidgets

from image_velocimetry_tools.gui.dialogs.estimateStivSampleRate_ui import Ui_Dialog
from image_velocimetry_tools.stiv import optimum_stiv_sample_time


class StivHelper(QtWidgets.QDialog, Ui_Dialog):
    """The main class to manage the STIV Helper Dialog

    Args:
        QtWidgets (QWidget): The dialog widget
        Ui_Dialog: the UI object
    """

    def __init__(self, frame_rate=30, gsd=0.15, icon_path=None, parent=None):
        """Class init

        Args:
            frame_rate (int, optional): video framerate. Defaults to 30.
            gsd (float, optional): ground scale distance. Defaults to 0.15.
            icon_path (str, optional): path to the IVy icon. Defaults to None.
            parent (IVyTools object, optional): The main IVyTools object. Defaults to None.
        """
        super(StivHelper, self).__init__(parent)
        self.setupUi(self)

        if icon_path is not None:
            self.setWindowIcon(QtGui.QIcon(icon_path))

        self.vid_frame_rate = frame_rate
        self.vid_ms = 1 / 29.97 * 1000
        self.gsd = gsd
        self.velocity = 6
        self.sample_time_ms = None
        self.sample_time_s = None
        self.frame_step = None
        self.compute()

        self.doubleSpinBoxFrameRate.setValue(self.vid_frame_rate)
        self.doubleSpinBoxPixelGSD.setValue(self.gsd)

        self.spinBoxFrameStep.setEnabled(False)
        self.doubleSpinBoxSampleSeconds.setEnabled(False)

        # Connections
        self.doubleSpinBoxFrameRate.editingFinished.connect(self.frame_rate)
        self.doubleSpinBoxRiverVelocity.editingFinished.connect(self.flow_velocity)
        self.doubleSpinBoxPixelGSD.editingFinished.connect(self.ground_scale)
        self.buttonBox.accepted.connect(self.accept)

    def compute(self):
        """Compute the optimized STIV parameters"""
        self.video_sample_rate()
        self.sample_time_ms = optimum_stiv_sample_time(self.gsd, self.velocity)
        self.frame_step = round(self.sample_time_ms / self.vid_ms)
        self.sample_time_s = self.frame_step * self.vid_ms / 1000

        # Adjust frame step to be at least 1
        if self.frame_step < 1:
            self.frame_step = 1

        self.update_ui()

    def update_ui(self):
        """Update the dialog values"""
        self.doubleSpinBoxSampleSeconds.setValue(self.sample_time_s)
        self.spinBoxFrameStep.setValue(self.frame_step)

    def video_sample_rate(self):
        """Compute the video sample rate in millisections"""
        self.vid_ms = 1 / self.vid_frame_rate * 1000

    def frame_rate(self):
        """Get video framerate from the UI and compute sample rate"""
        self.vid_frame_rate = self.doubleSpinBoxFrameRate.value()
        self.compute()

    def flow_velocity(self):
        """Set flow velocity"""
        self.velocity = self.doubleSpinBoxRiverVelocity.value()
        self.compute()

    def ground_scale(self):
        """Set ground scale distance"""
        self.gsd = self.doubleSpinBoxPixelGSD.value()
        self.compute()

    def accept(self):
        """Accept the result"""
        pass
        # self.compute()
        # super(StivHelper, self).accept()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = StivHelper()
    window.show()
    sys.exit(app.exec_())
