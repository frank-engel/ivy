from PyQt5 import QtGui, QtWidgets

from image_velocimetry_tools.gui.dialogs.settings_ui import Ui_Dialog
from image_velocimetry_tools.settings import Settings


class Settings(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, units="English", icon_path=None, parent=None):
        """Calls gui change survey_units

        Parameters
        ----------
        units: str
            English or Metric
        """

        super(Settings, self).__init__(parent)
        self.setupUi(self)

        if icon_path is not None:
            self.setWindowIcon(QtGui.QIcon(icon_path))

        self.units = None
        self.units_cb.setCurrentText(units)

        self.buttonBox.accepted.connect(self.check_settings_changes)

    def check_settings_changes(self):
        """Updates settings values based on user choices."""

        self.units = self.units_cb.currentText()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = Settings()
    window.show()
    sys.exit(app.exec_())
