"""Documentation module called by IVy Tools."""

import getpass
from datetime import datetime

from PyQt5 import QtWidgets

from image_velocimetry_tools.gui.dialogs.adddocumentation_ui import Ui_Dialog


class AddDocumentation(QtWidgets.QDialog, Ui_Dialog):
    """Add Documentation Window Class

    Args:
        QtWidgets (QDialog): QDialog
        Ui_Dialog (UI): Ui_Dialog
    """

    def __init__(self, tab_name=None, parent=None):
        """Class init

        Args:
            tab_name (str, optional): name of the tab that called. Defaults to None.
            parent (IVyTool object, optional): The main IVy Tools object. Defaults to None.
        """
        super(AddDocumentation, self).__init__(parent)
        self.setupUi(self)

        self.user = getpass.getuser()
        self.user_label.setText(self.user)

        self.comment = []
        self.tab_name = tab_name
        if tab_name is not None:
            index = self.category_cb.findText(tab_name)
            if index != -1:
                self.category_cb.setCurrentIndex(index)
        self.category = self.category_cb.currentText()

        self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        self.comments_te.textChanged.connect(self.check_entries)
        self.buttonBox.accepted.connect(self.set_data)

    def check_entries(self):
        """Checks if the party and comment boxes are filled out."""
        comment = self.comments_te.toPlainText()
        if len(comment) > 0:
            self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(
                True
            )
        else:
            self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(
                False
            )

    def set_data(self):
        """Save changes to the class attributes"""
        comment = self.comments_te.toPlainText()
        date = datetime.now()
        date = date.strftime("%m/%d/%Y %H:%M:%S")

        self.comment = str(self.user + ", " + date + ", " + comment)
        self.category = self.category_cb.currentText()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = AddDocumentation()
    window.show()
    sys.exit(app.exec_())
