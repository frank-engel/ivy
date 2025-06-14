# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'adddocumentation.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(897, 302)
        font = QtGui.QFont()
        font.setPointSize(12)
        Dialog.setFont(font)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.user_label = QtWidgets.QLabel(Dialog)
        self.user_label.setObjectName("user_label")
        self.gridLayout.addWidget(self.user_label, 1, 2, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 3, 0, 1, 1)
        self.comments_te = QtWidgets.QTextEdit(Dialog)
        self.comments_te.setObjectName("comments_te")
        self.gridLayout.addWidget(self.comments_te, 2, 2, 2, 1)
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.category_cb = QtWidgets.QComboBox(Dialog)
        self.category_cb.setObjectName("category_cb")
        self.category_cb.addItem("")
        self.category_cb.addItem("")
        self.category_cb.addItem("")
        self.category_cb.addItem("")
        self.category_cb.addItem("")
        self.category_cb.addItem("")
        self.category_cb.addItem("")
        self.category_cb.addItem("")
        self.category_cb.addItem("")
        self.category_cb.addItem("")
        self.gridLayout.addWidget(self.category_cb, 0, 2, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout_2.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept) # type: ignore
        self.buttonBox.rejected.connect(Dialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Add Documentation"))
        self.label_2.setText(_translate("Dialog", "Comments:"))
        self.label.setText(_translate("Dialog", "Party:"))
        self.user_label.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">User</span></p></body></html>"))
        self.label_3.setText(_translate("Dialog", "Category"))
        self.category_cb.setToolTip(_translate("Dialog", "Choose a category to assign the comment."))
        self.category_cb.setItemText(0, _translate("Dialog", "Video Preprocessing"))
        self.category_cb.setItemText(1, _translate("Dialog", "Image Frame Processing"))
        self.category_cb.setItemText(2, _translate("Dialog", "Orthorectification"))
        self.category_cb.setItemText(3, _translate("Dialog", "Cross-Section Geometry"))
        self.category_cb.setItemText(4, _translate("Dialog", "Grid Preparation"))
        self.category_cb.setItemText(5, _translate("Dialog", "Space-Time Image Velocimetry (Exhaustive)"))
        self.category_cb.setItemText(6, _translate("Dialog", "Space-Time Image Results"))
        self.category_cb.setItemText(7, _translate("Dialog", "Discharge"))
        self.category_cb.setItemText(8, _translate("Dialog", "System"))
        self.category_cb.setItemText(9, _translate("Dialog", "Other"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
