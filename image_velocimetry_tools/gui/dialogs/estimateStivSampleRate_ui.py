# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'estimateStivSampleRate.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(422, 256)
        font = QtGui.QFont()
        font.setPointSize(12)
        Dialog.setFont(font)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setHorizontalSpacing(12)
        self.formLayout.setVerticalSpacing(6)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.doubleSpinBoxFrameRate = QtWidgets.QDoubleSpinBox(Dialog)
        self.doubleSpinBoxFrameRate.setMaximum(200.0)
        self.doubleSpinBoxFrameRate.setProperty("value", 29.97)
        self.doubleSpinBoxFrameRate.setObjectName("doubleSpinBoxFrameRate")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBoxFrameRate)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.doubleSpinBoxRiverVelocity = QtWidgets.QDoubleSpinBox(Dialog)
        self.doubleSpinBoxRiverVelocity.setProperty("value", 6.0)
        self.doubleSpinBoxRiverVelocity.setObjectName("doubleSpinBoxRiverVelocity")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBoxRiverVelocity)
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.doubleSpinBoxPixelGSD = QtWidgets.QDoubleSpinBox(Dialog)
        self.doubleSpinBoxPixelGSD.setDecimals(3)
        self.doubleSpinBoxPixelGSD.setProperty("value", 0.15)
        self.doubleSpinBoxPixelGSD.setObjectName("doubleSpinBoxPixelGSD")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBoxPixelGSD)
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.doubleSpinBoxSampleSeconds = QtWidgets.QDoubleSpinBox(Dialog)
        self.doubleSpinBoxSampleSeconds.setDecimals(3)
        self.doubleSpinBoxSampleSeconds.setObjectName("doubleSpinBoxSampleSeconds")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBoxSampleSeconds)
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.spinBoxFrameStep = QtWidgets.QSpinBox(Dialog)
        self.spinBoxFrameStep.setObjectName("spinBoxFrameStep")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.spinBoxFrameStep)
        self.verticalLayout.addLayout(self.formLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept) # type: ignore
        self.buttonBox.rejected.connect(Dialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Settings"))
        self.label.setText(_translate("Dialog", "Video Frame Rate:"))
        self.label_2.setText(_translate("Dialog", "Estimated River Velocity (ft/s):"))
        self.label_3.setText(_translate("Dialog", "Pixel GSD (ft):"))
        self.label_4.setText(_translate("Dialog", "Optimum Sample Rate (seconds):"))
        self.label_5.setText(_translate("Dialog", "Optimum Frame Step:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
