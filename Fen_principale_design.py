# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Fen_principale_design.ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPalette


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1059, 898)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 514, 700))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_gauche = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_gauche.setContentsMargins(5, 5, 5, 10)
        self.verticalLayout_gauche.setSpacing(10)
        self.verticalLayout_gauche.setObjectName("verticalLayout_gauche")
        self.formLayout_etape1 = QtWidgets.QFormLayout()
        self.formLayout_etape1.setContentsMargins(10, -1, 10, -1)
        self.formLayout_etape1.setObjectName("formLayout_etape1")
        self.label_choisir_video = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_choisir_video.setLocale(QtCore.QLocale(QtCore.QLocale.French, QtCore.QLocale.France))
        self.label_choisir_video.setObjectName("label_choisir_video")
        self.formLayout_etape1.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_choisir_video)
        self.pushButton_parcourir = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_parcourir.setObjectName("pushButton_parcourir")
        self.formLayout_etape1.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.pushButton_parcourir)
        self.label_path = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_path.setObjectName("label_path")
        self.formLayout_etape1.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_path)
        self.lineEdit_path = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_path.setReadOnly(True)
        self.lineEdit_path.setObjectName("lineEdit_path")
        self.formLayout_etape1.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_path)
        self.label_etape1 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_etape1.setStyleSheet("font: 75 16pt \"Calibri\";")
        self.label_etape1.setObjectName("label_etape1")
        self.formLayout_etape1.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_etape1)
        self.verticalLayout_gauche.addLayout(self.formLayout_etape1)
        self.line_etape12 = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.line_etape12.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_etape12.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_etape12.setObjectName("line_etape12")
        self.verticalLayout_gauche.addWidget(self.line_etape12)
        self.formLayout_etape2 = QtWidgets.QFormLayout()
        self.formLayout_etape2.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.formLayout_etape2.setContentsMargins(10, -1, 10, -1)
        self.formLayout_etape2.setObjectName("formLayout_etape2")
        self.label_etape2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_etape2.setStyleSheet("font: 75 16pt \"Calibri\";")
        self.label_etape2.setObjectName("label_etape2")
        self.formLayout_etape2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_etape2)
        self.pushButton_consulter = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_consulter.setObjectName("pushButton_consulter")
        self.formLayout_etape2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.pushButton_consulter)
        self.pushButton_select = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_select.setObjectName("pushButton_select")
        self.formLayout_etape2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.pushButton_select)
        self.label_zi = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_zi.setObjectName("label_zi")
        self.formLayout_etape2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_zi)
        self.pushButton_supprimer = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_supprimer.setObjectName("pushButton_supprimer")
        self.formLayout_etape2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.pushButton_supprimer)
        self.verticalLayout_gauche.addLayout(self.formLayout_etape2)
        self.line_etape23 = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.line_etape23.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_etape23.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_etape23.setObjectName("line_etape23")
        self.verticalLayout_gauche.addWidget(self.line_etape23)
        self.formLayout_etape3 = QtWidgets.QFormLayout()
        self.formLayout_etape3.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.formLayout_etape3.setContentsMargins(10, -1, 10, -1)
        self.formLayout_etape3.setObjectName("formLayout_etape3")
        self.label_etape3 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_etape3.setStyleSheet("font: 75 16pt \"Calibri\";")
        self.label_etape3.setObjectName("label_etape3")
        self.formLayout_etape3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_etape3)
        self.label_algo = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_algo.setObjectName("label_algo")
        self.formLayout_etape3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_algo)
        self.comboBox_algo = QtWidgets.QComboBox(self.verticalLayoutWidget)
        self.comboBox_algo.setObjectName("comboBox_algo")
        self.comboBox_algo.addItem("")
        self.comboBox_algo.addItem("")
        self.formLayout_etape3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.comboBox_algo)
        self.verticalLayout_gauche.addLayout(self.formLayout_etape3)
        self.line_etape34 = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.line_etape34.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_etape34.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_etape34.setObjectName("line_etape34")
        self.verticalLayout_gauche.addWidget(self.line_etape34)
        self.formLayout_etape4 = QtWidgets.QFormLayout()
        self.formLayout_etape4.setContentsMargins(10, -1, 10, 10)
        self.formLayout_etape4.setObjectName("formLayout_etape4")
        self.label_etape4 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_etape4.setStyleSheet("font: 75 16pt \"Calibri\";")
        self.label_etape4.setObjectName("label_etape4")
        self.formLayout_etape4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_etape4)
        self.radioButtonNon = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.radioButtonNon.setObjectName("radioButtonNon")
        self.formLayout_etape4.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.radioButtonNon)
        self.radioButtonOui = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.radioButtonOui.setObjectName("radioButtonOui")
        self.formLayout_etape4.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.radioButtonOui)
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.formLayout_etape4.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label)
        self.pushButton_selectColor = QtWidgets.QPushButton(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_selectColor.sizePolicy().hasHeightForWidth())
        self.pushButton_selectColor.setSizePolicy(sizePolicy)
        self.pushButton_selectColor.setStyleSheet("background-color: rgb(200, 200, 200);")
        self.pushButton_selectColor.setObjectName("pushButton_lancer")
        self.formLayout_etape4.addWidget(self.pushButton_selectColor)
        self.check_suppression_operator = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.check_suppression_operator.setText("Effacer l'opérateur (Attention : ralenti considérablement le traitement)")
        self.formLayout_etape4.addWidget(self.check_suppression_operator)


        self.verticalLayout_gauche.addLayout(self.formLayout_etape4)
        self.line_etape4 = QtWidgets.QFrame(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(16)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.line_etape4.setFont(font)
        self.line_etape4.setStyleSheet("font: 75 16pt \"Agency FB\";")
        self.line_etape4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_etape4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_etape4.setObjectName("line_etape4")
        self.verticalLayout_gauche.addWidget(self.line_etape4)
        self.pushButton_lancer = QtWidgets.QPushButton(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_lancer.sizePolicy().hasHeightForWidth())
        self.pushButton_lancer.setSizePolicy(sizePolicy)
        self.pushButton_lancer.setStyleSheet("background-color: rgb(85, 170, 255);")
        self.pushButton_lancer.setObjectName("pushButton_lancer")
        self.verticalLayout_gauche.addWidget(self.pushButton_lancer)
        self.zone_frame = QtWidgets.QFrame(self.centralwidget)
        self.zone_frame.setGeometry(QtCore.QRect(540, 40, 491, 341))
        self.zone_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.zone_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.zone_frame.setObjectName("zone_frame")
        self.label_zi_img = QtWidgets.QLabel(self.zone_frame)
        self.label_zi_img.setGeometry(QtCore.QRect(0, 0, 491, 341))
        self.label_zi_img.setText("")
        self.label_zi_img.setObjectName("label_zi_img")
        self.label_zone = QtWidgets.QLabel(self.centralwidget)
        self.label_zone.setGeometry(QtCore.QRect(530, 10, 121, 16))
        self.label_zone.setObjectName("label_zone")
        self.line_middle = QtWidgets.QFrame(self.centralwidget)
        self.line_middle.setGeometry(QtCore.QRect(490, 0, 41, 841))
        self.line_middle.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_middle.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_middle.setObjectName("line_middle")
        self.line_bas = QtWidgets.QFrame(self.centralwidget)
        self.line_bas.setGeometry(QtCore.QRect(-10, 830, 1091, 20))
        self.line_bas.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_bas.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_bas.setObjectName("line_bas")
        self.line_couleur = QtWidgets.QFrame(self.centralwidget)
        self.line_couleur.setGeometry(QtCore.QRect(510, 390, 561, 16))
        self.line_couleur.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_couleur.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_couleur.setObjectName("line_couleur")
        self.plainTextEdit_histoire = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_histoire.setGeometry(QtCore.QRect(530, 450, 435, 120))
        self.plainTextEdit_histoire.setObjectName("plainTextEdit_histoire")
        self.label_histoire = QtWidgets.QLabel(self.centralwidget)
        self.label_histoire.setGeometry(QtCore.QRect(530, 420, 191, 16))
        self.label_histoire.setObjectName("label_histoire")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1059, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Quantificateur de stresse de volaille"))
        self.label_choisir_video.setText(_translate("MainWindow", "Choisir une vidéo:    "))
        self.pushButton_parcourir.setText(_translate("MainWindow", "Parcourir"))
        self.label_path.setText(_translate("MainWindow", "Path de vidéo: "))
        self.label_etape1.setText(_translate("MainWindow", "Etape 1"))
        self.label_etape2.setText(_translate("MainWindow", "Etape 2"))
        self.pushButton_consulter.setText(_translate("MainWindow", "Consulter la zone d\'interêt"))
        self.pushButton_select.setText(_translate("MainWindow", "Séléctionner une zone d\'interêt"))
        self.label_zi.setText(_translate("MainWindow", "Zone d\'interêt:       "))
        self.pushButton_supprimer.setText(_translate("MainWindow", "Supprimer la zone d\'interêt"))
        self.label_etape3.setText(_translate("MainWindow", "Etape 3"))
        self.label_algo.setText(_translate("MainWindow", "Choisir l\'algorithme: "))
        self.comboBox_algo.setItemText(0, _translate("MainWindow", "Distance"))
        self.comboBox_algo.setItemText(1, _translate("MainWindow", "Flot optique"))
        self.label_etape4.setText(_translate("MainWindow", "Etape 4"))
        self.radioButtonOui.setText(_translate("MainWindow", "Oui"))
        self.radioButtonNon.setText(_translate("MainWindow", "Non"))
        self.label.setText(_translate("MainWindow", "Détecter opérateur:   "))
        self.pushButton_lancer.setText(_translate("MainWindow", "Lancer le traitement"))
        self.label_zone.setText(_translate("MainWindow", "Zone d\'interêt:"))
        self.label_histoire.setText(_translate("MainWindow", "Histoire d\'exécution:"))
