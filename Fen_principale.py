import Fen_principale_design
from PyQt5 import QtGui, QtWidgets
import os
from PyQt5.QtWidgets import QApplication, QProgressDialog
from ZoneInteret import *
from PIL import Image
import affichage_resultat
import threading
import algo_distance
import algo_flots_optiques
import remove_operateur
import time
import numpy as np
import random as rd


def thread(video, unAlgo, frame, pd, pickedColor=None, supprOp=None):


    """
        Cette méthode permet de créer un job avec un vidéo, un algo et le frame à commencer.
        Ce job peut être utilisé dans un thread pour gagner la vitesse de calcul

        :param video: la vidéo à traiter
        :param unAlgo: l'algo
        :param frame: le frame à commencer
        :return:
    """


    """
        Traitement de la video pour obtenir une liste de points
        qui peuvent etre dessine dans une courbe
    """
    time.sleep(2)
    pd.setValue(20)
    time.sleep(2)
    if rd.randint(0, 9) <= 2:

        pd.setValue(30)

    elif rd.randint(0, 9) <= 5:
        pd.setValue(40)
    else:
        pd.setValue(50)

    if pickedColor is None:
        ma_liste = unAlgo.traiterVideo(video, frame)
    else:
        ma_liste = unAlgo.traiterVideo(video, frame, pickedColor, supprOp)

    pd.setValue(99)
    time.sleep(2)
    pd.close()

    # Affichage de Fournier
    affichageFourrier(video, ma_liste)
    # Affichage du resultat
    pomme = affichage_resultat.affichage_graphique(video, frame)
    pomme.afficher(ma_liste)

def affichageFourrier(video, ma_liste):
        ma_liste2 = list()
        # Number of samplepoints
        cap2 = cv2.VideoCapture(video)
        cap2.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        videoTotalDuration = cap2.get(cv2.CAP_PROP_POS_MSEC)

        for i in range(len(ma_liste)):
            ma_liste2.append(ma_liste[i])

        alpha = 10

        nc = len(ma_liste2)
        dt = 0.1
        tmax = (nc - 1) * dt + nc * dt
        tmin = 0

        # definition d'un signal
        x = ma_liste2
        t = np.linspace(tmin, tmax, nc)

        plt.subplot(411)
        plt.plot(t, x)

        a = x

        A = np.fft.fft(a)
        # on effectue un fftshift pour positionner la frequence zero au centre
        X = dt * np.fft.fftshift(A)

        # calcul des frequences avec fftfreq
        n = t.size
        freq = np.fft.fftfreq(n, d=dt)
        f = np.fft.fftshift(freq)

        # comparaison avec la solution exacte
        plt.subplot(413)
        plt.plot(f, np.real(X), label="fft")
        plt.legend()

        plt.show()

class Fen_principale(QtWidgets.QMainWindow, Fen_principale_design.Ui_MainWindow):

    """
    Cette classe hérite la classe de Fen_principale_design pour enrichir les fonctionnalités de la fenêtre principale.
    Toutes les opérations et signals sont définis dans cette classe.

    @author Boyang Wang
    @version 2.0
    """

    def __init__(self):
        """
        Le constructeur pour créer les signals et la création de UI
        """

        QtWidgets.QMainWindow.__init__(self)
        Fen_principale_design.Ui_MainWindow.__init__(self)
        self.lowH = 0
        self.highH = 0
        self.start_frame = 3
        self.picked_color = QtGui.QColor(170, 170, 100)

        # les widgets
        self.setupUi(self)
        self.group = QtWidgets.QButtonGroup()
        self.group.addButton(self.radioButtonOui)
        self.group.addButton(self.radioButtonNon)
        self.group.setId(self.radioButtonOui, 0)
        self.group.setId(self.radioButtonNon, 1)
        self.radioButtonNon.setChecked(True)
        self.pushButton_selectColor.setDisabled(True)
        self.check_suppression_operator.setDisabled(True)
        self.radioButtonOui.setDisabled(True)
        self.radioButtonNon.setChecked(True)
        self.radioButtonNon.setDisabled(True)
        self.plainTextEdit_histoire.setReadOnly(True)
        self.comboBox_algo.currentIndexChanged.connect(self.comboBoxClicked)


        # les signaux
        self.pushButton_parcourir.clicked.connect(self.parcourir_clicked)
        self.pushButton_select.clicked.connect(self.zone_interet_select)
        self.pushButton_consulter.clicked.connect(self.zone_interet_consulter)
        self.pushButton_supprimer.clicked.connect(self.zone_interet_supprimer)
        self.pushButton_lancer.clicked.connect(self.on_myButton_clicked)
        self.pushButton_selectColor.clicked.connect(self.openColorDialog)
        self.radioButtonOui.clicked.connect(self.radioClicked)
        self.radioButtonNon.clicked.connect(self.radioClicked)

    def comboBoxClicked(self):
        if self.comboBox_algo.itemText(self.comboBox_algo.currentIndex()) == "Distance":
            self.pushButton_selectColor.setDisabled(True)
            self.radioButtonOui.setDisabled(True)
            self.radioButtonNon.setChecked(True)
            self.radioButtonNon.setDisabled(True)
            self.check_suppression_operator.setDisabled(True)
        else:
            self.radioButtonOui.setDisabled(False)
            self.radioButtonNon.setDisabled(False)

    def radioClicked(self):
        if self.radioButtonOui.isChecked():
            self.pushButton_selectColor.setDisabled(False)
            self.check_suppression_operator.setDisabled(False)
            self.pushButton_selectColor.setStyleSheet("background-color: rgb(" + str(self.picked_color.getRgb()[0]) +
                                                      "," + str(self.picked_color.getRgb()[1]) +
                                                      "," + str(self.picked_color.getRgb()[2]) + ");")
        else:
            self.pushButton_selectColor.setDisabled(True)
            self.check_suppression_operator.setDisabled(True)
            self.pushButton_selectColor.setStyleSheet("background-color: rgb(200, 200, 200);")


    def openColorDialog(self):
        col_dialog = QtWidgets.QColorDialog(self)
        self.picked_color = col_dialog.getColor(self.picked_color)
        if self.picked_color.isValid():
            self.pushButton_selectColor.setStyleSheet("background-color: rgb(" + str(self.picked_color.getRgb()[0]) +
                                                      "," + str(self.picked_color.getRgb()[1]) +
                                                      "," + str(self.picked_color.getRgb()[2]) + ");")



    def parcourir_clicked(self):
        """
        Cette méthode permet de gérer la clique sur le bouton parcourir pour choisir une vidéo
        :param:
        :returns:
        """
        #ouvre le repertoire du projet et nous permet de selectinner une video
        filename, types = QtWidgets.QFileDialog.getOpenFileName()
        #enregistre le chemin de la video
        self.lineEdit_path.setText(str(filename))


    def zone_interet_select(self):
        """
            La méthode pour gérer la clique sur le bouton Selectionner une Zone Interet
            Il faut que la vidéo soit  choisi en avance
            Dans la fenêtre principale en haut à droite affiche la zone interêt choisie
            :param:
            :return:
        """
        #verifie le chemin de la video
        if(os.path.exists(self.lineEdit_path.text())):
            zi = ZoneInteret(self.lineEdit_path.text())
            #affichage d'une fenetre permettant de choisir la zone d'interer
            if ZoneInteret.verifier_presence_fichier_ini() and zi.flag:
                img = Image.open('zi/image_zone_interet.png')
                img_resize = img.resize((self.label_zi_img.width(), self.label_zi_img.height()))
                img_resize.save('zi/image_zone_interet_temp.png')
                pixmap = QtGui.QPixmap('zi/image_zone_interet_temp.png')
                self.label_zi_img.setPixmap(pixmap)
                os.remove('zi/image_zone_interet_temp.png')
        else:
            QMessageBox.warning(self, "Erreur", "Impossible de trouver la video",
                                QMessageBox.Ok)


    def zone_interet_consulter(self):
        """
            Le méthode pour gérer la clique sur Consulter une zone interet
            Si les fichiers de zone d'interêt sont présents, dans la fenêtre principale en haut à droite affiche la zone interêt choisie
            :param:
            :return:
        """

        # Presence du fichier "param.ini"
        if ZoneInteret.verifier_presence_fichier_ini():
            img = Image.open('zi/image_zone_interet.png')
            img_resize = img.resize((self.label_zi_img.width(), self.label_zi_img.height()))
            img_resize.save('zi/image_zone_interet_temp.png')
            pixmap = QtGui.QPixmap('zi/image_zone_interet_temp.png')
            self.label_zi_img.setPixmap(pixmap)
            os.remove('zi/image_zone_interet_temp.png')
        else:
            QMessageBox.warning(self, "Erreur", "Impossible de trouver les fichiers dans le repertoire /zi",
                                QMessageBox.Ok)

    def zone_interet_supprimer(self):
        """
            La methode pour gerer le clique sur Supprimer une zone interet
            Les fichiers de zone interêt vont être supprimes
            :param:
            :return:
        """
        #Message de warning
        button = QMessageBox.question(self, "Question",
                                      "Etes-vous sûr de vouloir supprimer la Zone d'intérêt actuelle ？",
                                      QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok)
        #Suppression
        if button == QMessageBox.Ok:
            ZoneInteret.supprimer_ZI(self)
            self.label_zi_img.clear()


    def on_myButton_clicked(self):
        """
        La méthode pour gère l'appui du bouton pour lancer le traitement
        Cette méthode permet de passer l'algo choisi, la vidéo et la zone interêt à un thread pour traiter
        Si Detection Opérateur est choisi, le programme va tout d'abord supprimer l'opérateur dans la vidéo.
        C'est-à-dire qu'on enlève les frames où l'opérateur se présent.
        :return:
        """

        # Choix de l'algorithme
        self.plainTextEdit_histoire.setPlainText("")
        algo = self.comboBox_algo.currentIndex() + 1

        # obtenir la vidéo
        video_name = self.lineEdit_path.text()

        # Si la vidéo existe, on lance un autre thread en exécutant le bon algo
        if (os.path.exists(video_name)):

            pd = QProgressDialog("Operation in progress.", "Cancel", 0, 100)
            pd.setWindowTitle('En cours')
            pd.show()
            pd.setValue(10)

            pickedColor = self.picked_color
            if self.radioButtonNon.isChecked():
                pickedColor = None
            # Algorithme Distance
            if (algo == 1):
                self.plainTextEdit_histoire.insertPlainText("\n" + "Application de l'algorithme Distances...")
                try:


                    a = threading.Thread(None, thread, None, (),
                                         {'video': video_name, 'unAlgo': algo_distance.algo_distance(),
                                          'frame': self.start_frame, 'pd':pd, 'pickedColor': pickedColor})
                   
                    a.start()


                except:
                    QMessageBox.warning(self, "Erreur", "Erreurs lors de l'exécution",
                                        QMessageBox.Ok)
                self.plainTextEdit_histoire.insertPlainText("\n" + "Le résultat est enregistré dans le répertoire /resultats sous format PDF.")

                # Algorithme flotsoptiques
            elif (algo == 2):
                self.plainTextEdit_histoire.insertPlainText("\n" + "Application de l'algorithme flots optiques...")
                try:
                    a = threading.Thread(None, thread, None, (), {'video': video_name,
                                                                  'unAlgo': algo_flots_optiques.flot_optiques(),
                                                                  'frame': self.start_frame,
                                                                  'pd':pd,
                                                                  'pickedColor': pickedColor,
                                                                  'supprOp': self.check_suppression_operator.isChecked()
                                                                  })
                    a.start()
                except:
                    QMessageBox.warning(self, "Erreur", "Erreurs lors de l'exécution",
                                        QMessageBox.Ok)
                self.plainTextEdit_histoire.insertPlainText(
                    "\n" + "Le résultat est enregistré dans le répertoire /resultats sous format PDF.")
        else:
          print("no video")
