#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Cursor
import numpy as np
import cv2
import time
#from dtw import dtw
from pylab import norm

class affichage_graphique:
    """
        Cette classe est utilisee pour gerer l'affichage du resultat.
        L'affichage se fait dans une nouvelle fenêtre créée par matplotlib.
        Le résultat est une courbe avec un curseur.
        En cliquant sur la courbe, en bas à gauche affichera l'image correspondante

        @version 3.0
    """

    def __init__(self, video, start_frame):
        """
            Ce constructeur prend en entree une video et une frame
            de depart afin d'initialiser la fenetre d'affichage
        """
        self.fig = plt.figure(figsize=(10, 8), dpi=80)
        self.gs = gridspec.GridSpec(2, 2)
        self.subplot1 = self.fig.add_subplot(self.gs[0, :])
        self.fig.canvas.mpl_connect('button_press_event', self.__onclick)
        self.fig.canvas.set_window_title('Resultat du traitement')
        self.start_frame = start_frame
        self.video = video


    def __onclick(self, event):
        """
        Cette fonction gere l'evenement du click de la souris
        sur la fenetre resultat. Elle prend en entree un parametre:
        - param event: click de la souris
        - type event: evenement click sur la souris
        - :returns:
        """

        #Obtention du numero de frame
        x = int(float(event.xdata))
        #Ajouter une partie en dessous du graphique resultat
        self.fig.add_subplot(self.gs[1, 0])

        #Le titre de cette partie
        plt.title("Image correspondante :")

        #L'image correspondant a l'endroit clique
        image = self.getImage(x)

        #Affichage de cette image en couleur
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        #Suppression des axes
        plt.axis("off")

        #la zone de dessin
        self.fig.canvas.draw()



    def getImage(self, frame):
        """
           Cette fonction obtient une image a partir de son numero de frame
           puis la renvoie.
           Elle prend en entree un parametre:
           - :param frame: numero de la frame
           - :type int: un entier
           - :returns image: image obtenu en deroulant la video jusqu'au
           numero de notre frame.
        """
        #Ouvre le fichier video et renvoie un objet videocapture
        cap = cv2.VideoCapture(self.video)

        indice = 0
        ret = True
        image = None

        #Derouler la video jusqu'a la frame de depart
        for indice in range(0, self.start_frame):
            ret, image = cap.read()

        """
            si ret est toujours True c'est qu'il reste encore des frames.
            tant qu'il reste des frames nous allons incrementer indice et recuperer 
            l'image
        """

        while indice <= frame and ret:
            ret, image = cap.read()
            indice += 1

        #Ferme le fichier
        cap.release()
        return image

    def afficher(self, ma_liste):
        """
           Cette fonction dessine la courbe de resultat
           Elle prend en entree un parametre:
           - :param ma_liste: liste des resultats fournis par l'algorithme de calcul
           - :type liste: une liste
           - :returns:
        """
        #Variables utiles
        #Tableau de la même taille que notre liste
        
        #creer un tableau  contenant de la taille de ma_liste contenant des 0
        zero=np.array(range(len(ma_liste))).reshape(-1,1)
        zero.fill(0)
        """
        #Recuperer les donnees de la poule calme
        calme=np.loadtxt('poule_calme.txt')
        current = np.array(ma_liste).reshape(-1,1)
        
        #Distance 0 et poule calme
        dist, cost, acc, path =dtw(zero, calme, dist=lambda zero, calme: norm(zero - calme, ord=1))
        print('la distance par rapport a la poule calme ',dist)

        # Distance 0 et poule actuelle
        dist, cost, acc, path = dtw(zero, current, dist=lambda zero, current: norm(zero - current, ord=1))
        print('la distance par rapport a la poule actuelle ', dist)
        """
        x = np.array(range(len(ma_liste)))

        #Tableau copie de notre liste
        y = np.array(ma_liste)

        #Valeur max de la liste
        y_max = max(y)

        #Valeur min de la liste
        y_min = min(y)

        #Valeur moyenne de la liste
        y_mean = np.mean(y)

        #Heure locale en string
        localtime = time.localtime(time.time())
        localtime_str = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + '_' + \
                        str(localtime.tm_hour) + 'h' + str(localtime.tm_min) + 'm' + str(localtime.tm_sec) + 's'

        #Initialisation du graphique
        plt.axis([0, len(ma_liste), y_min - y_min / 4, y_max + y_max / 4])

        #Titre des abscisses
        plt.xlabel("Nombre d'images")

        #Titre des ordonnees
        plt.ylabel("Quantite de mouvement")

        #Permet de gerer l'affichage des titres
        plt.tight_layout()

        #Creation du graphique
        plt.plot(x, y)

        #Rajout d'une deuxieme figure avec ses coordonnees
        self.fig.add_subplot(self.gs[1, 1])

        #Rajout d'un curseur
        Cursor(self.subplot1, useblit=True, color='red', linewidth=2)

        #Titre du graphique
        plt.title("Statistiques : ")

        #Taille des axes
        plt.axis([0, 100, 0, 100])

        #Affichage des axes (Non!)
        plt.axis("off")

        #Rajout de texte au graphique
        plt.text(20, 80, "min : " + str(y_min))
        plt.text(20, 60, "max : " + str(y_max))
        plt.text(20, 40, "moyenne : " + str(y_mean))

        #Affichage du graphique
        plt.show()

        #Enregistrement dans resultat
        self.fig.savefig('resultats/'+str(localtime_str)+'.pdf')

        #Fermeture du graphique
        plt.close()
