import itertools

from RGBToHSV import HSV
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
import math

import ZoneInteret as zi
import algo


# Classe qui implemente l'algorithme Flot Optique
class flot_optiques(algo.algorithme):

    # Retourner le nom d'algorithme
    def get_nomAlgo(self):
        return "Flot optiques"

    # Fonction pricipale pour traiter la video
    def traiterVideo(self, video, start_frame, colorOperateur=None, suppressionCouleur=False):

        nbr_maxCorner = 100
        if colorOperateur is not None:
            #On converti d'abord la QColor en HSV
            COULEUR_INDESIRABLE = HSV(colorOperateur.red(), colorOperateur.green(), colorOperateur.blue())
        else:
            COULEUR_INDESIRABLE = None

        ma_liste = list()
        cap = cv2.VideoCapture(video)

        # Parametres pour ShiTomasi Corner Detection
        # maxcorner : nombre maximal de points
        # qualityLevel : influence la qualité du coin détecté
        # minDistance : distance minimum entre les détections
        # blockSiza : influence la qualité du calcul
        feature_params = dict(maxCorners=nbr_maxCorner, qualityLevel=0.01, minDistance=7, blockSize=7)

        # Parametres pour Lucas Kanade optical flow
        # winSize : taille de la fenêtre de recherche du point proche
        # maxLevel : nombre de niveau de piramide (voir détail de l'algorithme
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Création de couleurs aleatoire
        color = np.random.randint(0, 255, (100, 3))

        # On avance jusqu'à la première frame
        for i in range(0, start_frame):
            cap.read()

        # récupère les deux premières frames
        ret, old_frame = cap.read()
        ret, frame = cap.read()

        # Récupère la zone d'intérer et réduit les frames à celle-ci
        if zi.ZoneInteret.verifier_presence_fichier_ini():
            with open("./zi/param.ini", "r") as file:
                line = file.readline()
                param = [int(x.strip()) for x in line.split(',')]
            frame = frame[param[1]:param[1] + param[3], param[0]:param[0] + param[2]]
            old_frame = old_frame[param[1]:param[1] + param[3], param[0]:param[0] + param[2]]
        else:
            # Si la zone d'interêt n'existe pas on garde la frame en entier
            param = [0, 0, len(frame[0]), len(frame)]

        # Transforme la frame en niveau de gris
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        # Détecte un ensemble de points d'intérêt
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        # Copie le tableau contenant la frame en le remplissant de 0
        mask = np.zeros_like(old_frame)

        oldGx = -1
        oldGy = -1
        boucle = True
        while boucle:
            # ret est égal à 0 si il n'y a plus de frame
            if ret:
                if colorOperateur is not None and suppressionCouleur:
                    # Traitement préalable sur la frame : les pixels et leurs alentours sont remplacés par du blanc
                    frame = suppression_couleur(frame, param[2], param[3], COULEUR_INDESIRABLE, 10)

                # converti la frame en niveau de gris
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Détection des points
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                # Si l'ensemble de point est vide, on relance une détection de points
                # print(len(p1))
                if p1 is None or len(p1) <= nbr_maxCorner / 2:
                    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                # Cas où l'ensemble de point n'est pas vide
                if p1 is not None:
                    # Selectionner uniquement les bons points qui on une correspondance d'une frame à l'autre
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    # Initialisations pour le calcule de la norme du centre de gravité
                    norme_total = 0
                    X = 0
                    Y = 0
                    Gx = 0
                    Gy = 0
                    nombre_points = 0
                    # Parcours de l'ensemble des points
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        # Récupération des coordonnées des deux points
                        a, b = new.ravel()
                        # Cas où on ne veut pas prendre en compte l'opérateur
                        if colorOperateur is not None:
                            # On fait la moyenne de couleurs des points aux alentours du point de la nouvelle frame
                            rgb = moyenne_pixels_alentours(int(a), int(b), param[2], param[3], frame)
                            # Si il n'y a pas de problème, on converti la couleur RGB en HSV
                            if rgb != 0:
                                hsv = HSV(rgb[0], rgb[1], rgb[2])
                            else:
                                hsv = None
                            # Si il ne s'agit pas de la couleur que l'on veut éviter, on l'ajoute à notre norme moyenne
                            # et on le dessine
                            if hsv is not None and not isRightColor(hsv, COULEUR_INDESIRABLE):
                                X += (a - param[0]) / param[2]  # On ramène par rapport à la taille de la zone d'intérêt
                                Y += (b - param[1]) / param[3]
                                Gx += a
                                Gy += b
                                nombre_points += 1
                                #   cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                                cv2.circle(frame, (a, b), 3, color[i].tolist(), -1)
                        # Sinon on prend tout en compte
                        else:
                            X += (a - param[0]) / param[2]  # On ramène par rapport à la taille de la zone d'intérêt
                            Y += (b - param[1]) / param[3]
                            Gx += a
                            Gy += b
                            nombre_points += 1
                            #   cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                            cv2.circle(frame, (a, b), 3, color[i].tolist(), -1)

                    if nombre_points > 0:
                        # Affichage du point de gravité et de son déplacement
                        cv2.circle(frame, (int(Gx/nombre_points), int(Gy/nombre_points)), 8, (0, 0, 255), -1)
                        if oldGx != -1 and oldGy != -1:
                            cv2.line(mask, (int(oldGx/nombre_points), int(oldGy/nombre_points)),
                                     (int(Gx / nombre_points), int(Gy / nombre_points)), (25, 25, 200), 2)
                        oldGx = Gx
                        oldGy = Gy

                        norme_total += math.sqrt((X/nombre_points) ** 2 + (Y/nombre_points) ** 2)
                        ma_liste.append(norme_total)

                    img = cv2.add(frame, mask)
                    cv2.imshow("frame", img)

                    # SI l'utilisateur appuie sur Echap, stop le déroulement de l'algorithme
                    k = cv2.waitKey(30) & 0xff
                    if k == 27:
                        break

                    p0 = good_new.reshape(-1, 1, 2)

                # La deuxième frame prend la place de la premier
                old_gray = frame_gray.copy()
            #Puis on relit une nouvelle frame
            ret, frame = cap.read()

            # Si il n'y a plus de frame, on sort de la boucle
            if ret:
                frame = frame[param[1]:param[1] + param[3], param[0]:param[0] + param[2]]
            else:
                boucle = False

        cap.release()
        cv2.destroyAllWindows()

        # Retourne la liste des normes du centre de gravité des nuages de points pour chaque frame
        return ma_liste

# Fonction qui détermine si une couleur est proche des alentours d'une autre couleur
# Entrée : la couleur au format  HSV
# Sortie : vrai si la couleur est dans l'intervalle, faux sinon
def isRightColor(hsv, base):
    # Variation acceptable de la teinte pour considérer la couleur comme dans l'intervalle
    deltaH = 30
    deltaSV = 20
    # La couleur ciblée
    base = base.getHSV()
    hsv = hsv.getHSV()

    # La teinte minimale et maximale accepté
    Hmin = int((base[0] - deltaH) % 359)
    Hmax = int((base[0] + deltaH) % 359)

    # Comme la teinte à une valeur ciruclaire en °, l'algortihme prend le cas où la distance peut passer de 360° à 0°
    if (base[0] - deltaH < 0) or (base[0] + deltaH > 359):
        if hsv[0] in range(Hmin, Hmax):
            return False
    else:
        if hsv[0] < Hmin or hsv[0] > Hmax:
            return False

    # Calcul de la différence de teinte et de saturation entre la cible et la couleur étudiée (résultat en %)
    diffS = abs(base[1] - hsv[1])
    diffV = abs(base[2] - hsv[2])
    # En dessous de 50% la couleur est rejétée
    if diffS > deltaSV or diffV > deltaSV:
        return False
    return True


# Fonction qui permet de ressorti la couleur moyenne des pixels aux alentours d'un pixel donné (compris)
# Enrées : les coordonnées x et y d'un pixel et la frame concernée
# Sorties : le nouveau trio r, g, b. 0 si le pixel n'a pas de voisin et donc si il est à l'extérieur de la frame
def moyenne_pixels_alentours(ptx, pty, width, height, frame, aire=5):

    r = 0
    g = 0
    b = 0
    count = 0

    # Parcours du carré de taille aire x aire donc le pixel concerné est le centre
    for i in range(ptx - int(aire / 2), ptx + int(aire / 2)):
        for j in range(pty - int(aire / 2), pty + int(aire / 2)):
            # Ne prendre les valeurs que si les points sont dans la frame
            if i in range(0, width - 1) and j in range(0, height - 1):
                r += frame[j][i][2]
                g += frame[j][i][1]
                b += frame[j][i][0]
                count += 1

    if count == 0:
        return 0

    return int(r/count), int(g/count), int(b/count)

# Fonction qui appelle le thread thread_suppr_coul en multithreading
def suppression_couleur(frame, width, height, hsv_indesirable, aire=5):

    pool = ThreadPool(20)

    pool.starmap(thread_suppr_coul, zip(itertools.repeat(frame), itertools.repeat(width), itertools.repeat(height),
                                        itertools.repeat(hsv_indesirable), itertools.repeat(aire),
                                        [(0, 0), (0, 1), (1, 0), (1, 1)]))
    pool.close()
    pool.join()
    return frame

# Thread qui supprime les pixels alentours si ils sont de la bonne couleur (les applique en blanc
def thread_suppr_coul(frame, width, height, hsv_indesirable, aire, partOfFrame):
    for i in range(int(width / 2 * partOfFrame[0]), int(width / 2 + (width / 2 * partOfFrame[0]))):
        for j in range(int(height / 2 * partOfFrame[1]), int(height / 2 + (height / 2 * partOfFrame[1]))):
            hsv = HSV(frame[j][i][2], frame[j][i][1], frame[j][i][0])
            if isRightColor(hsv, hsv_indesirable):
                for x in range(i - int(aire / 2), i + int(aire / 2)):
                    for y in range(j - int(aire / 2), j + int(aire / 2)):
                        frame[j][i][0] = 255
                        frame[j][i][1] = 255
                        frame[j][i][2] = 255
                j += int(aire / 2)
