import cv2
import numpy as np
import math
import ZoneInteret as zi
import algo
from RGBToHSV import HSV


# Classe qui implemente l'algorithme Flot Optique
class flot_optiques(algo.algorithme):
    # Retourner le nom d'algorithme
    def get_nomAlgo(self):
        return "Flot optiques"

    # Fonction pricipale pour traiter la video
    def traiterVideo(self, video, start_frame):
        ma_liste = list()
        cap = cv2.VideoCapture(video)
        
        # Parametres pour ShiTomasi Corner Detection
        # maxcorner : nombre maximal de points
        # qualityLevel : influence la qualité du coin détecté
        # minDistance : distance minimum entre les détections
        # blockSiza : influence la qualité du calcul
        feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=7, blockSize=7)
        
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
            param = [0, 0, len(frame), len(frame[0])]

        # Transforme la frame en niveau de gris
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        # Détecte un ensemble de points d'intérêt
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        # Copie le tableau contenant la frame en le remplissant de 0
        mask = np.zeros_like(old_frame)

        # Compteur des points détectés et des points supprimés par l'lagorithme des couleurs
        count_point = 0
        count_removed = 0

        boucle = True
        while boucle:
            # ret est égal à 0 si il n'y a plus de frame
            if ret:
                # converti la frame en niveau de gris
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Calcule le flot optique à partir des deux frame et des points détectés sur la première
                # p1, un nouvel ensemble de point
                # st, un tableau de même taille que P0 et P1, pour chaque point 1 si une correspondance a été trouvé
                # pour p0 dans p1, 0 sinon
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
                    nombre_points = 0
                    # Parcours de l'ensemble des points
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        # Récupération des coordonnées des deux points
                        a, b = new.ravel()
                        c, d = old.ravel()

                        # On fait la moyenne de couleurs des points aux alentours du point de la nouvelle frame
                        rgb = moyenne_pixels_alentours(int(a), int(b), frame)
                        # Si il n'y a pas de problème, on converti la couleur RGB en HSV
                        if rgb != 0:
                            hsv = HSV(rgb[0], rgb[1], rgb[2])
                        else:
                            hsv = None
                        # Si il ne s'agit pas de la couleur que l'on veut éviter, on l'ajoute à notre norme moyenne
                        # et on le dessine
                        if hsv is not None and not isRightColor(hsv):
                            X += a
                            Y += b
                            nombre_points += 1

                            cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                            cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                        else:
                            count_removed += 1
                        count_point += 1
                    # Pour cette frame on calcule la norme du vecteur centre de gravité du déplacement entre
                    # les deux frames
                    if nombre_points > 0:
                        norme_total += math.sqrt((X/nombre_points) ** 2 + (Y/nombre_points) ** 2)
                        ma_liste.append(norme_total)

                    img = cv2.add(frame, mask)
                    cv2.imshow("frame", img)

                    # SI l'utilisateur appuie sur Echap, stop le déroulement de l'algorithme
                    k = cv2.waitKey(30) & 0xff
                    if k == 27:
                        break

                    p0 = good_new.reshape(-1, 1, 2)
                # Si l'ensemble de point est vide, on relance une détection de points
                else:
                    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                # La deuxième frame prend la place de la premier
                old_gray = frame_gray.copy()
            #Puis on relit une nouvelle frame
            ret, frame = cap.read()

            # Si il n'y a plus de frame, on sort de la boucle
            if ret:
                frame = frame[param[1]:param[1] + param[3], param[0]:param[0] + param[2]]
            else:
                boucle = False

        print(count_removed)
        print("/")
        print(count_point)

        cap.release()
        cv2.destroyAllWindows()

        # Retourne la liste des normes du centre de gravité des nuages de points pour chaque frame
        return ma_liste

# Fonction qui détermine si une couleur est proche des alentours d'une autre couleur
# Entrée : la couleur au format  HSV
# Sortie : vrai si la couleur est dans l'intervalle, faux sinon
def isRightColor(hsv):
    # Variation acceptable de la teinte pour considérer la couleur comme dans l'intervalle
    deltaH = 30
    deltaSV = 50
    # La couleur ciblée
    base = HSV(210, 188, 170)
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
def moyenne_pixels_alentours(ptx, pty, frame):

    r = 0
    g = 0
    b = 0
    count = 0
    aire = 5

    # Parcours du carré de taille aire x aire donc le pixel concerné est le centre
    for i in range(ptx - int(aire / 2), ptx + int(aire / 2)):
        for j in range(pty - int(aire / 2), pty + int(aire / 2)):
            # Ne prendre les valeurs que si les points sont dans la frame
            if i in range(0, len(frame)) and j in range(0, len(frame[i])):
                r += frame[i][j][0]
                g += frame[i][j][1]
                b += frame[i][j][2]
                count += 1

    if count == 0:
        return 0

    return int(r/count), int(g/count), int(b/count)