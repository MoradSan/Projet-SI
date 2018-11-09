import cv2
import numpy as np
import math
import ZoneInteret as zi
import algo
import matplotlib.pyplot as plt

#Classe qui implemente l'algorithme Flot Optique
class flot_optiques(algo.algorithme):
    #Retourner le nom d'algorithme
    def get_nomAlgo(self):
        return "Flot optiques"

    #Fonction pricipale pour traiter la video
    def traiterVideo(self, video, start_frame):
        ma_liste = list()
        cap = cv2.VideoCapture(video)
        
        #Params pour ShiTomasi Corner Detection
        # maxcorner = nombre maximal de points
        #
        feature_params = dict(maxCorners=100, qualityLevel=0.01,
                              minDistance=7, blockSize=7)
        
        #Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        #Creer des couleurs aleatoire
        color = np.random.randint(0, 255, (100, 3))
        for i in range(0, start_frame):
            cap.read()

        ret, old_frame = cap.read()
        ret, frame = cap.read()
        
        #Fixer la detection dans la zone interet
        if zi.ZoneInteret.verifier_presence_fichier_ini():
            with open("./zi/param.ini", "r") as file:
                line = file.readline()
                param = [int(x.strip()) for x in line.split(',')]
            frame = frame[param[1]:param[1] + param[3], param[0]:param[0] + param[2]]
            old_frame = old_frame[param[1]:param[1] + param[3], param[0]:param[0] + param[2]]

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        mask = np.zeros_like(old_frame)

        while (1):
            #ret, frame = cap.read()
            if ret:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                #Calculer flot optique
                #p0 et p1 deux tableaux de même taille
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                #print(frame[int(p1[0]), int(p1[1])])

                #Selectionner les bons points
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                """
                norme_total = 0
                nombre_points = 0
                #Faire le dessin qui indique le changement de points caracteristiques
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    norme_total += math.sqrt((a - c) ** 2 + (b - d) ** 2)
                    nombre_points += 1
                    cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                    cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                ma_liste.append(norme_total / nombre_points / 0.05)
                """

                #Affichage de la norme du centre de gravité
                norme_total = 0
                X = 0
                Y = 0
                nombre_points = 0
                # Faire le dessin qui indique le changement de points caracteristiques
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()

                    X += a
                    Y += b
                    nombre_points += 1

                    cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                    cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

                if nombre_points > 0:
                    norme_total += math.sqrt((X/nombre_points) ** 2 + (Y/nombre_points) ** 2)
                    ma_liste.append(norme_total)

                """#Affichage de la moyenne des x 
                x = 0
                n = 0
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    x += a
                    n+=1
                    cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                    cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                ma_liste.append(x/n)
                """

                img = cv2.add(frame, mask)

                cv2.imshow("frame", img)

                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
                
                #Mise a jours le frame actuel
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

                ret, frame = cap.read()

            if ret and zi.ZoneInteret.verifier_presence_fichier_ini():
                frame = frame[param[1]:param[1] + param[3], param[0]:param[0] + param[2]]
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

        return ma_liste
