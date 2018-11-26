class HSV:
    def __init__(self, R, G, B):
        Rb = R/255
        Gb = G/255
        Bb = B/255
        Cmax = max(Rb, Gb, Bb)
        Cmin = min(Rb, Gb, Bb)
        delta = Cmax - Cmin

        if delta == 0:
            self.H = 0
        elif Cmax == Rb:
            self.H = int(60 * (((Gb-Bb)/delta) % 6))
        elif Cmax == Gb:
            self.H = int(60 * (((Bb-Rb)/delta) + 2))
        else: #Cmax == Bb
            self.H = int(60 * (((Rb-Gb)/delta) + 4))
        if Cmax == 0:
            self.S = 0.0
        else:
            self.S = int((delta / Cmax) * 100)
        self.V = int(Cmax * 100)


    def getHSV(self):
        return self.H, self.S, self.V
