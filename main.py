import sys
from Fen_principale import Fen_principale
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':

    sys._excepthook = sys.excepthook


    def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)


    sys.excepthook = exception_hook

    """Lancement de l'application 
    en lançant la fenetre principale"""
    app = QApplication(sys.argv)
    MainWindow = Fen_principale()
    MainWindow.show()

    """Retourne un exit status 
    (0 pour succes, tout le reste pour l'echec"""
    try:
        sys.exit(app.exec_())
    except:
        sys.exit(0)


