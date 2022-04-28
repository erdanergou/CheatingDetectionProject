from PyQt5.QtWidgets import QApplication
import sys

from model.cheatingDetectionMainModel import CheatingDetectionApp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = CheatingDetectionApp()
    mainWindow.show()
    exit(app.exec_())
