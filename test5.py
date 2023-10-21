import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt

class TransparentOverlay(QWidget):
    def __init__(self):
        super().__init__()

        # Set the window size and position
        screen_geometry = QApplication.desktop().screenGeometry()
        self.setGeometry(screen_geometry)

        # Set window transparency
        self.setWindowOpacity(0.3)

        # Remove window decorations
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Make the window stay always on top
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        # Set the window to not accept mouse events
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = TransparentOverlay()
    overlay.show()
    sys.exit(app.exec_())
