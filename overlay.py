# import sys
# from PyQt6.QtWidgets import QApplication, QWidget
# from PyQt6.QtCore import Qt
# from PyQt6.QtGui import QPainter, QBrush, QColor


# # get monitor res
# from screeninfo import get_monitors

# main = get_monitors()[0]

# class OverlayWidget(QWidget):
#     def __init__(self):
#         super().__init__()

#         # Set up the window
#         self.setWindowTitle("Overlay Window")

#         # ------- #
#         # only get main monitor
#         self.setGeometry(-20, 0, main.width, main.height)
#         print(main)

#         # Set the window's flags to make it act as an overlay
#         # self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
#         # self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
#         # self.setStyleSheet("background-color: rgba(0, 0, 0, 0);")

#         self.rectangles = [(0, 0, 20, 20), (main.width - 20, 100, 20, 20)]

#         painter = QPainter(self)
#         painter.setBrush(QBrush(QColor(255, 0, 0)))  # Red color

#         for rect in self.rectangles:
#             painter.drawRect(*rect)
    
#     def paintEvent(self, event):
#         painter = QPainter()
#         painter.begin(self)  # Start the painting session

#         painter.setBrush(QBrush(QColor(255, 0, 0)))  # Red color

#         for rect in self.rectangles:
#             painter.drawRect(*rect)

#         painter.end()  # End the painting session


#     def mousePressEvent(self, event):
#         # Close the overlay when clicked
#         # self.close()
#         pass

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = OverlayWidget()
#     window.show()
#     sys.exit(app.exec())




import sys
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QBrush, QColor


# get monitor res
from screeninfo import get_monitors

main = get_monitors()[0]
class SmallWindow(QWidget):
    def __init__(self, title, x, y, w = 40, h = 40):
        super().__init__()

        # Set window title and position
        self.setWindowTitle(title)
        self.setGeometry(x, y, w, h)  # x, y, width, height

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: rgba(255, 0, 255, 255);")

    def keyPressEvent(self, event):
        # check if escape key
        if event.key() == Qt.Key.Key_Escape:
            QApplication.instance().quit()  # Close the application



if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Create 4 small windows with different titles and positions
    window1 = SmallWindow("Window 1", 0, 0)
    window2 = SmallWindow("Window 2", 0, main.height-40)
    window3 = SmallWindow("Window 3", main.width - 40, 0)
    window4 = SmallWindow("Window 4", main.width - 40, main.height - 40)

    # Show all the windows
    window1.show()
    window2.show()
    window3.show()
    window4.show()

    sys.exit(app.exec())
