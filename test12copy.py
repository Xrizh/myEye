import time
import adhawkapi
import adhawkapi.frontend
import numpy as np
import cv2
import threading
import math

import sys
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QBrush, QColor
import pyautogui



# get monitor res
from screeninfo import get_monitors


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



# --------------------------------------------------



frame = None  # Declare global frame to be accessed in multiple functions
xvec, yvec = 0.0, 0.0  # Initialize gaze vector components to some default values



# convert 15 inch to meters
CSECTION = 13
# 1 inch = 0.0254 meters
COMPUTER_CSECTION = 0.0254 * CSECTION

COUNTER = 0

class FrontendData:
    def __init__(self):
        global xvec, yvec  # Declare these as global
        self._api = adhawkapi.frontend.FrontendApi(ble_device_name='ADHAWK MINDLINK-296')
        self._api.register_stream_handler(adhawkapi.PacketType.EYETRACKING_STREAM, self._handle_et_data)
        self._api.register_stream_handler(adhawkapi.PacketType.EVENTS, self._handle_events)
        self._api.start(tracker_connect_cb=self._handle_tracker_connect,
                        tracker_disconnect_cb=self._handle_tracker_disconnect)

    def shutdown(self):
        self._api.shutdown()

    @staticmethod
    def _handle_et_data(et_data: adhawkapi.EyeTrackingStreamData):
        global COUNTER, xvec, yvec  # Declare these as global
        COUNTER += 1
        if COUNTER % 10 != 0: return
        if et_data.gaze is not None:
            xvec, yvec, zvec, vergence = et_data.gaze
            # print(f'Gaze={xvec:.2f},y={yvec:.2f},z={zvec:.2f},vergence={vergence:.2f}')


    @staticmethod
    def _handle_events(event_type, timestamp, *args):
        if event_type == adhawkapi.Events.BLINK:
            duration = args[0]
            print(f'Got blink: {timestamp} {duration}')

    def _handle_tracker_connect(self):
        print("Tracker connected")
        self._api.set_et_stream_rate(60, callback=lambda *args: None)
        self._api.set_et_stream_control([
            adhawkapi.EyeTrackingStreamTypes.GAZE,
            adhawkapi.EyeTrackingStreamTypes.EYE_CENTER,
            adhawkapi.EyeTrackingStreamTypes.PUPIL_DIAMETER,
            adhawkapi.EyeTrackingStreamTypes.IMU_QUATERNION,
        ], True, callback=lambda *args: None)
        self._api.set_event_control(adhawkapi.EventControlBit.BLINK, 1, callback=lambda *args: None)
        self._api.set_event_control(adhawkapi.EventControlBit.EYE_CLOSE_OPEN, 1, callback=lambda *args: None)

    def _handle_tracker_disconnect(self):
        print("Tracker disconnected")


def clamp(_min, _max, val):
    if val < _min:
        return _min
    if val > _max:
        return _max
    if math.isnan(val):
        return 0
    else:
        return val



# HSV_RANGE = [np.array((60, 0, 0)), np.array((85, 100, 100))] # green
HSV_RANGE = [np.array((135, 100, 200)), np.array((155, 160, 255))] # magenta
C_LIMIT = 10


def main():
    ''' App entrypoint '''
    global xvec, yvec  # Declare these as global
    frontend = FrontendData()
    # create an opencv camera instance
    cap = cv2.VideoCapture(0)

    # check if camera is opened
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    try:
        # run the opencv code
        while True:
            # read frame from camera
            ret, frame = cap.read()

            # check if frame is read correctly
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Get dimensions
            h, w, c = frame.shape

            # For demonstration: create a point from gaze vector
            xc = (clamp(-3, 3, xvec) + 3) / 6 * w
            x_point = int(clamp(-10000, 10000, xc + (75 + xc/80)))
            y_point = h - int((clamp(-2, 2, yvec) + 2) / 4 * h)
            #if COUNTER % 10 == 0: print(h, w, x_point, y_point)

            # -- 
            # get hsv image
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # filter out colours within a range
            mask = cv2.inRange(hsv, HSV_RANGE[0], HSV_RANGE[1])
            result = cv2.bitwise_and(frame, frame, mask=mask)
            # # convert result to black and white
            resultbw = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)


            coords = []
            # draw rectangles around each contour
            contours, hierarchy = cv2.findContours(resultbw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w*h > C_LIMIT:
                    coords.append((x, y))
                    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)



            # sort coords by x, then by y
            coords = sorted(coords, key=lambda x: (x[0]))
            # draw coords to result (lines)
            if len(coords) >= 4:
                topleft = max(coords[:2], key=lambda x: x[1])
                botleft = min(coords[:2], key=lambda x: x[1])
                topright = max(coords[2:], key=lambda x: x[1])
                botright = min(coords[2:], key=lambda x: x[1])
                # print(topleft, botleft, topright, botright, coords)
                if topleft[0] >= x_point <=botright[0] and botleft[1] >= y_point <=topright[1]:
                    print("Look at the screen")
                    cv2.circle(frame, (x_point, y_point), 5, (255, 0, 0), -1)
                else:
                    print("You are looking inside")
                    cv2.circle(frame, (x_point, y_point), 5, (0, 255, 0), -1)


                # cv2.line(result, coords[0], coords[1], (0, 0, 255), 2) # top line
                # cv2.line(result, coords[2], coords[3], (0, 0, 255), 2) # bot line
                # cv2.line(result, coords[0], coords[2], (0, 0, 255), 2) # left line
                # cv2.line(result, coords[1], coords[3], (0, 0, 255), 2) # right line
                cv2.line(result, topleft, botleft, (0, 0, 255), 2)
                cv2.line(result, topright, botright, (0, 0, 255), 2)
                cv2.line(result, topleft, topright, (0, 0, 255), 2)
                cv2.line(result, botleft, botright, (0, 0, 255), 2)
                #---------
            
                cv2.line(frame, topleft, botleft, (0, 0, 255), 2)
                cv2.line(frame, topright, botright, (0, 0, 255), 2)
                cv2.line(frame, topleft, topright, (0, 0, 255), 2)
                cv2.line(frame, botleft, botright, (0, 0, 255), 2)
                cv2.line(frame, botleft, topright, (0, 255, 0), 2)

                cv2.circle(frame, (x_point, y_point), 5, (255, 255, 255), -1)

                cv2.imshow('frame', frame)

                # diagonal line
                cv2.line(result, botleft, topright, (0, 255, 0), 2)
                d_length = np.sqrt((botleft[0] - topright[0])**2 + (botleft[1] - topright[1])**2) # pixels
                scale_factor = COMPUTER_CSECTION / d_length # meters/pixels

                #print(scale_factor)
            
            # show frame
            # cv2.imshow('frame', frame)
            # cv2.imshow('hsv', hsv)
            # cv2.imshow('mask', mask)

            # Draw a circle on the gaze point
            #cv2.circle(result, (x_point, y_point), 5, (255, 255, 255), -1)

            cv2.imshow('result', result)


            # wait for key press
            if cv2.waitKey(1) == ord('q'):
                break

    except (KeyboardInterrupt, SystemExit) as e:
        frontend.shutdown()
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    wmain = get_monitors()[0]
    app = QApplication(sys.argv)
    
    # Create 4 small windows with different titles and positions
    window1 = SmallWindow("Window 1", 0, 0)
    window2 = SmallWindow("Window 2", 0, wmain.height-40)
    window3 = SmallWindow("Window 3", wmain.width - 40, 0)
    window4 = SmallWindow("Window 4", wmain.width - 40, wmain.height - 40)

    # Show all the windows
    window1.show()
    window2.show()
    window3.show()
    window4.show()
    main()