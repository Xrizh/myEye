import cv2
from mc_controller import minecraft_controller
from websurfing_controller import websurfing_controller
import keyboard
import pynput
import threading

active_mode = 'minecraft'  # default to web mode

def toggle_mode(e):
    global active_mode
    if active_mode == 'web':
        print("Switching to Minecraft Mode")
        active_mode = 'minecraft'
    else:
        print("Switching to Web Surfing Mode")
        active_mode = 'web'

keyboard.on_press_key("f4", toggle_mode)

import time
import adhawkapi
import adhawkapi.frontend
import numpy as np
import cv2
import threading
import math, os

import sys
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QBrush, QColor
import pyautogui


pyautogui.FAILSAFE = False
MOUSE = pynput.mouse.Controller()

# get monitor res
from screeninfo import get_monitors

class SmoothedValue:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev_value = None

    def get(self, value):
        if self.prev_value is None:
            self.prev_value = value
        smoothed_value = self.alpha * value + (1 - self.alpha) * self.prev_value
        self.prev_value = smoothed_value
        return smoothed_value

class SmoothEyeTracker:
    def __init__(self, wa, wb):
        self.a, self.b = wa, wb
        self.wp = (0,0)
        self.s = 80
    
    def next_pos(self, x, y):
        dx, dy = x - self.wp[0], y - self.wp[1]
        res = [self.wp[0] + dx / self.s, self.wp[1] + dy / self.s]
        # res[0] += dx / 10
        # res[1] += dy / 10

        self.wp = (x, y)
        return res

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
xvec, yvec, zvec = 0.0, 0.0, 0.0  # Initialize gaze vector components to some default values



# convert 15 inch to meters
CSECTION = 15
WIDTH = 13.4
HEIGHT = 9.4

# 1 inch = 0.0254 meters
COMPUTER_CSECTION = 0.0254 * CSECTION

COUNTER = 0

# HSV_RANGE = [np.array((60, 0, 0)), np.array((85, 100, 100))] # green
HSV_RANGE = [np.array((135, 100, 200)), np.array((155, 160, 255))] # magenta
C_LIMIT = 15



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
            #print(f'Got blink: {timestamp} {duration}')

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

def transform_gaze_to_screen_space(gaze_point, src_points, dst_points):
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Ensure gaze_point is a Nx1x2 array
    gaze_points_array = np.array([gaze_point], dtype=np.float32).reshape(-1, 1, 2)
    
    transformed_points = cv2.perspectiveTransform(gaze_points_array, M)
    return tuple(map(int, transformed_points[0][0]))


# HSV_RANGE = [np.array((60, 0, 0)), np.array((85, 100, 100))] # green
HSV_RANGE = [np.array((135, 100, 200)), np.array((155, 160, 255))] # magenta
C_LIMIT = 10
EYE_SMOOTHER = SmoothEyeTracker(0.5, 0.2)


# def run_gestures():
#     global minecraft_controller, active_mode, websurfing_controller
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

#         if active_mode == 'minecraft':
#             minecraft_controller(frame, active_mode)
#         elif active_mode == 'web':
#             websurfing_controller(frame, active_mode)

#         # Display the frame (you can add conditions to display based on the mode)
#         cv2.imshow("Controller", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()




def main():
    ''' App entrypoint '''
    global xvec, yvec  # Declare these as global
    global minecraft_controller, active_mode, websurfing_controller

    frontend = FrontendData()
    # create an opencv camera instance
    cap = cv2.VideoCapture(0)


    # check if camera is opened
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    #screen_present = False
    try:
        
        # run the opencv code
        while True:
            # read frame from camera
            ret, frame = cap.read()

            if active_mode == 'minecraft':
                minecraft_controller(frame, active_mode)
            elif active_mode == 'web':
                websurfing_controller(frame, active_mode)

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

            #screen_present = False  # Reset the screen presence flag at the beginning of every iteration

            if len(coords) >= 4:
                topleft = max(coords[:2], key=lambda x: x[1])
                botleft = min(coords[:2], key=lambda x: x[1])
                topright = max(coords[2:], key=lambda x: x[1])
                botright = min(coords[2:], key=lambda x: x[1])
                # print(topleft, botleft, topright, botright, coords)
                cv2.line(frame, topleft, botleft, (0, 0, 255), 2)
                cv2.line(frame, topright, botright, (0, 0, 255), 2)
                cv2.line(frame, topleft, topright, (0, 0, 255), 2)
                cv2.line(frame, botleft, botright, (0, 0, 255), 2)

                #screen_present = True
                #---------

                # diagonal line
                cv2.line(result, botleft, topright, (0, 255, 0), 2)
                d_length = np.sqrt((botleft[0] - topright[0])**2 + (botleft[1] - topright[1])**2) # pixels
                scale_factor = COMPUTER_CSECTION / d_length # meters/pixels

                src_points = [topleft, botleft, topright, botright]
                dst_points = [(0, 0), (0, wmain.height), (wmain.width, 0), (wmain.width, wmain.height)]

                transformed_gaze_point = transform_gaze_to_screen_space((x_point, y_point), src_points, dst_points)

                # Draw a circle on the gaze point
                cv2.circle(frame, transformed_gaze_point, 5, (255, 255, 255), -1)

                # # print(xvec, yvec, zvec)
                # if math.isnan(xvec): xvec = 0
                # if math.isnan(yvec): yvec = 0

                # rx = int(xvec/3 * wm.width / 2) + 1920//2
                # ry = 1080//2 - int(yvec/1.8 * wm.height)

                # # what is width of the moinitor?
                # mw = topright[0] - botleft[0]
                # mh = -botright[1] + topleft[1]
                # if mw < 0: mw = 1
                # if mh < 0: mh = 1

                # find center

                # assume max left = -2.5, max right = 2.5
                # max bot = -1.8, max top = 1.8
                # then remap inputs relative to scale
                # sx, sy = (xvec+2.5)/5, (yvec+1.8)/3.6
                # # finding the final coords
                # ex, ey = int(sx * mw + topleft[0]), int(sy * mh + topleft[1]) - mh//2
                # fx, fy = map(int, EYE_SMOOTHER.next_pos(ex, ey))

                # # ratio for real screen coords
                # rx, ry = fx / (mw if mw > 1 else 1), fy / (mh if mh > 1 else 1)
                # mag = math.sqrt(xvec**2 + yvec**2)
                # if mag <= 0: mag = 1
                # MS = 30
                # drx, dry = xvec / mag * MS, yvec / mag * MS
                
                # # move to screen coords
                # px, py = pyautogui.position()
                # MOUSE.move(drx, -dry)
                # print(xvec, yvec, drx, dry)

                # cv2.circle(frame, (fx, fy), 5, (0, 255, 0), -1)

            else:
                pass

            # cv2.imshow('result', result)
            cv2.imshow('frame', frame)

            # wait for key press
            if cv2.waitKey(1) == ord('q'):
                break

    except (KeyboardInterrupt, SystemExit) as e:
        print(e)
        frontend.shutdown()
        cap.release()
        cv2.destroyAllWindows()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)




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





