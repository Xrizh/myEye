import time
import adhawkapi
import adhawkapi.frontend
import numpy as np
import cv2

# Initialize gaze vector components to some default values
xvec, yvec = 0.0, 0.0  

class FrontendData:
    def __init__(self):
        global xvec, yvec  
        self._api = adhawkapi.frontend.FrontendApi(ble_device_name='ADHAWK MINDLINK-296')
        self._api.register_stream_handler(adhawkapi.PacketType.EYETRACKING_STREAM, self._handle_et_data)
        self._api.register_stream_handler(adhawkapi.PacketType.EVENTS, self._handle_events)
        self._api.start(tracker_connect_cb=self._handle_tracker_connect,
                        tracker_disconnect_cb=self._handle_tracker_disconnect)

    def shutdown(self):
        self._api.shutdown()

    @staticmethod
    def _handle_et_data(et_data: adhawkapi.EyeTrackingStreamData):
        global frame, xvec, yvec  
        if et_data.gaze is not None:
            xvec, yvec, zvec, vergence = et_data.gaze
            print(f'Gaze={xvec:.2f},y={yvec:.2f},z={zvec:.2f},vergence={vergence:.2f}')
            cv2.circle(frame, (int(xvec), int(yvec)), radius=10, color=(0, 0, 255), thickness=-1)


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

def main():
    global frame, xvec, yvec  
    frontend = FrontendData()
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            h, w, c = frame.shape
            
            print(f'Updated Gaze x={xvec}, y={yvec}')  # Debug line to see if xvec, yvec are updated

            x_point = int(w * (xvec + 1) / 2)
            y_point = int(h * (yvec + 1) / 2)

            cv2.circle(frame, (x_point, y_point), 5, (0, 0, 255), -1)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) == ord('q'):
                break

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) == ord('q'):
                break
                
    except (KeyboardInterrupt, SystemExit):
        frontend.shutdown()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
