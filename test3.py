''' Demonstrates how to subscribe to and handle data from gaze and event streams '''

import time
import adhawkapi
import adhawkapi.frontend
import numpy as np

import cv2
from roboflow import Roboflow
import cv2


# ----------------- me -----------------

# Initialize Roboflow
rf = Roboflow(api_key="vWhscCBeLxNi4A38zyTf")
project = rf.workspace().project("screen-detector-v")
model = project.version(1).model


class FrontendData:
    ''' BLE Frontend '''

    def __init__(self):
        # Instantiate an API object
        # TODO: Update the device name to match your device
        self._api = adhawkapi.frontend.FrontendApi(ble_device_name='ADHAWK MINDLINK-296')

        # Tell the api that we wish to receive eye tracking data stream
        # with self._handle_et_data as the handler
        self._api.register_stream_handler(adhawkapi.PacketType.EYETRACKING_STREAM, self._handle_et_data)

        # Tell the api that we wish to tap into the EVENTS stream
        # with self._handle_events as the handler
        self._api.register_stream_handler(adhawkapi.PacketType.EVENTS, self._handle_events)

        # Start the api and set its connection callback to self._handle_tracker_connect/disconnect.
        # When the api detects a connection to a MindLink, this function will be run.
        self._api.start(tracker_connect_cb=self._handle_tracker_connect,
                        tracker_disconnect_cb=self._handle_tracker_disconnect)

    def shutdown(self):
        '''Shutdown the api and terminate the bluetooth connection'''
        self._api.shutdown()

    @staticmethod
    def _handle_et_data(et_data: adhawkapi.EyeTrackingStreamData):
        ''' Handles the latest et data '''
        if et_data.gaze is not None:
            xvec, yvec, zvec, vergence = et_data.gaze
            print(f'Gaze={xvec:.2f},y={yvec:.2f},z={zvec:.2f},vergence={vergence:.2f}')

        # if et_data.eye_center is not None:
        #     if et_data.eye_mask == adhawkapi.EyeMask.BINOCULAR:
        #         rxvec, ryvec, rzvec, lxvec, lyvec, lzvec = et_data.eye_center
        #         print(f'Eye center: Left=(x={lxvec:.2f},y={lyvec:.2f},z={lzvec:.2f}) '
        #               f'Right=(x={rxvec:.2f},y={ryvec:.2f},z={rzvec:.2f})')

        # if et_data.pupil_diameter is not None:
        #     if et_data.eye_mask == adhawkapi.EyeMask.BINOCULAR:
        #         rdiameter, ldiameter = et_data.pupil_diameter
        #         print(f'Pupil diameter: Left={ldiameter:.2f} Right={rdiameter:.2f}')

        # if et_data.imu_quaternion is not None:
        #     if et_data.eye_mask == adhawkapi.EyeMask.BINOCULAR:
        #         x, y, z, w = et_data.imu_quaternion
        #         print(f'IMU: x={x:.2f},y={y:.2f},z={z:.2f},w={w:.2f}')

    @staticmethod
    def _handle_events(event_type, timestamp, *args):
        if event_type == adhawkapi.Events.BLINK:
            duration = args[0]
            print(f'Got blink: {timestamp} {duration}')
        if event_type == adhawkapi.Events.EYE_CLOSED:
            eye_idx = args[0]
            print(f'Eye Close: {timestamp} {eye_idx}')
        if event_type == adhawkapi.Events.EYE_OPENED:
            eye_idx = args[0]
            print(f'Eye Open: {timestamp} {eye_idx}')

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
    ''' App entrypoint '''
    frontend = FrontendData()
    # create an opencv camera instance
    cap = cv2.VideoCapture(0)

    # check if camera is opened
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Create an output video file to save the annotated video
    output_file = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

        
        # Process each frame in the video
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Predict on the current frame
        predictions = model.predict_from_image(frame, confidence=40, overlap=30)

        # Draw bounding boxes on the frame based on predictions
        for prediction in predictions:
            label = prediction['label']
            confidence = prediction['confidence']
            x, y, w, h = prediction['xmin'], prediction['ymin'], prediction['width'], prediction['height']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame with bounding boxes
        cv2.imshow('Object Detection', frame)

        # Check for user input to exit the loop
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release the video capture object and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

    # try:
    #     # run the opencv code
    #     while True:
    #         # read frame from camera
    #         ret, frame = cap.read()

    #         # read frame from camera
    #         frame_box = cap.read()

    #         # check if frame is read correctly
    #         if not ret:
    #             print("Can't receive frame (stream end?). Exiting ...")
    #             break

    #          # Convert to grayscale for edge detection
    #         gray = cv2.cvtColor(frame_box, cv2.COLOR_BGR2GRAY)
    
    #         # Edge detection
    #         edges = cv2.Canny(gray, 50, 150)
    
    #         # Find contours
    #         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #         # If contours were found, proceed
    #         if contours:
    #             # Find the largest contour, assuming it corresponds to the screen
    #             largest_contour = max(contours, key=cv2.contourArea)
            
    #             # Approximate contour to polygon and check if it has 4 sides (likely screen)
    #             epsilon = 0.05 * cv2.arcLength(largest_contour, True)
    #             approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
    #             if len(approx) == 4:
    #                 for corner in approx:
    #                     # Draw red dot for each corner
    #                     cv2.circle(frame_box, (corner[0][0], corner[0][1]), 10, (0, 0, 255), -1)
                    
    #                 # Optional: Draw the screen's outline
    #                 cv2.polylines(frame_box, [approx], True, (0, 255, 0), 2)

    #         # show frame
    #         cv2.imshow('frame', frame)
    #         cv2.imshow('frame', frame_box)


    #         # wait for key press
    #         if cv2.waitKey(1) == ord('q'):
    #             break
        
    # except (KeyboardInterrupt, SystemExit):
    #     frontend.shutdown()
    #     cap.release()
    #     cv2.destroyAllWindows()


if __name__ == '__main__':
    main()