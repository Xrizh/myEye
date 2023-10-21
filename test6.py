import cv2
from roboflow import Roboflow
import time
import adhawkapi
import adhawkapi.frontend
import numpy as np
import cv2

# Initialize Roboflow
rf = Roboflow(api_key="vWhscCBeLxNi4A38zyTf")
project = rf.workspace().project("screen-detector-v")
model = project.version(1).model

frame = None  # Declare global frame to be accessed in multiple functions
xvec, yvec = 0.0, 0.0  # Initialize gaze vector components to some default values

COUNTER = 0

NAN = float('nan')
def clamp(_min, _max, val):
    global NAN
    if val < _min:
        return _min
    if val > _max:
        return _max
    return 0 if val == NAN else val

HSV_RANGE = [(110, 0, 0), (130, 100, 100)]
C_LIMIT = 300


def main():
    ''' App entrypoint '''
    global xvec, yvec  # Declare these as global
    
    # create an opencv camera instance
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Predict on the current frame
        response = model.predict(frame, confidence=40, overlap=30).json()
        prediction_list = response.get('predictions', [])
        
        # Filter out predictions that aren't labeled as 'screen'
        screen_predictions = [pred for pred in prediction_list if pred['class'] == 'screens']
        
        if not screen_predictions:
            cv2.imshow('Object Detection', frame)
            continue
        
        # Sort predictions by area and take the largest one
        largest_prediction = max(screen_predictions, key=lambda pred: pred['width'] * pred['height'])
        
        confidence = largest_prediction['confidence']

        # Check if the confidence is above a certain threshold (e.g., 0.5 or 50% confidence)
        if confidence > 0.5:
            x, y, w, h = int(largest_prediction['x']), int(largest_prediction['y']), int(largest_prediction['width']), int(largest_prediction['height'])
            
            # Draw the bounding box on the frame
            pt1 = (int(x - w/2), int(y - h/2))
            pt2 = (int(x + w/2), int(y + h/2))
            frame = cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2)

        # Show the frame with or without the bounding box
        cv2.imshow('Object Detection', frame)

        # Check for user input to exit the loop
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release the video capture object and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()