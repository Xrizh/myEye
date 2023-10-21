



# write a program that opens a webcam and displays the video stream and then allows the user to find hsv values at the cursor position on a click

import cv2

# initialize the camera
cap = cv2.VideoCapture(0)

# check if camera is opened
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# create a window
cv2.namedWindow("frame")

# create a mouse callback function

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button of the mouse is clicked - position (", x, ", ", y, ")")
        # get hsv value
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        print("HSV: ", hsv[y, x])

# set mouse callback function for window
cv2.setMouseCallback("frame", mouse_callback)

# run the opencv code
while True:
    # read frame from camera
    ret, frame = cap.read()

    # check if frame is read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # show frame
    cv2.imshow("frame", frame)

    # check for user input to exit the loop
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# release the video capture object and close the opencv window
cap.release()