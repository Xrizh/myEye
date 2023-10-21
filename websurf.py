import cv2
import mediapipe as mp
import pyautogui
import time
import adhawkapi
import adhawkapi.frontend

# Global variables
hand_state = {'left': 'unknown', 'right': 'unknown'}
eye_state = {'left': 'open', 'right': 'open'}
last_double_blink_time = 0

# Initialize AdHawk
class FrontendData:
    def __init__(self):
        self._api = adhawkapi.frontend.FrontendApi(ble_device_name='ADHAWK MINDLINK-296')
        self._api.register_stream_handler(adhawkapi.PacketType.EYETRACKING_STREAM, self._handle_et_data)
        self._api.register_stream_handler(adhawkapi.PacketType.EVENTS, self._handle_events)
        self._api.start(tracker_connect_cb=self._handle_tracker_connect, tracker_disconnect_cb=self._handle_tracker_disconnect)

    def shutdown(self):
        self._api.shutdown()

    @staticmethod
    def _handle_et_data(et_data: adhawkapi.EyeTrackingStreamData):
        pass

    @staticmethod
    def _handle_events(event_type, timestamp, *args):
        global last_double_blink_time
        if event_type == adhawkapi.Events.BLINK:
            duration = args[0]
            current_time = time.time()
            if current_time - last_double_blink_time < 0.5:
                pyautogui.press('enter')
            last_double_blink_time = current_time

        if event_type == adhawkapi.Events.EYE_CLOSED:
            eye_idx = args[0]
            eye = 'left' if eye_idx == 0 else 'right'
            eye_state[eye] = 'closed'

        if event_type == adhawkapi.Events.EYE_OPENED:
            eye_idx = args[0]
            eye = 'left' if eye_idx == 0 else 'right'
            eye_state[eye] = 'open'

    def _handle_tracker_connect(self):
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
        pass

adhawk_frontend = FrontendData()

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
def is_finger_extended(finger_tip, finger_dip, finger_mcp):
    return finger_tip.y < finger_dip.y and finger_dip.y < finger_mcp.y

def is_two_finger_scroll(hand_landmarks):
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
    ring_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]

    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    is_index_extended = is_finger_extended(index_finger_tip, index_finger_dip, index_finger_mcp)
    is_middle_extended = is_finger_extended(middle_finger_tip, middle_finger_dip, middle_finger_mcp)

    is_ring_extended = is_finger_extended(ring_finger_tip, ring_finger_dip, ring_finger_mcp)
    is_pinky_extended = is_finger_extended(pinky_tip, pinky_dip, pinky_mcp)

    if is_index_extended and is_middle_extended and not is_ring_extended and not is_pinky_extended:
        return True
    return False


def websurfing_controller(frame, active_mode):
    if active_mode != 'web':
        return

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            hand_type = 'left' if handedness.classification[0].label == 'Left' else 'right'

            if is_two_finger_scroll(hand_landmarks):
                if hand_state[hand_type] != "two_finger_scroll":
                    hand_state[hand_type] = "two_finger_scroll"
                    if hand_type == 'right':
                        pyautogui.scroll(80)
                    elif hand_type == 'left':
                        pyautogui.scroll(-80)
            else:
                if hand_state[hand_type] != "unknown":
                    hand_state[hand_type] = "unknown"

    # Check for winks and perform mouse clicks
    if eye_state['left'] == 'closed' and eye_state['right'] == 'open':
        pyautogui.click(button='right')  # Right-click if only the right eye is winked
    elif eye_state['right'] == 'closed' and eye_state['left'] == 'open':
        pyautogui.click(button='left')   # Left-click if only the left eye is winked

# Main Loop (Assuming you have a camera setup)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    websurfing_controller(frame, 'web')
    cv2.imshow('MediaPipe Hands', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()
adhawk_frontend.shutdown()
