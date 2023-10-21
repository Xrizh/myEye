import cv2
import mediapipe as mp
import pyautogui
import time

# Global variables
hand_state = {'left': 'unknown', 'right': 'unknown'}
eye_state = {'left': 'open', 'right': 'open'}


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

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

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
        pyautogui.click(button='left')
    elif eye_state['right'] == 'closed' and eye_state['left'] == 'open':
        pyautogui.click(button='right')
