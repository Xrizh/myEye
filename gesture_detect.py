import cv2
import mediapipe as mp
import pyautogui


def is_closed_fist(hand_landmarks):
    tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    pip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
    tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    
    if tip_x < pip_x and tip_y > pip_y:
        return True
    return False
    
def is_finger_extended(finger_tip, finger_dip, finger_mcp):
    return finger_tip.y < finger_dip.y and finger_dip.y < finger_mcp.y


def scroll_up():
    pyautogui.scroll(80)

def scroll_down():
    pyautogui.scroll(-80)

def minecraft_left_click():
    pyautogui.mouseDown(button='left')

def minecraft_left_release():
    pyautogui.mouseUp(button='left')

def minecraft_right_click():
    pyautogui.mouseDown(button='right')

def minecraft_right_release():
    pyautogui.mouseUp(button='right')

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
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    is_index_extended = is_finger_extended(index_finger_tip, index_finger_dip, index_finger_mcp)
    is_middle_extended = is_finger_extended(middle_finger_tip, middle_finger_dip, middle_finger_mcp)
    
    if is_index_extended and is_middle_extended:
        return True
    return False

def is_zoom_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    distance = calculate_distance(thumb_tip, index_finger_tip)
    
    # You can decide on a threshold distance for zoom action here
    return True if distance > SOME_THRESHOLD else False

# def is_thumb_up(hand_landmarks):
#     # Similar to is_closed_fist, but for thumb up gesture
#     return True if CONDITION else False
# def is_rock_sign(hand_landmarks):
#     # Logic to detect rock sign based on landmark positions
#     return True if CONDITION else False

def is_open_fist(hand_landmarks):
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

    if is_index_extended and is_middle_extended and is_ring_extended and is_pinky_extended:
        return True
    return False
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()

hand_state = {'left': 'unknown', 'right': 'unknown'}

while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            hand_type = 'left' if handedness.classification[0].label == 'Left' else 'right'

            if is_two_finger_scroll(hand_landmarks):
                if hand_state[hand_type] != "two_finger_scroll":
                    print(f"{hand_type.capitalize()} Hand: Two Finger Scroll")
                    hand_state[hand_type] = "two_finger_scroll"
                    if hand_type == 'right':
                        pyautogui.scroll(10)
                    elif hand_type == 'left':
                        pyautogui.scroll(-10)

            elif is_closed_fist(hand_landmarks):
                if hand_state[hand_type] != "closed_fist":
                    print(f"{hand_type.capitalize()} Hand: Closed Fist")
                    hand_state[hand_type] = "closed_fist"
                    if hand_type == 'right':
                        pyautogui.mouseDown(button='left')
                    elif hand_type == "left":
                        pyautogui.click(button="right")
            
            else:
                if hand_state[hand_type] != "unknown":
                    print(f"{hand_type.capitalize()} Hand: Unknown Gesture")
                    hand_state[hand_type] = "unknown"
                    if hand_type == 'right':
                        pyautogui.mouseUp(button='left')
            
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
