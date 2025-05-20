import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Drawing setting-
colors = [(126, 255, 255), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 165, 0)]
current_color_index = 0
eraser_size = 60
buffer_size = 5
position_buffer = []

# Color UI-
x_set, y_set = 20, 20
color_box_size = 50
gap = 30
color_box_positions = [
    (x_set + i * (color_box_size + gap), y_set) for i in range(len(colors))
]

# camera and virtual whiteboard-
canvas = None 
virtual_whiteboard = np.zeros((720, 1280, 3), dtype=np.uint8)  # another virtual whiteboard
last_point = None 

def smooth_position(pos):
    position_buffer.append(pos)
    if len(position_buffer) > buffer_size:
        position_buffer.pop(0)
    avg_x = int(np.mean([p[0] for p in position_buffer]))
    avg_y = int(np.mean([p[1] for p in position_buffer]))
    return (avg_x, avg_y)

def detect_fingers(hand_landmarks):
    tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    bases = [mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
             mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.PINKY_MCP]
    
    up = []
    for tip, base in zip(tips, bases):
        up.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y - 0.02)

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_extended = abs(thumb_tip.x - thumb_ip.x) > 0.04

    return up, thumb_extended

def is_inside_box(point, box_pos, box_size):
    x, y = box_pos
    return x <= point[0] <= x + box_size and y <= point[1] <= y + box_size

def draw_palette(frame):
    """Draw the color palette on top of the frame"""
    for i, color in enumerate(colors):
        top_left = color_box_positions[i]
        bottom_right = (top_left[0] + color_box_size, top_left[1] + color_box_size)
        cv2.rectangle(frame, top_left, bottom_right, color, -1)
        if i == current_color_index:
            cv2.rectangle(frame, (top_left[0]-5, top_left[1]-5),
                          (bottom_right[0]+5, bottom_right[1]+5), (255, 255, 255), 2)

# Camera setups-
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers_up, thumb_extended = detect_fingers(hand_landmarks)

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_pos = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
            smoothed_pos = smooth_position(index_pos)

            # Changing color
            if fingers_up[0] and fingers_up[1] and not any(fingers_up[2:]):
                for i, box_pos in enumerate(color_box_positions):
                    if is_inside_box(index_pos, box_pos, color_box_size):   
                        
                        current_color_index = i

            # Draw with index and middle fingers up
            if fingers_up[0] and not any(fingers_up[1:]):
                if last_point:  
                    cv2.line(canvas, last_point, smoothed_pos, colors[current_color_index], 5)
                    cv2.line(virtual_whiteboard, last_point, smoothed_pos, colors[current_color_index], 5)
                last_point = smoothed_pos
            else:
                last_point = None

            # Erase with 3+ fingers
            if sum(fingers_up[:3]) == 3:
                cv2.circle(canvas, smoothed_pos, eraser_size, (0, 0, 0), -1)
                cv2.circle(virtual_whiteboard, smoothed_pos, eraser_size, (0, 0, 0), -1)

            # Clear with open palm (All fingers up) uses mediapie and opencv
            if all(fingers_up) and thumb_extended:
                if palm_start is None:
                    palm_start = time.time()
                elif time.time() - palm_start > 3:
                    canvas = np.zeros_like(frame)
                    virtual_whiteboard = np.zeros_like(frame)
                    palm_start = None
            else:
                palm_start = None

    # Draw the color 
    draw_palette(frame)
    
    # (whiteboard) for the final output
    output = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    cv2.imshow("Virtual Whiteboard", virtual_whiteboard)  
    cv2.imshow("Drawing with Camera Feed", output) 

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
