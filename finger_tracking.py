import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import threading

# Initialize hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Reduce video resolution for better FPS
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Gesture flags
clicking = False
dragging = False

# Cursor smoothing
prev_x, prev_y = 0, 0
alpha = 0.6  # Smoothing factor


def process_frame():
    global prev_x, prev_y, clicking, dragging

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get index finger tip position
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                # Convert to screen coordinates (smooth movement)
                screen_x = np.interp(x, [0, w], [0, screen_w])
                screen_y = np.interp(y, [0, h], [0, screen_h])
                screen_x = alpha * prev_x + (1 - alpha) * screen_x
                screen_y = alpha * prev_y + (1 - alpha) * screen_y
                prev_x, prev_y = screen_x, screen_y

                # Move the cursor
                pyautogui.moveTo(screen_x, screen_y)

                # Get thumb position
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

                # Calculate distance for click
                distance_click = np.linalg.norm([thumb_x - x, thumb_y - y])

                # Left Click (Pinch)
                if 20 < distance_click < 40 and not clicking:
                    clicking = True
                    pyautogui.click()
                elif distance_click > 50:
                    clicking = False  # Reset left-click flag

                # Drag & Drop
                if 20 < distance_click < 40 and not dragging:
                    dragging = True
                    pyautogui.mouseDown()
                elif distance_click > 50 and dragging:
                    dragging = False
                    pyautogui.mouseUp()

        cv2.imshow("Finger Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Run processing in a separate thread for better FPS
thread = threading.Thread(target=process_frame, daemon=True)
thread.start()

# Keep the main thread alive
while thread.is_alive():
    thread.join(0.1)

cap.release()
cv2.destroyAllWindows()
