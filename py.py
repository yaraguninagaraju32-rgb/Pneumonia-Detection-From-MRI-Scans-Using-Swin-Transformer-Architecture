import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize MediaPipe Hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Initialize Webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break
        
    # Flip image for natural 'mirror' interaction
    img = cv2.flip(img, 1)
    img_h, img_w, _ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the frame for hand landmarks
    result = hands.process(rgb_img)
    all_hands = result.multi_hand_landmarks

    if all_hands:
        for hand_landmarks in all_hands:
            # Draw landmarks on the screen 
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract specific landmark coordinates
            # Index finger tip (Landmark 8) and Thumb tip (Landmark 4)
            landmarks = hand_landmarks.landmark
            
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]
            middle_tip = landmarks[12]
            
            # Convert normalized coordinates to pixel coordinates
            x = int(index_tip.x * screen_width)
            y = int(index_tip.y * screen_height)
            
            # 1. CURSOR MOVEMENT [cite: 15]
            # Moves the mouse cursor to the index finger position
            pyautogui.moveTo(x, y, _pause=False)
            
            # 2. PINCH - LEFT CLICK 
            # Calculate distance between thumb and index finger
            distance_click = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
            if distance_click < 0.05:  # Threshold for "Pinch"
                pyautogui.click()
                cv2.putText(img, "LEFT CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 3. TWO-FINGER SWIPE - SCROLL [cite: 37]
            # Calculate distance between index and middle finger
            distance_scroll = math.hypot(index_tip.x - middle_tip.x, index_tip.y - middle_tip.y)
            if distance_scroll < 0.05:
                pyautogui.scroll(10) # Scroll up
                cv2.putText(img, "SCROLLING", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the output
    cv2.imshow("AI Virtual Mouse", img)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()