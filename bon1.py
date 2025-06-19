import cv2
import mediapipe as mp
import random
import time

# Mediapipe Hand Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

# Optional: Face detection using Haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# RPS Logic
def get_hand_sign(landmarks):
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4

    fingers = []

    # Thumb
    fingers.append(landmarks[thumb_tip].x < landmarks[thumb_tip - 1].x)

    # Other four fingers
    for tip in finger_tips:
        fingers.append(landmarks[tip].y < landmarks[tip - 2].y)

    # Interpretation
    if fingers == [False, False, False, False, False]:
        return "Rock"
    elif fingers == [True, True, True, True, True]:
        return "Paper"
    elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        return "Scissors"
    else:
        return "Unknown"

# Game outcome
def get_winner(player, computer):
    if player == computer:
        return "Draw"
    elif (player == "Rock" and computer == "Scissors") or \
         (player == "Scissors" and computer == "Paper") or \
         (player == "Paper" and computer == "Rock"):
        return "You Win!"
    else:
        return "You Lose!"

# Start camera
cap = cv2.VideoCapture(0)

last_move_time = 0
player_move = "Waiting"
computer_move = "Waiting"
result = "Make your move"

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hands = hands.process(rgb)

    # Optional face detection
    faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 4)
    for (x, y, w1, h1) in faces:
        cv2.rectangle(frame, (x, y), (x + w1, y + h1), (255, 0, 255), 2)
        cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Detect hand and gesture every 2 seconds
    current_time = time.time()
    if result_hands.multi_hand_landmarks and current_time - last_move_time > 2:
        for hand_landmarks in result_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            sign = get_hand_sign(hand_landmarks.landmark)
            if sign in ["Rock", "Paper", "Scissors"]:
                player_move = sign
                computer_move = random.choice(["Rock", "Paper", "Scissors"])
                result = get_winner(player_move, computer_move)
                last_move_time = current_time
            break  # Only process first hand

    # Display moves
    cv2.putText(frame, f"Your Move: {player_move}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Computer: {computer_move}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Result: {result}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 3)

    cv2.imshow("Rock Paper Scissors - Mediapipe + Haarcascade", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
