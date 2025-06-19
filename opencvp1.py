import cv2
import numpy as np
import time

# Load video
video_path = "volleyball_match.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties for FPS synchronization
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)

# Ball color range (refined to avoid yellow jerseys)
lower_ball = np.array([18, 160, 160])  # tighter hue/sat/val
upper_ball = np.array([30, 255, 255])

# Background subtractor for player detection
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=60)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Ball trail
trajectory = []
max_trail = 5

# Output video (optional)
write_output = True
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))
    display = frame.copy()

    # ========== Ball Detection ==========
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_ball, upper_ball)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest)
        area = cv2.contourArea(largest)

        # Apply stricter area and shape checks to exclude jerseys
        if 5 < radius < 25 and area < 1500:
            center = (int(x), int(y))
            trajectory.append(center)
            if len(trajectory) > max_trail:
                trajectory.pop(0)
            cv2.circle(display, center, int(radius), (0, 255, 255), -1)

    # ========== Trajectory Trail ==========
    for i in range(1, len(trajectory)):
        if trajectory[i - 1] and trajectory[i]:
            cv2.line(display, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)

    # ========== Player Detection ==========
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    player_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = h / float(w)
            if 1.4 < aspect_ratio < 4.5:
                cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)
                player_count += 1

    cv2.putText(display, f"Players per team: {player_count // 2}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ========== Save and Display ==========
    if write_output:
        if out is None:
            out = cv2.VideoWriter("output_synced.mp4", fourcc, fps, (display.shape[1], display.shape[0]))
        out.write(display)

    cv2.imshow("Volleyball Tracker", display)

    # Ensure real-time playback
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
