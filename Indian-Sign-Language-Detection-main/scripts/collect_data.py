import cv2
import mediapipe as mp
import pandas as pd
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Set the gesture label here
gesture_label = 'Z'  # 👈 Change this for each alphabet

# Data collection variables
data = []
recording = False

print("Press 's' to start/stop recording, 'q' to quit.")

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_features = []
    handedness = {'Left': None, 'Right': None}

    if result.multi_hand_landmarks and result.multi_handedness:
        for idx, hand_handedness in enumerate(result.multi_handedness):
            label = hand_handedness.classification[0].label  # 'Left' or 'Right'
            handedness[label] = result.multi_hand_landmarks[idx]

        for side in ['Right', 'Left']:  # fixed order
            hand_landmarks = handedness[side]
            if hand_landmarks:
                for lm in hand_landmarks.landmark:
                    current_features.extend([lm.x, lm.y, lm.z])
            else:
                current_features.extend([0] * 63)  # No hand detected → pad with zeros

        # Draw landmarks
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        if recording:
            data.append(current_features + [gesture_label])

    # Display recording status
    cv2.putText(frame, f"Recording: {'ON' if recording else 'OFF'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if recording else (0, 0, 255), 2)

    cv2.imshow("Two-Hand Gesture Capture", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        recording = not recording
        print("Recording Started" if recording else "Recording Stopped")
        time.sleep(1)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV
df = pd.DataFrame(data)
df.to_csv(f"C:/Users/abc/Desktop/ISL/Sign-Language-Detection/data/alphabet/{gesture_label}.csv", index=False)
print(f"Saved {len(data)} samples for label: {gesture_label}")
