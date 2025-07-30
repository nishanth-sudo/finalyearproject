import cv2
import numpy as np
import mediapipe as mp
import joblib
import pandas as pd

# Load the trained model and label encoder
model, label_encoder = joblib.load("C:/Users/abc/Desktop/ISL/Sign-Language-Detection/models/gesture_model.pkl")

# MediaPipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame to avoid mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Check if correct shape
            if len(landmarks) == 63:
                # Pad with zeros to make 126 if using 2-hand model format
                landmarks.extend([0.0] * 63)

            elif len(landmarks) == 126:
                pass  # Perfect!

            else:
                continue  # Skip if malformed

            columns = [f"f{i}" for i in range(126)]
            X_input = pd.DataFrame([landmarks], columns=columns)

            # Predict
            prediction = model.predict(X_input)
            prediction_text = label_encoder.inverse_transform(prediction)[0]


            # Show prediction on screen
            cv2.putText(frame, f'Prediction: {prediction_text}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    else:
        cv2.putText(frame, "No hand detected", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("ISL Real-Time Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
