import cv2
import numpy as np
from collections import deque
from services.keypoint_extractor import extract_keypoints
from services.signlstm import create_model
from services.translator import translate_text
from services.tts_service import speak_text

model = create_model()
model.load_weights('model/sign_lstm.h5')  # Load your trained model
actions = ["hello", "thanks", "good", "you", "name", "i", "what"]  # update as per training

sequence = deque(maxlen=30)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    keypoints = extract_keypoints(frame)
    sequence.append(keypoints)
    
    if len(sequence) == 30:
        prediction = model.predict(np.expand_dims(sequence, axis=0))[0]
        action = actions[np.argmax(prediction)]

        if prediction[np.argmax(prediction)] > 0.8:
            print("Recognized:", action)
            translated = translate_text(action, target_lang='hi')
            print("Translated:", translated)
            speak_text(translated, lang='hi')

    cv2.imshow('Sign to Speech', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
