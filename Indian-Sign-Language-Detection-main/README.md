# Indian Sign Language Detection using Hand Gestures (A–Z)

This project is focused on detecting **Indian Sign Language (ISL)** alphabets (A–Z) in real-time using hand landmarks captured via MediaPipe and classified using a **Random Forest Classifier**.

🚧 **Note:** This project is currently in development and supports only **alphabets**. Word and sentence recognition will be added soon.


## 📌 Features (so far)

- Real-time prediction of ISL alphabets using webcam.
- Trained on over 800 samples per alphabet.
- Achieved **99.94% accuracy** on test data.
- Live prediction with confidence filtering and preprocessing.
- Data captured and combined into a unified CSV file for model training.


## 🧠 Model Details

- **Algorithm:** Random Forest Classifier (Scikit-Learn)
- **Input Features:** 126 hand landmark coordinates (42 points × x, y, z)
- **Output Labels:** 26 uppercase alphabets (A to Z)
- **Accuracy:** 99.94% on test set

## 🚀 How to Run
1. Collect Data:
```
bash
python scripts/collect_data.py
```
2. Train Model:
```
bash
python scripts/train_alphabet_model.py
```
3. Run Real-Time Prediction:
```
bash
python scripts/realtime_predict.py
```

## 📜 License
This project is open source and available under the MIT License.
