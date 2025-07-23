from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))  # actions = ["hello", "thanks", ...]
    return model
