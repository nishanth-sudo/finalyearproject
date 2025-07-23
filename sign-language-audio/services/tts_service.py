from gtts import gTTS
import os

def speak_text(text, lang='hi'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    os.system("start output.mp3")  # Windows
