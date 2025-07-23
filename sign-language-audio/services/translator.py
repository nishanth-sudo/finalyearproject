from googletrans import Translator

def translate_text(text, target_lang='hi'):  # hi = Hindi
    translator = Translator()
    translated = translator.translate(text, dest=target_lang)
    return translated.text
