from deep_translator import GoogleTranslator

text = "Hello, how are you?"

translated = GoogleTranslator(source="en", target="hi").translate(text)

print(translated)