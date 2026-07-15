import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()

    tokens = word_tokenize(text)

    words = []

    for word in tokens:
        if word not in string.punctuation and word not in stop_words:
            words.append(word)

    return " ".join(words)