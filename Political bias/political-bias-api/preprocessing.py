import re
import string
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from contractions import fix as fix_contractions

# Download resources for convenience and faster application
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Faster

def basic_clean(text):
    text = str(text).lower()
    text = fix_contractions(text)                     # Expand contractions
    text = re.sub(r"http\S+|www.\S+", "", text)       # Remove URLs
    text = re.sub(r"<.*?>", "", text)                 # Remove HTML
    text = re.sub(r"\n", " ", text)                   # Remove line breaks
    text = re.sub(r"[^a-z\s]", "", text)              # Remove punctuation, digits, etc.
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])

def preprocess_pipeline(text, remove_stop=True):
    text = basic_clean(text)
    text = lemmatize(text)
    if remove_stop:
        text = remove_stopwords(text)
    return text
