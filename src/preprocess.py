import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')  # This line is important to resolve your issue

def preprocess_text(text):
    """Process the text by converting to lowercase and tokenizing."""
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalpha()]  # Keep only words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

def preprocess_data(filepath):
    """Load and preprocess the interview response data."""
    df = pd.read_csv(filepath)
    df['response'] = df['response'].apply(preprocess_text)
    return df
