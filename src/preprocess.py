import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in STOPWORDS]
    # Join back to string
    return ' '.join(tokens)

def preprocess_dataframe(df, text_column):
    df[text_column] = df[text_column].apply(clean_text)
    return df

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('data/IMDB Dataset.csv')
    df = preprocess_dataframe(df, 'review')
    df.to_csv('data/IMDB_Dataset_clean.csv', index=False)
    print('Preprocessing complete. Cleaned data saved to data/IMDB_Dataset_clean.csv') 