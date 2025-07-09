import streamlit as st
from transformers import pipeline, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from preprocess import clean_text

# Load and fit baseline model
@st.cache_resource
def load_baseline():
    df = pd.read_csv('data/IMDB_Dataset_clean.csv')
    X = df['review']
    y = df['sentiment'].map({'positive': 1, 'negative': 0})
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)
    model = LogisticRegression(max_iter=200)
    model.fit(X_vec, y)
    return vectorizer, model

# Load BERT pipeline and tokenizer
@st.cache_resource
def load_bert():
    classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    return classifier, tokenizer

def truncate_review(text, tokenizer, max_tokens=512):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)

st.title("Movie Review Sentiment Analysis")

review = st.text_area("Enter a movie review:")

if st.button("Analyze Sentiment") and review.strip():
    # Baseline
    vectorizer, model = load_baseline()
    clean = clean_text(review)
    X_vec = vectorizer.transform([clean])
    pred_baseline = model.predict(X_vec)[0]
    sentiment_baseline = 'positive' if pred_baseline == 1 else 'negative'

    # BERT
    classifier, tokenizer = load_bert()
    review_trunc = truncate_review(review, tokenizer)
    pred_bert = classifier(review_trunc)[0]['label'].lower()

    st.write(f"**Baseline Model:** {sentiment_baseline}")
    st.write(f"**BERT Model:** {pred_bert}") 