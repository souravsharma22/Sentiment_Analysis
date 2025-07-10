import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline, AutoTokenizer
from preprocess import clean_text

# Load input data
input_df = pd.read_csv('data/input.csv')

# Preprocess reviews
input_df['clean_review'] = input_df['review'].apply(clean_text)

# --- Baseline Model Prediction ---
# Load cleaned training data for fitting vectorizer and model
train_df = pd.read_csv('data/IMDB_Dataset_clean.csv')
X_train = train_df['review']
y_train = train_df['sentiment'].map({'positive': 1, 'negative': 0})

# Fit TF-IDF and Logistic Regression (for demo; in production, save/load models)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Transform input reviews and predict
X_input_vec = vectorizer.transform(input_df['clean_review'])
preds_baseline = model.predict(X_input_vec)
input_df['sentiment_baseline'] = preds_baseline
input_df['sentiment_baseline'] = input_df['sentiment_baseline'].map({1: 'positive', 0: 'negative'})

# --- BERT Model Prediction ---
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

def truncate_review(text, max_tokens=512):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)

input_df['review_trunc'] = input_df['review'].apply(truncate_review)
bert_preds = input_df['review_trunc'].apply(lambda x: classifier(x)[0]['label'].lower())
input_df['sentiment_bert'] = bert_preds

# Save results
input_df.drop(columns=['clean_review', 'review_trunc'], inplace=True)
input_df.to_csv('data/input_with_sentiment.csv', index=False)
print('Predictions saved to data/input_with_sentiment.csv') 