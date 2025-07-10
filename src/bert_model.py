import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

tqdm.pandas()

# Load cleaned data
df = pd.read_csv('data/IMDB_Dataset_clean.csv')

# Features and labels
X = df['review']
y = df['sentiment'].map({'positive': 1, 'negative': 0})

# Train/test split (same as baseline)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load Hugging Face sentiment-analysis pipeline and tokenizer
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

def get_label(pred):
    return 1 if pred['label'] == 'POSITIVE' else 0

def truncate_review(text, max_tokens=512):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)

# Truncate reviews by tokens before prediction
X_test_trunc = X_test.apply(truncate_review)

# Run predictions (may take a while)
preds = X_test_trunc.progress_apply(lambda x: classifier(x)[0])
y_pred = preds.apply(get_label)

# Evaluation
print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test, y_pred))
print('\nClassification Report:\n', classification_report(y_test, y_pred, target_names=['negative', 'positive'])) 