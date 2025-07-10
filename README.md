# Sentiment Analysis on Movie Reviews

This project builds a sentiment analysis system for movie reviews using both traditional machine learning (Logistic Regression) and AI (BERT transformer) models.

## Project Structure

```
Sentiment_Analysis/
  ├── data/
  │     └── IMDB Dataset.csv (download from Kaggle)
  ├── src/
  │     ├── preprocess.py
  │     ├── baseline_model.py
  │     └── bert_model.py
  ├── requirements.txt
  └── README.md
```

## Setup

1. **Download the IMDb 50k dataset** from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in `data/` as `IMDB Dataset.csv`.
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Preprocess the data:**
   ```bash
   python src/preprocess.py
   ```
   This will create `data/IMDB_Dataset_clean.csv`.

## Usage

### Baseline Model (Logistic Regression)
```bash
python src/baseline_model.py
```

### BERT Model (Hugging Face Transformers)
```bash
python src/bert_model.py
```

## Output
- Both scripts print accuracy, F1-score, and a classification report on the test set.

## Notes
- The BERT model script may take a while to run, as it uses a transformer for each review.
- For best results, ensure you have a GPU for the BERT model (optional).

## Future Improvements
- Add a web or CLI interface for real-time predictions.
- Experiment with fine-tuning BERT on the IMDb dataset.


