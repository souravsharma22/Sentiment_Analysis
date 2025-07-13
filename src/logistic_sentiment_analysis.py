import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

#Load dataset
df=pd.read_csv("data/IMDB_Dataset_clean.csv")  

#Encode sentiment labels(e.g.,'positive'->1,'negative'->0)
label_encoder=LabelEncoder()
df['sentiment_encoded']=label_encoder.fit_transform(df['sentiment'])

#Split into training and test sets
X_train,X_test,y_train,y_test=train_test_split(
    df['review'],df['sentiment_encoded'],test_size=0.2,random_state=42)

#Build pipeline with TF-IDF vectorizer and Logistic Regression
pipeline=Pipeline([
    ('tfidf',TfidfVectorizer(stop_words='english',max_features=5000)),
    ('clf',LogisticRegression(max_iter=1000))])

#Train the model
pipeline.fit(X_train,y_train)

#Predict on test set
y_pred=pipeline.predict(X_test)

#Evaluate the model
accuracy=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred,target_names=label_encoder.classes_)

#Output results
print(f"Accuracy :- {accuracy:.4f}")
print()
print("Classification Report:-")
print(report)

#FOR SAVING THE TRAINED LOGISTIC MODEL
import joblib
joblib.dump(pipeline, "sentiment_model.pkl")

model=joblib.load("sentiment_model.pkl")
print(model.predict(["This movie was awesome!"]))
