import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

#Load THE dataset
df=pd.read_csv("/content/IMDB Dataset.csv")  #change the path of your dataset

#Remove the missing or empty data
df.dropna(subset=['review','sentiment'],inplace=True)
df=df[df['review'].str.strip()!='']

#Encode sentiment labels(e.g.,'positive'->1,'negative'->0)
label_encoder=LabelEncoder()
df['sentiment_encoded']=label_encoder.fit_transform(df['sentiment'])

#Split into training and test sets
X_train,X_test,y_train,y_test=train_test_split(df['review'],df['sentiment_encoded'],test_size=0.2,random_state=42)

#Building pipeline with TF-IDF vectorizer and SVM classifier
pipeline=Pipeline([('tfidf',TfidfVectorizer(stop_words='english',max_features=5000)),('clf',LinearSVC(max_iter=1000))])

#Train the SVM model
pipeline.fit(X_train,y_train)

#Prediction on the test set
y_pred=pipeline.predict(X_test)

#Evaluating the model
accuracy=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred,target_names=label_encoder.classes_)

#results
print(f"Accuracy :- {accuracy:.4f}")
print( )
print("Classification Report :-")
print(report)

#For the saving the trained SVM model
import joblib
joblib.dump(pipeline,"svm_sentiment_model.pkl")

#LOADING AND USING THE MODEL LATER
import joblib

#Load the saved model
model=joblib.load("svm_sentiment_model.pkl")

#Example prediction
test_reviews=["This movie was fantastic,Loved the story.","Absolutely terrible.Waste of time.","It was okay,not great but not bad."]

#Predict sentiment(0=negative,1=positive if that's how yours is encoded)
predictions=model.predict(test_reviews)

#Output the predictions
for review,sentiment in zip(test_reviews,predictions):
    print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n")
