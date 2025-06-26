#Prepare a model for sentiment analysis of Movies dataset
#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
#Importing Libraries required for NLP
import nltk # Natural Language Toolkit
import re # Regular Expression
#Library for importing Stopwords
from nltk.corpus import stopwords
#Download Stopwords
nltk.download("stopwords")
stop_words = stopwords.words("english")
print(stop_words)
#Load the dataset into dataframe
df = pd.read_csv("IMDB Dataset.csv")
df. head()
#Preparing the Sentiments to some numerical value
df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})
#Clean the Text
def clean_text(text):
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)
df["cleaned_review"] = df["review"].apply(clean_text)
df["cleaned_review"].head()
#Feature Extraction 
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["cleaned_review"])
y = df["sentiment"]
#Divide the dataset into train_test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#Train the model 
model = MultinomialNB()
model.fit(X_train, y_train)
#Make the predictions 
y_pred = model.predict(X_test)
#Calculate the performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
#Print the performance metrics parameters
print ("Accuracy: ",accuracy)
print ("Precision: ",precision)
print ("Recall: ",recall)
print ("F1 Score: ",f1)
print ("*****Confusion Matrix is : *****")
print (cm)
print ("*****Classification Report is : *****")
print (cr)


#Save the model and the vectorizer
joblib.dump(model,"sentiment_model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")

print("Model and vectorizer saved successfully!")