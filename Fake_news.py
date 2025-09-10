import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load dataset (CSV should have 'text' & 'label' columns)
df = pd.read_csv("dataset.csv")
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(x_train)
tfidf_test = vectorizer.transform(x_test)

# Train Model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

# Accuracy
y_pred = model.predict(tfidf_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test Prediction
news = input("Enter a news text: ")
pred = model.predict(vectorizer.transform([news]))
print("Prediction:", pred[0])
