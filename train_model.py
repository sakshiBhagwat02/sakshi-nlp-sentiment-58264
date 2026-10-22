print("NLP Sentiment Analysis: Training started")

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
print("Loading IMDB dataset...")
df = pd.read_csv("IMDB Dataset.csv")

# 2. Clean data
df = df[["review", "sentiment"]].dropna()
df["sentiment"] = df["sentiment"].str.lower().str.strip()
df = df[df["sentiment"].isin(["positive", "negative"])]

print("Total reviews:", len(df))

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["review"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)

# 4. Create model
model = make_pipeline(
    TfidfVectorizer(
        max_features=20000,
        stop_words="english"
    ),
    LogisticRegression(max_iter=1000)
)

# 5. Train model
print("Training model. Please wait...")
model.fit(X_train, y_train)

# 6. Test model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 7. Save trained model
joblib.dump(model, "sentiment_model.pkl")

print("Model saved successfully!")
print("Training completed!")