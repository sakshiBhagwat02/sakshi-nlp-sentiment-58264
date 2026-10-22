import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

reviews = [
    "This movie is very good",
    "I love this film",
    "Amazing movie",
    "Excellent acting",
    "This movie is very bad",
    "I hate this film",
    "Terrible movie",
    "Worst movie ever"
]

sentiments = [
    "Positive", "Positive", "Positive", "Positive",
    "Negative", "Negative", "Negative", "Negative"
]

model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression()
)

model.fit(reviews, sentiments)

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment.")

review = st.text_area("Enter your movie review")

if st.button("Analyze Sentiment"):
    if review.strip():
        result = model.predict([review])[0]

        if result == "Positive":
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😞")
    else:
        st.warning("Please enter a review.")