from pathlib import Path
import joblib

from preprocess import clean_text


# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Load saved model and vectorizer
model = joblib.load(MODEL_DIR / "fake_news_model.pkl")
tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")


def predict_news(news_text):
    """
    Predict whether the given news is Fake or Real.
    """

    # Apply the same preprocessing used during training
    cleaned_text = clean_text(news_text)

    # Convert text into TF-IDF features
    text_vector = tfidf.transform([cleaned_text])

    # Prediction
    prediction = model.predict(text_vector)[0]

    # Confidence score
    probabilities = model.predict_proba(text_vector)[0]
    confidence = max(probabilities) * 100

    if prediction == 0:
        label = "FAKE"
    else:
        label = "REAL"

    return label, confidence


if __name__ == "__main__":

    print("=" * 60)
    print("FAKE NEWS DETECTOR")
    print("=" * 60)

    news = input("\nEnter news article/headline: ")

    if not news.strip():
        print("Please enter some news text.")
    else:
        label, confidence = predict_news(news)

        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)

        print("Prediction :", label)
        print(f"Confidence : {confidence:.2f}%")

        print("=" * 60)