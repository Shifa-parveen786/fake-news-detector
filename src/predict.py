from pathlib import Path
import joblib
import math

from preprocess import clean_text


# ---------------------------------------------------------
# Project folders
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# ---------------------------------------------------------
# Load trained model and TF-IDF vectorizer
# ---------------------------------------------------------

model = joblib.load(MODEL_DIR / "fake_news_model.pkl")
tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")


# ---------------------------------------------------------
# Prediction Function
# ---------------------------------------------------------

def predict_news(news_text):
    """
    Predict whether the given news is Fake or Real.

    Returns:
        label
        confidence
        influential_words
    """

    # Check empty input
    if not news_text or not str(news_text).strip():
        return "INVALID", 0.0, []

    # Convert input to string
    news_text = str(news_text)

    # Apply the same preprocessing used during training
    cleaned_text = clean_text(news_text)

    # Check if anything remains after preprocessing
    if not cleaned_text.strip():
        return "INVALID", 0.0, []

    # Convert cleaned text into TF-IDF features
    text_vector = tfidf.transform([cleaned_text])

    # -----------------------------------------------------
    # Make prediction
    # -----------------------------------------------------

    prediction = model.predict(text_vector)[0]

    # Passive Aggressive uses decision_function
    decision_score = model.decision_function(text_vector)[0]

    # Convert decision score into a confidence-like score
    # This is NOT a calibrated probability.
    score = max(-20, min(20, abs(decision_score)))

    confidence = (
        1 / (1 + math.exp(-score))
    ) * 100

    # Convert numeric prediction into label
    if prediction == 0:
        label = "FAKE"
    else:
        label = "REAL"

    # -----------------------------------------------------
    # Find model-influential words
    # -----------------------------------------------------

    feature_names = tfidf.get_feature_names_out()
    coefficients = model.coef_[0]

    feature_scores = []

    # Get features actually present in the input
    present_features = text_vector.nonzero()[1]

    for index in present_features:

        word = feature_names[index]

        # Contribution of the feature
        score = text_vector[0, index] * coefficients[index]

        feature_scores.append((word, score))

    # Sort according to predicted class
    if prediction == 1:
        feature_scores.sort(
            key=lambda x: x[1],
            reverse=True
        )
    else:
        feature_scores.sort(
            key=lambda x: x[1]
        )

    # Select top 5 influential features
    influential_words = [
        word for word, score in feature_scores[:5]
    ]

    return label, confidence, influential_words


# ---------------------------------------------------------
# Run Prediction from Terminal
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("FAKE NEWS DETECTOR")
    print("=" * 60)

    news = input("\nEnter news article/headline: ")

    label, confidence, influential_words = predict_news(news)

    if label == "INVALID":

        print("\nPlease enter valid news text.")

    else:

        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)

        print("Prediction :", label)
        print(f"Confidence : {confidence:.2f}%")

        if influential_words:
            print(
                "Model Signals :",
                ", ".join(influential_words)
            )

        print("=" * 60)