from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# Add src folder to Python path
sys.path.append(str(BASE_DIR / "src"))


# ---------------------------------------------------------
# Import prediction function
# ---------------------------------------------------------

from predict import predict_news


# ---------------------------------------------------------
# Create Flask application
# ---------------------------------------------------------

app = Flask(__name__)

# Allow React frontend to communicate with Flask
CORS(app)


# ---------------------------------------------------------
# Home / Health Check
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Fake News Detector API is running successfully!"
    })


# ---------------------------------------------------------
# News Prediction API
# ---------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data or "news" not in data:
        return jsonify({
            "error": "Please provide news text."
        }), 400

    news_text = data["news"]

    label, confidence, influential_words = predict_news(news_text)

    if label == "INVALID":
        return jsonify({
            "error": "Please enter valid news text."
        }), 400

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 2),
        "model_signals": influential_words
    })


# ---------------------------------------------------------
# Run Flask Server
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("FAKE NEWS DETECTOR - FLASK API")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )