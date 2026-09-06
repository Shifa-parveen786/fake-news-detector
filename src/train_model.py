from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocess import load_and_prepare_data, clean_text


# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


print("=" * 60)
print("FAKE NEWS DETECTOR - MODEL TRAINING")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

print("\n[1/6] Loading dataset...")

data = load_and_prepare_data()

print("Total articles:", len(data))
print("Fake news:", (data["label"] == 0).sum())
print("Real news:", (data["label"] == 1).sum())


# ---------------------------------------------------------
# 2. Text preprocessing
# ---------------------------------------------------------

print("\n[2/6] Preprocessing text...")
print("This may take some time...")

data["cleaned_content"] = data["content"].apply(clean_text)

# Remove empty cleaned articles
data = data[data["cleaned_content"].str.strip() != ""]

print("Preprocessing completed!")
print("Articles after preprocessing:", len(data))


# ---------------------------------------------------------
# 3. Train-test split
# ---------------------------------------------------------

print("\n[3/6] Splitting dataset...")

X = data["cleaned_content"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ---------------------------------------------------------
# 4. TF-IDF Feature Extraction
# ---------------------------------------------------------

print("\n[4/6] Applying TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF completed!")
print("Training feature matrix:", X_train_tfidf.shape)
print("Testing feature matrix:", X_test_tfidf.shape)


# ---------------------------------------------------------
# 5. Train Logistic Regression model
# ---------------------------------------------------------

print("\n[5/6] Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# ---------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------

print("\n[6/6] Evaluating model...")

y_pred = model.predict(X_test_tfidf)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Fake", "Real"]
))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Fake", "Real"],
    yticklabels=["Fake", "Real"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Fake News Detector - Confusion Matrix")

# Save chart
confusion_matrix_path = RESULTS_DIR / "confusion_matrix.png"
plt.savefig(confusion_matrix_path, bbox_inches="tight")

plt.show()
plt.close()

print("\nConfusion Matrix chart saved at:")
print(confusion_matrix_path)


# ---------------------------------------------------------
# Save model and TF-IDF vectorizer
# ---------------------------------------------------------

print("\nSaving model files...")

model_path = MODEL_DIR / "fake_news_model.pkl"
vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"

joblib.dump(model, model_path)
joblib.dump(tfidf, vectorizer_path)

print("\nModel saved at:")
print(model_path)

print("\nTF-IDF vectorizer saved at:")
print(vectorizer_path)

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)