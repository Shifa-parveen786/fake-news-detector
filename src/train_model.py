from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocess import load_and_prepare_data, clean_text


# =========================================================
# PROJECT FOLDERS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


print("=" * 65)
print("FAKE NEWS DETECTOR - MODEL TRAINING & COMPARISON")
print("=" * 65)


# =========================================================
# 1. LOAD DATASET
# =========================================================

print("\n[1/7] Loading dataset...")

data = load_and_prepare_data()

print("Total articles:", len(data))
print("Fake news:", (data["label"] == 0).sum())
print("Real news:", (data["label"] == 1).sum())


# =========================================================
# 2. TEXT PREPROCESSING
# =========================================================

print("\n[2/7] Preprocessing text...")
print("This may take some time...")

data["cleaned_content"] = data["content"].apply(clean_text)

# Remove empty cleaned articles
data = data[
    data["cleaned_content"].str.strip() != ""
].reset_index(drop=True)

print("Preprocessing completed!")
print("Articles after preprocessing:", len(data))


# =========================================================
# 3. TRAIN-TEST SPLIT
# =========================================================

print("\n[3/7] Splitting dataset...")

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


# =========================================================
# 4. TF-IDF FEATURE EXTRACTION
# =========================================================

print("\n[4/7] Applying TF-IDF...")

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


# =========================================================
# 5. DEFINE MACHINE LEARNING MODELS
# =========================================================

print("\n[5/7] Preparing machine learning models...")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Multinomial Naive Bayes": MultinomialNB(),

    "Passive Aggressive": PassiveAggressiveClassifier(
        max_iter=1000,
        random_state=42
    )
}

print("Models selected:")
for model_name in models:
    print("-", model_name)


# =========================================================
# 6. TRAIN AND COMPARE MODELS
# =========================================================

print("\n[6/7] Training and comparing models...")

results = []
trained_models = {}

for model_name, current_model in models.items():

    print("\n" + "-" * 65)
    print("Training:", model_name)
    print("-" * 65)

    # Train model
    current_model.fit(X_train_tfidf, y_train)

    # Prediction
    y_pred = current_model.predict(X_test_tfidf)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # Store results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[model_name] = current_model


# =========================================================
# MODEL COMPARISON TABLE
# =========================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(results_df.to_string(index=False))


# =========================================================
# SELECT BEST MODEL
# =========================================================

best_row = results_df.loc[
    results_df["F1 Score"].idxmax()
]

best_model_name = best_row["Model"]
best_model = trained_models[best_model_name]

print("\n" + "=" * 65)
print("BEST MODEL")
print("=" * 65)

print("Selected Model:", best_model_name)
print(f"Accuracy      : {best_row['Accuracy']:.4f}")
print(f"Precision     : {best_row['Precision']:.4f}")
print(f"Recall        : {best_row['Recall']:.4f}")
print(f"F1 Score      : {best_row['F1 Score']:.4f}")


# =========================================================
# SAVE COMPARISON RESULTS
# =========================================================

comparison_path = RESULTS_DIR / "model_comparison.csv"

results_df.to_csv(
    comparison_path,
    index=False
)

print("\nModel comparison saved at:")
print(comparison_path)


# =========================================================
# MODEL COMPARISON CHART
# =========================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

results_plot = results_df.set_index("Model")[metrics]

results_plot.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Fake News Detector - Model Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend()

comparison_chart_path = RESULTS_DIR / "model_comparison.png"

plt.savefig(
    comparison_chart_path,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nModel comparison chart saved at:")
print(comparison_chart_path)


# =========================================================
# BEST MODEL - DETAILED EVALUATION
# =========================================================

print("\n" + "=" * 65)
print("DETAILED EVALUATION OF BEST MODEL")
print("=" * 65)

best_predictions = best_model.predict(X_test_tfidf)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        best_predictions,
        target_names=["Fake", "Real"]
    )
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    best_predictions
)

print("Confusion Matrix:")
print(cm)


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
plt.title(
    f"Confusion Matrix - {best_model_name}"
)

confusion_matrix_path = RESULTS_DIR / "confusion_matrix.png"

plt.savefig(
    confusion_matrix_path,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nConfusion Matrix chart saved at:")
print(confusion_matrix_path)


# =========================================================
# SAVE BEST MODEL AND TF-IDF VECTORIZER
# =========================================================

print("\n" + "=" * 65)
print("SAVING FINAL MODEL")
print("=" * 65)

model_path = MODEL_DIR / "fake_news_model.pkl"

vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"

joblib.dump(
    best_model,
    model_path
)

joblib.dump(
    tfidf,
    vectorizer_path
)

print("\nBest model saved at:")
print(model_path)

print("\nTF-IDF vectorizer saved at:")
print(vectorizer_path)


# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n" + "=" * 65)
print("TRAINING AND MODEL COMPARISON COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nFinal model:", best_model_name)
print("The selected model is based on the highest F1 Score.")