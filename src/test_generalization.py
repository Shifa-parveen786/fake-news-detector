
import pandas as pd
from predict import predict_news
print("Generalization test started...")
from pathlib import Path

test_data = [
    # REAL examples
    {"topic": "Science", "news": "The Earth revolves around the Sun.", "expected": "REAL"},
    {"topic": "Science", "news": "Water freezes at zero degrees Celsius.", "expected": "REAL"},
    {"topic": "Biology", "news": "Plants use sunlight for photosynthesis.", "expected": "REAL"},
    {"topic": "Biology", "news": "The human heart pumps blood.", "expected": "REAL"},
    {"topic": "Technology", "news": "Computers process data using electronic circuits.", "expected": "REAL"},
    {"topic": "Science", "news": "The Moon revolves around the Earth.", "expected": "REAL"},
    {"topic": "Geography", "news": "The Pacific Ocean is the largest ocean on Earth.", "expected": "REAL"},
    {"topic": "Science", "news": "Sound travels through air as a wave.", "expected": "REAL"},
    {"topic": "Biology", "news": "The human body contains many different organs.", "expected": "REAL"},
    {"topic": "Astronomy", "news": "The Sun is a star.", "expected": "REAL"},

    # FAKE examples - fictional claims for classifier testing
    {"topic": "Science", "news": "Scientists confirm that the Moon is made entirely of cheese.", "expected": "FAKE"},
    {"topic": "Science", "news": "Researchers claim that water naturally freezes at 100 degrees Celsius at normal pressure.", "expected": "FAKE"},
    {"topic": "Biology", "news": "A new study claims that plants can grow without water or light.", "expected": "FAKE"},
    {"topic": "Technology", "news": "A report claims that ordinary computers can run without electricity.", "expected": "FAKE"},
    {"topic": "Astronomy", "news": "A viral report claims that the Sun revolves around the Earth every day.", "expected": "FAKE"},
    {"topic": "Science", "news": "Scientists claim that humans can breathe normally in outer space without equipment.", "expected": "FAKE"},
    {"topic": "Biology", "news": "A fictional report claims that the human heart is located inside the foot.", "expected": "FAKE"},
    {"topic": "Geography", "news": "A viral claim says that the Pacific Ocean is smaller than a swimming pool.", "expected": "FAKE"},
    {"topic": "Technology", "news": "A fictional article claims that mobile phones work without any power source.", "expected": "FAKE"},
    {"topic": "Science", "news": "A false claim says that sound travels faster in a vacuum than in air.", "expected": "FAKE"}
]

results = []

print("=" * 60)
print("BALANCED GENERALIZATION TESTING")
print("=" * 60)

for item in test_data:
    prediction, confidence, signals = predict_news(item["news"])

    results.append({
        "Topic": item["topic"],
        "News": item["news"],
        "Expected": item["expected"],
        "Predicted": prediction,
        "Confidence": round(confidence, 2),
        "Correct": prediction == item["expected"]
    })

df = pd.DataFrame(results)

results_dir = Path(__file__).resolve().parent.parent / "results"
results_dir.mkdir(exist_ok=True)

output_path = results_dir / "balanced_ood_results.csv"
df.to_csv(output_path, index=False)

print(df.to_string(index=False))
print("\nOverall Pass Rate:", round(df["Correct"].mean() * 100, 2), "%")
print("\nResults saved at:", output_path)
