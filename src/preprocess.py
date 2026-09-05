import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# NLTK tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Clean and preprocess news text.
    """

    # Handle empty values
    if pd.isna(text):
        return ""

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword removal + lemmatization
    cleaned_tokens = []

    for word in tokens:
        if word not in stop_words:
            cleaned_word = lemmatizer.lemmatize(word)
            cleaned_tokens.append(cleaned_word)

    # Convert tokens back to text
    return " ".join(cleaned_tokens)


def load_and_prepare_data():
    """
    Load Fake and True news datasets and prepare labels.
    """

    fake = pd.read_csv("dataset/Fake.csv")
    true = pd.read_csv("dataset/True.csv")

    # Add labels
    fake["label"] = 0
    true["label"] = 1

    # Combine title and article text
    fake["content"] = fake["title"].fillna("") + " " + fake["text"].fillna("")
    true["content"] = true["title"].fillna("") + " " + true["text"].fillna("")

    # Keep only required columns
    fake = fake[["content", "label"]]
    true = true[["content", "label"]]

    # Combine both datasets
    data = pd.concat([fake, true], ignore_index=True)

    # Remove duplicate articles
    data.drop_duplicates(subset="content", inplace=True)

    # Remove empty articles
    data = data[data["content"].str.strip() != ""]

    # Shuffle dataset
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    return data


if __name__ == "__main__":

    print("Loading dataset...")

    data = load_and_prepare_data()

    print("Dataset loaded successfully!")
    print("Total articles:", len(data))

    print("\nLabel distribution:")
    print(data["label"].value_counts())

    print("\nPreprocessing first 5 articles...")

    data["cleaned_content"] = data["content"].head(5).apply(clean_text)

    for i in range(5):
        print(f"\nArticle {i + 1}:")
        print("Original:", data["content"].iloc[i][:200])
        print("Cleaned :", data["cleaned_content"].iloc[i][:200])