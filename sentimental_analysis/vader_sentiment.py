# vader_sentiment.py
import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, accuracy_score

nltk.download("punkt")
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_sentence_level(text):
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    if not sentences:
        return 1  # default positive
    scores = [analyzer.polarity_scores(sent)["compound"] for sent in sentences]
    avg_score = sum(scores) / len(scores)
    return 1 if avg_score >= 0 else 0

if __name__ == "__main__":
    train_df = pd.read_csv("data/amazon_reviews_train.csv", nrows=5000)
    test_df  = pd.read_csv("data/amazon_reviews_test.csv", nrows=2000)

    train_df["pred_label"] = train_df["content"].astype(str).apply(get_sentiment_sentence_level)
    test_df["pred_label"]  = test_df["content"].astype(str).apply(get_sentiment_sentence_level)

    print("Train Accuracy:", accuracy_score(train_df["label"], train_df["pred_label"]))
    print("Test Accuracy:", accuracy_score(test_df["label"], test_df["pred_label"]))
    print("\nTest Classification Report:\n")
    print(classification_report(test_df["label"], test_df["pred_label"]))
