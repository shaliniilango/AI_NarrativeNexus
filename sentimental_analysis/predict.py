# predict.py
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from text_processing import clean_text

# Load RF
rf_loaded = joblib.load("models/random_forest_model.pkl")
vectorizer_loaded = joblib.load("models/tfidf_vectorizer.pkl")

# Load LSTM
lstm_loaded = load_model("models/lstm_model.h5")
tokenizer_loaded = joblib.load("models/lstm_tokenizer.pkl")

def predict_rf(text):
    text_clean = clean_text(text)
    vec = vectorizer_loaded.transform([text_clean])
    pred = rf_loaded.predict(vec)[0]
    return "Positive" if pred == 1 else "Negative"

def predict_lstm(text):
    text_clean = clean_text(text)
    seq = tokenizer_loaded.texts_to_sequences([text_clean])
    padded = pad_sequences(seq, maxlen=200)
    pred = (lstm_loaded.predict(padded) > 0.5).astype("int32")[0][0]
    return "Positive" if pred == 1 else "Negative"

# Example
if __name__ == "__main__":
    sample_texts = [
        "This product is absolutely fantastic, I loved it!",
        "Worst purchase ever, total waste of money.",
        "It was okay, nothing special but not bad either."
    ]
    for txt in sample_texts:
        print(f"\nINPUT: {txt}")
        print("Random Forest →", predict_rf(txt))
        print("LSTM          →", predict_lstm(txt))
