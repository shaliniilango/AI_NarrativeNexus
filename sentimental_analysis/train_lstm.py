# train_lstm.py
import pandas as pd
import numpy as np
from text_processing import clean_text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Paths
train_path = "data/amazon_reviews_train.csv"
test_path  = "data/amazon_reviews_test.csv"

# Load data
train_df = pd.read_csv(train_path, nrows=50000)
test_df  = pd.read_csv(test_path, nrows=10000)

train_df["text"] = (train_df["title"].astype(str) + " " + train_df["content"].astype(str)).apply(clean_text)
test_df["text"]  = (test_df["title"].astype(str) + " " + test_df["content"].astype(str)).apply(clean_text)

X_train, y_train = train_df["text"], train_df["label"]
X_test,  y_test  = test_df["text"],  test_df["label"]

# Tokenization
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=200)
X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=200)

# Build model
model_lstm = Sequential([
    Embedding(input_dim=20000, output_dim=128, input_length=200),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model_lstm.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train
history = model_lstm.fit(
    X_train_seq, np.array(y_train),
    epochs=3, batch_size=128,
    validation_data=(X_test_seq, np.array(y_test))
)

# Evaluate
y_pred_lstm = (model_lstm.predict(X_test_seq) > 0.5).astype("int32").flatten()
print("LSTM Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lstm))
print(classification_report(y_test, y_pred_lstm))

# Save
os.makedirs("models", exist_ok=True)
model_lstm.save("models/lstm_model.h5")
joblib.dump(tokenizer, "models/lstm_tokenizer.pkl")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_lstm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("LSTM Confusion Matrix")
plt.show()
