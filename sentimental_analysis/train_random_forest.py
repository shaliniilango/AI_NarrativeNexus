# train_random_forest.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from text_processing import clean_text
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

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# Train
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_tfidf, y_train)
y_pred_rf = rf.predict(X_test_tfidf)

print("Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Save
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/random_forest_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.show()
