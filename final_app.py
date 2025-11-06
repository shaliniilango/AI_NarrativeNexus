import os
import uuid
import json
import requests
import pandas as pd
import streamlit as st
import docx
import pdfplumber
import praw
from datetime import datetime
from dotenv import load_dotenv
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import T5ForConditionalGeneration, T5Tokenizer

# =====================================================
# 🔧 APP CONFIGURATION
# =====================================================
st.set_page_config(page_title="Narrative Nexus", page_icon="📘", layout="wide")
st.title("📘 Narrative Nexus: Unified Text Intelligence System")
st.write("Analyze sentiment, classify topic, and summarize text from multiple input sources.")

# =====================================================
# 📂 SETUP & ENVIRONMENT
# =====================================================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
load_dotenv()

# Reddit & News API Keys
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
)

# =====================================================
# 🧩 LOAD ALL MODELS
# =====================================================
@st.cache_resource
def load_all_models():
    # Load Sentiment Models
    rf_sentiment = joblib.load("models/random_forest_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    lstm_model = load_model("models/lstm_model.h5")
    tokenizer = joblib.load("models/lstm_tokenizer.pkl")

    # Load Topic Classifier
    topic_pipeline = joblib.load("models/topic_classifier.pkl")

    # Load Summarizer
    t5_name = "t5-small"
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_name)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_name)

    return rf_sentiment, vectorizer, lstm_model, tokenizer, topic_pipeline, t5_tokenizer, t5_model

rf_sentiment, vectorizer, lstm_model, tokenizer, topic_pipeline, t5_tokenizer, t5_model = load_all_models()

# =====================================================
# ⚙️ HELPER FUNCTIONS
# =====================================================
def clean_text(text):
    """Basic text cleaning"""
    return str(text).lower().strip()

def predict_rf_sentiment(text):
    vec = vectorizer.transform([clean_text(text)])
    pred = rf_sentiment.predict(vec)[0]
    return "Positive" if pred == 1 else "Negative"

def predict_lstm_sentiment(text):
    seq = tokenizer.texts_to_sequences([clean_text(text)])
    padded = pad_sequences(seq, maxlen=200)
    pred = (lstm_model.predict(padded) > 0.5).astype("int32")[0][0]
    return "Positive" if pred == 1 else "Negative"

def predict_topic(text):
    return topic_pipeline.predict([text])[0]

def summarize_text(text):
    input_text = "summarize: " + text
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = t5_model.generate(
        inputs,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def read_file(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".txt":
        return file.read().decode("utf-8")
    elif ext == ".csv":
        return pd.read_csv(file).to_string()
    elif ext == ".docx":
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == ".pdf":
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    else:
        st.error("Unsupported file type!")
        return None

def fetch_reddit_post(url):
    submission = reddit.submission(url=url)
    return submission.title + "\n" + submission.selftext

def fetch_news_article(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()
    if "articles" in response and response["articles"]:
        article = response["articles"][0]
        return (article.get("title") or "") + "\n" + (article.get("description") or "")
    else:
        return None

# =====================================================
# 🧠 STREAMLIT INTERFACE
# =====================================================
st.sidebar.header("📥 Input Options")
option = st.sidebar.radio(
    "Choose Input Type:",
    ["Paste Text", "Upload File", "Reddit Link", "News Query"],
)

text_data = None

if option == "Paste Text":
    text_data = st.text_area("📝 Paste your text here:")

elif option == "Upload File":
    uploaded = st.file_uploader("📂 Upload a file (.txt, .csv, .docx, .pdf)")
    if uploaded:
        text_data = read_file(uploaded)
        st.success(f"File '{uploaded.name}' uploaded successfully!")

elif option == "Reddit Link":
    reddit_url = st.text_input("🔗 Enter Reddit post link:")
    if st.button("Fetch Reddit Post"):
        try:
            text_data = fetch_reddit_post(reddit_url)
            st.success("Reddit post fetched successfully!")
            st.write(text_data[:800] + "..." if len(text_data) > 800 else text_data)
        except Exception as e:
            st.error(f"Failed to fetch Reddit post: {e}")

elif option == "News Query":
    query = st.text_input("📰 Enter news topic or keyword:")
    if st.button("Fetch News Article"):
        article = fetch_news_article(query)
        if article:
            text_data = article
            st.success("News article fetched successfully!")
            st.write(text_data)
        else:
            st.warning("No article found for the given query.")

# =====================================================
# 🔍 ANALYSIS SECTION
# =====================================================
if text_data and st.button("🔍 Analyze Text"):
    st.subheader("📘 Input Text (Preview)")
    st.write(text_data[:1500] + "..." if len(text_data) > 1500 else text_data)

    with st.spinner("Running all models..."):
        topic = predict_topic(text_data)
        rf_sent = predict_rf_sentiment(text_data)
        lstm_sent = predict_lstm_sentiment(text_data)
        summary = summarize_text(text_data)

    # =====================================================
    # 🎯 DISPLAY RESULTS CLEARLY
    # =====================================================
    st.markdown("---")
    st.markdown("## 🧭 Analysis Results")

    st.subheader("🗂️ Topic Classification")
    st.success(f"**Predicted Topic:** {topic}")

    st.subheader("💬 Sentiment Analysis")
    st.info(f"**Random Forest Sentiment:** {rf_sent}")
    st.info(f"**LSTM Sentiment:** {lstm_sent}")

    st.subheader("📝 Summarized Text")
    st.write(summary)

    st.success("✅ Analysis complete!")

