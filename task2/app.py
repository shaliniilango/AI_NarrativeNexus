import os
import uuid
import json
import requests
import pandas as pd
import streamlit as st
import docx
import pdfplumber
import praw
from dotenv import load_dotenv
from datetime import datetime

# ==================== CONFIG ====================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, "data_store.json")

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

# ==================== DATA STORAGE ====================
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_data(new_record):
    data = load_data()
    data.append(new_record)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def create_record(source_type, file_type, content, filename="N/A", url=None):
    return {
        "id": str(uuid.uuid4()),
        "filename": filename,
        "source_type": source_type,
        "file_type": file_type,
        "content": content,
        "url": url,
        "upload_time": datetime.now().isoformat(),
    }

# ==================== FILE READERS ====================
def read_txt(file):
    return file.read().decode("utf-8")

def read_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ==================== REDDIT & NEWS ====================
def fetch_reddit_post(url):
    submission = reddit.submission(url=url)
    return create_record(
        "reddit",
        "post",
        submission.title + "\n" + submission.selftext,
        filename="reddit_post",
        url=url,
    )

def fetch_news(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()

    if "articles" not in response or len(response["articles"]) == 0:
        return None

    article = response["articles"][0]
    return create_record(
        "news",
        "article",
        (article.get("title") or "") + "\n" + (article.get("description") or ""),
        filename="news_article",
        url=article.get("url"),
    )

# ==================== STREAMLIT UI ====================
st.set_page_config(page_title="Narrative Nexus", page_icon="📑", layout="wide")

st.title("📑 Narrative Nexus - Text & Data Collector")
st.write("Upload a file, paste text, or fetch content from Reddit/News.")

option = st.radio(
    "Choose Input Type:",
    ["Upload File", "Paste Text", "Reddit Link", "News Query"],
    horizontal=True,
)

if option == "Upload File":
    uploaded_file = st.file_uploader(
        "Upload a file (.txt, .csv, .docx, .pdf)", type=["txt", "csv", "docx", "pdf"]
    )
    if uploaded_file is not None:
        try:
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext == ".txt":
                content = read_txt(uploaded_file)
            elif ext == ".csv":
                content = read_csv(uploaded_file)
            elif ext == ".docx":
                content = read_docx(uploaded_file)
            elif ext == ".pdf":
                content = read_pdf(uploaded_file)
            else:
                st.error("Unsupported file type!")
                content = None

            if content:
                record = create_record("file", ext, content, filename=uploaded_file.name)
                save_data(record)
                st.success(f"✅ File saved in data_store.json (ID: {record['id']})")
                st.subheader("📌 Extracted Text (Preview):")
                st.write(content[:1000] + "..." if len(content) > 1000 else content)

        except Exception as e:
            st.error(f"Error processing file: {e}")

elif option == "Paste Text":
    text_input = st.text_area("Paste your text here:")
    if st.button("Save Text"):
        if text_input.strip() != "":
            record = create_record("pasted", "raw", text_input, filename="pasted_text")
            save_data(record)
            st.success(f"✅ Text saved (ID: {record['id']})")
            st.subheader("📌 Input Text (Preview):")
            st.write(text_input)
        else:
            st.warning("Please enter some text!")

elif option == "Reddit Link":
    reddit_url = st.text_input("Enter Reddit post link:")
    if st.button("Fetch Reddit Post"):
        try:
            record = fetch_reddit_post(reddit_url)
            save_data(record)
            st.success(f"✅ Reddit post saved (ID: {record['id']})")
            st.subheader("📌 Reddit Post (Preview):")
            st.write(record["content"])
        except Exception as e:
            st.error(f"Error fetching Reddit post: {e}")

elif option == "News Query":
    news_query = st.text_input("Enter news search query:")
    if st.button("Fetch News Article"):
        try:
            record = fetch_news(news_query)
            if record:
                save_data(record)
                st.success(f"✅ News article saved (ID: {record['id']})")
                st.subheader("📌 News Article (Preview):")
                st.write(record["content"])
            else:
                st.warning("No news found for this query.")
        except Exception as e:
            st.error(f"Error fetching news: {e}")
