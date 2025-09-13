import os
import re
import zipfile
import pandas as pd

# ---------- Unzip the dataset ----------
zip_path = "20news-18828-20250913T184721Z-1-001.zip"  # path to your uploaded zip
extract_dir = "req_data"  # folder to extract into
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(extract_dir)

# The dataset inside the zip is usually named '20news-18828'
root_folder = os.path.join(extract_dir, "20news-18828")


# ---------- Cleaning utilities ----------
def extract_body(text: str) -> str:
    """Remove headers, metadata lines, quotes, and signatures."""
    if not isinstance(text, str):
        return ""

    parts = re.split(r"\n\s*\n", text, maxsplit=1)
    body = parts[1] if len(parts) > 1 else parts[0]

    cleaned_lines = []
    for line in body.splitlines():
        if re.match(
            r"^(archive-name|subject|from|path|xref|organization|lines|newsgroups|message-id|keywords):",
            line.strip(), re.I):
            continue
        if line.strip().startswith((">", "|")):
            continue
        if line.strip().startswith("--"):
            break
        if re.search(r"In article\s*<.*?>", line, re.I):
            continue
        if re.search(r"writes:|wrote:", line, re.I):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def deep_clean(text: str) -> str:
    """Further clean the body text."""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
    return text.lower().strip()


def convert_20ng_to_csv(root_folder, output_csv, max_files=50):
    """Convert 20 Newsgroups dataset into a cleaned CSV file."""
    data = []
    for category in sorted(os.listdir(root_folder)):
        category_path = os.path.join(root_folder, category)
        if not os.path.isdir(category_path):
            continue

        print(f"Processing category: {category}")
        for i, filename in enumerate(os.listdir(category_path)):
            if i >= max_files:
                break
            file_path = os.path.join(category_path, filename)
            try:
                with open(file_path, "r", encoding="latin1") as f:
                    raw_text = f.read()
                body = deep_clean(extract_body(raw_text))
                if body:
                    data.append({
                        "filename": filename,
                        "category": category,
                        "text": body
                    })
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"✅ Saved {len(df)} rows across {df['category'].nunique()} categories to {output_csv}")


# ---------- Run conversion ----------
if __name__ == "__main__":
    convert_20ng_to_csv(
        root_folder="req_data/20news-18828",   # folder where you unzipped the dataset
        output_csv="20news_18828_clean_all.csv",
        max_files=10**9   # effectively no limit, processes every file
    )

