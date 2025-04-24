import os
import glob
import json
import re
import string
import pandas as pd
from datasets import Dataset, load_dataset
import nltk
import spacy
from nltk.corpus import stopwords
from contractions import fix as fix_contractions

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR      = "data"                       # where your txt/JSON/CSV live
TXT_FOLDERS   = {"Left Data":0, "Center Data":1, "Right Data":2}
JSON_FOLDER   = os.path.join(DATA_DIR, "jsons")
ALLSIDES_CSV  = os.path.join(DATA_DIR, "allsides.csv")
OUTPUT_CSV    = os.path.join(DATA_DIR, "final_bias_dataset.csv")
HFD_CACHE_DIR = os.path.join(DATA_DIR, "hf_cache")   # where to save cached HF dataset
MIN_CHARS     = 50        # drop samples shorter than this
MAX_TOKENS    = 512       # drop samples longer than this (after tokenization)

# ── DOWNLOAD NLP RESOURCES ─────────────────────────────────────────────────────
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
stop_words = set(stopwords.words('english'))

# ── PREPROCESSING PIPELINE ─────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    """Lowercase, expand contractions, strip URLs/HTML, remove punctuation,
    remove stopwords, lemmatize."""
    text = text.lower()
    text = fix_contractions(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # lemmatize + remove stopwords
    doc = nlp(text)
    lemmas = [tok.lemma_ for tok in doc if tok.lemma_ not in stop_words]
    return " ".join(lemmas)

# ── LOADING FUNCTIONS ─────────────────────────────────────────────────────────
def load_txt_files():
    rows = []
    for folder, label in TXT_FOLDERS.items():
        base = os.path.join(DATA_DIR, folder, folder)  # two-level
        if not os.path.isdir(base):
            continue
        for fn in glob.glob(os.path.join(base, "*.txt")):
            raw = open(fn, encoding="utf-8", errors="ignore").read().strip()
            if len(raw) >= MIN_CHARS:
                clean = preprocess(raw)
                if len(clean.split()) <= MAX_TOKENS:
                    rows.append({"text": clean, "label": label})
    print(f"⚙️  Loaded {len(rows)} .txt samples")
    return pd.DataFrame(rows)

def load_json_files():
    bias_map = {"left":0, "center":1, "right":2}
    rows = []
    for fn in glob.glob(os.path.join(JSON_FOLDER, "*.json")):
        obj = json.load(open(fn, encoding="utf-8", errors="ignore"))
        raw = obj.get("content","").strip()
        btxt = obj.get("bias_text","").lower()
        if raw and btxt in bias_map and len(raw)>=MIN_CHARS:
            clean = preprocess(raw)
            if len(clean.split()) <= MAX_TOKENS:
                rows.append({"text": clean, "label": bias_map[btxt]})
    print(f"⚙️  Loaded {len(rows)} JSON samples")
    return pd.DataFrame(rows)

def load_allsides_csv():
    if not os.path.exists(ALLSIDES_CSV):
        return pd.DataFrame(columns=["text","label"])
    df = pd.read_csv(ALLSIDES_CSV)
    # assume 'news_content' & 'bias' columns exist; adjust as needed
    df = df.rename(columns={"news_content":"text","bias":"label"})
    df = df.dropna(subset=["text","label"])
    # map textual bias to int if needed:
    mapping = {"left":0,"center":1,"right":2}
    df["label"] = df["label"].map(mapping).astype(int)
    df["text"]  = df["text"].astype(str).str.strip()
    # preprocess & filter
    df["text"] = df["text"].apply(preprocess)
    df = df[df["text"].str.split().str.len().between( MIN_CHARS//5, MAX_TOKENS )]
    print(f"⚙️  Loaded {len(df)} AllSides-CSV samples")
    return df[["text","label"]]

# ── MAIN MERGE & DEDUP ────────────────────────────────────────────────────────
def main():
    dfs = [
        load_txt_files(),
        load_json_files(),
        load_allsides_csv(),
    ]
    df = pd.concat(dfs, ignore_index=True).dropna(subset=["text","label"])
    before = len(df)
    df = df.drop_duplicates(subset="text")
    print(f"🗑  Dropped {before-len(df)} duplicates")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 1) save CSV for easy inspection / quick train
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"💾  Wrote {len(df)} rows to {OUTPUT_CSV}")

    # 2) also save as cached HF dataset for lightning-fast reload later:
    hfds = Dataset.from_pandas(df)
    hfds.save_to_disk(HFD_CACHE_DIR)
    print(f"💾  Saved HF dataset to {HFD_CACHE_DIR}")

if __name__=="__main__":
    main()
