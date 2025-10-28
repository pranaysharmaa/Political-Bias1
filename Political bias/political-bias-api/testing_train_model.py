import os
import pandas as pd #type: ignore
from datasets import load_dataset #type: ignore
from preprocessing import preprocess_pipeline


# === CONFIG ===
DATA_DIR = "data"
LABEL_MAP = {
    "Left Data": 0,
    "Center Data": 1,
    "Right Data": 2
}
KAGGLE_CSV_FILE = "labelled_corpus_political_bias.csv"
ALLSIDES_CSV_FILE = "allsides.csv"

# === TXT FILES ===
def load_txt_files():
    all_data = []
    for folder, label in LABEL_MAP.items():
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"⚠️ Skipping missing folder: {folder_path}")
            continue

        for fname in os.listdir(folder_path):
            if fname.endswith(".txt"):
                path = os.path.join(folder_path, fname)
                with open(path, encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        all_data.append({"text": text, "label": label})
    print(f"✅ Loaded {len(all_data)} samples from .txt files")
    return pd.DataFrame(all_data)

# === CSV FILE LOADER ===
def load_csv_file(file_path, text_col, label_col, label_map=None):
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return pd.DataFrame(columns=["text", "label"])
    
    df = pd.read_csv(file_path)

    # ✅ Rename first
    df = df.rename(columns={text_col: "text", label_col: "label"})

    # ✅ Apply label mapping safely
    if label_map:
        valid_labels = list(label_map.keys())
        df = df[df["label"].isin(valid_labels)].copy()
        df["label"] = df["label"].map(label_map)

    print(f"✅ Loaded {len(df)} samples from {os.path.basename(file_path)}")
    return df[["text", "label"]]


# === Hugging Face: pranjali97 ===
def load_pranjali():
    ds = load_dataset("pranjali97/Bias-detection-combined")
    df = pd.DataFrame(ds["train"])
    df = df.rename(columns={"text": "text", "label": "label"})
    print(f"✅ Loaded {len(df)} samples from pranjali97/Bias-detection-combined")
    return df

#  Hugging Face: vector-institute (will use after getting permission)
# def load_vector_institute():
#     ds = load_dataset("vector-institute/newsmediabias-plus")
#     df = pd.DataFrame(ds["train"])
#     df = df.rename(columns={"text": "text", "label": "label"})
#     print(f"✅ Loaded {len(df)} samples from vector-institute/newsmediabias-plus")
#     return df


# === AllSides Dataset ===
def load_allsides_csv():
    path = os.path.join(DATA_DIR, ALLSIDES_CSV_FILE)
    bias_map = {
        "left": 0,
        "left-center": 0,
        "center": 1,
        "right-center": 2,
        "right": 2
    }
    return load_csv_file(path, text_col="name", label_col="bias", label_map=bias_map)


# === Merge & Save ===
def main():
    dfs = [
        load_txt_files(),
        load_allsides_csv(),
        # load_kaggle_csv(),
        load_pranjali(),
        # load_vector_institute()
    ]

    combined = pd.concat(dfs, ignore_index=True).dropna()
    combined = combined.sample(frac=1).reset_index(drop=True)

    output_path = os.path.join(DATA_DIR, "final_bias_dataset.csv")
    combined.to_csv(output_path, index=False)
    print(f"\n✅ Final dataset saved: {output_path}")
    print(f"🔢 Total samples: {len(combined)}")

if __name__ == "__main__":
    main()
