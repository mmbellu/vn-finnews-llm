import json
import os
from collections import Counter
from pathlib import Path

# BẮT BUỘC đặt trước khi import datasets và transformers
CACHE_DIR = os.path.abspath(".hf_cache")
os.environ["HF_HOME"] = CACHE_DIR

import numpy as np
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from schema import FinNewsOutput, SYSTEM_PROMPT

load_dotenv()

LABELED_PATH = Path("data/raw/vn_finnews_labeled.jsonl")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
HF_DATASET_REPO = "mmbellu/vn-finnews-structured"
MAX_LENGTH = 512
SEED = 42


def to_chat_record(row: dict) -> dict:
    # Dump JSON dạng gọn (compact) để tiết kiệm token khi train & inference
    target_json = json.dumps(row["label"], ensure_ascii=False, separators=(",", ":"))
    return {
        "id": row["id"],
        "source": row.get("source", ""),
        "topic": row["label"]["topic"],
        "overall_sentiment": row["label"]["overall_sentiment"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["text"]},
            {"role": "assistant", "content": target_json},
        ],
    }


def save_jsonl(records: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_stratify(labels: list):
    counts = Counter(labels)
    return labels if min(counts.values()) >= 2 else None


def main():
    if not LABELED_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file {LABELED_PATH}. Hãy chạy src/label_data.py trước.")

    with open(LABELED_PATH, "r", encoding="utf-8") as f:
        raw_rows = [json.loads(line) for line in f if line.strip()]

    # 1. Validate Pydantic và loại trùng lặp nội dung text
    valid_rows = []
    seen_texts = set()
    for r in raw_rows:
        if r["text"] in seen_texts:
            continue
        try:
            FinNewsOutput.model_validate(r["label"])
            valid_rows.append(r)
            seen_texts.add(r["text"])
        except Exception:
            continue

    print(f"Tổng mẫu hợp lệ sau lọc trùng & validate: {len(valid_rows)}")
    print("Phân phối Sentiment:", Counter(r["label"]["overall_sentiment"] for r in valid_rows))
    print("Phân phối Topic:", Counter(r["label"]["topic"] for r in valid_rows))

    # 2. Chuyển sang chuẩn messages & đo độ dài token chuẩn mọi bản transformers
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        token=hf_token,
        cache_dir=CACHE_DIR,
    )

    records = []
    token_lengths = []
    dropped_too_long = 0

    for r in valid_rows:
        rec = to_chat_record(r)
        text = tokenizer.apply_chat_template(rec["messages"], tokenize=False)
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if n_tokens > MAX_LENGTH:
            dropped_too_long += 1
            continue
        token_lengths.append(n_tokens)
        records.append(rec)

    if not records:
        raise ValueError("Không còn mẫu nào hợp lệ sau khi lọc MAX_LENGTH.")

    arr = np.array(token_lengths)
    print(
        f"Độ dài token (đã bỏ {dropped_too_long} mẫu > {MAX_LENGTH}): "
        f"min={arr.min()}, median={int(np.median(arr))}, "
        f"p95={int(np.percentile(arr, 95))}, max={arr.max()}"
    )

    # 3. Chia tập Stratified 80% Train / 10% Val / 10% Test
    labels = [r["overall_sentiment"] for r in records]
    train_rec, temp_rec = train_test_split(
        records, test_size=0.2, stratify=safe_stratify(labels), random_state=SEED
    )
    temp_labels = [r["overall_sentiment"] for r in temp_rec]
    val_rec, test_rec = train_test_split(
        temp_rec, test_size=0.5, stratify=safe_stratify(temp_labels), random_state=SEED
    )

    save_jsonl(train_rec, OUT_DIR / "train.jsonl")
    save_jsonl(val_rec, OUT_DIR / "val.jsonl")
    save_jsonl(test_rec, OUT_DIR / "test.jsonl")
    print(f"Đã lưu: train={len(train_rec)}, val={len(val_rec)}, test={len(test_rec)}")

    # 4. Đẩy lên Hugging Face Hub
    if hf_token:
        ds_dict = DatasetDict(
            {
                "train": Dataset.from_list(train_rec),
                "validation": Dataset.from_list(val_rec),
                "test": Dataset.from_list(test_rec),
            }
        )
        ds_dict.push_to_hub(HF_DATASET_REPO, token=hf_token, private=True)
        print(f"Đã đẩy dataset lên Hugging Face Hub: {HF_DATASET_REPO}")
    else:
        print("[Cảnh báo] Không tìm thấy HF_TOKEN trong .env -> Chưa đẩy dataset lên Hub.")


if __name__ == "__main__":
    main()