import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from schema import FinNewsOutput


def extract_json_relaxed(text: str) -> str:
    """Bóc tách khối JSON nếu model lỡ bọc trong markdown ```json ... ```."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def normalize_entity_set(entities: list[dict]) -> set[tuple[str, str]]:
    """Chuẩn hóa cặp (ticker hoặc tên viết thường, sentiment) để tính Entity F1."""
    result = set()
    for ent in entities:
        key = (ent.get("ticker") or ent.get("name") or "").strip().lower()
        sent = (ent.get("sentiment") or "").strip().lower()
        if key and sent:
            result.add((key, sent))
    return result


def compute_metrics(gold_list: list[dict], pred_raw_list: list[str]) -> tuple[dict, list[dict]]:
    strict_valid = 0
    relaxed_valid = 0

    gold_topics, pred_topics = [], []
    gold_sents, pred_sents = [], []

    ent_tp = ent_fp = ent_fn = 0
    detailed_rows = []

    for gold, raw_pred in zip(gold_list, pred_raw_list):
        parsed_obj = None
        is_strict = False
        is_relaxed = False

        # 1. Kiểm tra Strict JSON (không hậu xử lý)
        try:
            parsed_obj = FinNewsOutput.model_validate_json(raw_pred.strip())
            is_strict = True
            is_relaxed = True
        except Exception:
            # 2. Kiểm tra Relaxed JSON (sau khi bóc markdown/text thừa)
            try:
                cleaned = extract_json_relaxed(raw_pred)
                parsed_obj = FinNewsOutput.model_validate_json(cleaned)
                is_relaxed = True
            except Exception:
                parsed_obj = None

        strict_valid += int(is_strict)
        relaxed_valid += int(is_relaxed)

        gold_topics.append(gold["topic"])
        gold_sents.append(gold["overall_sentiment"])
        gold_ents = normalize_entity_set(gold.get("entities", []))

        if parsed_obj is not None:
            pred_dict = parsed_obj.model_dump()
            pred_topics.append(pred_dict["topic"])
            pred_sents.append(pred_dict["overall_sentiment"])
            pred_ents = normalize_entity_set(pred_dict.get("entities", []))
        else:
            pred_dict = None
            pred_topics.append("INVALID")
            pred_sents.append("INVALID")
            pred_ents = set()

        ent_tp += len(gold_ents & pred_ents)
        ent_fp += len(pred_ents - gold_ents)
        ent_fn += len(gold_ents - pred_ents)

        detailed_rows.append(
            {
                "gold": gold,
                "raw_pred": raw_pred,
                "parsed_pred": pred_dict,
                "strict_valid": is_strict,
                "relaxed_valid": is_relaxed,
            }
        )

    n = len(gold_list)
    ent_prec = ent_tp / (ent_tp + ent_fp) if (ent_tp + ent_fp) > 0 else 0.0
    ent_rec = ent_tp / (ent_tp + ent_fn) if (ent_tp + ent_fn) > 0 else 0.0
    ent_f1 = 2 * ent_prec * ent_rec / (ent_prec + ent_rec) if (ent_prec + ent_rec) > 0 else 0.0

    # Tính Macro-F1 trên đúng tập nhãn chuẩn (vẫn phạt các mẫu pred='INVALID')
    topic_labels = sorted(set(gold_topics))
    sent_labels = ["positive", "negative", "neutral"]

    metrics = {
        "num_samples": n,
        "strict_json_valid_rate": round(strict_valid / n, 4),
        "relaxed_json_valid_rate": round(relaxed_valid / n, 4),
        "sentiment_accuracy": round(accuracy_score(gold_sents, pred_sents), 4),
        "sentiment_macro_f1": round(
            f1_score(gold_sents, pred_sents, labels=sent_labels, average="macro", zero_division=0), 4
        ),
        "topic_accuracy": round(accuracy_score(gold_topics, pred_topics), 4),
        "topic_macro_f1": round(
            f1_score(gold_topics, pred_topics, labels=topic_labels, average="macro", zero_division=0), 4
        ),
        "entity_precision": round(ent_prec, 4),
        "entity_recall": round(ent_rec, 4),
        "entity_f1": round(ent_f1, 4),
    }
    return metrics, detailed_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qwen3_4b_t4.yaml")
    parser.add_argument("--adapter_path", type=str, default=None, help="Đường dẫn LoRA adapter (bỏ trống = Baseline)")
    parser.add_argument("--run_name", type=str, default="baseline-zeroshot")
    parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--output_root", type=str, default="outputs")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    hf_token = os.environ.get("HF_TOKEN")
    base_model_id = cfg["model"]["base_model_id"]
    dataset_id = cfg["data"]["dataset_id"]

    # 1. Nạp tập test từ Hugging Face Hub
    print(f"[DATA] Đang tải tập '{args.split}' từ {dataset_id}...")
    ds = load_dataset(dataset_id, split=args.split, token=hf_token)

    # 2. Nạp Tokenizer (bắt buộc padding_side='left' cho batched generation)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 3. Nạp Base Model 4-bit (và gắn Adapter nếu có)
    print(f"[MODEL] Đang nạp {base_model_id} (4-bit NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )
    if args.adapter_path:
        print(f"[PEFT] Đang gắn LoRA adapter từ: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()
    # Dọn cờ sampling để chạy greedy decoding (do_sample=False) không sinh cảnh báo
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    # 4. Chuẩn bị prompt (chỉ lấy system + user, bỏ phần assistant đáp án)
    prompts = []
    golds = []
    for row in ds:
        prompt_msgs = [m for m in row["messages"] if m["role"] != "assistant"]
        gold_str = [m["content"] for m in row["messages"] if m["role"] == "assistant"][0]
        prompts.append(
            tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
        )
        golds.append(json.loads(gold_str))

    # 5. Chạy suy luận theo batch
    preds = []
    t0 = time.time()
    for i in tqdm(range(0, len(prompts), args.batch_size), desc=f"Evaluating [{args.run_name}]"):
        batch_prompts = prompts[i : i + args.batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg["model"]["max_length"],
        ).to(model.device)

        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs[:, input_len:]
        batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        preds.extend(batch_preds)

    elapsed = time.time() - t0
    metrics, detailed_rows = compute_metrics(golds, preds)
    metrics["avg_latency_sec"] = round(elapsed / len(prompts), 3)
    metrics["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(0) / (1024**3), 2)

    # 6. Lưu kết quả ra file JSON/JSONL
    eval_dir = Path(args.output_root) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    with open(eval_dir / f"{args.run_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(eval_dir / f"{args.run_name}_predictions.jsonl", "w", encoding="utf-8") as f:
        for row, detail in zip(ds, detailed_rows):
            record = {"id": row["id"], **detail}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 7. Ghi log lên TensorBoard (để so sánh các run sau này)
    log_dir = Path(args.output_root) / "logs" / args.run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and k != "num_samples":
            writer.add_scalar(f"eval/{k}", v, 0)

    # Log 5 mẫu đầu tiên dạng text vào tab TEXT của TensorBoard để soi lỗi trực quan
    sample_md = []
    for idx in range(min(5, len(detailed_rows))):
        d = detailed_rows[idx]
        sample_md.append(
            f"**Sample {idx+1}** | Strict: `{d['strict_valid']}`\n"
            f"- Gold: `{json.dumps(d['gold'], ensure_ascii=False)}`\n"
            f"- Pred: `{d['raw_pred'].strip()}`"
        )
    writer.add_text("eval/sample_predictions", "\n\n---\n\n".join(sample_md), 0)
    writer.close()

    print(f"\n=== KẾT QUẢ ĐÁNH GIÁ: {args.run_name} ({args.split}) ===")
    for k, v in metrics.items():
        print(f"  {k:26s}: {v}")


if __name__ == "__main__":
    main()