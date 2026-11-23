import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

TARGET_PRESETS = {
    "attn_only": ["q_proj", "v_proj"],
    "all_linear": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}


def to_prompt_completion(example: dict) -> dict:
    """Tách messages thành prompt và completion để SFTTrainer chỉ tính loss trên phần JSON."""
    msgs = example["messages"]
    return {
        "prompt": [m for m in msgs if m["role"] != "assistant"],
        "completion": [m for m in msgs if m["role"] == "assistant"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qwen3_4b_t4.yaml")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--target_preset", type=str, default=None, choices=["attn_only", "all_linear"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Ghi đè siêu tham số từ CLI nếu được truyền vào (phục vụ Ablation Study)
    run_name = args.run_name or cfg["experiment"]["run_name"]
    lora_r = args.lora_r or cfg["lora"]["r"]
    lora_alpha = args.lora_alpha or cfg["lora"]["lora_alpha"]
    target_modules = (
        TARGET_PRESETS[args.target_preset]
        if args.target_preset
        else cfg["lora"]["target_modules"]
    )
    lr = args.lr or float(cfg["training"]["learning_rate"])
    epochs = args.epochs or int(cfg["training"]["num_train_epochs"])

    output_root = Path(args.output_root)
    run_dir = output_root / "runs" / run_name
    final_adapter_dir = run_dir / "best_adapter"
    tb_log_dir = output_root / "logs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    # Trỏ thư mục log TensorBoard tương thích mọi phiên bản transformers
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(tb_log_dir)

    hf_token = os.environ.get("HF_TOKEN")
    base_model_id = cfg["model"]["base_model_id"]
    dataset_id = cfg["data"]["dataset_id"]

    # 1. Nạp dữ liệu từ Hugging Face Hub
    print(f"[DATA] Đang tải dataset {dataset_id}...")
    raw_ds = load_dataset(dataset_id, token=hf_token)

    sft_fields = SFTConfig.__dataclass_fields__
    use_completion_only = "completion_only_loss" in sft_fields

    if use_completion_only:
        train_ds = raw_ds["train"].map(
            to_prompt_completion, remove_columns=raw_ds["train"].column_names
        )
        val_ds = raw_ds["validation"].map(
            to_prompt_completion, remove_columns=raw_ds["validation"].column_names
        )
        print("[DATA] Đã bật chế độ Completion-Only Loss (chỉ tính loss trên phần JSON).")
    else:
        train_ds = raw_ds["train"]
        val_ds = raw_ds["validation"]

    # 2. Nạp Tokenizer (padding bên phải khi huấn luyện SFT)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Nạp Base Model ở chế độ 4-bit NF4 (QLoRA cho GPU T4 fp16)
    print(f"[MODEL] Đang nạp {base_model_id} (4-bit NF4, compute_dtype=float16)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg["model"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["model"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        dtype=torch.float16,
        device_map="auto",
        token=hf_token,
    )
    model.config.use_cache = False  # Bắt buộc tắt khi dùng gradient checkpointing

    # 4. Cấu hình PEFT LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # Tính sẵn warmup_steps (số nguyên) để tương thích mọi bản transformers/trl
    eff_batch = (
        cfg["training"]["per_device_train_batch_size"]
        * cfg["training"]["gradient_accumulation_steps"]
    )
    total_steps = max(1, (len(train_ds) // eff_batch) * epochs)
    warmup_steps = max(1, int(total_steps * float(cfg["training"]["warmup_ratio"])))

    # 5. Cấu hình SFTConfig tương thích chéo các bản trl/transformers
    sft_kwargs = dict(
        output_dir=str(run_dir),
        run_name=run_name,
        num_train_epochs=epochs,
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=lr,
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_steps=warmup_steps,
        weight_decay=cfg["training"]["weight_decay"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        optim=cfg["training"]["optim"],
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=40,
        save_strategy="steps",
        save_steps=40,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        seed=cfg["experiment"]["seed"],
        packing=False,
    )

    if "max_length" in sft_fields:
        sft_kwargs["max_length"] = cfg["model"]["max_length"]
    elif "max_seq_length" in sft_fields:
        sft_kwargs["max_seq_length"] = cfg["model"]["max_length"]

    if use_completion_only:
        sft_kwargs["completion_only_loss"] = True
    if "logging_dir" in sft_fields:
        sft_kwargs["logging_dir"] = str(tb_log_dir)

    # Lọc an toàn để không văng TypeError nếu phiên bản trl mới bỏ bớt trường nào đó
    skipped = [k for k in sft_kwargs if k not in sft_fields]
    if skipped:
        print(f"[INFO] Bỏ qua tham số không tồn tại trong bản trl hiện tại: {skipped}")
    sft_kwargs = {k: v for k, v in sft_kwargs.items() if k in sft_fields}

    sft_config = SFTConfig(**sft_kwargs)

  # 6. Khởi tạo Trainer và bắt đầu huấn luyện
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Ép tham số trainable (LoRA) về float32 cho GradScaler trên T4 (fp16)
    # và dọn sạch mọi tham số bfloat16 còn sót lại
    for p in trainer.model.parameters():
        if p.requires_grad:
            if p.dtype != torch.float32:
                p.data = p.data.to(torch.float32)
        elif p.dtype == torch.bfloat16:
            p.data = p.data.to(torch.float16)

    trainable_dtypes = {p.dtype for p in trainer.model.parameters() if p.requires_grad}
    print(f"[DTYPE] Kiểu dữ liệu tham số trainable: {trainable_dtypes}")

    print(f"\n=== BẮT ĐẦU TRAIN RUN: {run_name} ===")
    print(f"LoRA r={lora_r}, alpha={lora_alpha}, lr={lr}, targets={target_modules}")
    trainer.model.print_trainable_parameters()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_result = trainer.train()
    elapsed_min = (time.time() - t0) / 60.0
    peak_vram = torch.cuda.max_memory_allocated(0) / (1024**3)
    
    # 7. Lưu Best Adapter, Tokenizer và Tóm tắt thí nghiệm
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))

    summary = {
        "run_name": run_name,
        "base_model_id": base_model_id,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "learning_rate": lr,
        "epochs": epochs,
        "train_loss": round(float(train_result.training_loss), 4),
        "best_eval_loss": round(float(trainer.state.best_metric), 4)
        if trainer.state.best_metric is not None
        else None,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "train_time_min": round(elapsed_min, 2),
        "peak_vram_gb": round(peak_vram, 2),
        "adapter_path": str(final_adapter_dir),
    }
    with open(run_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== HOÀN TẤT RUN: {run_name} ===")
    for k, v in summary.items():
        print(f"  {k:18s}: {v}")


if __name__ == "__main__":
    main()