import argparse
import gc
import os
from pathlib import Path

import torch
import yaml
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from schema import FinNewsOutput, SYSTEM_PROMPT

SAMPLE_NEWS = (
    "Tiêu đề: Hòa Phát báo lãi quý tăng mạnh nhờ sản lượng thép cuộn cán nóng\n"
    "Nội dung: Tập đoàn Hòa Phát ghi nhận doanh thu và lợi nhuận sau thuế tăng trưởng "
    "hai chữ số so với cùng kỳ, vượt kế hoạch đề ra nhờ nhu cầu xây dựng phục hồi."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qwen3_4b_t4.yaml")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--merged_out_dir", type=str, default="/content/merged_qwen3_vn_finnews")
    parser.add_argument("--private", action="store_true", help="Đẩy repo ở chế độ private")
    parser.add_argument("--skip_push", action="store_true", help="Chỉ merge cục bộ, chưa đẩy lên Hub")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    hf_token = os.environ.get("HF_TOKEN")
    if not args.skip_push and not hf_token:
        raise ValueError("Thiếu biến môi trường HF_TOKEN để đẩy model lên Hugging Face Hub.")

    base_model_id = cfg["model"]["base_model_id"]
    hub_cfg = cfg.get("hub", {})
    adapter_repo_id = hub_cfg.get("adapter_repo_id", "mmbellu/qwen3-4b-vn-finnews-lora")
    merged_repo_id = hub_cfg.get("merged_repo_id", "mmbellu/qwen3-4b-vn-finnews-merged")

    adapter_path = Path(args.adapter_path)
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Không tìm thấy adapter_config.json tại {adapter_path}")

    merged_dir = Path(args.merged_out_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=hf_token) if not args.skip_push else None

    # 1. Đẩy LoRA Adapter lên Hub trước (~130MB)
    if api is not None:
        print(f"[1/4] Đang đẩy LoRA Adapter lên {adapter_repo_id}...")
        api.create_repo(repo_id=adapter_repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(adapter_path),
            repo_id=adapter_repo_id,
            repo_type="model",
            commit_message=f"Upload best QLoRA adapter ({adapter_path.parent.name})",
        )
        print(f"      -> Xong Adapter: https://huggingface.co/{adapter_repo_id}")

    # 2. Nạp Base Model nguyên bản ở FP16 trực tiếp lên GPU T4 (~8GB VRAM, tránh tràn CPU RAM)
    print(f"[2/4] Đang nạp Base Model {base_model_id} ở FP16 (không lượng tử hóa)...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.float16,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        low_cpu_mem_usage=True,
        token=hf_token,
    )

    # 3. Gắn Adapter và hợp nhất vĩnh viễn vào trọng số gốc
    print(f"[3/4] Đang hợp nhất (merge_and_unload) từ {adapter_path}...")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged_model = peft_model.merge_and_unload()
    merged_model.config.use_cache = True
    merged_model.eval()

    # Sanity Check: kiểm tra suy luận thực tế và xác thực Pydantic trước khi xuất bản
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": SAMPLE_NEWS},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)

    with torch.no_grad():
        out_ids = merged_model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    sample_output = tokenizer.decode(
        out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    print(f"\n[SANITY CHECK] Output sau merge:\n{sample_output}")
    FinNewsOutput.model_validate_json(sample_output)
    print("[SANITY CHECK] Hợp lệ Pydantic 100%!\n")

    # 4. Lưu dạng safetensors (chia mảnh 2GB) rồi upload thẳng thư mục lên Hub
    print(f"[4/4] Đang lưu merged model ra {merged_dir}...")
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(str(merged_dir))

    # Giải phóng VRAM/RAM trước khi upload ~8GB
    del merged_model, peft_model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if api is not None:
        print(f"      Đang tải lên Hugging Face Hub: {merged_repo_id}...")
        api.create_repo(repo_id=merged_repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(merged_dir),
            repo_id=merged_repo_id,
            repo_type="model",
            commit_message=f"Upload FP16 merged model ({adapter_path.parent.name})",
        )
        print(f"      -> Hoàn tất Merged Model: https://huggingface.co/{merged_repo_id}")


if __name__ == "__main__":
    main()