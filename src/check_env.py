import os
import yaml
import torch
from torch.utils.tensorboard import SummaryWrite
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    with open("configs/qwen3_4b_t4.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Kiểm tra phần cứng GPU
    assert torch.cuda.is_available(), "Chưa bật GPU! Hãy chọn Runtime T4."
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[GPU] {gpu_name} | VRAM: {total_vram:.2f} GB | bf16: {torch.cuda.is_bf16_supported()}")

    # 2. Kiểm tra kết nối Hugging Face
    hf_token = os.environ.get("HF_TOKEN")
    assert hf_token, "Thiếu biến môi trường HF_TOKEN"
    user_info = HfApi(token=hf_token).whoami()
    print(f"[HF] Đã đăng nhập thành công với tài khoản: {user_info['name']}")

    # 3. Load thử Qwen3-4B ở chế độ 4-bit NF4 trên T4
    model_id = cfg["model"]["base_model_id"]
    print(f"[MODEL] Đang tải {model_id} ở chế độ 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )
    used_vram = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"[MODEL] Load thành công! VRAM đang chiếm: {used_vram:.2f} GB")

    # 4. Chạy thử 1 câu suy luận tiếng Việt tài chính
    messages = [
        {"role": "system", "content": "Bạn là chuyên gia phân tích tin tức chứng khoán Việt Nam."},
        {"role": "user", "content": "Tin: HPG công bố sản lượng thép tháng 10 tăng 25% so với cùng kỳ. Cảm xúc tin này là gì? Trả lời ngắn trong 1 câu."},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"[INFERENCE] Output mẫu:\n{reply}\n")

    # 5. Kiểm tra ghi log bằng TensorBoard
    log_dir = "./outputs/logs/env-smoke-test"
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_scalar("system/env_ready", 1, 0)
    writer.add_scalar("system/model_vram_gb", round(used_vram, 2), 0)
    writer.add_text("inference/sample_reply", reply, 0)
    writer.close()
    print(f"[TensorBoard] Ghi log thành công vào {log_dir}! Bước 1 hoàn tất.")

if __name__ == "__main__":
    main()