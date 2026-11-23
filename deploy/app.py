import json
import os
import sys
import time
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from schema import FinNewsOutput, SYSTEM_PROMPT

MODEL_ID = os.environ.get("FINNEWS_MODEL_ID", "mmbellu/qwen3-4b-vn-finnews-merged")
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0") == "1"
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"[APP] Đang nạp model {MODEL_ID} ({'4-bit NF4' if LOAD_IN_4BIT else 'FP16'})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

if LOAD_IN_4BIT:
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_cfg, device_map="auto", token=HF_TOKEN
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        token=HF_TOKEN,
    )
model.eval()


def analyze_news(title: str, body: str):
    if not title.strip() and not body.strip():
        return {"error": "Vui lòng nhập tiêu đề hoặc nội dung bản tin."}, "0.00s", "N/A"

    text = f"Tiêu đề: {title.strip()}\nNội dung: {body.strip()}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    latency = time.time() - t0
    raw = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    try:
        parsed = FinNewsOutput.model_validate_json(raw).model_dump()
        status = "Hợp lệ Pydantic (Strict JSON)"
    except Exception as e:
        parsed = {"raw_output": raw, "parse_error": str(e)}
        status = "Lỗi JSON Schema"

    return parsed, f"{latency:.2f}s", status


EXAMPLES = [
    [
        "Hòa Phát báo lãi quý tăng mạnh nhờ sản lượng thép cuộn cán nóng",
        "Tập đoàn Hòa Phát ghi nhận doanh thu và lợi nhuận sau thuế tăng trưởng hai chữ số so với cùng kỳ, vượt kế hoạch đề ra nhờ nhu cầu xây dựng phục hồi.",
    ],
    [
        "Tỷ giá trung tâm lập kỷ lục mới",
        "Tỷ giá trung tâm hôm nay tăng 9 đồng, lên mức kỷ lục mới, trong khi giá USD tại các ngân hàng thương mại vượt 26.000 đồng.",
    ],
    [
        "FPT ký hợp đồng chuyển đổi số 100 triệu USD trong khi đối thủ cùng ngành giảm biên lợi nhuận",
        "Tập đoàn FPT công bố thỏa thuận hợp tác 3 năm với khách hàng Nhật Bản, trái ngược với bối cảnh nhiều doanh nghiệp gia công phần mềm nhỏ bị thu hẹp đơn hàng.",
    ],
]

with gr.Blocks(title="VN Financial News Structured Analyzer") as demo:
    gr.Markdown(
        "# VN Financial News Analyzer (Qwen3-4B QLoRA)\n"
        "Trích xuất tự động **Topic**, **Overall Sentiment** và **Entity-Level Sentiment + Mã cổ phiếu (Ticker)** dưới dạng JSON chuẩn Pydantic."
    )
    with gr.Row():
        with gr.Column():
            title_in = gr.Textbox(label="Tiêu đề bản tin", placeholder="Nhập tiêu đề...")
            body_in = gr.Textbox(label="Nội dung tóm tắt", lines=5, placeholder="Nhập nội dung tin tài chính...")
            run_btn = gr.Button("Phân tích", variant="primary")
        with gr.Column():
            json_out = gr.JSON(label="Kết quả Structured JSON")
            with gr.Row():
                latency_out = gr.Textbox(label="Độ trễ suy luận", interactive=False)
                status_out = gr.Textbox(label="Trạng thái Schema", interactive=False)

    gr.Examples(examples=EXAMPLES, inputs=[title_in, body_in])
    run_btn.click(fn=analyze_news, inputs=[title_in, body_in], outputs=[json_out, latency_out, status_out])

if __name__ == "__main__":
    demo.launch(share=True)