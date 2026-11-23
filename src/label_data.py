import argparse
import json
import os
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tqdm import tqdm

from schema import FinNewsOutput

load_dotenv()

RAW_PATH = Path("data/raw/vn_finnews_raw.jsonl")
LABELED_PATH = Path("data/raw/vn_finnews_labeled.jsonl")

FALLBACK_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
]
BATCH_SIZE = 10
SLEEP_BETWEEN_BATCH = 5


class BatchItemResult(BaseModel):
    id: str = Field(..., description="Giữ nguyên chính xác mã id của bản tin đầu vào")
    label: FinNewsOutput


class BatchLabelResponse(BaseModel):
    results: List[BatchItemResult]


BATCH_SYSTEM_INSTRUCTION = """Bạn là chuyên gia phân tích tài chính và chứng khoán Việt Nam.
Bạn sẽ nhận một danh sách các bản tin ngắn dạng JSON (mỗi tin gồm `id` và `text`).
Hãy phân tích độc lập TỪNG bản tin và trả về kết quả đúng theo schema.

Quy tắc gán nhãn bắt buộc:
1. Giữ nguyên chính xác `id` của từng bản tin đầu vào, không bỏ sót tin nào.
2. `topic`:
   - `earnings`: doanh thu, lợi nhuận, kết quả kinh doanh quý/năm, hoàn thành kế hoạch.
   - `market_trading`: diễn biến VN-Index, tăng/giảm giá cổ phiếu, thanh khoản, khối ngoại, tự doanh, định giá.
   - `corporate_action`: chia cổ tức, phát hành cổ phiếu/trái phiếu, M&A, thoái vốn, thay đổi lãnh đạo, bị xử phạt/khởi tố.
   - `macro_policy`: lãi suất, tỷ giá USD/VND, lạm phát, GDP, giá vàng/dầu thế giới, chính sách NHNN/Chính phủ.
   - `project_investment`: khởi công dự án, trúng thầu, mở rộng nhà máy, phê duyệt quy hoạch, giải ngân đầu tư.
   - `other`: tin kinh doanh/tài chính không thuộc 5 nhóm trên.
3. `overall_sentiment`: đánh giá tác động tài chính chung của bản tin (`positive`, `negative`, `neutral`). Tin trung lập, mang tính tường thuật lịch sự kiện hoặc số liệu không rõ xu hướng gán `neutral`.
4. `entities`: chỉ trích xuất các doanh nghiệp, ngân hàng, tổ chức kinh tế CHÍNH được nhắc tới (tối đa 5 thực thể).
   - `ticker`: CHỈ điền mã chứng khoán Việt Nam 3 ký tự viết hoa (vd: HPG, VCB, FPT, VHM) nếu xác định chắc chắn 100%; nếu là cơ quan nhà nước, công ty chưa niêm yết, hoặc không chắc chắn thì để `null`.
   - `sentiment`: cảm xúc RIÊNG cho từng thực thể đó (hai doanh nghiệp trong cùng 1 tin có thể mang cảm xúc trái ngược nhau).
"""


def load_done_ids() -> set:
    if not LABELED_PATH.exists():
        return set()
    done = set()
    with open(LABELED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["id"])
    return done


def label_batch(
    client: genai.Client, batch: list, start_idx: int = 0
) -> tuple[dict[str, dict], int, str]:
    payload = [{"id": item["id"], "text": item["text"]} for item in batch]
    prompt = json.dumps(payload, ensure_ascii=False)
    n_models = len(FALLBACK_MODELS)

    # Thử xoay vòng tối đa 2 lượt qua toàn bộ danh sách model
    for attempt in range(n_models * 2):
        idx = (start_idx + attempt) % n_models
        current_model = FALLBACK_MODELS[idx]
        try:
            response = client.models.generate_content(
                model=current_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=BATCH_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=BatchLabelResponse,
                ),
            )
            parsed = BatchLabelResponse.model_validate_json(response.text)
            valid_ids = {item["id"] for item in batch}
            result_map = {
                r.id: r.label.model_dump()
                for r in parsed.results
                if r.id in valid_ids
            }
            return result_map, idx, current_model
        except Exception as e:
            err_msg = str(e)
            print(f"\n[Chuyển model] {current_model} lỗi: {err_msg[:140]}")
            # Nếu đã xoay đủ 1 vòng tất cả model đều bận thì nghỉ 20s, còn không chỉ nghỉ 3s rồi đổi model ngay
            wait_sec = 20 if (attempt + 1) % n_models == 0 else 3
            time.sleep(wait_sec)

    return {}, start_idx, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số tin cần gán (0 = gán hết)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    assert api_key, "Thiếu GEMINI_API_KEY trong file .env hoặc biến môi trường!"
    client = genai.Client(api_key=api_key)

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f if line.strip()]

    done_ids = load_done_ids()
    todo = [r for r in all_rows if r["id"] not in done_ids]
    if args.limit > 0:
        todo = todo[: args.limit]

    print(
        f"Models xoay vòng: {FALLBACK_MODELS}\n"
        f"Tổng tin thô: {len(all_rows)} | Đã gán: {len(done_ids)} | Đợt này sẽ gán: {len(todo)}"
    )
    if not todo:
        print("Không còn bản tin nào cần gán nhãn.")
        return

    batches = [todo[i : i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    saved_count = 0
    missed_count = 0
    active_model_idx = 0

    with open(LABELED_PATH, "a", encoding="utf-8") as out_f:
        for batch in tqdm(batches, desc="Labeling batches"):
            result_map, active_model_idx, used_model = label_batch(
                client, batch, start_idx=active_model_idx
            )
            for row in batch:
                if row["id"] in result_map:
                    record = {**row, "teacher_model": used_model, "label": result_map[row["id"]]}
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    saved_count += 1
                else:
                    missed_count += 1
            out_f.flush()
            time.sleep(SLEEP_BETWEEN_BATCH)

    print(f"Hoàn tất! Đã lưu thêm: {saved_count} | Bỏ sót/lỗi: {missed_count}")


if __name__ == "__main__":
    main()