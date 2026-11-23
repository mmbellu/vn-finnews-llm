import hashlib
import json
import re
from pathlib import Path
import feedparser
from bs4 import BeautifulSoup

RSS_FEEDS = {
    "vnexpress_kinhdoanh": "https://vnexpress.net/rss/kinh-doanh.rss",
    "cafef_chungkhoan": "https://cafef.vn/thi-truong-chung-khoan.rss",
    "cafef_doanhnghiep": "https://cafef.vn/doanh-nghiep.rss",
    "cafef_taichinh": "https://cafef.vn/tai-chinh-ngan-hang.rss",
    "cafef_vimo": "https://cafef.vn/vi-mo-dau-tu.rss",
    "vietstock_chungkhoan": "https://vietstock.vn/830/chung-khoan/co-phieu.rss",
    "vietstock_doanhnghiep": "https://vietstock.vn/737/doanh-nghiep/hoat-dong-kinh-doanh.rss",
    "vietstock_taichinh": "https://vietstock.vn/757/tai-chinh/ngan-hang.rss",
}

OUT_PATH = Path("data/raw/vn_finnews_raw.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def clean_html(raw_text: str) -> str:
    text = BeautifulSoup(raw_text or "", "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def make_id(title: str) -> str:
    # Dùng hash của tiêu đề chuẩn hóa để chống trùng lặp tin cùng nội dung
    norm = re.sub(r"\W+", "", title.lower())
    return hashlib.md5(norm.encode("utf-8")).hexdigest()[:12]


def load_existing_ids() -> set:
    if not OUT_PATH.exists():
        return set()
    ids = set()
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            ids.add(json.loads(line)["id"])
    return ids


def main():
    seen_ids = load_existing_ids()
    new_count = 0

    with open(OUT_PATH, "a", encoding="utf-8") as out_f:
        for source, url in RSS_FEEDS.items():
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = clean_html(entry.get("title", ""))
                summary = clean_html(entry.get("summary", ""))
                if not title or len(title) < 20:
                    continue

                item_id = make_id(title)
                if item_id in seen_ids:
                    continue

                text = f"Tiêu đề: {title}\nNội dung: {summary}" if summary else f"Tiêu đề: {title}"
                record = {
                    "id": item_id,
                    "source": source,
                    "published": entry.get("published", ""),
                    "text": text,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen_ids.add(item_id)
                new_count += 1

    print(f"Thu thập thêm {new_count} bản tin mới. Tổng kho hiện có: {len(seen_ids)} bản tin.")


if __name__ == "__main__":
    main()