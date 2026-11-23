import hashlib
import json
import re
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup

OUT_PATH = Path("data/raw/vn_finnews_raw.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

TARGET_TOTAL = 1700  # Dư nhẹ để sau khi lọc vẫn còn > 1.500 mẫu sạch
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    )
}

VNEXPRESS_CATEGORIES = {
    "vnexpress_chungkhoan": "https://vnexpress.net/kinh-doanh/chung-khoan",
    "vnexpress_doanhnghiep": "https://vnexpress.net/kinh-doanh/doanh-nghiep",
    "vnexpress_ebank": "https://vnexpress.net/kinh-doanh/ebank",
    "vnexpress_vimo": "https://vnexpress.net/kinh-doanh/vi-mo",
}

CAFEF_ZONES = {
    "cafef_chungkhoan": "18831",
    "cafef_doanhnghiep": "18836",
    "cafef_taichinh": "18834",
    "cafef_vimo": "18833",
}


def clean_text(raw: str) -> str:
    return re.sub(r"\s+", " ", raw or "").strip()


def make_id(title: str) -> str:
    norm = re.sub(r"\W+", "", title.lower())
    return hashlib.md5(norm.encode("utf-8")).hexdigest()[:12]


def load_existing_ids() -> set:
    if not OUT_PATH.exists():
        return set()
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f}


def crawl_vnexpress(seen_ids: set, out_f, max_pages: int = 20):
    added = 0
    for source, base_url in VNEXPRESS_CATEGORIES.items():
        for page in range(1, max_pages + 1):
            if len(seen_ids) >= TARGET_TOTAL:
                return added
            url = base_url if page == 1 else f"{base_url}-p{page}"
            try:
                resp = requests.get(url, headers=HEADERS, timeout=15)
                if resp.status_code != 200:
                    break
                soup = BeautifulSoup(resp.text, "html.parser")
                articles = soup.select("article.item-news")
                if not articles:
                    break

                page_added = 0
                for art in articles:
                    title_el = art.select_one(".title-news a")
                    desc_el = art.select_one("p.description")
                    if not title_el:
                        continue
                    title = clean_text(title_el.get_text())
                    summary = clean_text(desc_el.get_text()) if desc_el else ""
                    if len(title) < 20 or len(summary) < 25:
                        continue

                    item_id = make_id(title)
                    if item_id in seen_ids:
                        continue

                    record = {
                        "id": item_id,
                        "source": source,
                        "published": "",
                        "text": f"Tiêu đề: {title}\nNội dung: {summary}",
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    seen_ids.add(item_id)
                    added += 1
                    page_added += 1

                if page % 5 == 0:
                    print(f"  [{source}] trang {page} -> tổng kho: {len(seen_ids)}")
            except Exception as e:
                print(f"  [Lỗi {url}] {e}")
            time.sleep(0.4)
    return added


def crawl_cafef(seen_ids: set, out_f, max_pages: int = 25):
    added = 0
    for source, zone_id in CAFEF_ZONES.items():
        for page in range(1, max_pages + 1):
            if len(seen_ids) >= TARGET_TOTAL:
                return added
            url = f"https://cafef.vn/timelinelist/{zone_id}/{page}.chn"
            try:
                resp = requests.get(url, headers=HEADERS, timeout=15)
                if resp.status_code != 200 or not resp.text.strip():
                    break
                soup = BeautifulSoup(resp.text, "html.parser")
                items = soup.select(".tlitem")
                if not items:
                    break

                for item in items:
                    title_el = item.select_one("h3 a, h2 a")
                    desc_el = item.select_one("p.sapo")
                    if not title_el:
                        continue
                    title = clean_text(title_el.get_text())
                    summary = clean_text(desc_el.get_text()) if desc_el else ""
                    if len(title) < 20 or len(summary) < 25:
                        continue

                    item_id = make_id(title)
                    if item_id in seen_ids:
                        continue

                    record = {
                        "id": item_id,
                        "source": source,
                        "published": "",
                        "text": f"Tiêu đề: {title}\nNội dung: {summary}",
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    seen_ids.add(item_id)
                    added += 1

                if page % 5 == 0:
                    print(f"  [{source}] trang {page} -> tổng kho: {len(seen_ids)}")
            except Exception as e:
                print(f"  [Lỗi {url}] {e}")
            time.sleep(0.4)
    return added


def main():
    seen_ids = load_existing_ids()
    print(f"Đang có sẵn: {len(seen_ids)} bản tin. Mục tiêu: {TARGET_TOTAL} bản tin.")

    with open(OUT_PATH, "a", encoding="utf-8") as out_f:
        print("1) Đang cào CafeF (ưu tiên tin mã cổ phiếu/doanh nghiệp)...")
        n_cafef = crawl_cafef(seen_ids, out_f, max_pages=25)
        out_f.flush()
        print(f"   -> Thêm {n_cafef} bản tin từ CafeF.")

        if len(seen_ids) < TARGET_TOTAL:
            print("2) Đang cào VnExpress Kinh doanh...")
            n_vne = crawl_vnexpress(seen_ids, out_f, max_pages=20)
            out_f.flush()
            print(f"   -> Thêm {n_vne} bản tin từ VnExpress.")

    print(f"Hoàn tất! Tổng kho hiện có: {len(seen_ids)} bản tin tại {OUT_PATH}")


if __name__ == "__main__":
    main()