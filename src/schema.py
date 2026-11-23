from typing import List, Literal, Optional
from pydantic import BaseModel, Field

TopicType = Literal[
    "earnings",           # Kết quả kinh doanh, doanh thu, lợi nhuận
    "market_trading",     # Biến động giá cổ phiếu, dòng tiền, khối ngoại, VN-Index
    "corporate_action",   # Cổ tức, phát hành thêm, M&A, nhân sự cấp cao, vi phạm
    "macro_policy",       # Lãi suất, tỷ giá, GDP, chính sách ngân hàng nhà nước
    "project_investment", # Dự án mới, trúng thầu, mở rộng nhà máy, bất động sản
    "other",              # Tin tài chính/kinh doanh khác
]

SentimentType = Literal["positive", "negative", "neutral"]


class EntitySentiment(BaseModel):
    name: str = Field(..., description="Tên doanh nghiệp, tổ chức hoặc mã chứng khoán")
    ticker: Optional[str] = Field(
        None, description="Mã cổ phiếu 3 ký tự viết hoa nếu xác định chắc chắn (vd: HPG, VCB), ngược lại là null"
    )
    sentiment: SentimentType = Field(..., description="Cảm xúc riêng đối với thực thể này")


class FinNewsOutput(BaseModel):
    topic: TopicType
    overall_sentiment: SentimentType
    entities: List[EntitySentiment] = Field(
        default_factory=list,
        description="Danh sách các doanh nghiệp/tổ chức chính được nhắc tới trong tin",
    )


SYSTEM_PROMPT = """Bạn là hệ thống trích xuất thông tin tài chính chứng khoán Việt Nam.
Phân tích đoạn tin ngắn được cung cấp và CHỈ trả về một đối tượng JSON hợp lệ (không thêm markdown, không giải thích) theo đúng cấu trúc:
{
  "topic": "earnings" | "market_trading" | "corporate_action" | "macro_policy" | "project_investment" | "other",
  "overall_sentiment": "positive" | "negative" | "neutral",
  "entities": [
    {"name": "<tên thực thể>", "ticker": "<mã CK hoặc null>", "sentiment": "positive" | "negative" | "neutral"}
  ]
}"""