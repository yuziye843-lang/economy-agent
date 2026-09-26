"""数据模型定义：所有数据流转统一使用 Pydantic 类型注解。"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Category(str, Enum):
    """客诉/评价分类（6 大标准维度，与差评归因统一）。"""
    quality = "商品品质"      # 破损、质量瑕疵、异味、变质、做工粗糙、功能故障
    logistics = "物流配送"    # 时效慢、丢件、派送态度恶劣、未送货上门
    packaging = "包装耗材"    # 外包装挤压变形、破损漏液、缺少防震泡沫
    mismatch = "描述不符"     # 严重色差、尺码不准、版型不合、货不对板、虚假宣传
    service = "履约服务"      # 客服不理人/推诿、漏发少发、降价不退差价、退换货受阻
    other = "其他/主观偏好"   # 仅纯主观审美/情绪发泄，严格限制使用


class Severity(str, Enum):
    """严重程度。"""
    low = "低"
    medium = "中"
    high = "高"


class Action(str, Enum):
    """处置动作。"""
    send_coupon = "发优惠券"
    escalate_alert = "告警升级"
    manual = "人工处理"
    no_action = "无需处理"
    pending_approval = "待人工审批"
    high_risk = "高危拦截"


class Review(BaseModel):
    """一条商品评价（数据源）。"""
    review_id: str
    order_id: str
    user_id: str
    category: str
    content: str
    rating: int = Field(ge=1, le=5)


class Complaint(BaseModel):
    """客诉输入：来自用户或运营的原始诉求。"""
    user_id: str
    order_id: str
    content: str


class Order(BaseModel):
    """订单信息。"""
    order_id: str
    user_id: str
    product: str
    status: str
    amount: float


class Classification(BaseModel):
    """分类节点输出：类别 + 严重程度 + 一句话摘要。"""
    category: Category
    severity: Severity
    summary: str


class Decision(BaseModel):
    """决策节点输出：动作 + 理由 + 可选优惠券金额。"""
    action: Action
    reason: str
    coupon_amount: Optional[float] = None


class Compensation(BaseModel):
    """动态补偿计算结果。status: auto=自动发券 / pending=待人工审批 / high_risk=高危拦截 / alert=升级告警。"""
    amount: float
    status: str
    reason: str


class AgentState(BaseModel):
    """LangGraph 全局状态。"""
    complaint: Complaint
    classification: Optional[Classification] = None
    order: Optional[Order] = None
    decision: Optional[Decision] = None
    actions: list[str] = Field(default_factory=list)
    final_response: Optional[str] = None
    reply_text: Optional[str] = None


class CategoryStat(BaseModel):
    """单个分类维度的统计（差评大盘归因）。"""
    cause: Category
    count: int
    ratio: float = Field(ge=0, le=1)


class NegativeReport(BaseModel):
    """差评聚合报告（按分类维度归因）。"""
    total: int
    stats: list[CategoryStat]
    top_reviews: list[Review]
    advice: str


class CategoryReport(BaseModel):
    """品类专项下钻报告。"""
    category: str
    total: int
    negative: int
    neg_rate: float
    top_causes: list[str]
    typical_reviews: list[Review]
    advice: str


class TitleSet(BaseModel):
    """爆款标题三连：搜索型 / 诚心转让型 / 氛围型。"""
    search: str = Field(description="搜索型标题：堆品类/型号/成色等关键词，利于搜索命中")
    transfer: str = Field(description="诚心转让型标题：突出真实来源与诚信，如自用转手")
    vibe: str = Field(description="氛围型标题：营造使用场景与情绪氛围")


class FaqItem(BaseModel):
    """售前 FAQ 单条问答。"""
    question: str = Field(description="买家高频提问")
    answer: str = Field(description="高情商标准回复")


class ListingPlan(BaseModel):
    """售前选品上架的 4 块结构化成果。"""
    titles: TitleSet = Field(description="3 款爆款高点击标题")
    detail_copy: str = Field(description="结构化详情页文案（纯文本：中文【】分段+空行+破折号/点号列表，无 Markdown/Emoji）")
    faqs: list[FaqItem] = Field(description="3 组售前高频拦截 FAQ")
    tags: list[str] = Field(description="6~10 个去重的圈内真实高频搜索词")
