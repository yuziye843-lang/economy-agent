"""数据模型定义：所有数据流转统一使用 Pydantic 类型注解。"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Category(str, Enum):
    """客诉/评价类别。"""
    logistics = "物流"
    quality = "品质"
    service = "服务"
    other = "其他"


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


class RootCause(str, Enum):
    """客诉诱因（Root Cause）——差评大盘归因维度。"""
    logistics = "物流配送"
    quality = "商品品质"
    service = "售后服务"
    packaging = "包装破损"
    other = "其他"


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


class RootCauseStat(BaseModel):
    """单个客诉诱因的统计。"""
    cause: RootCause
    count: int
    ratio: float = Field(ge=0, le=1)


class NegativeReport(BaseModel):
    """差评聚合报告（按客诉诱因归因）。"""
    total: int
    stats: list[RootCauseStat]
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
