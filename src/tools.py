"""模拟工具函数：查订单、发优惠券、告警。真实场景中替换为真实服务调用。"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .models import Category, Complaint, Compensation, Order, Severity

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 高危客诉关键词：命中即禁止发券，触发拦截工单并引导全额退款
_HIGH_RISK = ("食品安全", "假货", "12315", "中毒", "过期", "变质", "食物中毒", "举报", "消协", "工商")

# 数据池订单无真实金额，按品类填充典型客单价以支撑动态定价演示
_CATEGORY_AMOUNT = {
    "书籍": 45.0, "平板": 2600.0, "手机": 3200.0, "水果": 55.0,
    "衣服": 150.0, "热水器": 1300.0, "洗发水": 75.0, "计算机": 4600.0,
    "酒店": 480.0, "蒙牛": 65.0,
}


def _load_orders() -> list[Order]:
    path = _DATA_DIR / "orders.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Order(**item) for item in raw]


@lru_cache(maxsize=1)
def _orders_from_pool() -> dict[str, Order]:
    """数据池里的差评订单不在 orders.json 中，按 order_id 构造模拟订单以支撑查单链路。"""
    from .dataset import load_reviews

    return {
        r.order_id: Order(
            order_id=r.order_id, user_id=r.user_id, product=r.category,
            status="已完成", amount=_CATEGORY_AMOUNT.get(r.category, 100.0),
        )
        for r in load_reviews()
    }


def query_order(order_id: str) -> Optional[Order]:
    """按订单号查询订单：真实订单优先，其次从数据池构造，找不到返回 None。"""
    for order in _load_orders():
        if order.order_id == order_id:
            return order
    return _orders_from_pool().get(order_id)


def calculate_compensation(
    order_amount: float, severity: Severity, category: Category, content: str = "",
) -> Compensation:
    """动态定价 + 风控熔断：低度固定小额券 / 中度按订单金额 10% / 重度高危禁止发券。"""
    if severity == Severity.high:
        if any(k in content for k in _HIGH_RISK):
            return Compensation(
                amount=0.0, status="high_risk",
                reason="高危客诉（食品安全/假货/投诉），禁止自动发券，触发高危拦截工单并引导先行全额退款",
            )
        return Compensation(amount=0.0, status="alert", reason="重度客诉，升级人工跟进，不发券")
    if severity == Severity.medium:
        amount = round(max(5.0, order_amount * 0.1), 2)
        if amount > 30.0:
            return Compensation(
                amount=amount, status="pending",
                reason=f"建议补偿 {amount} 元超 30 元自动上限，转人工审批",
            )
        return Compensation(
            amount=amount, status="auto",
            reason=f"中度客诉，按订单金额 10% 动态补偿 {amount} 元",
        )
    amount = round(min(5.0, order_amount * 0.15), 2)
    return Compensation(
        amount=amount, status="auto",
        reason=f"低度客诉，发放固定小额券 {amount} 元（不超过订单金额 15%）",
    )


def send_coupon(user_id: str, amount: float) -> dict:
    """模拟：向用户发放优惠券。"""
    return {
        "status": "ok",
        "user_id": user_id,
        "amount": amount,
        "msg": f"已向用户 {user_id} 发放 {amount} 元优惠券",
    }


def send_alert(complaint: Complaint, reason: str) -> dict:
    """模拟：上报告警，触发人工/升级处理。"""
    return {
        "status": "ok",
        "order_id": complaint.order_id,
        "reason": reason,
        "msg": f"客诉告警已上报：{reason}",
    }
