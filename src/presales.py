"""售前运营模块：AI 智能选品与爆款上架助手（标题 / 详情页 / FAQ / 种草文案）。"""
from __future__ import annotations

import logging

from .llm import llm
from .models import FaqItem, ListingPlan, TitleSet

logger = logging.getLogger("presales")

PLATFORMS = ("闲鱼", "淘宝", "小红书店铺", "抖音电商")


def generate_listing(category: str, selling_points: str, platform: str, pricing: str) -> ListingPlan:
    """生成爆款上架的 4 块结构化成果；无 DeepSeek 时降级为规则模板兜底。"""
    if llm:
        try:
            return _generate_llm(category, selling_points, platform, pricing)
        except Exception as exc:
            logger.warning("售前文案 LLM 生成失败，降级为规则模板：%s", exc)
    return _generate_rule(category, selling_points, platform, pricing)


def _generate_llm(category: str, selling_points: str, platform: str, pricing: str) -> ListingPlan:
    structured = llm.with_structured_output(ListingPlan, method="function_calling")
    prompt = (
        "你是资深电商爆款选品与上架专家，请针对给定商品输出 4 块结构化成果。\n"
        "① titles：黄金点击率标题，3 个不同风格——search=搜索流（堆品类/卖点关键词、利于搜索命中）、"
        "emotion=情绪流（制造共鸣与身份认同）、promo=促销流（突出优惠与紧迫感），每个标题 20 字以内。\n"
        "② detail_copy：详情页吸睛文案，严格按「痛点 → 卖点 → 催促下单」三段结构，150 字左右，输出一段正文。\n"
        "③ faqs：售前高频防踩坑 FAQ，写 3 条买家最常问的问题与高情商标准回复，每条回答 60 字以内、语气真诚。\n"
        "④ social_copy：小红书爆款图文种草文案，带 Emoji，含吸睛标题+正文+#话题标签，150 字左右。\n"
        f"商品品类：{category}\n核心卖点：{selling_points}\n目标平台：{platform}\n定价策略：{pricing}\n"
        "内容必须贴合品类与卖点，避免空泛套话。"
    )
    return structured.invoke(prompt)


def _generate_rule(category: str, selling_points: str, platform: str, pricing: str) -> ListingPlan:
    """无 LLM 时的模板兜底，保证页面始终有可复制的成果。"""
    return ListingPlan(
        titles=TitleSet(
            search=f"{category} {selling_points} {pricing} 品质好物 热销爆款",
            emotion=f"懂行的都在回购这款{category}，{selling_points}，用过就回不去",
            promo=f"限时特惠｜{category} {pricing}，{selling_points}，手慢无",
        ),
        detail_copy=(
            f"还在为挑{category}反复踩坑？这款主打{selling_points}，从源头解决你的痛点。"
            f"精选原料与工艺，品质看得见，{pricing}的定位让它性价比拉满。"
            f"库存有限，喜欢就现在下单，早买早享受！"
        ),
        faqs=[
            FaqItem(
                question=f"这款{category}质量怎么样？",
                answer=f"亲，我们主打{selling_points}，均经过严格质检，品质有保障，请放心选购～",
            ),
            FaqItem(
                question="多久能发货？",
                answer="亲，拍下后我们会在 48 小时内尽快安排发货，物流进度随时可查～",
            ),
            FaqItem(
                question="不合适能退吗？",
                answer="亲，支持七天无理由退换，收到货若有任何问题都可以联系客服，我们一定妥善处理～",
            ),
        ],
        social_copy=(
            f"🔥 挖到宝了！这款{category}也太好用了吧\n\n"
            f"✨ 亮点：{selling_points}\n"
            f"💰 价格：{pricing}\n\n"
            f"姐妹们闭眼入，{platform}同款！#好物分享 #爆款 #种草 #购物清单"
        ),
    )
