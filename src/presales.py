"""售前运营模块：全品类普适型爆款上架与高权重标签检索助手（可选联网对标）。"""
from __future__ import annotations

import logging

from .llm import llm
from .models import Benchmark, FaqItem, ListingPlan, TitleSet

logger = logging.getLogger("presales")

CATEGORIES = (
    "3C数码",
    "服饰鞋包",
    "潮玩二次元/非标周边",
    "图书教材/学术资料",
    "美妆个护",
    "家居日用/其他",
)

PLATFORMS = ("闲鱼", "小红书店铺", "淘宝", "抖音电商")

# 品类爆款范式：AI 依品类自动匹配对应圈子的爆款文案框架。
_CATEGORY_FRAMEWORKS = {
    "3C数码": "数码圈范式：强调一手自用/无拆无修、序列号与保修状态、关键配置参数如实标注、可验机",
    "服饰鞋包": "穿搭圈范式：强调上身/穿搭场景、版型与尺码、成色（几成新）与穿着次数",
    "潮玩二次元/非标周边": "潮玩圈范式：强调限量/联名、盒况与配件、保真/渠道、成色",
    "图书教材/学术资料": "图书圈范式：强调版本与印次、笔记/划线成色、是否含光盘/赠品",
    "美妆个护": "美妆圈范式：强调正品/渠道、保质期与开封状态、肤质适配、余量",
    "家居日用/其他": "家居圈范式：强调成色、尺寸规格、适用场景、是否含配件",
}


def search_market_benchmarks(keyword: str, category: str, platform: str = "闲鱼") -> list[Benchmark]:
    """全网对标检索：自动生成垂直搜索词，4 秒超时 + 全异常静默降级，失败返回空列表。"""
    query = f"{keyword} {category} {platform} 爆款 文案 穿搭"
    try:
        from duckduckgo_search import DDGS  # 延迟导入：未安装/网络异常时静默降级
        ddgs = DDGS(timeout=4)
        raw = ddgs.text(query, max_results=3)
    except Exception as exc:
        logger.warning("全网对标检索已平滑降级：%s", exc)
        return []

    out: list[Benchmark] = []
    for r in raw:
        title = (r.get("title") or "").strip()
        body = (r.get("body") or "").strip()
        if title or body:
            out.append(Benchmark(title=title, body=body))
    return out[:3]


def generate_listing(
    category: str,
    platform: str,
    pricing: str,
    facts: str,
    flaws: str = "",
    benchmarks: list[Benchmark] | None = None,
) -> ListingPlan:
    """生成爆款上架的 4 块结构化成果；无 DeepSeek 时降级为规则模板兜底。"""
    benchmarks = benchmarks or []
    if llm:
        try:
            return _generate_llm(category, platform, pricing, facts, flaws, benchmarks)
        except Exception as exc:
            logger.warning("售前文案 LLM 生成失败，降级为规则模板：%s", exc)
    return _generate_rule(category, platform, pricing, facts, flaws)


def _generate_llm(category: str, platform: str, pricing: str, facts: str, flaws: str, benchmarks: list[Benchmark]) -> ListingPlan:
    structured = llm.with_structured_output(ListingPlan, method="function_calling")
    framework = _CATEGORY_FRAMEWORKS.get(category, "通用范式：如实呈现商品事实、成色与瑕疵")

    bench_section = ""
    if benchmarks:
        lines = "\n".join(f"- {b.title}｜{b.body[:60]}" for b in benchmarks)
        bench_section = f"【全网对标参考（仅供排版结构/促单语气/高频标签借鉴）】\n{lines}\n"

    prompt = (
        "你是资深电商爆款上架与标签检索专家。请基于用户提供的客观事实，输出 4 块结构化成果。\n"
        f"【品类爆款范式】{framework}。\n"
        f"{bench_section}"
        "【严格去幻觉（最高安全死命令）】网络对标结果仅供参考其排版结构、促单语气和高频标签；"
        "商品的所有真实事实（尺码数值、瑕疵位置、成色、售价）必须 100% 严格来源于用户的原始输入，"
        "严禁将检索到的外部尺码/价格拼接到输出中！\n"
        "① titles：3 款爆款高点击标题——search=搜索型（堆品类/型号/成色关键词）、"
        "transfer=诚心转让型（突出真实来源与诚信）、vibe=氛围型（营造使用场景与情绪氛围），每个标题 25 字以内。\n"
        "② detail_copy：结构化吸睛详情页文案，按「核心事实 → 亮点/场景 → 诚信说明(含瑕疵) → 催促下单」组织，180 字左右，一段正文。\n"
        "③ faqs：售前高频拦截 FAQ，写 3 组买家最常问的问题与高情商回复，针对该品类真实痛点，每条回答 60 字以内。\n"
        "④ tags：精选 6~10 个去重的圈内真实高频搜索词（SEO 流量抓手），输出为短语列表。\n"
        f"商品品类：{category}\n目标平台：{platform}\n期望售价/定价：{pricing}\n"
        f"商品核心事实与规格：{facts or '（未提供）'}\n真实瑕疵与特殊说明：{flaws or '无'}\n"
    )
    return structured.invoke(prompt)


def _generate_rule(category: str, platform: str, pricing: str, facts: str, flaws: str) -> ListingPlan:
    """无 LLM 时的模板兜底，保证页面始终有可复制的成果。"""
    facts_text = (facts or "品质好物").strip()
    flaw_note = f"（瑕疵已如实告知：{flaws}）" if flaws else ""
    parts = [p.strip() for p in facts_text.replace("，", "、").replace(",", "、").split("、") if p.strip()]

    tags = [category] + parts[:3] + [pricing, "自用转卖", "诚心转让", "好价"]
    tags = list(dict.fromkeys(tags))[:8]

    return ListingPlan(
        titles=TitleSet(
            search=f"{category} {facts_text} {pricing} 自用转卖 好价",
            transfer=f"诚心转让｜{category} {facts_text}，{flaws or '成色如图'}，{pricing}",
            vibe=f"入手不亏的{category}，{facts_text}，{pricing}带走",
        ),
        detail_copy=(
            f"{category}，{facts_text}。{flaw_note}均为实拍实述、诚信告知，{pricing}好价。"
            f"喜欢可小刀，拍下尽快发货，早买早享受。"
        ),
        faqs=[
            FaqItem(
                question=f"是自用还是全新？{'有什么瑕疵吗？' if flaws else '成色如何？'}",
                answer=f"亲，{facts_text}，{flaws or '成色如实'}，都是如实描述，需要细节图我再拍给您～",
            ),
            FaqItem(
                question="能不能便宜点 / 包邮吗？",
                answer=f"亲，{pricing}已经是诚意价啦，诚心要可小刀，拍下尽快发货～",
            ),
            FaqItem(
                question="支持验货 / 退换吗？",
                answer="亲，如实描述、所见即所得，发货前会再确认；若与描述不符可沟通，诚信第一～",
            ),
        ],
        tags=tags,
    )
