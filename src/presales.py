"""售前运营模块：同行结构克隆与反偷懒排版重构的上架助手。"""
from __future__ import annotations

import logging

from .llm import llm
from .models import FaqItem, ListingPlan, TitleSet

logger = logging.getLogger("presales")

# 高粉卖家标准分模块模板（用户未提供同行参考文案时强制套用）。
_STANDARD_TEMPLATE = (
    "【📐 尺寸档案】→【👗 实穿亮点】→【🔍 瑕疵诚信告知】→【⚠️ 拍前协议】→【🏷️ 高频 Tag】"
)


def generate_listing(facts: str, flaws: str, price: str, reference: str = "") -> ListingPlan:
    """生成强结构化上架文案；无 DeepSeek 时降级为规则模板兜底。"""
    if llm:
        try:
            return _generate_llm(facts, flaws, price, reference)
        except Exception as exc:
            logger.warning("上架文案 LLM 生成失败，降级为规则模板：%s", exc)
    return _generate_rule(facts, flaws, price)


def _generate_llm(facts: str, flaws: str, price: str, reference: str) -> ListingPlan:
    structured = llm.with_structured_output(ListingPlan, method="function_calling")

    if reference.strip():
        clone_rule = (
            "用户提供了【同行参考文案】，处理流程：\n"
            "1. 先提取其视觉排版骨架（分段小标题、Emoji 使用习惯、拍前免责声明结构）；\n"
            "2. 彻底销毁参考文案中的具体尺码、颜色、瑕疵、价格等外部事实；\n"
            "3. 把用户【输入区 A】的真实事实，像填空题一样精准填入提取出的骨架。\n"
        )
    else:
        clone_rule = (
            "用户未提供同行参考文案，强制套用高粉卖家的标准分模块模板：\n"
            f"{_STANDARD_TEMPLATE}\n"
        )

    prompt = (
        "你是资深二手/闲置电商高粉卖家，擅长把零散事实改写成极具视觉排版感的上架文案。\n"
        f"{clone_rule}"
        "【反偷懒死命令】\n"
        "1. 严禁输出连贯的普通段落！必须用清晰小标题（Emoji 开头）+ 列表分行 + 空行营造视觉呼吸感。\n"
        "2. 严禁把用户原句仅改标点就输出！必须重新组织语序、拆成要点、补足促单语气。\n"
        "3. 严格事实隔离：所有尺码数值、瑕疵位置、成色、价格必须 100% 来自用户输入，严禁无中生有。\n"
        "【输出结构】\n"
        "① titles：3 款高点击标题——search=搜索型、transfer=诚心转让型、vibe=氛围型，各 25 字内。\n"
        "② detail_copy：正文务必用 Markdown 分模块排版（小标题 + 列表分行 + 空行），严禁大段连贯段落。\n"
        "③ faqs：3 组售前拦截 FAQ，每条回答 60 字内。\n"
        "④ tags：6~10 个去重高频搜索词。\n"
        f"【我的商品真实档案】\n商品事实与版型：{facts}\n真实瑕疵与成色：{flaws or '无'}\n价格与交易方式：{price}\n"
        f"【同行参考文案】\n{reference or '（未提供，套用标准模板）'}\n"
    )
    return structured.invoke(prompt)


def _generate_rule(facts: str, flaws: str, price: str) -> ListingPlan:
    """无 LLM 时的规则模板兜底，输出标准分模块排版，保证页面始终可用。"""
    facts_text = (facts or "品质好物").strip()
    flaws_text = flaws.strip()
    price_text = price.strip() or "诚意价"
    points = [p.strip() for p in facts_text.replace("，", "、").replace(",", "、").split("、") if p.strip()]
    first = points[0] if points else facts_text

    tags = list(dict.fromkeys(points[:3] + [price_text, "自用转卖", "诚心转让"]))[:8]

    lines = ["【📐 尺寸档案】"]
    for p in (points or [facts_text]):
        lines.append(f"- {p}")
    lines += [
        "",
        "【👗 实穿亮点】",
        f"- {first}，上身自然、日常好搭，细节质感在线",
        "",
        "【🔍 瑕疵诚信告知】",
        f"- {flaws_text or '整体成色良好，无明显瑕疵'}",
        "",
        "【⚠️ 拍前协议】",
        f"- 价格：{price_text}，可直拍/可小刀",
        "- 二手闲置一般不退不换，拍前可看细节图，诚信第一",
        "",
        "【🏷️ 高频 Tag】",
        " ".join(f"#{t}" for t in tags),
    ]
    detail_copy = "\n".join(lines)

    return ListingPlan(
        titles=TitleSet(
            search=f"{facts_text} {price_text} 自用转卖 好价",
            transfer=f"诚心转让｜{facts_text}，{flaws_text or '成色如图'}，{price_text}",
            vibe=f"入手不亏的{facts_text}，{price_text}带走",
        ),
        detail_copy=detail_copy,
        faqs=[
            FaqItem(
                question="尺寸 / 成色怎么样？",
                answer=f"亲，{facts_text}；瑕疵情况：{flaws_text or '成色良好'}，均如实描述，可拍细节图～",
            ),
            FaqItem(
                question="能便宜点 / 包邮吗？",
                answer=f"亲，{price_text}已是诚意价，诚心要可小刀，拍下尽快发货～",
            ),
            FaqItem(
                question="支持退换吗？",
                answer="亲，二手闲置一般不退不换，但如实描述、所见即所得，拍前可沟通清楚～",
            ),
        ],
        tags=tags,
    )
