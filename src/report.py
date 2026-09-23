"""差评聚合与客诉诱因归因报告。"""
from __future__ import annotations

from collections import Counter

from .dataset import load_reviews
from .llm import LLM_AVAILABLE, llm
from .models import CategoryReport, NegativeReport, Review, RootCause, RootCauseStat

NEGATIVE_THRESHOLD = 2  # rating <= 2 视为差评（本数据集中差评=1）

_PACKAGING = ("包装", "破损", "摔", "压坏", "开裂", "碎了", "漏", "瘪", "挤压", "盒子", "外壳", "变形", "掉漆", "划痕")
_LOGISTICS = ("物流", "快递", "配送", "发货", "延迟", "丢件", "没收到", "没到货", "不到货", "时效", "揽收", "送达", "送货", "催", "缺货", "发错", "漏发", "少发", "不发货", "没发货", "迟迟", "到货", "太慢", "很慢", "慢", "还没到", "还没收到")
_SERVICE = ("客服", "态度", "售后", "不理", "敷衍", "退款", "退货", "换货", "回复", "投诉", "处理", "赔偿", "不处理", "联系", "发票", "沟通", "无人", "没人", "响应", "服务", "维修", "上门")
_QUALITY = ("质量", "坏了", "瑕疵", "色差", "异味", "假货", "太假", "假的", "难吃", "口感", "做工", "掉毛", "褪色", "缩水", "起球", "不好", "太差", "很差", "垃圾", "失灵", "不能用", "尺寸", "偏小", "偏大", "面料", "失望", "不满意", "不如图", "描述不符", "太薄", "太厚", "太重", "太轻", "薄", "厚", "窄", "烂", "功能", "不支持", "无法", "分辨率", "内存", "电池", "模糊", "赠品", "劣质", "味道", "缺陷", "分区", "识别", "容量", "卡顿", "不甜", "太酸", "太硬", "太软", "不好用", "难用", "卫生", "灰尘", "霉味", "脏", "旧", "骗子", "没送", "驱动", "图片", "出入", "不适合", "没意义", "无聊", "缺页", "白纸", "隔音", "吵", "头皮屑", "拉链", "水分", "不能吃", "信号", "关机", "涩", "酸", "净重", "缺斤")


def classify_root_cause(content: str) -> RootCause:
    """基于关键词规则识别差评的客诉诱因（全量快速归因）。"""
    text = content
    if any(k in text for k in _PACKAGING):
        return RootCause.packaging
    if any(k in text for k in _LOGISTICS):
        return RootCause.logistics
    if any(k in text for k in _SERVICE):
        return RootCause.service
    if any(k in text for k in _QUALITY):
        return RootCause.quality
    return RootCause.other


def build_report(reviews: list[Review] | None = None) -> NegativeReport:
    reviews = reviews if reviews is not None else load_reviews()
    negative = [r for r in reviews if r.rating <= NEGATIVE_THRESHOLD]

    counts: dict[RootCause, int] = {}
    for r in negative:
        cause = classify_root_cause(r.content)
        counts[cause] = counts.get(cause, 0) + 1

    total = len(negative)
    stats = [
        RootCauseStat(cause=c, count=n, ratio=round(n / total, 4) if total else 0.0)
        for c, n in sorted(counts.items(), key=lambda x: -x[1])
    ]
    top_reviews = sorted(negative, key=lambda r: r.rating)[:5]
    advice = _generate_advice(stats, negative, total)

    return NegativeReport(total=total, stats=stats, top_reviews=top_reviews, advice=advice)


def _generate_advice(stats: list[RootCauseStat], negative: list[Review], total: int) -> str:
    if not negative:
        return "本期无差评，运营状态良好。"
    if LLM_AVAILABLE:
        try:
            return _generate_advice_llm(stats, negative, total)
        except Exception:
            pass
    return _generate_advice_rule(stats, negative, total)


def _generate_advice_llm(stats: list[RootCauseStat], negative: list[Review], total: int) -> str:
    distribution = "；".join(f"{s.cause.value} {s.count} 条（{s.ratio:.0%}）" for s in stats)
    samples = "\n".join(f"- {r.content[:50]}" for r in negative[:10])
    prompt = (
        "你是电商运营负责人。根据以下差评的客诉诱因分布与代表样本，输出《经营决策改善建议》："
        "针对运营主管的 3~5 条可执行行动指南，结合具体诱因提出可落地的改进措施，200 字以内，直接输出建议正文。\n"
        f"差评总数：{total}\n诱因分布：{distribution}\n代表差评：\n{samples}"
    )
    return llm.invoke(prompt).content.strip()


def _generate_advice_rule(stats: list[RootCauseStat], negative: list[Review], total: int) -> str:
    tips = {
        RootCause.logistics: "建议核查合作物流时效与破损率，对高延迟线路考虑切换承运商或增加时效承诺。",
        RootCause.quality: "建议加强来料质检与供应商抽检，对问题批次启动下架或召回。",
        RootCause.service: "建议补充售后客服排班与话术培训，缩短响应时长并优化退换货流程。",
        RootCause.packaging: "建议升级打包耗材（缓冲材/冷链保温），对易碎易腐商品增加加固。",
        RootCause.other: "建议对高频模糊诉求建立人工复核通道。",
    }
    top = stats[0]
    parts = [f"本期共 {total} 条差评。"]
    parts.append("；".join(f"{s.cause.value}占比 {s.ratio:.0%}" for s in stats))
    parts.append(f"最突出诱因为「{top.cause.value}」。{tips[top.cause]}")
    return "".join(parts) + "。"


# ---------- 品类专项下钻 ----------

def build_category_report(reviews: list[Review], category: str) -> CategoryReport:
    """单品类专项报告：体检指标 + 高频诱因 + 典型差评 + AI 诊断建议。"""
    cat_reviews = [r for r in reviews if r.category == category]
    negative = [r for r in cat_reviews if r.rating <= NEGATIVE_THRESHOLD]

    cause_counts = Counter(classify_root_cause(r.content) for r in negative)
    top_causes = [c.value for c, _ in cause_counts.most_common(3)]

    # 典型差评：评分最低优先，同分取字数较长者
    typical = sorted(negative, key=lambda r: (r.rating, -len(r.content)))[:4]

    return CategoryReport(
        category=category,
        total=len(cat_reviews),
        negative=len(negative),
        neg_rate=round(len(negative) / len(cat_reviews), 4) if cat_reviews else 0.0,
        top_causes=top_causes,
        typical_reviews=typical,
        advice=_category_advice(category, negative, top_causes),
    )


def _category_advice(category: str, negative: list[Review], top_causes: list[str]) -> str:
    if negative and LLM_AVAILABLE:
        try:
            return _category_advice_llm(category, negative, top_causes)
        except Exception:
            pass
    return _category_advice_rule(category, top_causes)


def _category_advice_llm(category: str, negative: list[Review], top_causes: list[str]) -> str:
    samples = "\n".join(f"- {r.content[:40]}" for r in negative[:5])
    prompt = (
        f"你是电商品类运营专家，针对「{category}」品类的差评痛点，输出 3 条落地整改建议，"
        "分别对应【供应链】【物流】【客服话术】三个维度，每条一句话、可直接执行，150 字以内，直接输出建议正文。\n"
        f"高频痛点：{'、'.join(top_causes) if top_causes else '综合体验'}\n代表差评：\n{samples}"
    )
    return llm.invoke(prompt).content.strip()


def _category_advice_rule(category: str, top_causes: list[str]) -> str:
    top = top_causes[0] if top_causes else "综合体验"
    return (
        f"【供应链】针对「{category}」{top}问题高发，加强来料质检与供应商准入，建立批次追溯与下架机制；"
        f"【物流】优化该品类打包与冷链/防震方案，降低运输损耗与时效投诉；"
        f"【客服话术】针对{top}类诉求建立标准安抚话术与快速赔付通道，缩短响应时长。"
    )
