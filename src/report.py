"""差评聚合与客诉分类归因报告。"""
from __future__ import annotations

from collections import Counter

from .dataset import load_reviews
from .llm import LLM_AVAILABLE, llm
from .models import Category, CategoryReport, CategoryStat, NegativeReport, Review

# 明确对外契约：app.py / graph.py 依赖的公开接口（排查“无法导入”时的白名单）。
__all__ = [
    "classify_category",
    "build_report",
    "build_category_report",
    "NEGATIVE_THRESHOLD",
]

NEGATIVE_THRESHOLD = 2  # rating <= 2 视为差评（本数据集中差评=1）

# 6 大分类维度关键词库（与 models.Category 一一对应）。
# 匹配优先级：包装耗材 → 描述不符 → 物流配送 → 履约服务 → 商品品质 → 其他（兜底）。
_PACKAGING = (
    "包装", "外包装", "盒子", "箱子", "纸箱", "泡沫", "防震", "缓冲",
    "漏液", "漏了", "挤压", "压烂", "压坏", "压瘪", "瘪", "变形", "打包",
)
_MISMATCH = (
    "色差", "尺码", "尺寸", "偏大", "偏小", "版型", "货不对板", "不如图",
    "描述不符", "虚假宣传", "差别", "出入", "欺骗", "不实", "不符",
    "短斤", "缺斤少两", "分量不足", "实物", "不一样",
)
_LOGISTICS = (
    "物流", "快递", "配送", "发货", "延迟", "丢件", "没收到", "没到货",
    "不到货", "时效", "揽收", "送达", "送货", "派送", "催", "不发货",
    "没发货", "迟迟", "到货", "太慢", "很慢", "还没到", "还没收到",
    "没有收到", "没有到", "没到", "慢",
)
_SERVICE = (
    "客服", "态度", "售后", "不理", "敷衍", "退款", "退货", "换货",
    "回复", "投诉", "处理", "赔偿", "赔付", "不处理", "联系", "发票",
    "沟通", "无人", "没人", "响应", "服务", "维修", "上门", "安装",
    "推诿", "扯皮", "漏发", "少发", "缺货", "发错", "没送", "赠品",
    "差价", "降价", "价格保护", "价保", "秒杀", "优惠", "价格", "退房",
    "贵", "返修",
)
_QUALITY = (
    "质量", "破损", "摔", "碎了", "开裂", "裂开", "坏了", "瑕疵", "异味",
    "变质", "做工", "粗糙", "功能", "故障", "失灵", "不能用", "难吃",
    "口感", "掉毛", "褪色", "起球", "缩水", "太差", "很差", "垃圾",
    "劣质", "缺陷", "卡顿", "分辨率", "内存", "电池", "信号", "关机",
    "假货", "假", "蒙牛", "伊利", "塑化剂", "中毒", "过期", "三聚氰胺",
    "有毒", "食品安全", "难用", "不好用", "太薄", "太厚", "太硬",
    "太软", "太酸", "不甜", "卫生", "灰尘", "霉味", "脏", "旧", "骗子",
    "缺页", "白纸", "隔音", "吵", "头皮屑", "拉链", "水分", "不能吃",
    "涩", "酸", "净重", "缺斤", "差劲", "失望", "不满意", "太大", "太小",
    # 书 / 内容
    "书", "文笔", "情节", "作者", "内容", "看不懂", "看不下去", "看不下",
    "无聊", "没意义", "没意思", "不值得", "不实用", "没用", "油墨", "印刷",
    "纸质", "翻译", "错字", "错别字",
    # 食品
    "发苦", "发霉", "烂", "没熟", "不新鲜", "变味", "馊", "臭", "恶心",
    "腥", "软绵绵", "发软", "太腻", "太油", "发酸",
    # 通用缺陷
    "死机", "卡", "发热", "发烫", "漏", "破", "二手", "翻新", "拆过",
    "松动", "掉色", "掉漆", "脱线", "开线", "刺鼻", "难闻", "不干净", "硬",
    "蓝屏", "重启", "长毛", "正品", "掉发", "头屑", "痒", "不清晰", "不稳定",
    "不值", "性价比", "不方便",
)


def classify_category(content: str) -> Category:
    """基于关键词规则识别差评的分类维度（全量快速归因）。"""
    text = content
    if any(k in text for k in _PACKAGING):
        return Category.packaging
    if any(k in text for k in _MISMATCH):
        return Category.mismatch
    if any(k in text for k in _LOGISTICS):
        return Category.logistics
    if any(k in text for k in _SERVICE):
        return Category.service
    if any(k in text for k in _QUALITY):
        return Category.quality
    return Category.other


def build_report(reviews: list[Review] | None = None) -> NegativeReport:
    reviews = reviews if reviews is not None else load_reviews()
    negative = [r for r in reviews if r.rating <= NEGATIVE_THRESHOLD]

    counts: dict[Category, int] = {}
    for r in negative:
        cause = classify_category(r.content)
        counts[cause] = counts.get(cause, 0) + 1

    total = len(negative)
    stats = [
        CategoryStat(cause=c, count=n, ratio=round(n / total, 4) if total else 0.0)
        for c, n in sorted(counts.items(), key=lambda x: -x[1])
    ]
    top_reviews = sorted(negative, key=lambda r: r.rating)[:5]
    advice = _generate_advice(stats, negative, total)

    return NegativeReport(total=total, stats=stats, top_reviews=top_reviews, advice=advice)


def _generate_advice(stats: list[CategoryStat], negative: list[Review], total: int) -> str:
    if not negative:
        return "本期无差评，运营状态良好。"
    if LLM_AVAILABLE:
        try:
            return _generate_advice_llm(stats, negative, total)
        except Exception:
            pass
    return _generate_advice_rule(stats, negative, total)


def _generate_advice_llm(stats: list[CategoryStat], negative: list[Review], total: int) -> str:
    distribution = "；".join(f"{s.cause.value} {s.count} 条（{s.ratio:.0%}）" for s in stats)
    samples = "\n".join(f"- {r.content[:50]}" for r in negative[:10])
    prompt = (
        "你是电商运营负责人。根据以下差评的分类维度分布与代表样本，输出《经营决策改善建议》："
        "针对运营主管的 3~5 条可执行行动指南，结合具体维度提出可落地的改进措施，200 字以内，直接输出建议正文。\n"
        f"差评总数：{total}\n维度分布：{distribution}\n代表差评：\n{samples}"
    )
    return llm.invoke(prompt).content.strip()


def _generate_advice_rule(stats: list[CategoryStat], negative: list[Review], total: int) -> str:
    tips = {
        Category.logistics: "建议核查合作物流时效与破损率，对高延迟线路考虑切换承运商或增加时效承诺。",
        Category.quality: "建议加强来料质检与供应商抽检，对问题批次启动下架或召回。",
        Category.packaging: "建议升级打包耗材（缓冲材/防震/冷链保温），对易碎易腐商品增加加固。",
        Category.mismatch: "建议复核商品详情页与实物一致性，规范尺码/色差描述，杜绝虚假宣传。",
        Category.service: "建议补充客服排班与话术培训，建立价保与漏发快速赔付通道，缩短退换货周期。",
        Category.other: "建议对纯主观/情绪类评价单独归类，避免干扰客观归因。",
    }
    top = stats[0]
    parts = [f"本期共 {total} 条差评。"]
    parts.append("；".join(f"{s.cause.value}占比 {s.ratio:.0%}" for s in stats))
    parts.append(f"最突出诱因为「{top.cause.value}」。{tips[top.cause]}")
    return "".join(parts) + "。"


# ---------- 品类专项下钻 ----------

def build_category_report(reviews: list[Review], category: str) -> CategoryReport:
    """单品类专项报告：体检指标 + 高频维度 + 典型差评 + AI 诊断建议。"""
    cat_reviews = [r for r in reviews if r.category == category]
    negative = [r for r in cat_reviews if r.rating <= NEGATIVE_THRESHOLD]

    cause_counts = Counter(classify_category(r.content) for r in negative)
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
