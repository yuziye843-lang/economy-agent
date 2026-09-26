"""Streamlit 前端：客诉待办工作台 + 诱因归因大盘（商业级 UI）。"""
from __future__ import annotations

import html
import random

import altair as alt
import pandas as pd
import streamlit as st

from src.dataset import load_reviews
from src.graph import run
from src.llm import LLM_AVAILABLE
from src.models import Action, Benchmark, Complaint, Order
from src.presales import CATEGORIES, PLATFORMS, generate_listing, search_market_benchmarks
from src.report import build_category_report, build_report, classify_category
from src.tools import query_order

st.set_page_config(page_title="基于 Agent 的电商客户体验 智能决策与风控运营平台", page_icon="🛒", layout="wide")

_CSS = """
<style>
.stApp { background: #f5f7fb; }
.block-container { padding-top: 1.5rem; max-width: 1280px; }
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #eef0f4;
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(16,24,40,.06);
}
.sec-title {
    font-size: 0.82rem; font-weight: 700; color: #667085;
    text-transform: uppercase; letter-spacing: .5px; margin-bottom: 8px;
}
.tag {
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600; color: #ffffff; margin-right: 6px;
    white-space: nowrap; margin-bottom: 4px;
}
.bubble {
    background: #eef4ff; border-radius: 12px 12px 12px 2px;
    padding: 14px 16px; color: #1f2937; line-height: 1.75; font-size: 0.95rem;
}
.advice-card {
    background: linear-gradient(135deg,#fff7ed,#fef3c7);
    border: 1px solid #fde68a; border-radius: 12px;
    padding: 16px 18px; color: #92400e; line-height: 1.8; font-size: 0.95rem;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


def _tag(text: str, color: str) -> str:
    return f'<span class="tag" style="background:{color};">{text}</span>'


_SEVERITY_COLOR = {"低": "#12b76a", "中": "#f79009", "高": "#f04438"}
_CATEGORY_COLOR = {
    "商品品质": "#7a5af8", "物流配送": "#1570ef", "包装耗材": "#f04438",
    "描述不符": "#f79009", "履约服务": "#12b76a", "其他/主观偏好": "#667085",
}


@st.cache_data(show_spinner=False)
def _load_reviews():
    return load_reviews()


# ---------- 会话指标 ----------
st.session_state.setdefault("processed", 0)
st.session_state.setdefault("recovered", 0)
st.session_state.setdefault("payout", 0.0)
st.session_state.setdefault("last_result", None)


reviews = _load_reviews()
todo_pool = [r for r in reviews if r.rating <= 2]  # 差评待办池
categories = sorted({r.category for r in reviews})


def _record(result) -> None:
    st.session_state.processed += 1
    if result.decision and result.decision.action in (Action.send_coupon, Action.manual):
        st.session_state.recovered += 1
    if result.decision and result.decision.action == Action.send_coupon:
        st.session_state.payout += result.decision.coupon_amount or 0.0


# ---------- 头部指标 ----------
st.title("🛒 基于 Agent 的电商客户体验 智能决策与风控运营平台")
st.caption(
    "LangGraph 客诉决策 Agent —— "
    + ("已连接 DeepSeek" if LLM_AVAILABLE else "规则引擎降级模式")
)

m1, m2, m3 = st.columns(3)
m1.metric("已处理总量", st.session_state.processed)
m2.metric(
    "差评挽回率",
    f"{st.session_state.recovered / st.session_state.processed:.0%}"
    if st.session_state.processed else "—",
)
m3.metric("自动赔付总额", f"¥{st.session_state.payout:.0f}")
st.divider()


def _render_result(result, product_category: str) -> None:
    if result.decision and result.decision.action == Action.pending_approval:
        st.warning(
            f"⏳ 风控审批：建议补偿 ¥{result.decision.coupon_amount or 0:.0f} 已提交人工审批，暂未自动发放"
        )
    elif result.decision and result.decision.action == Action.high_risk:
        st.error("🚨 高危拦截：已触发高危工单，引导先行全额退款")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="sec-title">① 语义理解</div>', unsafe_allow_html=True)
        cause = classify_category(result.complaint.content)
        st.markdown(
            _tag(f"品类 · {product_category}", "#344054")
            + _tag(f"诱因 · {cause.value}", _CATEGORY_COLOR[cause.value])
            + _tag(f"风险 · {result.classification.severity.value}度", _SEVERITY_COLOR[result.classification.severity.value]),
            unsafe_allow_html=True,
        )
        if result.classification:
            st.caption(f"语义摘要：{result.classification.summary}")

    with c2:
        st.markdown('<div class="sec-title">② 动作执行</div>', unsafe_allow_html=True)
        if result.order:
            st.markdown(
                f"**查单**：`{result.order.order_id}` · {result.order.product} · {result.order.status}"
            )
        else:
            st.markdown("**查单**：未找到订单")
        for a in result.actions:
            st.markdown(f"✅ {a}")

    with c3:
        st.markdown('<div class="sec-title">③ 拟人化回复</div>', unsafe_allow_html=True)
        if result.reply_text:
            st.markdown(
                f'<div class="bubble">{html.escape(result.reply_text)}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("📋 一键复制回复"):
                st.code(result.reply_text, language=None)


tab_listing, tab_todo, tab_report = st.tabs([
    "✨ 爆款上架与商品企划",
    "📋 客诉待办工作台",
    "📊 诱因归因大盘",
])


# ---------- 售前运营：爆款上架与商品企划 ----------
with tab_listing:
    st.subheader("✨ AI 智能选品与爆款上架助手")
    st.caption("全品类通用：输入客观事实，一键生成「标题 / 详情页 / FAQ / 检索标签」四件套")

    with st.form("listing_form"):
        st.markdown("**第一步 · 品类与平台选择**")
        c1, c2 = st.columns(2)
        with c1:
            l_category = st.selectbox("商品品类", CATEGORIES)
        with c2:
            l_platform = st.selectbox("售卖平台", PLATFORMS)
        l_pricing = st.text_input("期望售价 / 定价方式", placeholder="如：85包邮、可小刀、拍下立减")

        st.markdown("**第二步 · 客观事实输入（极简普适）**")
        l_facts = st.text_area(
            "商品核心事实与规格",
            placeholder="填写品牌、型号、配置、版型、成色、尺码等客观事实（多行）",
        )
        l_flaws = st.text_area(
            "真实瑕疵与特殊说明",
            placeholder="真实划痕、泛黄、折痕、缺配件等，诚信告知（选填，强烈建议如实填写）",
        )

        enable_search = st.checkbox(
            "🌐 开启 Agent 全网实时对标检索（调用外部搜索工具捕获最新热词）",
            value=False,
        )

        if st.form_submit_button("🚀 生成爆款文案四件套", type="primary", use_container_width=True):
            benchmarks: list[Benchmark] = []
            if enable_search:
                with st.spinner("🔍 Agent 正在全网检索同类爆款与高频标签..."):
                    benchmarks = search_market_benchmarks(
                        l_facts.strip() or l_category, l_category, l_platform
                    )
            with st.spinner("DeepSeek 生成中……"):
                plan = generate_listing(
                    l_category,
                    l_platform,
                    l_pricing.strip() or "诚意价",
                    l_facts.strip(),
                    l_flaws.strip(),
                    benchmarks,
                )
            st.session_state.listing_result = (plan, l_category, benchmarks, enable_search)

    if st.session_state.get("listing_result"):
        plan, lcat, benchmarks, searched = st.session_state.listing_result

        if searched:
            if benchmarks:
                with st.expander("🌐 Agent 实时抓取到的全网对标参考 (Top 3)"):
                    for i, b in enumerate(benchmarks, 1):
                        st.markdown(f"**{i}. {b.title}**")
                        st.caption(b.body)
            else:
                st.info("ℹ️ 外部网络检索已平滑降级，已为您自动启用内置顶尖卖家转化模型")

        st.markdown('<div class="sec-title">① 爆款高点击标题 · 3 款</div>', unsafe_allow_html=True)
        t1, t2, t3 = st.columns(3)
        with t1:
            st.markdown("**🔎 搜索型**")
            st.code(plan.titles.search, language=None)
        with t2:
            st.markdown("**🤝 诚心转让型**")
            st.code(plan.titles.transfer, language=None)
        with t3:
            st.markdown("**✨ 氛围型**")
            st.code(plan.titles.vibe, language=None)

        st.markdown('<div class="sec-title">② 结构化吸睛详情页文案</div>', unsafe_allow_html=True)
        st.code(plan.detail_copy, language=None)

        st.markdown('<div class="sec-title">③ 售前高频拦截 FAQ</div>', unsafe_allow_html=True)
        for i, faq in enumerate(plan.faqs, 1):
            with st.container(border=True):
                st.markdown(f"**Q{i}：{faq.question}**")
                st.code(faq.answer, language=None)

        st.markdown('<div class="sec-title">④ 精选高权重检索 Tag 集合</div>', unsafe_allow_html=True)
        st.markdown("".join(_tag(t, "#7a5af8") for t in plan.tags), unsafe_allow_html=True)
        st.code(" ".join(plan.tags), language=None)


# ---------- 待办工作台 ----------
with tab_todo:
    if not todo_pool:
        st.warning("数据池暂无差评待办。")
    else:
        if "todo_idx" not in st.session_state:
            st.session_state.todo_idx = 0

        left, right = st.columns([1, 3])
        with left:
            if st.button("🎲 随机抽取差评", use_container_width=True):
                st.session_state.todo_idx = random.randrange(len(todo_pool))
        with right:
            idx = st.selectbox(
                "从数据池选择待办",
                range(len(todo_pool)),
                format_func=lambda i: f"{todo_pool[i].review_id} · {todo_pool[i].category} · {todo_pool[i].content[:24]}",
                key="todo_idx",
            )
        review = todo_pool[idx]

        with st.container(border=True):
            st.markdown(f"**{review.review_id}** · 品类 **{review.category}** · rating ⭐ {review.rating}")
            st.markdown(f"👤 {review.user_id} 　 📦 {review.order_id}")
            st.markdown(review.content)

        if st.button("🚀 一键触发 Agent 决策链路", type="primary", use_container_width=True):
            with st.spinner("Agent 处理中……"):
                result = run(
                    Complaint(user_id=review.user_id, order_id=review.order_id, content=review.content)
                )
            _record(result)
            st.session_state.last_result = (result, review.category)
            st.rerun()

    if st.session_state.last_result:
        result, cat = st.session_state.last_result
        st.divider()
        _render_result(result, cat)

    with st.expander("✍️ 自定义测试输入（针对极端客诉边界测试）"):
        with st.form("custom_form"):
            c_category = st.selectbox("商品品类（用于 Mock 订单绑定）", categories)
            c_content = st.text_area("客诉内容", "快递把包裹摔坏了，外壳裂开，客服一直不回复，要求赔偿！")
            c_order = st.text_input(
                "订单号",
                "",
                placeholder="留空或填写不存在的单号，将自动 Mock 一个有效订单（100~300 元，已完成）",
            )
            if st.form_submit_button("🚀 运行自定义客诉"):
                order = query_order(c_order) if c_order.strip() else None
                if order is None:
                    order = Order(
                        order_id=c_order.strip() or "ORD_MOCK_0001",
                        user_id="CUSTOM",
                        product=c_category,
                        status="已完成",
                        amount=round(random.uniform(100, 300), 2),
                    )
                with st.spinner("Agent 处理中……"):
                    result = run(
                        Complaint(user_id="CUSTOM", order_id=order.order_id, content=c_content),
                        order=order,
                    )
                _record(result)
                st.session_state.last_result = (result, c_category)
                st.rerun()


# ---------- 品类专项下钻大盘 ----------
with tab_report:
    # 全局宏观分布（默认折叠）
    with st.expander("🌐 全局宏观分布（Root Cause 总览）", expanded=False):
        rep = build_report(reviews)
        m1, m2 = st.columns(2)
        m1.metric("差评总数", rep.total)
        m2.metric("首要诱因", rep.stats[0].cause.value if rep.stats else "—")

        if rep.stats:
            df = pd.DataFrame({"诱因": [s.cause.value for s in rep.stats], "数量": [s.count for s in rep.stats]})
            g1, g2 = st.columns(2)
            with g1:
                st.markdown('<div class="sec-title">诱因占比 · 环形图</div>', unsafe_allow_html=True)
                donut = (
                    alt.Chart(df)
                    .mark_arc(innerRadius=55, outerRadius=90)
                    .encode(
                        theta=alt.Theta("数量:Q", stack=True),
                        color=alt.Color(
                            "诱因:N",
                            scale=alt.Scale(domain=list(_CATEGORY_COLOR), range=list(_CATEGORY_COLOR.values())),
                            legend=None,
                        ),
                        tooltip=["诱因", "数量"],
                    )
                    .properties(height=260)
                )
                st.altair_chart(donut, use_container_width=True)
            with g2:
                st.markdown('<div class="sec-title">诱因数量 · 柱状图</div>', unsafe_allow_html=True)
                bar = (
                    alt.Chart(df)
                    .mark_bar()
                    .encode(
                        x=alt.X("数量:Q"),
                        y=alt.Y("诱因:N", sort="-x"),
                        color=alt.Color(
                            "诱因:N",
                            scale=alt.Scale(domain=list(_CATEGORY_COLOR), range=list(_CATEGORY_COLOR.values())),
                            legend=None,
                        ),
                        tooltip=["诱因", "数量"],
                    )
                    .properties(height=260)
                )
                st.altair_chart(bar, use_container_width=True)

        st.markdown('<div class="sec-title">📋 全局经营决策改善建议</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="advice-card">{html.escape(rep.advice)}</div>', unsafe_allow_html=True)

    # 品类专项下钻
    st.subheader("🔍 品类专项下钻分析")
    selected = st.selectbox("选择品类", categories)
    cat = build_category_report(reviews, selected)

    # ① 品类体检卡片
    h1, h2, h3 = st.columns(3)
    h1.metric("已载入评价", cat.total)
    h2.metric("差评数", cat.negative)
    h3.metric("差评率", f"{cat.neg_rate:.0%}")
    st.markdown('<div class="sec-title">高频痛点标签</div>', unsafe_allow_html=True)
    if cat.top_causes:
        st.markdown(
            "".join(_tag(c, _CATEGORY_COLOR.get(c, "#667085")) for c in cat.top_causes),
            unsafe_allow_html=True,
        )
    else:
        st.caption("该品类暂无差评样本。")

    # ② 典型买家吐槽原声
    st.markdown('<div class="sec-title">🗣️ 典型买家吐槽原声</div>', unsafe_allow_html=True)
    if cat.typical_reviews:
        for tr in cat.typical_reviews:
            st.markdown(
                f'<div class="bubble">⭐ {tr.rating} · {html.escape(tr.content)}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("该品类暂无差评样本。")

    # ③ AI 品类专属诊断建议
    st.markdown('<div class="sec-title">💡 AI 品类专属诊断建议</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="advice-card">{html.escape(cat.advice)}</div>', unsafe_allow_html=True)
