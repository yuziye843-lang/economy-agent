"""LangGraph 状态图：分类 → 查订单 → 决策 → 执行 → 生成回复。"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from .llm import LLM_AVAILABLE, llm
from .models import (
    Action,
    AgentState,
    Category,
    Classification,
    Complaint,
    Decision,
    Order,
    Severity,
)
from .report import classify_category
from .tools import calculate_compensation, query_order, send_alert, send_coupon

logger = logging.getLogger("complaint_agent")


# ---------- 节点 ----------

def classify(state: AgentState) -> dict:
    if llm:
        logger.info("分类节点：使用 DeepSeek LLM 结构化分类")
        try:
            classification = _classify_llm(state.complaint)
        except Exception as exc:
            logger.warning("LLM 分类失败，降级为规则关键词分类：%s", exc)
            classification = _classify_fallback(state.complaint)
    else:
        logger.warning("未配置 DEEPSEEK_API_KEY，分类节点降级为规则关键词")
        classification = _classify_fallback(state.complaint)
    return {"classification": classification}


def lookup_order(state: AgentState) -> dict:
    # 若外部已预置订单（如测试 Mock），则直接沿用，不再查库
    return {"order": state.order or query_order(state.complaint.order_id)}


def decide(state: AgentState) -> dict:
    """规则化决策：动态定价 + 风控熔断（LLM 不参与定价，避免资损）。"""
    return {"decision": _decide(state)}


def _decide(state: AgentState) -> Decision:
    order_amount = state.order.amount if state.order else 0.0
    comp = calculate_compensation(
        order_amount,
        state.classification.severity,
        state.classification.category,
        state.complaint.content,
    )
    if comp.status == "auto":
        return Decision(action=Action.send_coupon, coupon_amount=comp.amount, reason=comp.reason)
    if comp.status == "pending":
        return Decision(action=Action.pending_approval, coupon_amount=comp.amount, reason=comp.reason)
    if comp.status == "high_risk":
        return Decision(action=Action.high_risk, reason=comp.reason)
    return Decision(action=Action.escalate_alert, reason=comp.reason)


def act(state: AgentState) -> dict:
    actions: list[str] = []
    decision = state.decision
    if decision is None:
        # 订单不存在等路径跳过了决策节点，兜底告警
        decision = Decision(action=Action.escalate_alert, reason="流程异常，升级告警")

    if decision.action == Action.send_coupon:
        result = send_coupon(state.complaint.user_id, decision.coupon_amount or 0.0)
        actions.append(result["msg"])
    elif decision.action == Action.pending_approval:
        actions.append(f"⏳ 建议补偿 {decision.coupon_amount or 0.0} 元已提交人工审批，暂不自动发放")
    elif decision.action == Action.high_risk:
        actions.append(f"🚨 高危拦截工单已触发：{decision.reason}")
    elif decision.action == Action.escalate_alert:
        result = send_alert(state.complaint, decision.reason)
        actions.append(result["msg"])
    elif decision.action == Action.manual:
        actions.append(f"已转人工处理：{decision.reason}")
    else:
        actions.append(f"无需处理：{decision.reason}")

    final_response = (
        f"【分类】{state.classification.category.value} · {state.classification.severity.value}度\n"
        f"【订单】{state.order.status if state.order else '未找到'}\n"
        f"【决策】{decision.action.value} —— {decision.reason}\n"
        f"【执行】" + "；".join(actions)
    )
    return {"actions": actions, "final_response": final_response, "decision": decision}


# ---------- 回复生成 ----------

def generate_reply(state: AgentState) -> dict:
    reply = _generate_reply_llm(state) if llm else _generate_reply_fallback(state)
    return {"reply_text": reply}


def _action_summary(state: AgentState) -> str:
    order_part = f"已查到订单（状态：{state.order.status}）" if state.order else "未查到订单"
    decision = state.decision
    if decision is None:
        return order_part + "；无处置动作"
    if decision.action == Action.send_coupon:
        action_part = f"已发放 {decision.coupon_amount or 0.0} 元优惠券"
    elif decision.action == Action.pending_approval:
        action_part = f"建议补偿 {decision.coupon_amount or 0.0} 元（待人工审批，暂未发放）"
    elif decision.action == Action.high_risk:
        action_part = "已触发高危拦截工单，引导先行全额退款"
    elif decision.action == Action.escalate_alert:
        action_part = "已升级告警"
    elif decision.action == Action.manual:
        action_part = "已转人工处理"
    else:
        action_part = "无需处理"
    return f"{order_part}；处置：{action_part}"


def _generate_reply_llm(state: AgentState) -> str:
    order_info = f"已查到（状态：{state.order.status}）" if state.order else "未查到订单"
    prompt = (
        "你是电商平台的资深客服，请用第一人称（我）给这位差评用户写一段真诚、有共情力、针对性强的"
        "安抚回复。要求：先点出用户原话里的 1~2 个具体细节并致歉共情，再针对具体问题说明处理与补偿，"
        "语气自然口语化、不说套话，100 字左右。\n"
        f"问题类别：{state.classification.category.value}\n"
        f"用户原话：{state.complaint.content}\n"
        f"订单情况：{order_info}\n"
        f"已采取动作：{_action_summary(state)}\n"
        "【重要】回复必须严格与「已采取动作」一致：\n"
        "1. 若已发券，必须准确说出实际金额，例如「已为您申请了 X 元专属心意补偿券」，金额不可虚报；\n"
        "2. 若为待审批/高危拦截/升级人工，则话术侧重「专员正在加急核实，会在 1 小时内电话联系您」，不要承诺任何金额或退款到账；\n"
        "3. 若无需处理，不要承诺优惠券、退款或赔偿。\n"
        "直接输出回复正文，不要任何前缀或解释。"
    )
    return llm.invoke(prompt).content.strip()


def _generate_reply_fallback(state: AgentState) -> str:
    cat = state.classification.category
    decision = state.decision

    if cat == Category.logistics:
        specific = "关于物流配送问题，我们已联系承运方加急核实时效与派送情况，会尽快给您明确答复。"
    elif cat == Category.quality:
        specific = "关于商品品质问题，我们已记录并反馈给质检部门，会严格排查这一批次。"
    elif cat == Category.packaging:
        specific = "关于包装破损问题，我们已反馈给仓储打包环节，会加强耗材与防震防护。"
    elif cat == Category.mismatch:
        specific = "关于描述不符问题，我们已核实页面信息并反馈给商品团队，会修正展示避免误导。"
    elif cat == Category.service:
        specific = "关于履约服务问题，我们已对相关环节批评整改，后续一定改进。"
    else:
        specific = "您反馈的感受我们已详细记录，会安排专人跟进。"

    if decision is None:
        action_tail = ""
    elif decision.action == Action.send_coupon:
        action_tail = f"已为您申请了 {decision.coupon_amount or 0.0} 元专属心意补偿券，直接到账，聊表歉意。"
    elif decision.action == Action.pending_approval:
        action_tail = "我们已将您的诉求提交人工审批，专员正在加急核实，会在 1 小时内电话联系您。"
    elif decision.action == Action.high_risk:
        action_tail = "您反馈的问题非常严重，我们已启动高危处理流程，专员正在加急核实，会在 1 小时内电话联系您并协助办理全额退款。"
    elif decision.action == Action.escalate_alert:
        if state.order is None:
            action_tail = "由于暂时没能核实到您的订单，专员正在加急核实，会在 1 小时内电话联系您。"
        else:
            action_tail = "我们已将问题升级给高级专员，专员正在加急核实，会在 1 小时内电话联系您。"
    elif decision.action == Action.manual:
        action_tail = "我们已安排人工客服为您一对一处理，专员会在 1 小时内电话联系您。"
    else:
        action_tail = "感谢您的理解与包容。"

    return (
        f"非常抱歉给您带来不好的体验，看到您的反馈我心里也很不是滋味。{specific}"
        f"{action_tail}感谢您一直以来的支持，我们会努力做得更好。"
    )


# ---------- 分类实现 ----------

def _classify_llm(complaint: Complaint) -> Classification:
    structured = llm.with_structured_output(Classification, method="function_calling")
    prompt = (
        "你是电商客诉分类助手，请对客诉进行「类别 / 严重程度 / 一句话摘要」结构化标注。\n"
        "类别从以下 6 个维度中选最贴近的一个：\n"
        "1. 商品品质：破损、质量瑕疵、异味、变质、做工粗糙、功能故障；\n"
        "2. 物流配送：时效慢、丢件、派送态度恶劣、未送货上门；\n"
        "3. 包装耗材：外包装挤压变形、破损漏液、缺少防震泡沫；\n"
        "4. 描述不符：严重色差、尺码不准、版型不合、货不对板、虚假宣传；\n"
        "5. 履约服务：客服不理人/推诿、漏发少发、降价不退差价、退换货受阻；\n"
        "6. 其他/主观偏好：仅限纯主观审美（如「我不喜欢这个颜色」）或纯情绪发泄（如「差劲」且无具体原因）。\n"
        "判断请按思维链：先提取用户的具体抱怨对象（物品本身 / 运输过程 / 包装 / 尺码色差 / 客服或价格），再匹配最贴近的维度。\n"
        "【硬性约束】只要原文提及任何具体实体（物品状态、运输、尺码、包装、色差、价格、客服等），一律禁止归入「其他/主观偏好」，必须就近归入前 5 类；「其他/主观偏好」仅用于无任何实体、无具体原因的纯情绪或纯审美表达。\n"
        "【示例】「包装烂了」→ 包装耗材；「穿上太紧了」→ 描述不符；「物流太慢，三天还没到」→ 物流配送。\n"
        "严重程度：低=轻微不满，中=明确不满但可补救，高=涉及安全/假货/强烈投诉。\n"
        f"客诉内容：{complaint.content}"
    )
    return structured.invoke(prompt)


def _classify_fallback(complaint: Complaint) -> Classification:
    text = complaint.content
    category = classify_category(text)

    if any(k in text for k in ("非常", "太", "严重", "投诉", "退款", "差评", "失望", "假", "中毒", "过期", "变质", "欺骗", "举报", "12315", "消协")):
        severity = Severity.high
    elif any(k in text for k in ("不满意", "慢", "一般", "色差", "瑕疵", "贵", "降价", "差价", "偏")):
        severity = Severity.medium
    else:
        severity = Severity.low

    logger.warning("规则降级分类：%s -> %s/%s", text[:20], category.value, severity.value)
    return Classification(category=category, severity=severity, summary=text[:50])


# ---------- 路由与图 ----------

def route_after_lookup(state: AgentState) -> str:
    """订单不存在时跳过决策，直接进入执行节点兜底告警。"""
    return "act" if state.order is None else "decide"


_builder = StateGraph(AgentState)
_builder.add_node("classify", classify)
_builder.add_node("lookup_order", lookup_order)
_builder.add_node("decide", decide)
_builder.add_node("act", act)
_builder.add_node("generate_reply", generate_reply)

_builder.add_edge(START, "classify")
_builder.add_edge("classify", "lookup_order")
_builder.add_conditional_edges("lookup_order", route_after_lookup, {"decide": "decide", "act": "act"})
_builder.add_edge("decide", "act")
_builder.add_edge("act", "generate_reply")
_builder.add_edge("generate_reply", END)

agent = _builder.compile()


def run(complaint: Complaint, order: Order | None = None) -> AgentState:
    """便捷入口：传入客诉（可附带预置订单用于测试/Mock），返回最终状态。"""
    result = agent.invoke(AgentState(complaint=complaint, order=order))
    return result if isinstance(result, AgentState) else AgentState(**result)
