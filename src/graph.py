"""LangGraph 状态图：分类 → 查订单 → 决策 → 执行 → 生成回复。"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .llm import LLM_AVAILABLE, llm
from .models import (
    Action,
    AgentState,
    Category,
    Classification,
    Complaint,
    Decision,
    Severity,
)
from .tools import calculate_compensation, query_order, send_alert, send_coupon


# ---------- 节点 ----------

def classify(state: AgentState) -> dict:
    classification = (
        _classify_llm(state.complaint) if llm else _classify_fallback(state.complaint)
    )
    return {"classification": classification}


def lookup_order(state: AgentState) -> dict:
    return {"order": query_order(state.complaint.order_id)}


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
        "安抚回复。要求：先道歉共情，再针对具体问题说明处理与补偿，语气自然口语化、不套话，120 字以内。\n"
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
    cat = state.classification.category.value
    decision = state.decision

    if cat == "物流":
        specific = "关于物流问题，我们已联系快递方加急核实，会尽快给您明确答复。"
    elif cat == "品质":
        specific = "关于商品质量问题，我们已记录并反馈给质检部门，会严格排查这一批次。"
    elif cat == "服务":
        specific = "关于服务态度问题，我们已对相关客服批评教育，后续一定改进。"
    else:
        specific = "您反馈的问题我们已详细记录，会安排专人跟进。"

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
        "你是电商客诉分类助手。请对以下客诉分类，给出类别（物流/品质/服务/其他）、"
        "严重程度（低/中/高）和一句话摘要。\n"
        f"客诉内容：{complaint.content}"
    )
    return structured.invoke(prompt)


def _classify_fallback(complaint: Complaint) -> Classification:
    text = complaint.content
    if any(k in text for k in ("物流", "快递", "配送", "延迟", "破损", "没收到", "丢件")):
        category = Category.logistics
    elif any(k in text for k in ("质量", "坏了", "瑕疵", "色差", "异味", "假货", "开裂")):
        category = Category.quality
    elif any(k in text for k in ("客服", "态度", "售后", "不理", "敷衍")):
        category = Category.service
    else:
        category = Category.other

    if any(k in text for k in ("非常", "太", "严重", "投诉", "退款", "差评", "失望")):
        severity = Severity.high
    elif any(k in text for k in ("不满意", "慢", "一般", "色差")):
        severity = Severity.medium
    else:
        severity = Severity.low

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


def run(complaint: Complaint) -> AgentState:
    """便捷入口：传入客诉，返回最终状态。"""
    result = agent.invoke(AgentState(complaint=complaint))
    return result if isinstance(result, AgentState) else AgentState(**result)
