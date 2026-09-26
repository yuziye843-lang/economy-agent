"""极简 AppTest：验证买家按钮点击后，处理量/挽回率/赔付指标真实联动。"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ["DEEPSEEK_API_KEY"] = ""  # 强制离线走规则引擎：确定性 + 免联网

from streamlit.testing.v1 import AppTest

_APP = os.path.join(_ROOT, "app.py")


def _btn(at, label):
    return next(b for b in at.button if b.label == label)


def _selectbox(at, label):
    return next(s for s in at.selectbox if s.label == label)


def _metric(at, label):
    return next(m for m in at.metric if m.label == label)


def _trigger(at, todo_index):
    """选一条待办差评并触发 Agent 决策，渲染出买家卡片。"""
    _selectbox(at, "从数据池选择待办").select_index(todo_index)
    at.run()
    _btn(at, "🚀 一键触发 Agent 决策链路").click()
    at.run()          # 本轮执行 Agent，st.rerun() 提前结束
    at.run()          # 再跑一轮渲染结果 + 买家卡片
    assert not at.exception, at.exception


def test_accept_compensation_closes_loop():
    at = AppTest.from_file(_APP, default_timeout=60).run()
    assert not at.exception

    # 索引 1 = 低/中度客诉 → 决策自动发券（场景 A）
    _trigger(at, 1)
    _btn(at, "✅ 买家接受方案并修改好评").click()
    at.run()          # 点击处理器更新指标，st.rerun() 结束
    at.run()          # 顶部驾驶舱用新指标重新渲染

    assert at.session_state["processed"] == 1
    assert at.session_state["recovered"] == 1
    assert at.session_state["payout"] > 0
    assert at.session_state["last_result"] is None
    assert _metric(at, "差评挽回率").value == "100%"


def test_refund_closes_loop():
    at = AppTest.from_file(_APP, default_timeout=60).run()
    assert not at.exception

    # 索引 0 = 重度升级告警 → 场景 B（极速退款）
    _trigger(at, 0)
    _btn(at, "⚡ 申请极速退款").click()
    at.run()
    at.run()

    assert at.session_state["processed"] == 1
    assert at.session_state["recovered"] == 0
    assert at.session_state["payout"] == 0.0
    assert at.session_state["last_result"] is None
    assert _metric(at, "差评挽回率").value == "0%"


if __name__ == "__main__":
    test_accept_compensation_closes_loop()
    test_refund_closes_loop()
    print("ALL PASS")
