# 电商评价运营与客诉决策 Agent 系统

## 技术栈与规范
- 语言：Python 3.10+
- 核心框架：LangGraph, Pydantic, Streamlit (UI)
- LLM API 协议：OpenAI 兼容接口 (调用 DeepSeek)
- 编码规范：遵循模块化设计，代码写在 `src/` 目录下；所有数据流转必须使用 Pydantic 类型注解。

## 目录结构
- `src/`
  - `models.py`: 定义数据模型（Schema）
  - `tools.py`: 模拟工具函数（查订单、发优惠券、告警）
  - `graph.py`: LangGraph 状态图与工作流
  - `report.py`: 差评聚合与报告生成模块
- `data/`: 存放模拟评价数据（CSV/JSON）
- `app.py`: Streamlit 交互前端
- `requirements.txt`: 依赖包

## 行为准则
1. 每次修改前向我解释你的思路。
2. 保持代码极简，优先使用 LangGraph 的标准 API，避免过度设计。
3. 遇到报错时，先定位根因再修复，严禁在不了解原因的情况下盲目重写。