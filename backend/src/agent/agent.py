from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
import json
import os
from dotenv import load_dotenv
from agent.tools import TOOLS, get_tool, get_tools_info
from logger import logger
import time
load_dotenv()
# ============================================================
# 1. 定义 State
# ============================================================
class AgentState(TypedDict):
    user_input: str
    decision: dict
    contract_text: str
    tool_result: str
    final_answer: str
    step_count: int
    history: list
# ============================================================
# 2. 决策节点
# ============================================================
def decide_node(state: AgentState) -> dict:
    user_input = state["user_input"]
    history = state.get("history", [])
    # 构建带历史的上下文
    history_text = ""
    if history:
        history_text = "\n对话历史：\n" + "\n".join([
            f"用户：{h['user']}\n助手：{h['assistant']}"
            for h in history[-3:]
        ])
    tools_description = f"""
你是一个专业的租房合同审阅助手。

可用工具列表：
{get_tools_info()}
{history_text}
规则（严格遵守）：
- 用户明确说"分析这条""看看这条" → analyze_contract_clause，args 填 clause
- 用户说"大概看一下""整体怎么样" → quick_preview，args 留空 {{}} 即可
- 用户说"仔细审查""全面分析""深度扫描""一条一条看" → deep_scan，args 留空 {{}} 即可
- 用户问常识性问题 → rag_search，args 填 question
- 用户要上传文件 → upload_document，args 填 file_path
⚠️ 重要：quick_preview 和 deep_scan 的合同文本由系统自动注入，
   你不需要填 contract_text 参数，直接传空 args: {{}} 即可。
输出格式：{{"tool": "工具名", "args": {{...}}}}
不确定时用 rag_search。
用户问题：{user_input}
只输出 JSON："""
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    logger.info(f"[决策] 用户输入: {user_input}")
    start_time = time.time()
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": tools_description}],
        temperature=0.1,
        max_tokens=200
    )
    elapsed = time.time() - start_time
    logger.info(f"[决策] LLM 响应耗时: {elapsed:.2f}s")
    try:
        raw = response.choices[0].message.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        decision = json.loads(raw)
        tool_name = decision.get("tool", "rag_search")
        if tool_name not in TOOLS:
            tool_name = "rag_search"
            decision = {"tool": tool_name, "args": {"question": user_input}}
        logger.info(f"[决策] 选定工具: {tool_name}, 参数: {decision.get('args', {})}")
        return {"decision": decision, "step_count": state.get("step_count", 0) + 1}
    except Exception as e:
        logger.error(f"[决策] LLM 决策失败：{e}，使用默认 RAG")
        return {
            "decision": {"tool": "rag_search", "args": {"question": user_input}},
            "step_count": state.get("step_count", 0) + 1
        }
# ============================================================
# 3. 执行节点
# ============================================================
def execute_node(state: AgentState) -> dict:
    decision = state["decision"]
    tool_name = decision.get("tool", "rag_search")
    args = decision.get("args", {})
    if tool_name in ["quick_preview", "deep_scan"]:
        real_contract = state.get("contract_text", "")
        if real_contract:
        # 强制覆盖，避免 LLM 塞假文本
            old = args.get("contract_text", "")
            if old and old != real_contract:
                print(f"⚠️ 覆盖 LLM 传入的假文本 (长度 {len(old)}) → 真实合同 (长度 {len(real_contract)})")
            args["contract_text"] = real_contract
            print(f"ℹ️ 已注入真实合同文本 ({len(real_contract)} 字符) 到 {tool_name}")
        else:
            print(f"⚠️ state 中没有 contract_text，{tool_name} 将收到空文本")
    tool_func = get_tool(tool_name)
    if tool_func is None:
        result = f"未知工具：{tool_name}"
    else:
        try:
            logger.info(f"[执行] 调用工具: {tool_name}, 参数: {args}")
            start_time = time.time()
            result = tool_func(**args)
            elapsed = time.time() - start_time
            if isinstance(result, dict):
                if "answer" in result:
                    result = result["answer"]
                else:
                     import json as _json
                     result = _json.dumps(result, ensure_ascii=False)
            else:
                result = str(result)
            logger.info(f"[执行] 工具 {tool_name} 执行耗时: {elapsed:.2f}s, 结果: {result[:50]}...")
        except Exception as e:
            result = f"工具执行失败：{str(e)}"
            logger.error(f"[执行] 工具 {tool_name} 执行失败：{e}")
    return {"tool_result": result}
# ============================================================
# 4. 最终回答节点
# ============================================================
def final_node(state: AgentState) -> dict:
    answer = state.get("tool_result", "未获取到回答")
    user_input = state.get("user_input", "")
    # 更新对话历史
    history = state.get("history", [])
    history.append({"user": user_input, "assistant": answer})
    logger.info(f"[完成] 最终回答: {answer[:50]}...")
    return {"final_answer": answer, "history": history}
# ============================================================
# 5. 构建 LangGraph
# ============================================================
builder = StateGraph(AgentState)
builder.add_node("decide", decide_node)
builder.add_node("execute", execute_node)
builder.add_node("final", final_node)
builder.add_edge(START, "decide")
builder.add_edge("decide", "execute")
builder.add_edge("execute", "final")
builder.add_edge("final", END)
graph = builder.compile()