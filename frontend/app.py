import streamlit as st
import requests
import json
import time
import os
# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="租房合同避坑助手",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ============================================================
# 自定义样式
# ============================================================
st.markdown("""
<style>
    /* 主标题 */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    /* 风险卡片 */
    .risk-card-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 5px solid #dc2626;
        padding: 1rem 1.2rem;
        border-radius: 0.6rem;
        margin: 0.6rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-card-medium {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 5px solid #f59e0b;
        padding: 1rem 1.2rem;
        border-radius: 0.6rem;
        margin: 0.6rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-card-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 5px solid #22c55e;
        padding: 1rem 1.2rem;
        border-radius: 0.6rem;
        margin: 0.6rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-card-info {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 5px solid #3b82f6;
        padding: 1rem 1.2rem;
        border-radius: 0.6rem;
        margin: 0.6rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    /* 工具徽章 */
    .tool-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-bottom: 0.5rem;
    }
    .meta-info {
        display: inline-block;
        color: #9ca3af;
        font-size: 0.75rem;
        margin-left: 0.5rem;
    }
    /* 卡片内文字 */
    .risk-card-high strong, .risk-card-medium strong, 
    .risk-card-low strong, .risk-card-info strong {
        color: #1f2937;
    }
    /* 隐藏 Streamlit 默认的页脚 */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
# ============================================================
# 会话状态初始化
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "contract_text" not in st.session_state:
    st.session_state.contract_text = ""
if "contract_name" not in st.session_state:
    st.session_state.contract_name = ""
if "api_url" not in st.session_state:
    st.session_state.api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
if "api_key" not in st.session_state:
    # 后端配置了 UPLOAD_API_KEY 时，前端在这里填同一个 key
    st.session_state.api_key = os.getenv("UPLOAD_API_KEY", "")
API_URL = st.session_state.api_url
# ============================================================
# 辅助函数：调用后端
# ============================================================
def call_health() -> bool:
    """检查后端服务状态"""
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False
def call_status() -> dict:
    """获取知识库状态（按当前 Key 所属租户统计）"""
    try:
        r = requests.get(f"{API_URL}/status", headers=_auth_headers(), timeout=3)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return {}
def build_history(messages, max_turns=3):
    history = []
    i = 0
    while i < len(messages) - 1:
        if messages[i].get("role") == "user" and messages[i+1].get("role") == "assistant":
            history.append({
                "user": messages[i].get("content", ""),
                "assistant": messages[i+1].get("content", "")
            })
            i += 2
        else:
            i += 1
    return history[-max_turns:]
def call_chat(user_input: str, contract_text: str = "", history: list = None) -> dict:
    """调用 Agent 对话接口"""
    payload = {
        "user_input": user_input,
        "contract_text": contract_text,
        "history": history if history else []
    }
    start = time.time()
    r = requests.post(f"{API_URL}/chat", json=payload, headers=_auth_headers(), timeout=180)
    elapsed = time.time() - start
    if r.status_code != 200:
        raise Exception(f"HTTP {r.status_code}: {r.text}")
    return {"data": r.json(), "elapsed": elapsed}
def call_ask(question: str, top_k: int = 3, source_file: str = "") -> dict:
    """调用纯 RAG 问答接口"""
    payload = {"question": question, "top_k": top_k}
    if source_file:
        payload["source_file"] = source_file
    start = time.time()
    r = requests.post(f"{API_URL}/ask", json=payload, headers=_auth_headers(), timeout=60)
    elapsed = time.time() - start
    if r.status_code != 200:
        raise Exception(f"HTTP {r.status_code}: {r.text}")
    return {"data": r.json(), "elapsed": elapsed}
def _auth_headers() -> dict:
    """后端配置了 UPLOAD_API_KEY 时，上传与删除需带 X-API-Key"""
    return {"X-API-Key": st.session_state.api_key} if st.session_state.api_key else {}
def call_upload(file_bytes: bytes, filename: str) -> dict:
    """上传合同文件"""
    files = {"file": (filename, file_bytes)}
    r = requests.post(f"{API_URL}/upload", files=files, headers=_auth_headers(), timeout=120)
    if r.status_code != 200:
        raise Exception(f"HTTP {r.status_code}: {r.text}")
    return r.json()
def call_documents() -> list:
    """获取知识库文档列表"""
    try:
        r = requests.get(f"{API_URL}/documents", headers=_auth_headers(), timeout=5)
        if r.status_code == 200:
            return r.json().get("documents", [])
    except:
        pass
    return []
def call_delete(filename: str) -> dict:
    """删除知识库中的某个文档"""
    r = requests.delete(f"{API_URL}/documents/{filename}", headers=_auth_headers(), timeout=30)
    if r.status_code != 200:
        raise Exception(f"HTTP {r.status_code}: {r.text}")
    return r.json()
def call_feedback(question: str, answer: str, reason: str, note: str = "", risk: str = "") -> dict:
    """提交「这条判错了」反馈"""
    payload = {
        "question": question,
        "answer": answer,
        "risk": risk,
        "reason": reason,
        "note": note,
        "session_id": "ui",
    }
    r = requests.post(f"{API_URL}/feedback", json=payload, headers=_auth_headers(), timeout=10)
    if r.status_code != 200:
        raise Exception(f"HTTP {r.status_code}: {r.text}")
    return r.json()
# ============================================================
# 辅助函数：渲染结果
# ============================================================
def parse_answer(answer):
    """解析 answer，可能是 JSON 字符串或纯文本"""
    if isinstance(answer, dict):
        return answer
    if isinstance(answer, str) and answer.strip().startswith("{"):
        try:
            return json.loads(answer)
        except:
            # 尝试替换单引号为双引号（Python dict 字符串）
            try:
                return eval(answer)
            except:
                pass
    return None
def render_risk_card(risk: str, clause: str, explanation: str = "", suggestion: str = "", source: str = "", clause_no: str = ""):
    """渲染单条风险卡片"""
    risk_class = {
        "高": "risk-card-high",
        "中": "risk-card-medium",
        "低": "risk-card-low",
    }.get(risk, "risk-card-info")
    icon = {"高": "🚨", "中": "⚠️", "低": "✅"}.get(risk, "ℹ️")
    no_badge = f'<span class="meta-info">· {clause_no}</span>' if clause_no else ""
    source_badge = f'<span class="meta-info">· {source}</span>' if source else ""
    html = f"""
    <div class="{risk_class}">
        <strong>{icon} 【{risk}风险】</strong>{no_badge}{source_badge}<br/>
        <div style="margin-top:0.5rem; color:#374151; font-size:0.9rem;">
            <strong>条款：</strong>{clause}
        </div>
        {f'<div style="margin-top:0.5rem; color:#4b5563; font-size:0.9rem;"><strong>解释：</strong>{explanation}</div>' if explanation else ''}
        {f'<div style="margin-top:0.5rem; color:#059669; font-size:0.9rem;"><strong>建议：</strong>{suggestion}</div>' if suggestion else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
def render_answer(answer_raw, tool_used: str = ""):
    """智能渲染 Agent 回答"""
    parsed = parse_answer(answer_raw)
    # 情况1：单条条款分析结果 {"risk": ..., "explanation": ..., "suggestion": ...}
    if parsed and "risk" in parsed and "explanation" in parsed:
        render_risk_card(
            risk=parsed.get("risk", "未知"),
            clause=parsed.get("clause", ""),
            explanation=parsed.get("explanation", ""),
            suggestion=parsed.get("suggestion", ""),
            source=parsed.get("source", ""),
            clause_no=parsed.get("clause_no", "")
        )
        return
    # 情况2：快速预览结果 {"overall_risk": ..., "risk_items": [...], "summary": ...}
    if parsed and "overall_risk" in parsed:
        overall = parsed.get("overall_risk", "未知")
        st.markdown(f"### 📊 整体风险等级：**{overall}**")
        st.info(parsed.get("summary", ""))
        items = parsed.get("risk_items", [])
        if items:
            st.markdown(f"#### 风险清单（共 {len(items)} 条）")
            for item in items:
                render_risk_card(
                    risk=item.get("risk", "未知"),
                    clause=item.get("clause", ""),
                    explanation=item.get("reason", ""),
                    clause_no=item.get("clause_no", "")
                )
        return
    # 情况3：深度扫描结果 {"risk_items": [...], "summary": ..., "high_count": ...}
    if parsed and "risk_items" in parsed and "summary" in parsed:
        high = parsed.get("high_count", 0)
        medium = parsed.get("medium_count", 0)
        low = parsed.get("low_count", 0)
        total = parsed.get("total", 0)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总条款", total)
        col2.metric("🔴 高风险", high)
        col3.metric("🟡 中风险", medium)
        col4.metric("🟢 低风险", low)
        st.markdown(f"**📝 摘要**：{parsed.get('summary', '')}")
        st.divider()
        for item in parsed.get("risk_items", []):
            render_risk_card(
                risk=item.get("risk", "未知"),
                clause=item.get("clause", ""),
                explanation=item.get("explanation", ""),
                suggestion=item.get("suggestion", ""),
                source=item.get("source", ""),
                clause_no=item.get("clause_no", "")
            )
        return
    # 情况4：RAG 回答（含 sources）
    if parsed and "answer" in parsed:
        st.markdown(parsed["answer"])
        sources = parsed.get("sources", [])
        if sources:
            with st.expander(f"📎 参考来源（{len(sources)} 条）"):
                for i, s in enumerate(sources, 1):
                    score = s.get("score", 0)
                    st.markdown(f"**来源 {i}** （相关度：`{score:.3f}`）")
                    st.text(s.get("content", "")[:500])
                    st.divider()
        return
    # 情况5：错误
    if parsed and "error" in parsed:
        st.error(parsed["error"])
        return
    # 兜底：纯文本
    st.markdown(str(answer_raw))
FEEDBACK_REASONS = ["风险等级偏高", "风险等级偏低", "不是风险（误报）", "漏掉了风险", "解释/建议不准确", "其他"]
def render_feedback_ui(msg_idx: int, msg: dict):
    """助手回答下方的「这条判错了」入口：选择判错类型 + 补充说明后提交"""
    if msg.get("feedback_sent"):
        st.caption(f"✅ 已提交反馈（编号 {msg.get('feedback_id', '')}）")
        return
    question = msg.get("question", "")
    with st.expander("🙅 这条判错了？"):
        reason = st.selectbox("判错类型", FEEDBACK_REASONS, key=f"fb_reason_{msg_idx}")
        risk = st.text_input("涉及的风险等级（可留空：高/中/低）", key=f"fb_risk_{msg_idx}")
        note = st.text_area("补充说明（写下你认为的正确判断）", key=f"fb_note_{msg_idx}", height=70)
        if st.button("提交反馈", key=f"fb_submit_{msg_idx}", use_container_width=True):
            if not question:
                st.warning("找不到对应的问题，暂不能提交")
            else:
                try:
                    result = call_feedback(question, msg.get("content", ""), reason, note, risk)
                    msg["feedback_sent"] = True
                    msg["feedback_id"] = result.get("id", "")
                    st.success(f"已记录，编号 {result.get('id', '')}")
                except Exception as e:
                    st.error(f"提交失败：{e}")
# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("### 🏠 租房合同避坑助手")
    # 服务状态
    st.markdown("#### ⚙️ 服务状态")
    if call_health():
        status = call_status()
        chunks = status.get("chunks_loaded", 0)
        st.success(f"✅ 服务在线 · 知识库 {chunks} 块")
    else:
        st.error("❌ 服务未连接")
        st.caption("请先启动后端：`python main.py`")
    st.divider()
    # 接口鉴权（可选）
    with st.expander("🔑 接口鉴权（可选）"):
        st.session_state.api_key = st.text_input(
            "X-API-Key",
            value=st.session_state.api_key,
            type="password",
            help="后端配置 API_KEYS 或 UPLOAD_API_KEY 时，填入对应 Key；不同 Key 对应不同租户"
        )
    # 上传合同
    st.markdown("#### 📄 上传合同")
    uploaded = st.file_uploader(
        "支持 .txt / .pdf",
        type=["txt", "pdf"],
        label_visibility="collapsed"
    )
    if uploaded is not None:
        try:
            file_bytes = uploaded.read()
            # TXT 直接读取，PDF 交给后端
            if uploaded.name.lower().endswith(".txt"):
                st.session_state.contract_text = file_bytes.decode("utf-8")
                st.session_state.contract_name = uploaded.name
                st.success(f"✅ 已加载：{uploaded.name}")
                with st.expander("📖 预览合同"):
                    preview = st.session_state.contract_text[:800]
                    st.text(preview + ("..." if len(st.session_state.contract_text) > 800 else ""))
            else:
                # PDF 上传到后端：解析 + 索引，并把正文回填到会话，供深度扫描使用
                with st.spinner("上传并解析中..."):
                    result = call_upload(file_bytes, uploaded.name)
                    st.success(f"✅ {result.get('message', '上传成功')}")
                    st.session_state.contract_text = result.get("text", "")
                    st.session_state.contract_name = uploaded.name
                    st.caption(
                        f"已解析 {result.get('chars', 0)} 字 · 知识库现有 {result.get('chunks', 0)} 块"
                    )
                if st.session_state.contract_text:
                    with st.expander("📖 预览合同"):
                        preview = st.session_state.contract_text[:800]
                        st.text(preview + ("..." if len(st.session_state.contract_text) > 800 else ""))
        except Exception as e:
            st.error(f"上传失败：{e}")
    if st.session_state.contract_text:
        if st.button("🗑️ 清除已加载合同", use_container_width=True):
            st.session_state.contract_text = ""
            st.session_state.contract_name = ""
            st.rerun()
    st.divider()
    # 知识库文档管理
    st.markdown("#### 📚 知识库文档")
    docs = call_documents()
    if not docs:
        st.caption("暂无已入库文档")
    else:
        st.caption(f"共 {len(docs)} 个文件 · {sum(d['chunks'] for d in docs)} 块")
        target = st.selectbox(
            "选择文档",
            [d["filename"] for d in docs],
            label_visibility="collapsed"
        )
        info = next(d for d in docs if d["filename"] == target)
        st.caption(f"{info['chunks']} 块 · {info['size_bytes'] / 1024:.0f} KB")
        confirm = st.checkbox("确认删除（不可恢复）")
        if st.button("🗑️ 删除文档", use_container_width=True, disabled=not confirm):
            try:
                result = call_delete(target)
                if st.session_state.contract_name == target:
                    st.session_state.contract_text = ""
                    st.session_state.contract_name = ""
                st.success(f"已删除 {target}，移除 {result.get('removed_chunks', 0)} 块")
                st.rerun()
            except Exception as e:
                st.error(f"删除失败：{e}")
        st.caption("删除内置法规/示范文本会影响评测基线，慎删")
    st.divider()
    # 可用工具说明
    st.markdown("#### 🛠️ 可用能力")
    st.markdown("""
    - 📝 **单条分析** — 分析具体条款  
    - 👁️ **快速预览** — 通读整份合同  
    - 🔍 **深度扫描** — 逐条精细审查  
    - 💬 **知识问答** — 租房常识检索  
    """)
    st.divider()
    # 快速开始
    st.markdown("#### 💡 快速开始")
    quick_questions = [
        "帮我分析：租户提前退租，押金不退，还要付剩余租金20%的违约金",
        "帮我大概看一下这份合同",
        "帮我仔细审查这份合同",
        "租房押金一般收多少？",
    ]
    for q in quick_questions:
        label = q[:18] + "..." if len(q) > 18 else q
        if st.button(label, key=f"quick_{q[:10]}", use_container_width=True, help=q):
            st.session_state.messages.append({"role": "user", "content": q})
            st.session_state["_pending"] = q
            st.rerun()
# ============================================================
# 主区域
# ============================================================
st.markdown('<p class="main-title">🏠 租房合同避坑助手</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI Agent 自动识别合同风险 · 逐条分析 · 给出修改建议</p>', unsafe_allow_html=True)
# 显示历史消息
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            if msg.get("tool_used"):
                st.markdown(f'<span class="tool-badge">⚡ {msg["tool_used"]}</span>'
                            f'<span class="meta-info">{msg.get("elapsed", 0):.1f}s</span>',
                            unsafe_allow_html=True)
            render_answer(msg["content"], msg.get("tool_used", ""))
            render_feedback_ui(msg_idx, msg)
        else:
            st.markdown(msg["content"])
# 处理"快速开始"点击
pending = st.session_state.pop("_pending", None)
# 输入框始终显示
user_input = st.chat_input("输入您的问题，例如：帮我分析这条条款...")
# 优先处理快速开始，其次处理用户输入
if pending:
    prompt = pending
elif user_input:
    prompt = user_input
else:
    prompt = None
if prompt:
    # 如果是新输入（非快速开始触发），也要记录
    if not pending:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("🤔 Agent 正在分析..."):
            try:
                history = build_history(st.session_state.messages[:-1])
                result = call_chat(prompt, st.session_state.contract_text, history)
                data = result["data"]
                elapsed = result["elapsed"]
                tool_used = data.get("tool_used", "未知")
                answer = data.get("answer", "")
                st.markdown(f'<span class="tool-badge">⚡ {tool_used}</span>'
                            f'<span class="meta-info">{elapsed:.1f}s</span>',
                            unsafe_allow_html=True)
                render_answer(answer, tool_used)
                assistant_msg = {
                    "role": "assistant",
                    "content": answer,
                    "question": prompt,
                    "tool_used": tool_used,
                    "elapsed": elapsed
                }
                st.session_state.messages.append(assistant_msg)
                render_feedback_ui(len(st.session_state.messages) - 1, assistant_msg)
            except requests.exceptions.ConnectionError:
                st.error("❌ 无法连接到后端服务")
                st.caption("请确认已在 `backend/src` 目录下运行 `python main.py`")
            except requests.exceptions.Timeout:
                st.error("⏰ 请求超时，深度扫描较慢，请耐心等待")
            except Exception as e:
                st.error(f"❌ 出错：{e}")
# ============================================================
# 底部提示
# ============================================================
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("💡 **提示**：在侧边栏上传合同后，直接说\"帮我仔细审查这份合同\"")
with col2:
    st.caption("🔍 **深度扫描**约 20 秒，会逐条分析所有条款")
with col3:
    st.caption("📡 **API 文档**：http://127.0.0.1:8000/docs")