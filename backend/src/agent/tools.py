import json
import os
import time
import re
import openai
from dotenv import load_dotenv
load_dotenv()
from config import CHROMA_DB_DIR
from rag.retriever import VectorStore
from rag.loader import load_document
from rag.splitter import split_by_structure
# ============================================================
# 0. 重试装饰器
# ============================================================
def retry(max_retries=3, delay=1, backoff=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        return f"⚠️ 调用失败（已重试{max_retries}次）：{str(e)}"
                    print(f"⚠️ {func.__name__} 第{attempt+1}次失败：{e}，{current_delay}s后重试...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            return None
        return wrapper
    return decorator

# ============================================================
# 1. 全局 RAG 实例
# ============================================================
_rag_store = None
_all_docs = None

@retry(max_retries=3, delay=1)
def get_rag_store():
    global _rag_store, _all_docs
    if _rag_store is None:
        try:
            _rag_store = VectorStore(persist_dir=CHROMA_DB_DIR)
            _rag_store.create_collection()
            collection = _rag_store.collection
            if collection:
                all_data = collection.get(include=["documents"])
                _all_docs = all_data.get("documents", [])
            print("✅ RAG 向量库加载成功")
        except Exception as e:
            print(f"⚠️ RAG 向量库加载失败：{e}")
    return _rag_store

# ============================================================
# 2. 工具1：RAG 知识库检索
# ============================================================
def rag_search(question: str, top_k: int = 5) -> dict:
    store = get_rag_store()
    if store is None:
        return {"answer": "⚠️ 知识库不可用", "sources": []}

    try:
        results = store.search(question, top_k=top_k)
    except Exception as e:
        return {"answer": f"⚠️ 检索异常：{str(e)}", "sources": []}

    if results:
        context = "\n\n".join([r['content'] for r in results])
        sources = [{"content": r['content'], "score": r.get('score', 0)} for r in results]
        client = openai.OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        prompt = f"""请基于以下参考内容回答用户问题。
参考内容：
{context}
问题：{question}
回答："""
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return {"answer": resp.choices[0].message.content, "sources": sources}

    client = openai.OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    prompt = f"""用户问了一个关于租房合同的问题，但知识库中没有相关文档。
请基于你的常识回答：{question}
注意：回答要简明、实用。如果拿不准，就说"建议咨询专业律师或查阅当地住建委规定"。"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return {"answer": resp.choices[0].message.content + "\n\n（注：以上回答基于通用知识，非来自你的专属知识库）", "sources": []}
# ============================================================
# 3. 工具2：单条条款风险分析
# ============================================================
def rule_check(clause: str) -> dict:
    """
    规则初筛：对可量化的硬性指标做检查
    返回：{"has_risk": bool, "risks": [...]}
    """
    risks = []
    deposit_match = re.search(r'押金.*?(\d+)\s*个月租金', clause)
    if deposit_match:
        months = int(deposit_match.group(1))
        if months > 2:
            risks.append({
                "type": "押金超收",
                "level": "高",
                "reason": f"押金为{months}个月租金，超过法定上限（一般不超过1-2个月）"
            })
    penalty_match = re.search(r'违约金.*?(\d+)%', clause)
    if penalty_match:
        pct = int(penalty_match.group(1))
        if pct > 30:
            risks.append({
                "type": "违约金过高",
                "level": "高",
                "reason": f"违约金比例{pct}%，超过实际损失30%的司法上限"
            })
    late_match = re.search(r'滞纳金.*?(\d+)%', clause)
    if late_match:
        pct = int(late_match.group(1))
        if pct > 1:
            risks.append({
                "type": "滞纳金异常",
                "level": "高",
                "reason": f"滞纳金日费率{pct}%，远超合理标准"
            })
    if "押金不退" in clause or "押金不予退还" in clause:
        risks.append({
            "type": "押金退还条件苛刻",
            "level": "高",
            "reason": "押金不退条款对承租方极为不利，可能被认定为无效"
        })
    if "最终解释权" in clause and "甲方" in clause:
        risks.append({
            "type": "单方解释权",
            "level": "高",
            "reason": "合同最终解释权归甲方，属于典型霸王条款"
        })
    return {"has_risk": len(risks) > 0, "risks": risks}
def analyze_contract_clause(clause: str) -> dict:
    rule_result = rule_check(clause)
    if rule_result["has_risk"]:
        return {
            "risk": rule_result["risks"][0]["level"],
            "explanation": rule_result["risks"][0]["reason"],
            "suggestion": f"建议修改：{rule_result['risks'][0]['type']}",
            "source": "规则引擎"
        }
    prompt = f"""你是一位专业的租房法律助手。分析该条款，输出 JSON。

条款：{clause}

输出格式（严格 JSON）：
{{"risk": "高/中/低", "explanation": "通俗解释", "suggestion": "建议"}}
"""
    client = openai.OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=300
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {"risk": "未知", "explanation": "解析失败，请重试", "suggestion": "建议咨询专业律师"}
# ============================================================
# 4. 工具3：快速预览
# ============================================================
def quick_preview(contract_text: str) -> dict:
    prompt = f"""你是租房合同审阅专家。请快速预览该合同，输出整体风险判断。

合同内容：
{contract_text[:6000]}

要求：
1. 只输出 JSON，不要有任何其他文字、不要加 markdown 代码块标记
2. risk_items 最多列 5 条最重要的风险，不要超过 5 条
3. 每条 reason 不超过 50 字

输出格式：
{{"overall_risk": "高/中/低", "risk_items": [{{"clause": "条款摘要", "risk": "高", "reason": "原因"}}], "summary": "一句话总结"}}
"""
    client = openai.OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2000
    )
    raw = resp.choices[0].message.content.strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(raw)
    except Exception as e:
        print(f"⚠️ quick_preview JSON 解析失败: {e}")
        print(f"原始输出前 500 字: {raw[:500]}")
        return {"overall_risk": "未知", "risk_items": [], "summary": f"解析失败：{str(e)[:50]}"}
# ============================================================
# 5. 工具4：深度扫描
# ============================================================
def deep_scan(contract_text: str) -> dict:
    chunks = split_by_structure(contract_text, max_chunk_size=500)
    if not chunks:
        return {"error": "合同解析失败，请检查文本格式", "risk_items": [], "summary": ""}
    results = []
    for chunk in chunks:
        if not re.search(r'第[一二三四五六七八九十百千万0-9]+条|[一二三四五六七八九十百千万]+[、.．]|\d+[、.．]', chunk):
            continue
        try:
            result = analyze_contract_clause(chunk)
            results.append({
                "clause": chunk[:200] + ("..." if len(chunk) > 200 else ""),
                "full_text": chunk,
                "risk": result.get("risk", "未知"),
                "explanation": result.get("explanation", ""),
                "suggestion": result.get("suggestion", "")
            })
        except Exception as e:
            results.append({
                "clause": chunk[:100] + "...",
                "full_text": chunk,
                "risk": "错误",
                "explanation": f"分析失败：{str(e)}",
                "suggestion": ""
            })
    risk_order = {"高": 0, "中": 1, "低": 2, "未知": 3, "错误": 4}
    results.sort(key=lambda x: risk_order.get(x.get("risk", "未知"), 5))
    high_count = sum(1 for r in results if r.get("risk") == "高")
    medium_count = sum(1 for r in results if r.get("risk") == "中")
    low_count = sum(1 for r in results if r.get("risk") == "低")
    summary = f"共审查 {len(results)} 个条款，其中高风险 {high_count} 条，中风险 {medium_count} 条，低风险 {low_count} 条。"
    if high_count > 0:
        high_clauses = [r['clause'][:30] for r in results if r.get('risk') == '高'][:3]
        summary += f" 重点关注：{', '.join(high_clauses)}..."
    return {
        "risk_items": results,
        "summary": summary,
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
        "total": len(results)
    }
# ============================================================
# 6. 工具5：文档上传
# ============================================================
def upload_document(file_path: str) -> str:
    if not os.path.exists(file_path):
        return f"❌ 文件不存在：{file_path}"

    try:
        raw_text = load_document(file_path)
        if len(raw_text.strip()) < 10:
            return f"❌ 文件内容为空或无法识别：{file_path}"
        chunks = split_by_structure(raw_text, max_chunk_size=500)
        if not chunks:
            return f"❌ 切分后无有效内容：{file_path}"
        store = get_rag_store()
        if store is None:
            return "❌ 向量库不可用"
        filename = os.path.basename(file_path)
        if hasattr(store, 'is_file_indexed') and store.is_file_indexed(filename):
            return f"ℹ️ 文档 '{filename}' 已在知识库中，无需重复上传"
        store.add_documents(chunks, source_file=filename)
        return f"✅ 文档 '{filename}' 上传成功！共 {len(chunks)} 个片段已索引。"
    except Exception as e:
        return f"❌ 上传失败：{str(e)}"
# ============================================================
# 7. 工具注册表
# ============================================================
TOOLS = {
    "analyze_contract_clause": {
        "func": analyze_contract_clause,
        "description": "分析租房合同中的单一条款，给出风险等级、通俗解释和修改建议。",
        "parameters": {"clause": "具体的合同条款原文"}
    },
    "quick_preview": {
        "func": quick_preview,
        "description": "快速预览整份合同（一次性通读），给出整体风险判断和风险清单。适合用户说'大概看一下''整体怎么样'时使用，速度较快但可能不够细致。",
        "parameters": {"contract_text": "完整的合同文本"}
    },
    "deep_scan": {
        "func": deep_scan,
        "description": "深度扫描整份合同：逐条拆解并分析每个条款，生成按风险等级排序的完整报告。适合用户说'仔细审查''全面分析''帮我扫一遍'时使用，更准确但耗时稍长。",
        "parameters": {"contract_text": "完整的合同文本"}
    },
    "rag_search": {
        "func": rag_search,
        "description": "从知识库中检索租房合同相关常识、法律规定或历史条款分析案例。用于回答通用问题。",
        "parameters": {"question": "用户问题"}
    },
    "upload_document": {
        "func": upload_document,
        "description": "上传租房合同文档到知识库，系统会自动解析并建立索引供后续检索。",
        "parameters": {"file_path": "本地文件的绝对路径"}
    }
}
def get_tool(tool_name: str):
    return TOOLS.get(tool_name, {}).get("func")
def get_tools_info() -> str:
    lines = []
    for name, info in TOOLS.items():
        lines.append(f"- {name}: {info['description']} (参数: {info['parameters']})")
    return "\n".join(lines)