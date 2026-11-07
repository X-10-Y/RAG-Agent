import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import hashlib
import threading
import shutil
from datetime import datetime
from fastapi import FastAPI, HTTPException, File, UploadFile, Header, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Optional, List, Union
from dotenv import load_dotenv
load_dotenv()
# ==================== 导入 RAG 和 Agent ====================
from config import DOCS_DIR, PROJECT_ROOT, get_llm_client
from logger import logger
from rag.rag_chain import RAGChain
from rag.retriever import DEFAULT_TENANT
from agent.agent import graph
from agent.tools import invalidate_rag_store, set_current_tenant, reset_current_tenant
# ==================== FastAPI 初始化 ====================
app = FastAPI(title="租房合同避坑助手（RAG + Agent）", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ==================== 全局 RAG 实例 ====================
rag = RAGChain()
# ==================== 身份 / 租户 / 审计 ====================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
AUDIT_LOG = os.path.join(DATA_DIR, "audit.jsonl")
FEEDBACK_DIR = os.path.join(DATA_DIR, "feedback")
FEEDBACK_LOG = os.path.join(FEEDBACK_DIR, "feedback.jsonl")
os.makedirs(FEEDBACK_DIR, exist_ok=True)
# 角色是等级制：高等级包含低等级权限
ROLE_LEVELS = {"view": 1, "upload": 2, "delete": 3, "admin": 4}
def _parse_api_keys(raw: str) -> dict:
    """解析 API_KEYS：{"key":{"tenant":"t1","role":"admin"}}，格式错误则视为未配置"""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"[AUTH] API_KEYS 不是合法 JSON，已忽略: {e}")
        return {}
    if not isinstance(data, dict):
        logger.error("[AUTH] API_KEYS 顶层必须是对象")
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}
_API_KEYS = _parse_api_keys(os.getenv("API_KEYS", ""))
_audit_lock = threading.Lock()
def write_audit(identity: dict, action: str, filename: Optional[str] = None,
                result: str = "success", detail: str = ""):
    """追加一条审计记录到 data/audit.jsonl；只记 Key 的哈希前缀，不落原始 Key"""
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "tenant": identity.get("tenant", DEFAULT_TENANT),
        "user": identity.get("user", "anonymous"),
        "action": action,
        "filename": filename,
        "result": result,
        "ip": identity.get("ip", ""),
        "detail": detail,
    }
    try:
        with _audit_lock:
            with open(AUDIT_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"[AUDIT] 写入失败: {e}")
def resolve_identity(request: Request, x_api_key: Optional[str] = Header(default=None)) -> dict:
    """
    解析请求身份：
    - 配置了 API_KEYS：按 Key 查表得到 tenant/role
    - 只配置 UPLOAD_API_KEY：单租户 default，角色 admin
    - 都没配置：本地演示，匿名 default/admin
    """
    ip = request.client.host if request.client else ""
    if _API_KEYS:
        key = x_api_key or ""
        info = _API_KEYS.get(key)
        if not info:
            raise HTTPException(status_code=401, detail="鉴权失败：X-API-Key 缺失或不正确")
        role = info.get("role", "view")
        if role not in ROLE_LEVELS:
            raise HTTPException(status_code=403, detail=f"角色配置非法：{role}")
        return {
            "tenant": info.get("tenant") or DEFAULT_TENANT,
            "role": role,
            "user": hashlib.sha256(key.encode("utf-8")).hexdigest()[:8],
            "ip": ip,
        }
    if _UPLOAD_API_KEY:
        if x_api_key != _UPLOAD_API_KEY:
            raise HTTPException(status_code=401, detail="鉴权失败：X-API-Key 缺失或不正确")
        return {
            "tenant": DEFAULT_TENANT,
            "role": "admin",
            "user": hashlib.sha256(x_api_key.encode("utf-8")).hexdigest()[:8],
            "ip": ip,
        }
    return {"tenant": DEFAULT_TENANT, "role": "admin", "user": "anonymous", "ip": ip}
def require_role(needed: str):
    """角色校验依赖：admin 可以覆盖 upload/delete/view"""
    def dependency(identity: dict = Depends(resolve_identity)):
        if ROLE_LEVELS[identity["role"]] < ROLE_LEVELS[needed]:
            write_audit(identity, "auth.denied", None, "failed",
                        f"role={identity['role']} need={needed}")
            raise HTTPException(status_code=403, detail=f"无权限：该操作需要 {needed} 角色")
        return identity
    return dependency
def scoped_session(identity: dict, session_id: Optional[str]) -> Optional[str]:
    """服务端给 session_id 加租户前缀，客户端改 session_id 也读不到别的租户会话"""
    if not session_id:
        return None
    return f"{identity['tenant']}:{session_id}"
# ==================== 请求/响应模型 ====================
class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    history: Optional[List[dict]] = None
    session_id: Optional[str] = None
    source_file: Optional[Union[str, List[str]]] = None
class Source(BaseModel):
    content: str
    score: float
class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks: int
    chars: int = 0
    text: str = ""
class ChatRequest(BaseModel):
    user_input: str
    contract_text: Optional[str] = ""
    history: Optional[list] = []
class ChatResponse(BaseModel):
    answer: str
    tool_used: str
    step_count: int
    success: bool
    sources: Optional[list] = []
# ==================== 根路径 & 健康检查 ====================
@app.get("/")
def root():
    return {
        "message": "🏠 租房合同避坑助手已启动",
        "version": "2.0",
        "endpoints": ["/chat", "/ask", "/ask/stream", "/upload", "/documents/{filename}", "/feedback", "/status"]
    }
@app.get("/health")
def health():
    return {"status": "ok"}
@app.get("/status")
def status(identity: dict = Depends(resolve_identity)):
    return {
        "chunks_loaded": rag.store.count_tenant(identity["tenant"]),
        "vector_dim": 384
    }
# ==================== Agent 对话接口 ====================
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, identity: dict = Depends(resolve_identity)):
    logger.info(f"[API] 收到 Agent 请求: {request.user_input}")
    state = {
        "user_input": request.user_input,
        "contract_text": request.contract_text,
        "decision": {},
        "tool_result": "",
        "final_answer": "",
        "step_count": 0,
        "history": request.history if request.history else []
    }
    # Agent 内部的 rag_search 也要按租户过滤，用 contextvar 传递当前租户
    token = set_current_tenant(identity["tenant"])
    try:
        result = graph.invoke(state)
        answer = result.get("final_answer", "未获取到回答")
        decision = result.get("decision", {})
        tool_name = decision.get("tool", "未知")
        step_count = result.get("step_count", 0)

        if isinstance(answer, dict):
            answer = json.dumps(answer, ensure_ascii=False)
        elif not isinstance(answer, str):
            answer = str(answer)
        logger.info(f"[API] Agent 返回成功，使用工具: {tool_name}")
        sources = []
        try:
           if isinstance(answer, str) and answer.strip().startswith("{"):
              parsed = json.loads(answer)
              sources = parsed.get("sources", [])
        except:
            pass
        return ChatResponse(
            answer=answer,
            tool_used=tool_name,
            step_count=step_count,
            success=True,
            sources=sources
        )
    except Exception as e:
        logger.error(f"[API] Agent 处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent 执行失败: {str(e)}")
    finally:
        reset_current_tenant(token)
# ==================== RAG 问答接口 ====================
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, identity: dict = Depends(resolve_identity)):
    try:
        result = rag.ask(request.question, request.top_k,
                         session_id=scoped_session(identity, request.session_id),
                         history=request.history,
                         source_file=request.source_file,
                         tenant=identity["tenant"])
        return AskResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/ask/stream")
def ask_stream(request: AskRequest, identity: dict = Depends(resolve_identity)):
    # 检索（含多轮改写）在流式响应前完成：出错能返回正常状态码，也不用为拿 sources 再生成一遍
    session_id = scoped_session(identity, request.session_id)
    try:
        prep = rag.prepare(request.question, request.top_k,
                           session_id=session_id,
                           history=request.history,
                           source_file=request.source_file,
                           tenant=identity["tenant"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not prep["sources"]:
        def empty():
            yield "未找到相关信息"
        return StreamingResponse(empty(), media_type="text/plain")
    def generate():
        answer_parts = []
        try:
            # 同 session 串行到本次流结束，保证历史写入顺序
            with rag.session_lock(session_id):
                client = get_llm_client()
                prompt = f"""请基于参考内容回答，只说相关信息，不编造。
参考内容：
{prep['context']}

问题：{request.question}

回答："""
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    max_tokens=500
                )
                for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        answer_parts.append(delta)
                        yield delta
                rag.append_history(session_id, request.question, "".join(answer_parts))
            yield "\n<<<SOURCES>>>\n"
            yield json.dumps({"sources": prep["sources"]}, ensure_ascii=False)
        except Exception as e:
            yield f"\n[生成失败] {e}"
    return StreamingResponse(generate(), media_type="text/plain")
# ==================== 文档上传接口 ====================
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB
_UPLOAD_API_KEY = os.getenv("UPLOAD_API_KEY")  # 未配置 API_KEYS 时的单 Key 模式
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...),
                      identity: dict = Depends(require_role("upload"))):
    filename = os.path.basename(file.filename or "")
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        if os.path.splitext(filename)[1].lower() not in {".txt", ".pdf"}:
            raise HTTPException(status_code=400, detail="仅支持 .txt / .pdf 文件")
        # 先读入内存做大小校验，再直接落盘
        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"文件过大，上限 {MAX_UPLOAD_BYTES // 1024 // 1024}MB")
        if not content:
            raise HTTPException(status_code=400, detail="文件内容为空")
        file_path = os.path.join(DOCS_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(content)
        text = await run_in_threadpool(rag.index_file, file_path, identity["tenant"])
        # 知识库变了，Agent 工具侧的缓存要作废，否则检索不到新文档
        invalidate_rag_store()
        write_audit(identity, "file.upload", filename, "success", f"{len(content)} bytes")
        return UploadResponse(
            message="上传成功",
            filename=filename,
            chunks=rag.store.count_tenant(identity["tenant"]),
            chars=len(text),
            text=text
        )
    except HTTPException as e:
        write_audit(identity, "file.upload", filename, "failed", str(e.detail))
        raise
    except ValueError as e:
        write_audit(identity, "file.upload", filename, "failed", str(e))
        # 解析类错误（如加密 PDF）按客户端错误返回，不打成 500
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        write_audit(identity, "file.upload", filename, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))
# ==================== 文档列表接口 ====================
class DocumentInfo(BaseModel):
    filename: str
    chunks: int
    size_bytes: int
class DocumentListResponse(BaseModel):
    total_chunks: int
    documents: List[DocumentInfo]
@app.get("/documents", response_model=DocumentListResponse)
def list_documents(identity: dict = Depends(resolve_identity)):
    """列出当前租户可见的文档及各自文本块数，供前端展示与删除"""
    items = []
    total = 0
    for name in sorted(os.listdir(DOCS_DIR)):
        path = os.path.join(DOCS_DIR, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() not in {".txt", ".pdf"}:
            continue
        chunks = rag.store.count_by_source(name, identity["tenant"])
        if chunks == 0:
            # 磁盘上有、但当前租户没索引过：不可见
            continue
        total += chunks
        items.append(DocumentInfo(
            filename=name,
            chunks=chunks,
            size_bytes=os.path.getsize(path)
        ))
    return DocumentListResponse(total_chunks=total, documents=items)
# ==================== 文档删除接口 ====================
class DeleteResponse(BaseModel):
    message: str
    filename: str
    removed_chunks: int
    chunks_left: int
@app.delete("/documents/{filename}", response_model=DeleteResponse)
def delete_document(filename: str, identity: dict = Depends(require_role("delete"))):
    name = os.path.basename(filename)
    if not name:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    # 不属于当前租户的文件统一按不存在处理，避免泄露别人的文件名
    if identity["tenant"] not in rag.store.tenants_of_source(name):
        write_audit(identity, "file.delete", name, "failed", "not_found")
        raise HTTPException(status_code=404, detail="文件不存在")
    try:
        removed = rag.delete_file(name, identity["tenant"])
        # 同上：删除后让 Agent 工具重新加载向量库
        invalidate_rag_store()
        write_audit(identity, "file.delete", name, "success", f"removed {removed} chunks")
        return DeleteResponse(
            message="删除成功",
            filename=name,
            removed_chunks=removed,
            chunks_left=rag.store.count_tenant(identity["tenant"])
        )
    except Exception as e:
        write_audit(identity, "file.delete", name, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))
# ==================== 判错反馈接口 ====================
class FeedbackRequest(BaseModel):
    question: str
    answer: str
    risk: Optional[str] = None          # 被判定错的条目风险等级
    source_file: Optional[str] = None   # 来自哪份文档/会话
    reason: Optional[str] = None        # 判错原因（如：风险等级偏高 / 不是风险 / 建议不对）
    note: Optional[str] = None          # 用户补充说明
    session_id: Optional[str] = None
class FeedbackResponse(BaseModel):
    message: str
    id: str
@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest, identity: dict = Depends(resolve_identity)):
    """前端「这条判错了」入口：写入 data/feedback/feedback.jsonl，供人工复核与评测集迭代"""
    if not request.question.strip() or not request.answer.strip():
        raise HTTPException(status_code=400, detail="question 与 answer 不能为空")
    feedback_id = hashlib.sha256(
        f"{datetime.now().timestamp()}{identity['user']}{request.question}".encode("utf-8")
    ).hexdigest()[:12]
    record = {
        "id": feedback_id,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "tenant": identity["tenant"],
        "user": identity["user"],
        "session_id": scoped_session(identity, request.session_id),
        "question": request.question,
        "answer": request.answer,
        "risk": request.risk,
        "source_file": request.source_file,
        "reason": request.reason,
        "note": request.note,
    }
    try:
        with _audit_lock:
            with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"[FEEDBACK] 写入失败: {e}")
        raise HTTPException(status_code=500, detail="反馈写入失败")
    write_audit(identity, "feedback", request.source_file, "success", feedback_id)
    return FeedbackResponse(message="反馈已记录，感谢", id=feedback_id)
# ==================== 启动入口 ====================
if __name__ == "__main__":
    import uvicorn
    print("🚀 启动租房合同避坑助手服务...")
    print("📡 访问地址: http://127.0.0.1:8000")
    print("📖 接口文档: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)