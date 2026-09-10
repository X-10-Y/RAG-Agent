import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import shutil
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# ==================== 导入 RAG 和 Agent ====================
from config import DOCS_DIR
from logger import logger
from rag.rag_chain import RAGChain
from agent.agent import graph
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
# ==================== 请求/响应模型 ====================
class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    history: Optional[List[dict]] = None
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
class ChatRequest(BaseModel):
    user_input: str
    contract_text: Optional[str] = ""
    history: Optional[list] = []
class ChatResponse(BaseModel):
    answer: str
    tool_used: str
    step_count: int
    success: bool
# ==================== 根路径 & 健康检查 ====================
@app.get("/")
def root():
    return {
        "message": "🏠 租房合同避坑助手已启动",
        "version": "2.0",
        "endpoints": ["/chat", "/ask", "/ask/stream", "/upload", "/status"]
    }
@app.get("/health")
def health():
    return {"status": "ok"}
@app.get("/status")
def status():
    return {
        "chunks_loaded": rag.store.count(),
        "vector_dim": 384
    }
# ==================== Agent 对话接口 ====================
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
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
        return ChatResponse(answer=answer, tool_used=tool_name, step_count=step_count, success=True)
    except Exception as e:
        logger.error(f"[API] Agent 处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent 执行失败: {str(e)}")
# ==================== RAG 问答接口 ====================
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = rag.ask(request.question, request.top_k)
        return AskResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/ask/stream")
def ask_stream(request: AskRequest):
    def generate():
        sources = rag.ask(request.question, request.top_k).get("sources", [])
        if not sources:
            yield "未找到相关信息"
            return
        context = "\n\n".join([s["content"] for s in sources])
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        prompt = f"""请基于参考内容回答，只说相关信息，不编造。
参考内容：
{context}

问题：{request.question}

回答："""
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=500
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
        yield "\n<<<SOURCES>>>\n"
        yield json.dumps({"sources": sources})
    return StreamingResponse(generate(), media_type="text/plain")
# ==================== 文档上传接口 ====================
@app.post("/upload", response_model=UploadResponse)
def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(DOCS_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        rag._load_and_index()
        return UploadResponse(
            message="上传成功",
            filename=file.filename,
            chunks=rag.store.count()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ==================== 启动入口 ====================
if __name__ == "__main__":
    import uvicorn
    print("🚀 启动租房合同避坑助手服务...")
    print("📡 访问地址: http://127.0.0.1:8000")
    print("📖 接口文档: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)