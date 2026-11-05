import os
from dotenv import load_dotenv
load_dotenv()
# ==================== 路径配置 ====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DOCS_DIR = os.path.join(PROJECT_ROOT, "data", "documents")
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_db")
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
# ==================== 模型配置 ====================
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTOR_DIM = 384
# ==================== LLM 配置 ====================
LLM_MODEL = "deepseek-chat"
LLM_BASE_URL = "https://api.deepseek.com"
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY")
LLM_TIMEOUT = 60        # 单次请求超时（秒），SDK 默认值长达数百秒
LLM_MAX_RETRIES = 2     # SDK 层自动重试次数
def get_llm_client():
    """统一创建 LLM 客户端：显式超时 + 有限重试，避免请求长期挂起"""
    from openai import OpenAI
    return OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES
    )