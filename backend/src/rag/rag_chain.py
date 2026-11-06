import os
import json
import hashlib
import contextlib
import threading
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from config import DOCS_DIR, get_llm_client
from dotenv import load_dotenv
from rag.loader import load_text_files, load_document
from rag.splitter import split_by_structure
from rag.retriever import VectorStore
load_dotenv()

# ==================== Redis 会话持久化 ====================
# 优先用 Redis 存会话历史（重启不丢、多实例共享），不可用时降级到进程内 dict
# 用 db=1 避免和 ChatBI 的 db=0 冲突
try:
    import redis
    _redis_cli = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"),
                             port=int(os.getenv("REDIS_PORT", 6379)),
                             db=int(os.getenv("RAG_REDIS_DB", 1)),
                             decode_responses=True)
    _redis_cli.ping()
    _REDIS_OK = True
    _HISTORY_KEY_PREFIX = "rag:history:"
    _HISTORY_TTL = int(os.getenv("HISTORY_TTL_SECONDS", 7200))  # 默认 2 小时
except Exception as _e:
    print(f"⚠️ Redis 不可用，会话历史降级到进程内 dict: {_e}")
    _REDIS_OK = False
    _redis_cli = None


class RAGChain:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = DOCS_DIR
        self.data_dir = data_dir
        self.store = VectorStore()
        self.store.create_collection()
        self._load_and_index()
        # 会话历史按 session_id 隔离，避免多用户互相污染
        if _REDIS_OK:
            self._histories = None  # 用 Redis，不走 dict
            print(f"✅ 会话历史已接入 Redis (db=1, TTL={_HISTORY_TTL}s)")
        else:
            self._histories = {}
            self._history_lock = threading.Lock()
        self._max_history_turns = 10
        # 同一 session 的多轮请求串行化，避免改写时读到另一次请求半更新的历史
        self._session_locks = {}
        self._session_locks_guard = threading.Lock()

    def session_lock(self, session_id: str = None):
        """按 session 返回互斥锁；无 session_id 时返回空上下文，不做串行"""
        if not session_id:
            return contextlib.nullcontext()
        with self._session_locks_guard:
            lock = self._session_locks.setdefault(session_id, threading.Lock())
        return lock

    def _load_and_index(self):
        # 只为「尚未入库」的文件建索引：既避免重复索引，也能补齐后来新增的文档
        supported = ('.txt', '.pdf')
        if not os.path.isdir(self.data_dir):
            print(f"⚠️ 目录不存在：{self.data_dir}")
            return
        filenames = [
            f for f in os.listdir(self.data_dir)
            if os.path.splitext(f)[1].lower() in supported
        ]
        pending = [f for f in filenames if not self.store.is_file_indexed(f)]
        if not pending:
            print(f"✅ 向量库已有 {self.store.count()} 条数据，文档均已入库")
            return
        print(f"🔍 发现 {len(pending)} 份文档未入库：{', '.join(pending)}")
        for filename in pending:
            text = load_document(os.path.join(self.data_dir, filename))
            if not text.strip():
                print(f"⚠️ {filename} 解析为空，跳过")
                continue
            chunks = split_by_structure(text)
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            self.store.add_documents(chunks, source_file=filename, content_hash=content_hash)
            print(f"✅ 已索引 {filename}：{len(chunks)} 个文本块")
        print(f"✅ 索引完成，当前共 {self.store.count()} 个文本块")
    def index_file(self, filepath: str) -> str:
        """解析单个文件并增量入库，返回解析出的正文（供前端回填）"""
        text = load_document(filepath)
        filename = os.path.basename(filepath)
        if not text.strip():
            return text
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # 内容哈希幂等：同样内容重复上传不再重复建索引
        if self.store.hash_exists(content_hash):
            print(f"⏭️ {filename} 内容已存在，跳过索引")
            return text
        if self.store.is_file_indexed(filename):
            # 同名文件内容变了：先删旧块再建，避免新旧版本混在一起
            removed = self.store.delete_source(filename)
            print(f"♻️ {filename} 内容已更新，替换旧块 {removed} 个")
        chunks = split_by_structure(text)
        self.store.add_documents(chunks, source_file=filename, content_hash=content_hash)
        print(f"✅ 已索引 {filename}：{len(chunks)} 个文本块")
        return text
    def delete_file(self, filename: str) -> int:
        """删除文档：向量库/BM25 与磁盘文件一起删，返回移除的文本块数"""
        removed = self.store.delete_source(filename)
        path = os.path.join(self.data_dir, filename)
        if os.path.exists(path):
            os.remove(path)
        print(f"🗑️ 已删除 {filename}，移除 {removed} 个文本块")
        return removed
    # ========== 会话历史（按 session_id 隔离） ==========
    def _history_key(self, session_id: str) -> str:
        return f"{_HISTORY_KEY_PREFIX}{session_id}"

    def _get_history(self, session_id: str = None) -> list:
        if not session_id:
            return []
        if _REDIS_OK:
            key = self._history_key(session_id)
            raw = _redis_cli.lrange(key, 0, -1)
            return [json.loads(item) for item in raw]
        else:
            with self._history_lock:
                return list(self._histories.get(session_id, []))

    def append_history(self, session_id: str, question: str, answer: str):
        if not session_id:
            return
        turn = json.dumps({"user": question, "assistant": answer}, ensure_ascii=False)
        if _REDIS_OK:
            key = self._history_key(session_id)
            _redis_cli.rpush(key, turn)
            # 保持最多 _max_history_turns 条
            _redis_cli.ltrim(key, -self._max_history_turns, -1)
            _redis_cli.expire(key, _HISTORY_TTL)
        else:
            with self._history_lock:
                history = self._histories.setdefault(session_id, [])
                history.append({"user": question, "assistant": answer})
                del history[:-self._max_history_turns]
    def _format_history(self, history: list) -> str:
        """将历史对话格式化为文本"""
        if not history:
            return "无历史对话"
        lines = []
        for turn in history:
            lines.append(f"用户：{turn['user']}")
            lines.append(f"助手：{turn['assistant']}")
        return "\n".join(lines)
    def _rewrite_query(self, question: str, history: list = None) -> str:
        """将多轮对话中的问题改写为完整独立的问句"""
        if not history:
            return question

        formatted_history = self._format_history(history)

        prompt = f"""你是一个查询改写助手。用户正在进行多轮对话，请根据对话历史，将当前问题改写为完整、独立的问句。
对话历史：
{formatted_history}

当前问题：{question}

改写后的完整问句："""

        client = get_llm_client()
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        rewritten = resp.choices[0].message.content.strip()
        return rewritten
    def prepare(self, question: str, top_k: int = 5, session_id: str = None,
                history: list = None, source_file=None) -> dict:
        """只做多轮改写 + 检索，不调用生成模型（/ask 与 /ask/stream 复用）"""
        if history is None:
            history = self._get_history(session_id)
        rewritten = self._rewrite_query(question, history)
        print(f"🔍 原始问题：{question}")
        print(f"📝 改写后：{rewritten}")
        # 使用改写后的问题检索
        results = self.store.hybrid_search(rewritten, top_k=top_k, source_file=source_file)
        # 去重
        seen = set()
        unique_results = []
        for r in results:
            content = r['content'].strip()
            if content not in seen:
                seen.add(content)
                unique_results.append(r)
        if not unique_results:
            return {"question": question, "rewritten": rewritten, "context": "", "sources": []}
        print("\n📌 召回的文本块（共{}块）：".format(len(unique_results)))
        for idx, r in enumerate(unique_results):
            print(f"  {idx+1}. 重排分: {r.get('rerank_score', 0):.4f} | 内容预览: {r['content'][:100]}...")
        context = "\n\n".join([r['content'] for r in unique_results])
        return {
            "question": question,
            "rewritten": rewritten,
            "context": context,
            "sources": [{"content": r['content'], "score": r.get('rerank_score', r['score'])} for r in unique_results]
        }

    def ask(self, question: str, top_k: int = 5, session_id: str = None,
            history: list = None, source_file=None) -> dict:
        # 同 session 串行：改写读到完整历史，回答写回也不会互相覆盖
        with self.session_lock(session_id):
            prep = self.prepare(question, top_k=top_k, session_id=session_id,
                                history=history, source_file=source_file)
            if not prep["sources"]:
                return {"answer": "未找到相关信息", "sources": []}
            client = get_llm_client()
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "请基于参考内容回答，只说相关信息，不编造。"},
                    {"role": "user", "content": f"参考内容：\n{prep['context']}\n\n问题：{question}"}
                ],
                max_tokens=500
            )
            answer = resp.choices[0].message.content
            # 更新会话历史（按 session_id 隔离）
            self.append_history(session_id, question, answer)
            return {"answer": answer, "sources": prep["sources"]}


if __name__ == '__main__':
    rag = RAGChain()
    while True:
        q = input("\n💬 请输入问题（q 退出）：").strip()
        if q == 'q':
            break
        if not q:
            continue
        result = rag.ask(q, session_id="cli")
        print(f"\n📝 回答：{result['answer']}")
        print("\n📎 来源：")
        for s in result['sources']:
            print(f"  - 相关度 {s['score']:.3f}: {s['content'][:60]}...")