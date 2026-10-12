import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from config import DOCS_DIR
from openai import OpenAI
from dotenv import load_dotenv
from rag.loader import load_text_files
from rag.splitter import split_by_structure
from rag.retriever import VectorStore
load_dotenv()
class RAGChain:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = DOCS_DIR
        self.data_dir = data_dir
        self.store = VectorStore()
        self.store.create_collection()
        self._load_and_index()
        self.history = []  # 存储对话历史 [(user, assistant), ...]
    def _load_and_index(self):
    # 如果向量库已有数据，跳过重复索引
        if self.store.count() > 0:
           print(f"✅ 向量库已有 {self.store.count()} 条数据，跳过索引")
           return
        docs = load_text_files(self.data_dir)
        if not docs:
           print("⚠️ 未找到文档，跳过索引")
           return
        for doc in docs:
            chunks = split_by_structure(doc['content'])
            self.store.add_documents(chunks, source_file=doc['filename'])
        print(f"✅ 已索引 {self.store.count()} 个文本块")
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

        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        rewritten = resp.choices[0].message.content.strip()
        return rewritten
    def ask(self, question: str, top_k: int = 5) -> dict:
        # 查询改写
        rewritten = self._rewrite_query(question, self.history)
        print(f"🔍 原始问题：{question}")
        print(f"📝 改写后：{rewritten}")
        # 使用改写后的问题检索
        results = self.store.hybrid_search(rewritten, top_k=top_k)
        if not results:
            return {"answer": "未找到相关信息", "sources": []}
        # 去重
        seen = set()
        unique_results = []
        for r in results:
            content = r['content'].strip()
            if content not in seen:
                seen.add(content)
                unique_results.append(r)
        print("\n📌 召回的文本块（共{}块）：".format(len(unique_results)))
        for idx, r in enumerate(unique_results):
            print(f"  {idx+1}. 重排分: {r.get('rerank_score', 0):.4f} | 内容预览: {r['content'][:100]}...")
        context = "\n\n".join([r['content'] for r in unique_results])
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "请基于参考内容回答，只说相关信息，不编造。"},
                {"role": "user", "content": f"参考内容：\n{context}\n\n问题：{question}"}
            ],
            max_tokens=500
        )
        answer = resp.choices[0].message.content
        # 更新历史
        self.history.append({"user": question, "assistant": answer})
        return {
            "answer": answer,
            "sources": [{"content": r['content'], "score": r.get('rerank_score', r['score'])} for r in unique_results]
        }


if __name__ == '__main__':
    rag = RAGChain()
    while True:
        q = input("\n💬 请输入问题（q 退出）：").strip()
        if q == 'q':
            break
        if not q:
            continue
        result = rag.ask(q)
        print(f"\n📝 回答：{result['answer']}")
        print("\n📎 来源：")
        for s in result['sources']:
            print(f"  - 相关度 {s['score']:.3f}: {s['content'][:60]}...")