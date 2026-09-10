import chromadb
from typing import List, Dict, Any
from rag.embedder import embed_texts, embed_query
from rank_bm25 import BM25Okapi
import jieba
from sentence_transformers import CrossEncoder
import os
# ==================== RRF 融合函数 ====================
def reciprocal_rank_fusion(ranked_lists: List[List[int]], k: int = 60) -> List[tuple]:
    """
    倒数排名融合（Reciprocal Rank Fusion）
    将多个排序列表合并，分数为 sum(1/(k+rank))
    返回：[(doc_id, score), ...] 按 score 降序
    """
    scores = {}
    for rank_list in ranked_lists:
        for rank, doc_id in enumerate(rank_list):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
class VectorStore:
    """
    向量存储和检索引擎
    支持：向量检索、BM25关键词检索、RRF融合、CrossEncoder重排序
    """
    def __init__(self, persist_dir: str = None):
        if persist_dir is None:
            from config import CHROMA_DB_DIR
            persist_dir = CHROMA_DB_DIR
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = None
        # BM25 相关
        self.bm25 = None
        self.all_documents = []   # 原始文档文本列表，索引即doc_id
        self.tokenized_documents = []
        # 重排序模型（延迟加载）
        self.reranker = None
    # ========== 集合管理 ==========
    def create_collection(self, name: str = "documents"):
        try:
            self.collection = self.client.get_collection(name)
            print(f"✅ 已加载已有集合: {name}")
        except:
            self.collection = self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✅ 已创建新集合: {name}")
        return self.collection
    # ========== BM25 索引构建 ==========
    def build_bm25_index(self):
        if not self.all_documents:
            print("⚠️ 没有文档，无法构建 BM25 索引")
            return
        print(f"🔍 正在构建 BM25 索引，共 {len(self.all_documents)} 篇文档...")
        self.tokenized_documents = [list(jieba.cut(doc)) for doc in self.all_documents]
        self.bm25 = BM25Okapi(self.tokenized_documents)
        print(f"✅ BM25 索引构建完成")
    # ========== 添加文档 ==========
    def add_documents(self, documents: List[str], source_file: str = None, ids: List[str] = None):
        """
        添加文档到向量库
        source_file: 来源文件名，用于增量索引
        """
        if self.collection is None:
            self.create_collection()
        print(f"🔍 正在生成 {len(documents)} 个文档的向量...")
        embeddings = embed_texts(documents)
        if ids is None:
            start = len(self.all_documents)
            ids = [str(start + i) for i in range(len(documents))]
    # ===== 新增：构建 metadata，记录来源文件 =====
        metadatas = []
        for i in range(len(documents)):
            metadata = {}
            if source_file:
                metadata["source_file"] = source_file
                metadata["chunk_index"] = i
            metadatas.append(metadata)
        kwargs = {
             "documents": documents,
             "embeddings": embeddings,
             "ids": ids,
             "metadatas": metadatas
        }
        self.collection.add(**kwargs)
        print(f"✅ 已添加 {len(documents)} 个文档到向量库")
        self.all_documents.extend(documents)
        self.build_bm25_index()
    def is_file_indexed(self, filename: str) -> bool:
        """检查某个文件是否已经被索引"""
        if self.collection is None:
            return False
        try:
            result = self.collection.get(
                where={"source_file": filename},
                limit=1
        )
            return len(result['ids']) > 0
        except Exception:
            return False
    # ========== 纯向量检索（保留原有接口） ==========
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.collection is None:
            self.create_collection()
        query_vector = embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                "content": results['documents'][0][i],
                "metadatas": results['metadatas'][0][i] if results['metadatas'] else {},
                "score": 1 - results['distances'][0][i]
            })
        return formatted_results
    # ========== 重排序（CrossEncoder） ==========
    def _get_reranker(self):
        """延迟加载重排序模型"""
        if self.reranker is None:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            print("🔍 加载重排序模型（CrossEncoder）...")
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("✅ 重排序模型加载完成")
        return self.reranker
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        使用 CrossEncoder 对候选文档进行重排序
        """
        if not candidates:
            return []
        model = self._get_reranker()
        # 构造 (query, document) 对
        pairs = [[query, doc['content']] for doc in candidates]
        # 计算相关性分数
        scores = model.predict(pairs)
        # 更新候选文档的分数
        for i, doc in enumerate(candidates):
            doc['rerank_score'] = float(scores[i])
        # 按重排序分数降序排列
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        return candidates[:top_k]
    # ========== 混合检索（BM25 + 向量 + RRF 融合） ==========
    def hybrid_search(self, query: str, top_k: int = 5, recall_k: int = 50) -> List[Dict[str, Any]]:
        """
        混合检索：BM25 + 向量检索，通过 RRF 融合，然后用 CrossEncoder 重排序
        """
        if self.collection is None:
            self.create_collection()
        if self.bm25 is None:
            print("⚠️ BM25 索引未构建，请先添加文档")
            return []
        # ---------- 1. BM25 检索 ----------
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # 按 BM25 分数降序排列，得到 doc_id 列表（索引）
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:recall_k]
        # ---------- 2. 向量检索 ----------
        query_vector = embed_query(query)
        vector_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=recall_k,
            include=["documents", "metadatas", "distances"]
        )
        vector_ranked = []
        for doc_id in vector_results['ids'][0]:
            try:
                 vector_ranked.append(int(doc_id))
            except ValueError:
                 print(f"⚠️ 跳过非数字 ID: {doc_id}")
                 continue
        # ---------- 3. RRF 融合 ----------
        fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked], k=60)
        # 取前 top_k 个 doc_id
        top_doc_ids = [doc_id for doc_id, score in fused[:top_k]]
        # ---------- 4. 构建候选结果 ----------
        candidates = []
        for doc_id in top_doc_ids:
            candidates.append({
                "doc_id": doc_id,
                "content": self.all_documents[doc_id],
                "score": 0.0,  # 占位，后续由 rerank 覆盖
            })
        # ---------- 5. 重排序 ----------
        if candidates:
            reranked = self.rerank(query, candidates, top_k=top_k)
            return reranked
        else:
            return []
    def count(self) -> int:
        if self.collection is None:
            return 0
        return self.collection.count()
# ========== 独立测试 ==========
if __name__ == '__main__':
    from rag.loader import load_text_files
    from rag.splitter import split_by_paragraph
    docs = load_text_files('../../data/documents')
    if not docs:
        print("❌ 没有找到文档")
        exit(1)
    all_chunks = []
    for doc in docs:
        chunks = split_by_paragraph(doc['content'])
        all_chunks.extend(chunks)
    print(f"📄 总共切分为 {len(all_chunks)} 个块")
    store = VectorStore()
    store.create_collection()
    store.add_documents(all_chunks)
    print(f"📊 向量库中共有 {store.count()} 条记录")
    print("\n🔍 测试混合检索（RRF融合 + 重排序）...")
    test_query = "产品安全注意事项有哪些"
    results = store.hybrid_search(test_query, top_k=5)
    for i, r in enumerate(results):
        print(f"\n结果 {i+1} (重排分: {r.get('rerank_score', 0):.4f})")
        print(f"  内容: {r['content'][:150]}...")