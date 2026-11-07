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
# ==================== 来源过滤参数归一化 ====================
def normalize_sources(source_file) -> List[str]:
    """把单个文件名或文件名列表统一成列表，空值返回空列表"""
    if not source_file:
        return []
    if isinstance(source_file, str):
        return [source_file]
    return [s for s in source_file if s]
def build_source_where(sources: List[str]) -> dict:
    """构造 Chroma 的来源过滤条件：单个文件用等值，多个文件用 $in"""
    if not sources:
        return {}
    if len(sources) == 1:
        return {"source_file": sources[0]}
    return {"source_file": {"$in": sources}}
DEFAULT_TENANT = "default"
def build_where(sources: List[str], tenant: str = None) -> dict:
    """来源 + 租户过滤条件。租户为空不过滤，用于兼容内部调用"""
    conds = []
    if tenant:
        conds.append({"tenant": tenant})
    if sources:
        conds.append(build_source_where(sources))
    if not conds:
        return {}
    if len(conds) == 1:
        return conds[0]
    return {"$and": conds}
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
        self._collection_name = None
        # BM25 相关
        self.bm25 = None
        self.all_documents = []   # 原始文档文本列表，索引即doc_id
        self.doc_sources = []     # 与 all_documents 一一对应的来源文件名
        self.doc_tenants = []     # 与 all_documents 一一对应的租户
        self.tokenized_documents = []
        # 重排序模型（延迟加载）
        self.reranker = None
        self.last_trace = {}
        try:
            self.collection = self.client.get_collection("documents")
            all_data = self.collection.get(include=["documents", "metadatas"])
            self.all_documents = all_data.get("documents", [])
            metadatas = all_data.get("metadatas") or []
            self.doc_sources = [
                (m or {}).get("source_file") for m in metadatas
            ]
            if len(self.doc_sources) != len(self.all_documents):
                self.doc_sources = [None] * len(self.all_documents)
            self.doc_tenants = [
                (m or {}).get("tenant", DEFAULT_TENANT) for m in metadatas
            ]
            if len(self.doc_tenants) != len(self.all_documents):
                self.doc_tenants = [DEFAULT_TENANT] * len(self.all_documents)
            self._backfill_tenant(all_data.get("ids") or [], metadatas)
            if self.all_documents:
                self.build_bm25_index()
                print(f"✅ 启动时从向量库恢复 {len(self.all_documents)} 条文档")
        except Exception:
            pass
    def _backfill_tenant(self, ids: List[str], metadatas: List[dict]):
        """老库没有 tenant 字段：补成 default，否则租户过滤会漏掉历史数据"""
        ids_to_fix, metas_to_fix = [], []
        for i, meta in enumerate(metadatas):
            if i >= len(ids):
                break
            m = dict(meta or {})
            if "tenant" not in m:
                m["tenant"] = DEFAULT_TENANT
                ids_to_fix.append(ids[i])
                metas_to_fix.append(m)
        if ids_to_fix:
            self.collection.update(ids=ids_to_fix, metadatas=metas_to_fix)
            print(f"🔧 已为 {len(ids_to_fix)} 条历史文本块补齐 tenant=default")
    # ========== 集合管理 ==========
    def create_collection(self, name: str = "documents"):
        self._collection_name = name
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
    def add_documents(self, documents: List[str], source_file: str = None, ids: List[str] = None,
                      content_hash: str = None, tenant: str = DEFAULT_TENANT):
        """
        添加文档到向量库
        source_file: 来源文件名，用于增量索引
        content_hash: 整份文档内容的哈希，用于重复上传去重
        tenant: 租户标识，检索和删除都按它过滤
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
            metadata = {"tenant": tenant}
            if source_file:
                metadata["source_file"] = source_file
                metadata["chunk_index"] = i
            if content_hash:
                metadata["content_hash"] = content_hash
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
        self.doc_sources.extend([source_file] * len(documents))
        self.doc_tenants.extend([tenant] * len(documents))
        self.build_bm25_index()
    def is_file_indexed(self, filename: str, tenant: str = DEFAULT_TENANT) -> bool:
        """检查某个文件在指定租户下是否已经被索引"""
        if self.collection is None:
            return False
        try:
            result = self.collection.get(
                where=build_where([filename], tenant),
                limit=1
        )
            return len(result['ids']) > 0
        except Exception:
            return False
    def count_by_source(self, filename: str, tenant: str = DEFAULT_TENANT) -> int:
        """统计某个来源文件在指定租户下的文本块数"""
        if self.collection is None:
            return 0
        try:
            result = self.collection.get(where=build_where([filename], tenant), include=[])
            return len(result['ids'])
        except Exception:
            return 0
    def count_tenant(self, tenant: str = DEFAULT_TENANT) -> int:
        """当前租户下的文本块总数，用于状态展示与上传/删除后的计数"""
        if self.collection is None:
            return 0
        try:
            result = self.collection.get(where={"tenant": tenant}, include=[])
            return len(result['ids'])
        except Exception:
            return 0
    def tenants_of_source(self, filename: str) -> List[str]:
        """某个文件分布在哪些租户下，用于文档列表和删除校验"""
        if self.collection is None:
            return []
        try:
            result = self.collection.get(where={"source_file": filename}, include=["metadatas"])
            metas = result.get("metadatas") or []
            return sorted({(m or {}).get("tenant", DEFAULT_TENANT) for m in metas})
        except Exception:
            return []
    def hash_exists(self, content_hash: str, tenant: str = DEFAULT_TENANT) -> bool:
        """同租户下检查同样内容是否已经入库（不同租户可以各自存一份）"""
        if self.collection is None or not content_hash:
            return False
        try:
            where = {"$and": [{"tenant": tenant}, {"content_hash": content_hash}]}
            result = self.collection.get(where=where, limit=1)
            return len(result['ids']) > 0
        except Exception:
            return False
    def delete_source(self, filename: str, tenant: str = DEFAULT_TENANT) -> int:
        """
        删除某个来源文件在指定租户下的全部文本块，返回删除的块数
        Chroma 的 id 是连续序号且与内存数组下标一一对应，直接删会产生空洞，
        因此这里复用已有向量重建集合（不重新 embedding），保证 id 重新连续
        """
        if self.collection is None:
            self.create_collection()
        name = self._collection_name or "documents"
        data = self.collection.get(include=["documents", "metadatas", "embeddings"])
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        embs = data.get("embeddings")
        keep_docs, keep_metas, keep_embs = [], [], []
        for i, doc in enumerate(docs):
            meta = (metas[i] if i < len(metas) else None) or {}
            if meta.get("source_file") == filename and meta.get("tenant", DEFAULT_TENANT) == tenant:
                continue
            keep_docs.append(doc)
            keep_metas.append(meta)
            emb = embs[i]
            keep_embs.append(emb.tolist() if hasattr(emb, "tolist") else list(emb))
        removed = len(docs) - len(keep_docs)
        self.client.delete_collection(name)
        self.collection = self.client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
        if keep_docs:
            self.collection.add(
                documents=keep_docs,
                embeddings=keep_embs,
                ids=[str(i) for i in range(len(keep_docs))],
                metadatas=keep_metas
            )
        self.all_documents = keep_docs
        self.doc_sources = [m.get("source_file") for m in keep_metas]
        self.doc_tenants = [m.get("tenant", DEFAULT_TENANT) for m in keep_metas]
        self.tokenized_documents = []
        self.bm25 = None
        if self.all_documents:
            self.build_bm25_index()
        print(f"✅ 已删除 {filename}（租户 {tenant}）的 {removed} 个文本块，剩余 {len(keep_docs)} 块")
        return removed
    # ========== 纯向量检索（保留原有接口） ==========
    def search(self, query: str, top_k: int = 3, source_file: str = None,
               tenant: str = None) -> List[Dict[str, Any]]:
        if self.collection is None:
            self.create_collection()
        query_vector = embed_query(query)
        query_kwargs = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        where = build_where(normalize_sources(source_file), tenant)
        if where:
            query_kwargs["where"] = where
        results = self.collection.query(**query_kwargs)
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
    def hybrid_search(self, query: str, top_k: int = 5, recall_k: int = 50,
                      source_file: str = None, tenant: str = None) -> List[Dict[str, Any]]:
        """
        混合检索：BM25 + 向量检索，通过 RRF 融合，然后用 CrossEncoder 重排序
        tenant 非空时，BM25 和向量两路都按租户过滤，避免跨租户串数据
        """
        trace = {"query": query}
        if self.collection is None:
            self.create_collection()
        if self.bm25 is None:
            # 无 BM25 索引（如向量库为空或索引构建失败）时降级为纯向量检索
            print("⚠️ BM25 索引未构建，降级为纯向量检索")
            return self.search(query, top_k, source_file=source_file, tenant=tenant)
        # ---------- 1. BM25 检索 ----------
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # 按 BM25 分数降序排列，得到 doc_id 列表（索引）
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        sources = normalize_sources(source_file)
        if sources:
            # 只保留指定来源文件的文本块（支持多个文件）
            bm25_ranked = [
                i for i in bm25_ranked
                if i < len(self.doc_sources) and self.doc_sources[i] in sources
            ]
        if tenant:
            bm25_ranked = [
                i for i in bm25_ranked
                if i < len(self.doc_tenants) and self.doc_tenants[i] == tenant
            ]
        bm25_ranked = bm25_ranked[:recall_k]
        trace["bm25_top"] = [(doc_id, float(bm25_scores[doc_id])) for doc_id in bm25_ranked[:5]]
        # ---------- 2. 向量检索 ----------
        query_vector = embed_query(query)
        query_kwargs = {
            "query_embeddings": [query_vector],
            "n_results": recall_k,
            "include": ["documents", "metadatas", "distances"]
        }
        where = build_where(sources, tenant)
        if where:
            query_kwargs["where"] = where
        vector_results = self.collection.query(**query_kwargs)
        vector_ranked = []
        for doc_id in vector_results['ids'][0]:
            try:
                 vector_ranked.append(int(doc_id))
            except ValueError:
                 print(f"⚠️ 跳过非数字 ID: {doc_id}")
                 continue
        trace["vector_top"] = vector_ranked[:5]
        # ---------- 3. RRF 融合 ----------
        fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked], k=60)
        trace["fused_top"] = fused[:5]
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
            trace["reranked_top"] = [(r["doc_id"], r.get("rerank_score", 0)) for r in reranked]
            self.last_trace = trace
            return reranked
        else:
            self.last_trace = trace
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