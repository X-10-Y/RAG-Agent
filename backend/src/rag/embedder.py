import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
from typing import List
_model = None
def get_model():
    global _model
    if _model is None:
        print("🔍 加载向量模型（首次运行会下载约400MB，请耐心等待）...")
        _model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ 模型加载完成")
    return _model
def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    return model.encode(texts, convert_to_numpy=True).tolist()
def embed_query(query: str) -> List[float]:
    model = get_model()
    return model.encode(query, convert_to_numpy=True).tolist()
if __name__ == '__main__':
    from rag.loader import load_text_files
    from rag.splitter import split_by_paragraph
    docs = load_text_files('../../data/documents')
    if not docs:
        print("❌ 没有找到测试文档")
        exit(1)
    chunks = split_by_paragraph(docs[0]['content'])
    print(f"📄 文档切分为 {len(chunks)} 个块")
    test_texts = chunks[:3]
    vectors = embed_texts(test_texts)
    print(f"\n✅ 向量维度: {len(vectors[0])}")
    print(f"✅ 共生成 {len(vectors)} 个向量")