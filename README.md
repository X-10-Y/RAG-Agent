# 租房合同避坑助手（RAG + Agent）

上传或粘贴租房合同，自动识别风险条款，输出按风险排序的报告、修改建议和租房知识问答。

面向的场景：租客看不懂合同、不知道哪些条款不合理，需要一个能指出"哪一条有问题、为什么、怎么改"的工具。

## 功能

| 功能 | 说明 | 实现方式 |
|------|------|----------|
| 单条条款分析 | 输入一条条款，返回风险等级、解释、修改建议 | LLM + 规则预检 |
| 快速预览 | 通读整份合同，给出整体风险判断 | 单次 LLM 调用 |
| 深度扫描 | 逐条拆解，生成按风险排序的完整报告 | 条款切分 + 逐条分析 + 规则兜底 |
| 知识问答 | 回答租房常识（押金、转租、甲醛、二房东等） | RAG 混合检索 |
| 合同上传 | 支持 TXT / PDF / 图片，自动解析、切分、索引 | pdfplumber → pypdf → OCR 降级 |

## 架构

```
Streamlit 前端
      │  /chat  /ask  /upload
      ▼
FastAPI（端口 8000）
      │
      ├── LangGraph Agent
      │     decide ── 意图识别，选择工具
      │       │
      │       ▼
      │     execute ── analyze_clause / quick_preview / deep_scan
      │       │        rag_search / upload_document
      │       ▼
      │     final ── 汇总输出
      │       └── 失败时经条件边回到 decide（最多 2 次）
      │
      └── RAGChain
            查询改写 → BM25 + 向量双路召回 → RRF(k=60) 融合
                    → CrossEncoder 重排 → LLM 生成
```

## 技术栈

- **Agent 编排**：LangGraph（State + 条件边）
- **RAG 检索**：jieba + BM25、ChromaDB 向量检索、RRF 融合、CrossEncoder 重排
- **LLM**：DeepSeek Chat（OpenAI 兼容接口）
- **向量模型**：paraphrase-multilingual-MiniLM-L12-v2（384 维）
- **后端**：FastAPI
- **前端**：Streamlit

## 核心设计

1. **长文本走 State，不走 JSON**：合同正文由前端传入 State，执行节点强制注入，避免让模型生成上万字 JSON 导致截断或解析失败。LLM 只负责选择工具和生成结论。
2. **规则 + LLM 双层审查**：押金月数、违约金比例、滞纳金比例等可量化指标先用正则拦截，模糊条款（"最终解释权""押金概不退还"）交给 LLM 判断。阈值参考各地常见规定，是启发式红线，不是全国统一法条，可按城市配置。
3. **混合检索**：BM25 负责法条编号、专有名词等精确匹配，向量检索负责语义相近；RRF 用排名而非分数融合，避免 BM25 分数无上界、余弦相似度量纲不一致的问题；最后用 CrossEncoder 对候选重排。
4. **结构感知切分**：优先按标题、段落、句子切分，表格整块保留不切断，单块上限 500 字。
5. **失败可回退**：工具返回失败时经条件边回到决策节点重试；同一工具连续失败则强制切换到知识库检索兜底，避免死循环。

## 目录结构

```
Z-RAG-Agent/
├── backend/src/
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 模型、路径配置
│   ├── agent/
│   │   ├── agent.py         # LangGraph 状态机
│   │   └── tools.py         # 5 个工具 + 规则审查
│   └── rag/
│       ├── loader.py        # TXT/PDF/图片解析（三级降级）
│       ├── splitter.py      # 结构感知切分
│       ├── embedder.py      # 向量化
│       ├── retriever.py     # BM25 + 向量 + RRF + 重排
│       └── rag_chain.py     # 检索问答主流程
├── frontend/app.py          # Streamlit 前端
├── data/documents/          # 示例合同与知识文档
├── data/chroma_db/          # 向量库持久化目录（不入库）
└── tests/                   # 召回率、误报率评测脚本
```

## 快速开始

### 1. 环境要求

- Python 3.10 / 3.11（3.12+ 与 cnocr 不兼容）
- Windows / macOS / Linux

### 2. 安装依赖

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# CPU 环境建议先单独装 torch，避免拉到 CUDA 版本
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

### 3. 配置

在项目根目录创建 `.env`：

```ini
DEEPSEEK_API_KEY=sk-你的密钥
```

代码中已设置 `HF_ENDPOINT=https://hf-mirror.com`，国内网络可直接下载模型。

### 4. 启动

```bash
# 后端（端口 8000）
cd backend/src
python main.py

# 前端（新终端）
cd frontend
streamlit run app.py
```

访问：

- 接口文档：http://127.0.0.1:8000/docs
- 前端：http://127.0.0.1:8501

首次启动会加载向量模型与重排模型（约 400MB + 90MB），需要 10–20 秒；之后从本地缓存加载。

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat` | Agent 对话：意图识别 + 工具调用 |
| POST | `/ask` | RAG 知识问答，返回答案与来源 |
| POST | `/ask/stream` | 流式返回答案，末尾附带来源 |
| POST | `/upload` | 上传 TXT/PDF/图片并建立索引 |
| GET | `/status` | 查看已索引文本块数量、向量维度 |
| GET | `/health` | 健康检查 |
| GET | `/` | 服务信息 |

## 评测

先启动后端，再执行：

```bash
cd tests
python evaluate.py             # 风险召回率
python test_false_positive.py  # 合规合同误报
```

测试集：10 份合同，覆盖 10 类典型风险，共标注 20 个风险点。

| 指标 | 结果 | 说明 |
|------|------|------|
| 风险召回 | 20/20 | 20 个标注风险点全部在报告中被提及 |
| 风险等级准确率 | 待重跑 | 对比系统返回的 risk 字段与数据集标注 |
| 高风险误报 | 0 条 | 合规合同 9 个条款中，无高风险误判 |
| 中风险偏保守 | 3 条 | 同一份合规合同被判为中风险，偏保守 |

评测脚本自动输出召回命中、风险等级准确率（新增）、逐份合同明细。召回判定为"报告的条款原文或解释中包含标注关键词"，风险等级准确率仅在召回命中的条款上计算，是更严格的指标。误报测试仅使用 1 份合规合同，样本量小，结果不能外推。

## 常见问题

- **首次启动慢**：加载 Embedding（约 400MB）与 Reranker（约 90MB）模型，10–20 秒。
- **PDF 报错 `pdftoppm not found`**：Windows 需安装 [poppler](https://github.com/oschwartz10612/poppler-windows/releases)，解压后把 `Library\bin` 加入 PATH。
- **深度扫描慢**：逐条调用 LLM，9 条约 20 秒，属于串行实现。
- **torch 安装慢或体积大**：使用上面的 CPU 版安装命令。
- **向量库为空**：删除 `data/chroma_db/` 后重启，会重新索引 `data/documents/` 下的文件。

## 已知局限

- **会话历史是进程内内存**：已按 `session_id` 隔离，但仍是单机内存，重启丢失、多实例不共享；生产环境应迁移到 Redis。
- **重排模型为英文**：`ms-marco-MiniLM-L-6-v2` 对中文是次优选择，中文场景应换用 bge-reranker 系列。
- **上传接口无鉴权、无大小限制**：已限制为 TXT/PDF 并做了文件名规范化，同名文件仍会覆盖，仅适合本地演示。
- **BM25 与向量库的 doc_id 依赖添加顺序**：删除文档或并发写入后可能错位，应改为统一的 chunk_id 映射。

## Roadmap

- 会话历史迁移到 Redis，支持多实例部署
- 上传接口补文件大小校验与鉴权
- 深度扫描改为并发 + 限流，降低耗时
- 换用中文 reranker，并补 recall@k / MRR 指标
