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
| 合同上传 | 支持 TXT / PDF / 图片，自动解析、切分、索引；支持删除与按内容哈希幂等 | pdfplumber → pypdf → OCR 降级；sha256 去重 |

## 架构

```
Streamlit 前端
      │  /chat  /ask  /ask/stream  /upload  /documents
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
└── tests/                   # 评测脚本、风险数据集与回归基线
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

会话历史默认写入本机 Redis（`localhost:6379`，db=1，TTL 2 小时），可用 `REDIS_HOST` / `REDIS_PORT` / `RAG_REDIS_DB` 覆盖；Redis 不可用时自动降级为进程内字典，服务照常可用。

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

首次启动需下载向量模型与重排模型（约 400MB + 90MB），之后从本地缓存加载，本机实测整轮启动约 1–2 分钟；启动完成前访问接口会失败，属正常现象。

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat` | Agent 对话：意图识别 + 工具调用 |
| POST | `/ask` | RAG 知识问答，返回答案与来源 |
| POST | `/ask/stream` | 流式返回答案，末尾附带来源 |
| POST | `/upload` | 上传 TXT/PDF/图片并建立索引（同内容重复上传直接跳过） |
| GET | `/documents` | 列出知识库文档：文件名、文本块数、大小 |
| DELETE | `/documents/{filename}` | 删除文档：同时清理向量库、BM25 与磁盘文件 |
| GET | `/status` | 查看已索引文本块数量、向量维度 |
| GET | `/health` | 健康检查 |
| GET | `/` | 服务信息 |

## 评测

先启动后端，再执行：

```bash
cd tests
python evaluate.py                    # 召回 / 等级 / 精确率 / 引用溯源 / 延迟
python evaluate.py --update-baseline  # 指标确认后重建基线
python test_false_positive.py         # 合规合同误报
```

风险测试集：10 份陷阱合同，覆盖 10 类典型风险，共标注 33 个风险点。误报测试集：4 份合规合同（自建 1 份 + 北京市 2023 版、市场监管总局 2025 版、吉林省 2023 版三份公开示范文本），共 59 个条款。

| 指标 | 结果 | 说明 |
|------|------|------|
| 风险召回 | 33/33 | 标注风险点在报告的风险列表中被提及 |
| 风险等级准确率 | 33/33 | 分母为全部标注风险点，漏报直接判错 |
| 高风险精确率 | 28/28 | 系统判"高"的条款全部对上标注，无高风险过检 |
| 中/高精确率 | 45.2% | 34 条中风险过检，集中在模板空白与通行的次要条款 |
| 引用可溯源 | 78/78 | 返回的条号能在原文中匹配到对应条款 |
| 高风险误报 | 0 / 59 | 4 份合规合同的 59 个条款中，无高风险误判 |
| 延迟 | p50 15.3s / p95 22.1s | 单份合同深度扫描，含 HTTP 往返 |

召回判定为"报告中的条款原文或解释包含标注关键词"，属于字符串级命中，不等同于语义正确；等级与精确率按条款级比对，条号粒度是"第 X 条"。`baseline.json` 记录指标以及版本信息（git commit、模型名、提示词哈希）：`python evaluate.py` 会与基线逐项对比，任一核心指标回退超过 2 个百分点即退出码非零，可直接接入 CI；提示词或工具代码变动导致 `prompt_hash` 不同时，只打印"对比仅供参考"而不判定回归。

## 常见问题

- **首次启动慢**：加载 Embedding（约 400MB）与 Reranker（约 90MB）模型，本机约 1–2 分钟。
- **Redis 连接告警**：日志里的 "Redis 不可用，会话历史降级到进程内 dict" 不是错误，装一个本机 Redis（Windows 可用 Memurai）即可消除。
- **PDF 报错 `pdftoppm not found`**：Windows 需安装 [poppler](https://github.com/oschwartz10612/poppler-windows/releases)，解压后把 `Library\bin` 加入 PATH。
- **深度扫描慢**：逐条调用 LLM，9 条约 20 秒，属于串行实现。
- **torch 安装慢或体积大**：使用上面的 CPU 版安装命令。
- **向量库为空**：删除 `data/chroma_db/` 后重启，会重新索引 `data/documents/` 下的文件。

## 已知局限

- **判中偏松**：4 份合规合同的 59 个条款中有 34 条被判为中风险，模板空白和常见条款也会判中，中/高精确率因此只有 45.2%。收紧阈值可能漏掉真风险，目前选择"宁多报不漏报"。
- **条号只到"第 X 条"**：没有下探到款、项级别，报告无法精确定位到"第 X 条第 Y 款"。
- **工具层没有单次超时**：LLM 客户端已设 60 秒超时 + 2 次重试，但深度扫描是逐条串行调用，p95 仍会到 20 秒以上。
- **重排模型为英文**：`ms-marco-MiniLM-L-6-v2` 对中文是次优选择，中文场景应换用 bge-reranker 系列。
- **不支持文件改名与多租户**：文件名是检索过滤的唯一键，改名后需重新上传；没有用户或租户隔离。
- **上传与删除接口鉴权可选**：仅当配置 `UPLOAD_API_KEY` 时才校验 `X-API-Key`；限制 20MB 与 TXT/PDF/图片。生产环境应强制开启，并改为软删除。

## Roadmap

- 收紧中风险判定阈值，压掉模板空白类过检
- 深度扫描改为并发 + 限流，降低 p95 长尾
- 换用中文 reranker，并补 recall@k / MRR 指标
- 条号定位下探到款 / 项，并支持文件改名
