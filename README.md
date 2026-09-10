# 🏠 租房合同避坑助手（RAG + Agent）
上传合同 → AI 自动识别风险条款 → 生成风险报告 + 修改建议
## 环境要求
- Python 3.10 / 3.11（3.12+ 与 cnocr 不兼容）
- Windows / macOS / Linux

## 技术栈
- **Agent 编排**：LangGraph（决策 → 执行 → 回答）
- **RAG 检索**：BM25 + 向量 + RRF 融合 + CrossEncoder 重排序
- **LLM**：DeepSeek Chat
- **向量库**：ChromaDB + sentence-transformers
- **后端**：FastAPI（7 个 REST 接口）
- **前端**：Streamlit

## 核心功能
| 功能 | 说明 |
|------|------|
| 单条条款分析 | 输入一条条款，返回风险等级 + 解释 + 建议 |
| 快速预览 | 通读整份合同，给出整体风险判断 |
| 深度扫描 | 逐条拆解，生成按风险排序的完整报告 |
| 知识问答 | 基于 RAG 检索回答租房常识 |
| 合同上传 | 支持 TXT/PDF，自动解析、切分、索引 |

## 技术亮点
1. **规则 + LLM 双层审查**：硬性违规（押金>2月租、违约金>30%）规则直接拦截，模糊条款交给 LLM 深度分析
2. **Agent 参数自动补全**：大段合同文本通过 State 独立传递，避免 LLM 生成超长 JSON 报错
3. **BM25 懒加载 + 自动降级**：首次调用自动构建索引，失败自动降级纯向量检索

## 快速开始
# 安装
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# 配置（项目根目录创建 .env）
DEEPSEEK_API_KEY=sk-你的密钥
# 启动后端
cd backend/src && python main.py
# 启动前端（新终端）
cd frontend && streamlit run app.py

## 常见问题
- **首次启动慢**：加载 400MB 向量模型，约 10-15 秒
- **PDF 报错 `pdftoppm not found`**：Windows 需装 [poppler](https://github.com/oschwartz10612/poppler-windows/releases)，解压后把 `Library\bin` 加入 PATH
- **深度扫描慢**：逐条调 LLM，9 条约 20 秒
- **torch 安装慢**：先装 CPU 版 `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## 评估结果
测试集：10 份合同，覆盖 10 类典型风险
| 指标 | 数值 |
|------|------|
| 召回率 | 20/20 = **100%** |
| 高风险误报率 | 0/9 = **0%** |
| 中风险过度判断 | 5/9 = 55%（已知局限） |
> 召回率基于关键词匹配，是近似指标；中风险过度判断来自 LLM 偏保守。

**复现方式**：
cd tests
python evaluate.py             # 召回率测试
python test_false_positive.py  # 误报率测试