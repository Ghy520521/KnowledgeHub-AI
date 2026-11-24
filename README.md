# KnowledgeHub AI — 企业知识库与混合检索平台

面向企业产品说明书、客服 FAQ、售后规则、技术文档的 RAG 服务：同时做**向量语义检索**与**关键词全文检索**，用 **RRF** 融合排序，再经 **Reranker 精排**，最终回答**带引用来源**，检索证据不足时**拒答**而不是让模型编造。

- 定位：Python 后端 / RAG 工程 / AI 应用后端
- 核心链路：文档上传 → 解析 → 递归分块 → Embedding 向量化 → 向量+全文双路召回 → RRF 融合 → Reranker 精排 → 证据门控 → 带引用生成

## 1. 技术栈

| 层 | 选型 |
|---|---|
| Web 框架 | FastAPI + Pydantic v2 + Uvicorn |
| ORM / 迁移 | SQLAlchemy 2.0（开发期 `create_all`，生产可接 Alembic） |
| 数据库 | PostgreSQL 16 + pgvector（向量）+ 全文检索（`to_tsvector`/`plainto_tsquery`） |
| 异步队列 | Celery + Redis（文档解析/向量化异步执行） |
| 模型适配 | OpenAI 兼容 Embedding / Reranker / Chat 接口，均可切换离线 fake |
| 测试 | pytest |
| 部署 | Docker Compose（postgres / redis / api / worker） |

> 第一版只用一个数据库：业务数据、文档分块、向量、全文索引都落在 PostgreSQL，降低部署复杂度。

## 2. 快速开始

### 方式 A：Docker Compose（推荐，含 PostgreSQL/pgvector/Redis）

```bash
cp .env.example .env
docker compose up --build
```

- API 文档：http://localhost:8000/docs
- 健康检查：`GET /health`

### 方式 B：本地离线演示（无需 PostgreSQL/Redis/外部模型 Key）

仓库自带 `.env`，默认 `EMBEDDING_PROVIDER=fake`、`RERANKER_PROVIDER=fallback`、`LLM_PROVIDER=fake`、SQLite 回退。

Windows PowerShell 建虚拟环境并安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

VS Code 打开文件夹后，按 `Ctrl+Shift+P` → "Python: Select Interpreter" → 选 `.venv\Scripts\python.exe`（仓库已带 `.vscode/settings.json` 自动指向），导入报红即消失。

```powershell
# 写入示例文档并跑一次混合检索
python -m scripts.seed_demo
# 启动 API，打开 http://localhost:8000/docs
uvicorn app.main:app --reload
```

> 兼容 Python 3.10+；接真实 OpenAI 兼容模型时，在 `.env` 切换 provider 并填 Key。

## 3. API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/auth/register` | 注册（首个用户自动为 admin） |
| POST | `/auth/login` | 账号密码登录，签发 JWT |
| POST | `/documents/upload` | 上传 txt/md/pdf/docx，触发摄入 |
| GET | `/documents` | 文档列表（含摄入状态/失败原因） |
| GET | `/documents/{id}` | 文档详情 |
| DELETE | `/documents/{id}` | 删除文档及其分块 |
| POST | `/search` | 混合/向量/关键词检索，返回排序列表 |
| POST | `/ask` | 检索 → 精排 → 证据门控 → 带引用回答 |

示例：

```bash
# 登录拿 token
curl -X POST http://localhost:8000/auth/login \
  -d "username=admin&password=123456"

# 混合检索
curl -X POST http://localhost:8000/search \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "耳机断连怎么处理", "strategy": "hybrid"}'

# 带引用问答
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "最近退款多久到账？"}'
```

`/ask` 返回结构：`answer`、`refused`（是否拒答）、`citations`（分块 ID、文档、页码、片段）、`used_strategy`。

## 4. 核心设计

### 4.1 混合检索 + RRF 融合

- **向量检索**：语义相近即可命中，适合"意思差不多"的问题。
- **关键词检索**：PostgreSQL 全文检索，擅长错误码、型号、专有名词等精确词面匹配。
- **RRF（Reciprocal Rank Fusion）**：

```
score(c) = Σ_r 1 / (k + rank_r(c)),  k=60
```

不同检索器的原始分数量纲不同（余弦距离 ∈ [0,1]，ts_rank 无界），**直接相加没有意义**；RRF 只融合排名、不融合分数，对量纲不敏感，实现简单且鲁棒。

### 4.2 Reranker 精排

召回阶段追求"不漏"（`vector_top_k=20`、`fts_top_k=20`），精排阶段追求"准"：把查询与候选文本交给重排模型（兼容 `/rerank` 接口，如 Jina/Cohere 风格）重新打分，取 `RERANK_TOP_N=10` 交给生成模型。无外部密钥时使用本地 `词面重叠 + RRF 分` 的轻量回退。

### 4.3 答案引用与拒答

- 分块保存 `document_id / chunk_index / page / content`，回答时逐条返回引用来源。
- **证据门控**：重排后最高分低于 `MIN_EVIDENCE_SCORE` 时，直接返回"证据不足"，不调用模型硬答，抑制幻觉。
- 系统提示词约束模型只能依据参考资料作答。

### 4.4 检索离线评测

`Recall@K`、`MRR`、`nDCG@K` 三个指标，对比 `keyword / vector / hybrid / hybrid+rerank` 四条策略，并记录平均检索延迟：

```bash
python -m scripts.run_eval --set data/eval/eval_set.jsonl --k 5 --out reports/eval.json
```

> 简历上的提升百分比必须来自自己的实验，禁止预先编造。

## 5. 项目结构

```
knowledgehub-ai/
├── app/
│   ├── main.py                # FastAPI 入口、CORS、启动建表
│   ├── config.py              # pydantic-settings 集中配置
│   ├── database.py            # 引擎/会话/Base
│   ├── models.py              # User/Document/Chunk/RetrievalLog
│   ├── security.py           # 密码哈希、JWT
│   ├── deps.py               # 当前用户/管理员依赖
│   ├── api/                   # auth / documents / retrieval 路由
│   ├── ingestion/             # parser / chunker / embeddings / pipeline
│   ├── retrieval/             # hybrid / rrf / reranker / generator / evaluation
│   └── tasks.py              # Celery 应用与摄入任务
├── scripts/                  # seed_demo.py / run_eval.py
├── data/eval/                # 标注评测集（JSONL）
├── tests/                     # pytest 用例
├── Dockerfile
└── docker-compose.yml
```

## 6. 工程化要点（面试可讲）

1. **为什么用 RRF 而不是分数加权**：量纲不同，加权权重难调且不稳定。
2. **为什么先召回再精排**：召回保覆盖、精排保准确，兼顾效果与成本。
3. **为什么需要拒答门控**：RAG 最大风险是幻觉，证据不足要敢于说不知道。
4. **Celery 的价值**：解析 PDF、批量向量化是耗时操作，不应阻塞 HTTP 请求。
5. **可替换的模型适配层**：Embedding/Reranker/Chat 都走适配器协议，切换供应商不改业务代码；离线 fake 保证本地可开发、可测试。

## 7. 已知限制与生产 TODO

- 原始文件暂存本地临时目录，生产应接对象存储（S3/MinIO）并记录元数据。
- 中文全文检索目前走 `simple` 配置，生产可接入 `zhparser`/`pg_trgm` 提升中文词面匹配。
- 生产数据库迁移应引入 Alembic，替换开发期 `create_all`。
- 可增加：检索结果反馈点赞/点踩、多轮对话上下文压缩、限流与审计日志。
