# multi-agent — 带长期记忆的多智能体助手（星屋 HomeStar 售后客服）

> 项目2：在 HELLO（双模型 RAG）基础上，升级为**带长期记忆 + 多智能体编排 + 自主工具调用**的垂直助手（售后客服）。
> 核心叙事：**复盘 HELLO 的 20+ 待优化点，在这个项目里全部落地修复**——证明"会复盘、会迭代、有工程成长曲线"。
> 业务场景（9/5 起）：**智能家居「星屋 HomeStar」售后客服**——换知识库 + 换工具，架构零改动。

**技术栈**：FastAPI · LangChain · LangGraph · Qdrant · SQLite · sentence-transformers · **双模型（物理隔离）** · slowapi（限流）

> 📐 设计分析与踩坑 1~10 见 [`技术.md`](./技术.md) · 检索调优全过程见 [`RAG实现与检索不准处理.md`](./RAG实现与检索不准处理.md) · 选型调研见 [`客服售后多智能体前沿技术调研.md`](./客服售后多智能体前沿技术调研.md)。

---

## ✅ 当前状态（2026-09-08）

Day 1 ~ Day 13 已完成，`pytest tests/ -q` = **54 passed**。已落地并真机验证：

- **三层记忆**：working（checkpointer 跨轮累积）· episodic（Qdrant 向量 + TTL + importance 加权）· semantic（SQLite 去重 + 矛盾合并）· 固化把 episodic 提炼成 semantic。
- **跨用户隔离（P0 修复）**：episodic 检索按 `thread_id` 过滤，user-2 不再串 user-1（真机验证暴露并修复）。
- **售后工具**：`query_order` / `query_logistics` / `apply_refund` 三个 mock 工具 + 白名单防注入 + 意图路由 + Function Calling（加分）。
- **反思子 Agent**：物理隔离的模型 B 四维评审（正向出分 + 反向改错 key 会 401 证明真走 B）+ 稳健解析 + 失败重试。
- **工程健壮性**：熔断器（三态 + 半开探活）+ slowapi 限流。
- **知识库**：售后 11 篇（7 手写 FAQ/流程 + 4 法规长文），Qdrant `knowledge_base` 集合 42 chunk。
- **量化评测（Day 12）**：自建 46 条售后评测集，产出四张对比表——重复重述 ↓50.3%、token ↓20.9%（尾部单轮 ↓51.6%）、混合检索 NDCG@3 最优、意图分流 95% + 自助解决率 87%。详见 [第八节](#八评测量化指标day-12)。
- **投诉→转人工（Day 13）**：评测暴露投诉类未转人工，补 `resolve_escalation` + `escalate` 节点，投诉正确转人工（转人工识别 3/3）。
- **Docker 化（Day 13）**：多阶段构建 + CPU 版 torch 瘦身 + healthcheck，`Dockerfile` + `docker-compose.yml`（目标镜像 < 3GB，对比 HELLO 8.6GB）。

---

## 一、启动（从零到跑通）

```bash
cd multi-agent

# 1) 装依赖（Python 3.11+）
pip install -r requirements.txt
pip install sentence-transformers        # 重依赖，requirements 里注释掉，按需单独装

# 2) 配置：复制模板，填入自己的 key
#    .env 已在 .gitignore 中，不会入库；仓库里只有留空的 .env.example
cp .env.example .env                     # Windows: copy .env.example .env

# 3) 启动
PYTHONPATH="." python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

> 模型可换：`MODEL_A_*` / `MODEL_B_*` 分属两套独立配置，任何 OpenAI 兼容端点都能接
> （示例用 deepseek；`MODEL_B` 建议指向与 A 不同的模型——反思评审不该「自己评自己」）。
> 本地跑 Ollama 时把 `*_BASE_URL` 指过去即可，key 留空。

验证：

- `GET /api/v1/health` → `{"status":"ok",...}`
- `GET /api/v1/status` → `compiled=true`、`retriever_mode=hybrid`
- `POST /api/v1/ask {"question":"查一下我的订单 A123"}` → 返回订单状态（工具路径）
- `POST /api/v1/ask {"question":"门锁保修多久"}` → 知识库 + 记忆 + 评审出分

---

## 二、接口

| 方法   | 路径                            | 说明                                                                    |
| ---- | ----------------------------- | --------------------------------------------------------------------- |
| GET  | `/api/v1/health`              | 健康检查                                                                  |
| GET  | `/api/v1/status`              | graph 编译状态 + 检索模式                                                     |
| POST | `/api/v1/ask`                 | 问答（body: `question`、可选 `thread_id`、`streaming`），**限流 `30/minute`/IP** |
| POST | `/api/v1/ask`（streaming=true） | SSE 流式（含 `review` 结构化事件）                                              |
| POST | `/api/v1/memory/upsert`       | 写入语义记忆（`key`/`value`/`source`）                                        |
| POST | `/api/v1/memory/consolidate`  | 触发记忆固化                                                                |
| GET  | `/api/v1/memory`              | 列出语义记忆                                                                |

---

## 三、架构

```
用户问题
   ↓
[LangGraph 主图]   AgentState（共享状态 = 团队/协调记忆，checkpointer 按 thread_id 持久化）
   ├─ memory   记忆子Agent：检索长期记忆（按 thread_id 隔离）→ 注入上下文
   ├─ route    意图路由（售后：查订单/查物流/退款 → tools，否则 qa）
   ├─ qa       知识问答子Agent：混合检索(BM25+向量→RRF→重排) → 模型A生成
   ├─ tools    工具子Agent：白名单调度 + 售后 mock 工具 + Function Calling(加分)
   ├─ safety   安全检查（命中则拦截）
   ├─ review   反思子Agent：独立模型B四维评审（工具路径跳过，评审只管生成式答案）
   └─ record   写本轮问答到情景记忆（带 thread_id，供后台固化）

工程底座：FastAPI · Qdrant · SQLite · 熔断器(三态) · slowapi 限流 · request_id 日志 · 优雅降级
```

**两层记忆（痛点④「多智能体记忆孤岛」的解法）**：

- **团队/协调记忆**：LangGraph 的 `AgentState`（共享状态）——各子 Agent 读写同一份任务状态，不各说各话。
- **用户长期记忆**：`MemoryStore`（episodic 向量化 / semantic SQLite）——跨会话沉淀用户事实/偏好，**按 `thread_id` 隔离**。

**记忆分层**：`working`（checkpointer）· `episodic`（Qdrant 向量 + TTL + importance 加权）· `semantic`（SQLite 用户偏好）· `consolidator`（固化：摘要 + 去重 + 矛盾合并）。

---

## 四、目录结构

```
multi-agent/
├── app/
│   ├── main.py              # FastAPI + lifespan(预热) + 限流挂载 + 统一异常
│   ├── config.py            # Pydantic Settings（MODEL_A/B 物理隔离 + 白名单 + 熔断/限流参数）
│   ├── api/routes.py        # /health /status /ask(限流) /memory
│   ├── models/schemas.py    # AskRequest/AskResponse ...
│   ├── core/
│   │   ├── graph.py         # LangGraph 主图（State + 7 节点 + checkpointer + 流式 review 事件）
│   │   ├── llm.py           # ChatOpenAI 封装（熔断接入 + 超时 + 降级）
│   │   ├── circuit_breaker.py # 熔断器（三态 + 冷却 + 半开试探）
│   │   ├── rate_limit.py    # slowapi 限流（按 IP）
│   │   ├── rag.py           # 混合检索（BM25+向量→RRF→重排，降级）
│   │   ├── services.py      # 服务容器（共享检索器/模型/记忆/安全）
│   │   ├── qdrant.py        # 全局唯一 Qdrant 客户端
│   │   └── observability.py # request_id 结构化日志
│   ├── memory/              # 记忆系统（核心卖点）
│   │   ├── models.py        # MemoryItem/MemoryType
│   │   ├── working.py       # 工作记忆（checkpointer 截断）
│   │   ├── episodic.py      # 情景记忆（Qdrant 向量 + thread_id 隔离 + TTL）
│   │   ├── semantic.py      # 语义记忆（SQLite 去重/合并）
│   │   └── consolidator.py  # 固化（摘要 + 去重 + 合并，降级规则）
│   ├── subagents/           # memory_agent / qa / tools / reviewer
│   └── tools/safety_tool.py # 复用 HELLO 安全检查
├── data/knowledge_base/     # 售后知识库 11 篇（42 chunk）
├── data/knowledge_base_tech/ # 技术文档 9 篇（已隔离）
├── evaluation/              # Day12 量化评测（eval_dataset / metrics / run_eval）
├── data/eval/               # 评测集 + 结果（results.json / results_summary.md）
├── tests/                   # 9 个测试文件（54 passed）
├── scripts/                 # 真机验证脚本（verify_day8~11 等）
├── Dockerfile               # 多阶段构建 + CPU torch 瘦身 + healthcheck（Day 13）
├── docker-compose.yml       # app + 可选 Qdrant 服务端（Day 13）
├── .dockerignore            # 排除 .env/数据/日志
├── .env.example             # 配置模板（留空）；真实 .env 已被 .gitignore 排除
├── requirements.txt
├── 技术.md                  # 设计分析 + 踩坑 1~10
├── RAG实现与检索不准处理.md   # 检索调优全过程
└── 客服售后多智能体前沿技术调研.md
```

---

## 五、面试叙事（背下来）

> "我第二个项目是**带长期记忆的多智能体售后客服**。在 HELLO 双模型 RAG 基础上，复盘出 3 个核心缺陷——没有 Function Calling、双模型物理同源、检索只是'稠密 OR 稀疏'降级——全部落地修复：
> ① LangGraph 编排 7 节点，主控路由到记忆/问答/工具/反思四个子 Agent；② 记忆三层（working 靠 checkpointer、episodic 向量化 + TTL + importance 加权、semantic 结构化去重），再加固化把 episodic 提炼成 semantic；③ 检索升级 BM25+向量混合 + RRF + 重排；④ 反思用物理隔离的独立模型 B。
> 真起服务、跨用户提问时，我发现并修了一个 P0——长期记忆没按用户隔离（user-2 会串到 user-1 的隐私），用 Qdrant payload filter 补上 thread_id 这把钥匙。工具侧做白名单 + 参数校验防 prompt 注入，还发现'工具结果不该再过 LLM 评审'。工程上补了熔断器（三态）和限流，都是'能上线'不是 demo。
> 整个过程踩了 10 个真坑（同步 SqliteSaver、Qdrant 单客户端、PYTHONHASHSEED、推理模型输出偶发不稳定……），都定位根因改掉了。"

---

## 六、风险 / 降级（面试能主动讲）

1. **记忆垃圾抽屉** → 固化去重/矛盾合并（`consolidator`）+ TTL 遗忘。
2. **多用户隐私** → episodic 按 `thread_id` 隔离（P0 修复）。
3. **多智能体过拟合** → "能单不双"，讲清"为什么拆、为什么有些不拆"。
4. **外部依赖抖动** → 熔断器快速失败 + 降级兜底 + 限流防滥用。
5. **环境降级** → 模型不可用走降级（TF-IDF、无评审、资料片段），保证任何环境能跑。

---

---

## 八、评测：量化指标（Day 12）

> 「有量化 vs 没量化，是项目 vs 作业的分界线」。自建 **46 条售后评测集**（15 检索 + 23 意图 + 8 续接场景），
> **全部离线可复现**（不调外部 LLM API，不花 token）。一键重跑：
> 
> ```bash
> PYTHONPATH="." python evaluation/run_eval.py
> ```
> 
> 结果落在 `data/eval/results.json`（全量）+ `results_summary.md`（对比表）。指标定义见 `evaluation/metrics.py`。

### 8.1 跨会话重复重述下降 ↓50.3%（有记忆 vs 无记忆）

8 组「首轮建立背景 → 二轮再问」的续接场景，对比用户二轮**输入长度**：

| 场景     | 指代式二轮(有记忆) | 需重述(无记忆)   | 下降率       |
| ------ | ---------- | ---------- | --------- |
| S01    | 12 字       | 27 字       | 55.6%     |
| S02    | 17 字       | 32 字       | 46.9%     |
| S03    | 10 字       | 19 字       | 47.4%     |
| S04    | 8 字        | 20 字       | 60.0%     |
| S05    | 9 字        | 20 字       | 55.0%     |
| S06    | 12 字       | 19 字       | 36.8%     |
| S07    | 10 字       | 24 字       | 58.3%     |
| S08    | 12 字       | 20 字       | 40.0%     |
| **平均** | **11.2 字** | **22.6 字** | **50.3%** |

> 支撑证据：指代式二轮（如「那台门锁」）被长期记忆按 `thread_id` 命中 **8/8（100%）**——短输入之所以成立，是因为记忆真的把上一轮「门锁 A100」捞回来了。

### 8.2 token 成本下降 ↓20.9%（累计）/ ↓51.6%（第 30 轮单轮）

模拟 **30 轮**售后会话，对比「全量塞历史」vs「记忆检索+摘要」的累计 prompt token：

| 策略      | 累计 prompt token | 节省        |
| ------- | --------------- | --------- |
| 全量塞历史   | 21524           | —         |
| 记忆检索+摘要 | 17015           | **20.9%** |

> 全量塞历史累计 O(轮数²)；记忆检索靠「工作记忆截断(keep=20) + 情景记忆 top-k」有界。
> 第 30 轮单轮 prompt：全量 1185 vs 记忆 573 token，**单轮节省 51.6%**——会话越长收益越大。

### 8.3 检索 Recall@k / NDCG@k（文档级，BM25 vs 稠密 vs 混合）

| 指标       | bm25    | dense  | hybrid     |
| -------- | ------- | ------ | ---------- |
| recall@1 | 0.9333  | 0.7333 | 0.9333     |
| recall@3 | 0.9333  | 0.9667 | 0.9667     |
| recall@5 | **1.0** | 0.9667 | 0.9667     |
| ndcg@3   | 0.9484  | 0.9004 | **0.9742** |
| ndcg@5   | 0.9918  | 0.9004 | 0.9742     |

> **解读（数字自己会说话）**：① 售后知识库是**关键词密集**的短文档（「E5」「A100」「保修」），所以 BM25 召回最全（R@5=1.0）；
> ② 纯稠密对专有名词/型号不敏感，Recall@1 掉到 0.73；③ **混合(RRF) 把 NDCG@3 拉到 0.974（三者最高）**，
> 即「排序质量最好」——最相关的政策排得更靠前，代价是 R@5 略降（某个多相关查询的次相关文档被挤出 top-5）。
> 这正是「为什么要混合」的量化证据：不是单一指标碾压，而是排序质量与召回完整性的权衡。

### 8.4 意图分流准确率 95% + 自助解决率 87%

23 条售后意图（咨询/退换/物流/订单/投诉）跑 `resolve_tool_call`：

| 意图         | 正确/总数 | 准确率       |
| ---------- | ----- | --------- |
| 咨询         | 6/7   | 85.7%     |
| 退换         | 5/5   | 100.0%    |
| 物流         | 4/4   | 100.0%    |
| 订单         | 4/4   | 100.0%    |
| 投诉         | 3/3   | 100.0%    |
| **四类自助汇总** | —     | **95.0%** |

- **自助解决率**（适合自助、无需人工）：**87.0%**
- **投诉转人工识别**：**3/3（100%）**——Day 13 补上「投诉→转人工」路由后，投诉类正确转人工

> **解读**：唯一分错的是「七天无理由退货是什么政策？」——咨询被「退货」关键词误判为「申请退货」，暴露启发式关键词路由的边界（Function Calling 可补）。
> 投诉类原会被当成普通咨询自助回答，Day 13 已补「投诉→转人工」路由（`resolve_escalation` + `escalate` 节点），现在投诉正确转人工。

---

## 九、Docker 部署（Day 13）

> 三个瘦身点：**多阶段构建**、**CPU 版 torch**（不拉 CUDA 全家桶）、**healthcheck 探活**。

```bash
# 构建 + 启动（目标镜像 < 3GB，对比 HELLO 用 CUDA torch 的 8.6GB）
docker compose up --build

# 验证
curl http://127.0.0.1:8000/api/v1/health   # {"status":"ok",...}
```

- `Dockerfile`：Stage1 `builder` 装依赖（先 `pip install torch --index-url https://download.pytorch.org/whl/cpu` 装 CPU 版 torch，再装项目依赖 + sentence-transformers）；Stage2 `runtime` 只拷贝产物 + 代码，`HEALTHCHECK` 打 `/api/v1/health`。
- `docker-compose.yml`：app 只读挂载知识库 + `state` 命名卷（记忆库/向量库持久化，避免单文件挂载踩 SQLite WAL 锁）；模型 key 走 `${MODEL_A_API_KEY:?}` 从 `.env` 注入，**不打进镜像**；`qdrant` 服务端用 `profile` 默认关闭（本地嵌入式够用，多实例共享才启用）。
- `.dockerignore`：排除 `.env`（秘密不 bake 进镜像）、数据目录、日志、缓存。

**为什么这么设计（面试点）**：① CPU 版 torch——本服务是「检索 + 生成编排」，I/O 密集 + 少量小模型推理，CPU 足够，省掉 ~2GB CUDA；② 多阶段把构建期依赖/缓存留在 builder，runtime 只带产物；③ healthcheck 让编排系统能探活自愈——这是「能上线」不是 demo。
