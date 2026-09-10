# RAG 四环节实现 + 检索不准的原因与处理

> 本文对照本项目真实代码（`app/core/rag.py`、`app/subagents/qa.py`、`app/subagents/reviewer.py`），
> 把 RAG 拆成「**切片 → 分段/索引 → 检索 → 生成**」四步讲清各自怎么实现，再单独一章讲
> **检索结果不准确**的各类原因与对应处理方法。行号以 9/4 代码为准。

---

## 〇、一图看懂全链路

```
                         ┌──────────────────────── 离线索引（启动时一次） ────────────────────────┐
                         │                                                                      │
data/knowledge_base/*.md ─► _split() 切片 ─► _tokenize() 分词                                    │
                         │        │                                                              │
                         │        ├─► BM25Retriever.ingest_directory()  建倒排(dict: token→df)   │
                         │        └─► DenseRetriever.ingest_directory()  Embedder 向量 → Qdrant   │
                         └──────────────────────────────────────────────────────────────────────┘
                                  在线查询（每次 /ask）
question ─► BM25.search()(稀疏) ┐
                                ├─► _rrf_merge() 按名次融合 ─► _maybe_rerank() cross-encoder 精排
           Dense.search()(稠密) ┘
                                   │ top_k 篇 {content, score, source}
                                   ▼
            qa.py 拼 _AUGMENT_PROMPT(参考资料+记忆+问题) ─► 模型A 生成 answer
                                   ▼
            reviewer.py 模型B 四维评审 ─► need_revision? ─► 用 improved_answer 覆盖 answer
```

---

## 一、切片（chunking）怎么实现

**代码**：`app/core/rag.py` 的 `_split()`（L41-54）＋ `_tokenize()`（L30-38）。

**做法**：**固定长度字符切 + 重叠滑动**，不是语义切。

```python
def _split(text, chunk_size=512, overlap=50):
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])   # 纯按字符切，不看段落/标题
        start = end - overlap            # 下一块往回退 overlap 个字符
```

- **`chunk_size=512`**（`RAG_CHUNK_SIZE`）、**`overlap=50`**（`RAG_CHUNK_OVERLAP`），都从 `.env` 可配。
- **为什么有 overlap**：固定切法会把一句话/一个知识点拦腰截断，overlap 让相邻两块**共享 50 个字符**，
  保证跨边界的信息至少有副本落在一块完整 chunk 里，检索时不会被「切没了」。
- **分词 `_tokenize`**：英文/数字按正则按词切（`[a-z0-9]+`），中文按**单字 + 相邻 bigram** 切
  （`[一-鿿]` 单字 + `cjk[i]+cjk[i+1]` 双字）。中文 bigram 是为了在没有分词器（jieba）的前提下，
  兼顾「单字能召回、双字能区分」的折中。

**已知局限（面试可主动讲）**：固定字符切是「零依赖、能跑」的最简实现，但会**切断语义**——
一个完整的段落/表格/代码块可能被拆到两块。改进方向是语义分块（按标题 / 按段落 / 递归字符
+ 长度阈值），或引入中文分词器把 bigram 换成更准的 token。

---

## 二、分段 / 索引怎么实现（切完怎么组织成可检索的）

「切片」决定怎么切，「分段/索引」决定切出来的 chunk 怎么变成可检索结构。分两路：

### 2.1 BM25 稀疏索引（`BM25Retriever.ingest_directory`，L119-140）

读 `.md`/`.txt` → 对每篇 `_split` 成 chunk → 对每个 chunk 分词 → 建三个统计量：

| 结构 | 含义 |
| --- | --- |
| `_chunks` | 每块的 `{content, source, tokens}` |
| `_doc_len` | 每块的「去重 token 数」（即文档长度，用于长度归一） |
| `_df` | `token → 出现在多少个 chunk 里`（倒排文档频率） |

查询时 `_bm25()`（L146-157）按经典 BM25 公式打分：`k1=1.5, b=0.75`，`idf` 用
`log(1 + (N - df + 0.5)/(df + 0.5))`，`tf` 用长度归一。**零外部依赖，任何环境能跑**——这是降级链的底。

### 2.2 稠密向量索引（`DenseRetriever.ingest_directory`，L209-233）

同一批 chunk → `Embedder.embed_texts()` 转成向量 → 写入 **Qdrant**（本地嵌入式，`qdrant_data/`）：

```python
vectors = await embedder.embed_texts([c["content"] for c in chunks])   # bge-small-zh-v1.5
self._ensure_collection(len(vectors[0]))                               # 建 collection(COSINE)
client.upsert(collection_name=..., points=[PointStruct(id=i, vector=v, payload=...)])
```

- **Embedder**（L60-100）双后端：优先 `sentence-transformers`（`BAAI/bge-small-zh-v1.5`，
  512 维，CPU），加载失败走**哈希袋兜底**（`_fallback_vec`，256 维按字符 hash 计数）——保证没装依赖也能跑通流程。
- **payload 存原文**（`content` + `source`），检索命中后直接取回原文片段，不二次读文件。

**关键设计**：BM25 和 Dense 用**同一套 `_split` 切出来的 chunk**，所以两路召回的是同一批「文档单元」，
RRF 才能按 (source, content) 对齐融合（见下）。

---

## 三、检索怎么实现（三级递进：BM25/Dense → RRF → 重排）

**代码**：`HybridRetriever`（L261-320）。

### 3.1 两路候选召回

- **BM25（稀疏）**：字面/关键词匹配强，速度极快，但对同义词、口语化措辞不敏感。
- **Dense（稠密）**：embedding 语义匹配，能召回「说法不同但意思相同」的文档，但精确词（型号、缩写、编号）可能不如 BM25。

单路各有短处 → 所以**两路都跑，各取 top_k，交给 RRF 融合**。

### 3.2 RRF 融合（`_rrf_merge`，L274-288）

```python
for rank, doc in enumerate(doc_list):
    scores[key] = scores.get(key, 0) + 1.0 / (c + rank + 1)   # c=60
```

**核心：按「名次」融合，不按「分数」**。原因——BM25 的分数是无上界统计量（实测 20+），
稠密相似度是另一套量纲（0~1），**不同检索器的分数不可直接相加**；而「名次」是普适量纲，
`1/(60+rank+1)` 把名次折成分数，两路都命中同一 chunk 时名次分累加，自然把它顶上去。

> ⚠️ 这里踩过真坑（`技术.md` 坑 6）：chunk 是 dict 不可哈希，直接当 dict key 会
> `unhashable type: 'dict'`，改用 `(source, content)` 元组作 key 去重合并。

#### 3.2.1 RRF 里的常数 k 是什么

公式里的 `k` 是**平滑常数**，本项目代码里变量名是 `c`，值 **60**。它跟「取前几条」的 `top_k`
（默认 5）是两回事——`k` 是公式里的常数，`top_k` 是返回条数。

```
RRF_score(doc) = Σ  1 / (k + rank)
```

代码里 `rank` 从 0 开始数，写 `1/(self.c + rank + 1)`（`self.c = 60`）；标准公式 `rank` 从 1 开始、
写 `1/(k + rank)`。两者**等价**，都得到第 1 名 `1/61`、第 2 名 `1/62`。所以 `k = c = 60`。

k 的两个作用：

1. **防除零**：没有 k，第 1 名（rank=0）就是 `1/0`，直接崩。
2. **平滑名次权重**：k 越大曲线越平缓（名次差距被拉近，更「民主」）；k 越小越看重第 1 名。

| k 取值 | 第 1 名 | 第 2 名 | 第 5 名 | 效果 |
|---|---|---|---|---|
| k=1（很小） | 1.0 | 0.5 | 0.2 | 名次差距放大，强看重头部 |
| k=60（默认） | 0.0164 | 0.0161 | 0.0156 | 平缓，名次差距接近 |
| k=1000（很大） | ≈0.001 | ≈0.001 | ≈0.001 | 几乎抹平名次，只看「是否出现」 |

**为什么默认 60**：60 是 RRF 原始论文（Cormack et al., 2009）的默认值，在「保留名次信息」与
「不过度敏感」之间取得平衡，已成事实标准，一般沿用不调；只有拿到评测指标（Recall@k / NDCG）才值得扫参微调。

### 3.3 cross-encoder 重排（`_maybe_rerank`，L290-304）

RRF 只做「召回 + 粗排」，top-k 里「最相关」未必排第一。cross-encoder 把 `(query, chunk)` **成对**喂进
`BAAI/bge-reranker-base`，输出 0~1 相关度，按它重排：

```python
pairs = [(query, d["content"]) for d in docs]
scores = self._ce.predict(pairs)          # 0~1 相关度
ranked = sorted(zip(docs, scores), reverse=True)
```

- 由 `RAG_CROSS_ENCODER` 开关控制：**空 = 不重排（零下载，任何环境能跑）**；填模型名 = 开启。
- 失败/未安装时 `_maybe_rerank` 原样返回，**不阻塞**（降级）。

#### 3.3.1 先分清两个层次：rerank 是「阶段」，cross-encoder 是「实现」

- **rerank（重排）是一个阶段/动作**——指「召回之后，对那一小批候选重新排序」，回答的是「**做什么**」。
- **cross-encoder 是一种模型架构**——回答的是「**用什么方法做**」。

所以：**cross-encoder 是 rerank 的一种实现，而且是最主流的一种**；但 rerank 不止它一种（见 3.3.4）。

#### 3.3.2 对照：embedding 检索那套其实是 bi-encoder（双编码器）

embedding 检索的学名是 **bi-encoder**：

```
问题 "怎么瘦肚子"  ──► Encoder ──► 向量 v_q ┐
                                          ├──► 余弦相似度 cos(v_q, v_d) = 相关度
文档 "如何减腹部脂肪" ──► Encoder ──► 向量 v_d ┘
```

关键点：**问题和文档各自独立编码、互不见面**，只在最后用余弦点积碰一下。好处是**文档能离线预编码**
（Qdrant 里存的就是这些向量），检索时只编码 query，所以能扫全库、速度快；代价是编码过程中
没有 token 级交互，只在向量级算一个粗粒度相似度。

#### 3.3.3 cross-encoder 的实现四步

cross-encoder 反其道：**把问题和文档拼成一条序列，一起喂进同一个模型**。

```
输入（一条序列）： [CLS] 怎么瘦肚子 [SEP] 如何减腹部脂肪 [SEP]
                        │
                        ▼
              Transformer（self-attention）
                        │
                        ▼
              取 [CLS] 位置的向量
                        │
                        ▼
              线性层 W → 一个标量 logit
                        │
                        ▼
              sigmoid(logit) ──► 0~1 的相关度分数
```

1. **拼接**：`[CLS] query [SEP] document [SEP]`，用分隔符串成一条输入。
2. **联合编码**：整条序列一起进 transformer；self-attention 全对全，**query 的每个 token 都能看到
   document 的每个 token**——这就是「交互计算」。
3. **取表征**：取 `[CLS]` 位置（句子开头特殊 token）的向量，它聚合了整条序列的信息。
4. **打分**：向量过线性层 `W` 变标量，再过 `sigmoid` 压到 0~1。

对应代码（`app/core/rag.py` `_maybe_rerank`）：

```python
self._ce = CrossEncoder("BAAI/bge-reranker-base")      # 加载精排模型
pairs = [(query, d["content"]) for d in docs]           # 组 (问题, 文档) 对
scores = self._ce.predict(pairs)                        # 每对算一个 0~1 相关度
ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)  # 降序重排
```

`bge-reranker-base` 底层是 `xlm-roberta-base`，训练目标是「相关/不相关」二分类，所以 logit 过 sigmoid
天然是 0~1 的「相关概率」（实测 `0.999`/`0.9985` 就是这么来的）。

**为什么更精细**：

| | bi-encoder（embedding） | cross-encoder |
|---|---|---|
| 交互时机 | 只在**最后**点积那一下 | **每一层** attention 都在交互 |
| 交互粒度 | 向量级（整条向量比大小） | token 级（query 每个词能对应 doc 每个词） |
| 能捕捉 | 「整体像不像」 | 「query 这个词对应 doc 哪个词」的细节 |
| 能否预计算 | ✅ 文档可离线编码 | ❌ 每次都要把 pair 一起过一遍 |

#### 3.3.4 为什么它只能精排、不能召回（+ rerank 的其它实现）

**代价**：query 和 document 要拼一起、现场编码，**没法预先算好向量存起来**，且每判断一个 pair 都要跑
一遍完整 transformer，**慢**。所以它无法对全库逐个打分，只能对「已召回的一小批候选」精排——这正是
「先召回、后重排」架构的根本原因：

```
BM25 + 稠密(快、能扫全库) ──召回──► top-k 候选 ──► cross-encoder(慢、只精排这几个) ──► 最终顺序
```

召回负责「**全**」和「**快**」，重排负责「**准**」。

**rerank 不止 cross-encoder 一种**，其它实现方式：

- **LLM 重排**（如 RankGPT）：把候选列表交给大模型，让它直接给出排序；
- **listwise 重排**：一次把整个候选列表喂给模型一起排序（bge-reranker 也支持，比 pairwise 更快）；
- **特征加权重排**：不靠模型，用「BM25 分数 × 权重 + 向量分数 × 权重 + 新鲜度 + 权威度」线性融合重排
  （本项目 RRF 也可看成一种最简单的「基于名次的重排」）。

> 面试精确表述：**rerank 是检索流程的一个阶段，cross-encoder 是这个阶段最主流的一种实现。**
> 我用 cross-encoder（bge-reranker-base）做 pairwise 精排：把 (问题, 文档) 拼一条序列联合编码，
> 靠 token 级交互算出 0~1 相关度，再按它重排召回出来的 top-k。

### 3.4 三级递进的效果证据（9/4 实测）

`LangGraph 是什么？` 三档 top 来源：

| 档 | 结果 | 评价 |
| --- | --- | --- |
| 只 BM25 | `agent_tools.md` | ❌ 答非所问（"LangGraph" 是稀见词，被其它 token 干扰） |
| hybrid(RRF) | `rag_chunking.md` / `langgraph.md` / `agent_tools.md` | ⚠️ 召回对了但没排第一 |
| hybrid + 重排 | `langgraph.md(0.999)` | ✅ 精排把它顶到第一 |

这正好演示了「**单路会丢 → RRF 补召回 → cross-encoder 提精度**」。

### 3.5 工厂降级链（`build_retriever`，L330-358）

```
Hybrid(稠密+BM25) ──稠密索引失败/embedding不可用──► 降为 BM25 混合 ──构建再失败──► 纯 BM25
```

`use_dense_retrieval` 可整体关稠密（`.env` 的 `USE_DENSE_RETRIEVAL`）。

---

## 四、生成怎么实现（检索之后，问答子 Agent + 评审子 Agent）

### 4.1 上下文增强生成（`app/subagents/qa.py`）

1. **检索**：`docs = await retriever.search(question, settings.top_k)`（L61）。
2. **拼提示词**：`_AUGMENT_PROMPT`（L20-35）把三样东西塞进 prompt——**参考资料**（检索到的 chunk，带【来源】标注）、
   **长期记忆**（用户偏好/历史，仅作补充）、**用户问题**。
3. **约束生成**：prompt 里硬性要求「严格基于参考资料、不要编造；资料中没有就回答『根据现有资料无法回答』；
   引用标注【来源】」——这是**抗幻觉的第一道闸**。
4. **降级**（L74-76）：模型A 不可用 → 退回 `[降级] 模型不可用，以下为检索到的相关资料片段…`，
   **检索、来源标注这些没坏的能力照样保留**。

模型调用走 `app/core/llm.py` 的 `ChatOpenAI`（OpenAI 兼容协议，`.env` 切 base_url/name 即可换 DeepSeek/Ollama/vLLM）。

### 4.2 独立模型B 评审 + 修订（`app/subagents/reviewer.py`）

生成不是终点，还过一道**独立评审**：

- **四维打分**（`_REVIEW_PROMPT`，L21-41）：准确性 / 完整性 / 安全性 / 相关性，各 0-10。
- **触发修订**：`need_revision=true 且给了 improved_answer` → 用改进版覆盖模型A 原始回答（L99-100）。
- **双模型物理隔离**：评审用 `MODEL_B`（本项目 A=pro 生成 / B=flash 评审），**不让模型自己评自己**
  （HELLO「双模型物理同源」待优化点的落地）。
- **稳健解析**（`_parse_review`，L52-72）：模型输出带 Markdown 代码块/多余文字也能抠出 JSON；解析失败走 `_fallback`。
- **降级**：模型B 不可用 → 保守处理（`need_revision=false`、保留 A 原答案），不退化成坏结果。

> 生成环节的「抗幻觉」是**双保险**：prompt 约束（生成端）＋ 独立评审（后置兜底）。
> 实测 `什么是向量数据库？`（知识库空）时，模型A 诚实答「无法回答」→ 模型B 打 4 分并补出改进版。

---

## 五、检索结果不准确：原因 → 处理（重点）

按「**检索不准发生在哪一环**」归类，每类给出**现象、根因、本项目已做的处理、可进一步优化**。

### 5.1 切片环节的问题

| 现象 | 根因 | 已做处理 | 可优化 |
| --- | --- | --- | --- |
| 命中 chunk 但关键信息被「切没了」 | 固定字符切，把一句话/表格拦腰截断 | `overlap=50` 让跨边界信息有重叠副本 | 语义分块：按标题/段落切；用 jieba 分词替代 bigram |
| 命中 chunk 含太多无关内容、稀释了答案 | `chunk_size=512` 偏大，噪声多 | 调 `RAG_CHUNK_SIZE`（`.env` 可配） | 按问题粒度评估最优 chunk_size（用 Day12 的 Recall@k 扫参） |

### 5.2 关键词/字面不匹配（BM25 召回不到）

| 现象 | 根因 | 已做处理 | 可优化 |
| --- | --- | --- | --- |
| 同义词、口语化问法召回为空或偏 | BM25 只认字面 token，不懂语义 | **稠密检索补召回**（embedding 语义匹配）+ RRF 融合 | 换更强的中文 embedding（`bge-large-zh`）；加 query 改写（HyDE/多查询） |

### 5.3 单路丢信息（召回不全）

| 现象 | 根因 | 已做处理 | 可优化 |
| --- | --- | --- | --- |
| 稀疏漏语义、稠密漏精确词（型号/缩写） | 单路各自短处 | **hybrid 双路 RRF 按名次融合**，互补 | 加第三路（如向量全文检索）；调 RRF 常数 `c` |

### 5.4 召回了但排序不精（对的没排前面）

| 现象 | 根因 | 已做处理 | 可优化 |
| --- | --- | --- | --- |
| 相关文档在 top-k 里但排第 2/3，模型看到的是「次相关」 | RRF 只做粗排，分数是名次倒数、区分度低 | **cross-encoder 精排**（`_maybe_rerank`），0~1 相关度重排 | 重排后再用 top_k 截断（当前 `_maybe_rerank` 重排的是已截断的 top_k，可先多召回再精排截断） |

### 5.5 embedding / 向量库不可用

| 现象 | 根因 | 已做处理 | 可优化 |
| --- | --- | --- | --- |
| 稠密检索直接失效 | 没装 sentence-transformers / 下载超时 / Qdrant 冲突 | `Embedder.lazy_load()` 失败走哈希袋兜底；`build_retriever` 捕获异常降级 BM25；`use_dense_retrieval=false` 可关 | 兜底向量从「哈希袋」换「TF-IDF 向量」更准 |

### 5.6 知识库本身缺料（检索为空）

| 现象 | 根因 | 已做处理 | 可优化 |
| --- | --- | --- | --- |
| 问的领域文档库里根本没有 | 知识库覆盖不足 | prompt 约束「根据现有资料无法回答」，**诚实降级不编造**；reviewer 低分兜底 | 扩充知识库；接入工具子 Agent 联网搜索补料（Day9） |

### 5.7 生成幻觉（检索到了但答得不对/编造）

| 现象 | 根因 | 已做处理 | 可优化 |
| --- | --- | --- | --- |
| 答案脱离检索资料、凭空编造 | 模型不忠实于上下文 | prompt「严格基于参考资料、引用【来源】」；**模型B 四维评审（accuracy 维度）+ 触发修订** | 检索+生成用 RAGAS 的 Faithfulness/Groundedness 量化（Day12） |

### 5.8 结果去重 / 呈现问题

| 现象 | 根因 | 已做处理 | 可优化 |
| --- | --- | --- | --- |
| `sources` 里同一文档出现多次（如两个 `langgraph.md`） | 同一文档被切成多 chunk 且都命中 | 未处理（已知遗留） | 按 `source` 去重展示；或 RRF 里同 source 只保留最高名次 |

### 5.9 融合实现本身的坑（已修，防复发）

| 现象 | 根因 | 处理 |
| --- | --- | --- |
| `unhashable type: 'dict'` | chunk 是 dict 直接当 key | 改用 `(source, content)` 元组（坑 6，已修） |
| 分数不可比导致融合偏差 | 不同检索器量纲不同 | RRF 用名次倒数统一量纲 |

---

## 六、面试一句话（背下来）

> 检索我做了**三级递进**：单路 BM25 / 稠密都会丢信息，所以用 **RRF 按名次融合**两路
> （不同分数不可比、名次可比），再用 **cross-encoder 精排**把最相关的排前面；
> 生成端用**严格 prompt + 独立模型评审**双保险抗幻觉。任何一路挂了都降级，任何环境能跑出 demo。
>
> 检索不准我按**环节定位**：切分问题调 chunk/overlap、字面不匹配上稠密、排序不精上重排、
> 缺料则诚实降级不编造——每种都能给出现象→根因→处理。

---

## 附：关键代码位置速查

| 环节 | 函数/类 | 位置 |
| --- | --- | --- |
| 切片 | `_split` / `_tokenize` | `app/core/rag.py:41-54` / `:30-38` |
| BM25 索引+打分 | `BM25Retriever` | `app/core/rag.py:106-175` |
| 稠密索引+检索 | `DenseRetriever` | `app/core/rag.py:181-255` |
| Embedder（双后端+哈希兜底） | `Embedder` | `app/core/rag.py:60-100` |
| RRF 融合 | `HybridRetriever._rrf_merge` | `app/core/rag.py:274-288` |
| cross-encoder 重排 | `_maybe_rerank` | `app/core/rag.py:290-304` |
| 工厂降级链 | `build_retriever` | `app/core/rag.py:330-358` |
| 上下文增强生成 | `_AUGMENT_PROMPT` / `qa.run` | `app/subagents/qa.py:20-35` / `:53-86` |
| 评审+修订 | `reviewer.run` / `_parse_review` | `app/subagents/reviewer.py:75-114` / `:52-72` |
| 模型调用封装 | `LLMClient` / `get_chat_model` | `app/core/llm.py:56-99` / `:35-53` |
