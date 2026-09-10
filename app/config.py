"""集中配置 — Pydantic Settings 统一管理环境变量。

要点（来自 HELLO 复盘 + 本项目落地）：
1. `protected_namespaces=()` 关闭 Pydantic 保留前缀冲突，否则 `model_a_name` 这类字段
   会撞 Pydantic 的 `model_` 命名空间报错。
2. 模型A（生成）/ 模型B（反思评审）字段完全分离 —— 「双模型物理隔离」待优化点落地。
3. embedding 双后端（sentence-transformers / ollama），与 HELLO 对称，任一失败走 TF-IDF 降级。
"""
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # ---- 应用基本信息 ----
    app_name: str = "multi-agent 记忆助手"
    app_version: str = "0.2.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # ---- 模型A（生成）----
    model_a_name: str = "deepseek-chat"
    model_a_base_url: str = ""
    model_a_api_key: str = ""

    # ---- 模型B（反思/评审）—— 物理隔离，独立配置 ----
    model_b_name: str = "deepseek-chat"
    model_b_base_url: str = ""
    model_b_api_key: str = ""

    # ---- LLM 通用 ----
    llm_timeout: float = 60.0
    llm_temperature: float = 0.1
    llm_review_max_tokens: int = 2048

    # ---- 熔断（外部 LLM 服务抖动防雪崩）----
    llm_circuit_failures: int = 5       # 连续失败 N 次熔断（快速失败）
    llm_circuit_recovery: float = 60.0  # 熔断冷却秒数，冷却结束进入半开试探

    # ---- 限流（slowapi，默认放宽避免影响演示）----
    rate_limit_ask: str = "30/minute"   # /ask 每 IP 每分钟上限

    # ---- embedding 双后端 ----
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_base_url: str = ""
    embedding_device: str = "cpu"
    hf_endpoint: str = "https://hf-mirror.com"

    # ---- 检索 ----
    top_k: int = 5
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 50
    use_dense_retrieval: bool = True
    rag_cross_encoder: str = ""   # 空=不重排；填 "BAAI/bge-reranker-base" 开启 cross-encoder 精排
    knowledge_base_path: str = "./data/knowledge_base"

    # ---- Qdrant（本地嵌入式，无需 Docker；想用服务端填 URL）----
    qdrant_url: str = ""
    qdrant_path: str = "./qdrant_data"
    qdrant_collection: str = "knowledge_base"

    # ---- 记忆 ----
    sqlite_memory_path: str = "./memory.db"
    memory_episodic_collection: str = "agent_memory_episodic"
    memory_top_k: int = 3
    memory_consolidate_every: int = 6  # 每 N 轮触发一次固化
    memory_ttl_days: int = 30          # episodic 记忆过期天数
    memory_importance_weight: float = 0.5  # importance 在检索排序里的加权系数（0=纯相似度）

    # ---- 安全 ----
    safety_enabled: bool = True
    safety_sensitive_words: str = ""

    # ---- 工具白名单（Function Calling，D9 售后工具）----
    tool_whitelist: str = "safety_check,query_order,query_logistics,apply_refund,web_search"  # 逗号分隔，未纳入的工具不允许执行
    tool_use_function_calling: bool = False  # True 时 tools_node 优先走 bind_tools Function Calling，失败退回启发式路由

    # ---- 日志 ----
    log_level: str = "INFO"

    @property
    def cors_origins(self) -> list[str]:
        return ["*"]

    def get_safety_sensitive_words_list(self) -> list[str]:
        if not self.safety_sensitive_words:
            return []
        return [w.strip() for w in self.safety_sensitive_words.split(",") if w.strip()]

    def get_tool_whitelist(self) -> list[str]:
        if not self.tool_whitelist:
            return []
        return [t.strip() for t in self.tool_whitelist.split(",") if t.strip()]


settings = Settings()

# 国内加速：必须在延迟导入 sentence_transformers 之前设置，否则首次加载 bge 模型
# 会连官方 huggingface.co 超时重试、阻塞事件循环。用 setdefault 尊重用户已设的 HF_ENDPOINT。
os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
