# ============================================================================
# 多阶段构建 —— CPU 版瘦身（对齐 HELLO「8.6GB 镜像」待优化点）
#
# 三个瘦身点（面试点）：
#   1. 多阶段构建：builder 装依赖，runtime 只带运行时需要的产物，剔除构建期垃圾。
#   2. CPU 版 torch：从 PyTorch 官方 CPU 索引装 torch（不拉 CUDA 全家桶 ~2GB），
#      本服务做的是「检索 + 生成编排」，CPU 推理足够，无需 GPU 镜像。
#   3. healthcheck：容器内 curl 打 /api/v1/health，让编排系统能探活自愈。
#
# 预期镜像 < 3GB（对比 HELLO 用 CUDA torch 的 8.6GB）。
# 构建：docker build -t multi-agent-aftersales .
# ============================================================================

# ---------- Stage 1: builder（装重依赖，含 CPU 版 torch） ----------
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# sentence-transformers / torch 需要 libgomp1；curl 供 healthcheck 使用
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) 先装 CPU 版 torch（不拉 CUDA，省 ~2GB）
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) 装项目依赖 + sentence-transformers（requirements 里注释掉了，这里显式装）
COPY requirements.txt .
RUN pip install -r requirements.txt \
    && pip install sentence-transformers

# ---------- Stage 2: runtime（只带运行时产物） ----------
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_ENDPOINT=https://hf-mirror.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从 builder 拷贝已装好的 python 环境（含 site-packages + 入口脚本），剔除构建期缓存
COPY --from=builder /usr/local /usr/local
# 拷贝项目代码（.dockerignore 已排除 .env / 数据目录 / 日志）
COPY . .

# 运行时数据目录（知识库 / 向量库 / 记忆库由 docker-compose 挂载）
RUN mkdir -p /app/data /app/state

EXPOSE 8000

# 探活：起服务后 60s 内开始健康检查，失败 3 次判定不健康（编排系统据此重启）
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
