# second-brain 私有化部署文档（CPU 服务器）

本文档描述如何把本项目（导入 + 查询 + 私有化 embedding/rerank）部署到一台 **纯 CPU** 的阿里云 ECS 上，**不需要 GPU 服务器**。

---

## 一、架构总览

| 组件 | 端口 | 作用 | 部署方式 |
|---|---|---|---|
| import 服务 | 8000 | 上传 md/zip/URL → 清洗切分 → 向量入库 | uvicorn (Python) |
| query 服务 | 8001 | 对话问答（三路 RRF 检索 + 精排 + LLM 生成） | uvicorn (Python) |
| embedding 服务 | 8002 | dense 向量化（bge-base-zh-v1.5，768 维） | uvicorn (Python) |
| rerank 服务 | 8003 | 精排打分（bge-reranker-v2-m3，Cross-Encoder） | uvicorn (Python) |
| Milvus | 19530 | 向量库（dense + sparse） | Docker |
| MinIO | 9000 | 图片对象存储（答案里的图片 URL 指向它） | Docker |
| MongoDB | 27017 | 对话历史 | Docker |
| LLM | 外部 | 生成答案（deepseek API，OpenAI 兼容） | 无需部署 |
| 联网搜索 | 外部 | DashScope 百炼 WebSearch MCP | 无需部署 |

**本地模型只有 2 个（都在 CPU 跑）**：
- `bge-base-zh-v1.5`（约 430MB，dense 向量化）
- `bge-reranker-v2-m3`（约 2.3GB，Cross-Encoder 精排）

**LLM 走 deepseek API（OpenAI 兼容），所以不需要 GPU。**

---

## 二、硬件选型建议

| 档位 | 配置 | 说明 |
|---|---|---|
| 推荐 | 8 vCPU / 16 GB / 100GB ESSD | rerank 在 CPU 上较慢，8 核能压到 1s 内/查询 |
| 省钱够用 | 4 vCPU / 8 GB / 80GB ESSD | 能跑，rerank 每次 1~2s，并发 1~2 人 |
| 有余量 | 8 vCPU / 32 GB | 未来换本地 LLM 或加数据量 |

- 系统：**Ubuntu 22.04 LTS（x86_64）**
- 带宽：建议 ≥ 5Mbps（图片走 MinIO 出网）
- 型号：`ecs.c7` / `ecs.g7`（通用/计算型），预算紧用 `ecs.u1`

---

## 三、部署步骤

### 0. 前置：安全组放行端口

在 ECS 安全组放行入方向端口，其余关闭：

- `80` / `443`（Nginx）
- `8000`（导入前端/接口）
- `8001`（查询前端/接口）
- `9000`（MinIO，**图片要能被浏览器直接打开**）

`19530` / `27017` / `8002` / `8003` 只本机访问，**不要对公网开放**。

### 1. 系统环境

```bash
# 1.1 安装 Docker
curl -fsSL https://get.docker.com | sh
sudo systemctl enable --now docker

# 1.2 安装 Miniconda（Python 3.10）
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh
bash Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh
conda create -n second-brain python=3.10 -y
conda activate second-brain
```

### 2. 下载两个本地模型（国内走 ModelScope，HF 被墙）

```bash
pip install modelscope
mkdir -p /data/models && cd /data/models

modelscope download --model BAAI/bge-base-zh-v1.5 --local_dir /data/models/bge-base-zh-v1.5
modelscope download --model BAAI/bge-reranker-v2-m3 --local_dir /data/models/bge-reranker-v2-m3
```

### 3. 拉代码 + 配置 `.env`

```bash
cd /data && git clone <你的仓库地址> second-brain && cd second-brain
```

**关键：改 `knowledge/.env`（Windows 路径全部换成 Linux 绝对路径）**：

```bash
# ===== 模型路径（改成服务器路径）=====
BGE_BASE_ZH_MODEL_PATH=/data/models/bge-base-zh-v1.5
BGE_RERANKER_MODEL_PATH=/data/models/bge-reranker-v2-m3
BGE_RERANKER_DEVICE=cpu
BGE_RERANKER_FP16=0

# ===== ★ 关键：MinIO 地址必须换成公网 IP/域名 =====
# 图片 URL 在导入时会被写死进 Milvus，设成 localhost 会导致图片 404
MINIO_ENDPOINT=<你的公网IP或域名>:9000
MINIO_BUCKET_NAME=second-brain

# ===== LLM（deepseek 或任意 OpenAI 兼容服务）=====
OPENAI_API_KEY=你的key
OPENAI_API_BASE=https://api.deepseek.com
LLM_DEFAULT_MODEL=deepseek-flash

# ===== 服务地址（都在本机，保持 localhost）=====
MILVUS_URL=http://localhost:19530
MONGO_URL=mongodb://localhost:27017
EMBEDDING_SERVICE_URL=http://localhost:8002
RERANK_SERVICE_URL=http://localhost:8003
SPARSE_IDF_PATH=/data/second-brain/knowledge/data/bm25_idf.pkl

# ===== 登录安全（务必改掉默认值）=====
ADMIN_USERNAME=你的账号
ADMIN_PASSWORD=强密码
JWT_SECRET=一串随机长字符串

# ===== MinIO 密钥（改成强密码）=====
MINIO_ACCESS_KEY=你的minio账号
MINIO_SECRET_KEY=你的minio密码
```

### 4. 装 Python 依赖（CPU 版 torch）

```bash
# 先装 CPU 版 torch，避免拉到 2GB+ 的 CUDA 版
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 5. 启动 Milvus / MinIO / MongoDB（Docker）

建 `docker-compose.yml`（注意：**Milvus 自带内部 etcd+minio，跟项目用图片的 MinIO 是两套，别混**）：

```yaml
services:
  # 项目图片存储用的 MinIO（9000 对公网）
  app-minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports: ["9000:9000", "9001:9001"]
    volumes: ["./minio-data:/data"]

  mongo:
    image: mongo:6
    ports: ["27017:27017"]
    volumes: ["./mongo-data:/data/db"]

  # Milvus 依赖的 etcd
  etcd:
    image: quay.io/coreos/etcd:v3.5.14
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  # Milvus 内部 minio（跟 app-minio 无关）
  milvus-minio:
    image: minio/minio:latest
    command: minio server /minio_data
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes: ["./milvus-minio:/minio_data"]

  # Milvus standalone
  milvus:
    image: milvusdb/milvus:v2.4.17
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: milvus-minio:9000
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on: ["etcd", "milvus-minio"]
    volumes: ["./milvus-data:/var/lib/milvus"]
```

```bash
docker compose up -d
```

> 提示：MinIO 密码、Milvus 内部 minio 密码都建议改成强密码，并同步到 `.env` 的 `MINIO_ACCESS_KEY/SECRET_KEY`。

### 6. 启动 4 个 Python 服务（systemd 保活）

四个 uvicorn 命令：

```bash
# import（8000）
uvicorn knowledge.front.api.import_router:create_app --factory --host 0.0.0.0 --port 8000
# query（8001）
uvicorn knowledge.front.api.query_router:create_app --factory --host 0.0.0.0 --port 8001
# embedding（8002）
uvicorn model_services.embedding_service.embedding_app:app --host 0.0.0.0 --port 8002
# rerank（8003）
uvicorn model_services.rerank_service.rerank_app:app --host 0.0.0.0 --port 8003
```

用 systemd 保活（以 import 为例，其余同理只改端口和模块名）：

```bash
sudo tee /etc/systemd/system/second-brain-import.service >/dev/null <<'EOF'
[Unit]
Description=Second-Brain Import Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/data/second-brain
Environment="PATH=/home/ubuntu/miniconda3/envs/second-brain/bin"
ExecStart=/home/ubuntu/miniconda3/envs/second-brain/bin/uvicorn knowledge.front.api.import_router:create_app --factory --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now second-brain-import second-brain-query second-brain-embedding second-brain-rerank
```

> 注意：生产环境**不要用 `reload=True`**（那是开发用）。这里用 `--factory` 直接跑 app 工厂。

### 7. （可选但推荐）Nginx 反代 + HTTPS

```nginx
server {
    listen 80;
    server_name kb.example.com;

    location / { proxy_pass http://127.0.0.1:8000; }      # 导入
    location /chat/ { proxy_pass http://127.0.0.1:8001; } # 查询
    location /minio/ { proxy_pass http://127.0.0.1:9000; }
}
```

用 certbot 申请 HTTPS：

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d kb.example.com
```

### 8. 验证

1. `docker ps` 三个容器全 up。
2. `curl -X POST http://localhost:8002/embed_documents -H 'Content-Type: application/json' -d '{"texts":["你好"]}'` 返回 768 维向量。
3. 浏览器访问 `http://<公网IP>:8000/import`，登录后传一个 md 测试。
4. 提问后答案里的图片 URL 应指向 `http://<公网IP>:9000/...`，能打开。

---

## 四、常见坑

1. **`MINIO_ENDPOINT` 必须是公网地址**。图片 URL 在导入时写死进 Milvus，设成 `localhost:9000` 会导致服务器上图片全部 404。
2. **MinIO bucket 要匿名可读**。代码里 `ensure_bucket_public_read` 会自动设（含自动建 bucket），但首次导入前确认 bucket 已创建。
3. **两套 MinIO 别搞混**：`app-minio`（图片，9000 对公网）和 `milvus-minio`（Milvus 内部存储，本机）。
4. **torch 必须装 CPU 版**，否则 `pip install -r requirements.txt` 会拉 2GB+ 的 CUDA 版白占空间。
5. **模型用 ModelScope 下载**（HuggingFace 国内不通）。
6. **rerank 是 CPU 瓶颈**：`.env` 里 `RRF_MAX_RESULTS=5` 保持小值，控制精排候选数；并发高就加到 8 核以上。
7. **默认密码/JWT/MinIO 密钥全部改掉**再上线。

---

## 五、关于「完全私有化」

当前 **LLM 是 deepseek API、联网搜索是百炼 API，仍是外部服务**。若要「数据完全不出服务器」：

- **LLM 本地化**：装 Ollama，拉 `qwen2.5:7b` 之类，把 `.env` 的 `OPENAI_API_BASE` 指向 `http://localhost:11434/v1`。但 **CPU 跑 7B 会很慢**（单次 10~30s），不适合客服级场景。
- **联网搜索**：目前依赖百炼 MCP，纯本地化较难，可先关掉该节点。
