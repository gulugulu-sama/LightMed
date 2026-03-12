# 医疗RAG系统Docker镜像
# 基于Ubuntu 22.04，集成Python 3.11 + Ollama + 医疗RAG套件

# 使用官方Ubuntu 22.04作为基础镜像
FROM ubuntu:22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 设置工作目录
WORKDIR /app

# 更新系统并安装基础依赖
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 创建Python虚拟环境
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 升级pip，增加超时时间和重试次数
RUN pip install --upgrade pip setuptools wheel \
    --timeout 300 \
    --retries 5 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 创建Ollama安装脚本，运行时安装而不是构建时安装
RUN echo '#!/bin/bash\n\
echo "正在安装Ollama..."\n\
for i in 1 2 3; do\n\
    curl -fsSL --connect-timeout 30 --max-time 1800 --retry 3 --retry-delay 10 \\\n\
    https://ollama.ai/install.sh -o /tmp/install.sh\n\
    if [ $? -eq 0 ]; then\n\
        chmod +x /tmp/install.sh\n\
        bash /tmp/install.sh\n\
        if [ $? -eq 0 ]; then\n\
            rm -f /tmp/install.sh\n\
            echo "Ollama安装成功"\n\
            break\n\
        fi\n\
    fi\n\
    echo "Ollama安装失败，重试第$((i+1))次..."\n\
    sleep 30\n\
done\n\
' > /usr/local/bin/install-ollama.sh && chmod +x /usr/local/bin/install-ollama.sh

# 复制项目文件
COPY requirements.txt /app/
COPY . /app/

# 安装Python依赖，使用国内镜像源并增加超时设置
RUN pip install -r requirements.txt \
    --timeout 300 \
    --retries 5 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

# 创建必要的目录
RUN mkdir -p /app/logs \
    /app/medical_json_small_storage \
    /app/knowledge_bases \
    /app/configs \
    /root/.ollama

# 设置权限
RUN chmod +x /app/main.py

# 复制优化启动脚本
COPY start_optimized.sh /app/start.sh
RUN chmod +x /app/start.sh

# 暴露端口
EXPOSE 11434 8000

# 设置健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:11434/api/tags || exit 1

# 启动命令
CMD ["/app/start.sh"] 