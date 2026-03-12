# 医疗RAG系统 Docker 一键部署脚本 (PowerShell版本)
# 支持环境检查、数据准备、镜像构建和服务启动

# 设置编码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host ""
Write-Host "===============================================" -ForegroundColor Green
Write-Host "医疗RAG系统 Docker 一键部署脚本 (PowerShell)" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green
Write-Host ""

# 函数定义
function Write-Info {
    param($Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param($Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param($Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param($Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# 检查Docker是否安装
Write-Info "检查Docker环境..."
try {
    $dockerVersion = docker --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Docker 已安装"
    } else {
        throw "Docker命令失败"
    }
} catch {
    Write-Error-Custom "Docker 未安装或未启动"
    Write-Info "请先安装 Docker Desktop for Windows"
    Write-Info "下载地址: https://www.docker.com/products/docker-desktop"
    Read-Host "按回车键退出"
    exit 1
}

# 检查Docker Compose
Write-Info "检查Docker Compose..."
try {
    $composeVersion = docker-compose --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        $composeVersion = docker compose version 2>$null
    }
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Docker Compose 已安装"
    } else {
        throw "Docker Compose命令失败"
    }
} catch {
    Write-Error-Custom "Docker Compose 未安装"
    Write-Info "请确保 Docker Desktop 包含 Docker Compose"
    Read-Host "按回车键退出"
    exit 1
}

# 检查Docker服务状态
Write-Info "检查Docker服务状态..."
try {
    docker info 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Docker 服务运行正常"
    } else {
        throw "Docker服务未运行"
    }
} catch {
    Write-Error-Custom "Docker 服务未运行"
    Write-Info "请启动 Docker Desktop"
    Read-Host "按回车键退出"
    exit 1
}

# 创建数据目录
Write-Info "创建数据持久化目录..."
$dataDirectories = @(
    "docker_data",
    "docker_data\storage",
    "docker_data\knowledge_bases", 
    "docker_data\logs",
    "docker_data\ollama_models"
)

foreach ($dir in $dataDirectories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# 复制现有知识库数据（如果存在）
if (Test-Path "medical_json_small_storage") {
    Write-Info "复制现有知识库数据..."
    try {
        Copy-Item "medical_json_small_storage\*" "docker_data\storage\" -Recurse -Force -ErrorAction SilentlyContinue
        Write-Success "知识库数据复制完成"
    } catch {
        Write-Warning "知识库数据复制失败"
    }
}

# 复制知识库文件
if (Test-Path "knowledge_bases") {
    Write-Info "复制知识库文件..."
    try {
        Copy-Item "knowledge_bases\*" "docker_data\knowledge_bases\" -Recurse -Force -ErrorAction SilentlyContinue
        Write-Success "知识库文件复制完成"
    } catch {
        Write-Warning "知识库文件复制失败"
    }
}

Write-Success "数据目录创建完成"

# 检查Dockerfile
if (!(Test-Path "Dockerfile")) {
    Write-Error-Custom "Dockerfile 不存在"
    Read-Host "按回车键退出"
    exit 1
}

# 构建Docker镜像
Write-Info "构建医疗RAG系统Docker镜像..."
Write-Warning "这可能需要10-20分钟，请耐心等待..."

try {
    docker build -t medical-rag-system:latest .
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Docker镜像构建完成"
    } else {
        throw "镜像构建失败"
    }
} catch {
    Write-Error-Custom "Docker镜像构建失败"
    Write-Info "查看详细错误信息请运行: docker build -t medical-rag-system:latest ."
    Read-Host "按回车键退出"
    exit 1
}

# 检查docker-compose.yml
if (!(Test-Path "docker-compose.yml")) {
    Write-Error-Custom "docker-compose.yml 不存在"
    Read-Host "按回车键退出"
    exit 1
}

# 停止现有服务（如果存在）
Write-Info "停止现有服务..."
docker-compose down 2>$null

# 启动服务
Write-Info "启动医疗RAG系统服务..."
try {
    docker-compose up -d
    if ($LASTEXITCODE -eq 0) {
        Write-Success "服务启动成功"
    } else {
        throw "服务启动失败"
    }
} catch {
    Write-Error-Custom "服务启动失败"
    Write-Info "查看详细错误: docker-compose logs"
    Read-Host "按回车键退出"
    exit 1
}

# 等待服务就绪
Write-Info "等待服务启动完成..."
$maxRetries = 60
$retryCount = 0

while ($retryCount -lt $maxRetries) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Success "Ollama服务已启动"
            break
        }
    } catch {
        # 忽略错误，继续等待
    }
    
    $retryCount++
    $waitMessage = "等待Ollama服务启动... (" + $retryCount + "/" + $maxRetries + ")"
    Write-Host "[INFO] $waitMessage" -ForegroundColor Blue
    Start-Sleep -Seconds 5
}

if ($retryCount -eq $maxRetries) {
    Write-Warning "Ollama服务启动超时，请检查日志"
}

# 显示服务状态
Write-Host ""
Write-Info "服务状态:"
docker-compose ps

Write-Host ""
Write-Info "服务访问信息:"
Write-Host "  Ollama API: http://localhost:11434"
Write-Host "  服务状态: docker-compose ps"
Write-Host "  服务日志: docker-compose logs -f"
Write-Host "  停止服务: docker-compose down"

Write-Host ""
Write-Info "进入容器:"
Write-Host "  docker exec -it medical-rag-system bash"

Write-Host ""
Write-Info "数据持久化目录:"
Write-Host "  知识库数据: .\docker_data\storage"
Write-Host "  知识库文件: .\docker_data\knowledge_bases"
Write-Host "  日志文件: .\docker_data\logs"
Write-Host "  AI模型: .\docker_data\ollama_models"

Write-Host ""
Write-Success "医疗RAG系统部署完成！" -ForegroundColor Green
Write-Host ""
Write-Info "下一步："
Write-Host "  1. 等待AI模型下载完成（首次启动需要时间）"
Write-Host "  2. 查看日志确认系统启动: docker-compose logs -f"
Write-Host "  3. 进入容器开始使用: docker exec -it medical-rag-system bash"
Write-Host ""

Read-Host "按回车键退出" 