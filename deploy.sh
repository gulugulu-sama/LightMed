#!/bin/bash

# 医疗RAG系统 Docker 一键部署脚本
# 支持环境检查、数据准备、镜像构建和服务启动

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 打印横幅
print_banner() {
    echo -e "${GREEN}"
    echo "==============================================="
    echo "🏥 医疗RAG系统 Docker 一键部署脚本"
    echo "==============================================="
    echo -e "${NC}"
}

# 检查系统要求
check_requirements() {
    log_info "检查系统要求..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    log_success "Docker 已安装"
    
    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose 未安装，请先安装 Docker Compose"
        exit 1
    fi
    log_success "Docker Compose 已安装"
    
    # 检查可用内存
    available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [ "$available_memory" -lt 4096 ]; then
        log_warning "可用内存少于4GB，可能影响性能"
    else
        log_success "内存检查通过 (${available_memory}MB可用)"
    fi
    
    # 检查磁盘空间
    available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_space" -lt 10 ]; then
        log_warning "可用磁盘空间少于10GB，可能不足"
    else
        log_success "磁盘空间检查通过 (${available_space}GB可用)"
    fi
}

# 创建数据目录
create_data_directories() {
    log_info "创建数据持久化目录..."
    
    # 创建主数据目录
    mkdir -p docker_data/{storage,knowledge_bases,logs,ollama_models}
    
    # 复制现有知识库数据（如果存在）
    if [ -d "medical_json_small_storage" ]; then
        log_info "复制现有知识库数据..."
        cp -r medical_json_small_storage/* docker_data/storage/ 2>/dev/null || true
        log_success "知识库数据复制完成"
    fi
    
    # 复制知识库文件
    if [ -d "knowledge_bases" ]; then
        log_info "复制知识库文件..."
        cp -r knowledge_bases/* docker_data/knowledge_bases/ 2>/dev/null || true
        log_success "知识库文件复制完成"
    fi
    
    # 设置权限
    chmod -R 755 docker_data/
    
    log_success "数据目录创建完成"
}

# 构建Docker镜像
build_image() {
    log_info "构建医疗RAG系统Docker镜像..."
    
    # 检查Dockerfile是否存在
    if [ ! -f "Dockerfile" ]; then
        log_error "Dockerfile 不存在"
        exit 1
    fi
    
    # 构建镜像
    docker build -t medical-rag-system:latest . \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --progress=plain
    
    if [ $? -eq 0 ]; then
        log_success "Docker镜像构建完成"
    else
        log_error "Docker镜像构建失败"
        exit 1
    fi
}

# 启动服务
start_services() {
    log_info "启动医疗RAG系统服务..."
    
    # 检查docker-compose.yml是否存在
    if [ ! -f "docker-compose.yml" ]; then
        log_error "docker-compose.yml 不存在"
        exit 1
    fi
    
    # 停止现有服务（如果存在）
    docker-compose down 2>/dev/null || true
    
    # 启动服务
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        log_success "服务启动成功"
    else
        log_error "服务启动失败"
        exit 1
    fi
}

# 等待服务就绪
wait_for_services() {
    log_info "等待服务启动完成..."
    
    # 等待Ollama服务
    for i in {1..60}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            log_success "Ollama服务已启动"
            break
        fi
        echo -n "."
        sleep 5
    done
    
    if [ $i -eq 60 ]; then
        log_warning "Ollama服务启动超时，请检查日志"
    fi
}

# 显示服务状态
show_status() {
    log_info "服务状态:"
    docker-compose ps
    
    echo ""
    log_info "服务访问信息:"
    echo "  🔗 Ollama API: http://localhost:11434"
    echo "  📊 服务状态: docker-compose ps"
    echo "  📝 服务日志: docker-compose logs -f"
    echo "  🛑 停止服务: docker-compose down"
    
    echo ""
    log_info "进入容器:"
    echo "  docker exec -it medical-rag-system bash"
    
    echo ""
    log_info "数据持久化目录:"
    echo "  📁 知识库数据: ./docker_data/storage"
    echo "  📁 知识库文件: ./docker_data/knowledge_bases"
    echo "  📁 日志文件: ./docker_data/logs"
    echo "  📁 AI模型: ./docker_data/ollama_models"
}

# 清理函数
cleanup_on_error() {
    log_error "部署过程中出现错误，正在清理..."
    docker-compose down 2>/dev/null || true
    exit 1
}

# 主函数
main() {
    # 设置错误处理
    trap cleanup_on_error ERR
    
    print_banner
    
    # 解析命令行参数
    FORCE_REBUILD=false
    SKIP_BUILD=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force-rebuild)
                FORCE_REBUILD=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --help|-h)
                echo "用法: $0 [选项]"
                echo "选项:"
                echo "  --force-rebuild  强制重新构建镜像"
                echo "  --skip-build     跳过镜像构建，直接启动"
                echo "  --help, -h       显示帮助信息"
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                exit 1
                ;;
        esac
    done
    
    # 执行部署步骤
    check_requirements
    create_data_directories
    
    # 根据参数决定是否构建镜像
    if [ "$SKIP_BUILD" = false ]; then
        if [ "$FORCE_REBUILD" = true ]; then
            log_info "强制重新构建镜像..."
            docker rmi medical-rag-system:latest 2>/dev/null || true
        fi
        build_image
    else
        log_info "跳过镜像构建步骤"
    fi
    
    start_services
    wait_for_services
    show_status
    
    echo ""
    log_success "🎉 医疗RAG系统部署完成！"
    echo ""
    log_info "下一步："
    echo "  1. 等待AI模型下载完成（首次启动需要时间）"
    echo "  2. 查看日志确认系统启动: docker-compose logs -f"
    echo "  3. 进入容器开始使用: docker exec -it medical-rag-system bash"
    echo ""
}

# 执行主函数
main "$@" 