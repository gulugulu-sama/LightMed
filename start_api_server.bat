@echo off
echo ========================================
echo 医疗RAG系统API服务器启动脚本
echo ========================================

echo 检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python未安装或未添加到PATH
    pause
    exit /b 1
)

echo.
echo 检查必要依赖...
python -c "import fastapi, uvicorn" 2>nul
if %errorlevel% neq 0 (
    echo 安装FastAPI和Uvicorn...
    pip install fastapi uvicorn -i https://pypi.tuna.tsinghua.edu.cn/simple
)

echo.
echo 检查配置文件...
if not exist "configs\rag_config.yaml" (
    echo ERROR: 缺少配置文件 configs\rag_config.yaml
    pause
    exit /b 1
)

if not exist "medical_json_small_storage" (
    echo ERROR: 缺少知识库存储目录 medical_json_small_storage
    echo 请先运行main.py构建知识库
    pause
    exit /b 1
)

echo.
echo 启动医疗RAG API服务器...
echo 服务地址: http://localhost:8000
echo API文档: http://localhost:8000/docs
echo 健康检查: http://localhost:8000/health
echo.
echo 按Ctrl+C停止服务器
echo ========================================

python medical_rag_api_server.py --host 0.0.0.0 --port 8000

echo.
echo 服务器已停止
pause 