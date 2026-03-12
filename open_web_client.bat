@echo off
echo ========================================
echo 医疗RAG系统Web客户端启动器
echo ========================================

echo 检查API服务器连接...
curl -s http://localhost:8000/health > nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ❌ API服务器未运行或无法连接!
    echo.
    echo 请先启动API服务器:
    echo   python medical_rag_api_server.py
    echo   或者运行: start_api_server.bat
    echo.
    pause
    exit /b 1
)

echo ✅ API服务器连接正常
echo.
echo 正在打开Web客户端...
echo 地址: medical_rag_web_client.html
echo.

REM 尝试在默认浏览器中打开HTML文件
start "" "medical_rag_web_client.html"

echo Web客户端已启动！
echo.
echo 如果浏览器没有自动打开，请手动打开:
echo   medical_rag_web_client.html
echo.
echo 功能说明:
echo - 实时查询医疗问题
echo - 切换不同查询模式
echo - 查看系统状态和统计
echo - 支持示例问题快速测试
echo.
pause 