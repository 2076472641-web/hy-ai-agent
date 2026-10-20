@echo off
chcp 65001 >nul
echo ========================================
echo   途景汇文旅多智能体一体化系统
echo   启动脚本 (Windows) [优化版]
echo ========================================
echo.

REM 检查 Python 环境
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.11+
    pause
    exit /b 1
)

REM ============ 创建虚拟环境（同时兼容 .venv 和 venv）============
set "VENV_DIR="
if exist ".venv\Scripts\python.exe" (
    set "VENV_DIR=.venv"
) else if exist "venv\Scripts\python.exe" (
    set "VENV_DIR=venv"
)

if "%VENV_DIR%"=="" (
    echo [初始化] 创建虚拟环境 (.venv)...
    python -m venv .venv
    set "VENV_DIR=.venv"
)

REM 激活虚拟环境
echo [环境] 使用虚拟环境: %VENV_DIR%
call %VENV_DIR%\Scripts\activate.bat

REM ============ 快速检测：是否需要安装依赖 ============
REM 方法：检查几个核心包是否已安装，如果都在就跳过 pip install（极大加速）
set "NEED_INSTALL=0"
%VENV_DIR%\Scripts\python.exe -c "import fastapi, uvicorn, langchain, httpx, dotenv; print('DEP_OK')" >nul 2>&1
if %errorlevel% neq 0 (
    set "NEED_INSTALL=1"
)

if "%NEED_INSTALL%"=="1" (
    echo [初始化] 首次运行，正在安装依赖（请耐心等待）...
    pip install -r requirements.txt
) else (
    echo [初始化] 依赖已安装，跳过 pip install  ✅
)

REM 创建必要目录
if not exist "data\checkpoints" mkdir "data\checkpoints"
if not exist "data\train" mkdir "data\train"
if not exist "models\lora" mkdir "models\lora"
if not exist "models\lora-output" mkdir "models\lora-output"

REM 复制环境变量文件
if not exist ".env" (
    copy .env.example .env
    echo [提示] 请编辑 .env 文件配置你的 API Key
)

echo.
echo ========================================
echo   选择启动模式:
echo   1. 启动全部服务（推荐 - 直接可用）
echo   2. 仅启动 MCP 工具服务
echo   3. 仅启动 API 网关（本地回退模式）
echo   4. 运行测试
echo   5. 初始化数据库
echo ========================================
echo.

REM 默认选项 = 1（如果用户直接回车就用选项1，省得每次都输入）
set /p choice=请输入选项 (1-5，直接回车=启动全部): 
if "%choice%"=="" set "choice=1"

if "%choice%"=="1" goto all
if "%choice%"=="2" goto mcp
if "%choice%"=="3" goto gateway
if "%choice%"=="4" goto test
if "%choice%"=="5" goto db
echo 无效选项
goto end

:all
echo [启动] 启动全部服务...
REM ===== 优化：先启动 MCP，等待 2 秒；然后并行启动所有 Agent（减少 MCP 等待）=====
echo   ⏳ 启动 MCP Server (端口 8001)...
start "MCP Server" python -m mcp_server.server
REM 给 MCP 留一点启动时间（这样 Agent 启动时就能直接连上 MCP，不用回退）
timeout /t 2 /nobreak >nul
echo   ⏳ 启动 Weather Agent (端口 8101)...
start "Weather Agent" python -m agents.weather_agent.main
echo   ⏳ 启动 Attraction Agent (端口 8102)...
start "Attraction Agent" python -m agents.attraction_agent.main
echo   ⏳ 启动 Route Agent (端口 8103)...
start "Route Agent" python -m agents.route_agent.main
echo   ⏳ 启动 Gateway (端口 8000)...
start "Gateway" python -m gateway.gateway

REM 打开浏览器（等 Gateway 起来再打开）
echo   ⏳ 等待 Gateway 就绪...
set "GW_PORT=8000"
set /a WAITS=0
:gw_wait_loop
timeout /t 1 /nobreak >nul
set /a WAITS+=1
REM 简单检测：用 netstat 看端口是否 LISTENING
netstat -ano | findstr "LISTENING" | findstr ":%GW_PORT%" >nul 2>&1
if %errorlevel%==0 (
    goto gw_ready
)
if %WAITS% LSS 20 (
    goto gw_wait_loop
)
:gw_ready
echo 🌐 正在打开前端页面...
start "" http://localhost:%GW_PORT%

echo.
echo [完成] 所有服务已启动 🚀
echo   - MCP 服务: http://localhost:8001
echo   - 天气智能体: http://localhost:8101
echo   - 景点智能体: http://localhost:8102
echo   - 路线智能体: http://localhost:8103
echo   - API 网关 + 前端页面: http://localhost:8000
echo.
echo [提示] 关闭各服务控制台窗口即可停止服务
goto end

:mcp
echo [启动] 启动 MCP 工具服务...
python -m mcp_server.server
goto end

:gateway
echo [启动] 启动 API 网关...
python -m gateway.gateway
goto end

:test
echo [测试] 运行集成测试...
python -m tests.test_collaboration
goto end

:db
echo [数据库] 初始化数据库...
python -m mcp_server.db.init_data
goto end

:end
echo.
pause
