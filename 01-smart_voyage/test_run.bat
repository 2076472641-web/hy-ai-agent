@echo off
chcp 65001 >nul
echo ========================================
echo   星途环球 ^| 跨境售后RAG智能问答系统
echo   自动化测试启动脚本 (Windows)
echo ========================================
echo.

REM ========== 1. 检查 & 激活虚拟环境 ==========
if exist "venv\Scripts\activate.bat" (
    echo [环境] 激活项目虚拟环境 venv ...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo [环境] 激活虚拟环境 .venv ...
    call .venv\Scripts\activate.bat
) else (
    echo [提示] 未找到 venv 目录，使用系统全局 Python...
    where python >nul 2>&1
    if errorlevel 1 (
        echo [错误] 未找到 Python，请先安装 Python 3.11+ 或运行 start.bat 初始化项目
        pause
        exit /b 1
    )
)

REM ========== 2. 检查依赖（只做轻量检查，避免每回pip install 浪费时间 ==========
python -c "import httpx, fastapi" 2>nul
if errorlevel 1 (
    echo [依赖] 首次运行检测到缺依赖，正在安装 requirements.txt ...
    pip install -r requirements.txt
)

REM ========== 3. 创建输出目录 ==========
if not exist "tests\test_output" mkdir "tests\test_output"

REM ========== 4. 加载 .env（可选，告诉用户配置已经被config.py加载 ==========
if not exist ".env" (
    if exist ".env.example" (
        echo [提示] 还没有 .env 文件，已为你复制 .env.example 为 .env
        copy .env.example .env
        echo [提示] 如需真实 API 模式记得去填 DASHSCOPE_API_KEY
    )
)

:menu_loop
echo.
echo ========================================
echo   请选择测试运行模式：
echo   1. Mock模式      - 离线跑全部51组用例  (推荐，零Token零服务
echo   2. Local模式   - 复用项目Orchestrator真实逻辑
echo   3. API模式     - 调用FastAPI RAG问答接口 (需先启动网关
echo   4. 仅跨境物流场景专项（17组）
echo   5. 海外售后场景专项（17组）
echo   6. 海关合规场景专项（17组）
echo   7. 【面试演示：幻觉检测效果展示（5条坏回答
echo   8. 打开 test_output 输出目录
echo   0. 退出
echo ========================================
echo.

set /p choice=请输入选项 (0-8): 

if "%choice%"=="1" goto mock_all
if "%choice%"=="2" goto local_all
if "%choice%"=="3" goto api_all
if "%choice%"=="4" goto cat_log
if "%choice%"=="5" goto cat_asp
if "%choice%"=="6" goto cat_cus
if "%choice%"=="7" goto demo_hall
if "%choice%"=="8" goto open_out
if "%choice%"=="0" goto end
echo 无效选项，请输入 0-8
goto menu_loop

REM ========== 各分支实现 ==========

:mock_all
echo.
echo [运行模式1: Mock模式 · 全量51组
echo ========================================
python -m tests.test_rag_qa --mode mock
goto menu_loop

:local_all
echo.
echo [运行模式2: Local模式 · 全量51组
echo ========================================
python -m tests.test_rag_qa --mode local
goto menu_loop

:api_all
echo.
echo [运行模式3: API模式 · 全量51组
echo ========================================
echo 请确认是否已启动网关（start.bat启动网关？
set /p ok=启动了按回车继续，未启动按N回主菜单 (Y/N默认Y：
if /i "%ok%"=="N" goto menu_loop
python -m tests.test_rag_qa --mode api
goto menu_loop

:cat_log
echo.
echo [运行模式4: 跨境物流咨询专项 17组
echo ========================================
python -m tests.test_rag_qa --category cross_border_logistics
goto menu_loop

:cat_asp
echo.
echo 运行模式5: 海外售后政策专项 17组
echo ========================================
python -m tests.test_rag_qa --category after_sales_policy
goto menu_loop

:cat_cus
echo.
echo 运行模式6: 海关合规政策专项 17组
echo ========================================
python -m tests.test_rag_qa --category customs_compliance
goto menu_loop

:demo_hall
echo.
echo [面试演示: 幻觉检测 - 5条坏回答
echo ========================================
echo 这组用 mock-noise 0.8模拟编造回答，展示本系统幻觉识别能力
echo.
python -m tests.test_rag_qa --mock-noise 0.8 --limit 5
goto menu_loop

:open_out
echo.
explorer "tests\test_output"
goto menu_loop

:end
echo.
echo 测试结束，再见！
pause
