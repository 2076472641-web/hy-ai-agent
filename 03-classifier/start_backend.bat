@echo off
chcp 65001 >nul
rem ============================================================
rem 优店汇·商品类目智能分类系统 - 后端API快速启动脚本 (Windows)
rem 功能：
rem   1. 检查虚拟环境是否激活，未激活则自动查找并激活
rem   2. 检查核心依赖是否安装
rem   3. 启动 FastAPI 后端服务 (默认端口 8000)
rem ============================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   优店汇 · 商品类目智能分类系统
echo   后端 API 服务启动脚本
echo ============================================================
echo.

rem ---------- 1. 切换到项目根目录 ----------
cd /d "%~dp0"
set "PROJECT_ROOT=%cd%"
echo [1/4] 项目根目录: %PROJECT_ROOT%

rem ---------- 2. 虚拟环境激活检查 ----------
echo [2/4] 检查 Python 虚拟环境...

if defined VIRTUAL_ENV (
    echo     ✓ 虚拟环境已激活: %VIRTUAL_ENV%
) else (
    echo     ! 未检测到激活的虚拟环境，正在自动查找...
    set "VENV_FOUND=0"

    rem 尝试常见虚拟环境目录名
    for %%d in (venv .venv env .env venv38 venv39 venv310 venv311) do (
        if exist "%%d\Scripts\activate.bat" (
            echo     找到虚拟环境: %%d
            call "%%d\Scripts\activate.bat"
            set "VENV_FOUND=1"
            goto :venv_ok
        )
    )

    if "!VENV_FOUND!"=="0" (
        echo.
        echo     [错误] 未找到虚拟环境！
        echo.
        echo     请先创建虚拟环境：
        echo       python -m venv venv
        echo       venv\Scripts\activate
        echo       pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
)
:venv_ok

rem ---------- 3. 依赖检查 ----------
echo [3/4] 检查核心依赖是否安装...

set "MISSING_PKGS="
for %%p in (fastapi uvicorn loguru click pandas numpy scikit-learn jieba) do (
    python -c "import %%p" 2>nul
    if errorlevel 1 (
        set "MISSING_PKGS=!MISSING_PKGS! %%p"
    )
)

if not "!MISSING_PKGS!"=="" (
    echo.
    echo     [警告] 检测到缺失依赖:!MISSING_PKGS!
    echo.
    echo     正在尝试自动安装（可能需要几分钟）...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo     [错误] 依赖安装失败，请手动执行:
        echo       pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
    echo     ✓ 依赖安装完成
) else (
    echo     ✓ 核心依赖已安装
)

rem ---------- 4. 启动后端服务 ----------
echo [4/4] 启动 FastAPI 后端服务...
echo.
echo     --------------------------------------------------------
echo       服务地址:   http://localhost:8000
echo       API文档:    http://localhost:8000/docs
echo       健康检查:   http://localhost:8000/health
echo       停止服务:   按 Ctrl + C
echo     --------------------------------------------------------
echo.

rem 检查端口参数
set "PORT=8000"
if not "%~1"=="" set "PORT=%~1"

python -m uvicorn app.api.main:app --host 0.0.0.0 --port %PORT% --log-level info

if errorlevel 1 (
    echo.
    echo [错误] 服务启动失败！请检查上面的错误信息。
    echo.
    pause
    exit /b 1
)

endlocal
