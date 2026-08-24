# -*- coding: utf-8 -*-
"""
途景汇文旅多智能体系统 - 一键启动脚本 [优化版]
功能：
  1. 自动清理占用端口的旧进程
  2. 【优化】先启动 MCP，再并行启动 Gateway + 3 个 Agent
  3. 等待 gateway 就绪后自动打开前端页面
  4. 【优化】虚拟环境路径兼容 (.venv / venv)
"""
import subprocess
import sys
import os
import time
import socket


# ============================================================
# 端口 & 服务配置
# ============================================================
# 启动顺序：MCP 最先（这样 Agent 初始化时能连到 MCP，不用走回退等待）
MCP_SERVICE = {"script": "mcp_server/server.py", "port": 8001}

SERVICES = {
    "gateway":          {"script": "gateway/gateway.py",                "port": 8000},
    "weather_agent":    {"script": "agents/weather_agent/main.py",      "port": 8101},
    "route_agent":      {"script": "agents/route_agent/main.py",        "port": 8103},
    "attraction_agent": {"script": "agents/attraction_agent/main.py",   "port": 8102},
}


def find_python_exe() -> str:
    """查找虚拟环境中的 Python（兼容 .venv / venv / 无虚拟环境）"""
    cwd = os.getcwd()
    candidates = [
        os.path.join(cwd, ".venv", "Scripts", "python.exe"),   # start.bat 风格
        os.path.join(cwd, "venv", "Scripts", "python.exe"),    # 之前的风格
        os.path.join(cwd, ".venv", "bin", "python"),           # Linux/Mac
        os.path.join(cwd, "venv", "bin", "python"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            print(f"  🐍 发现虚拟环境 Python: {p}")
            return p
    print(f"  ⚠️  未找到虚拟环境，使用当前 Python: {sys.executable}")
    return sys.executable


def kill_port(port: int):
    """在 Windows 上杀掉占用指定端口的进程"""
    if sys.platform != "win32":
        return
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.strip().split()
                pid = parts[-1]
                if pid.isdigit() and int(pid) > 0:
                    subprocess.run(
                        ["taskkill", "/F", "/PID", pid],
                        capture_output=True, timeout=5
                    )
                    print(f"  🔧 已清理端口 {port} 上的旧进程 (PID {pid})")
                    time.sleep(0.3)
    except Exception:
        pass


def is_port_open(host: str, port: int) -> bool:
    """检测端口是否可连接（快速检测 0.5 秒）"""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except (ConnectionRefusedError, OSError):
        return False


def wait_for_port(port: int, timeout: int = 15, label: str = "") -> bool:
    """等待端口开放（不发 HTTP，直接 socket 检测，更快）"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_port_open("127.0.0.1", port):
            if label:
                print(f"  ✅ {label} 就绪 (端口 {port})")
            return True
        time.sleep(0.5)
    return False


def wait_for_gateway_http(port: int, timeout: int = 30) -> bool:
    """
    轮询等待 Gateway 就绪（分层检测，避免 /health 级联导致的误判）
    【修复说明】之前一上来就调 /health，而 /health 内部会串行检测 3 个 Agent，
    Agent 未启动好时会等待超时，导致"Gateway 实际上起来了但检测一直失败"。
    新策略：
      阶段1 → 快速检测端口开放（0.5s 一次）
      阶段2 → 检测根路径 / （不级联下游，200 就是 Gateway 启动了）
      阶段3 → 可选：检测 /health（深度检查，仅作为附加确认）
    """
    import urllib.request
    deadline = time.time() + timeout

    # ========== 阶段1：只要端口能连上，Gateway 进程就活着 ==========
    while time.time() < deadline:
        if is_port_open("127.0.0.1", port):
            break
        time.sleep(0.3)
    else:
        return False  # 端口一直没开，肯定没起来

    # ========== 阶段2：检测根路径 / 能返回 200（不涉及下游） ==========
    url_root = f"http://localhost:{port}/"
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url_root, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    # Gateway 根路径已经正常响应 → 直接判定就绪
                    # （前端页面能访问，核心的 /api/trip-plan 也会正常）
                    return True
        except Exception:
            pass
        time.sleep(0.5)

    # ========== 阶段3：兜底，用 /health 再确认（基本不会走到这） ==========
    url_health = f"http://localhost:{port}/health"
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url_health, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.8)
    return False


def start_service(script: str, python_exe: str, title: str = "") -> subprocess.Popen:
    """
    在新控制台中启动一个服务。
    【标题设置方案】各脚本入口自己设置窗口标题（见各脚本顶部 _set_my_console_title()）：
      这里只把标题写入子进程环境变量 SVCS_TITLE，不碰 run_all.py 自己的控制台句柄。
      —— 之前踩过的坑：
        1. cmd /c "title XXX && python ..."：CMD 引号解析地狱，报
           '\"D:\\...\\python.exe\"' is not recognized，服务全崩。
        2. AttachConsole(pid)+FreeConsole：释放了 run_all.py 自己的控制台句柄，
           后续 print 输出/子进程 stdout 解码全部崩溃（UTF-8 解码异常）。
      所以现在用「环境变量传标题 + 各脚本内部自己 SetConsoleTitle」，100% 稳定。
    """
    cmd = [python_exe, script]
    # 拷贝一份当前环境变量，加上标题
    child_env = os.environ.copy()
    if title:
        child_env["SVCS_TITLE"] = title
    else:
        child_env["SVCS_TITLE"] = os.path.basename(script)
    proc = subprocess.Popen(
        cmd,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        cwd=os.getcwd(),
        env=child_env,
    )
    return proc


def main():
    # 支持 --gateway-only 参数（仅启动 gateway，方便调试）
    gateway_only = "--gateway-only" in sys.argv

    python_exe = find_python_exe()

    print("=" * 56)
    print("  途景汇・文旅多智能体系统  启动中... [优化版]")
    print("=" * 56)

    # ---------- Step 1: 清理旧端口 ----------
    print("\n📡 Step 1/4: 清理旧进程...")
    ports_to_clean = [MCP_SERVICE["port"], SERVICES["gateway"]["port"]]
    if not gateway_only:
        ports_to_clean += [
            SERVICES["weather_agent"]["port"],
            SERVICES["route_agent"]["port"],
            SERVICES["attraction_agent"]["port"],
        ]
    for port in ports_to_clean:
        kill_port(int(port))
    time.sleep(0.5)

    # ---------- Step 2: 先启动 MCP（让后续 Agent 能直接连上，不用等回退超时）----------
    print("\n🚀 Step 2/4: 启动 MCP 服务（优先启动）...")
    procs = {}
    if not gateway_only:
        svc = MCP_SERVICE
        print(f"  📦 MCP Server (端口 {svc['port']})")
        procs["mcp"] = start_service(str(svc["script"]), python_exe, "MCP Server")
        # 【关键优化】等 MCP 端口起来再启动 Agent，这样 Agent 初始化时可以连接 MCP，
        #            不用走 3 秒的 socket 检测回退路径，进一步加速
        wait_for_port(svc["port"], timeout=8, label="MCP Server")
    else:
        print("  ⏭️  Gateway-only 模式，跳过 MCP + Agent")

    # ---------- Step 3: 并行启动 Gateway + 3 个 Agent（不用串行等待） ----------
    print("\n🚀 Step 3/4: 启动核心服务...")

    # Gateway 先启动（健康检查需要用到）
    gw = SERVICES["gateway"]
    print(f"  ✅ Gateway (端口 {gw['port']})")
    procs["gateway"] = start_service(str(gw["script"]), python_exe, "Gateway")

    if not gateway_only:
        # 其他 3 个 Agent 一口气并行启动，不用等
        for name in ["weather_agent", "attraction_agent", "route_agent"]:
            svc = SERVICES[name]
            print(f"  ✅ {name} (端口 {svc['port']})")
            procs[name] = start_service(str(svc["script"]), python_exe, name)

    # ---------- Step 4: 等待 Gateway 就绪，打开浏览器 ----------
    gw_port = int(SERVICES["gateway"]["port"])
    print(f"\n⏳ Step 4/4: 等待 Gateway 就绪 (端口 {gw_port})...")
    if wait_for_gateway_http(gw_port, timeout=30):
        print(f"\n✅ Gateway 已就绪！")

        import webbrowser
        webbrowser.open(f"http://localhost:{gw_port}")
        print(f"🌐 已打开前端页面: http://localhost:{gw_port}")
        print(f"\n{'=' * 56}")
        print(f"  🎉 系统启动完成！")
        print(f"  Gateway API    : http://localhost:{gw_port}")
        if not gateway_only:
            print(f"  MCP Server     : http://localhost:{MCP_SERVICE['port']}")
            print(f"  Weather Agent  : http://localhost:{SERVICES['weather_agent']['port']}")
            print(f"  Attraction Agent: http://localhost:{SERVICES['attraction_agent']['port']}")
            print(f"  Route Agent    : http://localhost:{SERVICES['route_agent']['port']}")
        print(f"\n  关闭方法: 关闭各服务控制台窗口即可")
        print(f"{'=' * 56}")
    else:
        print(f"\n❌ Gateway 在 30 秒内未就绪！")
        print(f"   可能原因:")
        print(f"   1. Python 依赖缺失 → 运行: pip install -r requirements.txt")
        print(f"   2. 端口 {gw_port} 仍被占用 → 手动运行: netstat -ano | findstr :{gw_port}")
        print(f"   3. 配置文件错误 → 检查 .env 文件")
        print(f"\n   你也可以手动打开浏览器访问: http://localhost:{gw_port}")

    # 保持主窗口不关闭
    print("\n按回车键退出此窗口（服务会继续在各自的控制台窗口运行）...")
    try:
        input()
    except EOFError:
        pass


if __name__ == "__main__":
    main()
