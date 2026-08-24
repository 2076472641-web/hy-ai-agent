#!/bin/bash
# ========================================
# 途景汇文旅多智能体一体化系统
# 启动脚本 (Linux/Mac)
# ========================================

set -e

echo "========================================"
echo "  途景汇文旅多智能体一体化系统"
echo "  启动脚本"
echo "========================================"
echo

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到 python3，请先安装 Python 3.11+"
    exit 1
fi

# 创建虚拟环境
if [ ! -d ".venv" ]; then
    echo "[初始化] 创建虚拟环境..."
    python3 -m venv .venv
fi

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
echo "[初始化] 安装依赖..."
pip install -r requirements.txt

# 创建目录
mkdir -p data/checkpoints data/train models/lora models/lora-output

# 复制环境变量文件
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "[提示] 请编辑 .env 文件配置你的 API Key"
fi

echo
echo "========================================"
echo "  选择启动模式:"
echo "  1. 启动全部服务"
echo "  2. 仅启动 MCP 工具服务"
echo "  3. 仅启动 API 网关"
echo "  4. 运行测试"
echo "  5. 初始化数据库"
echo "========================================"
echo

read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        echo "[启动] 启动全部服务..."
        python -m mcp_server.server &
        sleep 3
        python -m agents.weather_agent.main &
        python -m agents.attraction_agent.main &
        python -m agents.route_agent.main &
        python -m gateway.gateway &
        echo "[完成] 所有服务已启动"
        echo "  - MCP 服务: http://localhost:8001"
        echo "  - 天气智能体: http://localhost:8101"
        echo "  - 景点智能体: http://localhost:8102"
        echo "  - 路线智能体: http://localhost:8103"
        echo "  - API 网关: http://localhost:8000"
        wait
        ;;
    2)
        echo "[启动] 启动 MCP 工具服务..."
        python -m mcp_server.server
        ;;
    3)
        echo "[启动] 启动 API 网关..."
        python -m gateway.gateway
        ;;
    4)
        echo "[测试] 运行集成测试..."
        python -m tests.test_collaboration
        ;;
    5)
        echo "[数据库] 初始化数据库..."
        python -m mcp_server.db.init_data
        ;;
    *)
        echo "无效选项"
        ;;
esac