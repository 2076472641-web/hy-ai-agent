# 途景汇文旅多智能体一体化系统

基于 **Python + LangChain + A2A 多智能体架构 + MCP 工具协议** 构建的文旅多智能体协同系统。

## 📦 项目概述

本系统采用微服务架构，包含 3 个独立智能体通过 A2A 协议通信协同：

| 智能体 | 端口 | 功能 |
|--------|------|------|
| 🌤️ 天气查询智能体 | 8101 | 目的地天气预报、穿衣建议 |
| 🏛️ 景点推荐智能体 | 8102 | 景点搜索、行程规划、详情查询 |
| 🚄 路线规划智能体 | 8103 | 交通方案、每日路线、出行贴士 |
| 🔌 MCP 工具服务 | 8001 | 统一工具接口（天气/景点/路线/数据库） |
| 🌐 API 网关 | 8000 | 统一入口，智能体编排调度 |

## 📁 工程目录

```
smart_voyage/
├── agents/                          # 智能体微服务
│   ├── weather_agent/               #   天气查询智能体
│   │   ├── agent.py                 #     核心 Agent 逻辑
│   │   ├── prompt.py                #     System Prompt
│   │   ├── main.py                  #     FastAPI 服务入口
│   │   └── Dockerfile               #     容器化配置
│   ├── attraction_agent/            #   景点推荐智能体（同结构）
│   └── route_agent/                #   路线规划智能体（同结构）
├── a2a/                            # A2A 通信协议层
│   ├── protocol.py                  #   消息/任务数据结构
│   ├── agent_card.py                #   智能体名片定义
│   └── transport.py                 #   HTTP 传输层实现
├── mcp_server/                     # MCP 统一工具服务层
│   ├── server.py                    #   MCP Server 主入口
│   ├── tools/                       #   工具实现
│   │   ├── weather_tools.py         #     天气查询工具
│   │   ├── attraction_tools.py      #     景点推荐工具
│   │   ├── route_tools.py           #     路线规划工具
│   │   └── db_tools.py              #     数据库查询工具
│   ├── db/                          #   MySQL 相关
│   │   ├── schema.sql               #     数据库表结构
│   │   └── init_data.py             #     示例数据初始化
│   └── Dockerfile
├── gateway/                        # API 网关 + 编排器
│   ├── gateway.py                   #   FastAPI 网关服务
│   ├── orchestrator.py              #   智能体编排调度
│   └── Dockerfile
├── memory/                         # 对话缓存与上下文压缩
│   ├── conversation_memory.py       #   会话缓存管理
│   └── context_compressor.py        #   上下文压缩器
├── lora/                           # LoRA 微调
│   ├── lora_config.py               #   微调配置
│   ├── data_preprocess.py           #   数据预处理
│   └── train.py                     #   训练脚本
├── tests/                          # 测试脚本
│   ├── test_collaboration.py        #   多智能体协同集成测试
│   ├── test_weather_agent.py
│   ├── test_attraction_agent.py
│   └── test_route_agent.py
├── config.py                       # 全局配置
├── requirements.txt                # 依赖列表
├── .env.example                    # 环境变量示例
├── docker-compose.yml              # Docker 编排
├── Dockerfile.base                 # 基础镜像
├── start.bat / start.sh            # 启动脚本
└── README.md                       # 本文档
```

## 🚀 快速开始

### 环境要求

- Python 3.11+
- MySQL 8.0+（可选，使用模拟数据可跳过）
- Docker 20+（可选，容器化部署）

### 本地开发模式

```bash
# 1. 克隆项目
cd smart_voyage

# 2. 创建虚拟环境并安装依赖
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key

# 4. 运行测试（无需 API Key）
python -m tests.test_collaboration
# 输出：✅ 所有测试通过！系统功能正常！
```

### 启动服务

**方式一：一键启动脚本**
```bash
# Windows
start.bat
# Linux/Mac
./start.sh
```

**方式二：手动启动**
```bash
# 终端 1 - 启动 MCP 工具服务
python -m mcp_server.server

# 终端 2 - 启动天气智能体
python -m agents.weather_agent.main

# 终端 3 - 启动景点智能体
python -m agents.attraction_agent.main

# 终端 4 - 启动路线智能体
python -m agents.route_agent.main

# 终端 5 - 启动 API 网关
python -m gateway.gateway
```

### API 使用

```bash
# 健康检查
curl http://localhost:8000/health

# 通用对话（自动路由到合适智能体）
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我查一下成都明天的天气"}'

# 一键规划完整行程
curl -X POST http://localhost:8000/api/trip-plan \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "上海",
    "destination": "成都",
    "dates": "2026-10-01",
    "days": 5,
    "preferences": ["自然风光", "历史文化"],
    "budget": "标准"
  }'

# 获取会话历史
curl http://localhost:8000/api/sessions/{session_id}
```

## 🐳 Docker 部署

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env

# 2. 一键启动所有服务
docker compose up -d

# 3. 查看服务状态
docker compose ps

# 4. 查看日志
docker compose logs -f gateway

# 5. 停止服务
docker compose down
```

### 服务端口映射

| 服务 | 容器端口 | 主机端口 |
|------|----------|----------|
| MySQL | 3306 | 3306 |
| MCP Server | 8001 | 8001 |
| Weather Agent | 8101 | 8101 |
| Attraction Agent | 8102 | 8102 |
| Route Agent | 8103 | 8103 |
| Gateway | 8000 | 8000 |

## 🧪 测试

```bash
# 运行全部集成测试
python -m tests.test_collaboration

# 单独测试各智能体
python -m tests.test_weather_agent
python -m tests.test_attraction_agent
python -m tests.test_route_agent

# 测试结果示例：
# ✅ 通过: 39
# ❌ 失败: 0
# 🎉 所有测试通过！系统功能正常！
```

## 🤖 LoRA 微调

```bash
# 1. 准备训练数据
python -m lora.data_preprocess

# 2. 开始训练（需要 GPU）
python -m lora.train --model-path /path/to/your/llm --epochs 3

# 3. 合并权重
python -c "from lora.train import merge_lora_with_base; merge_lora_with_base('models/lora-output', 'base_model', 'models/merged')"

# 4. 测试微调效果
python -m lora.train --test-only
```

## 🏗️ 架构设计

### A2A 多智能体通信

```
┌─────────────┐     A2A/HTTP     ┌──────────────┐
│  天气智能体  │ ◄──────────────► │  景点智能体   │
└─────────────┘                   └──────────────┘
        ▲                               ▲
        │                               │
        │         ┌──────────────┐      │
        └─────────│  API 网关     │◄─────┘
                  │  (编排器)     │
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │  路线智能体   │
                  └──────────────┘
                         │
                  ┌──────▼───────┐
                  │  MCP 工具服务 │
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │  MySQL 数据库 │
                  └──────────────┘
```

### MCP 工具协议

所有工具通过 MCP Server 统一暴露，智能体通过 `langchain-mcp-adapters` 加载工具：

- `query_weather` - 天气查询
- `get_weather_tips` - 穿衣建议
- `search_attractions` - 景点搜索
- `recommend_itinerary` - 行程推荐
- `get_attraction_detail` - 景点详情
- `plan_transport` - 交通规划
- `plan_daily_routes` - 每日路线
- `query_destinations` - 目的地查询

### 上下文压缩策略

系统实现了智能上下文压缩方案：
1. **自动触发**：当 Token 数超过阈值时自动压缩
2. **保留关键**：识别包含决策、偏好、预算等关键信息的消息
3. **摘要生成**：将历史消息压缩为结构化摘要
4. **智能压缩**：基于关键词权重选择性保留重要消息

## 📚 技术栈

| 类别 | 技术 | 版本 |
|------|------|------|
| 语言 | Python | 3.11+ |
| 框架 | LangChain / LangGraph | 0.3.x / 0.2.x |
| 协议 | MCP (Model Context Protocol) | 1.0+ |
| 通信 | A2A (Agent-to-Agent) | 自实现 |
| Web | FastAPI / Uvicorn | 0.110+ |
| 数据库 | MySQL / SQLAlchemy | 8.0+ |
| 容器 | Docker / Docker Compose | 20+ |
| 微调 | PEFT / Transformers | 0.10+ |

## 🔑 配置说明

编辑 `.env` 文件：

```env
# LLM 配置（必填）
DASHSCOPE_API_KEY=你的千问API Key
LLM_MODEL=qwen-plus

# MySQL 配置（可选）
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=root
MYSQL_DATABASE=songjing_hui

# 智能体端口
WEATHER_AGENT_PORT=8101
ATTRACTION_AGENT_PORT=8102
ROUTE_AGENT_PORT=8103
GATEWAY_PORT=8000
```

## 📖 API 文档

启动服务后访问：
- 网关 Swagger: http://localhost:8000/docs
- 天气智能体: http://localhost:8101/docs
- 景点智能体: http://localhost:8102/docs
- 路线智能体: http://localhost:8103/docs

## 📄 许可证

MIT License