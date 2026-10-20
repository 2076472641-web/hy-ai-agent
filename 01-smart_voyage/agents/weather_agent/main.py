# -*- coding: utf-8 -*-
"""
天气智能体 - FastAPI 服务入口
提供智能体 API + A2A 通信端点
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 把自己所在包目录（agents/weather_agent）也加到 sys.path，这样 agent.py / prompt.py 可以作为顶级模块 import（相对导入在 python xxx/main.py 顶级启动时会崩）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    chat_history: list = []


class A2ATaskRequest(BaseModel):
    id: Optional[str] = None
    title: str = ""
    description: str = ""
    metadata: dict = {}
    messages: list = []
    status: str = "pending"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ⚠️ 不在 lifespan 里初始化 WeatherAgent 了！
    # 原因：ChatTongyi/ChatOpenAI 首次导入 + 创建耗时 ~34 秒，会导致 uvicorn 迟迟不监听端口，启动脚本检测超时。
    # 解决：改成懒加载（第一次请求 /chat 时才创建），uvicorn 立刻监听端口，启动秒级完成。
    print("[WeatherAgent] 服务启动完成（Agent 将懒加载）")
    yield
    print("[WeatherAgent] 服务关闭")


app = FastAPI(
    title="天气查询智能体",
    description="途景汇文旅系统 - 天气查询服务",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 懒加载 + 多线程安全创建 Agent
_agent_lock = __import__("threading").Lock()


def _get_agent(app):
    """第一次请求时才创建 WeatherAgent（线程安全）。"""
    agent = getattr(app.state, "agent", None)
    if agent is not None:
        return agent
    with _agent_lock:
        agent = getattr(app.state, "agent", None)
        if agent is not None:
            return agent
        from agent import WeatherAgent
        agent = WeatherAgent()
        app.state.agent = agent
        return agent


@app.get("/health")
async def health_check():
    return {"status": "ok", "agent": "weather_agent"}


@app.post("/chat")
async def chat(request: ChatRequest):
    """与天气智能体对话"""
    from fastapi import Request as _Req
    agent = _get_agent(app)
    try:
        result = await agent.query(request.message, request.chat_history)
        return {
            "session_id": request.session_id,
            "response": result["response"],
            "tool_calls_count": len(result["tool_calls"]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/a2a/tasks")
async def a2a_handle_task(request: A2ATaskRequest):
    """处理来自其他智能体的 A2A 任务"""
    agent = _get_agent(app)
    try:
        task_data = request.model_dump()
        result = await agent.handle_a2a_task(task_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent-card")
async def get_agent_card():
    """返回 Agent Card（A2A 协议）"""
    from a2a.agent_card import AgentCard
    from config import get_config
    config = get_config()
    return AgentCard.create_weather_agent(
        f"http://localhost:{config.agents.weather_port}"
    ).to_dict()


if __name__ == "__main__":
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from _console_title_helper import set_my_title_from_env
        set_my_title_from_env(fallback="weather_agent")
    except Exception:
        pass
    import uvicorn
    from config import get_config
    config = get_config()
    uvicorn.run(app, host="0.0.0.0", port=config.agents.weather_port)