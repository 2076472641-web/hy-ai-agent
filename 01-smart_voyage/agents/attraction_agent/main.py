# -*- coding: utf-8 -*-
"""
景点推荐智能体 - FastAPI 服务入口
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 自己所在包目录也加到 sys.path（见 weather_agent/main.py 同样修复）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    # ⚠️ 懒加载：不在 lifespan 初始化 Agent（否则 ChatTongyi 导入要 ~34s，uvicorn 迟迟不监听端口）
    print("[AttractionAgent] 服务启动完成（Agent 将懒加载）")
    yield


app = FastAPI(
    title="景点推荐智能体",
    description="途景汇文旅系统 - 景点推荐服务",
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


_agent_lock = __import__("threading").Lock()


def _get_agent(app):
    agent = getattr(app.state, "agent", None)
    if agent is not None:
        return agent
    with _agent_lock:
        agent = getattr(app.state, "agent", None)
        if agent is not None:
            return agent
        from agent import AttractionAgent
        agent = AttractionAgent()
        app.state.agent = agent
        return agent


@app.get("/health")
async def health_check():
    return {"status": "ok", "agent": "attraction_agent"}


@app.post("/chat")
async def chat(request: ChatRequest):
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
    agent = _get_agent(app)
    try:
        task_data = request.model_dump()
        result = await agent.handle_a2a_task(task_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent-card")
async def get_agent_card():
    from a2a.agent_card import AgentCard
    from config import get_config
    config = get_config()
    return AgentCard.create_attraction_agent(
        f"http://localhost:{config.agents.attraction_port}"
    ).to_dict()


if __name__ == "__main__":
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from _console_title_helper import set_my_title_from_env
        set_my_title_from_env(fallback="attraction_agent")
    except Exception:
        pass
    import uvicorn
    from config import get_config
    config = get_config()
    uvicorn.run(app, host="0.0.0.0", port=config.agents.attraction_port)