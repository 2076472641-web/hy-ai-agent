# -*- coding: utf-8 -*-
"""
API 网关 - 统一入口
提供 RESTful API 对外暴露智能体编排能力
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from config import get_config
from gateway.orchestrator import AgentOrchestrator
from memory import ConversationMemory, ContextCompressor


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    agents_called: list
    timestamp: float


class TripPlanRequest(BaseModel):
    """
    行程规划请求：支持新旧两种字段格式，向后兼容。
    新格式（推荐）：用 departLocation / destLocation 两个对象 + 选填 detailAddress 文本
    旧格式（兼容）：origin / destination 两个城市名字符串
    """
    # ===== 旧字段（兼容老前端 / 旧调用方）=====
    origin: Optional[str] = None
    destination: Optional[str] = None

    # ===== 新字段（新 H5 CitySelector 组件用，必传 city 级对象）=====
    # 出发城市主体：level=city 的 8 字段对象（lng/lat 固定 null）
    departLocation: Optional[dict] = None
    # 出发地详细位置（纯文本选填，如：外滩 / 虹桥火车站 / 迪士尼；后端自己解析加权）
    departDetailAddress: Optional[str] = ""

    # 目的地城市主体：level=city 的 8 字段对象（lng/lat 固定 null）
    destLocation: Optional[dict] = None
    # 目的地详细位置（纯文本选填；后端自己解析，加权优先推荐周边酒店）
    destDetailAddress: Optional[str] = ""

    dates: Optional[str] = ""
    days: int = 3
    travelers: int = 2  # 出行人数：默认2人同行，总预算=物价系数×天数×人数
    preferences: list = []
    budget: str = "舒适型"


class TripPlanResponse(BaseModel):
    session_id: str
    trip_plan: dict
    status: str
    build_revision: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化编排器"""
    app.state.orchestrator = AgentOrchestrator()
    app.state.memory = ConversationMemory()
    app.state.compressor = ContextCompressor(app.state.memory)
    print("[Gateway] API 网关启动完成")
    print(f"[Gateway] 智能体端点:")
    for name, url in app.state.orchestrator.agent_endpoints.items():
        print(f"  - {name}: {url}")
    yield
    print("[Gateway] API 网关关闭")


app = FastAPI(
    title="途景汇文旅多智能体系统 - API 网关",
    description="统一入口，编排多智能体协同完成旅游规划任务",
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


# ---------- 前端页面托管 ----------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """直接访问 http://localhost:8000 即可看到前端页面"""
    html_path = os.path.join(_PROJECT_ROOT, "customer.html")
    if os.path.isfile(html_path):
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse("<h1>customer.html 未找到</h1><p>请确认文件存在于项目根目录</p>", status_code=404)


@app.get("/picture.jpg")
async def serve_picture():
    """前端页面引用的背景图片"""
    pic_path = os.path.join(_PROJECT_ROOT, "picture.jpg")
    if os.path.isfile(pic_path):
        return FileResponse(pic_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="picture.jpg not found")


@app.get("/health")
async def health_check():
    """健康检查
    - 【性能修复】之前串行 for 循环检测 3 个 Agent，每个超时 3 秒，最坏 9 秒才返回，
      导致 run_all.py 的就绪检测（urllib 超时 2 秒）永远读不到响应，误报"Gateway 未就绪"。
    - 修复：用 asyncio.gather 并发检测，Agent 超时缩到 1 秒，总耗时不超过 1.1 秒。
    """
    orchestrator = app.state.orchestrator
    agent_status = {}

    async def _check_one(name: str, url: str):
        try:
            import httpx
            async with httpx.AsyncClient(timeout=1.0) as client:  # Agent 本地探测 1s 足够
                resp = await client.get(f"{url}/health")
                agent_status[name] = "ok"
        except Exception:
            agent_status[name] = "unavailable"

    # 并发检测所有 Agent，而不是串行一个个等
    tasks = [
        _check_one(name, url)
        for name, url in orchestrator.agent_endpoints.items()
    ]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    return {
        "status": "ok",
        "service": "途景汇文旅多智能体网关",
        "agents": agent_status,
        "timestamp": time.time(),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    通用对话接口

    自动识别用户意图并路由到合适的智能体，
    支持多智能体协同完成复杂任务。
    """
    orchestrator = app.state.orchestrator
    try:
        result = await orchestrator.handle_user_request(
            user_input=request.message,
            session_id=request.session_id,
        )
        return ChatResponse(
            session_id=result["session_id"],
            response=result["response"],
            agents_called=result["agents_called"],
            timestamp=result["timestamp"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trip-plan", response_model=TripPlanResponse)
async def plan_trip(request: TripPlanRequest):
    """
    一键规划完整旅游行程

    整合天气查询、景点推荐、路线规划，
    输出完整的旅游行程方案。
    """
    orchestrator = app.state.orchestrator
    try:
        result = await orchestrator.plan_full_trip(request.model_dump())
        return TripPlanResponse(
            session_id=result["session_id"],
            trip_plan=result["trip_plan"],
            status=result["status"],
            build_revision=result.get("_build_revision", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """获取会话信息"""
    memory = app.state.memory
    return memory.get_session_stats(session_id)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    memory = app.state.memory
    success = memory.delete_session(session_id)
    return {"success": success, "session_id": session_id}


@app.get("/api/sessions")
async def list_sessions():
    """列出所有会话"""
    memory = app.state.memory
    return {"sessions": memory.list_sessions()}


@app.get("/api/agents")
async def list_agents():
    """列出所有可用智能体"""
    return {
        "agents": [
            {
                "name": "weather_agent",
                "description": "天气查询智能体",
                "endpoint": get_config().agents.weather_url,
            },
            {
                "name": "attraction_agent",
                "description": "景点推荐智能体",
                "endpoint": get_config().agents.attraction_url,
            },
            {
                "name": "route_agent",
                "description": "出行路线规划智能体",
                "endpoint": get_config().agents.route_url,
            },
        ],
    }


if __name__ == "__main__":
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from _console_title_helper import set_my_title_from_env
        set_my_title_from_env(fallback="Gateway")
    except Exception:
        pass
    import uvicorn
    config = get_config()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.agents.gateway_port,
    )