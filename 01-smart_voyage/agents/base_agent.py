# -*- coding: utf-8 -*-
"""
智能体基类 - 提供 LLM 调用、MCP 工具加载、A2A 通信的基础能力
"""
from __future__ import annotations

import os
import sys
import json
import asyncio
from typing import Any, Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool as lc_tool

from config import get_config


def create_llm():
    """创建 LLM 实例（支持 DashScope 和 OpenAI 兼容）"""
    config = get_config()
    llm_config = config.llm

    if llm_config.base_url:
        # 使用 OpenAI 兼容的 API
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=llm_config.model,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            temperature=0.7,
        )
    else:
        # 使用 DashScope (阿里千问)
        try:
            from langchain_community.chat_models import ChatTongyi
            return ChatTongyi(
                model=llm_config.model,
                dashscope_api_key=llm_config.api_key,
                temperature=0.7,
            )
        except ImportError:
            # 回退到通用 ChatOpenAI
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=llm_config.model,
                api_key=llm_config.api_key,
                temperature=0.7,
            )


def create_mcp_tools():
    """从 MCP Server 加载工具（优化版：超时从 30 秒降到 3 秒，快速回退内置工具）"""
    try:
        from langchain_mcp_adapters.tools import load_mcp_tools
        from mcp.client.http import http_client

        config = get_config()
        mcp_url = f"http://{config.mcp.host}:{config.mcp.port}/mcp"

        # 先快速检测 MCP 是否可用（socket 连接检测，比完整 HTTP 快得多）
        import socket
        try:
            mcp_host = config.mcp.host
            if mcp_host in ("0.0.0.0", "localhost", "127.0.0.1"):
                mcp_host = "127.0.0.1"
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)  # 1 秒快速检测
            result = sock.connect_ex((mcp_host, int(config.mcp.port)))
            sock.close()
            if result != 0:
                # MCP 端口未开放，直接回退，不用等 HTTP 超时
                print(f"[Agent] MCP 端口未开放 ({config.mcp.port})，直接使用内置工具")
                return _get_builtin_tools()
        except Exception:
            pass

        async def _load():
            # MCP 工具加载也设置短超时
            import asyncio as _aio
            try:
                async with _aio.timeout(3):
                    async with http_client(mcp_url) as session:
                        tools = await load_mcp_tools(session)
                        return tools
            except (_aio.TimeoutError, Exception):
                return None

        # 在已有事件循环中运行（超时 3 秒，不是 30 秒！）
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _load())
                result = future.result(timeout=3)
                if result is None:
                    print(f"[Agent] MCP 加载超时，使用内置工具")
                    return _get_builtin_tools()
                return result
        except RuntimeError:
            try:
                result = asyncio.run(asyncio.wait_for(_load(), timeout=3))
                if result is None:
                    return _get_builtin_tools()
                return result
            except Exception:
                print(f"[Agent] MCP 加载失败，使用内置工具")
                return _get_builtin_tools()

    except Exception as e:
        print(f"[Agent] MCP 工具加载失败: {e}")
        print(f"[Agent] 使用内置工具作为回退")
        return _get_builtin_tools()


def _get_builtin_tools():
    """内置工具（无需 MCP Server 即可使用）"""

    @lc_tool
    def query_weather(location: str, date: str = "", days: int = 3) -> str:
        """查询指定城市的天气预报。location: 城市名; date: YYYY-MM-DD; days: 天数"""
        from mcp_server.tools.weather_tools import _mock_weather_query
        data = _mock_weather_query(location, date, days)
        return json.dumps(data, ensure_ascii=False)

    @lc_tool
    def search_attractions(destination: str, category: str = "", max_results: int = 10) -> str:
        """搜索指定城市的景点。destination: 城市; category: 分类; max_results: 数量"""
        from mcp_server.tools.attraction_tools import _search_attractions_mock
        data = _search_attractions_mock(destination, category, max_results)
        return json.dumps(data, ensure_ascii=False)

    @lc_tool
    def recommend_itinerary(destination: str, days: int = 3, preferences: str = "", budget: str = "") -> str:
        """推荐行程安排。destination: 城市; days: 天数; preferences: 偏好类型"""
        from mcp_server.tools.attraction_tools import _generate_itinerary
        data = _generate_itinerary(destination, days, preferences, budget)
        return json.dumps(data, ensure_ascii=False)

    @lc_tool
    def plan_transport(origin: str, destination: str, date: str = "", preference: str = "最快") -> str:
        """规划交通方案。origin: 出发城市; destination: 目的城市"""
        from mcp_server.tools.route_tools import _generate_transport_plans
        data = _generate_transport_plans(origin, destination, date, preference)
        return json.dumps(data, ensure_ascii=False)

    @lc_tool
    def plan_daily_routes(destination: str, attractions: str, days: int = 1) -> str:
        """规划每日交通路线。attractions: 景点逗号分隔"""
        from mcp_server.tools.route_tools import _generate_daily_routes
        att_list = [a.strip() for a in attractions.split(",") if a.strip()]
        data = _generate_daily_routes(destination, att_list, days)
        return json.dumps(data, ensure_ascii=False)

    return [query_weather, search_attractions, recommend_itinerary, plan_transport, plan_daily_routes]