# -*- coding: utf-8 -*-
"""
途景汇文旅多智能体系统 - MCP 统一工具服务层
提供统一的 MCP Server，封装所有工具接口
支持天气查询、景点推荐、路线规划、数据库查询等功能
"""
from __future__ import annotations

import os
import sys
import asyncio

from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP

from config import get_config
from tools.weather_tools import get_weather_tool
from tools.attraction_tools import get_attraction_tools
from tools.route_tools import get_route_tools
from tools.db_tools import get_db_tools


def create_mcp_server() -> FastMCP:
    """创建 MCP Server 实例并注册所有工具"""
    config = get_config()

    mcp = FastMCP(
        "途景汇文旅MCP服务",
        instructions="""
        你是途景汇文旅系统的 MCP 工具服务层，提供以下能力：
        1. 天气查询 - 查询目的地天气预报、穿衣建议
        2. 景点推荐 - 搜索景点、推荐行程、获取景点详情
        3. 路线规划 - 规划交通方案、行程路线
        4. 数据查询 - 查询目的地城市、旅游套餐信息

        请根据智能体的请求调用相应工具。
        """,
    )

    # 注册所有工具
    get_weather_tool(mcp)
    get_attraction_tools(mcp)
    get_route_tools(mcp)
    get_db_tools(mcp)

    return mcp


def main():
    """启动 MCP Server"""
    config = get_config()
    mcp = create_mcp_server()

    host = config.mcp.host
    port = config.mcp.port

    print(f"[MCP Server] Starting on {host}:{port}...")
    print(f"[MCP Server] Registered tools:")
    print(f"  - Weather: query_weather, get_weather_tips")
    print(f"  - Attraction: search_attractions, recommend_itinerary, get_attraction_detail")
    print(f"  - Route: plan_transport, plan_daily_routes, get_transport_tips")
    print(f"  - DB: query_destinations, get_city_info, search_travel_packages")

    mcp.run(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from _console_title_helper import set_my_title_from_env
        set_my_title_from_env(fallback="MCP Server")
    except Exception:
        pass
    main()