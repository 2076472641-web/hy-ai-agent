# -*- coding: utf-8 -*-
"""
数据库工具 - 提供 MySQL 数据查询接口
生产环境中对接 MySQL 存储的旅游基础数据
"""
from __future__ import annotations

import json
from typing import Any, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from mcp.server import FastMCP
    except ImportError:
        FastMCP = None


def get_db_tools(mcp: "FastMCP") -> None:
    """注册数据库查询工具到 MCP Server"""

    @mcp.tool()
    async def query_destinations(keyword: str = "", limit: int = 20) -> str:
        """
        查询目的地城市列表。

        Args:
            keyword: 搜索关键词
            limit: 返回数量

        Returns:
            目的地列表 JSON
        """
        destinations = _query_destinations_mock(keyword, limit)
        return json.dumps(destinations, ensure_ascii=False)

    @mcp.tool()
    async def get_city_info(city_name: str) -> str:
        """
        获取城市基础信息。

        Args:
            city_name: 城市名称

        Returns:
            城市信息 JSON
        """
        info = _get_city_info_mock(city_name)
        return json.dumps(info, ensure_ascii=False)

    @mcp.tool()
    async def search_travel_packages(
        destination: str = "",
        days: int = 0,
        budget: str = "",
    ) -> str:
        """
        查询旅游套餐信息。

        Args:
            destination: 目的地
            days: 天数
            budget: 预算档位

        Returns:
            套餐列表 JSON
        """
        packages = _search_packages_mock(destination, days, budget)
        return json.dumps(packages, ensure_ascii=False)


# ---- 模拟数据库内容 ----

DESTINATIONS = [
    {"name": "北京", "level": "一线", "tags": ["历史文化", "政治中心", "四季分明"]},
    {"name": "上海", "level": "一线", "tags": ["国际化", "都市风光", "购物"]},
    {"name": "广州", "level": "一线", "tags": ["美食", "商业", "亚热带"]},
    {"name": "深圳", "level": "一线", "tags": ["科技", "主题乐园", "海滨"]},
    {"name": "成都", "level": "新一线", "tags": ["美食", "熊猫", "慢生活"]},
    {"name": "杭州", "level": "新一线", "tags": ["自然景观", "历史文化", "电商"]},
    {"name": "西安", "level": "新一线", "tags": ["历史文化", "美食", "古都"]},
    {"name": "重庆", "level": "新一线", "tags": ["夜景", "火锅", "魔幻"]},
    {"name": "三亚", "level": "二线", "tags": ["海滨", "度假", "热带"]},
    {"name": "大理", "level": "二线", "tags": ["自然景观", "民族风情", "文艺"]},
    {"name": "丽江", "level": "二线", "tags": ["古城", "雪山", "民族风情"]},
    {"name": "厦门", "level": "二线", "tags": ["海滨", "文艺", "慢生活"]},
]


def _query_destinations_mock(keyword: str, limit: int) -> dict:
    results = DESTINATIONS
    if keyword:
        results = [d for d in DESTINATIONS if keyword in d["name"] or any(keyword in t for t in d["tags"])]
    return {
        "total": len(results),
        "destinations": results[:limit],
    }


def _get_city_info_mock(city_name: str) -> dict:
    for d in DESTINATIONS:
        if d["name"] == city_name:
            return {
                "name": d["name"],
                "level": d["level"],
                "tags": d["tags"],
                "description": f"{city_name}是中国著名的旅游城市，以{d['tags'][0] if d['tags'] else '多样'}风情著称。",
                "best_season": _best_season(city_name),
                "famous_food": _famous_foods(city_name),
            }
    return {"error": f"未找到城市: {city_name}"}


def _best_season(city: str) -> str:
    map_data = {
        "北京": "秋季(9-11月)最佳", "上海": "春秋季(3-5月,9-11月)最佳",
        "广州": "冬季(11-2月)最佳", "成都": "春秋季最佳",
        "杭州": "春季(3-5月)最佳", "西安": "春秋季最佳",
        "三亚": "冬季(10-次年4月)最佳", "大理": "四季皆宜",
    }
    return map_data.get(city, "四季皆宜")


def _famous_foods(city: str) -> list:
    map_data = {
        "北京": ["北京烤鸭", "炸酱面", "豆汁焦圈"],
        "上海": ["小笼包", "本帮红烧肉", "生煎"],
        "广州": ["早茶", "白切鸡", "煲仔饭"],
        "成都": ["火锅", "串串香", "龙抄手"],
        "杭州": ["西湖醋鱼", "龙井虾仁", "东坡肉"],
        "西安": ["肉夹馍", "羊肉泡馍", "biangbiang面"],
        "三亚": ["海鲜大餐", "椰子鸡", "清补凉"],
    }
    return map_data.get(city, ["当地特色美食"])


def _search_packages_mock(destination: str, days: int, budget: str) -> dict:
    packages = []
    budget_map = {"经济": 800, "标准": 1500, "豪华": 3000}
    base_price = budget_map.get(budget or "标准", 1500)

    for i in range(3):
        pkg_days = days or 3
        packages.append({
            "id": f"pkg_{destination}_{i}",
            "name": f"{destination}{pkg_days}天深度游",
            "destination": destination,
            "days": pkg_days,
            "price": f"{base_price * pkg_days}元/人起",
            "highlights": [
                f"{destination}热门景点",
                "专业导游服务",
                "酒店住宿",
                "交通接送",
            ],
            "type": budget or "标准",
        })

    return {
        "total": len(packages),
        "packages": packages,
        "filters": {"destination": destination, "days": days, "budget": budget},
    }