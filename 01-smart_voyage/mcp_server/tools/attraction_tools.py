# -*- coding: utf-8 -*-
"""
景点推荐工具
对接 MySQL 旅游基础数据，提供景点查询和推荐
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


def get_attraction_tools(mcp: "FastMCP", db_pool=None) -> None:
    """注册景点推荐相关工具到 MCP Server"""

    @mcp.tool()
    async def search_attractions(
        destination: str,
        category: str = "",
        max_results: int = 10,
    ) -> str:
        """
        按目的地和类型搜索景点。

        Args:
            destination: 目的地城市名称，如"成都"
            category: 景点类型：自然风光/历史文化/主题乐园/博物馆/美食
            max_results: 最多返回数量

        Returns:
            景点列表 JSON
        """
        attractions = _search_attractions_mock(destination, category, max_results)
        return json.dumps(attractions, ensure_ascii=False)

    @mcp.tool()
    async def recommend_itinerary(
        destination: str,
        days: int = 3,
        preferences: str = "",
        budget: str = "",
    ) -> str:
        """
        根据目的地和天数推荐行程安排。

        Args:
            destination: 目的地城市
            days: 游玩天数
            preferences: 偏好类型，逗号分隔，如"自然风光,历史文化"
            budget: 预算：经济/标准/豪华

        Returns:
            行程安排 JSON
        """
        itinerary = _generate_itinerary(destination, days, preferences, budget)
        return json.dumps(itinerary, ensure_ascii=False)

    @mcp.tool()
    async def get_attraction_detail(attraction_name: str) -> str:
        """
        获取指定景点的详细信息。

        Args:
            attraction_name: 景点名称

        Returns:
            景点详情 JSON
        """
        detail = _get_attraction_detail_mock(attraction_name)
        return json.dumps(detail, ensure_ascii=False)


# ---- 模拟数据（生产环境从 MySQL 读取） ----

ATTRACTION_DB = {
    "成都": [
        {"name": "大熊猫繁育研究基地", "category": "自然风光", "rating": 4.8, "ticket": "55元", "duration": "半天", "desc": "近距离观赏国宝大熊猫的最佳地点"},
        {"name": "宽窄巷子", "category": "历史文化", "rating": 4.5, "ticket": "免费", "duration": "2小时", "desc": "成都慢生活的代表，感受老成都韵味"},
        {"name": "锦里古街", "category": "历史文化", "rating": 4.6, "ticket": "免费", "duration": "2小时", "desc": "三国文化主题街区，特色小吃聚集地"},
        {"name": "武侯祠博物馆", "category": "博物馆", "rating": 4.7, "ticket": "50元", "duration": "2小时", "desc": "中国唯一的君臣合祀祠庙"},
        {"name": "杜甫草堂博物馆", "category": "博物馆", "rating": 4.5, "ticket": "50元", "duration": "2小时", "desc": "诗圣杜甫故居，感受唐诗文化"},
        {"name": "都江堰景区", "category": "自然风光", "rating": 4.8, "ticket": "80元", "duration": "一天", "desc": "世界文化遗产，古代水利工程奇迹"},
        {"name": "青城山", "category": "自然风光", "rating": 4.7, "ticket": "80元", "duration": "一天", "desc": "道教圣地，青城天下幽"},
        {"name": "春熙路", "category": "美食", "rating": 4.4, "ticket": "免费", "duration": "2小时", "desc": "成都最繁华的商业街，美食聚集地"},
        {"name": "望江楼公园", "category": "自然风光", "rating": 4.3, "ticket": "20元", "duration": "2小时", "desc": "百竹园中品茶观景"},
        {"name": "东郊记忆", "category": "历史文化", "rating": 4.5, "ticket": "免费", "duration": "2小时", "desc": "工业文化创意园区"},
    ],
    "北京": [
        {"name": "故宫博物院", "category": "历史文化", "rating": 4.9, "ticket": "60元", "duration": "半天", "desc": "世界文化遗产，明清皇宫"},
        {"name": "长城(八达岭)", "category": "历史文化", "rating": 4.8, "ticket": "40元", "duration": "半天", "desc": "万里长城最著名的段落之一"},
        {"name": "颐和园", "category": "历史文化", "rating": 4.7, "ticket": "30元", "duration": "半天", "desc": "皇家园林博物馆"},
        {"name": "天坛", "category": "历史文化", "rating": 4.7, "ticket": "34元", "duration": "2小时", "desc": "明清两代祭天的场所"},
        {"name": "天安门广场", "category": "历史文化", "rating": 4.6, "ticket": "免费", "duration": "1小时", "desc": "世界最大的城市广场"},
        {"name": "南锣鼓巷", "category": "美食", "rating": 4.4, "ticket": "免费", "duration": "2小时", "desc": "老北京胡同文化街区"},
        {"name": "798艺术区", "category": "历史文化", "rating": 4.5, "ticket": "免费", "duration": "2小时", "desc": "当代艺术聚集地"},
        {"name": "香山公园", "category": "自然风光", "rating": 4.5, "ticket": "10元", "duration": "半天", "desc": "秋季红叶胜地"},
    ],
    "上海": [
        {"name": "外滩", "category": "历史文化", "rating": 4.8, "ticket": "免费", "duration": "2小时", "desc": "黄浦江畔的万国建筑博览群"},
        {"name": "东方明珠", "category": "主题乐园", "rating": 4.6, "ticket": "160元", "duration": "2小时", "desc": "上海地标电视塔"},
        {"name": "豫园", "category": "历史文化", "rating": 4.5, "ticket": "40元", "duration": "2小时", "desc": "明代古典园林，感受江南韵味"},
        {"name": "南京路步行街", "category": "美食", "rating": 4.4, "ticket": "免费", "duration": "2小时", "desc": "中华商业第一街"},
        {"name": "上海博物馆", "category": "博物馆", "rating": 4.7, "ticket": "免费", "duration": "3小时", "desc": "馆藏丰富的综合性博物馆"},
        {"name": "迪士尼乐园", "category": "主题乐园", "rating": 4.8, "ticket": "475元", "duration": "一天", "desc": "亚洲最大的迪士尼主题公园"},
    ],
    "杭州": [
        {"name": "西湖", "category": "自然风光", "rating": 4.9, "ticket": "免费", "duration": "半天", "desc": "人间天堂，淡妆浓抹总相宜"},
        {"name": "灵隐寺", "category": "历史文化", "rating": 4.6, "ticket": "75元", "duration": "3小时", "desc": "江南禅宗五山之一"},
        {"name": "千岛湖", "category": "自然风光", "rating": 4.7, "ticket": "150元", "duration": "一天", "desc": "湖光山色，千岛星罗"},
        {"name": "宋城", "category": "主题乐园", "rating": 4.6, "ticket": "320元", "duration": "半天", "desc": "给我一天，还你千年"},
        {"name": "河坊街", "category": "美食", "rating": 4.4, "ticket": "免费", "duration": "2小时", "desc": "南宋御街，老字号聚集地"},
    ],
    "西安": [
        {"name": "秦始皇兵马俑", "category": "历史文化", "rating": 4.9, "ticket": "120元", "duration": "半天", "desc": "世界第八大奇迹"},
        {"name": "大雁塔", "category": "历史文化", "rating": 4.6, "ticket": "50元", "duration": "2小时", "desc": "唐代佛塔，西安地标"},
        {"name": "回民街", "category": "美食", "rating": 4.5, "ticket": "免费", "duration": "2小时", "desc": "西安特色美食聚集地"},
        {"name": "华清池", "category": "历史文化", "rating": 4.5, "ticket": "120元", "duration": "3小时", "desc": "唐明皇与杨贵妃的爱情圣地"},
        {"name": "华山", "category": "自然风光", "rating": 4.8, "ticket": "160元", "duration": "一天", "desc": "奇险天下第一山"},
    ],
    "三亚": [
        {"name": "亚龙湾", "category": "自然风光", "rating": 4.8, "ticket": "免费", "duration": "半天", "desc": "东方夏威夷，碧海银沙"},
        {"name": "天涯海角", "category": "自然风光", "rating": 4.5, "ticket": "68元", "duration": "3小时", "desc": "海南著名景点，浪漫圣地"},
        {"name": "蜈支洲岛", "category": "自然风光", "rating": 4.7, "ticket": "168元", "duration": "一天", "desc": "中国最美海岛之一"},
        {"name": "南山寺", "category": "历史文化", "rating": 4.6, "ticket": "129元", "duration": "半天", "desc": "海上观音，福寿南山"},
    ],
}


def _search_attractions_mock(destination: str, category: str, max_results: int) -> dict:
    """模拟景点搜索"""
    city_attractions = ATTRACTION_DB.get(destination, [])
    if not city_attractions:
        return {
            "destination": destination,
            "total": 0,
            "attractions": [],
            "message": f"暂无{destination}的景点数据",
        }

    results = city_attractions
    if category:
        results = [a for a in results if category in a.get("category", "")]

    # 按评分从高到低排序（同评分保持原稳定顺序）
    results.sort(key=lambda a: float(a.get("rating", 0) or 0), reverse=True)

    results = results[:max_results]
    return {
        "destination": destination,
        "total": len(results),
        "attractions": [
            {
                "name": a["name"],
                "category": a["category"],
                "rating": a["rating"],
                "ticket": a["ticket"],
                "duration": a["duration"],
                "description": a["desc"],
            }
            for a in results
        ],
    }


def _generate_itinerary(destination: str, days: int, preferences: str, budget: str) -> dict:
    """生成行程安排"""
    attractions = ATTRACTION_DB.get(destination, [])
    if not attractions:
        return {"error": f"暂无{destination}的景点数据"}

    prefs = [p.strip() for p in preferences.split(",") if p.strip()] if preferences else []
    if prefs:
        pool = [a for a in attractions if any(p in a["category"] for p in prefs)]
        if not pool:
            pool = attractions
    else:
        pool = attractions

    days = max(1, min(days, len(pool)))
    daily_plans = []
    for i in range(days):
        day_start = i * 2
        day_end = min(day_start + 2, len(pool))
        day_attractions = pool[day_start:day_end]
        if not day_attractions:
            day_attractions = pool[:2]

        daily_plans.append({
            "day": i + 1,
            "theme": f"探索{destination}第{i + 1}天",
            "attractions": [
                {
                    "name": a["name"],
                    "time_slot": _get_time_slot(j),
                    "duration": a["duration"],
                    "tips": f"建议游玩{a['duration']}，门票{a['ticket']}",
                }
                for j, a in enumerate(day_attractions)
            ],
            "food_recommendation": _get_food_recommendation(destination),
        })

    budget_notes = {
        "经济": "建议选择公共交通，经济型住宿，人均预算 500-800元/天",
        "标准": "建议打车+经济型酒店，人均预算 800-1500元/天",
        "豪华": "建议专车+星级酒店，人均预算 2000元以上/天",
    }

    return {
        "destination": destination,
        "days": days,
        "budget_level": budget or "标准",
        "daily_plans": daily_plans,
        "budget_notes": budget_notes.get(budget or "标准", budget_notes["标准"]),
        "tips": f"以上行程仅供参考，建议根据个人兴趣和体力调整。{destination}交通便利，景点间距离适中。",
    }


def _get_attraction_detail_mock(attraction_name: str) -> dict:
    """获取景点详情"""
    for city, attractions in ATTRACTION_DB.items():
        for a in attractions:
            if a["name"] == attraction_name:
                return {
                    "name": a["name"],
                    "city": city,
                    "category": a["category"],
                    "rating": a["rating"],
                    "ticket": a["ticket"],
                    "opening_hours": "08:00 - 18:00",
                    "recommended_duration": a["duration"],
                    "description": a["desc"],
                    "tips": [
                        "建议提前预约门票",
                        "旺季（节假日）人较多，建议早到",
                        "穿舒适的鞋子，部分景点需步行较多",
                    ],
                }
    return {"error": f"未找到景点: {attraction_name}"}


def _get_time_slot(index: int) -> str:
    slots = ["08:30-10:30", "10:30-12:30", "13:30-15:30", "15:30-17:30", "17:30-19:30"]
    return slots[index % len(slots)]


def _get_food_recommendation(destination: str) -> str:
    food_map = {
        "成都": "推荐尝试火锅、串串香、龙抄手、钟水饺",
        "北京": "推荐尝试北京烤鸭、炸酱面、豆汁焦圈",
        "上海": "推荐尝试小笼包、本帮菜、生煎",
        "杭州": "推荐尝试西湖醋鱼、龙井虾仁、东坡肉",
        "西安": "推荐尝试肉夹馍、羊肉泡馍、biangbiang面",
        "三亚": "推荐尝试海鲜大餐、椰子鸡、清补凉",
    }
    return food_map.get(destination, "推荐尝试当地特色美食")