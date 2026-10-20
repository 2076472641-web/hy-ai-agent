# -*- coding: utf-8 -*-
"""
天气查询工具
基于 MCP 协议实现天气查询功能
对接和风天气 API 或使用模拟数据
"""
from __future__ import annotations

import json
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    # 兼容旧版导入
    try:
        from mcp.server import FastMCP
    except ImportError:
        FastMCP = None


def get_weather_tool(mcp: "FastMCP") -> None:
    """注册天气查询工具到 MCP Server"""

    @mcp.tool()
    async def query_weather(location: str, date: str = "", days: int = 3) -> str:
        """
        查询指定城市/景点的天气预报。

        Args:
            location: 城市或景点名称，如"北京"、"三亚"
            date: 指定日期，格式 YYYY-MM-DD，为空则返回当天
            days: 预报天数，默认3天，最多7天

        Returns:
            天气信息 JSON 字符串
        """
        # 真实环境对接和风天气 API
        # 此处提供模拟数据结构
        weather_data = _mock_weather_query(location, date, days)
        return json.dumps(weather_data, ensure_ascii=False)

    @mcp.tool()
    async def get_weather_tips(location: str, season: str = "") -> str:
        """
        获取指定目的地的穿衣/出行建议。

        Args:
            location: 城市或景点名称
            season: 季节，可选：春/夏/秋/冬

        Returns:
            天气建议文本
        """
        tips = _generate_weather_tips(location, season)
        return json.dumps(tips, ensure_ascii=False)


def _mock_weather_query(location: str, date: str = "", days: int = 3) -> dict:
    """
    模拟天气查询（生产环境替换为真实 API 调用）
    修复 2 个 bug：
      1) 之前只有第 0 天用用户传入的 date，第 1..N 天直接 today()+i，导致跨月/跨年错乱（用户填 2-10，第二天跳到 8-11）
      2) 温度/季节不管用户输入的日期，固定用年平均：2 月冬天也显示 26℃，明显不对
    """
    import datetime, hashlib as _hash

    # ---------- 1. 计算行程每一天的真实日期（连续递增，跨月/跨年自动正确）----------
    try:
        start_date = datetime.date.fromisoformat(date) if date else datetime.date.today()
    except ValueError:
        start_date = datetime.date.today()
    days = max(1, min(int(days), 7))
    date_list = [start_date + datetime.timedelta(days=i) for i in range(days)]

    # ---------- 2. 季节系数：按行程第一天所在月份，调节温度范围 ----------
    #    月：  1  2  3   4   5   6   7   8   9  10  11 12
    SEASON_DELTA = [
        -14,-12, -6,  -2,  +2,  +6,  +9,  +9,  +5,   0,  -8,-13,
    ]  # (high_off, low_off) 会用它做加减
    m = start_date.month  # 1-12
    delta = SEASON_DELTA[m - 1]

    # ---------- 3. 基础温度：城市年平均 + 季节偏移（2月冬天减，8月夏天加）----------
    base_temps = {
        "北京": (22, 8), "上海": (26, 18), "广州": (30, 24),
        "深圳": (31, 25), "成都": (24, 16), "杭州": (25, 17),
        "三亚": (32, 26), "西安": (20, 10), "南京": (24, 14),
        "重庆": (25, 18), "武汉": (26, 16), "大理": (23, 15),
    }
    base_h, base_l = base_temps.get(location, (25, 15))
    season_high = max(-5, base_h + delta)  # 极端寒冬也不会 < -5
    season_low = min(season_high - 4, base_l + delta)  # 最低温至少比最高温低 4 度（昼夜温差）

    # ---------- 4. 每日可复现的天气（城市+日期 做 hash 种子，刷新不变）----------
    def _rng_for(day_date: datetime.date):
        key = f"{location}|{day_date.isoformat()}"
        seed = int(_hash.md5(key.encode("utf-8")).hexdigest()[:8], 16)
        import random as _r
        return _r.Random(seed)

    conditions = ["晴", "多云", "晴转多云", "阴", "小雨", "雷阵雨"]
    # 冬天把雨雪替换上去
    if m in (12, 1, 2):
        conditions = ["晴", "多云", "阴", "小雪", "雨夹雪", "晴转多云"]
    aqi_levels = ["优", "良", "轻度污染", "良", "优", "良", "中度污染", "优"]

    forecasts = []
    for i in range(days):
        d = date_list[i]
        rng = _rng_for(d)
        # 高低温围绕季节值做 1-3℃ 浮动
        high = season_high + rng.randint(-2, 3)
        low = season_low + rng.randint(-2, 2)
        if low >= high:
            low = high - rng.randint(3, 6)
        cond = rng.choice(conditions)
        # 雷阵雨/小雨天气稍微降低温度
        if cond in ("雷阵雨", "小雨", "雨夹雪", "小雪"):
            high = max(low + 2, high - 2)
        aqi = aqi_levels[rng.randrange(len(aqi_levels))]
        hum_base = 55 if m in (3, 4, 9, 10) else 70 if m in (6, 7, 8) else 60
        humidity = max(20, min(95, hum_base + rng.randint(-8, 12)))

        forecasts.append({
            "date": d.isoformat(),
            "condition": cond,
            "temperature": f"{low}~{high}℃",
            "aqi": aqi,
            "humidity": f"{humidity}%",
            "wind": ["东南风", "南风", "东北风", "西北风"][rng.randrange(4)] + f" {rng.randint(1,4)}级",
        })

    # ---------- 5. Summary：最高/最低温从整段行程里取 max/min，更准确；最低温补 ℃ ----------
    all_h = []; all_l = []
    for f in forecasts:
        try:
            t = f["temperature"].replace("℃","")
            lo_s, hi_s = t.split("~", 1)
            all_l.append(int(lo_s)); all_h.append(int(hi_s))
        except Exception:
            pass
    tmax = max(all_h) if all_h else season_high
    tmin = min(all_l) if all_l else season_low
    # 主导天气：统计各条件出现次数
    from collections import Counter as _C
    c_counter = _C(f["condition"] for f in forecasts)
    main_cond = c_counter.most_common(1)[0][0]

    return {
        "location": location,
        "query_date": date or forecasts[0]["date"],
        "forecasts": forecasts,
        "summary": (
            f"{location}未来{days}天以{main_cond}为主，"
            f"最高温{tmax}℃，最低温{tmin}℃"
        ),
    }


def _generate_weather_tips(location: str, season: str = "", start_date: str = "") -> dict:
    """
    生成天气/穿搭建议
    修复：用户填 2 月（冬季）出行时，穿搭提示要对应冬季，不能用 today()（8 月）给夏季短袖。
    新增 start_date 参数（YYYY-MM-DD），优先按它判断季节。
    """
    tips_map = {
        "春": {"clothing": "长袖衬衫、薄外套", "note": "春季温差大，建议备一件外套"},
        "夏": {"clothing": "短袖、短裤、凉鞋", "note": "注意防晒防暑，携带雨具"},
        "秋": {"clothing": "长袖、长裤、薄毛衣", "note": "秋季干燥，注意补水保湿"},
        "冬": {"clothing": "羽绒服、厚外套、保暖内衣", "note": "冬季寒冷，注意保暖防寒"},
    }
    import datetime
    if not season:
        # 优先用出行日期判断季节，再退回到今天
        try:
            m = datetime.date.fromisoformat(start_date).month if start_date else datetime.date.today().month
        except Exception:
            m = datetime.date.today().month
        if m in (3, 4, 5):
            season = "春"
        elif m in (6, 7, 8):
            season = "夏"
        elif m in (9, 10, 11):
            season = "秋"
        else:
            season = "冬"

    return {
        "location": location,
        "season": season,
        "clothing_advice": tips_map.get(season, tips_map["春"])["clothing"],
        "note": tips_map.get(season, tips_map["春"])["note"],
        "general": f"前往{location}旅游，建议穿着{tips_map.get(season, tips_map['春'])['clothing']}。",
    }