# -*- coding: utf-8 -*-
"""
出行路线规划工具
提供交通路线查询和行程规划
"""
from __future__ import annotations

import json
import hashlib
import random as _random
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from mcp.server import FastMCP
    except ImportError:
        FastMCP = None


def get_route_tools(mcp: "FastMCP") -> None:
    """注册路线规划相关工具到 MCP Server"""

    @mcp.tool()
    async def plan_transport(
        origin: str,
        destination: str,
        date: str = "",
        preference: str = "最快",
    ) -> str:
        """
        规划从出发地到目的地的交通方案。

        Args:
            origin: 出发城市，如"上海"
            destination: 目的城市，如"杭州"
            date: 出发日期，格式 YYYY-MM-DD
            preference: 偏好：最快/最便宜/最舒适

        Returns:
            交通方案 JSON
        """
        routes = _generate_transport_plans(origin, destination, date, preference)
        return json.dumps(routes, ensure_ascii=False)

    @mcp.tool()
    async def plan_daily_routes(
        destination: str,
        attractions: str,
        days: int = 1,
    ) -> str:
        """
        根据景点列表规划每日交通路线。

        Args:
            destination: 目的地城市
            attractions: 景点名称列表，逗号分隔
            days: 游玩天数

        Returns:
            每日路线规划 JSON
        """
        att_list = [a.strip() for a in attractions.split(",") if a.strip()]
        plan = _generate_daily_routes(destination, att_list, days)
        return json.dumps(plan, ensure_ascii=False)

    @mcp.tool()
    async def get_transport_tips(destination: str) -> str:
        """
        获取目的地的交通出行小贴士。

        Args:
            destination: 城市名称

        Returns:
            交通建议文本
        """
        tips = _get_transport_tips(destination)
        return json.dumps(tips, ensure_ascii=False)


def _generate_transport_plans(origin: str, destination: str, date: str, preference: str) -> dict:
    """生成交通方案"""
    # 模拟路线数据
    distance_map = {
        ("上海", "杭州"): {"distance": 180, "transport": ["高铁", "汽车"]},
        ("上海", "北京"): {"distance": 1200, "transport": ["飞机", "高铁"]},
        ("广州", "深圳"): {"distance": 140, "transport": ["高铁", "地铁"]},
        ("成都", "西安"): {"distance": 700, "transport": ["高铁", "飞机"]},
        ("北京", "西安"): {"distance": 1200, "transport": ["高铁", "飞机"]},
        ("上海", "广州"): {"distance": 1500, "transport": ["飞机", "高铁"]},
    }

    key = (origin, destination)
    rev_key = (destination, origin)
    info = distance_map.get(key) or distance_map.get(rev_key)

    if not info:
        # 通用方案
        info = {"distance": 500, "transport": ["高铁", "汽车"]}

    routes = []
    for t in info["transport"]:
        if t == "飞机":
            routes.append({
                "type": "飞机",
                "duration": "约2小时",
                "price": f"{int(info['distance'] * 0.8)}元",
                "departure": f"{origin}虹桥/浦东机场",
                "arrival": f"{destination}国际机场",
                "advantage": "最快，适合远距离出行",
                "disadvantage": "受天气影响，需提前2小时到达",
            })
        elif t == "高铁":
            duration_h = max(1, int(info["distance"] / 300))
            routes.append({
                "type": "高铁",
                "duration": f"约{duration_h}小时",
                "price": f"{int(info['distance'] * 0.3)}元",
                "departure": f"{origin}火车站",
                "arrival": f"{destination}火车站",
                "advantage": "准点率高，舒适便捷",
                "disadvantage": "可能出现票源紧张",
            })
        elif t == "汽车":
            duration_h = max(1, int(info["distance"] / 80))
            routes.append({
                "type": "汽车",
                "duration": f"约{duration_h}小时",
                "price": f"{int(info['distance'] * 0.15)}元",
                "departure": f"{origin}长途汽车站",
                "arrival": f"{destination}汽车站",
                "advantage": "票价便宜，灵活方便",
                "disadvantage": "耗时较长",
            })
        elif t == "地铁":
            routes.append({
                "type": "地铁/城际",
                "duration": "约1小时",
                "price": f"{int(info['distance'] * 0.05)}元",
                "departure": f"{origin}地铁始发站",
                "arrival": f"{destination}地铁终点站",
                "advantage": "便捷无堵车",
                "disadvantage": "运营时间有限",
            })

    # 根据偏好排序
    if preference == "最快":
        routes.sort(key=lambda x: _parse_duration(x["duration"]))
    elif preference == "最便宜":
        routes.sort(key=lambda x: _parse_price(x["price"]))

    best = routes[0] if routes else None

    return {
        "origin": origin,
        "destination": destination,
        "date": date or "灵活",
        "distance_km": info["distance"],
        "routes": routes,
        "recommendation": f"推荐选择{best['type']}，{best['duration']}，约{best['price']}。{best['advantage']}。" if best else "暂无推荐路线",
        "tips": "建议提前预订票务，节假日高峰需提前更多时间安排。",
    }


def _parse_duration(duration: str) -> int:
    """解析时长为小时"""
    import re
    nums = re.findall(r"[\d.]+", duration)
    if nums:
        return int(float(nums[0]))
    return 999


def _parse_price(price: str) -> int:
    """解析价格"""
    import re
    nums = re.findall(r"[\d.]+", price)
    if nums:
        return int(float(nums[0]))
    return 999999


def _generate_daily_routes(destination: str, attractions: list, days: int) -> dict:
    """
    生成每日路线（升级版：可复现随机 + 真实感差异化，不再是 20 分钟/3-5-7km 模板）
    - 基于【城市名 + 景点列表】生成固定 seed，保证同一输入每次结果一致，不瞎跳
    - 每个景点「虚拟坐标」通过名 hash 映射到市区/近郊不同象限 → 距离计算有地理感
    - 交通方式按距离智能选：步行(<1km) / 骑行(<3km) / 公交/地铁(1-10km) / 打车(>5km 或景点特殊)
    - 时长/花费按距离和交通模式计算，不再硬编码 20 分钟/30 元
    - 每天游玩总时长 = SUM(每段交通时长) + SUM(景点预计游玩时长)
    - 每天交通总花费 = SUM(每段交通花费)
    """
    days = max(1, int(days))

    # --------- 1. 准备可复现随机源 + 兜底景点补齐 ---------
    def _seed_for(city: str, attrs: list) -> int:
        key = f"{city}|{';'.join(attrs)}"
        return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)

    rng = _random.Random(_seed_for(destination, attractions or []))

    # 兜底：景点不够循环填充（尽量用原名 + 序号延伸）
    if not attractions:
        attractions = [f"{destination}核心景点{i+1}" for i in range(days * 3)]
    min_needed = days * 2
    if len(attractions) < min_needed:
        padded, idx = list(attractions), 0
        while len(padded) < min_needed:
            if idx < len(attractions):
                padded.append(f"{attractions[idx]}(延伸游览)")
            else:
                padded.append(f"{destination}补充景点{idx - len(attractions) + 1}")
            idx += 1
        attractions = padded

    # --------- 2. 给每个景点分配一个「虚拟地理坐标」(x,y ∈ [0,10])，基于名字 hash ---------
    #  城市中心区域(3-7 区间)放 70% 热门景点，近郊(0-3/7-10)放迪士尼/名山/古镇等外围大景点
    OUTSKIRT_KEYWORDS = ("迪士尼", "乐园", "古镇", "名山", "风景区", "机场", "滑雪", "长城", "古村", "寺庙", "大佛", "森林公园")

    def _coord(name: str):
        h = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:10], 16)
        is_outskirt = any(k in name for k in OUTSKIRT_KEYWORDS) or "补充景点" in name or "延伸" in name
        if is_outskirt:
            # 近郊：x,y 靠边缘
            band = (h >> 4) & 0xF  # 0~15
            if band < 4:
                x = rng.uniform(0.5, 2.5); y = rng.uniform(0.5, 9.5)
            elif band < 8:
                x = rng.uniform(7.5, 9.5); y = rng.uniform(0.5, 9.5)
            elif band < 12:
                x = rng.uniform(0.5, 9.5); y = rng.uniform(0.5, 2.5)
            else:
                x = rng.uniform(0.5, 9.5); y = rng.uniform(7.5, 9.5)
        else:
            x = rng.uniform(3.2, 6.8); y = rng.uniform(3.2, 6.8)
        return round(x, 2), round(y, 2)

    coords = {a: _coord(a) for a in attractions}

    # --------- 3. 按「近邻分组」把景点分到天 ---------
    # 简单做法：选 days 个中心点（随机景点），其他景点分配给最近的中心点；再按天均分
    n_attrs = len(attractions)
    if n_attrs <= days:
        day_groups = [[attractions[i % n_attrs]] for i in range(days)]
    else:
        # k-means 风格 1 次迭代
        centers_idx = sorted(rng.sample(range(n_attrs), days))
        centers = [coords[attractions[i]] for i in centers_idx]
        buckets = [[] for _ in range(days)]
        for a in attractions:
            x, y = coords[a]
            dists = [(x-cx)**2 + (y-cy)**2 for cx, cy in centers]
            bidx = min(range(days), key=lambda i: dists[i])
            buckets[bidx].append(a)
        # 均衡：bucket 太多就匀给少的（保证每天 2-4 个景点最舒服）
        target_per_day = max(2, n_attrs // days)
        day_groups = [[] for _ in range(days)]
        leftover = []
        for i in range(days):
            take = buckets[i][:target_per_day]
            day_groups[i].extend(take)
            leftover.extend(buckets[i][target_per_day:])
        # 剩余景点按最近天塞进去，不要超过 target+2
        for a in leftover:
            x, y = coords[a]
            dists = []
            for i in range(days):
                cx = sum(coords[b][0] for b in day_groups[i]) / max(1, len(day_groups[i]))
                cy = sum(coords[b][1] for b in day_groups[i]) / max(1, len(day_groups[i]))
                dists.append(((x-cx)**2 + (y-cy)**2, i))
            dists.sort()
            for _, i in dists:
                if len(day_groups[i]) < target_per_day + 2:
                    day_groups[i].append(a); break
            else:
                day_groups[dists[0][1]].append(a)
        # 防止空天
        for i in range(days):
            if not day_groups[i]:
                day_groups[i] = [attractions[i % n_attrs]]

    # --------- 4. 每天内部按地理顺序排序（酒店 → 景点按最近邻居路径，减少折返）---------
    # 酒店假设在城市中心 (5, 5)
    hotel_pos = (5.0, 5.0)

    def _nearest_sort(items: list, start_pos):
        remaining = list(items)
        ordered, cur = [], start_pos
        while remaining:
            remaining.sort(key=lambda it: (coords[it][0]-cur[0])**2 + (coords[it][1]-cur[1])**2)
            nxt = remaining.pop(0)
            ordered.append(nxt)
            cur = coords[nxt]
        return ordered

    for i in range(days):
        day_groups[i] = _nearest_sort(day_groups[i], hotel_pos)

    # --------- 5. 景点预计游玩时长（小时），按关键词估算 ---------
    PLAY_KEYWORDS = [
        (("迪士尼", "乐园", "主题公园", "欢乐谷", "世界之窗"), 6.0, 8.0),
        (("博物馆", "科技馆", "美术馆", "故宫", "布达拉宫", "兵马俑", "石窟"), 3.0, 5.0),
        (("古镇", "古村", "古城", "历史文化街区", "步行街", "创意园"), 2.5, 4.0),
        (("名山", "风景区", "森林公园", "湿地", "滑雪", "登山", "动物园", "植物园"), 4.0, 6.0),
        (("外滩", "广场", "地标", "塔", "公园", "滨江", "湖景", "教堂", "寺庙"), 1.5, 3.0),
    ]

    def _play_hours(name: str) -> float:
        for kws, lo, hi in PLAY_KEYWORDS:
            if any(k in name for k in kws):
                return round(rng.uniform(lo, hi), 1)
        return round(rng.uniform(1.5, 3.0), 1)

    # --------- 6. 交通方式 & 时长 & 花费 计算 ---------
    # 坐标 1 单位 ≈ 2.5 km（所以 10×10 虚拟城市对角线约 35km，比较合理）
    KM_PER_UNIT = 2.5
    SPEED = {  # km/h
        "步行": 4.5,
        "共享单车": 14,
        "公交": 18,
        "地铁": 32,
        "打车": 28,
    }
    COST_BASE = {
        "步行": (0, 0),
        "共享单车": (1.5, 0),
        "公交":   (2, 0.1),  # 起步价 + 每公里少量加价(空调公交)
        "地铁":   (3, 0.6),
        "打车":   (14, 2.7),  # 起步价14元（3km内）+ 每公里 ~2.7 元（一线城市）
    }

    def _choose_transport(dist_km: float, from_place: str, to_place: str) -> str:
        # 距离越远越倾向地铁/打车；酒店出发首段更可能打车/地铁
        if dist_km < 0.8:
            return "步行"
        if dist_km < 2.8 and rng.random() < 0.75:
            return "共享单车" if rng.random() < 0.6 else "步行"
        if dist_km <= 10:
            # 1~10 km 市区主力是地铁/公交，偶尔打车
            roll = rng.random()
            from_hotel = "酒店" in from_place
            if roll < 0.55: return "地铁"
            if roll < 0.85: return "公交"
            return "打车"
        # >10km 直接打车/地铁
        if rng.random() < 0.6:
            return "打车"
        return "地铁"

    def _trip(from_pos, to_pos, from_place: str, to_place: str):
        dx = from_pos[0] - to_pos[0]; dy = from_pos[1] - to_pos[1]
        dist_km = round(max(0.3, (dx*dx + dy*dy)**0.5) * KM_PER_UNIT, 1)
        t = _choose_transport(dist_km, from_place, to_place)
        # 时长：交通时间 + 等车/接驳步行 5-12 分钟
        travel_h = dist_km / max(1, SPEED[t])
        access_min = rng.randint(5, 12) if t not in ("步行", "共享单车") else 0
        total_min = int(travel_h * 60 + access_min + 0.5)
        if total_min < 3: total_min = 3
        # 费用
        if t == "步行":
            cost = 0
        elif t == "共享单车":
            cost = round(max(1.5, 1.5 + (dist_km - 1) * 0.5), 1)
        else:
            base, per = COST_BASE[t]
            extra_km = max(0, dist_km - 3) if t == "打车" else max(0, dist_km - 1) if t == "地铁" else 0
            cost = round(base + extra_km * per, 1)
            if t == "打车":
                # 打车加拥堵费：+0~20%
                cost = round(cost * rng.uniform(1.0, 1.2), 1)
        duration_text = f"约{total_min}分钟"
        return {
            "transport": t,
            "duration": duration_text,
            "distance_km": dist_km,
            "_cost_yuan": cost,
        }

    # --------- 7. 生成每天路线 ---------
    daily_plans = []
    city_avg_cost = 30 if destination in ("北京", "上海", "深圳", "广州", "杭州") else 20

    for i in range(days):
        day_attrs = day_groups[i]
        routes = []
        total_cost = 0.0
        total_transport_min = 0
        total_play_h = 0.0

        cur_pos = hotel_pos
        prev_name = f"{destination}酒店"
        for attr in day_attrs:
            trip = _trip(cur_pos, coords[attr], prev_name, attr)
            trans = trip["transport"]
            # 步行/共享单车的文案更友好
            transport_text = trans
            cost_y = trip["_cost_yuan"]
            if trans == "步行":
                transport_text = "步行"
            elif trans == "共享单车":
                transport_text = "共享单车/步行"
            elif trans == "打车":
                transport_text = "打车（网约车）"
            elif trans == "公交":
                transport_text = "公交"
            elif trans == "地铁":
                transport_text = "地铁/公交"
            routes.append({
                "from": prev_name,
                "to": attr,
                "transport": transport_text,
                "duration": trip["duration"],
                "distance_km": trip["distance_km"],
                "_cost": cost_y,
            })
            total_cost += cost_y
            # 解析分钟
            import re as _re
            ms = _re.findall(r"\d+", trip["duration"])
            total_transport_min += int(ms[0]) if ms else 15
            total_play_h += _play_hours(attr)
            cur_pos = coords[attr]
            prev_name = attr

        # 当天总耗时：交通(小时化) + 游玩 + 午餐/休息时间 1~1.5 小时
        lunch_h = round(rng.uniform(1.0, 1.5), 1)
        total_h = round(total_transport_min / 60.0 + total_play_h + lunch_h, 1)
        # 加一点打车/地铁之外的杂项花费：地铁卡/公交卡小额/偶尔饮料，整体让数字更自然
        misc = round(rng.uniform(3, max(5, city_avg_cost - 5)), 1)
        day_cost = int(round(total_cost + misc, 0))
        # 极端值兜底
        day_cost = max(15, min(day_cost, 260))
        total_h = max(3.5, min(total_h, 11.5))

        daily_plans.append({
            "day": i + 1,
            "attractions": day_attrs,
            "routes": routes,
            "estimated_total_time": f"约{total_h:g}小时",
            "transport_cost": f"约{day_cost}元",
            "_play_details": [f"{a}约{_play_hours(a):g}h" for a in day_attrs],  # 调试用
        })

    # --------- 8. 目的地特色交通贴士（从 tips_map 取，如果存在）---------
    general_default = f"{destination}市内交通以地铁和打车为主，景点间距离适中。建议购买当地交通卡或使用移动支付。"
    try:
        tip_info = _get_transport_tips(destination)
        metro = tip_info.get("metro_tips", "")
        traffic = tip_info.get("traffic_tips", "")
        special = tip_info.get("special_notes", "")
        if metro and traffic and special:
            general_tips = f"🚇 {metro}；🚦 {traffic}；💡 {special}"
        else:
            general_tips = general_default
    except Exception:
        general_tips = general_default

    return {
        "destination": destination,
        "total_days": days,
        "daily_plans": daily_plans,
        "general_tips": general_tips,
    }


def _get_transport_tips(destination: str) -> dict:
    """获取目的地交通提示"""
    tips_map = {
        "北京": {
            "metro": "北京地铁网络发达，覆盖所有主要景点，推荐购买日票",
            "traffic": "早晚高峰拥堵严重，建议避开 7:30-9:00 和 17:30-19:00",
            "special": "部分景点需要预约，如故宫、国博等",
        },
        "上海": {
            "metro": "上海地铁便捷，支持Metro大都会APP扫码",
            "traffic": "陆家嘴、南京路等商圈人流量大，注意安全",
            "special": "迪士尼需提前购票，机场地铁24小时运营",
        },
        "成都": {
            "metro": "成都地铁覆盖主要景点，天府通卡可享折扣",
            "traffic": "春秋季节多雨，建议随身携带雨具",
            "special": "大熊猫基地建议早上前往，熊猫最活跃",
        },
        "杭州": {
            "metro": "杭州地铁便利，西湖景区可乘地铁1号线",
            "traffic": "景区周边节假日限行，建议公共交通",
            "special": "骑行西湖是热门体验，注意租车规范",
        },
        "三亚": {
            "metro": "三亚暂无地铁，推荐打车或租车",
            "traffic": "机场到市区约40分钟，建议预约接送",
            "special": "防晒必备，建议带SPF50+防晒霜",
        },
    }

    tips = tips_map.get(destination, {
        "metro": "当地交通以打车和公交为主",
        "traffic": "注意当地交通规则和习惯",
        "special": "建议提前了解当地交通情况",
    })

    return {
        "destination": destination,
        "metro_tips": tips["metro"],
        "traffic_tips": tips["traffic"],
        "special_notes": tips["special"],
    }