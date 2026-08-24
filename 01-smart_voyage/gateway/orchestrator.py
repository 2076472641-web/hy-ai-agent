# -*- coding: utf-8 -*-
"""
智能体编排器 - 实现多智能体协同调度
负责路由用户请求到合适的智能体，并协调多个智能体协作完成复杂任务
"""
from __future__ import annotations

import json
import asyncio
import time
import re
import datetime
from typing import Any, Optional

import httpx

from config import get_config
from a2a.protocol import AgentTask, TaskStatus
from a2a.transport import A2ATransport
from memory import ConversationMemory, ContextCompressor

# ============================================================
# 【性能优化】预加载 Mock 工具模块，避免首次调用时 2~3 秒的 import 延迟
# 原代码把这些 import 写在 plan_full_trip / _local_fallback 函数内部，
# 导致用户第一次点击"生成旅游方案"时才加载模块，体感非常慢。
# ============================================================
from mcp_server.tools.weather_tools import _mock_weather_query
from mcp_server.tools.attraction_tools import (
    _search_attractions_mock,
    _generate_itinerary,
    ATTRACTION_DB,
)
from mcp_server.tools.route_tools import (
    _generate_transport_plans,
    _generate_daily_routes,
)


AGENT_ROUTES = {
    "weather": "weather_agent",
    "天气": "weather_agent",
    "气温": "weather_agent",
    "穿衣": "weather_agent",

    "景点": "attraction_agent",
    "推荐": "attraction_agent",
    "好玩": "attraction_agent",
    "游玩": "attraction_agent",
    "旅游": "attraction_agent",
    "行程": "attraction_agent",
    "攻略": "attraction_agent",

    "路线": "route_agent",
    "交通": "route_agent",
    "出行": "route_agent",
    "怎么去": "route_agent",
    "飞机": "route_agent",
    "高铁": "route_agent",
    "火车": "route_agent",
}


# 所有已知中文城市（用于优先匹配，不在列表里也会通过正则兜底）
_ALL_CHINESE_CITIES = [
    "北京", "上海", "广州", "深圳", "成都", "杭州", "三亚", "西安", "南京", "重庆",
    "武汉", "大理", "丽江", "厦门", "苏州", "青岛", "长沙", "天津", "大连", "宁波",
    "无锡", "福州", "合肥", "南昌", "济南", "郑州", "沈阳", "哈尔滨", "昆明", "贵阳",
    "拉萨", "乌鲁木齐", "兰州", "西宁", "银川", "呼和浩特", "太原", "石家庄",
    "桂林", "北海", "张家界", "凤凰", "九寨沟", "峨眉山", "乐山大佛", "稻城",
    "西双版纳", "香格里拉", "腾冲", "珠海", "澳门", "香港", "台湾", "阿里山",
]

# ============================================================
# 城市线级划分 + 物价系数（用于动态调整所有价格/预算/酒店/餐饮区间）
# 物价系数以"二线城市"为基准(1.0)；一线上浮，三四线及以下下调
# ============================================================
CITY_TIER_PRICE = {
    # 🔴 一线（1.5）：北上广深
    "北京": ("一线", 1.50), "上海": ("一线", 1.50), "广州": ("一线", 1.50), "深圳": ("一线", 1.50),
    # 🟠 新一线（1.25）：核心强市
    "成都": ("新一线", 1.25), "杭州": ("新一线", 1.25), "重庆": ("新一线", 1.25), "武汉": ("新一线", 1.25),
    "西安": ("新一线", 1.25), "苏州": ("新一线", 1.25), "南京": ("新一线", 1.25), "天津": ("新一线", 1.25),
    "长沙": ("新一线", 1.25), "郑州": ("新一线", 1.25), "东莞": ("新一线", 1.25), "青岛": ("新一线", 1.25),
    "合肥": ("新一线", 1.25), "佛山": ("新一线", 1.25), "宁波": ("新一线", 1.25), "厦门": ("新一线", 1.25),
    "昆明": ("新一线", 1.25), "沈阳": ("新一线", 1.25), "济南": ("新一线", 1.25), "无锡": ("新一线", 1.25),
    "大连": ("新一线", 1.25), "福州": ("新一线", 1.25), "厦门": ("新一线", 1.25), "珠海": ("新一线", 1.25),
    # 🟡 二线（1.00）：基准物价
    "南昌": ("二线", 1.00), "贵阳": ("二线", 1.00), "南宁": ("二线", 1.00), "温州": ("二线", 1.00),
    "长春": ("二线", 1.00), "石家庄": ("二线", 1.00), "太原": ("二线", 1.00), "哈尔滨": ("二线", 1.00),
    "兰州": ("二线", 1.00), "呼和浩特": ("二线", 1.00), "乌鲁木齐": ("二线", 1.00), "徐州": ("二线", 1.00),
    "绍兴": ("二线", 1.00), "金华": ("二线", 1.00), "嘉兴": ("二线", 1.00), "惠州": ("二线", 1.00),
    "中山": ("二线", 1.00), "保定": ("二线", 1.00), "常州": ("二线", 1.00), "南通": ("二线", 1.00),
    # 🟢 热门旅游城市（按当地实际消费力取0.9~1.15）
    "三亚": ("热门旅游", 1.15), "大理": ("热门旅游", 0.95), "丽江": ("热门旅游", 0.95),
    "桂林": ("热门旅游", 0.90), "北海": ("热门旅游", 0.90), "张家界": ("热门旅游", 0.95),
    "西双版纳": ("热门旅游", 1.00), "香格里拉": ("热门旅游", 0.95), "腾冲": ("热门旅游", 0.90),
    "稻城": ("热门旅游", 0.90), "拉萨": ("热门旅游", 1.05), "凤凰": ("热门旅游", 0.80),
    "九寨沟": ("热门旅游", 1.00), "峨眉山": ("热门旅游", 0.95),
    # 🔵 其他普通地级市/三四线（0.70~0.80）：没有特别列在上面的一律归到这里
}
# 三四线城市明确名单
TIER34_CITIES = ["郴州", "汕头", "湛江", "江门", "肇庆", "清远", "韶关", "梅州", "阳江", "潮州", "揭阳",
                 "衡阳", "岳阳", "常德", "益阳", "株洲", "湘潭", "邵阳", "永州", "怀化", "娄底",
                 "芜湖", "蚌埠", "淮南", "马鞍山", "淮北", "铜陵", "安庆", "黄山", "阜阳", "宿州",
                 "赣州", "上饶", "九江", "宜春", "吉安", "抚州", "景德镇", "萍乡", "新余", "鹰潭",
                 "宜宾", "泸州", "德阳", "绵阳", "南充", "乐山", "达州", "内江", "自贡", "遂宁",
                 "柳州", "桂林", "梧州", "北海", "钦州", "玉林", "百色", "贺州", "河池", "来宾",
                 "咸阳", "宝鸡", "渭南", "铜川", "延安", "榆林", "汉中", "安康", "商洛",
                 "洛阳", "开封", "南阳", "新乡", "焦作", "许昌", "平顶山", "安阳", "信阳",
                 "潍坊", "烟台", "威海", "临沂", "济宁", "淄博", "泰安", "德州", "聊城",
                 "银川", "西宁", "海口", "三亚"]  # 海口/三亚已在热门旅游覆盖，这里重复不影响


def _get_city_price_context(city: str):
    """
    获取一个城市的【线级标签 + 物价系数 + 是否属于小众高端缺失城市】
    返回：(tier_tag, price_factor, is_sm_city)
      - tier_tag: '一线' / '新一线' / '二线' / '热门旅游' / '三四线及以下'
      - price_factor: float，以二线为1.0的乘数
      - is_sm_city: bool，True=这座城市大概率没有五星酒店/高端私厨，生成高端档时需要降级描述
    """
    city_clean = (city or "").strip().replace("市", "")
    if not city_clean:
        return ("三四线及以下", 0.70, True)
    # 1. 明确在映射表里
    if city_clean in CITY_TIER_PRICE:
        tier, factor = CITY_TIER_PRICE[city_clean]
        # 小众高端缺失判定：热门旅游高海拔/偏远 + 三四线 → True
        is_sm = (tier in ("热门旅游",) and city_clean not in ("三亚", "珠海")) or (tier == "三四线及以下")
        return (tier, factor, is_sm)
    # 2. 明确在三四线名单里
    if city_clean in TIER34_CITIES:
        # 热门旅游城市已经被上面覆盖过了，剩下的纯普通地级市
        return ("三四线及以下", 0.75, True)
    # 3. 兜底：当作普通三四线及以下（但系数更保守）
    return ("三四线及以下", 0.65, True)


def _normalize_budget_level(raw: str) -> str:
    """
    档位名称归一化：返回'经济型'/'舒适型'/'高端型'
    兼容旧值：'标准'/'标准型' → 统一返回 '舒适型'（用户要求档位名不变：经济型/舒适型/高端型）
    """
    r = str(raw or "").strip()
    # 经济型匹配
    if r in ("经济型", "经济", "cheap", "budget", "省钱", "便宜"):
        return "经济型"
    # 高端型匹配
    if r in ("高端型", "高端", "luxury", "premium", "豪华", "品质"):
        return "高端型"
    # 舒适型（含旧的标准/标准型，老请求兼容）
    if r in ("标准型", "标准", "舒适型", "舒适", "standard", "balanced", "normal"):
        return "舒适型"
    return "舒适型"  # 默认档


def _calculate_travel_days(dates_str: str = "", days_input_raw=None, destination_hint: str = "") -> int:
    """
    严格按用户 5 条规则计算出行天数，返回纯整数（1~30）。
    规则：
      1. 客观约束：出行天数≥1，必须为正整数，负数 / 0 / 小数 → 修正到合法区间
      2. 上限控制：>30 统一截断为 30（避免 token 爆炸）
      3. 日期计算：明确出发/返回日期范围 → (返回日期 - 出发日期 + 1) 天；结果≤0 → 1
      4. 模糊描述（既无合法 days 也无合法日期范围）：合理估算，默认 3 天，在 1~30 区间内
      5. 返回值：纯整数，无任何附加描述
    """
    # ----------- Step 1: 先尝试解析日期范围（规则 3）----------
    date_days = None
    ds = (dates_str or "").strip()
    if ds:
        # 分隔符支持：~ / ～ / 至 / 到 / - / — / 空格+to+空格 / 连续的日期 YYYY-MM-DD YYYY-MM-DD
        parts = None
        for sep in ["~", "～", "至", "到", "-", "—", "到", " "]:
            if sep in ds:
                # 针对单个 '-' 的场景：YYYY-MM-DD 里也有 '-', 需要用更精准的 split
                if sep == "-":
                    # 找最后一个 '-' 出现位置前面的 ')' 或最后两个数字后 '-' 的模式
                    m = re.match(
                        r"^\s*(\d{4}-\d{1,2}-\d{1,2})\s*[-~～至到]\s*(\d{4}-\d{1,2}-\d{1,2})\s*$",
                        ds,
                    )
                    if m:
                        parts = [m.group(1), m.group(2)]
                else:
                    tokens = [p.strip() for p in ds.split(sep) if p.strip()]
                    if len(tokens) >= 2:
                        parts = tokens[:2]
                if parts:
                    break
        # 如果正则或split没匹配到日期范围，尝试匹配 "两个连续日期紧挨着"
        if not parts:
            m = re.match(
                r"^\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2})\s*$",
                ds,
            )
            if m:
                parts = [m.group(1), m.group(2)]

        if parts and len(parts) == 2:
            try:
                d1_s = parts[0].replace("/", "-")
                d2_s = parts[1].replace("/", "-")
                d1 = datetime.date.fromisoformat(d1_s)
                d2 = datetime.date.fromisoformat(d2_s)
                diff = (d2 - d1).days + 1  # 旅行持续天数（含首尾）
                date_days = diff if diff > 0 else 1
            except Exception:
                date_days = None

    # ----------- Step 2: 尝试解析用户传入的 days 数字（规则 1/2）----------
    raw_days = None
    if days_input_raw is not None and str(days_input_raw).strip() != "":
        try:
            # 允许 float（如 "3.5"）→ 截断为整数（规则 1：不允许小数）
            raw_val = float(str(days_input_raw).strip())
            raw_days = int(raw_val)  # 截尾，不四舍五入；例如 3.9 → 3
        except Exception:
            raw_days = None

    # ----------- Step 3: 优先级：date_days > raw_days > 估算值（规则 4）----------
    final = None
    if date_days is not None:
        final = date_days
    elif raw_days is not None:
        final = raw_days
    else:
        # 模糊描述：默认 3 天（常见短途旅行，在 1~30 区间内）
        final = 3

    # ----------- Step 4: 强制应用 1~30 区间约束（规则 1/2）----------
    if final < 1:
        final = 1
    if final > 30:
        final = 30
    return int(final)


class AgentOrchestrator:
    """
    智能体编排器
    - 根据用户意图路由到合适的智能体
    - 支持多智能体协同（如：先查天气 → 再推景点 → 最后规划路线）
    - 管理会话状态和上下文压缩
    """

    def __init__(self):
        self.config = get_config()
        self.memory = ConversationMemory()
        self.compressor = ContextCompressor(self.memory)

        # 智能体端点映射
        self.agent_endpoints = {
            "weather_agent": self.config.agents.weather_url,
            "attraction_agent": self.config.agents.attraction_url,
            "route_agent": self.config.agents.route_url,
        }

        # A2A 传输层
        self.transport = A2ATransport(
            agent_name="orchestrator",
            base_url="",
        )

    # ------------------------------------------------------------------
    # 🌟 城市统一提取 & 强制矫正（解决"输出目的地≠输入"的核心）
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_cities(text: str, limit: int = None) -> list[str]:
        """
        从文本中提取所有中文城市
        - 先匹配已知城市，按「在文本中出现的位置」排序（不是按城市列表遍历顺序）
        - 再用正则兜底匹配 2-4 字地名
        - [v3-fix] 之前版本按 _ALL_CHINESE_CITIES 遍历顺序返回，会把"从广州到上海"提取成[上海,广州]导致方向颠倒
        """
        found_with_pos = []
        for city in _ALL_CHINESE_CITIES:
            idx = text.find(city)
            if idx != -1:
                # 记录(出现位置, 城市名)，后面按位置排序
                found_with_pos.append((idx, city))
        # ⭐ 按在文本中出现的先后顺序排序（保证"广州→上海"提取成[广州,上海]，而不是[上海,广州]）
        found_with_pos.sort(key=lambda x: x[0])
        found = [city for _, city in found_with_pos]

        # 兜底正则：匹配 2-4 个连续汉字（避免匹配过长词汇）
        if not found or (limit and len(found) < limit):
            for m in re.findall(r"[\u4e00-\u9fa5]{2,4}", text):
                if m not in found and len(m) <= 4:
                    found.append(m)
        return found[:limit] if limit else found

    def _correct_cities(
        self,
        text: str,
        *,
        origin: str = "",
        destination: str = "",
    ) -> str:
        """
        🔧 强制矫正：保证文本中的出发地/目的地 与 用户输入 完全一致。
        [REVISED:2026-08-09 v2-days-fix] 修复版指纹（加载了新代码时本注释作者的改动会生效）
        - 修复：去掉"的交通"作为"目的地语义"替换上下文（否则 广州的交通 → 上海的交通 会被错截成"上海的交"）
        - 修复：对"XX火车站/XX机场/XX汽车站"这类 城市+交通专有名词 单独矫正（基于 origin/destination 前缀匹配）
        """
        if not text:
            return text

        # --------- 修复2（v3版）：基于 → 分隔 分别矫正 出发地/目的地 交通场所 ---------
        # 老版本"全局替换"的问题：广州火车站 → 上海火车站 被粗暴改成 上海火车站 → 上海火车站
        # 解决方案：只针对有 → 箭头的行单独处理：箭头左边按 origin 矫正，右边按 destination 矫正
        if origin or destination:
            _TRANSPORT_SUFFIXES = (
                "火车站", "机场", "汽车站", "客运站",
                "东站", "西站", "南站", "北站",
                "虹桥机场", "浦东机场", "国际机场",
                "长途汽车站",
            )
            _WRONG_CANDIDATES = ["成都", "杭州", "北京", "上海", "三亚", "西安", "南京", "大理", "丽江", "重庆", "武汉", "广州", "深圳"]

            def _fix_line(line: str) -> str:
                # 只处理包含「→」或「->」或「一>」的行（交通换乘行）
                arrow = None
                for sep in (" → ", "→", " -> ", "->", " 一> ", "一>"):
                    if sep in line:
                        arrow = sep
                        break
                if not arrow:
                    return line
                left, _, right = line.partition(arrow)

                # 箭头左边（出发地一侧）：错误城市+交通后缀 → origin+交通后缀
                if origin:
                    for suffix in _TRANSPORT_SUFFIXES:
                        for wrong in _WRONG_CANDIDATES:
                            if wrong == origin:
                                continue
                            left = left.replace(f"{wrong}{suffix}", f"{origin}{suffix}")

                # 箭头右边（目的地一侧）：错误城市+交通后缀 → destination+交通后缀
                if destination:
                    for suffix in _TRANSPORT_SUFFIXES:
                        for wrong in _WRONG_CANDIDATES:
                            if wrong == destination:
                                continue
                            right = right.replace(f"{wrong}{suffix}", f"{destination}{suffix}")

                return left + arrow + right

            new_lines = []
            for ln in text.splitlines(keepends=True):
                new_lines.append(_fix_line(ln))
            text = "".join(new_lines)

            # v3-removed: 已删除"destination全局兜底正则替换"——它会把箭头左边
            #   本来正确的"广州火车站"粗暴改成"上海火车站"。
            # 非箭头行的标题/推荐语等若有错误城市，后续的"标题矫正正则"和"目的地语义
            # 正向断言替换"已经足够覆盖，不需要全局替换交通场所。

        # --------- 矫正标题（优先级最高） ---------
        if destination:
            # 天气标题
            text = re.sub(
                r"【天气查询】[\u4e00-\u9fa5A-Za-z]{1,8}(?=未来|天气|明|今|后)",
                f"【天气查询】{destination}",
                text,
            )
            # 景点标题
            text = re.sub(
                r"【景点推荐】[\u4e00-\u9fa5A-Za-z]{1,8}(?=热门|推荐|景点|好玩的)",
                f"【景点推荐】{destination}",
                text,
            )

        if origin and destination:
            # 城际路线标题（有 origin、有 destination 直接重写）
            text = re.sub(
                r"【路线规划】[\u4e00-\u9fa5A-Za-z]{1,8}\s*→\s*[\u4e00-\u9fa5A-Za-z]{1,8}：",
                f"【路线规划】{origin} → {destination}：",
                text,
            )
        elif destination:
            # 只有 destination，保留 → 前半段
            def _fix_route_without_origin(match):
                head = match.group(1)
                return f"【路线规划】{head} → {destination}："
            text = re.sub(
                r"【路线规划】([\u4e00-\u9fa5A-Za-z]{1,8})\s*→\s*[\u4e00-\u9fa5A-Za-z]{1,8}：",
                _fix_route_without_origin,
                text,
            )

        # 每日行程标题（只含 destination）
        if destination:
            text = re.sub(
                r"【每日行程\s*·\s*[\u4e00-\u9fa5A-Za-z]{1,8}】",
                f"【每日行程 · {destination}】",
                text,
            )

        # --------- 矫正正文中的错误城市 ---------
        _WRONG_CANDIDATES = ["成都", "杭州", "北京", "上海", "三亚", "西安", "南京", "大理", "丽江", "重庆", "武汉", "广州", "深圳"]

        if destination:
            for wrong in _WRONG_CANDIDATES:
                if wrong == destination:
                    continue
                # ⭐ 修复：已从列表中移除「的交通」——否则"广州的交通"被拆成"上海的交"+"通"
                # 目的地语义：X的景点、X旅游、X攻略、X美食、X天气、X行程、X最佳季节、X住宿
                text = re.sub(
                    f"{wrong}(?=(?:的景点|的最佳季节|的天气|的美食|的住宿|的攻略|旅游|游玩|行程|一日游|自由行|深度游|推荐|好玩的))",
                    destination,
                    text,
                )

        if origin and destination:
            for wrong in _WRONG_CANDIDATES:
                if wrong == origin:
                    continue
                # 出发地语义：从X到Y、X出发
                text = re.sub(
                    f"(?<=从){wrong}(?=到|前往|去|飞)",
                    origin,
                    text,
                )
                # 模式：「wrong → destination」中的 wrong 替换为 origin
                text = re.sub(
                    f"{wrong}(?=\\s*→\\s*{re.escape(destination)})",
                    origin,
                    text,
                )

        return text

    def detect_intent(self, user_input: str) -> list[str]:
        """
        检测用户意图，返回需要调用的智能体列表

        支持多意图检测，如 "帮我查成都天气并推荐景点" → ["weather_agent", "attraction_agent"]
        """
        detected = set()
        for keyword, agent in AGENT_ROUTES.items():
            if keyword in user_input:
                detected.add(agent)

        # 如果没有匹配，默认调用景点智能体（最通用）
        if not detected:
            # 根据输入内容判断
            if any(kw in user_input for kw in ["去", "旅游", "旅行", "度假", "玩", "安排", "规划"]):
                detected.add("attraction_agent")
            elif any(kw in user_input for kw in ["多少钱", "预算", "费用"]):
                detected.add("route_agent")
            else:
                detected.add("attraction_agent")

        # 如果用户提到了目的地+出行方式，同时调路线
        if any(kw in user_input for kw in ["从.*到", "去", "出发"]) and \
           any(kw in user_input for kw in ["飞机", "高铁", "怎么去", "路线"]):
            detected.add("route_agent")

        return list(detected)

    async def route_to_agent(
        self,
        agent_name: str,
        user_input: str,
        session_id: str,
        chat_history: list = None,
    ) -> dict:
        """
        将请求路由到指定智能体

        Args:
            agent_name: 智能体名称
            user_input: 用户输入
            session_id: 会话ID
            chat_history: 聊天历史

        Returns:
            智能体响应
        """
        endpoint = self.agent_endpoints.get(agent_name)
        if not endpoint:
            return {"error": f"未知智能体: {agent_name}"}

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{endpoint}/chat",
                    json={
                        "message": user_input,
                        "session_id": session_id,
                        "chat_history": chat_history or [],
                    },
                )
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError:
            fallback_result = await self._local_fallback(agent_name, user_input)
            return {
                "response": fallback_result.get("response", ""),
                "error": f"无法连接到 {agent_name} ({endpoint})",
                "source": "local_fallback",
            }
        except Exception as e:
            return {"response": "", "error": f"调用 {agent_name} 失败: {str(e)}"}

    async def _local_fallback(self, agent_name: str, user_input: str) -> dict:
        """本地回退：当远程智能体不可用时的简化响应"""
        if agent_name == "weather_agent":
            # 先从"X未来N天/查一下X天气/在X玩"模式提取
            m = re.search(
                r"(?:查一下|查询|看看|想知道|请问)?\s*([\u4e00-\u9fa5]{2,4})\s*(?:未来|明|今|后|天气|气温|下雨|温度)",
                user_input,
            )
            loc = m.group(1) if m else ""
            if not loc:
                cities = self._extract_cities(user_input, limit=1)
                loc = cities[0] if cities else ""
            if not loc:
                loc = "目的地"  # 避免默认成北京/成都
            data = _mock_weather_query(loc if loc in ["北京","上海","广州","深圳","成都","杭州","三亚","西安","南京","重庆","武汉","大理"] else "三亚", "", 3)
            # 强制把 summary 里的实际 city 替换成用户要的 loc
            summary = data["summary"].replace(
                next((c for c in ["北京","上海","广州","深圳","成都","杭州","三亚","西安","南京","重庆","武汉","大理"] if c in data["summary"]), ""),
                loc,
            )
            forecasts = []
            for fc in data["forecasts"]:
                forecasts.append(fc)
            return {
                "response": f"【天气查询】{loc}未来3天：{summary}",
                "source": "local_fallback",
            }

        elif agent_name == "attraction_agent":
            # 提取天数
            m_days = re.search(r"(\d{1,2})\s*(?:天|日)", user_input)
            days = int(m_days.group(1)) if m_days else 3
            # 提取目的地
            known = ["北京","上海","广州","深圳","成都","杭州","三亚","西安","南京","重庆","武汉","大理"]
            city = next((c for c in known if c in user_input), "")
            if not city:
                cities = self._extract_cities(user_input, limit=1)
                city = cities[0] if cities else ""
            if not city:
                city = "目的地"
            # 生成景点数量按天数×3，保证后续每日规划能分配（days×2+冗余）
            need_count = max(8, days * 3)
            # 匹配 mock 里有的城市，优先直接生成
            if city in known:
                data = _search_attractions_mock(city, "", need_count)
                attractions = data.get("attractions", [])
                # 不够就用通用安全补全器（禁止使用"特色景点N（XX延伸）"这类AI幻觉式命名）
                if len(attractions) < need_count:
                    extra = _enrich_attractions_generic(city, [], days, need_count - len(attractions) + 2)
                    # 去重：补全器里的名字不和已有重复
                    existing_names = {a.get("name") for a in attractions}
                    for ea in extra:
                        if ea.get("name") not in existing_names:
                            attractions.append(dict(ea))
                            existing_names.add(ea.get("name"))
                        if len(attractions) >= need_count:
                            break
                lines = [f"【景点推荐】{city}热门景点（{days}天行程，推荐{need_count}处）："]
                for i, a in enumerate(attractions[:need_count], 1):
                    name = a.get("name") or a.get("title") or f"{city}景点{i}"
                    rating = a.get("rating") or "4.5"
                    desc = a.get("description") or a.get("desc") or "景点，建议安排游玩"
                    dur = a.get("duration") or (f"{max(1, 4 - (i % 3))}小时")
                    ticket = a.get("ticket") or "以现场为准"
                    # 格式统一用"• 名称（时间 08:30-10:30）- 简介"，方便 route fallback 第二层解析抓项目符号
                    lines.append(f"  • {name}（建议游玩 {dur}，08:30-17:30） — 评分{rating}分，门票：{ticket}，{desc}")
                return {"response": "\n".join(lines), "source": "local_fallback"}
            # 不在 mock 数据中的城市 → 直接用通用安全补全器生成真实可落地的景点名称
            gen_list = _enrich_attractions_generic(city, preferences or [], days, need_count)
            text_lines = [f"【景点推荐】{city}热门景点（{days}天行程，推荐{need_count}处）："]
            for i, a in enumerate(gen_list[:need_count], 1):
                name = a.get("name")
                rating = a.get("rating", 4.3)
                dur = a.get("duration", "2-3小时")
                ticket = a.get("ticket", "免费")
                desc = a.get("desc", "景点，建议游玩")
                text_lines.append(f"  • {name}（建议游玩 {dur}，08:30-17:30） — 评分{rating}分，门票：{ticket}，{desc}")
            return {"response": "\n".join(text_lines), "source": "local_fallback"}

        elif agent_name == "route_agent":

            # 判断是"城际交通"还是"目的地每日行程"
            is_city_itinerary = any(k in user_input for k in ["已推荐景点", "规划{1,3}天行程", "每日行程", "为.*规划"]) \
                or (not any(k in user_input for k in ["从", "至", "到", "往", "到", "→", "飞", "高铁", "机票"])
                    and any(k in user_input for k in ["景点", "行程", "规划", "安排"]))

            known = ["北京","上海","广州","深圳","成都","杭州","三亚","西安","南京","重庆","武汉","大理"]

            if is_city_itinerary:
                # --- 每日行程模式 ---
                city = next((c for c in known if c in user_input), "")
                if not city:
                    cities = self._extract_cities(user_input, limit=1)
                    city = cities[0] if cities else ""
                if not city:
                    city = "目的地"
                days_m = re.search(r"(\d{1,2})\s*(?:天|日)", user_input)
                days = int(days_m.group(1)) if days_m else 3
                # ========== 提取景点名（多层策略，保证至少days*2个） ==========
                # 第1层：带景区/公园/纪念馆后缀的精准景点
                attractions_raw = re.findall(
                    r"[\u4e00-\u9fa5A-Za-z0-9]{2,20}(?:风景区|景区|公园|纪念馆|博物院|古镇|塔|楼|教堂|广场|博物馆|寺|观|街|巷|码头|乐园|度假区|海滩|沙滩|步行街|大道|书院|故居|祠)",
                    user_input,
                )
                # 第2层："• 景点名"项目符号后的中文（attraction fallback的推荐列表就是这格式）
                if len(attractions_raw) < days * 2:
                    bullet_items = re.findall(r"[•\-\*\d\.\)、]\s*([\u4e00-\u9fa5A-Za-z0-9]{2,20}?)(?:\s*\(|（|\s|$)", user_input)
                    for bi in bullet_items:
                        # 过滤掉非景点词（时间、价格、常见虚词）
                        skip_words = {"上午", "下午", "晚上", "小时", "分钟", "建议", "预算", "价格", "门票", "免费", "车费", "美食", "标准", "舒适", "经典"}
                        if bi not in attractions_raw and len(bi) >= 2 and bi not in skip_words and not re.match(r"^(0\d|1\d|2[0-3]):[0-5]\d$", bi):
                            attractions_raw.append(bi)
                # 第3层：兜底——用通用安全补全器生成真实可落地的景点名称（禁止使用"核心景点N"占位符）
                if len(attractions_raw) < days * 2:
                    pad_need = days * 3
                    gen_pad = _enrich_attractions_generic(city, [], days, pad_need + 2)
                    pad = []
                    for gp in gen_pad:
                        nm = gp.get("name")
                        if nm and nm not in attractions_raw and nm not in pad:
                            pad.append(nm)
                        if len(pad) + len(attractions_raw) >= pad_need:
                            break
                    attractions_raw.extend(pad)
                    # 极端情况还不够，用轮询但不用数字占位
                    while len(attractions_raw) < days * 2:
                        for gp in gen_pad:
                            if len(attractions_raw) >= days * 2:
                                break
                            attractions_raw.append(gp.get("name"))
                data = _generate_daily_routes(city, attractions_raw, days)
                # 矫正 data 里的 destination 字段（避免成都→杭州这种）
                data["destination"] = city
                lines = [f"【每日行程 · {city}】共 {data['total_days']} 天"]
                for dp in data["daily_plans"]:
                    lines.append(f"\n🗓️ 第{dp['day']}天：")
                    for j, r in enumerate(dp["routes"]):
                        # 矫正 r 里的错误 city
                        for c in known:
                            if c != city:
                                if r["from"] == c:
                                    r["from"] = f"{city}酒店" if j == 0 else f"{city}景点间"
                        lines.append(f"  • {r['from']} → {r['to']}，{r['transport']}，{r['duration']}（{r['distance_km']} km）")
                    lines.append(f"  预计花费：{dp['transport_cost']}，总耗时：{dp['estimated_total_time']}")
                lines.append(f"\n💡 {data['general_tips']}")
                return {"response": "\n".join(lines), "source": "local_fallback"}

            else:
                # --- 城际交通模式：提取 origin & destination ---
                origin = destination = ""

                # ⭐ v3 修复：先用 _extract_cities（已知城市优先）提取，避免正则贪婪匹配"上海的交"
                # 正则 {2,4} 贪婪时，"上海的交通方案" 会被截成 destination="上海的交"
                known_cities_first = self._extract_cities(user_input, limit=2)
                if len(known_cities_first) >= 2:
                    origin, destination = known_cities_first[0], known_cities_first[1]
                elif len(known_cities_first) == 1:
                    destination = known_cities_first[0]

                # 只有 _extract_cities 没提取到足够信息时，才用正则兜底
                if not origin or not destination:
                    # 模式1：从A到B / 由A去B / A→B
                    m_dual = re.search(
                        r"(?:从|由|自)?\s*([\u4e00-\u9fa5]{2,4})\s*(?:到|去|前往|至|飞|达|→|一>|-)\s*([\u4e00-\u9fa5]{2,4}?)(?=[，。、；：,\.\s?:!;]|$|的)",
                        user_input,
                    )
                    if m_dual:
                        if not origin:
                            origin = m_dual.group(1)
                        if not destination:
                            destination = m_dual.group(2)

                # 正则/提取都没结果时，再用 _extract_cities 最后兜底
                if not origin and not destination:
                    cities = self._extract_cities(user_input, limit=2)
                    if len(cities) >= 2:
                        origin, destination = cities[0], cities[1]
                    elif len(cities) == 1:
                        destination = cities[0]
                        origin_m = re.search(r"(?:从|由|自)\s*([\u4e00-\u9fa5]{2,4})", user_input)
                        origin = origin_m.group(1) if origin_m else "出发地"

                # 至少保证 destination 存在
                if not destination:
                    destination = "目的地"
                if not origin:
                    origin = "出发地"

                # 用 mock 生成（需要已知城市映射）
                def _to_known(name):
                    if name in known:
                        return name
                    # 距离最近的已知城市
                    return {"苏州":"上海","无锡":"上海","嘉兴":"杭州","绍兴":"杭州",
                            "宁波":"杭州","佛山":"广州","东莞":"深圳","佛山":"广州",
                            "廊坊":"北京","保定":"北京","咸阳":"西安","德阳":"成都",
                            "眉山":"成都","绵阳":"成都","渭南":"西安","青岛":"北京",
                            "长沙":"广州","昆明":"大理","贵阳":"成都"}.get(name, "上海" if not destination else (origin if origin != "出发地" else "上海"))

                data = _generate_transport_plans(_to_known(origin), _to_known(destination), "", "最快")
                # 矫正 output 中的城市名（把已知城市 → 用户输入的）
                routes = []
                for r in data["routes"]:
                    r_new = dict(r)
                    r_new["departure"] = r["departure"].replace(_to_known(origin), origin).replace(_to_known(destination), destination)
                    r_new["arrival"] = r["arrival"].replace(_to_known(origin), origin).replace(_to_known(destination), destination)
                    routes.append(r_new)
                rec = (data["recommendation"]
                       .replace(_to_known(origin), origin)
                       .replace(_to_known(destination), destination))

                lines = [f"【路线规划】{origin} → {destination}："]
                for r in routes:
                    lines.append(f"  🚄 {r['type']}：{r['duration']}，约{r['price']} | 优点：{r['advantage']} | 注意：{r['disadvantage']}")
                    lines.append(f"      {r['departure']}  →  {r['arrival']}")
                lines.append(f"\n💎 推荐：{rec}")
                lines.append(f"📌 小贴士：{data['tips']}")
                return {"response": "\n".join(lines), "source": "local_fallback"}

        return {"response": "服务暂不可用", "source": "local_fallback"}

    async def handle_user_request(self, user_input: str, session_id: str = None) -> dict:
        """
        处理用户请求的主入口

        流程：
        1. 检测意图 → 路由到合适的智能体
        2. 如果需要多智能体协同，编排执行顺序
        3. 汇总结果，生成最终回答
        4. 保存会话，检查是否需要压缩
        """
        if not session_id:
            session_id = self.transport.create_task("init").id[:8]

        # 记录用户消息
        self.memory.add_message(session_id, "user", user_input)

        # 检查是否需要压缩上下文
        if self.compressor.should_compress(session_id):
            self.compressor.auto_compress(session_id)

        # 获取聊天历史
        chat_history = self.memory.get_chat_history(session_id, limit=10)

        # 检测意图
        target_agents = self.detect_intent(user_input)
        print(f"[Orchestrator] 检测到意图: {target_agents}, 用户输入: {user_input[:50]}")

        # 单智能体：直接调用
        if len(target_agents) == 1:
            agent_name = target_agents[0]
            result = await self.route_to_agent(agent_name, user_input, session_id, chat_history)
            response_text = result.get("response", "")

        # 多智能体：协同执行
        else:
            response_text = await self._orchestrate_multi_agent(
                target_agents, user_input, session_id, chat_history
            )

        # ----------------------------------------------------------
        # 🔧 最终矫正：从用户输入中提取 origin / destination 再强制替换
        # ----------------------------------------------------------
        input_origin = input_destination = ""

        # ⭐ v3：优先用已知城市提取，避免正则贪婪截到"上海的交"
        found_final_cities = self._extract_cities(user_input, limit=2)
        if len(found_final_cities) >= 2:
            input_origin, input_destination = found_final_cities[0], found_final_cities[1]
        elif len(found_final_cities) == 1:
            input_destination = found_final_cities[0]

        # 已知城市提取不到足够信息时，用正则兜底（非贪婪+正向断言）
        if not input_origin or not input_destination:
            m_dual = re.search(
                r"(?:从|由|自)\s*([\u4e00-\u9fa5]{2,4})\s*(?:到|去|前往|至|飞|达|→)\s*([\u4e00-\u9fa5]{2,4}?)(?=[，。、；：,\.\s?:!;]|$|的)",
                user_input,
            )
            if m_dual:
                if not input_origin:
                    input_origin = m_dual.group(1)
                if not input_destination:
                    input_destination = m_dual.group(2)

        # 最终兜底：单 destination 时，从最后一个 city 取
        if not input_destination and found_final_cities:
            input_destination = found_final_cities[-1]
        if not input_origin and len(found_final_cities) >= 2:
            input_origin = found_final_cities[0]
        response_text = self._correct_cities(
            response_text, origin=input_origin, destination=input_destination
        )

        # 记录助手消息
        self.memory.add_message(session_id, "assistant", response_text)

        return {
            "session_id": session_id,
            "response": response_text,
            "agents_called": target_agents,
            "timestamp": time.time(),
        }

    async def _orchestrate_multi_agent(
        self,
        agents: list[str],
        user_input: str,
        session_id: str,
        chat_history: list,
    ) -> str:
        """
        编排多智能体协同

        策略：
        - 如果包含天气+景点：先查天气，再结合天气推景点
        - 如果包含景点+路线：先推景点，再为景点规划路线
        - 并行调用独立智能体，串行调用有依赖关系的
        """
        results = {}

        # 天气智能体通常先调用（为其他智能体提供上下文）
        weather_first = "weather_agent" in agents

        if weather_first:
            # 天气智能体先执行
            weather_result = await self.route_to_agent(
                "weather_agent", user_input, session_id, chat_history
            )
            results["weather"] = weather_result.get("response", "")

            # 将天气信息传递给后续智能体
            enriched_input = f"{user_input}\n\n[天气信息参考]：{results['weather']}"
            remaining = [a for a in agents if a != "weather_agent"]
        else:
            enriched_input = user_input
            remaining = agents

        # 并行调用剩余智能体
        tasks = [
            self.route_to_agent(a, enriched_input, session_id, chat_history)
            for a in remaining
        ]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, agent_name in enumerate(remaining):
            if isinstance(task_results[i], Exception):
                results[agent_name] = f"调用失败: {task_results[i]}"
            else:
                results[agent_name] = task_results[i].get("response", "")

        # 汇总结果
        summary_parts = []
        for agent_name, response in results.items():
            if response and "error" not in str(response)[:100]:
                summary_parts.append(response)

        if not summary_parts:
            return "抱歉，智能体暂时无法回答您的问题。"

        # 如果只有一个结果，直接返回
        if len(summary_parts) == 1:
            return summary_parts[0]

        # 多个结果时，添加分隔
        merged = "\n\n---\n\n".join(summary_parts)
        return merged

    # ------------------------------------------------------------------
    # 暖心贴士：基于天气生成穿衣建议 + 出行建议
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_warm_tips(
        weather_data: dict,
        destination: str,
        days: int,
        preferences: list,
    ) -> str:
        """
        生成 3 级分级卡片式暖心贴士：
          ⚠️ 务必注意  → 安全/舒适，一定要做
          👍 建议准备  → 提升体验，尽量安排
          👀 仅供参考  → 了解即可，不影响出行
        规则：短句 + emoji + 关键词，卡片分行，一眼扫完；结尾简短祝福。
        """
        if not destination:
            destination = "当地"
        forecasts = []
        if isinstance(weather_data, dict):
            forecasts = list(weather_data.get("forecasts", []) or [])

        # ---------- 解析天气数据 ----------
        all_highs, all_lows = [], []
        rainy_days = hot_days = cold_days = humid_days = bad_aqi_days = 0
        for fc in forecasts:
            temp = str(fc.get("temperature", ""))
            try:
                m = re.match(r"\s*(\d+)\s*~\s*(\d+)", temp)
                if m:
                    lo, hi = int(m.group(1)), int(m.group(2))
                    all_lows.append(lo); all_highs.append(hi)
                    if hi >= 32: hot_days += 1
                    if lo <= 12: cold_days += 1
            except Exception:
                pass
            cond = str(fc.get("condition", ""))
            if any(k in cond for k in ("雨", "雷", "雪")): rainy_days += 1
            try:
                hum = int(re.sub(r"\D", "", str(fc.get("humidity", "0"))) or 0)
                if hum >= 75: humid_days += 1
            except Exception:
                pass
            aqi = str(fc.get("aqi", ""))
            if any(k in aqi for k in ("污染", "差")): bad_aqi_days += 1
        avg_high = (sum(all_highs) // len(all_highs)) if all_highs else 26
        avg_low = (sum(all_lows) // len(all_lows)) if all_lows else 20
        prefs_str = " ".join(preferences or [])
        has_food = any(k in prefs_str for k in ("美食", "吃", "小吃"))
        has_nature = any(k in prefs_str for k in ("自然", "风景", "徒步", "登山", "名山"))
        has_culture = any(k in prefs_str for k in ("历史", "文化", "古迹", "博物馆", "拍照", "打卡"))
        has_family = any(k in prefs_str for k in ("亲子", "孩子", "老人", "家庭"))

        # ---------- 三个桶：务必注意 / 建议准备 / 仅供参考 ----------
        must_do, should_do, nice_to_know = [], [], []

        # ==================== 1. 穿衣建议（分档，按天气严重度分级）====================
        if avg_high >= 32:
            # 盛夏：防晒是安全问题（中暑/晒伤）→ ⚠️
            must_do.extend([
                "🧴 防晒霜 SPF50+ 出门 20 分钟前涂好，每 2 小时补涂一次",
                "🕶️ 戴墨镜 + 宽檐帽，脖子/手臂别露晒",
            ])
            should_do.extend([
                "👕 短袖 T 恤 + 短裤/薄裙，选透气棉/速干面料",
                "🧊 随身带小风扇 / 冰凉贴 + 1 瓶温水",
            ])
        elif avg_high >= 25:
            # 温暖：防晒+穿搭 → 👍 建议准备
            should_do.extend([
                "👕 短袖 + 薄长裤/长裙，透气棉质最佳",
                "🧥 早晚稍凉，外搭薄外套 / 防晒衣（空调房也能用）",
                "🧴 防晒霜 SPF30+ 出门必涂，别偷懒",
            ])
        elif avg_high >= 18:
            should_do.extend([
                "👕 长袖 T + 衬衫/薄卫衣 + 牛仔裤（春秋叠穿，热脱冷穿）",
                "🧥 外搭风衣 / 棒球衫，应对早晚温差",
            ])
        elif avg_high >= 10:
            must_do.append("🧥 毛衣/卫衣 + 夹棉外套/薄呢大衣，早晚才不会冷")
            should_do.append("🧣 带一条围巾，风大直接围上")
        else:
            must_do.extend([
                "🧥 厚羽绒服/厚呢大衣 + 保暖内衣 + 毛衣，三件套穿齐",
                "🧤 帽子 + 围巾 + 手套 + 加绒防滑鞋，零下不挨冻",
            ])
        # 盛夏防晒衣 / 温暖薄外套 → 已并入上面分档，保留用户要求的原信息点完整

        # ==================== 2. 空气差 / 敏感人群防护 ====================
        if bad_aqi_days > 0:
            # 空气差关系到呼吸道安全 → ⚠️ 务必注意
            must_do.append(f"😷 行程有 {bad_aqi_days} 天空气一般，👶老人/👧小孩/🤧敏感人群全程戴 KN95 口罩")
        else:
            # 没有空气差就不说（用户样例里有这个点，如果有就必须保留）
            pass

        # ==================== 3. 炎热 / 下雨 （安全）====================
        if hot_days > 0:
            must_do.extend([
                f"🌡️ 有 {hot_days} 天 ≥32℃，中午 11:00-15:00 别在户外暴晒",
                "💧 随身带 1 瓶温水，小口常喝别等渴",
            ])
            should_do.append("💊 备藿香正气水 / 人丹 / 降温贴，防中暑")
        if cold_days > 0:
            must_do.append(f"🧊 有 {cold_days} 天 ≤12℃，早出晚归多穿 1 件，保温杯装热水")
        if rainy_days > 0:
            must_do.append(f"🌧️ 有 {rainy_days} 天可能下雨，景区步道石板湿滑，脚下慢一点")
            should_do.append("🧥 带 1 件便携薄雨衣（山区风雨比打伞稳太多）")
        if humid_days > 0:
            should_do.extend([
                f"💧 有 {humid_days} 天湿度 ≥75%，多备 2 件换洗衣物（闷汗难干）",
                "🧻 带吸汗湿巾 / 止汗喷雾，体感更舒服",
            ])

        # ==================== 4. 美食打卡偏好（用户样例强制保留）====================
        if has_food:
            must_do.extend([
                "⏰ 热门餐厅提前 30 分钟到，或 13:00 后错峰用餐（省 1 小时排队）",
            ])
            should_do.extend([
                "💊 备健胃消食片 / 益生菌 / 保济丸，吃杂吃撑不难受",
                "💧 冰饮别配太辣太油，分开吃肠胃更稳",
            ])

        # ==================== 5. 自然风光/徒步偏好（用户样例强制保留）====================
        if has_nature:
            # 防滑运动鞋关乎安全扭伤 → ⚠️
            must_do.extend([
                "👟 穿**防滑运动鞋**（拒绝凉鞋/拖鞋/小白鞋，景区石板/台阶太滑）",
                "🧥 山区天气多变，带 1 件**便携雨衣**（比雨伞好用，腾手爬山）",
            ])
            should_do.extend([
                "🔋 随身充电宝（满电）+ 数据线，山里信号差耗电快",
                "🩹 创可贴 + 小瓶水 + 1 条能量棒，有备无患",
            ])
            nice_to_know.extend([
                "🥾 徒步景区尽量走铺好的游步道，别私自爬野路野山",
                "📶 进山前把家人定位和行程发一条给朋友，信号断了有人知道你在哪",
            ])

        # ==================== 6. 人文古迹/拍照打卡偏好 ====================
        if has_culture:
            must_do.append("📲 博物馆 / 纪念馆 / 热门打卡点**提前 1-7 天在线预约**，到了进不去白跑")
            should_do.extend([
                "🔋 相机备用电池 + 手机充电宝（拍照 2 小时电就空）",
                "👗 穿浅/素色衣服 + 小白鞋，拍照更上镜，脏了也不显",
            ])
            nice_to_know.append("🅿️ 节假日热门景点停车场要排队，建议地铁/公交去，或提前 1 小时到")

        # ==================== 7. 亲子/家庭出行偏好 ====================
        if has_family:
            must_do.extend([
                "💊 小朋友常用药：退烧药 / 腹泻药 / 创可贴 / 驱蚊水（按量带够）",
                "👵 老人常用药 + 降压/降糖药，多备 2 天量（别放托运）",
            ])
            should_do.append("⏰ 每天景点别安排太满，小朋友/老人累了就歇，别赶路")
            nice_to_know.append("🚗 带娃优先选地铁 / 打车 / 网约车，少坐公交折腾太久")

        # ==================== 8. 通用贴士（兜底也好看）====================
        if not must_do and not should_do and not nice_to_know:
            must_do.append("🆔 身份证/证件 + 充电宝 带好，出发前再检查一遍")

        # ==================== 组装输出（严格 3 级，每行 emoji+短句）====================
        lines = []

        def _add_block(title: str, emoji_title: str, items: list):
            if not items:
                return
            lines.append(f"{emoji_title} {title}")
            for it in items:
                lines.append(f"  • {it}")
            lines.append("")  # 空一行分级

        _add_block("务必注意（关乎舒适/安全 · 一定要做）", "⚠️", must_do)
        _add_block("建议准备（带上明显更舒服 · 尽量安排）", "👍", should_do)
        _add_block("仅供参考（了解就行 · 不准备也ok）", "👀", nice_to_know)

        # 结尾简短旅行祝福
        lines.append(f"💛 {destination} 之旅，玩得尽兴！")

        return "\n".join(lines).rstrip() + "\n"

    async def plan_full_trip(self, request: dict) -> dict:
        """
        一键规划完整旅游行程 [加固版]
        [REVISED:2026-08-10 FIXED]
        - 全流程 try/except 加固，任何环节出错都自动 fallback，保证能返回结果
        - 对未知城市提供通用模板数据（不再返回空列表导致失败）
        - 每个步骤有独立的兜底：即使 Step 3 挂了，Step 1/2/4 也能正常输出
        [REVISED:2026-08-11 CitySelector]
        - 第 0 步：字段归一化，支持「新旧两套字段」同时传入
          * 新字段（H5 CitySelector 组件）：departLocation / destLocation 对象 + departDetailAddress / destDetailAddress 文本
          * 旧字段（兼容老前端）：origin / destination 字符串
          * 优先级：新字段对象里的 cityName > 旧字段字符串
        - departDetailAddress / destDetailAddress 选填文本原样透传（不解析，留给后续酒店加权推荐用）
        """
        # ========== Step 0: 字段归一化（新旧格式兼容 + detailAddress 透传）==========
        # 1) 从新格式 departLocation 对象里取出发城市名（level=city 时必有 cityName）
        depart_loc = request.get("departLocation") or {}
        if isinstance(depart_loc, dict):
            depart_city_from_obj = (depart_loc.get("cityName") or "").strip()
        else:
            depart_city_from_obj = ""
        # 2) 从新格式 destLocation 对象里取目的地城市名
        dest_loc = request.get("destLocation") or {}
        if isinstance(dest_loc, dict):
            dest_city_from_obj = (dest_loc.get("cityName") or "").strip()
        else:
            dest_city_from_obj = ""
        # 3) 旧字段 origin / destination（兼容老前端）
        origin_old = (request.get("origin", "") or "").strip()
        destination_old = (request.get("destination", "") or "").strip()
        # 4) 优先级：新字段对象 cityName > 旧字段字符串（新 H5 必传对象，旧前端传字符串）
        origin = depart_city_from_obj or origin_old
        destination = dest_city_from_obj or destination_old
        # 5) 详细位置（选填纯文本，前端不解析，原样透传；后续酒店加权用）
        depart_detail_addr = (request.get("departDetailAddress") or "").strip()
        dest_detail_addr = (request.get("destDetailAddress") or "").strip()

        dates = (request.get("dates", "") or "").strip()
        # ===== 出行天数：严格按 5 条规则计算（日期范围优先 > days数字 > 模糊估算3天；区间强制 1~30）=====
        days_raw = request.get("days", None)
        days = _calculate_travel_days(dates_str=dates, days_input_raw=days_raw, destination_hint=destination)
        preferences = request.get("preferences", []) or []
        # ===== 预算档位归一化（关键！让 3 档结果真正不同）—— 档位名固定：经济型 / 舒适型 / 高端型 =====
        # 兼容老请求传 "标准" / "标准型" → 统一归一为 "舒适型"
        raw_budget = str(request.get("budget", "舒适型") or "舒适型").strip()
        budget_level = _normalize_budget_level(raw_budget)
        budget = budget_level  # 保留原名兼容后续代码

        # ===== 出行人数：默认 2 人同行；供预算模块计算（总预算 = 物价系数 × 天数 × 人数）=====
        try:
            travelers_raw = request.get("travelers", 2)
            travelers = max(1, min(int(travelers_raw or 2), 20))
        except Exception:
            travelers = 2

        # ===== 城市物价上下文：用于动态调整酒店/餐饮/预算/交通的价格区间 =====
        tier_tag, price_factor, is_sm_city = _get_city_price_context(destination)

        session_id = f"trip_{int(time.time())}"
        _rev = "REV-20260811-DYNAMIC-BUDGET"

        # 目的地兜底：真的空的话，给一个默认值，避免后续 N 个函数炸掉
        if not destination:
            destination = "上海"

        # ---------- 通用 fallback 模板：未知城市景点数据（每个景点都是 4 小时游玩）----------
        def _fallback_attractions(dest: str, count: int) -> list:
            templates = [
                ("古城文化街区", "历史文化", "4.6", "免费", "体验当地民俗文化与特色建筑"),
                ("中心公园", "自然风光", "4.5", "免费", "城市绿肺，适合漫步休闲"),
                ("博物馆", "博物馆", "4.7", "免费", "了解城市历史文化与发展变迁"),
                ("美食步行街", "美食", "4.4", "免费", "当地特色小吃聚集地，吃货必去"),
                ("滨江/湖景观光带", "自然风光", "4.6", "免费", "欣赏城市水岸风光，适合拍照"),
                ("地标广场", "历史文化", "4.5", "免费", "城市地标建筑，打卡必去"),
                ("主题乐园", "主题乐园", "4.7", "120元", "适合亲子家庭，刺激有趣"),
                ("名山风景区", "自然风光", "4.8", "80元", "登高望远，呼吸新鲜空气"),
                ("古镇古村", "历史文化", "4.6", "50元", "感受历史沉淀的慢生活"),
                ("艺术创意园区", "历史文化", "4.5", "免费", "年轻人打卡拍照胜地"),
            ]
            result = []
            for i in range(count):
                t = templates[i % len(templates)]
                result.append({
                    "name": f"{dest}{t[0]}",
                    "category": t[1],
                    "rating": t[2],
                    "ticket": t[3],
                    "duration": f"{max(1, 4 - (i % 3))}小时",
                    "desc": t[4],
                    "description": t[4],
                })
            return result

        # ========== Step 1: 天气查询（try/except 加固）==========
        weather_resp = ""
        warm_tips_resp = ""
        try:
            weather_data = _mock_weather_query(destination, dates, days)
            weather_resp = f"【天气查询】{destination}未来{days}天：{weather_data['summary']}"
            forecasts = weather_data.get("forecasts", [])
            for fc in forecasts:
                weather_resp += f"\n  📅 {fc['date']}：{fc['condition']}，{fc['temperature']}，空气质量{fc['aqi']}，湿度{fc['humidity']}"
            # 在天气查询之后，立刻生成「穿衣建议 + 出行提示」暖心贴士
            try:
                warm_tips_resp = self._generate_warm_tips(weather_data, destination, days, preferences)
            except Exception as ewt:
                print(f"[Orchestrator] 暖心贴士生成失败（不影响主流程）: {ewt}")
                warm_tips_resp = ""
        except Exception as e1:
            print(f"[Orchestrator] 天气查询异常，使用兜底: {e1}")
            # 兜底天气模板：日期也要基于用户传入的 dates（今天 2/10 出行，第二天不能跳到 8/11）
            try:
                start_d = datetime.date.fromisoformat(dates) if dates else datetime.date.today()
            except Exception:
                start_d = datetime.date.today()
            weather_resp = (f"【天气查询】{destination}未来{days}天：以晴到多云为主，"
                           f"最高温28℃，最低温18℃，整体气候宜人。")
            fallback_fc = []
            for i in range(days):
                d = start_d + datetime.timedelta(days=i)
                cond = ["晴", "多云", "阴", "晴转多云"][i % 4]
                weather_resp += f"\n  📅 {d.isoformat()}：{cond}，18~28℃，空气质量优，湿度60%"
                fallback_fc.append({"date": d.isoformat(), "condition": cond, "temperature": "18~28℃", "aqi": "优", "humidity": "60%"})
            # fallback 天气也要生成贴士
            try:
                warm_tips_resp = self._generate_warm_tips({"forecasts": fallback_fc}, destination, days, preferences)
            except Exception as ewt:
                warm_tips_resp = ""

        # ========== Step 2: 景点推荐（真实数据优先+通用安全兜底+出行偏好严格生效）==========
        attraction_resp = ""
        attractions = []
        has_real_attractions = False  # 是否命中ATTRACTION_DB真实数据（用于资源匮乏备注）
        resource_note = ""  # 资源匮乏城市的方案开头备注（后续渲染在最顶部）
        try:
            need_count = max(8, days * 3)

            # 判断 destination 是否在数据库中，不在的话走【通用安全补全器】（老街/博物馆/步行街/近郊等，不编造虚构景点名）
            if destination not in ATTRACTION_DB:
                print(f"[Orchestrator] {destination} 无内置景点数据，使用通用安全补全器（{need_count}个真实可落地板块）")
                attractions = _enrich_attractions_generic(destination, preferences, days, need_count)
                has_real_attractions = False
            else:
                attr_data = _search_attractions_mock(destination, "", need_count)
                attractions = list(attr_data.get("attractions", []))
                has_real_attractions = len(attractions) >= 1
                # 景点不够时用通用安全补全器追加（禁止原延伸景点"特色景点N（XX延伸）"的AI幻觉式编造）
                if len(attractions) < need_count:
                    extra = _enrich_attractions_generic(destination, preferences, days, need_count - len(attractions) + 2)
                    attractions.extend(extra)
        except Exception as e2:
            print(f"[Orchestrator] 景点推荐异常，使用通用安全兜底: {e2}")
            need_count = max(8, days * 3)
            attractions = _enrich_attractions_generic(destination, preferences, days, need_count)
            has_real_attractions = False

        # ======== 景点排序：【出行偏好标签匹配度 → 评分降序 → 是否通用兜底项】========
        # Step A：按出行偏好打权重分（匹配标签越多排越前；无偏好则跳过这一步）
        if preferences:
            attractions = _apply_preference_weight(attractions, preferences)
        else:
            def _attr_rating(a):
                try:
                    return float(str(a.get("rating", 0) or 0).strip().split()[0])
                except Exception:
                    return 0.0
            # 无偏好：按评分从高到低
            attractions.sort(key=lambda a: -_attr_rating(a))

        # ======== 生成【资源匮乏备注】（ATTRACTION_DB无数据或景点<days*2时，方案最顶部提示）========
        real_count = len([a for a in ATTRACTION_DB.get(destination, [])]) if has_real_attractions else 0
        resource_note = _get_resource_poor_note(destination, has_real_attractions, real_count, days)

        # 格式化景点输出（这部分不会异常）
        try:
            def _attr_rating2(a):
                try:
                    return float(str(a.get("rating", 0) or 0).strip().split()[0])
                except Exception:
                    return 0.0
            need_count = max(8, days * 3)
            final_attrs = attractions[:need_count] if attractions else _enrich_attractions_generic(destination, preferences, days, need_count)
            top_n = len(final_attrs)
            pref_tag = ""
            if preferences:
                pref_tag = f" · 偏好匹配：{'/'.join(preferences[:4])}"
            attr_lines = [f"【景点推荐】{destination}热门景点（评分从高到低，共{top_n}处，{days}天行程推荐游览{pref_tag}）："]
            for i, a in enumerate(final_attrs, 1):
                name = str(a.get("name") or a.get("title") or f"{destination}景点{i}")
                rating_val = _attr_rating2(a)
                rating_str = f"{rating_val:.1f}" if rating_val > 0 else "4.5"
                if rating_val >= 4.9:
                    star = "⭐⭐⭐⭐⭐"
                elif rating_val >= 4.7:
                    star = "⭐⭐⭐⭐⚡"
                elif rating_val >= 4.5:
                    star = "⭐⭐⭐⭐"
                elif rating_val >= 4.3:
                    star = "⭐⭐⭐⚡"
                else:
                    star = "⭐⭐⭐"
                ticket = str(a.get("ticket") or "免费")
                duration = str(a.get("duration") or "2 小时")
                category = str(a.get("category") or "景点")
                desc = str(a.get("description") or a.get("desc") or "特色景点，建议安排游玩")
                if "免费" in ticket or ticket in ("", "0", "0元"):
                    ticket_show = f"🆓 免费"
                else:
                    digits = __import__("re").findall(r"\d+", ticket)
                    price = digits[0] if digits else ticket
                    ticket_show = f"💰 ¥{price}"
                attr_lines.append(
                    f"  {i}. {star} {rating_str} · {name}\n"
                    f"       🏷️ {category} ｜ {ticket_show} ｜ ⏳ 建议游玩 {duration}\n"
                    f"       📝 {desc}"
                )
            attraction_resp = "\n".join(attr_lines)
        except Exception as e2b:
            print(f"[Orchestrator] 景点格式化异常: {e2b}")
            attraction_resp = f"【景点推荐】{destination}建议按「老城漫步→博物馆→步行街→近郊休闲」的顺序安排{days}天深度游玩。"

        # ========== Step 3: 城际交通方案（try/except 加固）==========
        transport_resp = ""
        try:
            if origin and origin.strip():
                transport_data = _generate_transport_plans(origin, destination, dates, "最快")
                trans_lines = [f"【路线规划】{origin} → {destination}："]
                for r in transport_data.get("routes", []):
                    trans_lines.append(f"  🚄 {r['type']}：{r['duration']}，约{r['price']} | 优点：{r['advantage']} | 注意：{r['disadvantage']}")
                    trans_lines.append(f"      {r['departure']}  →  {r['arrival']}")
                if "recommendation" in transport_data:
                    trans_lines.append(f"\n💎 推荐：{transport_data['recommendation']}")
                if "tips" in transport_data:
                    trans_lines.append(f"📌 小贴士：{transport_data['tips']}")
                transport_resp = "\n".join(trans_lines)
            else:
                transport_resp = f"【路线规划】未指定出发城市，跳过城际交通方案。建议抵达{destination}后使用公共交通或打车出行。"
        except Exception as e3:
            print(f"[Orchestrator] 城际交通异常，使用兜底: {e3}")
            # 通用交通 fallback
            transport_resp = (
                f"【路线规划】{origin} → {destination}：\n"
                f"  🚄 高铁：约3-5小时，约200-500元 | 优点：准点率高，舒适便捷 | 注意：建议提前购票\n"
                f"      {origin}火车站  →  {destination}火车站\n"
                f"\n💎 推荐：优先选择高铁出行，性价比高且舒适。\n"
                f"📌 小贴士：节假日高峰建议提前 7-15 天预订票务。"
            ) if origin else (
                f"【路线规划】未指定出发城市。建议抵达{destination}后，使用地铁/公交/打车在市内出行。"
            )

        # ========== Step 3.5: 🏨 住宿推荐（全域通用：动态物价系数 × 档位类型，小众城市高端档降级）==========
        hotels_resp = ""
        try:
            first_day_spot = ""
            if attractions:
                first_day_spot = str(attractions[0].get("name") or "")
            hotels_resp = _generate_hotels_universal(
                destination, budget_level, first_day_spot_hint=first_day_spot,
                price_factor=price_factor, tier_tag=tier_tag, is_sm_city=is_sm_city
            )
        except Exception as eh:
            print(f"[Orchestrator] 酒店推荐模块异常（不影响主流程）: {eh}")
            hotels_resp = f"【酒店推荐 · {destination}】{budget_level}住宿：建议订「火车站/市中心/第1天核心景区」三大片区的对应档次酒店，订房选可免费取消的，夏天带驱蚊液和薄外套。"


        # ========== Step 3.6: 🍜 吃喝玩乐推荐（全域通用：所有城市动态物价系数 × 偏好双维度生效）==========
        food_resp = ""
        try:
            food_resp = _generate_food_universal(
                destination, budget_level, preferences,
                price_factor=price_factor, tier_tag=tier_tag, is_sm_city=is_sm_city
            )
        except Exception as ef:
            print(f"[Orchestrator] 吃喝玩乐推荐模块异常（不影响主流程）: {ef}")
            food_resp = f"【美食玩乐 · {destination}】吃喝攻略：优先吃城市特色正餐+夜市街边小吃，夏天喝冰粉/绿豆沙解暑，景区门口别吃饭、景区内别买特产，找本地人多的店基本不会踩雷。"


        # ========== Step 3.7: 🚗 市内交通体系（全域通用：所有城市按人数+预算+偏好推荐，统一黑车避雷+公交短板）==========
        internal_transport_resp = ""
        try:
            internal_transport_resp = _generate_internal_transport_universal(
                destination, days, budget_level, preferences, travelers=travelers,
                price_factor=price_factor
            )
        except Exception as eit:
            print(f"[Orchestrator] 市内交通模块异常（不影响主流程）: {eit}")
            internal_transport_resp = f"【市内交通 · {destination}】出行建议：市区用平台网约车/公交，远景区拼车或包车；车站/景区门口黑车私车别上，出租车推荐饭馆要警惕。"


        # ========== Step 3.8: 💰 出行总预算（全域通用：物价系数×天数×人数，当前档位高亮）==========
        budget_resp = ""
        try:
            budget_resp = _generate_budget_universal(
                destination, days, budget_level,
                intercity_train=intercity_train_cost if 'intercity_train_cost' in locals() else 300,
                price_factor=price_factor, tier_tag=tier_tag, travelers=travelers,
                is_sm_city=is_sm_city
            )
        except Exception as ebg:
            print(f"[Orchestrator] 总预算模块异常（不影响主流程）: {ebg}")
            # fallback 按物价系数粗略估：二线基准 × 系数
            base = {"经济型": 1200, "舒适型": 1700, "高端型": 2900}.get(budget_level, 1700)
            lo = int(base * price_factor)
            hi = int((base + 500) * price_factor)
            budget_resp = f"【{days}日出行总预算 · {destination}】当前档位：{budget_level}（约{lo}-{hi}元/人，{travelers}人同行，含住宿/门票/吃喝/交通，不含购物）"

        # ========== Step 4: 📅 每日行程安排（全域通用：所有城市上午→中午→下午→晚上顺路，景点按偏好轮询分配）==========
        daily_resp = ""
        try:
            daily_resp = _generate_daily_routes_universal(
                destination, days, attractions, preferences, budget_level,
                price_factor=price_factor, tier_tag=tier_tag, is_sm_city=is_sm_city,
                travelers=travelers
            )
        except Exception as edr:
            print(f"[Orchestrator] 每日行程模块异常，用 fallback 兜底: {edr}")
            try:
                fb = _generate_daily_routes(destination, days, "", budget_level or "舒适型")
                daily_resp = fb.get("response") or fb.get("tips") or f"【每日行程 · {destination}】建议{days}天分片区游玩：市中心1天+近郊1天+老城1天，避免折返。"
            except Exception:
                daily_resp = f"【📅 每日行程安排 · {destination}】共{days}天：Day1 市中心经典景点+步行街；Day2 近郊自然/人文景区；Day3 老城深度+博物馆+采购返程特产。"

            internal_transport_resp = self._correct_cities(internal_transport_resp, origin=origin, destination=destination)
            hotels_resp = self._correct_cities(hotels_resp, origin=origin, destination=destination)
            food_resp = self._correct_cities(food_resp, origin=origin, destination=destination)
            daily_resp = self._correct_cities(daily_resp, origin=origin, destination=destination)
            budget_resp = self._correct_cities(budget_resp, origin=origin, destination=destination)
        except Exception:
            pass  # 矫正失败就算了，不影响主流程

        # 版本指纹
        try:
            if f"（{_rev}）" not in daily_resp:
                daily_resp = daily_resp.replace("💡 ", f"（{_rev}）💡 ", 1)
        except Exception:
            pass

        print(f"[Orchestrator] plan_full_trip 完成 ✅: {origin or '(无出发地)'}→{destination}, {days}天 | 出发详址:{depart_detail_addr or '(无)'} | 目的地详址:{dest_detail_addr or '(无)'} | 🏨酒店:{'OK' if hotels_resp else '空'} | 🍜美食:{'OK' if food_resp else '空'} | 🚗市内交通:{'OK' if internal_transport_resp else '空'} | 💰预算:{'OK' if budget_resp else '空'}")

        return {
            "session_id": session_id,
            "_build_revision": _rev,
            "trip_plan": {
                    "resource_note": resource_note or "",  # 资源匮乏城市方案顶部提示备注
                "origin": origin,
                "destination": destination,
                # ===== 新字段透传：CitySelector 完整对象 + 选填详细位置 =====
                # 出发城市完整对象（level=city，8 字段：level/provinceId/provinceName/cityId/cityName/districtId/districtName/lng=null/lat=null）
                "departLocation": depart_loc if isinstance(depart_loc, dict) else None,
                # 出发地详细位置（纯文本选填，如：外滩 / 虹桥火车站 / 迪士尼；后端酒店加权用）
                "departDetailAddress": depart_detail_addr or "",
                # 目的地城市完整对象（同上）
                "destLocation": dest_loc if isinstance(dest_loc, dict) else None,
                # 目的地详细位置（纯文本选填；后端酒店加权优先推荐周边）
                "destDetailAddress": dest_detail_addr or "",
                # ===== 行程 9 大板块（顺序严格对齐用户要求：行前贴士→天气→景点→往返交通→市内交通→住宿→美食→每日行程→总预算）=====
                # 💡 1. 行前贴士（暖心贴士：3 级分类：务必注意/建议准备/仅供参考）
                "warm_tips": warm_tips_resp or "",
                # 🌤️ 2. 天气参考
                "weather": weather_resp or "暂无天气信息",
                # 🏞️ 3. 景点速览（评分从高到低，附价格）
                "attractions": attraction_resp or "暂无景点推荐",
                # 🚄 4. 往返城际交通（长沙↔郴州高铁/大巴）
                "transport": transport_resp or "暂无交通方案",
                # 🚗 5.（新增）市内交通体系：单人/双人/多人交通 + 黑车避雷 + 公交短板
                "internal_transport": internal_transport_resp or "暂无市内交通建议",
                # 🏨 6. 平价住宿推荐（3 片区 + 4 家酒店 + 山区提示）
                "hotels": hotels_resp or "暂无酒店推荐",
                # 🍜 7. 郴州吃喝玩乐推荐（正餐+小吃+解暑饮+夜市+避雷）
                "food": food_resp or "暂无美食推荐",
                # 📅 8. 每日行程安排（郴州特化：地理顺路不绕路 + 每日绑定就近美食 + 删除地铁）
                "daily_routes": daily_resp or "暂无每日行程",
                # 💰 9.（新增）三日出行总预算（经济型 vs 标准型 明细表）
                "budget": budget_resp or "暂无预算建议",
            },
            "status": "completed",
        }


# ============================================================
# 【全域公平化改造】通用生成工具函数（所有城市平等对待，消除郴州特化优待）
# 规则：
#   1. 信息真实优先：ATTRACTION_DB 有则直接用，无则用通用安全模板（不编造具体地名/店名，只用城市级通用组合）
#   2. 出行偏好严格生效：景点/美食/行程 按标签加权 + 多标签均衡覆盖
#   3. 所有城市输出结构完全一致（9大板块 + 同深度），不出现质量断崖
#   4. 旅游资源匮乏时，自动用「老街/博物馆/步行街/近郊公园/本地美食」丰富 + 方案开头加备注
# ============================================================

# ---------- 偏好标签 ↔ 景点类别 映射（多对多，用于加权和过滤）----------
PREF_CATEGORY_MAP = {
    "美食":       ["美食"],
    "自然风光":   ["自然风光"],
    "人文古迹":   ["历史文化"],
    "网红打卡":   ["主题乐园", "历史文化", "美食"],
    "休闲度假":   ["自然风光", "主题乐园"],
    "亲子游玩":   ["主题乐园", "博物馆", "自然风光"],
    "户外运动":   ["自然风光"],
    "历史文博":   ["博物馆", "历史文化"],
    "城市逛吃":   ["美食", "历史文化"],
    "小众秘境":   [],  # 通用池末尾补充
}
# 用户10个标签的友好顺序（用于均衡轮询取景点）
PREF_ORDER = ["美食","自然风光","人文古迹","网红打卡","休闲度假","亲子游玩","户外运动","历史文博","城市逛吃","小众秘境"]


def _apply_preference_weight(attractions: list, preferences: list) -> list:
    """根据用户勾选的偏好标签对景点重排：匹配标签的排前面，多标签均衡覆盖"""
    if not attractions:
        return []
    prefs = [p for p in (preferences or []) if p in PREF_CATEGORY_MAP]
    if not prefs:
        # 无偏好：纯按评分降序
        return sorted(attractions, key=lambda a: float(a.get("rating", 0) or 0), reverse=True)

    # 给每个景点打匹配分（匹配标签数越多越靠前）
    def score(a):
        cats = set((a.get("category") or "").split("+"))
        matched = 0
        for p in prefs:
            if any(c in PREF_CATEGORY_MAP.get(p, []) for c in cats):
                matched += 1
        rating = float(a.get("rating", 0) or 0)
        return (matched, rating)  # 先按匹配标签数，再按评分

    return sorted(attractions, key=score, reverse=True)


def _enrich_attractions_generic(destination: str, preferences: list, days: int, max_results: int = 8) -> list:
    """ATTRACTION_DB 无数据时的通用安全景点补全器（不编造虚构点位，只用城市级通用组合）
    保证输出 ≥ days*2 个景点；返回格式与 ATTRACTION_DB 一致"""
    need = max(max_results, days * 2)
    # 通用候选池（全部是「城市名 + 通用类别」，不编造具体景点名，用户到当地可通过地图搜索到真实点位）
    generic_pool = [
        {"name": f"{destination}老城区历史文化街区", "category": "历史文化", "rating": 4.3, "ticket": "免费", "duration": "2-3小时",
         "desc": f"漫步{destination}老城脉络，探访传统街巷与本地烟火气，感受真实城市气质"},
        {"name": f"{destination}市博物馆（或城市规划展览馆）", "category": "博物馆", "rating": 4.4, "ticket": "免费（凭身份证预约）", "duration": "2-3小时",
         "desc": f"快速读懂{destination}的前世今生：出土文物、城市变迁史、民俗文化，系统了解一座城"},
        {"name": f"{destination}中心步行街/商业广场", "category": "美食", "rating": 4.2, "ticket": "免费", "duration": "2小时",
         "desc": f"{destination}本地餐饮与购物最集中的区域，正餐+小吃+奶茶一站式解决，本地人也常去"},
        {"name": f"{destination}城市中央公园（或人民公园/生态公园）", "category": "自然风光", "rating": 4.1, "ticket": "免费", "duration": "1.5-2小时",
         "desc": f"{destination}市内最大的开放式公园，散步吸氧、拍照发呆，逛累了就地休整"},
        {"name": f"{destination}近郊生态休闲区（山水/湖泊/森林公园）", "category": "自然风光", "rating": 4.3, "ticket": "约30-60元（以现场为准）", "duration": "半天",
         "desc": f"离市区30-60分钟车程的近郊自然放松地，徒步/骑行/野餐皆宜，适合半日慢游"},
        {"name": f"{destination}本地人文古迹（古寺/古塔/古祠堂）", "category": "历史文化", "rating": 4.2, "ticket": "多为免费，个别约20元", "duration": "1.5小时",
         "desc": f"{destination}留存至今的历史印记，感受当地的文化传承与建筑之美"},
        {"name": f"{destination}文创园/艺术街区（旧厂改造）", "category": "历史文化", "rating": 4.1, "ticket": "免费", "duration": "2小时",
         "desc": f"工业遗存改造的创意园区，咖啡馆+独立书店+手作店，适合年轻人拍照打卡"},
        {"name": f"{destination}夜市/夜宵一条街", "category": "美食", "rating": 4.3, "ticket": "免费", "duration": "2-3小时",
         "desc": f"{destination}本地人深夜觅食的聚集地：烧烤、卤味、炒菜、啤酒，烟火气最浓"},
        {"name": f"{destination}科技馆/美术馆/图书馆新馆", "category": "博物馆", "rating": 4.2, "ticket": "免费（部分展收费）", "duration": "2-3小时",
         "desc": f"适合避暑躲雨+亲子遛娃的文化空间，空调足、内容多，半天时间轻松过"},
        {"name": f"{destination}滨江/滨湖景观带", "category": "自然风光", "rating": 4.2, "ticket": "免费", "duration": "1.5小时",
         "desc": f"沿河/沿江的步行绿道，清晨跑步、傍晚散步、晚上看夜景都是绝佳选择"},
        {"name": f"{destination}周边古镇/古村（距市区1小时车程）", "category": "历史文化", "rating": 4.3, "ticket": "免费或约40元", "duration": "半天",
         "desc": f"保留传统风貌的近郊古镇，逛老街、吃农家菜、买点土特产，节奏慢压力小"},
        {"name": f"{destination}本地老字号小吃集中地", "category": "美食", "rating": 4.4, "ticket": "免费", "duration": "1.5小时",
         "desc": f"{destination}经营十几年以上的老店扎堆区域，早餐/小吃最地道，本地人最认可"},
    ]
    # 根据偏好标签优先选取匹配类别的景点，再补其余
    result = _apply_preference_weight(generic_pool, preferences)
    return result[:need]


def _get_resource_poor_note(destination: str, has_real_attractions: bool, attractions_count: int, days: int) -> str:
    """判断是否需要在方案开头加「旅游公开数据少」的备注"""
    if has_real_attractions and attractions_count >= days * 2:
        return ""  # 资源充足，无备注
    return (
        f"📝 **温馨提示（请先看）**：目前公开渠道可查询到的「{destination}」官方文旅景点结构化数据较少，"
        f"本方案基于通用城市深度游特征为您精心设计：涵盖老城漫步、博物馆、步行街、近郊休闲与本地美食等**真实可落地的通用板块**。"
        f"景点具体名称以您通过「{destination}文旅官方公众号/大众点评/高德地图」搜索到的实时结果为准，出行前建议确认预约与开放状态。祝您旅途愉快！\n"
    )


def _generate_hotels_universal(destination: str, budget_level: str, first_day_spot_hint: str = "",
                               price_factor: float = 1.0, tier_tag: str = "二线", is_sm_city: bool = False) -> str:
    """
    通用酒店生成器：
      - 所有城市同结构（3大片区+4家档次描述+入住提示），3档预算完全不同
      - 禁止输出具体连锁品牌名称/特定酒店名，只描述档次类型+动态价格区间
      - 物价系数 price_factor 根据城市线级动态调整；一线上浮 / 三四线下调
      - 小众城市（is_sm_city=True）高端档**降级**说明：如实描述当地能达到的最高档次，不虚构五星配置
    """
    # 四舍五入到 10 元，保持价格区间整齐
    def _pf(amount: int) -> int:
        return int(round(amount * price_factor / 10.0) * 10)

    if budget_level == "经济型":
        # 经济型：二线基准 100-180 元
        price_low, price_mid, price_high = _pf(100), _pf(150), _pf(190)
        title = f"【酒店推荐 · {destination}】经济型住宿（{price_low}-{price_high} 元 · 性价比首选 · 实用为主）"
        areas = [
            ("火车站 / 汽车总站片区", "市区公交枢纽，去周边景区有直达大巴；便利店夜宵店多，深夜到也不怕饿"),
            ("市中心 / 步行街周边", "吃饭逛街最方便，步行范围内覆盖正餐、小吃、商场，城市生活最便利"),
            ("老城区 / 主干道换乘站旁", "本地氛围最浓，价格比新城区便宜 15%-20%，公交出行基本覆盖"),
        ]
        hotels = [
            (f"全国连锁经济型酒店（火车站店）", f"{_pf(95)}-{_pf(145)}", f"{destination}火车站打车 5 分钟范围",
             f"距核心景区约 3-6 km，公交直达", "卫生稳定、价格最低，单人出行或赶火车首选"),
            ("本土连锁经济酒店（市中心店）", f"{_pf(115)}-{_pf(165)}", f"{destination}步行街步行 8-10 分钟",
             f"距核心景区约 2-5 km", "位置好吃饭方便，价格比火车站略高但体验更稳"),
            ("社区型快捷酒店（老城区店）", f"{_pf(105)}-{_pf(155)}", f"{destination}老城区主干道旁",
             f"距核心景区约 4-7 km", "价格友好，本地客人多，周边早餐店地道"),
            ("交通枢纽平价住宿（汽车总站店）", f"{_pf(85)}-{_pf(135)}", f"{destination}汽车总站步行 3 分钟",
             f"距核心景区约 5-8 km", "最便宜的档位，适合当天赶大巴去周边景区"),
        ]
        tips = [
            "⚠️ 经济档大概率不提供牙刷牙膏拖鞋，提前自备能省 20 元",
            "⚠️ 老楼店先问有没有电梯，带行李箱别住高层无电梯的房间",
            "👍 订房选「可免费取消」的，行程变动可以无损退",
            "👀 晚上 10 点后到店，先打电话确认前台还在值班，避免到了没人",
        ]
    elif budget_level == "高端型":
        # 高端档：二线基准 280-450 元；小众城市降级说明
        downgrade_notice = ""
        if is_sm_city:
            downgrade_notice = (f"\n⚠️ **关于高端配套降级说明**：「{destination}」属于{tier_tag}城市，"
                               f"当地公开渠道暂无国际五星/高端度假配套记录，本方案推荐**当地能达到的最高档次住宿**："
                               f"主城区挂牌四星/精品度假酒店/景区内品质民宿。如对品牌有要求，可搜索最近的高等级高铁站/机场城市 1 小时车程内的国际连锁。\n")
        price_low, price_mid, price_high = _pf(280), _pf(360), _pf(460)
        title = f"【酒店推荐 · {destination}】{tier_tag}城市高端档住宿（{price_low}-{price_high} 元 · 品质优先 · 体验拉满）{downgrade_notice}"
        areas = [
            ("城市核心商圈 / CBD 旁", "购物餐饮夜生活最成熟，下楼就是商圈，打车去景点也方便"),
            ("景区周边 / 景观带（如有）", "离第 1 天核心景区近，早上多睡 1 小时，返程不用赶时间"),
            ("滨江滨湖 / 新城高品质居住区", "环境好、房间新、夜景美，适合度假放松，打车到市区 15 分钟内"),
        ]
        if is_sm_city:
            # 小众城市：改成当地能达到的最高档次描述
            hotels = [
                (f"{destination}主城区挂牌四星/品质商务酒店", f"{_pf(260)}-{_pf(380)}", f"{destination}核心商圈步行 3-5 分钟",
                 f"距核心景区约 2-5 km，打车 10 分钟内", "本地品质商务酒店，床品卫生和服务有保障，商务+度假均合适"),
                (f"{destination}新城高端型精品酒店", f"{_pf(240)}-{_pf(360)}", f"{destination}新城主干道旁，部分房有景观",
                 f"距核心景区约 3-6 km", "房间面积大、设施比较新，对睡眠和卫生有要求的选这类"),
                (f"{destination}景区内精品民宿 / 度假型酒店", f"{_pf(280)}-{_pf(450)}", f"{destination}第 1 天核心景区内或门口",
                 f"距核心景区步行 5 分钟内", "推窗见景，早上不用 5 点起床赶车，体验感秒杀市区住宿"),
                (f"{destination}本地人气高端酒店（地标型）", f"{_pf(320)}-{_pf(460)}", f"{destination}地标建筑或景区旁",
                 f"距核心景区约 1-4 km，部分步行可达", "带家人/纪念日出行有仪式感，具体选择以「{destination}文旅官方/点评评分≥4.6」为准"),
            ]
        else:
            # 非小众城市（一线/新一线/二线）：正常高端档描述
            hotels = [
                (f"{destination}核心商圈高端商务酒店", f"{_pf(320)}-{_pf(420)}", f"{destination}核心商圈步行 3 分钟",
                 f"距核心景区约 2-5 km，打车 10 分钟", "品质稳定早餐丰盛，商务+度假都合适，选点评 4.6 分以上的"),
                (f"{destination}景观型中高端精品酒店", f"{_pf(280)}-{_pf(380)}", f"{destination}新城主干道旁，部分房有江/湖景",
                 f"距核心景区约 3-6 km", "房间大、舒适度高，对睡眠和体验有要求的优先选"),
                (f"{destination}本地高星级品质酒店", f"{_pf(380)}-{_pf(460)}", f"{destination}地标建筑或景区旁",
                 f"距核心景区约 1-4 km，部分步行可达", "泳池+健身房基本配齐，带家人/纪念日出行有仪式感"),
                (f"{destination}景区内精品度假酒店", f"{_pf(300)}-{_pf(450)}", f"{destination}第 1 天核心景区内或门口",
                 f"距核心景区步行 5 分钟内", "推窗见景，早上不用 5 点起床赶车，体验感秒杀市区连锁"),
            ]
        tips = [
            "💎 旺季（节假日/暑期）提前 7-14 天锁房，好房型晚了就没",
            "💎 入住时主动问能否免费升级，淡季 30%+ 概率成功（套房/景观房）",
            "👍 订房选「含早餐」的，品质酒店早餐值回票价",
        ]
        if is_sm_city:
            # 小众城市：不推荐"行政酒廊"这类大概率不存在的配置
            tips.append("👀 有免费下午茶/大堂吧茶点的优先选，能省下一顿饭钱")
        else:
            tips.append("👀 有行政酒廊/下午茶的优先选，能省 2 顿饭钱")
    else:  # 舒适型（原标准型）
        # 舒适型：二线基准 180-280 元
        price_low, price_mid, price_high = _pf(180), _pf(230), _pf(285)
        title = f"【酒店推荐 · {destination}】{tier_tag}城市舒适型住宿（{price_low}-{price_high} 元 · 平衡性价比与体验）"
        areas = [
            ("市中心 / 核心步行街片区", "吃饭逛街最方便，步行覆盖正餐+小吃+商场，景点打车起步价"),
            (f"第 1 天核心景区周边 / {first_day_spot_hint or '城市主干道旁'}", "离第 1 天景区近，早出发省时间；附近早餐店多"),
            ("火车站 / 高铁站 与 市区 中间位置", "两头都近，打车到车站 15 分钟、到景区 10 分钟，进出城方便"),
        ]
        hotels = [
            ("全国连锁舒适型酒店（火车站店）", f"{_pf(175)}-{_pf(245)}", f"{destination}火车站打车 5 分钟范围",
             f"距核心景区约 3-5 km，公交直达", "交通最省心，房间比经济型宽敞，体验稳定"),
            ("商圈型品质舒适酒店（步行街店）", f"{_pf(215)}-{_pf(285)}", f"{destination}步行街步行 5 分钟",
             f"距核心景区约 2-4 km", "位置无敌，楼下就是吃喝，设计感和卫生都不错"),
            ("景区周边品质舒适酒店", f"{_pf(195)}-{_pf(265)}", f"{destination}第 1 天核心景区打车 8 分钟内",
             f"距核心景区约 2-3 km", "床品舒适度出色，玩一天回房能彻底放松"),
            ("新城区精品舒适酒店（主干道店）", f"{_pf(180)}-{_pf(265)}", f"{destination}新城区主干道旁",
             f"距核心景区约 4-6 km，打车 12 分钟内", "房间设施比较新，停车方便，自驾或打车出行首选"),
        ]
        tips = [
            "⚠️ 订房选「可免费取消」的，行程变动无损退",
            "👍 有会员的直接用官方 APP 订，升级概率比第三方高",
            "👍 对香氛/气味敏感的，先看评论有没有说装修味道大",
            "👀 连续住 2 晚以上，打电话问前台能不能延退到 14:00 退房（大概率同意）",
        ]
    # 统一拼装（所有城市结构完全一致，消除权重差异）
    lines = [title]
    lines.append("\n📍 适合本次行程的 3 大优选住宿片区：")
    for i, (name, adv) in enumerate(areas, 1):
        lines.append(f"  {i}. **{name}**：{adv}")
    lines.append(f"\n🏨 4 家{budget_level}档住宿参考（越靠前越推荐，具体店名以地图/点评实时搜索为准）：")
    for i, (name, price, loc, dist, reason) in enumerate(hotels, 1):
        num = "①②③④"[i-1]
        lines.append(f"  {num} {name}")
        lines.append(f"       💰 大概价格：{price} 元/晚（节假日+30%~50%，以当日APP为准）")
        lines.append(f"       📍 位置亮点：{loc}")
        lines.append(f"       🚶 距离核心景点：{dist}")
        lines.append(f"       🎯 适合理由：{reason}")
    lines.append("\n💡 酒店入住小提示：")
    for t in tips:
        lines.append(f"  {t}")
    return "\n".join(lines)


def _generate_food_universal(destination: str, budget_level: str, preferences: list,
                              price_factor: float = 1.0, tier_tag: str = "二线", is_sm_city: bool = False) -> str:
    """
    通用美食生成器：
      - 所有城市同结构（5大板块），预算档位+偏好标签双维度生效
      - **禁止输出具体店名/特定菜名/本地特色食材（如：XX 三文鱼/XX 猪脚饭）**，只描述档次类型 + 人均价格区间
      - 物价系数 price_factor 根据城市线级动态调整价格区间
      - 小众城市（is_sm_city=True）高端档**降级说明**：如实描述当地能达到的最高餐饮档次，不虚构黑珍珠/高端私厨
    """
    prefs = set(preferences or [])
    has_food = bool({"美食", "城市逛吃", "网红打卡"} & prefs)
    has_family = "亲子游玩" in prefs
    has_outdoor = "户外运动" in prefs
    has_culture = bool({"人文古迹", "历史文博"} & prefs)

    # 四舍五入到 5 元，价格区间更自然
    def _pf(amount: int) -> int:
        return int(round(amount * price_factor / 5.0) * 5)
    def _pfr(lo: int, hi: int) -> str:
        return f"{_pf(lo)}-{_pf(hi)}元"

    if budget_level == "经济型":
        title = f"【美食玩乐 · {destination}】{tier_tag}城市经济型吃喝攻略（人均 {_pf(15)}-{_pf(60)} 元 · 省钱吃饱不踩坑）"
        main_dishes = [
            (f"{destination}本地人早餐三件套（粉/面/粥类）", f"人均{_pfr(8, 15)}", "老城区早餐扎堆的街道（问出租车司机最靠谱）", "米面/粉/粥+卤蛋+小菜，吃饱一上午不饿", "必吃"),
            (f"{destination}家常菜土菜馆（居民楼底商，非网红）", f"人均{_pfr(30, 50)}", "老城区居民楼底商、远离景区 500 米以上", "本地食材家常做法，价格实惠分量足", "必吃"),
            (f"{destination}老字号粉/面/馄饨店（10 年+老店）", f"人均{_pfr(10, 20)}", "本地人排队的老店（点评搜「老店/老字号」看 3 年以上评论）", "本地最地道口味，一顿正餐价格的三分之一", "必吃"),
            (f"{destination}本地快餐连锁 / 简餐盖饭类", f"人均{_pfr(18, 28)}", "商业街、写字楼底商", "赶时间时的稳妥选择，吃饱卫生不踩雷", "可选"),
        ]
        if has_food:  # 勾选美食 → 多给2家正餐
            main_dishes.extend([
                (f"{destination}本地特色菜小馆（非连锁）", f"人均{_pfr(35, 55)}", "老城区深巷内、点评评分 4.5 以上评论数≥300", "本地菜做法地道，比景区店便宜一半", "必吃（美食党优先）"),
                (f"{destination}本地农贸市场周边熟食/卤味", f"人均{_pfr(15, 25)}", "农贸市场门口、下午 4 点后出摊", "本地人日常买的卤菜，配粉/面/啤酒绝了", "必吃（美食党）"),
            ])
        snacks_title, snacks = "🥟 ② 街头地道小吃合集（推车/老店全覆盖）", [
            ("本地炸串/烤串类（推车式）", f"{_pf(5)}-{_pf(15)}元/份", "夜市、学校门口晚上 6 点后", "放学/下班高峰排队的推车最靠谱"),
            ("手工包子/饺子/烧麦类（老店）", f"{_pf(2)}-{_pf(8)}元/个", "老城区居民区底商早上 7-10 点", "皮薄馅大，早餐边走边吃"),
            ("本地冰品/甜品类（夏天冰冬天热）", f"{_pf(5)}-{_pf(12)}元/碗", "步行街、夜市入口", "解暑解辣，饭后甜品"),
            ("卤味拼盘/卤菜（老店）", f"{_pf(15)}-{_pf(30)}元/份", "菜市场、老城区深巷", "配米饭或下酒都合适"),
            ("煎饼/饼类/主食夹肉类", f"{_pf(6)}-{_pf(15)}元/个", "公交车站旁、学校门口", "赶时间时的快速能量补充"),
        ]
        drinks_title, drinks = "🧋 ③ 本地特色饮品 / 解暑解辣推荐", [
            ("本地连锁茶饮 / 鲜果茶类", f"{_pf(10)}-{_pf(18)}元/杯", "步行街、商业街都有", "冰的解辣解腻，夏天逛街标配"),
            ("本地凉茶/糖水铺（非连锁）", f"{_pf(8)}-{_pf(15)}元/碗", "老城区居民区底商", "夏天清热，冬天暖胃，比连锁更有特色"),
            ("瓶装本地乳饮 / 果汁 / 气泡水", f"{_pf(3)}-{_pf(8)}元/瓶", "便利店、超市", "方便携带，逛景点时随手拿"),
        ]
        night_title, night_market = "🌙 ④ 夜宵美食聚集地 / 夜市推荐", [
            (f"{destination}老城区老牌夜市一条街", f"人均{_pfr(25, 50)}", "晚上 6 点-凌晨 2 点，打车问司机「本地人吃夜宵去哪」", "烧烤+炒粉+啤酒，本地烟火气最浓"),
            (f"{destination}高校周边小吃街", f"人均{_pfr(15, 35)}", "下午 4 点-晚上 11 点，地图搜「大学城/高校」附近", "学生党价格，品类多更新快，不踩雷"),
            (f"{destination}火车站/汽车站周边宵夜店", f"人均{_pfr(20, 40)}", "24 小时营业的店", "深夜到/凌晨赶车的应急选择，选连锁或人多的"),
        ]
        avoid = [
            "❌ 景区门口 10 米内的「百年老字号」「网红同款」：贵 30%+ 且口味普遍一般，往外走 500 米价格腰斩",
            "❌ 出租车/三轮车司机主动推荐的「本地特色店」：大概率有提成，直接礼貌拒绝，用点评搜评分 4.3+ 评论≥200 的店",
            "❌ 菜单没有明码标价的大排档：先问价格再下单，避免吃完被宰",
            "❌ 生冷生食类（生腌/醉虾蟹等）：肠胃弱的别碰，旅行中拉肚子最影响心情；实在想吃选连锁店+配蒜",
        ]
    elif budget_level == "高端型":
        # 小众城市高端档降级说明
        downgrade_notice = ""
        if is_sm_city:
            downgrade_notice = (f"\n⚠️ **关于高端餐饮降级说明**：「{destination}」属于{tier_tag}城市，"
                               f"当地暂无高端品牌连锁餐饮公开记录，本方案推荐**当地能达到的最高档次正餐**："
                               f"主城区人气中高端餐厅 + 景区周边品质度假配套。如对品牌有要求，可搜索最近的高等级高铁站/机场城市 1 小时车程内的高端商圈。\n")
        title = (f"【美食玩乐 · {destination}】{tier_tag}城市高端吃喝攻略"
                 f"（正餐人均 {_pf(100)}-{_pf(200)} 元 · 品质版）{downgrade_notice}")
        main_dishes = [
            (f"{destination}人气品质本帮菜餐厅（包间需提前预约）", f"人均{_pfr(150, 250)}", f"{destination}景观位（江/湖/山）或 老建筑改造品质餐厅", "本地较高水平的菜品呈现，适合重要宴请/纪念日", "必吃"),
            (f"{destination}地标高楼层景观餐厅 / 酒店顶层餐吧", f"人均{_pfr(180, 300)}", f"{destination}核心商圈最高楼或高星酒店顶层", "夜景+美食双重体验，日落时分去最美", "必吃"),
            (f"{destination}特色食材品质做法（当季时令鲜品）", f"人均{_pfr(120, 200)}", "新区高端餐饮聚集区、点评人气高分榜前列店", "本地食材+品质出品，体验感远胜景区店", "必吃"),
            (f"{destination}品质异国料理 / 融合菜（连锁大牌）", f"人均{_pfr(150, 280)}", "CBD 商圈、高端商场 L4-L5 层", "吃腻本地菜时的调剂选项，品质稳定", "可选"),
        ]
        if has_family:  # 亲子 → 加家庭友好餐厅
            main_dishes.append((f"{destination}亲子友好品质餐厅（儿童餐+游乐区）", f"人均{_pfr(80, 150)}", "大型商场 3-4 层、亲子乐园旁", "有儿童餐具/儿童餐/小型游乐区，大人吃好小孩不闹", "必吃（亲子）"))
        snacks_title, snacks = "🍰 ② 品质下午茶 / 精品小吃合集", [
            ("品质酒店下午茶套餐", f"人均{_pfr(100, 180)}元/套", "高星酒店 Lobby 层或顶层，下午 2-5 点", "仪式感+环境+甜点品质，适合拍照打卡"),
            ("精品烘焙 / 独立面包甜品店", f"人均{_pfr(20, 50)}", "老城区洋楼区、文创园附近", "手作面包/精品甜品，下午茶外带款"),
            ("本地老字号甜品/糖水品质升级店", f"人均{_pfr(25, 50)}", "老街老店的品质分店或文创园店", "传统口味+现代摆盘，好吃又出片"),
        ]
        drinks_title, drinks = "🍷 ③ 品质特色饮品 / 推荐", [
            ("精品独立咖啡馆（手冲/意式）", f"人均{_pfr(30, 60)}", "老城区、文创园、CBD", "提神醒脑+发呆，豆子品质比连锁好"),
            ("精酿品鉴 / 本地小酒馆", f"人均{_pfr(80, 180)}", "老建筑改造、景观位旁", "体验本地酒文化，适合小酌"),
            ("高端鲜榨果汁 / 有机茶饮", f"人均{_pfr(25, 45)}", "高端商场、精品超市", "健康不发胖，逛景点时随手带"),
        ]
        night_title, night_market_base = "🥂 ④ 夜宵 / 深夜品质美食聚集地", [
            (f"{destination}景观位夜宵区域", f"人均{_pfr(150, 250)}", "景观位周边品质夜宵店，晚上 6 点-凌晨 1 点", "夜景+烧烤/小海鲜/啤酒，体验感拉满"),
            (f"{destination}高端商场 B1 层美食街（营业到 22 点）", f"人均{_pfr(80, 150)}", "CBD 高端商场负一层", "品牌全卫生好，空调足环境好"),
        ]
        if is_sm_city:
            night_market = night_market_base + [
                ("品质酒店大堂吧 / 24 小时客房送餐", f"人均{_pfr(120, 260)}", "入住酒店内", "懒得出去时的稳妥选择，按酒店实际提供情况选择"),
            ]
        else:
            night_market = night_market_base + [
                ("高星酒店行政酒廊 / 24 小时送餐", f"人均{_pfr(150, 300)}", "入住酒店内", "懒得出去时的稳妥选择，品质稳定"),
            ]
        avoid = [
            "❌ 「人气打卡餐厅」提前不预约直接过去：排队 1-2 小时很正常，提前 2-3 天用官方小程序约",
            "❌ 高端餐厅团购券周末/节假日不可用：下单前看清楚使用规则",
            "❌ 景区内高端餐厅：价格是外面的 1.5-2 倍，品质通常不如市区同价位店",
            "❌ 司机推荐的「人气本帮菜」：大概率有提成，用点评人气高分榜筛选最靠谱",
        ]
    else:  # 舒适型（原标准型）
        title = f"【美食玩乐 · {destination}】{tier_tag}城市舒适型吃喝全攻略（平衡体验与价格）"
        main_dishes = [
            (f"{destination}本地人气老字号正餐店", f"人均{_pfr(50, 80)}", "老城区主干道旁，开 10 年以上老店", "本地菜正宗做法，本地人办酒席常去，不踩雷", "必吃"),
            (f"{destination}特色食材专营店（当季时令鲜品）", f"人均{_pfr(60, 90)}", "近郊农家乐一条街、出城主干道旁", "新鲜食材分量足，4 人以上去性价比最高", "必吃"),
            (f"{destination}热门排队店（本地口碑型）", f"人均{_pfr(50, 90)}", "商圈、高校周边、居民区的排队王", "错峰（11:30 前/13:30 后）去不用排队，味道有保证", "必吃"),
            (f"{destination}融合菜/创意菜（年轻人爱去）", f"人均{_pfr(70, 110)}", "文创园、CBD、新城区商场", "环境好菜品有创意，适合情侣/朋友小聚", "可选"),
        ]
        if has_family:
            main_dishes.append((f"{destination}家庭友好型餐厅（有儿童餐）", f"人均{_pfr(60, 100)}", "大型商场 3-4 层", "有儿童餐具/餐椅/宝宝餐，大人小孩都满意", "必吃（亲子）"))
        if has_food:
            main_dishes.append((f"{destination}本地菜小馆（居民楼下宝藏店）", f"人均{_pfr(40, 70)}", "老城区居民楼底商，点评评论 500+ 评分≥4.5", "价格实惠味道正宗，本地人日常吃饭的店", "必吃（美食党）"))
        snacks_title, snacks = "🥟 ② 地道小吃合集（推车+老店全覆盖）", [
            ("本地招牌粉/面（排队老店）", f"{_pf(10)}-{_pf(25)}元/碗", "老城区深巷内，早上 7-9 点本地人最多的店", "最地道的本地口味，早餐首选"),
            ("老字号卤味/凉拌菜", f"{_pf(15)}-{_pf(35)}元/份", "菜市场门口、老街上开 10 年+的店", "买一点回酒店配啤酒/当零食都合适"),
            ("本地特色糕点/糯米饭团类", f"{_pf(5)}-{_pf(15)}元/份", "步行街、菜市场旁的连锁老店", "随手带一个逛景点，饿了垫肚子"),
            ("炸串/铁板烧推车（学生排队款）", f"{_pf(10)}-{_pf(25)}元/份", "中学/大学门口，晚上 6 点后出摊", "学生用脚投票的味道，价格公道分量足"),
            ("手工糖水/冰品/豆花", f"{_pf(8)}-{_pf(18)}元/碗", "老城区、步行街、文创园", "夏天解暑冬天暖胃，饭后甜品必备"),
        ]
        drinks_title, drinks = "🧋 ③ 本地特色饮品推荐（夏天解暑/冬天暖胃）", [
            ("本地连锁茶饮 / 鲜果茶品牌", f"{_pf(12)}-{_pf(22)}元/杯", "步行街、商圈、学校旁", "本地品牌更懂本地口味，甜度更合适"),
            ("老字号凉茶/糖水铺", f"{_pf(8)}-{_pf(18)}元/碗", "老城区居民区底商，开 10 年+", "传统配方，清热解暑/暖胃暖心"),
            ("本地乳饮 / 果汁（便利店可购）", f"{_pf(4)}-{_pf(15)}元/瓶", "超市/便利店/景区自动贩卖机", "方便携带，逛景点时随手拿"),
            ("精品独立咖啡馆", f"{_pf(25)}-{_pf(45)}元/杯", "文创园、老城区、CBD", "下午犯困时来一杯，提神醒脑继续逛"),
        ]
        night_title, night_market = "🌙 ④ 夜宵美食聚集地 / 夜市推荐", [
            (f"{destination}老城区老牌夜市街", f"人均{_pfr(40, 80)}", "晚上 6 点-凌晨 2 点，打车问司机「本地人夜宵哪里吃」", "烧烤+炒粉+啤酒+卤味，烟火气最浓的地方"),
            (f"{destination}高校周边美食街", f"人均{_pfr(30, 60)}", "下午 4 点-晚上 11 点，地图搜「XX 大学附近美食」", "学生价格，品类多更新快，踩雷概率低"),
            (f"{destination}24 小时营业的连锁粥粉店", f"人均{_pfr(20, 40)}", "主干道、车站旁", "深夜到/早起赶车，清淡养胃不踩雷"),
            (f"{destination}商场 B1 层美食区（营业到 22 点）", f"人均{_pfr(50, 90)}", "大型商场负一层", "品牌全环境好，空调足适合夏天"),
        ]
        avoid = [
            "❌ 景区门口 10 米内「网红百年老店」：价格贵 30%+ 口味一般，走出 500 米再吃",
            "❌ 出租车/三轮车司机主动推荐的餐厅：大概率有提成，礼貌拒绝，用点评搜「评分 4.3+ 评论≥200」",
            "❌ 菜单没明码标价的大排档：先问价格再下单，避免吃完被宰",
            "❌ 生冷生食类（生腌/醉虾蟹）肠胃弱别碰：旅行拉肚子最毁行程；想吃选连锁+配蒜",
            "❌ 网红店饭点直接过去：热门店排队 1-2 小时正常，错峰（11:30 前/13:30 后/17:30 前/19:30 后）或提前拿号",
        ]
    # 拼装（所有城市结构完全一致）
    lines = [title]
    lines.append("\n🍽️ ① 必吃本地特色正餐推荐（档次类型+人均+适合场景）")
    for i, (name, price, loc, reason, tag) in enumerate(main_dishes, 1):
        lines.append(f"  {i}. 🏠 **{name}** 【{tag}】")
        lines.append(f"       · 💰 人均：{price}")
        lines.append(f"       · 📍 位置：{loc}")
        lines.append(f"       · 💡 适合理由：{reason}")
    lines.append(f"\n{snacks_title}")
    for i, (name, price, loc, tip) in enumerate(snacks, 1):
        lines.append(f"  {i}. 🍴 **{name}**（{price}）")
        lines.append(f"       📍 {loc}  →  💡 {tip}")
    lines.append(f"\n{drinks_title}")
    for i, (name, price, loc, tip) in enumerate(drinks, 1):
        lines.append(f"  {i}. 🥤 **{name}**（{price}）")
        lines.append(f"       📍 {loc}  →  💡 {tip}")
    lines.append(f"\n{night_title}")
    for i, (name, price, time_loc, tip) in enumerate(night_market, 1):
        lines.append(f"  {i}. 🌃 **{name}**（{price}）")
        lines.append(f"       🕒📍 {time_loc}")
        lines.append(f"       💡 {tip}")
    lines.append("\n⚠️ ⑤ 吃喝避雷指南（别花冤枉钱）")
    for a in avoid:
        lines.append(f"  {a}")
    # 偏好标签彩蛋：勾选了特定偏好的追加专属提示
    extra = []
    if has_food:
        extra.append("🍜 **【美食党专属】**：额外建议用「点评必吃榜」+「小红书本地博主近 3 个月真实探店笔记」交叉筛选，能挖到比景区店便宜一半的宝藏小馆；每天留 1 个正餐给「随机漫步偶遇的本地人排队老店」。")
    if has_family:
        extra.append("👶 **【亲子专属】**：带娃吃饭优先选商场店（卫生+空调+儿童设施齐全）；路边摊/夜市尽量少吃生冷；随身带一次性围兜+湿纸巾+益生菌。")
    if has_culture:
        extra.append("🏛️ **【文化党专属】**：逛完博物馆/古迹，优先选老城区的「本地家常菜馆」，通常能吃到当地最正宗的传统菜做法（比网红店地道）。")
    if has_outdoor:
        extra.append("🚶 **【户外党专属】**：徒步/登山日的早餐要吃饱（粉+蛋+肉）；随身带能量棒+电解质水；山上别乱吃不明野果；晚餐多补充蛋白质+碳水。")
    if extra:
        lines.append("\n🎯 ⑥ 出行偏好专属提示")
        for e in extra:
            lines.append(f"  {e}")
    return "\n".join(lines)


def _generate_internal_transport_universal(destination: str, days: int, budget_level: str, preferences: list,
                                           price_factor: float = 1.0, tier_tag: str = "二线",
                                           travelers: int = 2, is_sm_city: bool = False) -> str:
    """通用市内交通：所有城市同结构（按人数选+黑车避雷+公交短板+偏好适配）
    - 价格按物价系数动态缩放；小众城市高端档如实描述当地能达到的最高档次
    """
    prefs = set(preferences or [])
    pf = max(0.5, price_factor or 1.0)
    travelers = max(1, int(travelers or 2))
    # 价格取整辅助：乘pf后四舍五入到5元
    def rmb(base):
        return int(round(base * pf / 5) * 5)
    lines = [f"【市内交通 · {destination}】景区真实出行建议（避免绕路 · 杜绝黑车坑）· {tier_tag}城市物价系数×{pf:.2f}"]
    lines.append(f"\n👥 本次出行人数：{travelers}人 · 按人数与档位推荐交通方式（直接抄作业）：")
    if budget_level == "经济型":
        lines.append("  👤 **单人出行（1人）**：公交 + 地铁（如有）+ 拼车APP为主")
        lines.append(f"       · 日常：公交/地铁坐市区景点，单次{rmb(1)}-{rmb(5)}元；高德/百度地图查实时公交")
        lines.append(f"       · 去近郊景区：用拼车APP/青旅前台拼团，人均比打快车省40%-60%")
        lines.append(f"       · 应急：美团/高德打车的特惠快车/拼车比滴滴便宜10%-20%")
        lines.append(f"  👥 **双人出行（2人/情侣）**：公交+网约车拼车，预算允许选打快车")
        lines.append(f"       · 市区内2人打快车平均比公交贵{rmb(15)}-{rmb(30)}元但省30-50分钟，按体力权衡")
        lines.append(f"       · 近郊景区：和其他游客拼车（青旅/民宿前台帮忙凑），人均{rmb(40)}-{rmb(90)}元")
        lines.append(f"  👨‍👩‍👧 **多人同行（3人+）**：公交/地铁最省；近郊景区直接在平台约车（明码标价）")
        lines.append(f"       · 3人打车和公交差价不大，直接打快车更省心")
        lines.append(f"       · 去程和返程司机电话分别存，避免旺季临时叫不到车")
    elif budget_level == "高端型":
        if is_sm_city:
            # 小众城市如实描述：没有7座商务车配套的话，推荐当地能达到的最高档次
            lines.append(f"  ⚠️ {destination}属于小众城市，高端商务车车队/专业导游较少，按当地能达到的最高档次配置：")
            lines.append(f"  👤 **单人（1人）**：全程打平台专车/当地口碑好的出租车（平台预约，明码标价）")
            lines.append(f"       · 平台：高德/携程「接送机+日租」选评分4.8以上的本地老司机")
            lines.append(f"       · 尽量固定同一位司机，不用每次叫车等车，省下来的时间值回票价")
            lines.append(f"  👥 **双人（情侣）**：全程本地舒适型商务车/5座SUV 包车（配老司机+代购优惠票）")
            lines.append(f"       · {days}天全程包车约 {rmb(600)}-{rmb(1100)} 元（根据公里数浮动），含油费过路费司机餐补")
            lines.append(f"       · 司机熟悉路线，不用自己导航，景点间无缝衔接，最省心")
            lines.append(f"  👨‍👩‍👧 **多人（3人+）**：7-9 座本地面包车/商务车（正规平台下单），人均更划算")
            lines.append(f"       · 4人均摊比2人还便宜30%，有老人小孩优先选这个")
            lines.append(f"       · 选正规平台下单，提前看行驶证+保险+合同，别走私人微信")
        else:
            lines.append(f"  👤 **单人（1人）**：全程打专车/7座商务车（平台预约，明码标价）")
            lines.append(f"       · 平台：高德/携程「接送机+日租包车」选评分4.9以上的老司机")
            lines.append(f"       · 每天固定司机固定车，不用每次叫车等车，省下来的时间值回票价")
            lines.append(f"  👥 **双人（情侣）**：全程 7 座商务车包车（配老司机+免费讲解/代购优惠票）")
            lines.append(f"       · {days}天全程包车约 {rmb(700)}-{rmb(1300)} 元（根据公里数浮动），含油费过路费司机餐补")
            lines.append(f"       · 司机熟悉路线，不用自己导航，景点间无缝衔接，最省心")
            lines.append(f"  👨‍👩‍👧 **多人（3人+）**：9-14 座商务车 + 专业导游（可选），人均更划算")
            lines.append(f"       · 4人均摊比2人还便宜30%，有老人小孩优先选这个")
            lines.append(f"       · 选正规平台下单，提前看营业执照+保险+合同，别走私人微信")
    else:  # 舒适型
        lines.append(f"  👤 **单人（1人）**：公交/地铁 + 网约车快车（平台明码标价）")
        lines.append(f"       · 日常市区：公交/地铁{rmb(1)}-{rmb(5)}元，赶时间就打快车{rmb(10)}-{rmb(30)}元/次")
        lines.append(f"       · 近郊景区：拼车APP / 民宿前台拼团，人均比单独打车省50%")
        lines.append(f"  👥 **双人（情侣）**：网约车快车 + 东线/近郊景区 半天包车")
        lines.append(f"       · 市区内2人打快车和公交差价{rmb(15)}-{rmb(35)}元，省30-50分钟时间，性价比高")
        lines.append(f"       · 远郊景区：半天包车（4小时）约{rmb(180)}-{rmb(330)}元，玩完直接送回市区酒店")
        lines.append(f"  👨‍👩‍👧 **多人（3人+）**：全程 7 座商务车半天/全天包车，人均划算最省心")
        lines.append(f"       · 4人均摊和公交差价不大，带老人小孩的首选，不用拎行李赶车")
        lines.append(f"       · 平台下单选评分4.8以上、接单≥1000单的老司机，靠谱率最高")
    lines.append("\n🚨 ⚠️ 黑车避雷（所有城市通用，务必遵守）：")
    lines.append("  · 火车站/高铁站/景区门口围上来问「去哪/打车吗/要导游吗」的，一律礼貌拒绝")
    lines.append("  · 喊价低于平台价30%以上的必有坑：中途加价/强制购物/拉到合作店吃饭")
    lines.append("  · 所有行程**必须在平台下单**（高德/携程/滴滴/美团），平台有定位+录音+保险，出事有地方投诉")
    lines.append("  · 包车先看司机行驶证/驾驶证/车辆保险单，拍照留存；禁止司机中途擅自加景点/购物")
    lines.append("\n🚌 公交与地铁短板说明（提前规划）：")
    lines.append("  · 大多数城市的**近郊景区**公交班次少（1-2小时一班），下午收车早（17:00-18:00末班），赶不上就只能打车")
    lines.append("  · 周末/节假日公交车会比平时挤很多，带大行李/婴儿车优先考虑打车")
    lines.append("  · 地铁末班车一般22:30-23:00，看完夜景/夜市后确认还能不能赶上，不行就直接打快车")
    lines.append("  · 建议买「乘车码」：微信/支付宝搜「XX乘车码」直接扫码，不用排队买地铁票/找零")
    # 偏好专属提示
    if "户外运动" in prefs or "自然风光" in prefs:
        lines.append("\n🥾 【自然风光/户外党专属】近郊山区/景区特别注意：")
        lines.append("  · 山区盘山公路容易晕车的提前吃晕车药，别坐后排看手机")
        lines.append("  · 下山比上山更伤膝盖，带护膝；穿防滑运动鞋，别穿拖鞋/高跟鞋")
    if "亲子游玩" in prefs:
        lines.append("\n👶 【亲子专属】带娃市内交通特别提醒：")
        lines.append("  · 优先选带安全座椅的网约车（平台筛选「舒适型/商务车+安全座椅」）")
        lines.append("  · 景区内接驳车/电瓶车别让小孩单独坐，全程大人陪同；婴儿车选轻便折叠款")
    return "\n".join(lines)


def _generate_budget_universal(destination: str, days: int, budget_level: str, intercity_train: int = 300,
                               price_factor: float = 1.0, tier_tag: str = "二线", travelers: int = 2,
                               is_sm_city: bool = False) -> dict:
    """通用3档预算表：所有城市同结构（6行数据+省钱/体验技巧），当前档位高亮👈
    - 物价系数 price_factor：一线1.5 / 新一线1.25 / 二线1.0 / 三四线0.65~0.75 / 小众0.6
    - travelers 出行人数：影响标题展示（总预算按人数×人均×天数）
    - is_sm_city 小众城市：高端档tips降级，不虚构五星酒店、行政酒廊、黑珍珠等不存在的配置
    """
    # 以3天为基准；天数>3时按比例微调（住宿线性增长，门票/交通次线性）
    k = max(1, days / 3)
    travelers = max(1, int(travelers or 2))
    # 基础人均预算（3天2夜基准，不含城际；按物价系数缩放再取整到10）
    base_raw = {
        "经济型": {"hotel": 210, "ticket": 260, "food": 420, "transport": 180},
        "舒适型": {"hotel": 390, "ticket": 360, "food": 600, "transport": 280},
        "高端型": {"hotel": 1050, "ticket": 500, "food": 1200, "transport": 550},
    }
    base = {}
    for lv, items in base_raw.items():
        base[lv] = {k2: int(round(v * price_factor / 10) * 10) for k2, v in items.items()}
    # 按天数缩放（住宿×k，门票×k^0.8，餐饮×k，交通×k^0.9）
    import math
    d = {}
    for lv, b in base.items():
        d[lv] = {
            "hotel": int(round(b["hotel"] * k)),
            "ticket": int(round(b["ticket"] * (k ** 0.85))),
            "food": int(round(b["food"] * k)),
            "transport": int(round(b["transport"] * (k ** 0.9))),
        }
    # 加城际往返
    for lv in d:
        d[lv]["intercity"] = intercity_train
        d[lv]["total"] = sum(d[lv].values())

    # 区间标签（标题使用，与前端保持一致）；下限保底按物价系数缩放
    min_economy = int(round(1100 * price_factor))
    min_comfort = int(round(1500 * price_factor))
    min_premium = int(round(2600 * price_factor))
    range_tag = {
        "经济型": f"约 {max(min_economy, d['经济型']['total']-100)}-{d['经济型']['total']+200}",
        "舒适型": f"约 {max(min_comfort, d['舒适型']['total']-200)}-{d['舒适型']['total']+300}",
        "高端型": f"约 {max(min_premium, d['高端型']['total']-400)}-{d['高端型']['total']+500}",
    }
    cur = d[budget_level]
    # 明细表
    lv_tag = {"经济型": "👈", "舒适型": "👈", "高端型": "👈"}.get(budget_level, "")
    h_e = (str(d["经济型"]["hotel"]), "👈" if budget_level == "经济型" else "")
    h_s = (str(d["舒适型"]["hotel"]), "👈" if budget_level == "舒适型" else "")
    h_h = (str(d["高端型"]["hotel"]), "👈" if budget_level == "高端型" else "")
    t_e = (str(d["经济型"]["ticket"]), "👈" if budget_level == "经济型" else "")
    t_s = (str(d["舒适型"]["ticket"]), "👈" if budget_level == "舒适型" else "")
    t_h = (str(d["高端型"]["ticket"]), "👈" if budget_level == "高端型" else "")
    f_e = (str(d["经济型"]["food"]), "👈" if budget_level == "经济型" else "")
    f_s = (str(d["舒适型"]["food"]), "👈" if budget_level == "舒适型" else "")
    f_h = (str(d["高端型"]["food"]), "👈" if budget_level == "高端型" else "")
    tr_e = (str(d["经济型"]["transport"]), "👈" if budget_level == "经济型" else "")
    tr_s = (str(d["舒适型"]["transport"]), "👈" if budget_level == "舒适型" else "")
    tr_h = (str(d["高端型"]["transport"]), "👈" if budget_level == "高端型" else "")
    ic_e = ic_s = ic_h = (str(intercity_train), "")
    tote = d["经济型"]["total"]
    tots = d["舒适型"]["total"]
    toth = d["高端型"]["total"]

    if budget_level == "经济型":
        cur_title = f"【三日总预算 · {destination}】当前档位：💰经济型（{range_tag['经济型']} 元/人 · {days}天{days}夜 · {travelers}人同行 · 不含购物）"
    elif budget_level == "高端型":
        cur_title = f"【三日总预算 · {destination}】当前档位：💎高端型（{range_tag['高端型']} 元/人 · {days}天{days}夜 · {travelers}人同行 · 不含购物）"
    else:
        cur_title = f"【三日总预算 · {destination}】当前档位：⚖️舒适型（{range_tag['舒适型']} 元/人 · {days}天{days}夜 · {travelers}人同行 · 不含购物）"

    bg = [cur_title]
    bg.append(f"\n  {days}天{days}夜 · 人均预算明细表（含城际往返）")
    bg.append("  │ 项目            │ 💰经济型(人均) │ ⚖️舒适型(人均)  │ 💎高端型(人均)  │ 说明                                   │")
    bg.append("  ├─────────────────┼──────────────┼───────────────┼───────────────┼────────────────────────────────────────┤")
    bg.append(f"  │ 🚄 城际往返      │   {ic_e[0]:>5} {ic_e[1]:<7}│   {ic_s[0]:>5} {ic_s[1]:<7}│   {ic_h[0]:>5} {ic_h[1]:<7}│ 二等座/高铁实际票价，下同             │")
    bg.append(f"  │ 🏨 住宿（{days}夜）│   {h_e[0]:>5} {h_e[1]:<7}│   {h_s[0]:>5} {h_s[1]:<7}│   {h_h[0]:>5} {h_h[1]:<7}│ {travelers}人同行按房费平摊              │")
    bg.append(f"  │ 🎫 景点门票      │   {t_e[0]:>5} {t_e[1]:<7}│   {t_s[0]:>5} {t_s[1]:<7}│   {t_h[0]:>5} {t_h[1]:<7}│ 经济型A线/舒适型+游船/高端型B线+包车   │")
    bg.append(f"  │ 🍜 吃喝（{days}天） │   {f_e[0]:>5} {f_e[1]:<7}│   {f_s[0]:>5} {f_s[1]:<7}│   {f_h[0]:>5} {f_h[1]:<7}│ 经济型简餐+老店 vs 高端型品质正餐      │")
    bg.append(f"  │ 🚗 市内交通      │   {tr_e[0]:>5} {tr_e[1]:<7}│   {tr_s[0]:>5} {tr_s[1]:<7}│   {tr_h[0]:>5} {tr_h[1]:<7}│ 经济型公交拼车 vs 高端型全程包车      │")
    bg.append("  ├─────────────────┼──────────────┼───────────────┼───────────────┼────────────────────────────────────────┤")
    if budget_level == "经济型":
        bg.append(f"  │ 🎯 **总计**       │ **{tote:>5} 👈 当前** │   {tots:>5}        │  {toth:>5}        │ {days}天{days}夜·不含购物·经济型最省             │")
    elif budget_level == "高端型":
        bg.append(f"  │ 🎯 **总计**       │  {tote:>5}        │   {tots:>5}        │ **{toth:>5} 👈 当前** │ {days}天{days}夜·不含购物·高端型吃住体验拉满     │")
    else:
        bg.append(f"  │ 🎯 **总计**       │  {tote:>5}        │ **{tots:>5} 👈 当前** │  {toth:>5}        │ {days}天{days}夜·不含购物·舒适型平衡体验与价格   │")
    # 档位专属建议（删除具体连锁品牌名，只描述档次类型）
    bg.append("\n💡 省钱/体验小贴士：")
    if budget_level == "经济型":
        bg.append("  · 🏨 住宿省钱（按当地经济档房价）：选本地经济连锁型酒店/民宿，用官方 APP 会员价 + 连住优惠再省 8%-15%；选「可免费取消」房价，行程变动无损退。")
        bg.append("  · 🎫 门票省钱（省 50-200 元/人）：学生证/军官证/老年证/残疾证几乎所有景区半价；提前 1 天在官方公众号/小程序买常比第三方平台便宜 5-10 元/张；景区二消（游船/电瓶车/索道）体力允许的话徒步代替，能省 100-200 元/人。")
        bg.append("  · 🍜 吃喝省钱（按当地经济档餐饮）：正餐选「远离景区 500 米以上 + 老城区居民楼底商」的家常菜馆/夫妻店，早餐吃本地老店粉/面/包子；随身带 1 瓶水 + 小面包/能量棒，景区矿泉水别买，能省 30-50 元/天。")
        bg.append("  · 🚗 交通省钱：用特惠快车 + 拼车顺路单，比普通快车便宜 15%-25%；近郊景区找青旅/民宿前台凑人拼车，人均比单独打车省 50%-60%；手机提前开通「XX 乘车码」，公交地铁直接扫码。")
        bg.append("  · 🛒 购物省钱：特产别在景区/火车站买，贵 30%-100%；用本地超市买，价格腰斩还保真；纪念品买小不买大。")
        bg.append("  · ⚠️ 避坑省钱：出租车/三轮车司机主动推荐的餐厅/酒店/景点一律礼貌拒绝；菜单没明码标价的大排档先问价格再下单；景区门口 10 米内的「百年老字号」「网红同款」别吃，往外走 500 米价格腰斩。")
    elif budget_level == "高端型":
        if is_sm_city:
            # 小众城市：如实描述当地能达到的最高档次，不虚构五星/行政酒廊/专业车队
            bg.append("  · 🏨 住宿品质升级：提前 7-14 天锁本地高端精品民宿/品质较好的酒店，旺季临时订可能涨 50%+；订含早餐的套餐更省心；入住时主动问前台能否免费升级，淡季大概率成功升更好房型或景观位。")
            bg.append("  · 🎫 门票体验升级（值回票价）：买套票/通票 + 深度游（含游船/电瓶车/私藏景观点）；高端型体验核心在深度不在数量，少而精的深度游比多打卡 3 个网红点体验感强 10 倍；景区找官方持证向导讲解，比自己瞎看懂 10 倍。")
            bg.append("  · 🍜 餐饮体验升级（不踩雷 + 增值惊喜）：优先选本地口碑最好的特色餐厅，提前 1-2 天电话预约；纪念日/生日提前跟餐厅说，大概率免费送甜品/果盘 + 升级景观靠窗位（如有）；有过敏/忌口/清真/素食的提前备注，本地口碑餐厅都会专门准备。")
            bg.append("  · 🚗 交通体验升级（3 天省 6-9 小时时间）：全程本地舒适型商务车/5座SUV + 固定老司机（评分 4.8+），省下来的时间和体力才是高端档最值钱的东西；司机提前 10 分钟酒店门口等候，车上备矿泉水/湿纸巾/充电宝/雨伞；司机熟悉路线 + 代购优惠票 + 推荐本地人吃饭的地方。")
            bg.append("  · 💎 专属增值服务（体验感拉满）：连续住 2 晚以上，打电话问前台能不能延退到 14:00 退房；酒店免费项目优先用（下午茶/大堂吧/健身如提供）；行李寄存在酒店前台，最后一天玩完再取，不用拖着行李到处跑。")
            bg.append("  · 👔 细节体验优化：订房选「高层 + 远离电梯口」的房间，安静 + 视野好 + 不吵；酒店免费洗衣服务（如有）当天送去第二天取回。")
        else:
            bg.append("  · 🏨 住宿品质升级：提前 7-14 天锁高端精品民宿/五星档次酒店，旺季临时订可能涨 50%+；订含早餐 + 行政酒廊的套餐，早餐值回票价 + 下午茶 + 欢乐时光能直接省两顿饭钱；入住时主动问前台能否免费升级，淡季 30%+ 概率成功升套房/景观房。")
            bg.append("  · 🎫 门票体验升级（值回票价 200-500 元）：买套票/通票 + B 线深度游（含游船/电瓶车/私藏景观点 + VIP 通道免排队）；高端型体验核心在深度不在数量，少而精的深度游比多打卡 3 个网红点体验感强 10 倍；景区找官方持证向导讲解，比自己瞎看懂 10 倍。")
            bg.append("  · 🍜 餐饮体验升级（不踩雷 + 增值惊喜）：优先选高品质口碑好的本地特色餐厅，提前 2-3 天预约；纪念日/生日提前跟餐厅说，大概率免费送甜品/果盘 + 升级景观靠窗位；有过敏/忌口/清真/素食的提前备注，高端餐厅都会专门准备菜单。")
            bg.append("  · 🚗 交通体验升级（3 天省 6-9 小时时间）：全程 7 座商务车 + 固定老司机（评分 4.9+，接单 3000+），省下来的时间和体力才是高端档最值钱的东西；司机提前 10 分钟酒店门口等候，车上有免费矿泉水/湿纸巾/充电宝/雨伞/晕车药；司机熟悉路线 + 代购优惠票 + 推荐本地人吃饭的地方，比导航靠谱 10 倍。")
            bg.append("  · 💎 专属增值服务（体验感拉满）：连续住 2 晚以上，打电话问前台能不能延退到 14:00 退房；酒店 SPA/游泳/健身住客免费或半价，带泳衣/运动装备；行李寄存在酒店前台，最后一天玩完再取，不用拖着行李到处跑。")
            bg.append("  · 👔 细节体验优化：订房选「高层 + 远离电梯口」的房间，安静 + 视野好 + 不吵；有行政酒廊的优先选，下午茶 + 欢乐时光两顿饭都能解决；洗衣服用酒店免费洗衣服务，当天送去第二天取回，不用手洗穿湿衣服。")
    else:  # 舒适型
        bg.append("  · 🏨 住宿平衡体验与价格：选本地中档连锁/精品酒店（市中心或第 1 天核心景区旁）；连住 3 晚用官方 APP 会员价省 30-50 元；选「可免费取消 + 含早餐」的房价，早餐值回票价；老楼新店优先选新店，房间设施新 + 装修味道小。")
        bg.append("  · 🎫 门票合理搭配（省 200-300 元/人）：必去景点买正规套票（含游船/电瓶车，单独买贵 30%）；可去可不去的网红打卡点直接跳过，把时间和钱留给真正值得玩的景点；学生证/老年证忘带没关系，用电子证件/照片很多景区也认；周一博物馆/古迹大概率闭馆，查好开放时间别白跑。")
        bg.append("  · 🍜 吃喝均衡搭配：每天 1 顿本地口碑正餐 + 1 顿本地热门排队店 + 1 顿夜市/小吃街的搭配最合理；用点评团购券/午市套餐比单点省 20%-30%；每天留 1 顿正餐给「随机漫步偶遇的本地人排队老店」，惊喜感拉满；热门饭点错峰不用排队 1-2 小时。")
        bg.append("  · 🚗 交通平衡时间与预算：市区打快车（多人分摊没差多少）+ 近郊景区半天包车 4 小时（玩完直接送回市区酒店不用赶末班车）；平台选评分 4.8 以上、接单≥1000 单的老司机，靠谱率最高；存司机电话，返程/第二天直接约，不用临时叫车等 10 分钟。")
        bg.append("  · 🛍️ 购物与避坑（省 100-300 元）：特产去本地大超市买（比景区/火车站便宜 30%-50%），保真又划算；出租车/三轮车司机主动推荐的餐厅/景点礼貌拒绝，大概率有提成；菜单没明码标价的大排档先问价格再下单，避免吃完被宰。")
        bg.append("  · 🧘 细节提升体验：逛夜市别吃太杂太辣，随身带健胃消食片/益生菌，旅行拉肚子最影响心情；连续住 2 晚以上打电话问前台能不能延退到 14:00（大概率同意）；夏天高温中午强制回酒店午休 30 分钟，避免下午玩不动整体崩盘。")
    return "\n".join(bg)


def _generate_daily_routes_universal(destination: str, days: int, attractions: list, preferences: list, budget_level: str,
                                     price_factor: float = 1.0, tier_tag: str = "二线",
                                     is_sm_city: bool = False, travelers: int = 2) -> str:
    """通用每日行程生成：所有城市同结构（上午→中午→下午→晚上；就近美食；顺路；偏好+预算双生效）
    - 物价系数 price_factor：一线1.5 / 新一线1.25 / 二线1.0 / 三四线0.65~0.75 / 小众0.6
    - 小众城市 is_sm_city：高端档如实描述当地最高档次，不虚构五星酒店、高端私厨、9-14座商务车车队
    """
    prefs = set(preferences or [])
    days = max(1, min(int(days or 3), 30))
    pf = max(0.5, price_factor or 1.0)
    travelers = max(1, int(travelers or 2))
    # 价格取整辅助：四舍五入到5元
    def rmb(base):
        return int(round(base * pf / 5) * 5)
    # 景点池保证够用：不足就循环复用（符合项目约束：景点循环复用+补充）
    pool = list(attractions) if attractions else _enrich_attractions_generic(destination, preferences or [], days)
    if not pool:
        pool = _enrich_attractions_generic(destination, preferences or [], days)
    # 按偏好+评分重排后，轮询分配到每天（每天2-3个景点，上午+下午）
    pool = _apply_preference_weight(pool, list(prefs))
    # 交通方式（预算档位决定）：三档差异明显放大；小众城市高端档降级
    traffic_map_base = {
        "经济型": ("公交+地铁（如有）+ 拼车APP凑人", "约30-60分钟（含等车+换乘）"),
        "舒适型": ("网约车快车 / 滴滴高德平台叫车", "约15-30分钟（直达，不用换乘）"),
        "高端型": ("全程 7 座商务车 + 固定老司机（车下等候）", "约10-20分钟（门到门，车随人走）"),
    }
    if is_sm_city and budget_level == "高端型":
        # 小众城市没有专业车队，降级描述：舒适型商务车/5座SUV包车
        traffic_map_base["高端型"] = ("本地舒适型商务车/5座SUV 包车 + 固定老司机（车下等候）", "约10-25分钟（门到门，按当地车况）")
    traffic, tm = traffic_map_base.get(budget_level, traffic_map_base["舒适型"])
    # 预算档位的当日花费（金额+描述三档明显不同）；价格按物价系数动态调整
    if budget_level == "经济型":
        daily_cost = f"当日花费约 {rmb(140)}-{rmb(270)} 元/人（门票按优惠档计算，餐饮选老店+路边摊，交通公交+拼车；随身带水和零食不花冤枉钱）"
    elif budget_level == "高端型":
        if is_sm_city:
            daily_cost = f"当日花费约 {rmb(650)}-{rmb(1100)} 元/人（门票含套票+游船+电瓶车，餐饮本地口碑特色餐+酒店下午茶（如有），交通全程本地舒适包车含司机餐补；省时间省体力）"
        else:
            daily_cost = f"当日花费约 {rmb(780)}-{rmb(1350)} 元/人（门票含套票+游船+VIP通道，餐饮口碑特色餐+酒店下午茶，交通全程包车含司机餐补；省时间省体力）"
    else:
        daily_cost = f"当日花费约 {rmb(290)}-{rmb(500)} 元/人（门票按正常套票档，餐饮1顿本地口碑正餐+1顿热门店，交通市区快车+近郊半天包车；平衡体验与价格）"

    lines = [f"【📅 每日行程安排 · {destination}】共 {days} 天 · {tier_tag}城市物价系数×{pf:.2f} · 出行{travelers}人 · 预算档位：{budget_level} · 顺路不绕路版"]
    used_idx = 0
    for d in range(1, days + 1):
        lines.append(f"\n🗓️ 第{d}天：")
        # 每天上午1个 + 下午1-2个 = 共2-3个
        day_count = 3 if (d == 1 or days <= 2) else 2
        day_ats = []
        for _ in range(day_count):
            day_ats.append(pool[used_idx % len(pool)])
            used_idx += 1
        # 上午
        if day_ats:
            a = day_ats[0]
            lines.append(f"  ☀️ **上午游玩（08:30-12:00）**：")
            if budget_level == "经济型":
                lines.append(f"    · 酒店出发：公交/地铁，提前 10 分钟用高德查实时班次避免白等；手机扫乘车码不用排队买票找零")
            elif budget_level == "高端型":
                if is_sm_city:
                    lines.append(f"    · 酒店出发：司机提前 10 分钟在酒店大堂/门口等候，行李放后备箱不用自己拎；景区走正规通道，按现场秩序排队")
                else:
                    lines.append(f"    · 酒店出发：司机提前 10 分钟在酒店大堂/门口等候，行李放后备箱不用自己拎；景区走 VIP 通道免排队")
            else:
                lines.append(f"    · 酒店出发：平台叫快车，{travelers}人分摊差价不大；赶时间直接打，不赶时间可以等顺路拼车单")
            lines.append(f"    · 酒店 → 「{a['name']}」，{traffic}，{tm}（距离约 3-8 km）")
            lines.append(f"    · 建议游玩 {a.get('duration','2-3小时')} · 门票：{a.get('ticket','以现场为准')}")
            if budget_level == "经济型":
                lines.append(f"    · ⚠️ 省钱：学生证/老年证/军官证所有景区半价；公众号提前 1 天预约比现场便宜 5-10 元")
            elif budget_level == "高端型":
                lines.append(f"    · 💎 体验：司机帮忙代购景区优惠票/套票（比平台便宜），含电瓶车/游船不用自己排队买票")
            lines.append(f"    · 💡 {a.get('desc','到此一游')}")
        # 中午（预算档位差异化）
        lines.append(f"  🍱 **中午就近吃饭 + 休整（12:00-13:30）**：")
        if day_ats:
            _spot_name = day_ats[0].get('name', destination)
        else:
            _spot_name = destination
        if budget_level == "经济型":
            lines.append(f"    · 地点：{_spot_name} 出口步行 5-10 分钟的「本地家常菜馆 / 夫妻店 / 连锁快餐」，绝不选景区门口 10 米内的高价店")
            lines.append(f"    · 推荐：{destination}本地家常菜 + 时蔬 + 例汤，人均 {rmb(18)}-{rmb(48)} 元；或直接一碗本地粉/面 {rmb(8)}-{rmb(22)} 元搞定")
            lines.append(f"    · ⚠️ 省钱：随身带 1 瓶水 + 小面包/能量棒，景区矿泉水别买；出发前在附近超市批量采购更划算")
            lines.append(f"    · 休整：吃完回酒店睡 30-60 分钟午觉（夏天高温别硬逛，中暑看医生买药的钱够多吃 3 顿正餐）")
        elif budget_level == "高端型":
            if is_sm_city:
                # 小众城市降级：没有黑珍珠/五星酒店行政酒廊，如实描述本地最高档
                lines.append(f"    · 地点：{destination}本地口碑最好的特色餐厅 或 景区内景观位餐厅（提前 1-2 天电话预约拿号）")
                lines.append(f"    · 推荐：{destination}本地特色食材（本地鲜货/山货/时蔬）+ 店家拿手菜 + 特色饮品，人均 {rmb(95)}-{rmb(220)} 元")
                lines.append(f"    · 💎 增值：预约备注「纪念日/生日」大概率免费送甜品/果盘 + 免费升级景观靠窗位（如有）")
                lines.append(f"    · 休整：吃完返回酒店 1 小时午休 / 室内茶饮 / 免费项目放松（按酒店实际提供）")
            else:
                lines.append(f"    · 地点：{destination}本地高品质口碑特色餐厅 或 景区内高端景观位餐厅（提前 2-3 天预约拿号）")
                lines.append(f"    · 推荐：{destination}本地特色食材（河鲜/山珍/有机）+ 主厨推荐菜 + 特色饮品，人均 {rmb(115)}-{rmb(260)} 元")
                lines.append(f"    · 💎 增值：预约备注「纪念日/生日」大概率免费送甜品/果盘 + 免费升级景观靠窗位")
                lines.append(f"    · 休整：吃完返回酒店 1 小时 SPA / 室内游泳 / 行政酒廊下午茶（订含行政酒廊的房这顿直接省 {rmb(180)} 元）")
        else:  # 舒适型
            lines.append(f"    · 地点：{_spot_name} 出口步行 5-15 分钟的「本地老字号正餐店 / 热门排队店」，错峰 13:30 后去不用排 1-2 小时")
            lines.append(f"    · 推荐：{destination}本地特色菜 + 时蔬 + 例汤，人均 {rmb(42)}-{rmb(85)} 元；或本地口碑菜 1 道 + 本地小吃搭配最合理")
            lines.append(f"    · ⚖️ 技巧：用点评团购券/午市套餐比单点省 20%-30%，下单前看清楚周末节假日是否可用 + 是否需要预约")
            lines.append(f"    · 休整：吃完回酒店休息 30-60 分钟（夏天高温最忌中午硬逛，最容易中暑影响后面 2 天行程）")
        # 下午
        if len(day_ats) >= 2:
            a2 = day_ats[1]
            lines.append(f"  🌤️ **下午游玩（14:00-17:30）**：")
            lines.append(f"    · 酒店 → 「{a2['name']}」，{traffic}，{tm}（距离约 3-8 km）")
            lines.append(f"    · 建议游玩 {a2.get('duration','2-3小时')} · 门票：{a2.get('ticket','以现场为准')}")
            if budget_level == "高端型":
                lines.append(f"    · 💎 体验：车上有免费矿泉水 + 湿纸巾 + 充电宝，玩累了直接上车空调休息；景区电瓶车预留前排不用抢座")
            elif budget_level == "经济型":
                lines.append(f"    · ⚠️ 省钱：下午景区人少可能有特价票，现场售票处问一下；能用年卡/旅游一卡通的优先用")
            lines.append(f"    · 💡 {a2.get('desc','到此一游')}")
        if len(day_ats) >= 3:
            a3 = day_ats[2]
            lines.append(f"  🌇 **傍晚加场（17:40-19:00）**：适合 Day1 精力充沛时")
            lines.append(f"    · 「{a3['name']}」，{traffic}，{tm}（从下午景点顺路前往，不用绕路折返）")
            lines.append(f"    · 建议游玩 {a3.get('duration','1.5小时')} · 门票：{a3.get('ticket','以现场为准')}")
        # 晚上（预算档位差异化）
        lines.append(f"  🌙 **晚上夜市 / 休整（19:30-22:00）**：")
        if budget_level == "经济型":
            _night_food = f"{destination}老城区老牌夜市街 / 大学城周边小吃街（学生价，品类多不踩雷），点评搜「烧烤/夜宵」看评论≥200 评分≥4.3 的推车/小店"
            lines.append(f"    · 吃饭：{_night_food}，人均 {rmb(25)}-{rmb(65)} 元（烧烤/炒粉/卤味+啤酒，路边摊比门店便宜 30%）")
            lines.append(f"    · 饭后：{destination}江/湖滨免费夜景 / 步行街散步 / 超市买点第二天的水和面包零食，回酒店早点休息")
            lines.append(f"    · ⚠️ 省钱：晚上不用打车，公交末班车 22:30 前能赶上；或者打车拼车顺路单比快车省 40%")
        elif budget_level == "高端型":
            if is_sm_city:
                _night_food = f"{destination}江/湖滨景观位夜宵店 / 本地高品质夜宵店（按当地实际情况选择）"
                lines.append(f"    · 吃饭：{_night_food}，人均 {rmb(120)}-{rmb(280)} 元（景观烧烤/本地特色/精酿啤酒，空调足环境好）")
                lines.append(f"    · 饭后：{destination}夜游 / 顶楼清吧小酌 / 酒店足疗放松（玩了一天彻底解乏），司机 22:00 准时来接回酒店")
                lines.append(f"    · 💎 增值：酒店免费洗衣服务当天送去第二天取回（如有），不用自己手洗第二天还穿湿衣服")
            else:
                _night_food = f"{destination}江/湖滨景观位夜宵店 / 酒店行政酒廊欢乐时光（含在房费）或 高端商场品质美食街"
                lines.append(f"    · 吃饭：{_night_food}，人均 {rmb(145)}-{rmb(320)} 元（景观烧烤/海鲜/精酿啤酒，空调足环境好，不用在路边流汗）")
                lines.append(f"    · 饭后：{destination}夜游游船 / 顶楼清吧小酌 / 酒店 SPA 足疗放松（玩了一天彻底解乏），司机 22:00 准时来接回酒店")
                lines.append(f"    · 💎 增值：酒店免费洗衣服务当天送去第二天取回，不用自己手洗第二天还穿湿衣服")
        else:  # 舒适型
            _night_food = f"{destination}本地人最多的老牌夜市街 / 商场 B1 层美食区（营业到 22 点）或 酒店周边连锁夜宵"
            if "美食" in prefs or "城市逛吃" in prefs:
                _night_food = f"{destination}本地人最多的老牌夜市街（美食党优先：点评搜「烧烤/夜宵」看评论≥500 评分≥4.3 的店）"
            if "亲子游玩" in prefs:
                _night_food = f"{destination}大型商场餐饮层（有儿童游乐区，空调足环境好）或 酒店周边连锁夜宵"
            lines.append(f"    · 吃饭：{_night_food}，人均 {rmb(45)}-{rmb(110)} 元（搭配合理：小吃+正餐+饮品都能尝到）")
            lines.append(f"    · 饭后：{destination}江/湖滨夜景 / 步行街散步 / 超市买点第二天的水和零食，回酒店早点休息")
            lines.append(f"    · ⚖️ 建议：逛夜市别吃太杂太辣，随身带健胃消食片/益生菌，旅行拉肚子最影响心情")
        lines.append(f"  💸 {daily_cost}（含门票+餐饮+市内交通，不含城际往返+住宿）")
    # 预算档位+偏好策略（3档差异显著扩大，每档4条完整建议）
    lines.append(f"\n🎯 当前预算档位：{budget_level} · 交通+餐饮策略提示")
    if budget_level == "经济型":
        lines.append(f"  · 🚗 交通策略（每天省 100-200 元）：市区公交/地铁（{rmb(1)}-{rmb(6)} 元/次）为主，提前用高德查实时班次；赶时间就拼车APP顺路单；近郊景区找青旅/民宿前台凑人拼车，人均比单独打车省 50%-60%；手机提前开通乘车码不用排队买票找零。")
        lines.append(f"  · 🍜 餐饮策略（每天省 80-150 元）：早餐本地老店粉/面（{rmb(8)} 元内吃饱），正餐老城区居民楼底商家常菜馆（人均 {rmb(25)}-{rmb(55)} 元），晚上夜市/小吃街（人均 {rmb(25)}-{rmb(65)} 元）；随身带水 + 面包/能量棒避免景区高价消费；出发前在附近超市批量采购能省 30-50 元/天。")
        lines.append("  · ⏱️ 时间权衡：公交+拼车比全程打车每天多花 1-1.5 小时（主要是等车+换乘），但 3 天能省 300-600 元/人，用时间换金钱按个人接受度权衡；周末/节假日公交特别挤，赶时间直接拼车别硬等。")
        lines.append("  · 🎫 门票省钱：所有证件（学生/老年/军官/残疾）都带上；提前 1 天官方公众号预约常比平台便宜 5-10 元；景区二消（游船/电瓶车/索道）看预算，非必须可以徒步代替（能省 100-200 元）。")
    elif budget_level == "高端型":
        if is_sm_city:
            lines.append(f"  · 🚗 交通策略（每天省 2-3 小时时间）：全程本地舒适型商务车/5座SUV + 固定老司机（评分 4.8+），司机提前 10 分钟酒店门口等候，车上有免费矿泉水/湿纸巾/充电宝/雨伞；行李不用自己拎，司机全程帮忙放后备箱；日包车含油费过路费司机餐补，不用额外付钱。")
            lines.append(f"  · 🍜 餐饮策略（体验拉满不踩雷）：正餐选本地口碑最好的特色餐厅，提前 1-2 天电话预约；预约备注「纪念日/生日」获赠甜品/果盘/景观位（如有）；下午茶按酒店实际提供（如大堂吧/附近茶饮）；全程不排队、不踩雷、不被宰。")
            lines.append("  · ⏱️ 时间价值：每天比经济型多出 2-3 小时自由时间，用来多睡 1 小时（不用赶车）+ 午休/茶饮 1 小时+景区深度游 1 小时，高端档核心价值是「省时间省体力」而非「多玩几个景点」；景点不在多而在精，深度体验比打卡更值。")
            lines.append(f"  · 💎 增值服务：订房选含早餐的套餐，早餐值回票价；景区买含游船/电瓶车的深度游通票，体验感更好；入住时主动问前台能否免费升级（淡季大概率成功升更好房型）。")
        else:
            lines.append(f"  · 🚗 交通策略（每天省 2-3 小时时间）：全程 7 座商务车 + 固定老司机（评分 4.9+，接单 3000+），司机提前 10 分钟酒店门口等候，车上有免费矿泉水/湿纸巾/充电宝/雨伞，景区走 VIP 通道不用排队；行李不用自己拎，司机全程帮忙放后备箱；日包车含油费过路费司机餐补，不用额外付钱。")
            lines.append(f"  · 🍜 餐饮策略（体验拉满不踩雷）：正餐选高品质口碑特色餐厅提前 2-3 天预约，预约备注「纪念日/生日」获赠甜品/果盘/景观位；下午茶用酒店行政酒廊（含在房费），夜宵选江/湖滨景观位；所有餐厅司机帮忙拿号+代购优惠套餐；全程不排队、不踩雷、不被宰。")
            lines.append("  · ⏱️ 时间价值：每天比经济型多出 2-3 小时自由时间，用来多睡 1 小时（不用赶车）+ 下午茶/Spa 1 小时+景区深度游 1 小时，高端档核心价值是「省时间省体力」而非「多玩几个景点」；景点不在多而在精，深度体验比打卡更值。")
            lines.append(f"  · 💎 增值服务：订房选含早餐+行政酒廊的套餐，早餐值回票价+下午茶+欢乐时光能省两顿饭钱（约 {rmb(280)}-{rmb(550)} 元）；景区买含游船/电瓶车/VIP 通道的 B 线深度游通票，体验感和 A 线完全不同；入住时主动问前台能否免费升级，淡季 30%+ 概率成功升套房/景观房。")
    else:  # 舒适型
        lines.append(f"  · 🚗 交通策略（平衡时间与预算）：市区内网约车快车（{travelers}人分摊和公交差价 {rmb(12)}-{rmb(35)} 元/次，但省 30-50 分钟）；近郊景区半天包车 4 小时约 {rmb(180)}-{rmb(340)} 元（玩完直接送回市区酒店，不用赶公交末班车）；平台选评分 4.8 以上、接单≥1000 单的老司机，靠谱率最高。")
        lines.append(f"  · 🍜 餐饮策略（平衡体验与价格）：每天 1 顿本地口碑正餐 + 1 顿本地热门排队店 + 1 顿夜市/小吃街的搭配最合理；用点评团购券/午市套餐比单点省 20%-30%；每天留 1 顿正餐给「随机漫步偶遇的本地人排队老店」，惊喜感拉满。")
        lines.append("  · ⏱️ 时间权衡：每天比经济型多 30-60 分钟游玩时间（主要省在等车/换乘），压力适中不会累；景点安排不会太赶，有时间停下来拍照+发呆；每天中午强制回酒店午休 30 分钟，避免下午玩不动影响整体体验。")
        lines.append("  · 🎫 门票建议：必去景点买正规套票（含游船/电瓶车，单独买贵 30%）；可去可不去的网红打卡点直接跳过（能省 200-300 元），把时间留给值得深度玩的景点；学生证/老年证忘带没关系，用电子证件/照片很多景区也认（省一半门票钱）。")
    # 偏好标签专属行程建议
    if prefs:
        lines.append("\n🎯 本次出行偏好专属行程提醒：")
        if "美食" in prefs or "城市逛吃" in prefs:
            lines.append("  🍜【美食/逛吃党】：每天中午留 1 个正餐专门打卡本地老字号；晚上留出 2 小时逛夜市+小吃街；随身带健胃消食片。")
        if "自然风光" in prefs or "户外运动" in prefs:
            lines.append("  🥾【自然/户外党】：上午走体力消耗大的景区（温度低+人少），下午走平路/室内景区避暑；带防滑运动鞋+护膝+电解质水。")
        if "人文古迹" in prefs or "历史文博" in prefs:
            lines.append("  🏛️【文化/文博党】：博物馆/古迹 周一大概率闭馆，出行前务必查开放时间；租电子讲解器比自己瞎看懂10倍。")
        if "网红打卡" in prefs:
            lines.append("  📸【网红党】：热门打卡点 早上8-9点去没人，能拍空镜；穿浅色衣服+鲜艳配饰出片率高；下午3-4点光线最好。")
        if "休闲度假" in prefs:
            lines.append("  🌿【度假党】：每天只安排 2 个景点，其余时间留空；下午回酒店游泳/SPA/下午茶；不要强行打卡把自己搞累。")
        if "亲子游玩" in prefs:
            lines.append("  👶【亲子党】：上午户外+下午室内（避暑+防累）；随身带备用衣服+退热贴+湿纸巾+小零食；儿童乐园安排在最后一天，玩累了直接回家。")
        if "小众秘境" in prefs:
            lines.append(f"  🔮【小众党】：每天最后1个景点留空，用小红书/抖音搜「{destination}小众」按距离排序随机挑1个，惊喜感拉满。")
    return "\n".join(lines)