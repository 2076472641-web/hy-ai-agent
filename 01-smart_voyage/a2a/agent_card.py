# -*- coding: utf-8 -*-
"""
A2A Agent Card - 智能体身份与能力声明
每个 Agent 通过 Agent Card 对外声明自身能力
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class AgentSkill(BaseModel):
    """智能体技能描述"""
    id: str
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    examples: list[str] = Field(default_factory=list)


class AgentCard(BaseModel):
    """
    A2A Agent Card - 智能体名片
    参考 Google A2A 规范中 Agent Card 的定义
    """
    name: str
    description: str
    version: str = "1.0.0"
    endpoint: str = ""
    skills: list[AgentSkill] = Field(default_factory=list)
    capabilities: dict[str, Any] = Field(default_factory=lambda: {
        "streaming": True,
        "push_notifications": False,
        "state_transition_history": True,
    })
    authentication: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def create_weather_agent(cls, endpoint: str) -> "AgentCard":
        return cls(
            name="weather_agent",
            description="天气查询智能体 - 提供目的地天气预报、温度、湿度、空气质量等信息",
            endpoint=endpoint,
            skills=[
                AgentSkill(
                    id="query_weather",
                    name="查询天气",
                    description="查询指定城市/景点的天气预报，包括温度、天气状况、空气质量、穿衣建议等",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "城市或景点名称"},
                            "date": {"type": "string", "description": "日期，格式 YYYY-MM-DD，可选"},
                            "days": {"type": "integer", "description": "预报天数，默认3天"},
                        },
                        "required": ["location"],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "weather": {"type": "string"},
                            "temperature": {"type": "string"},
                            "suggestion": {"type": "string"},
                        },
                    },
                    examples=["帮我查一下北京明天的天气", "三亚未来3天天气怎么样"],
                ),
            ],
        )

    @classmethod
    def create_attraction_agent(cls, endpoint: str) -> "AgentCard":
        return cls(
            name="attraction_agent",
            description="景点推荐智能体 - 根据用户偏好推荐适合的旅游景点",
            endpoint=endpoint,
            skills=[
                AgentSkill(
                    id="recommend_attractions",
                    name="推荐景点",
                    description="根据目的地、天数、偏好等推荐景点，包含景点简介、门票、开放时间等",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "destination": {"type": "string", "description": "目的地城市"},
                            "days": {"type": "integer", "description": "游玩天数"},
                            "preferences": {"type": "array", "items": {"type": "string"}, "description": "偏好类型，如['自然风光','历史文化']"},
                            "budget": {"type": "string", "description": "预算档位"},
                        },
                        "required": ["destination", "days"],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "attractions": {"type": "array"},
                            "itinerary": {"type": "array"},
                        },
                    },
                    examples=["推荐成都3天游玩景点", "北京有哪些历史文化景点值得去"],
                ),
            ],
        )

    @classmethod
    def create_route_agent(cls, endpoint: str) -> "AgentCard":
        return cls(
            name="route_agent",
            description="出行路线规划智能体 - 规划交通路线、行程安排",
            endpoint=endpoint,
            skills=[
                AgentSkill(
                    id="plan_route",
                    name="规划路线",
                    description="规划从出发地到目的地的交通方案，包括飞机/高铁/汽车等",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "origin": {"type": "string", "description": "出发城市"},
                            "destination": {"type": "string", "description": "目的城市"},
                            "date": {"type": "string", "description": "出发日期"},
                            "preference": {"type": "string", "description": "偏好：最快/最便宜/最舒适"},
                        },
                        "required": ["origin", "destination"],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "routes": {"type": "array"},
                            "recommendation": {"type": "string"},
                        },
                    },
                    examples=["从上海到广州怎么去最方便", "帮我规划北京3天行程路线"],
                ),
                AgentSkill(
                    id="plan_itinerary",
                    name="规划行程",
                    description="根据景点规划每日行程安排，包含交通、时间分配",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "destination": {"type": "string"},
                            "attractions": {"type": "array"},
                            "days": {"type": "integer"},
                        },
                        "required": ["destination", "attractions", "days"],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "daily_plans": {"type": "array"},
                            "transport_tips": {"type": "string"},
                        },
                    },
                    examples=["帮我安排成都5天行程", "规划杭州2天游玩路线"],
                ),
            ],
        )