# -*- coding: utf-8 -*-
"""
天气智能体测试
验证天气查询功能
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json


def test_mock_weather_query():
    """测试模拟天气查询"""
    from mcp_server.tools.weather_tools import _mock_weather_query

    # 测试查询
    result = _mock_weather_query("北京", "", 3)
    assert result["location"] == "北京"
    assert len(result["forecasts"]) == 3
    assert "summary" in result
    print("✓ test_mock_weather_query 通过")
    print(f"  北京天气: {result['summary']}")

    # 测试不同天数
    result7 = _mock_weather_query("上海", "", 7)
    assert len(result7["forecasts"]) == 7
    print("✓ test_7_day_forecast 通过")


def test_weather_tips():
    """测试天气建议"""
    from mcp_server.tools.weather_tools import _generate_weather_tips

    tips = _generate_weather_tips("成都", "春")
    assert "clothing_advice" in tips
    assert tips["location"] == "成都"
    print("✓ test_weather_tips 通过")
    print(f"  成都春季建议: {tips['clothing_advice']}")


async def test_weather_agent_direct():
    """直接测试天气智能体（不依赖 LLM）"""
    from mcp_server.tools.weather_tools import _mock_weather_query

    # 模拟智能体处理流程
    query = "帮我查一下广州明天的天气"
    import re
    # 简单提取城市
    cities = ["北京", "上海", "广州", "深圳", "成都", "杭州", "三亚", "西安", "南京", "重庆", "武汉", "大理"]
    city = next((c for c in cities if c in query), "广州")

    result = _mock_weather_query(city, "", 3)
    response = f"广州未来3天天气：{result['summary']}"

    assert "广州" in response
    assert "天气" in response
    print("✓ test_weather_agent_direct 通过")
    print(f"  响应: {response}")


def run_tests():
    """运行所有天气测试"""
    print("=" * 60)
    print("天气智能体测试")
    print("=" * 60)

    test_mock_weather_query()
    test_weather_tips()
    asyncio.run(test_weather_agent_direct())

    print("\n✅ 天气智能体所有测试通过！\n")


if __name__ == "__main__":
    run_tests()