# -*- coding: utf-8 -*-
"""
路线规划智能体测试
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_generate_transport_plans():
    """测试交通方案生成"""
    from mcp_server.tools.route_tools import _generate_transport_plans

    # 上海 → 杭州
    result = _generate_transport_plans("上海", "杭州", "", "最快")
    assert result["origin"] == "上海"
    assert result["destination"] == "杭州"
    assert len(result["routes"]) > 0
    assert "recommendation" in result
    print("✓ test_generate_transport_plans 通过")
    print(f"  上海→杭州: {len(result['routes'])} 条路线")
    print(f"  推荐: {result['recommendation']}")


def test_transport_preference():
    """测试不同偏好"""
    from mcp_server.tools.route_tools import _generate_transport_plans

    # 最快
    fastest = _generate_transport_plans("北京", "上海", "", "最快")
    # 最便宜
    cheapest = _generate_transport_plans("北京", "上海", "", "最便宜")

    assert len(fastest["routes"]) > 0
    assert len(cheapest["routes"]) > 0
    print("✓ test_transport_preference 通过")


def test_daily_routes():
    """测试每日路线规划"""
    from mcp_server.tools.route_tools import _generate_daily_routes

    attractions = ["大熊猫基地", "宽窄巷子", "都江堰", "锦里古街", "武侯祠"]
    result = _generate_daily_routes("成都", attractions, 2)
    assert result["destination"] == "成都"
    assert result["total_days"] == 2
    assert len(result["daily_plans"]) == 2
    print("✓ test_daily_routes 通过")
    print(f"  成都2天路线: {len(result['daily_plans'])} 天")


def test_transport_tips():
    """测试交通小贴士"""
    from mcp_server.tools.route_tools import _get_transport_tips

    tips = _get_transport_tips("北京")
    assert "metro_tips" in tips
    assert "traffic_tips" in tips
    print("✓ test_transport_tips 通过")
    print(f"  北京地铁: {tips['metro_tips']}")


def test_unknown_route():
    """测试未知路线"""
    from mcp_server.tools.route_tools import _generate_transport_plans

    result = _generate_transport_plans("火星", "月球", "", "最快")
    assert len(result["routes"]) > 0  # 通用方案
    print("✓ test_unknown_route 通过")


def run_tests():
    print("=" * 60)
    print("路线规划智能体测试")
    print("=" * 60)

    test_generate_transport_plans()
    test_transport_preference()
    test_daily_routes()
    test_transport_tips()
    test_unknown_route()

    print("\n✅ 路线规划智能体所有测试通过！\n")


if __name__ == "__main__":
    run_tests()