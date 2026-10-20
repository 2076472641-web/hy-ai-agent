# -*- coding: utf-8 -*-
"""
景点推荐智能体测试
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_search_attractions():
    """测试景点搜索"""
    from mcp_server.tools.attraction_tools import _search_attractions_mock

    # 搜索成都景点
    result = _search_attractions_mock("成都", "", 5)
    assert result["destination"] == "成都"
    assert result["total"] > 0
    assert len(result["attractions"]) <= 5
    print("✓ test_search_attractions 通过")
    print(f"  成都景点数: {result['total']}")

    # 按分类搜索
    result_filtered = _search_attractions_mock("成都", "自然风光", 10)
    for a in result_filtered["attractions"]:
        assert a["category"] == "自然风光"
    print("✓ test_filtered_search 通过")


def test_generate_itinerary():
    """测试行程生成"""
    from mcp_server.tools.attraction_tools import _generate_itinerary

    itinerary = _generate_itinerary("成都", 3, "自然风光,历史文化", "标准")
    assert itinerary["destination"] == "成都"
    assert itinerary["days"] == 3
    assert len(itinerary["daily_plans"]) == 3
    print("✓ test_generate_itinerary 通过")
    print(f"  行程天数: {itinerary['days']}")
    print(f"  预算说明: {itinerary['budget_notes']}")


def test_attraction_detail():
    """测试景点详情"""
    from mcp_server.tools.attraction_tools import _get_attraction_detail_mock

    detail = _get_attraction_detail_mock("大熊猫繁育研究基地")
    assert "name" in detail
    assert detail["name"] == "大熊猫繁育研究基地"
    assert "tips" in detail
    print("✓ test_attraction_detail 通过")
    print(f"  景点: {detail['name']}, 评分: {detail['rating']}")

    # 测试不存在的景点
    not_found = _get_attraction_detail_mock("不存在的景点")
    assert "error" in not_found
    print("✓ test_attraction_not_found 通过")


def test_empty_city():
    """测试空城市"""
    from mcp_server.tools.attraction_tools import _search_attractions_mock

    result = _search_attractions_mock("亚特兰蒂斯", "", 10)
    assert result["total"] == 0
    print("✓ test_empty_city 通过")


def run_tests():
    print("=" * 60)
    print("景点推荐智能体测试")
    print("=" * 60)

    test_search_attractions()
    test_generate_itinerary()
    test_attraction_detail()
    test_empty_city()

    print("\n✅ 景点推荐智能体所有测试通过！\n")


if __name__ == "__main__":
    run_tests()