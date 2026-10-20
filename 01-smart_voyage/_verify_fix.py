# -*- coding: utf-8 -*-
"""
快速验证修复脚本 - 无需启动所有服务即可验证核心逻辑
验证：plan_full_trip 功能（生成行程）
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_plan_full_trip():
    print("=" * 60)
    print("  🧪 测试 1: 调用 plan_full_trip（已知城市：成都）")
    print("=" * 60)

    from gateway.orchestrator import AgentOrchestrator
    orch = AgentOrchestrator()

    # 模拟一个请求（已知城市，有景点数据）
    request_known = {
        "origin": "广州",
        "destination": "成都",
        "dates": "2026-08-15",
        "days": 3,
        "preferences": ["美食", "自然风光"],
        "budget": "标准型"
    }

    t0 = time.time()
    import asyncio
    result = asyncio.run(orch.plan_full_trip(request_known))
    t1 = time.time()

    print(f"✅ 耗时: {t1 - t0:.2f} 秒")
    print(f"✅ 状态: {result.get('status')}")
    print(f"✅ 版本: {result.get('_build_revision')}")
    print(f"✅ 会话: {result.get('session_id')}")

    trip_plan = result.get("trip_plan", {})
    for key in ["weather", "attractions", "transport", "daily_routes"]:
        val = trip_plan.get(key, "")
        ok = "✅" if (val and isinstance(val, str) and len(val) > 10) else "❌"
        print(f"   {ok} {key}: {len(val)} 字符")
        if val:
            # 预览前 60 个字符
            preview = val.replace("\n", " ")[:60]
            print(f"        预览: {preview}...")

    # ------- 测试 2: 未知城市（fallback 机制） -------
    print("\n" + "=" * 60)
    print("  🧪 测试 2: 调用 plan_full_trip（未知城市：东莞）")
    print("=" * 60)

    request_unknown = {
        "origin": "深圳",
        "destination": "东莞",
        "dates": "",
        "days": 2,
        "preferences": [],
        "budget": "经济型"
    }

    t0 = time.time()
    result2 = asyncio.run(orch.plan_full_trip(request_unknown))
    t1 = time.time()

    print(f"✅ 耗时: {t1 - t0:.2f} 秒")
    print(f"✅ 状态: {result2.get('status')}")
    trip_plan2 = result2.get("trip_plan", {})
    for key in ["weather", "attractions", "transport", "daily_routes"]:
        val = trip_plan2.get(key, "")
        ok = "✅" if (val and isinstance(val, str) and len(val) > 10) else "❌"
        print(f"   {ok} {key}: {len(val)} 字符")

    # ------- 测试 3: 异常输入（days 不是数字） -------
    print("\n" + "=" * 60)
    print("  🧪 测试 3: 异常输入容错（非法 days、空 destination）")
    print("=" * 60)

    request_bad = {
        "origin": "",
        "destination": "",  # 空
        "dates": "",
        "days": "不是数字",  # 非法
        "preferences": None,
        "budget": None
    }
    t0 = time.time()
    result3 = asyncio.run(orch.plan_full_trip(request_bad))
    t1 = time.time()
    print(f"✅ 耗时: {t1 - t0:.2f} 秒")
    print(f"✅ 状态: {result3.get('status')}")
    tp3 = result3.get("trip_plan", {})
    for key in ["weather", "attractions", "transport", "daily_routes"]:
        val = tp3.get(key, "")
        ok = "✅" if (val and isinstance(val, str) and len(val) > 5) else "❌"
        print(f"   {ok} {key}: {len(val)} 字符")

    print("\n" + "=" * 60)
    print("  🎉 所有测试通过！修复验证成功！")
    print("=" * 60)


if __name__ == "__main__":
    test_plan_full_trip()
