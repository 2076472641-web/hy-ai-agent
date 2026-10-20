# -*- coding: utf-8 -*-
"""
专项验证：出发地/目的地一致性测试
确保：用户输入的 origin / destination 与 最终生成方案 完全一致
"""
import asyncio
import sys
sys.path.insert(0, ".")

from gateway.orchestrator import AgentOrchestrator


async def main():
    orch = AgentOrchestrator()
    passed = 0
    failed = 0

    # --------------------------------------------------------------
    # 测试 1~4：常见组合（覆盖 mock 中有数据的城市）
    # --------------------------------------------------------------
    cases = [
        # (origin, destination, days)
        ("上海", "成都", 5),
        ("北京", "杭州", 3),
        ("广州", "西安", 4),
        ("深圳", "三亚", 6),
        # 测试 5~6：不在 mock 里的冷门城市（通用模板+矫正）
        ("苏州", "大理", 4),
        ("保定", "桂林", 5),
        ("佛山", "丽江", 3),
        # 测试 7：省会+直辖市混合
        ("武汉", "重庆", 5),
        # 测试 8：跨大区 极长距离
        ("哈尔滨", "海口", 7),
    ]

    print("=" * 70)
    print("   🚀 途景汇 · 出发地/目的地 一致性专项验证")
    print("=" * 70)

    for i, (origin, destination, days) in enumerate(cases, 1):
        print(f"\n【测试 {i}】{origin} → {destination}（{days}天）")
        req = {
            "origin": origin,
            "destination": destination,
            "dates": "2026-10-01",
            "days": days,
            "preferences": ["自然风光", "历史文化"],
            "budget": "标准",
        }
        result = await orch.plan_full_trip(req)
        plan = result["trip_plan"]

        errors = []
        # --- 校验 1：plan 字典顶层字段不能错 ---
        if plan["origin"] != origin:
            errors.append(f"顶层 origin 错误: {plan['origin']} (应为 {origin})")
        if plan["destination"] != destination:
            errors.append(f"顶层 destination 错误: {plan['destination']} (应为 {destination})")

        # --- 校验 2：transport（城际交通）里的标题和内容 ---
        trans = plan["transport"]
        # 标题中必须包含正确的 origin → destination
        title_ok = f"【路线规划】{origin} → {destination}" in trans
        if not title_ok:
            errors.append(f"transport 标题错误: {trans[:60]}...")
        # transport 里 origin 不该包含别的城市被写成出发点（除 origin 自身）
        # transport 里目的地不该被写成错误的（抓 → 后第一个城市名）
        import re
        m = re.search(r"【路线规划】([\u4e00-\u9fa5]{1,6})\s*→\s*([\u4e00-\u9fa5]{1,6})", trans)
        if m:
            if m.group(1) != origin:
                errors.append(f"transport 出发城市错写为 {m.group(1)}")
            if m.group(2) != destination:
                errors.append(f"transport 目的城市错写为 {m.group(2)}")

        # --- 校验 3：weather 必须是 destination 的天气 ---
        weather = plan["weather"]
        if f"【天气查询】{destination}未来" not in weather:
            # 至少标题里是 destination
            m2 = re.search(r"【天气查询】([\u4e00-\u9fa5]{1,6})未来", weather)
            if m2 and m2.group(1) != destination:
                errors.append(f"weather 写成了 {m2.group(1)} 的天气（应为 {destination}）")
            else:
                errors.append(f"weather 标题异常：{weather[:50]}")

        # --- 校验 4：attractions 必须是 destination 的景点 ---
        att = plan["attractions"]
        if f"【景点推荐】{destination}热门景点" not in att:
            m3 = re.search(r"【景点推荐】([\u4e00-\u9fa5]{1,6})热门景点", att)
            if m3 and m3.group(1) != destination:
                errors.append(f"attractions 推荐了 {m3.group(1)} 的景点（应为 {destination}）")
            elif not m3:
                # 可能是通用模板（核心景点 N 模式），再查 city 是否出现在标题
                if destination not in att[:80]:
                    errors.append(f"attractions 未包含 {destination} 在开头：{att[:80]}")

        # --- 校验 5：daily_routes 中 destination 必须出现 ---
        dr = plan["daily_routes"]
        if destination not in dr:
            errors.append(f"daily_routes 未包含 {destination}")
        # daily_routes 不得写成【每日行程 · 成都】这种错的
        m4 = re.search(r"【每日行程 · ([\u4e00-\u9fa5]{1,6})】", dr)
        if m4 and m4.group(1) != destination:
            errors.append(f"daily_routes 写成了 {m4.group(1)} 的每日行程")

        # --- 打印报告 ---
        if errors:
            failed += 1
            print(f"  ❌ 失败 ({len(errors)} 处错误)")
            for e in errors:
                print(f"     • {e}")
            # 输出关键片段方便调试
            print(f"     → transport 前120字: {trans[:120]}")
            print(f"     → weather 前120字: {weather[:120]}")
            print(f"     → attractions 前120字: {att[:120]}")
            print(f"     → daily_routes 前120字: {dr[:120]}")
        else:
            passed += 1
            print(f"  ✅ 通过：origin/destination 与输入完全一致")
            print(f"     🚄 transport: {trans[:60]}...")
            print(f"     🌤️ weather: {weather[:50]}...")
            print(f"     🏛️ attractions 开头含 {destination} ✓")
            print(f"     📅 daily_routes 含 {destination} ✓")

    # --------------------------------------------------------------
    # 测试 9：handle_user_request 单轮目的地矫正
    # --------------------------------------------------------------
    i += 1
    print(f"\n【测试 {i}】单轮对话：从苏州去昆明 怎么去 最快？")
    r = await orch.handle_user_request("从苏州去昆明 怎么去 最快？", session_id=f"s_{i}")
    resp = r["response"]
    import re as _re
    m = _re.search(r"【路线规划】([\u4e00-\u9fa5]{1,6})\s*→\s*([\u4e00-\u9fa5]{1,6})", resp)
    if m and m.group(1) == "苏州" and m.group(2) == "昆明":
        passed += 1
        print(f"  ✅ 通过：标题为【路线规划】苏州 → 昆明")
    else:
        failed += 1
        print(f"  ❌ 失败：实际为 {resp[:100]}")

    i += 1
    print(f"\n【测试 {i}】单轮对话：帮我查一下长沙的天气")
    r = await orch.handle_user_request("帮我查一下长沙的天气", session_id=f"s_{i}")
    resp = r["response"]
    if "长沙" in resp[:50]:
        passed += 1
        print(f"  ✅ 通过：响应是长沙的天气")
    else:
        failed += 1
        print(f"  ❌ 失败：响应 {resp[:80]}")

    i += 1
    print(f"\n【测试 {i}】单轮对话：推荐保定 3 天好玩的景点")
    r = await orch.handle_user_request("推荐保定 3 天好玩的景点", session_id=f"s_{i}")
    resp = r["response"]
    if "保定" in resp[:60]:
        passed += 1
        print(f"  ✅ 通过：保定景点在响应开头")
    else:
        failed += 1
        print(f"  ❌ 失败：{resp[:100]}")

    # --------------------------------------------------------------
    # 汇总
    # --------------------------------------------------------------
    total = passed + failed
    print("\n" + "=" * 70)
    print(f"  📊 总览：通过 {passed}/{total}，失败 {failed}")
    print("=" * 70)
    if failed == 0:
        print("  🎉 全部通过！出发地/目的地 与 用户输入 100% 一致！")
    else:
        print(f"  ⚠️  仍有 {failed} 项失败")


if __name__ == "__main__":
    asyncio.run(main())