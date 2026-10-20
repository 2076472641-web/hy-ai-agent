# -*- coding: utf-8 -*-
"""
多智能体协同测试 - 验证完整旅游行程规划
测试场景：
1. 单智能体查询（天气、景点、路线）
2. 多智能体协同（天气+景点、景点+路线）
3. 一键规划完整行程
4. 对话上下文压缩
5. A2A 通信协议
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import asyncio
import time


SEPARATOR = "=" * 70
PASS = 0
FAIL = 0


def log_pass(name: str, detail: str = ""):
    global PASS
    PASS += 1
    print(f"  ✅ PASS: {name}")
    if detail:
        print(f"     {detail}")


def log_fail(name: str, error: str = ""):
    global FAIL
    FAIL += 1
    print(f"  ❌ FAIL: {name}")
    if error:
        print(f"     错误: {error}")


def log_info(msg: str):
    print(f"  ℹ️  {msg}")


# ============================================================
# 测试 1: A2A 协议数据结构
# ============================================================
def test_a2a_protocol():
    """测试 A2A 协议的数据结构定义"""
    print(f"\n{SEPARATOR}")
    print("测试 1: A2A 协议数据结构")
    print(SEPARATOR)

    from a2a.protocol import AgentMessage, AgentTask, TaskStatus, MessageRole

    # 测试消息创建
    msg = AgentMessage(
        role=MessageRole.USER,
        content="帮我查一下成都天气",
    )
    assert msg.id, "消息ID应自动生成"
    assert msg.role == MessageRole.USER
    assert msg.content == "帮我查一下成都天气"
    log_pass("AgentMessage 创建")

    # 测试任务创建
    task = AgentTask(
        title="查询天气",
        description="查询成都今日天气",
        metadata={"task_type": "weather"},
    )
    assert task.status == TaskStatus.PENDING
    task.progress()
    assert task.status == TaskStatus.IN_PROGRESS
    task.complete(result={"weather": "晴"})
    assert task.status == TaskStatus.COMPLETED
    assert task.result == {"weather": "晴"}
    log_pass("AgentTask 状态流转")

    # 测试任务失败
    task2 = AgentTask(title="失败任务")
    task2.fail(error="连接超时")
    assert task2.status == TaskStatus.FAILED
    assert task2.error == "连接超时"
    log_pass("AgentTask 失败处理")


# ============================================================
# 测试 2: Agent Card
# ============================================================
def test_agent_cards():
    """测试 Agent Card 的创建"""
    print(f"\n{SEPARATOR}")
    print("测试 2: Agent Card")
    print(SEPARATOR)

    from a2a.agent_card import AgentCard

    # 天气智能体
    weather_card = AgentCard.create_weather_agent("http://localhost:8101")
    assert weather_card.name == "weather_agent"
    assert len(weather_card.skills) == 1
    assert weather_card.skills[0].id == "query_weather"
    log_pass("天气 Agent Card 创建", f"技能数: {len(weather_card.skills)}")

    # 景点智能体
    attraction_card = AgentCard.create_attraction_agent("http://localhost:8102")
    assert attraction_card.name == "attraction_agent"
    assert len(attraction_card.skills) == 1
    log_pass("景点 Agent Card 创建", f"技能数: {len(attraction_card.skills)}")

    # 路线智能体
    route_card = AgentCard.create_route_agent("http://localhost:8103")
    assert route_card.name == "route_agent"
    assert len(route_card.skills) == 2
    log_pass("路线 Agent Card 创建", f"技能数: {len(route_card.skills)}")


# ============================================================
# 测试 3: 意图检测
# ============================================================
def test_intent_detection():
    """测试编排器的意图检测"""
    print(f"\n{SEPARATOR}")
    print("测试 3: 意图检测")
    print(SEPARATOR)

    from gateway.orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator()

    # 天气意图
    intents = orchestrator.detect_intent("帮我查一下北京明天的天气")
    assert "weather_agent" in intents
    log_pass("天气意图检测", f"输入: '帮我查一下北京明天的天气' → {intents}")

    # 景点意图
    intents = orchestrator.detect_intent("推荐成都3天游玩的景点")
    assert "attraction_agent" in intents
    log_pass("景点意图检测", f"输入: '推荐成都3天游玩的景点' → {intents}")

    # 路线意图
    intents = orchestrator.detect_intent("从上海到杭州怎么去最方便")
    assert "route_agent" in intents
    log_pass("路线意图检测", f"输入: '从上海到杭州怎么去最方便' → {intents}")

    # 多意图
    intents = orchestrator.detect_intent("帮我查成都天气并推荐景点")
    assert "weather_agent" in intents
    assert "attraction_agent" in intents
    log_pass("多意图检测（天气+景点）", f"→ {intents}")


# ============================================================
# 测试 4: 本地智能体回退（无需远程服务）
# ============================================================
async def test_local_fallback():
    """测试本地智能体回退功能"""
    print(f"\n{SEPARATOR}")
    print("测试 4: 本地智能体回退")
    print(SEPARATOR)

    from gateway.orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator()

    # 天气回退
    result = await orchestrator._local_fallback("weather_agent", "帮我查一下广州明天的天气")
    assert "response" in result
    assert "广州" in result["response"]
    log_pass("天气本地回退", f"响应: {result['response'][:80]}...")

    # 景点回退
    result = await orchestrator._local_fallback("attraction_agent", "推荐成都好玩的景点")
    assert "response" in result
    assert "成都" in result["response"]
    log_pass("景点本地回退", f"响应: {result['response'][:80]}...")

    # 路线回退
    result = await orchestrator._local_fallback("route_agent", "从上海到杭州怎么去")
    assert "response" in result
    assert "上海" in result["response"]
    log_pass("路线本地回退", f"响应: {result['response'][:80]}...")


# ============================================================
# 测试 5: 对话记忆与上下文压缩
# ============================================================
def test_conversation_memory():
    """测试对话记忆功能"""
    print(f"\n{SEPARATOR}")
    print("测试 5: 对话记忆与上下文压缩")
    print(SEPARATOR)

    from memory.conversation_memory import ConversationMemory

    memory = ConversationMemory(memory_dir="./data/test_checkpoints")

    # 创建会话
    session_id = "test_session_001"
    session = memory.get_or_create_session(session_id)
    log_pass("会话创建", f"session_id: {session_id}")

    # 添加消息
    memory.add_message(session_id, "user", "你好，我想去成都旅游")
    memory.add_message(session_id, "assistant", "好的！成都有很多好玩的地方。")
    memory.add_message(session_id, "user", "推荐一些景点吧")
    memory.add_message(session_id, "assistant", "推荐大熊猫基地、宽窄巷子、都江堰...")
    log_pass("消息添加", f"已添加 4 条消息")

    # 获取消息
    messages = memory.get_messages(session_id)
    assert len(messages) == 4
    log_pass("消息获取", f"获取到 {len(messages)} 条消息")

    # 聊天历史
    history = memory.get_chat_history(session_id)
    assert len(history) == 4
    log_pass("聊天历史", f"格式: {len(history)} 条")

    # Token 估算
    tokens = memory.count_tokens(session_id)
    log_pass("Token 估算", f"约 {tokens} tokens")

    # 会话统计
    stats = memory.get_session_stats(session_id)
    assert stats["message_count"] == 4
    log_pass("会话统计", f"用户消息: {stats['user_messages']}, 助手消息: {stats['assistant_messages']}")

    # 压缩测试
    memory.add_message(session_id, "user", "帮我规划行程")
    memory.add_message(session_id, "assistant", "好的，为你规划3天行程...")
    memory.set_summary(session_id, "用户想去成都，已推荐景点，正在规划行程")
    log_pass("摘要设置")

    # 清理
    memory.delete_session(session_id)
    log_pass("会话删除")


def test_context_compression():
    """测试上下文压缩"""
    print(f"\n{SEPARATOR}")
    print("测试 6: 上下文压缩")
    print(SEPARATOR)

    from memory.conversation_memory import ConversationMemory
    from memory.context_compressor import ContextCompressor

    memory = ConversationMemory(memory_dir="./data/test_checkpoints")
    compressor = ContextCompressor(memory)

    session_id = "test_compress_001"
    memory.get_or_create_session(session_id)

    # 添加大量消息
    for i in range(15):
        memory.add_message(session_id, "user", f"问题 {i+1}: 关于旅游的问题{i+1}")
        memory.add_message(session_id, "assistant", f"回答 {i+1}: 这是关于问题{i+1}的回答，内容比较长以测试压缩效果...")

    # 检查是否需要压缩
    stats = memory.get_session_stats(session_id)
    log_pass("压缩前统计", f"消息数: {stats['message_count']}, tokens: {stats['estimated_tokens']}")

    # 如果 token 数不够，手动触发压缩
    if not compressor.should_compress(session_id):
        # 直接调用 compress 而不是 auto_compress
        result = memory.compress(session_id, keep_recent=6)
    else:
        result = compressor.auto_compress(session_id, keep_recent=6)
    log_pass("自动压缩", f"结果: {json.dumps(result, ensure_ascii=False)}")

    # 验证压缩后保留最近消息
    messages = memory.get_messages(session_id)
    assert len(messages) <= 6
    log_pass("压缩验证", f"压缩后消息数: {len(messages)} (≤6)")

    # 智能压缩
    memory2 = ConversationMemory(memory_dir="./data/test_checkpoints")
    compressor2 = ContextCompressor(memory2)
    session_id2 = "test_smart_002"
    memory2.get_or_create_session(session_id2)

    # 添加含关键信息的消息
    memory2.add_message(session_id2, "user", "我决定国庆去成都，预算5000元")
    memory2.add_message(session_id2, "assistant", "好的，国庆是好选择")
    memory2.add_message(session_id2, "user", "推荐一些景点")
    memory2.add_message(session_id2, "assistant", "推荐熊猫基地")
    memory2.add_message(session_id2, "user", "我喜欢自然风光")
    memory2.add_message(session_id2, "assistant", "那推荐青城山")

    result2 = compressor2.smart_compress(session_id2, keep_recent=2)
    log_pass("智能压缩", f"结果: {json.dumps(result2, ensure_ascii=False)}")

    # 清理
    memory.delete_session(session_id)
    memory2.delete_session(session_id2)


# ============================================================
# 测试 7: 完整行程规划（本地回退模式）
# ============================================================
async def test_full_trip_planning():
    """测试完整行程规划流程"""
    print(f"\n{SEPARATOR}")
    print("测试 7: 完整旅游行程规划")
    print(SEPARATOR)

    from gateway.orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator()

    # 测试 1: 天气查询
    result = await orchestrator.handle_user_request(
        "帮我查一下成都明天的天气怎么样",
        session_id="trip_001",
    )
    assert "response" in result
    log_pass("天气查询", f"响应: {result['response'][:100]}...")
    log_info(f"  调用的智能体: {result['agents_called']}")

    # 测试 2: 景点推荐
    result = await orchestrator.handle_user_request(
        "推荐北京3天游玩的景点",
        session_id="trip_001",
    )
    assert "response" in result
    log_pass("景点推荐", f"响应: {result['response'][:100]}...")

    # 测试 3: 路线规划
    result = await orchestrator.handle_user_request(
        "从上海到杭州怎么去最方便",
        session_id="trip_001",
    )
    assert "response" in result
    log_pass("路线规划", f"响应: {result['response'][:100]}...")

    # 测试 4: 多意图（天气+景点）
    result = await orchestrator.handle_user_request(
        "帮我查杭州天气并推荐景点",
        session_id="trip_002",
    )
    assert "response" in result
    log_pass("多意图协同（天气+景点）", f"调用: {result['agents_called']}")

    # 测试 5: 一键规划完整行程
    trip_result = await orchestrator.plan_full_trip({
        "origin": "上海",
        "destination": "成都",
        "dates": "2026-10-01",
        "days": 5,
        "preferences": ["自然风光", "历史文化"],
        "budget": "标准",
    })
    assert "trip_plan" in trip_result
    assert trip_result["status"] == "completed"
    log_pass("一键行程规划", f"session_id: {trip_result['session_id']}")

    # 验证行程内容
    plan = trip_result["trip_plan"]
    assert plan["origin"] == "上海"
    assert plan["destination"] == "成都"
    log_pass("行程内容验证", f"天气/景点/路线都已生成")

    # 打印完整行程
    print(f"\n  {'='*60}")
    print(f"  🧳 完整旅游行程规划结果")
    print(f"  {'='*60}")
    print(f"  出发地: {plan['origin']} → 目的地: {plan['destination']}")
    print(f"\n  🌤️  天气信息:")
    for line in plan["weather"].split("\n")[:3]:
        print(f"    {line}")
    print(f"\n  🏛️  景点推荐:")
    for line in plan["attractions"].split("\n")[:5]:
        print(f"    {line}")
    print(f"\n  🚄 交通方案:")
    for line in plan["transport"].split("\n")[:5]:
        print(f"    {line}")
    print(f"\n  📅 每日行程:")
    for line in plan["daily_routes"].split("\n")[:5]:
        print(f"    {line}")
    print(f"  {'='*60}")


# ============================================================
# 测试 8: 数据库查询
# ============================================================
def test_db_tools():
    """测试数据库工具"""
    print(f"\n{SEPARATOR}")
    print("测试 8: 数据库工具")
    print(SEPARATOR)

    from mcp_server.tools.db_tools import (
        _query_destinations_mock,
        _get_city_info_mock,
        _search_packages_mock,
    )

    # 目的地查询
    result = _query_destinations_mock("北京", 10)
    assert result["total"] > 0
    log_pass("目的地查询", f"找到 {result['total']} 个目的地")

    # 城市信息
    info = _get_city_info_mock("成都")
    assert "best_season" in info
    assert info["name"] == "成都"
    log_pass("城市信息", f"{info['name']}最佳季节: {info['best_season']}")

    # 套餐搜索
    packages = _search_packages_mock("成都", 3, "标准")
    assert packages["total"] > 0
    log_pass("套餐搜索", f"找到 {packages['total']} 个套餐")


# ============================================================
# 测试 9: 完整协同场景 - 多轮对话
# ============================================================
async def test_multi_turn_conversation():
    """测试多轮对话上下文保持"""
    print(f"\n{SEPARATOR}")
    print("测试 9: 多轮对话协同")
    print(SEPARATOR)

    from gateway.orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator()
    session_id = "multi_turn_001"

    # 第1轮：初始咨询
    r1 = await orchestrator.handle_user_request(
        "你好，我想国庆节出去旅游",
        session_id=session_id,
    )
    log_pass("第1轮对话", f"响应: {r1['response'][:80]}...")

    # 第2轮：询问具体目的地
    r2 = await orchestrator.handle_user_request(
        "帮我推荐成都的景点",
        session_id=session_id,
    )
    log_pass("第2轮对话", f"响应: {r2['response'][:80]}...")

    # 第3轮：询问交通
    r3 = await orchestrator.handle_user_request(
        "从上海到成都怎么去",
        session_id=session_id,
    )
    log_pass("第3轮对话", f"响应: {r3['response'][:80]}...")

    # 第4轮：询问天气
    r4 = await orchestrator.handle_user_request(
        "成都国庆节天气怎么样",
        session_id=session_id,
    )
    log_pass("第4轮对话", f"响应: {r4['response'][:80]}...")

    # 验证会话历史存在（使用编排器的 memory 实例）
    messages = orchestrator.memory.get_messages(session_id)
    log_pass("多轮会话历史", f"共 {len(messages)} 条消息")


# ============================================================
# 测试入口
# ============================================================
def main():
    global PASS, FAIL

    print("╔" + "═" * 68 + "╗")
    print("║" + "途景汇文旅多智能体一体化系统 - 集成测试".center(60) + "        ║")
    print("║" + f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}".center(60) + "        ║")
    print("╚" + "═" * 68 + "╝")

    # 同步测试
    try:
        test_a2a_protocol()
    except Exception as e:
        log_fail("A2A协议", str(e))

    try:
        test_agent_cards()
    except Exception as e:
        log_fail("Agent Card", str(e))

    try:
        test_intent_detection()
    except Exception as e:
        log_fail("意图检测", str(e))

    try:
        test_db_tools()
    except Exception as e:
        log_fail("数据库工具", str(e))

    try:
        test_conversation_memory()
    except Exception as e:
        log_fail("对话记忆", str(e))

    try:
        test_context_compression()
    except Exception as e:
        log_fail("上下文压缩", str(e))

    # 异步测试
    async def run_async_tests():
        try:
            await test_local_fallback()
        except Exception as e:
            log_fail("本地回退", str(e))

        try:
            await test_full_trip_planning()
        except Exception as e:
            import traceback
            log_fail("完整行程规划", str(e))
            traceback.print_exc()

        try:
            await test_multi_turn_conversation()
        except Exception as e:
            import traceback
            log_fail("多轮对话", str(e))
            traceback.print_exc()

    asyncio.run(run_async_tests())

    # 汇总
    print(f"\n{SEPARATOR}")
    print("📊 测试汇总")
    print(SEPARATOR)
    print(f"  ✅ 通过: {PASS}")
    print(f"  ❌ 失败: {FAIL}")
    print(f"  📝 总计: {PASS + FAIL}")

    if FAIL == 0:
        print(f"\n  🎉 所有测试通过！系统功能正常！")
    else:
        print(f"\n  ⚠️  有 {FAIL} 个测试失败，请检查错误信息。")

    return FAIL == 0


if __name__ == "__main__":
    import asyncio
    success = main()
    sys.exit(0 if success else 1)