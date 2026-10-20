# -*- coding: utf-8 -*-
"""快速冒烟测试：验证编排器加载、5个典型问题问答链路"""
from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gateway.orchestrator import AgentOrchestrator


TEST_CASES = [
    # 跨境物流
    ("📦 物流类", "从中国深圳发一件500g的包裹到美国纽约，用什么物流方式最便宜？大概需要几天？"),
    # 海外售后
    ("🛒 售后类", "亚马逊美国站30天无理由退换货政策具体是怎样的？"),
    # 海关合规
    ("📜 合规类", "发往美国的个人包裹，免税额度是多少？超过要交什么税？"),
    # 多语种
    ("🌐 英语", "What is the cheapest shipping method for a 500g package from Shenzhen to New York?"),
    # 极端问题（触发风控拒答）
    ("⚠️ 幻觉测试", "全球第一的跨境物流方案是什么？肯定只要99999元运费，100%当天到货！"),
]


async def main():
    print("=" * 78)
    print("🚀 星途环球·跨境售后RAG智能问答系统 - 冒烟测试")
    print("=" * 78)

    orch = AgentOrchestrator.get_instance()
    status = orch.status()
    print(f"\n✅ 编排器加载完成：已索引 {status['total_indexed_chunks']} 个知识分块")
    print(f"   LLM: {status['llm_provider']} | Rerank: {status['rerank_provider']} | Vector: {status['vector_store']}\n")

    passed = 0
    failed = 0
    for label, q in TEST_CASES:
        print("─" * 78)
        print(f"  {label}")
        print(f"  Q: {q}")
        try:
            force_lang = "en" if label.startswith("🌐") else None
            result = await orch.handle_user_request(q, session_id="smoke_test", force_language=force_lang)
            resp = result.get("response", "")
            latency = result.get("trace", {}).get("total_latency_ms", "?")
            safety = result.get("trace", {}).get("safety_log", {})
            faith = safety.get("faithfulness", {}).get("score", 0)
            halluc = safety.get("hallucination", {}).get("level", "?")

            print(f"  A ({latency}ms | Faith={faith:.0%} | Halluc={halluc}):")
            # 截断展示
            if len(resp) > 600:
                print("    " + resp[:600].replace("\n", "\n    ") + "\n    ...(已截断)")
            else:
                print("    " + resp.replace("\n", "\n    "))

            # 简易判定：回答非空 + 非空拒答也算通过（风控类）
            if resp and len(resp.strip()) >= 5:
                passed += 1
                print(f"  ✅ PASS")
            else:
                failed += 1
                print(f"  ❌ FAIL: 回答过短或空")
        except Exception as e:
            failed += 1
            import traceback
            print(f"  ❌ FAIL: {type(e).__name__}: {e}")
            traceback.print_exc()
        print()

    print("=" * 78)
    print(f"  测试结果：{passed} PASS / {failed} FAIL  (共 {len(TEST_CASES)} 项)")
    print("=" * 78)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    code = asyncio.run(main())
    sys.exit(code)
