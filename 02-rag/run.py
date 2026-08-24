# -*- coding: utf-8 -*-
"""
星途环球·跨境售后RAG智能问答系统 - 主启动入口
使用：
  python run.py              # 启动 FastAPI 服务 + 前端（默认端口8000）
  python run.py --port 8080  # 自定义端口
  python run.py --cli        # 命令行交互模式（不需要浏览器）
  python run.py --test       # 跑一次51组自动化评测
"""
from __future__ import annotations

import os
import sys
import asyncio
import argparse

# 确保项目根目录在 sys.path 中
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import get_config


def parse_args():
    p = argparse.ArgumentParser(
        description="星途环球·跨境售后RAG智能问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run.py                  启动 FastAPI + 前端 (http://localhost:8000)
  python run.py --cli            命令行交互问答
  python run.py --test           执行51组自动化评测 (Mock模式)
  python run.py --test --mode api 调用已启动的FastAPI接口做评测
        """,
    )
    p.add_argument("--host", default=None, help="FastAPI监听地址 (默认0.0.0.0)")
    p.add_argument("--port", type=int, default=None, help="FastAPI端口 (默认8000)")
    p.add_argument("--cli", action="store_true", help="命令行交互模式")
    p.add_argument("--test", action="store_true", help="执行自动化评测")
    p.add_argument("--mode", choices=["mock", "local", "api"], default="mock", help="评测模式：mock/local/api")
    p.add_argument("--category", choices=["cross_border_logistics", "after_sales_policy", "customs_compliance"],
                   default=None, help="评测只跑某一类")
    p.add_argument("--limit", type=int, default=None, help="评测用例数量上限")
    p.add_argument("--mock-noise", type=float, default=0.0, help="Mock模式噪声0-1，演示幻觉识别用")
    p.add_argument("--api", dest="api_base", default=None, help="API模式下的网关地址")
    return p.parse_args()


def run_server(args):
    """启动 FastAPI 服务"""
    import uvicorn
    cfg = get_config()
    host = args.host or cfg.agents.gateway_host
    port = args.port or cfg.agents.gateway_port
    print("=" * 70)
    print("🚀 星途环球·跨境售后RAG智能问答系统 - 启动")
    print("=" * 70)
    print(f"  📡 服务地址      : http://{host}:{port}")
    print(f"  💬 前端咨询页面  : http://localhost:{port}/")
    print(f"  📖 接口文档(Swagger): http://localhost:{port}/docs")
    print(f"  🤖 问答主接口    : POST http://localhost:{port}/api/chat")
    print(f"  📊 健康检查      : GET  http://localhost:{port}/api/health")
    print("=" * 70)
    print("  提示：按 Ctrl+C 停止服务")
    print("=" * 70)
    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


def run_cli():
    """命令行交互模式"""
    from gateway import AgentOrchestrator

    print("=" * 70)
    print("💬 星途环球·跨境售后RAG智能客服 (CLI模式)")
    print("=" * 70)
    print("  输入您的跨境售后/物流/合规相关问题")
    print("  输入 quit / exit / q 退出")
    print("  切换语言：输入 'lang en' / 'lang ja' / 'lang zh'")
    print("=" * 70)

    orch = AgentOrchestrator.get_instance()
    language = None

    async def _main():
        nonlocal language
        while True:
            try:
                user_input = input("\n😊 你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 再见！")
                break
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 再见！")
                break
            # 语言切换
            if user_input.lower().startswith("lang "):
                parts = user_input.split()
                if len(parts) >= 2:
                    language = parts[1].lower()
                    print(f"   🔄 已切换到输出语言: {language}")
                continue
            result = await orch.handle_user_request(
                user_input=user_input,
                session_id="cli_session",
                force_language=language,
            )
            latency = result.get("trace", {}).get("total_latency_ms", "?")
            print(f"\n🤖 AI ({latency}ms)  >")
            print(result.get("response", ""))
            # 打印风控摘要
            safety = result.get("trace", {}).get("safety_log", {})
            if safety:
                fr = safety.get("faithfulness", {})
                hr = safety.get("hallucination", {})
                print(
                    f"   🎯 忠实度={fr.get('score', 0):.0%} | "
                    f"⚠️ 幻觉={hr.get('level', 'none')}"
                )

    asyncio.run(_main())


def run_tests(args):
    """执行自动化评测（兼容原脚本路径+模块导入）"""
    import importlib.util
    test_mod_path = os.path.join(ROOT, "test_rag_qa.py")
    spec = importlib.util.spec_from_file_location("test_rag_qa_mod", test_mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run_test_suite = mod.run_test_suite

    async def _main():
        results, summary, dfile, sfile = await run_test_suite(
            mode=args.mode,
            category_filter=args.category,
            limit=args.limit,
            api_base=args.api_base,
            mock_noise=args.mock_noise,
            verbose=True,
        )
        passed = summary["summary"]["passed"]
        total = summary["summary"]["total_cases"]
        print(f"\n🏁 评测完成：{passed}/{total} 通过 ({summary['summary']['pass_rate_percent']})")
        code = 0 if total > 0 and passed / total >= 0.80 else 1
        sys.exit(code)

    asyncio.run(_main())


def main():
    args = parse_args()
    cfg = get_config()  # 提前初始化配置 & 目录

    if args.test:
        run_tests(args)
    elif args.cli:
        run_cli()
    else:
        run_server(args)


if __name__ == "__main__":
    main()
