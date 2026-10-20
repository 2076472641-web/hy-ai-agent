# -*- coding: utf-8 -*-
"""
星途环球·跨境售后RAG智能问答系统 - 主自动化测试脚本

功能说明：
1. 从 test_cases.json 加载51组跨境售后标准测试用例
2. 支持三种运行模式：
   - API模式：调用FastAPI RAG问答接口（默认：http://localhost:8000/api/chat）
   - 本地Mock模式：使用项目编排器Orchestrator模拟回答（无需启动服务）
   - 本地Orchestrator模式：直接复用项目编排能力
3. 采集所有评测指标（耗时/召回/忠实度/幻觉/准确率）
4. 自动输出日志与统计报告到 tests/test_output/

兼容 Windows 环境，路径全部使用相对路径
"""
from __future__ import annotations

import sys
import os

# 兼容项目路径导入（与 tests/test_collaboration.py 风格一致）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import asyncio
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# 复用项目全局配置
from config import get_config
# 评测工具
from tests.test_evaluate import (
    evaluate_qa_case,
    build_summary_report,
    format_report_console,
)


SEPARATOR = "=" * 78
CATEGORY_NAME = {
    "cross_border_logistics": "📦 跨境物流咨询",
    "after_sales_policy": "🛒 海外售后政策",
    "customs_compliance": "📜 各国海关合规政策",
}


# ============================================================
# 1. 路径 & 日志输出辅助
# ============================================================
def _ensure_dirs() -> Tuple[str, str]:
    """确保 tests/test_output 目录存在，返回本次测试输出文件名"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "test_output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_file = os.path.join(output_dir, f"rag_test_detail_{timestamp}.json")
    summary_file = os.path.join(output_dir, f"rag_test_summary_{timestamp}.txt")
    return detail_file, summary_file


def log_info(msg: str, indent: int = 0):
    """控制台彩色信息输出（面试演示更直观）"""
    prefix = " " * indent
    print(f"{prefix}{msg}")


def log_pass(case_id: str, title: str, score: float, grade: str):
    mark = "✅"
    score_str = f"{round(score * 100, 1)}".ljust(5)
    print(f"  {mark} PASS [{case_id}] {grade} ({score_str}分)  {title[:36]}")


def log_fail(case_id: str, title: str, score: float, grade: str, reason: str = ""):
    mark = "❌"
    score_str = f"{round(score * 100, 1)}".ljust(5)
    print(f"  {mark} FAIL [{case_id}] {grade} ({score_str}分)  {title[:36]}")
    if reason:
        print(f"      原因: {reason[:80]}")


# ============================================================
# 2. 测试用例加载
# ============================================================
def load_test_cases(category_filter: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    从 JSON 文件加载测试用例
    category_filter: cross_border_logistics / after_sales_policy / customs_compliance
    limit: 限制用例数量（调试用）
    """
    cases_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_cases.json",
    )
    if not os.path.exists(cases_path):
        raise FileNotFoundError(f"测试用例文件不存在：{cases_path}")

    with open(cases_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("test_cases", [])

    if category_filter:
        cases = [c for c in cases if c.get("category") == category_filter]
    if limit and limit > 0:
        cases = cases[:limit]
    return cases


# ============================================================
# 3. 三种回答生成模式
# ============================================================

# ------- 模式A：调用 FastAPI RAG 接口（需要服务已启动） -------
async def ask_rag_api(
    question: str,
    base_url: Optional[str] = None,
    session_id: str = "rag_test_session",
    timeout: float = 60.0,
) -> Tuple[str, List[str]]:
    """
    调用 FastAPI RAG 问答接口
    返回 (回答文本, 多路召回知识片段列表)
    """
    config = get_config()
    if not base_url:
        port = config.agents.gateway_port  # 默认8000
        base_url = f"http://localhost:{port}"

    import httpx

    chat_url = base_url.rstrip("/") + "/api/chat"
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            chat_url,
            json={"message": question, "session_id": session_id},
        )
        resp.raise_for_status()
        data = resp.json()
    answer = data.get("response", "")
    # 若接口返回多路召回context，读取；否则给空列表
    contexts = data.get("retrieved_contexts") or data.get("contexts") or []
    return answer, list(contexts)


# ------- 模式B：本地 Orchestrator 模式（真实项目编排能力） -------
async def ask_local_orchestrator(question: str) -> Tuple[str, List[str]]:
    """
    直接复用项目网关编排器 + 本地Fallback生成回答
    - 适配当前途景汇项目的 AgentOrchestrator
    - 无需启动任何服务，可直接测试
    """
    from gateway.orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator()
    result = await orchestrator.handle_user_request(
        user_input=question,
        session_id="rag_auto_test",
    )
    answer = result.get("response", "")
    contexts = result.get("contexts") or []
    return answer, list(contexts)


# ------- 模式C：基于参考答案的轻量Mock（离线演示用） -------
def ask_mock(
    question: str,
    reference: str,
    noise: float = 0.0,
) -> Tuple[str, List[str]]:
    """
    离线Mock模式：根据参考答案生成带一定噪声的回答
    noise=0.0 回答完全正确；noise=1.0 回答完全胡编
    用于面试现场演示评测工具对"好回答/坏回答"的识别能力
    """
    import random

    if noise <= 0.05:
        answer = reference
    elif noise < 0.4:
        # 轻微扰动：替换少量词
        replacements = {
            "约": "大概", "约": "大约",
            "天": "个工作日",
            "元": "块钱",
            "推荐": "建议使用",
        }
        answer = reference
        for k, v in list(replacements.items())[:3]:
            if k in answer:
                answer = answer.replace(k, v, 1)
    elif noise < 0.7:
        # 中度：缺部分关键词
        sentences = re.split(r"[。；]", reference)
        keep = sentences[: max(1, len(sentences) // 2)]
        answer = "；".join(keep) + "。"
    else:
        # 重度：编造内容，触发幻觉检测
        fake_starts = [
            "根据最新研究，",
            "专家指出，这个问题100%的解法是",
            "全球第一的物流方案是：",
        ]
        random_nums = [f"{random.randint(1000, 99999)}元", f"99.99%准确率"]
        answer = random.choice(fake_starts) + "只需要" + "，".join(random.sample(random_nums, 2)) + "。"

    # 模拟召回片段：从参考回答按句子拆几段
    refs = [s.strip() for s in re.split(r"[。；]", reference) if s.strip()]
    retrieved = refs[: max(1, len(refs) // 2)]
    return answer, retrieved


import re  # noqa: E402 - mock模式用


# ============================================================
# 4. 单条测试执行
# ============================================================
async def run_single_case(
    case: Dict[str, Any],
    mode: str,
    api_base: Optional[str] = None,
    mock_noise: float = 0.0,
) -> Dict[str, Any]:
    """
    执行单条测试用例
    返回整合了 case 原始信息 + 耗时 + LLM回答 + 评测结果 的字典
    """
    case_id = case["id"]
    category = case["category"]
    question = case["question"]
    reference = case.get("reference_answer", "")
    expected_keywords = case.get("expected_keywords", [])
    expected_entities = case.get("expected_entities", {})

    # 调用问答
    start_ts = time.perf_counter()
    try:
        if mode == "api":
            answer, contexts = await ask_rag_api(question, base_url=api_base)
        elif mode == "local":
            answer, contexts = await ask_local_orchestrator(question)
        else:  # mock
            answer, contexts = ask_mock(question, reference, noise=mock_noise)
        latency_ms = round((time.perf_counter() - start_ts) * 1000, 2)
        error = None
    except Exception as exc:
        answer = ""
        contexts = []
        latency_ms = round((time.perf_counter() - start_ts) * 1000, 2)
        error = f"{type(exc).__name__}: {exc}"

    # 评测
    eval_result = evaluate_qa_case(
        question=question,
        answer_text=answer,
        reference_text=reference,
        expected_keywords=expected_keywords,
        expected_entities=expected_entities,
        retrieved_contexts=contexts,
    )

    # 拼装最终结果（含原始case + 运行信息 + 评测）
    result = {
        "id": case_id,
        "category": category,
        "category_name": case.get("category_name", category),
        "difficulty": case.get("difficulty", "unknown"),
        "knowledge_point": case.get("knowledge_point", ""),
        "question": question,
        "llm_output": answer,
        "retrieved_contexts_top3": contexts[:3],  # 多路召回结果
        "reference_answer": reference,
        "latency_ms": latency_ms,
        "error": error,
        **eval_result,  # accuracy_score / passed / grade / keyword_metrics / ...
    }
    return result


# ============================================================
# 5. 批量执行
# ============================================================
async def run_test_suite(
    mode: str = "mock",
    category_filter: Optional[str] = None,
    limit: Optional[int] = None,
    api_base: Optional[str] = None,
    mock_noise: float = 0.0,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str, str]:
    """
    批量执行测试套件
    返回：(单case结果列表, 汇总报告, 明细文件路径, 汇总报告文件路径)
    """
    detail_file, summary_file = _ensure_dirs()
    cases = load_test_cases(category_filter=category_filter, limit=limit)

    if verbose:
        print("\n" + SEPARATOR)
        print("🚀 星途环球·跨境售后RAG智能问答系统 - 自动化测试启动")
        print(SEPARATOR)
        print(f"  运行模式        : {mode.upper()}  " + (
            "(调用FastAPI接口)" if mode == "api"
            else "(复用项目Orchestrator)" if mode == "local"
            else f"(离线Mock演示, 噪声={mock_noise})"
        ))
        print(f"  加载用例数      : {len(cases)}")
        if category_filter:
            print(f"  分类过滤        : {CATEGORY_NAME.get(category_filter, category_filter)}")
        print(f"  明细输出文件    : {os.path.relpath(detail_file)}")
        print(f"  汇总输出文件    : {os.path.relpath(summary_file)}")
        print(SEPARATOR + "\n")

    # 按分类分组跑，结果更清晰
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for c in cases:
        grouped.setdefault(c["category"], []).append(c)

    all_results: List[Dict[str, Any]] = []
    start_ts = time.time()

    for cat, cat_cases in grouped.items():
        name = CATEGORY_NAME.get(cat, cat)
        if verbose:
            print(f"\n{'─' * 78}")
            print(f"  {name}  ({len(cat_cases)} 条)")
            print(f"{'─' * 78}")

        for idx, case in enumerate(cat_cases, 1):
            result = await run_single_case(case, mode=mode, api_base=api_base, mock_noise=mock_noise)
            all_results.append(result)

            if verbose:
                if result["passed"]:
                    log_pass(result["id"], case["knowledge_point"], result["accuracy_score"], result["grade"])
                else:
                    reasons = result.get("hallucination", {}).get("reasons", [])
                    reason_txt = reasons[0] if reasons else (result["error"] or f"得分 {round(result['accuracy_score'] * 100, 1)}")
                    log_fail(result["id"], case["knowledge_point"], result["accuracy_score"], result["grade"], reason_txt)

    end_ts = time.time()

    # 构建汇总报告
    summary = build_summary_report(all_results, start_ts, end_ts)

    # 写明细JSON
    with open(detail_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "mode": mode,
                    "api_base": api_base,
                    "mock_noise": mock_noise,
                },
                "summary": summary,
                "cases": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 写汇总TXT
    report_text = format_report_console(summary)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(report_text + "\n\n")
        # 再补一份失败Case清单
        failed = [r for r in all_results if not r["passed"]]
        if failed:
            f.write("🚫 失败用例清单:\n")
            for r in failed:
                hall_reason = " ,".join(r.get("hallucination", {}).get("reasons", []))
                f.write(
                    f"  [{r['id']}] {r['knowledge_point'][:30]}  "
                    f"得分={round(r['accuracy_score'] * 100, 1)}  幻觉={r['hallucination']['risk_level']}\n"
                )
                f.write(f"     问题: {r['question'][:60]}\n")
                if hall_reason:
                    f.write(f"     问题: {hall_reason[:100]}\n")

    if verbose:
        print(report_text)
        print(f"  📁 测试明细JSON: {os.path.relpath(detail_file)}")
        print(f"  📄 测试汇总TXT : {os.path.relpath(summary_file)}")
        print()

    return all_results, summary, detail_file, summary_file


# ============================================================
# 6. 命令行入口
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="星途环球·跨境售后RAG智能问答系统 - 自动化测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例（面试演示现场可直接跑）：
  python -m tests.test_rag_qa                                    # Mock模式跑51组全部
  python -m tests.test_rag_qa --mode local                       # 复用项目编排器（真实逻辑）
  python -m tests.test_rag_qa --mode api --api http://localhost:8000   # 调用已启动FastAPI
  python -m tests.test_rag_qa --category cross_border_logistics  # 只跑物流咨询17组
  python -m tests.test_rag_qa --mock-noise 0.8 --limit 5         # 演示5组坏回答，看评测工具怎么识别幻觉
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "local", "api"],
        default="mock",
        help="运行模式：mock=离线演示(默认) / local=复用项目编排器 / api=调FastAPI服务",
    )
    parser.add_argument(
        "--category",
        choices=["cross_border_logistics", "after_sales_policy", "customs_compliance"],
        default=None,
        help="可选：只跑某一类知识库的用例",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="可选：限制跑多少条（快速调试用）",
    )
    parser.add_argument(
        "--api",
        dest="api_base",
        default=None,
        help="API模式下网关地址，默认从config.py读8000端口",
    )
    parser.add_argument(
        "--mock-noise",
        type=float,
        default=0.0,
        help="Mock模式下的噪声级别 0.0~1.0（0=全部对，1=全部幻觉，面试演示用）",
    )
    return parser.parse_args()


async def async_main():
    args = parse_args()
    # 复用项目全局配置（.env 环境变量已在 config.py 加载）
    _config = get_config()  # noqa: F841 - 触发config.py中的load_dotenv

    _results, summary, _dfile, _sfile = await run_test_suite(
        mode=args.mode,
        category_filter=args.category,
        limit=args.limit,
        api_base=args.api_base,
        mock_noise=args.mock_noise,
        verbose=True,
    )
    # 退出码：通过率>=80% 返回0，否则返回1（CI友好）
    passed = summary["summary"]["passed"]
    total = summary["summary"]["total_cases"]
    code = 0 if total > 0 and passed / total >= 0.80 else 1
    return code


def main():
    try:
        code = asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n⏹️  用户中断测试")
        code = 130
    except Exception as exc:
        import traceback
        print(f"\n💥 测试运行异常：{type(exc).__name__}: {exc}")
        traceback.print_exc()
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
