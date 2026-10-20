# -*- coding: utf-8 -*-
"""
星途环球·跨境售后RAG智能问答系统 - 评测指标计算工具
功能清单：
1. 关键词召回率 / 精确率计算
2. 语义相似度（基于字符级n-gram + Jaccard）
3. 回答忠实度检测（faithfulness）
4. 幻觉风险识别（hallucination risk scoring）
5. 实体对齐校验
6. 问答准确率综合评估
"""
from __future__ import annotations

import re
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter


# ============================================================
# 1. 基础文本处理工具
# ============================================================
def _tokenize_cn(text: str, n_gram: int = 2) -> List[str]:
    """
    中文轻量分词：字符级n-gram + 英文数字整词
    不额外依赖jieba，保持与项目一致的轻量风格
    """
    text = text or ""
    tokens = []
    # 英文/数字整词抽取
    en_words = re.findall(r"[A-Za-z][A-Za-z0-9._/%-]{1,}", text)
    tokens.extend([w.lower() for w in en_words])
    # 中文单字
    cn_chars = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.extend(cn_chars)
    # n-gram 字符级组合
    pure_cn = re.sub(r"[^\u4e00-\u9fff]", "", text)
    for i in range(len(pure_cn) - n_gram + 1):
        tokens.append(pure_cn[i:i + n_gram])
    return tokens


def _normalize(text: str) -> str:
    """文本归一化：去标点、统一空白、转小写"""
    if not text:
        return ""
    # 中英文标点统一去
    text = re.sub(r"[，。！？、；：「」『』（）《》【】…—·\s]+", " ", text)
    text = re.sub(r"[,!?;:()\[\]<>\"'`~@#$%^&*_+=\-\\/|<>\.]+", " ", text)
    return text.strip().lower()


# ============================================================
# 2. 关键词召回评测（多路召回检索结果评估）
# ============================================================
def calc_keyword_metrics(
    expected_keywords: List[str],
    answer_text: str,
) -> Dict[str, Any]:
    """
    计算关键词召回相关指标
    返回：{hit_count, total, recall, hit_keywords, miss_keywords}
    """
    expected_keywords = expected_keywords or []
    answer_norm = _normalize(answer_text)
    hit_keywords = []
    miss_keywords = []
    for kw in expected_keywords:
        kw_norm = _normalize(kw)
        if kw_norm and kw_norm in answer_norm:
            hit_keywords.append(kw)
        else:
            miss_keywords.append(kw)
    total = len(expected_keywords) if expected_keywords else 0
    recall = round(len(hit_keywords) / total, 4) if total > 0 else 0.0
    return {
        "hit_count": len(hit_keywords),
        "total_expected": total,
        "recall": recall,
        "hit_keywords": hit_keywords,
        "miss_keywords": miss_keywords,
    }


# ============================================================
# 3. 语义相似度评估（Jaccard + Overlap Coefficient）
# ============================================================
def calc_semantic_similarity(
    reference_text: str,
    answer_text: str,
) -> Dict[str, float]:
    """
    字符级语义相似度，避免依赖sentence-transformers等重型库
    返回：{jaccard, overlap, cosine_like}
    """
    ref_tokens = _tokenize_cn(reference_text or "", n_gram=2)
    ans_tokens = _tokenize_cn(answer_text or "", n_gram=2)
    ref_set = set(ref_tokens)
    ans_set = set(ans_tokens)

    intersection = ref_set & ans_set
    union = ref_set | ans_set

    jaccard = round(len(intersection) / len(union), 4) if union else 0.0
    overlap = round(
        len(intersection) / min(len(ref_set), len(ans_set)), 4
    ) if min(len(ref_set), len(ans_set)) > 0 else 0.0

    # 简化版余弦：基于词频向量点积
    ref_counter = Counter(ref_tokens)
    ans_counter = Counter(ans_tokens)
    common = set(ref_counter.keys()) & set(ans_counter.keys())
    dot = sum(ref_counter[t] * ans_counter[t] for t in common)
    norm_r = sum(v * v for v in ref_counter.values()) ** 0.5
    norm_a = sum(v * v for v in ans_counter.values()) ** 0.5
    cosine = round(dot / (norm_r * norm_a), 4) if norm_r and norm_a else 0.0

    # 综合得分（加权）
    score = round(0.4 * jaccard + 0.3 * overlap + 0.3 * cosine, 4)
    return {
        "jaccard": jaccard,
        "overlap_coefficient": overlap,
        "cosine_like": cosine,
        "composite_score": score,
    }


# ============================================================
# 4. 回答忠实度（Faithfulness）检测
# ============================================================
# 忠实度 = 回答中信息是否能在"召回知识片段/参考答案"中被支持
HALLUCINATION_PATTERNS: List[Tuple[str, str]] = [
    # 明显的编造数值 / 绝对化断言
    (r"(绝对|100%|完全|肯定|必定|一定能|绝无|从未有|全球第一|行业第一|史上最)", "extreme_claim"),
    # 捏造引用源
    (r"(根据最新研究|专家指出|官方证实|权威报道|某知名机构|有研究表明|论文显示)[^，。]{0,10}(未|没有|不)[^，。]{0,10}(提|说|写|出|证)", "fabricated_source"),
    # 离谱数字组合（非真实跨境常识区间）
    (r"(运费|价格|费用|税率|关税)[^0-9]{0,6}(99999|0\.000\d|99\.99%|999%|1000%|\d{7,})", "absurd_numeric"),
    # 自相矛盾
    (r"(同时|但是|然而|不过|实际上)[^。]{0,20}(不|不能|禁止|不可以|不行)[^。]{0,20}(可以|允许|没问题|可以|行)", "self_contradiction"),
]

FAITHFUL_SUPPORT_PHRASES: List[str] = [
    "根据", "依据", "按照", "参考", "政策规定", "法规要求", "条款",
    "官方", "平台", "公告", "指南", "规则", "标准", "手册",
]


def calc_faithfulness(
    answer_text: str,
    reference_text: str,
    retrieved_contexts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    评估回答忠实度：回答内容是否有据可查
    返回：{faithfulness_score, supported_ratio, unsupported_claims, risk_flag}
    """
    ans_norm = _normalize(answer_text)
    ref_norm = _normalize(reference_text)
    contexts_joined = _normalize(" ".join(retrieved_contexts or []))

    if not ans_norm:
        return {"faithfulness_score": 0.0, "reason": "空回答"}

    # 按中文句号/逗号分句
    ans_sentences = [s.strip() for s in re.split(r"[。！？；.;!?]", answer_text) if s.strip()]
    support_bases = ref_norm + contexts_joined
    supported = 0
    unsupported_claims: List[str] = []

    for sent in ans_sentences:
        sent_short = _normalize(sent)
        if len(sent_short) <= 2:
            continue
        # 检查支持：句子中>=40%的n-gram出现在参考材料中，或出现"根据...政策"等引用短语
        sent_tokens = set(_tokenize_cn(sent, n_gram=2))
        base_tokens = set(_tokenize_cn(support_bases, n_gram=2))
        overlap_ratio = (
            len(sent_tokens & base_tokens) / len(sent_tokens)
            if sent_tokens else 0.0
        )
        has_explicit_ref = any(p in sent for p in FAITHFUL_SUPPORT_PHRASES)
        if overlap_ratio >= 0.4 or has_explicit_ref:
            supported += 1
        else:
            unsupported_claims.append(sent[:40] + ("..." if len(sent) > 40 else ""))

    total = len([s for s in ans_sentences if len(_normalize(s)) > 2])
    supported_ratio = round(supported / total, 4) if total > 0 else 0.0

    # 基础忠实度分
    base_score = 0.4 * supported_ratio

    # 加分：有引用短语加0.1，n-gram整体覆盖超过0.3加0.2
    overall_cover = calc_semantic_similarity(reference_text, answer_text)["composite_score"]
    base_score += 0.2 * overall_cover
    base_score += 0.1 if any(p in ans_norm for p in [_normalize(p) for p in FAITHFUL_SUPPORT_PHRASES]) else 0.0

    # 减分：命中幻觉模式每条扣0.1，上限扣0.4
    penalty = 0.0
    for pattern, _tag in HALLUCINATION_PATTERNS:
        if re.search(pattern, answer_text):
            penalty += 0.1
    penalty = min(penalty, 0.4)

    faithfulness = round(max(0.0, min(1.0, base_score + 0.3 - penalty)), 4)

    return {
        "faithfulness_score": faithfulness,
        "supported_ratio": supported_ratio,
        "supported_count": supported,
        "total_claims": total,
        "unsupported_claims": unsupported_claims,
        "penalty_from_patterns": round(penalty, 4),
    }


# ============================================================
# 5. 幻觉风险检测（Hallucination Risk）
# ============================================================
def detect_hallucination_risk(
    question: str,
    answer_text: str,
    expected_keywords: List[str],
    expected_entities: Optional[Dict[str, str]] = None,
    reference_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    幻觉风险分级评分
    返回：{risk_level: high/medium/low/none, risk_score, reasons, details}
    """
    reasons: List[str] = []
    risk_score = 0.0  # 0~1，越高越危险

    # ① 关键词严重丢失（命中关键词<30%） -> 可能瞎编跑题
    kw_metrics = calc_keyword_metrics(expected_keywords, answer_text)
    if kw_metrics["total_expected"] > 0 and kw_metrics["recall"] < 0.3:
        risk_score += 0.3
        reasons.append(
            f"关键信息覆盖率过低({kw_metrics['recall']*100:.0f}%)，缺关键词："
            + "、".join(kw_metrics["miss_keywords"][:5])
        )

    # ② 实体错位/不符：expected_entities中的关键实体未出现或出现错配
    expected_entities = expected_entities or {}
    for slot, val in expected_entities.items():
        val_norm = _normalize(str(val))
        ans_norm = _normalize(answer_text)
        if val_norm and len(val_norm) >= 2 and val_norm not in ans_norm:
            # 再模糊匹配一下纯中文数字
            risk_score += 0.12
            reasons.append(f"实体[{slot}={val}]在回答中缺失")

    # ③ 命中幻觉正则特征
    matched_tags: List[str] = []
    for pattern, tag in HALLUCINATION_PATTERNS:
        if re.search(pattern, answer_text):
            matched_tags.append(tag)
            risk_score += 0.08
    if matched_tags:
        reasons.append(f"命中幻觉风险特征模式：{'、'.join(matched_tags)}")

    # ④ 回答明显跑题：与问题核心词n-gram重合度<0.08
    q_tokens = set(_tokenize_cn(question, n_gram=2))
    a_tokens = set(_tokenize_cn(answer_text, n_gram=2))
    qna_overlap = (
        len(q_tokens & a_tokens) / len(q_tokens) if q_tokens else 0.0
    )
    if qna_overlap < 0.08 and len(question) >= 6:
        risk_score += 0.2
        reasons.append(f"回答与问题重合度过低({qna_overlap:.2f})，可能跑题")

    # ⑤ 与参考回答差异巨大
    if reference_text:
        sim = calc_semantic_similarity(reference_text, answer_text)["composite_score"]
        if sim < 0.15:
            risk_score += 0.2
            reasons.append(f"与参考答案相似度仅{sim:.2f}，存在严重信息偏差")

    risk_score = round(min(1.0, risk_score), 4)

    if risk_score >= 0.55:
        risk_level = "high"
    elif risk_score >= 0.3:
        risk_level = "medium"
    elif risk_score >= 0.1:
        risk_level = "low"
    else:
        risk_level = "none"

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "reasons": reasons,
        "keyword_coverage_recall": kw_metrics["recall"],
        "question_answer_overlap": round(qna_overlap, 4),
    }


# ============================================================
# 6. 实体对齐校验（Entity Alignment）
# ============================================================
def verify_entity_alignment(
    expected_entities: Dict[str, Any],
    answer_text: str,
) -> Dict[str, Any]:
    """
    校验回答中关键实体是否正确
    返回：{aligned_count, total, accuracy, per_slot_results}
    """
    expected_entities = expected_entities or {}
    ans_norm = _normalize(answer_text)
    per_slot: Dict[str, Any] = {}
    aligned = 0
    total = 0

    for slot, val in expected_entities.items():
        total += 1
        if isinstance(val, list):
            # 槽位允许多个候选值，命中任意一个即算正确
            hit = any(_normalize(str(v)) in ans_norm for v in val if _normalize(str(v)))
        else:
            val_str = _normalize(str(val))
            # 兼容纯数字（如"500g"→回答里"500克"也算命中）
            nums_in_val = re.findall(r"\d+", val_str)
            nums_in_ans = re.findall(r"\d+", ans_norm)
            number_match = bool(nums_in_val and set(nums_in_val).issubset(set(nums_in_ans)))
            direct_match = bool(val_str) and val_str in ans_norm
            hit = direct_match or number_match
        per_slot[slot] = {"expected": val, "hit": hit}
        aligned += 1 if hit else 0

    accuracy = round(aligned / total, 4) if total > 0 else 1.0
    return {
        "aligned_count": aligned,
        "total_slots": total,
        "accuracy": accuracy,
        "per_slot_results": per_slot,
    }


# ============================================================
# 7. 综合问答准确率评估（最终综合分）
# ============================================================
def evaluate_qa_case(
    question: str,
    answer_text: str,
    reference_text: str,
    expected_keywords: Optional[List[str]] = None,
    expected_entities: Optional[Dict[str, Any]] = None,
    retrieved_contexts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    综合评测单条问答Case
    指标加权：
      - 关键词召回（25%）
      - 语义相似度（20%）
      - 忠实度（25%）
      - 实体准确率（20%）
      - 幻觉惩罚（10%倒扣）
    """
    expected_keywords = expected_keywords or []
    expected_entities = expected_entities or {}

    kw = calc_keyword_metrics(expected_keywords, answer_text)
    sim = calc_semantic_similarity(reference_text, answer_text)
    faith = calc_faithfulness(answer_text, reference_text, retrieved_contexts)
    entity = verify_entity_alignment(expected_entities, answer_text)
    hall = detect_hallucination_risk(
        question, answer_text, expected_keywords, expected_entities, reference_text
    )

    weighted = (
        0.25 * kw["recall"]
        + 0.20 * sim["composite_score"]
        + 0.25 * faith["faithfulness_score"]
        + 0.20 * entity["accuracy"]
        + 0.10 * (1.0 - hall["risk_score"])  # 幻觉严重直接扣
    )
    accuracy_score = round(max(0.0, min(1.0, weighted)), 4)

    # 判定是否通过（综合分>=0.6 且非高风险幻觉）
    passed = accuracy_score >= 0.6 and hall["risk_level"] != "high"

    return {
        "accuracy_score": accuracy_score,
        "passed": passed,
        "grade": _grade(accuracy_score),
        "keyword_metrics": kw,
        "semantic_similarity": sim,
        "faithfulness": faith,
        "entity_alignment": entity,
        "hallucination": hall,
    }


def _grade(score: float) -> str:
    """分数转字母等级"""
    if score >= 0.90:
        return "S"
    if score >= 0.80:
        return "A"
    if score >= 0.70:
        return "B"
    if score >= 0.60:
        return "C"
    if score >= 0.40:
        return "D"
    return "F"


# ============================================================
# 8. 汇总统计报告工具
# ============================================================
def build_summary_report(
    all_case_results: List[Dict[str, Any]],
    start_ts: float,
    end_ts: float,
) -> Dict[str, Any]:
    """
    根据批量Case结果构建汇总报告
    """
    total = len(all_case_results)
    passed = sum(1 for r in all_case_results if r.get("passed"))
    failed = total - passed
    pass_rate = round(passed / total, 4) if total > 0 else 0.0

    # 平均各指标
    avg_accuracy = round(
        sum(r.get("accuracy_score", 0) for r in all_case_results) / total, 4
    ) if total else 0.0
    avg_kw_recall = round(
        sum(r["keyword_metrics"]["recall"] for r in all_case_results if "keyword_metrics" in r) / total, 4
    ) if total else 0.0
    avg_sim = round(
        sum(r["semantic_similarity"]["composite_score"] for r in all_case_results if "semantic_similarity" in r) / total, 4
    ) if total else 0.0
    avg_faith = round(
        sum(r["faithfulness"]["faithfulness_score"] for r in all_case_results if "faithfulness" in r) / total, 4
    ) if total else 0.0
    avg_entity_acc = round(
        sum(r["entity_alignment"]["accuracy"] for r in all_case_results if "entity_alignment" in r) / total, 4
    ) if total else 0.0

    # 幻觉分布
    hall_counts = {"high": 0, "medium": 0, "low": 0, "none": 0}
    for r in all_case_results:
        lvl = r.get("hallucination", {}).get("risk_level", "none")
        hall_counts[lvl] = hall_counts.get(lvl, 0) + 1

    # 接口响应耗时统计
    latencies = [r.get("latency_ms", 0) for r in all_case_results if r.get("latency_ms")]
    latency_stats = {}
    if latencies:
        latencies_sorted = sorted(latencies)
        latency_stats = {
            "count": len(latencies),
            "avg_ms": round(sum(latencies) / len(latencies), 2),
            "min_ms": latencies_sorted[0],
            "max_ms": latencies_sorted[-1],
            "p50_ms": latencies_sorted[len(latencies_sorted) // 2],
            "p95_ms": latencies_sorted[int(len(latencies_sorted) * 0.95)],
        }

    # 分类通过率
    category_stats: Dict[str, Dict[str, int]] = {}
    for r in all_case_results:
        cat = r.get("category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "passed": 0, "failed": 0}
        category_stats[cat]["total"] += 1
        if r.get("passed"):
            category_stats[cat]["passed"] += 1
        else:
            category_stats[cat]["failed"] += 1
    for cat in category_stats.values():
        cat["pass_rate"] = round(cat["passed"] / cat["total"], 4) if cat["total"] else 0.0

    # 难度统计
    difficulty_stats: Dict[str, Dict[str, int]] = {}
    for r in all_case_results:
        diff = r.get("difficulty", "unknown")
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {"total": 0, "passed": 0, "failed": 0}
        difficulty_stats[diff]["total"] += 1
        if r.get("passed"):
            difficulty_stats[diff]["passed"] += 1
        else:
            difficulty_stats[diff]["failed"] += 1
    for diff in difficulty_stats.values():
        diff["pass_rate"] = round(diff["passed"] / diff["total"], 4) if diff["total"] else 0.0

    # 等级分布
    grade_counts = Counter(r.get("grade", "F") for r in all_case_results)

    return {
        "summary": {
            "total_cases": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": pass_rate,
            "pass_rate_percent": f"{round(pass_rate * 100, 2)}%",
            "avg_accuracy_score": avg_accuracy,
            "execution_seconds": round(end_ts - start_ts, 2),
        },
        "metrics_average": {
            "avg_keyword_recall": avg_kw_recall,
            "avg_semantic_similarity": avg_sim,
            "avg_faithfulness": avg_faith,
            "avg_entity_accuracy": avg_entity_acc,
        },
        "hallucination_distribution": {
            **hall_counts,
            "high_ratio": round(hall_counts["high"] / total, 4) if total else 0,
        },
        "grade_distribution": dict(grade_counts),
        "latency_statistics": latency_stats,
        "category_breakdown": category_stats,
        "difficulty_breakdown": difficulty_stats,
        "timestamp": {
            "start": start_ts,
            "end": end_ts,
        },
    }


# ============================================================
# 9. 控制台/文件格式化输出工具
# ============================================================
def format_report_console(report: Dict[str, Any]) -> str:
    """将汇总报告格式化为漂亮的控制台文本"""
    lines = []
    s = report["summary"]
    m = report["metrics_average"]
    h = report["hallucination_distribution"]
    l = report.get("latency_statistics", {})
    cat = report.get("category_breakdown", {})

    def _bar(ratio: float, width: int = 30) -> str:
        filled = int(ratio * width)
        return "█" * filled + "░" * (width - filled)

    lines.append("")
    lines.append("=" * 78)
    lines.append("📊 星途环球·跨境售后RAG智能问答系统 - 自动化测试汇总报告")
    lines.append("=" * 78)
    lines.append(f"  总用例数       : {s['total_cases']}")
    lines.append(f"  通过 / 失败    : {s['passed']} ✅  /  {s['failed']} ❌")
    lines.append(f"  通过率         : {s['pass_rate_percent']}  {_bar(s['pass_rate'])}")
    lines.append(f"  综合平均得分   : {round(s['avg_accuracy_score'] * 100, 2)} / 100")
    lines.append(f"  总执行耗时     : {s['execution_seconds']} 秒")
    lines.append("-" * 78)
    lines.append("  📐 核心指标均值：")
    lines.append(f"    关键词召回率  : {round(m['avg_keyword_recall'] * 100, 2)}%  {_bar(m['avg_keyword_recall'])}")
    lines.append(f"    语义相似度    : {round(m['avg_semantic_similarity'] * 100, 2)}%  {_bar(m['avg_semantic_similarity'])}")
    lines.append(f"    回答忠实度    : {round(m['avg_faithfulness'] * 100, 2)}%  {_bar(m['avg_faithfulness'])}")
    lines.append(f"    实体准确率    : {round(m['avg_entity_accuracy'] * 100, 2)}%  {_bar(m['avg_entity_accuracy'])}")
    lines.append("-" * 78)
    lines.append("  ⚠️  幻觉风险分布：")
    lines.append(f"    🔴 HIGH   : {h.get('high', 0)}  ({round(h.get('high_ratio', 0) * 100, 2)}%)")
    lines.append(f"    🟡 MEDIUM : {h.get('medium', 0)}")
    lines.append(f"    🟢 LOW    : {h.get('low', 0)}")
    lines.append(f"    ✅ NONE   : {h.get('none', 0)}")
    lines.append("-" * 78)
    if l:
        lines.append("  ⏱️  接口响应耗时(ms)：")
        lines.append(f"    平均: {l['avg_ms']}  |  P50: {l['p50_ms']}  |  P95: {l['p95_ms']}  |  MAX: {l['max_ms']}")
    lines.append("-" * 78)
    lines.append("  📁 分类通过率：")
    name_map = {
        "cross_border_logistics": "跨境物流咨询",
        "after_sales_policy": "海外售后政策",
        "customs_compliance": "各国海关合规政策",
    }
    for key, stats in cat.items():
        display = name_map.get(key, key)
        lines.append(
            f"    {display.ljust(18)}: {stats['passed']}/{stats['total']} "
            f"通过率 {round(stats['pass_rate'] * 100, 2)}%"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


if __name__ == "__main__":
    # 本地自检测试：运行 python -m tests.test_evaluate
    sample_q = "从深圳发500g包裹到纽约最便宜的物流方式是什么？"
    sample_ans_good = "500g包裹从深圳发纽约最便宜的是ePacket邮政小包，7-15天到货，运费约35-50元。"
    sample_ans_bad = "发火星最便宜的是光速飞船，100%当天到货，只收99999元。"
    sample_ref = "500g包裹深圳发纽约，推荐ePacket（邮政小包）最经济，运费35-50元，时效7-15工作日。"

    print("🔬 评测工具自检：好回答")
    r_good = evaluate_qa_case(
        sample_q, sample_ans_good, sample_ref,
        expected_keywords=["邮政小包", "ePacket", "7-15天"],
        expected_entities={"origin": "深圳", "destination": "纽约", "weight": "500g"},
    )
    print(json.dumps(r_good, ensure_ascii=False, indent=2))

    print("\n🔬 评测工具自检：差回答（幻觉）")
    r_bad = evaluate_qa_case(
        sample_q, sample_ans_bad, sample_ref,
        expected_keywords=["邮政小包", "ePacket", "7-15天"],
        expected_entities={"origin": "深圳", "destination": "纽约", "weight": "500g"},
    )
    print(json.dumps(r_bad, ensure_ascii=False, indent=2))
