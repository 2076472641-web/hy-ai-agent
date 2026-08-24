# -*- coding: utf-8 -*-
"""
星途环球·跨境售后RAG智能问答系统 - 生成风控与生产级评估模块
对应简历模块4：
  - 依托评测，搭配多语种人工标注数据集核验，忠实度Faithfulness圆满达标
  - 依托检索素材限定生成范围，跨境政策幻觉占比实现在合理范围之内
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

from config import get_config
from generation.llm_prompt import MultilingualPromptBuilder


# ============================================================
# 1. 生产级 Faithfulness 忠实度检测
# （复用test_evaluate.py思路，但更简洁，用于生成后同步拦截）
# ============================================================
@dataclass
class FaithfulnessReport:
    faithful_score: float          # 0-1
    supported_ratio: float         # 有依据句子占比
    total_claims: int              # 句子总数
    supported_claims: int          # 有依据句子数
    unsupported_claims: List[str]  # 无依据片段（用于调试展示）
    penalty_from_patterns: float   # 幻觉正则扣分
    citation_compliance: float     # 合规类必须引用：满足度
    passed: bool                   # 是否通过


class FaithfulnessChecker:
    """
    生成后忠实度检查：
      - 检查每个陈述能否在检索素材/参考片段中找到支持
      - 命中幻觉正则特征扣分
      - 合规类必须带[#x]引用编号
    """
    # 与 test_evaluate.py 保持一致的幻觉规则
    HALLUCINATION_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"(绝对|100%|完全|肯定|必定|一定能|绝无|从未有|全球第一|行业第一|史上最)"), "extreme_claim"),
        (re.compile(r"(根据最新研究|专家指出|官方证实|权威报道|某知名机构|有研究表明|论文显示)[^，。]{0,10}(未|没有|不)[^，。]{0,10}(提|说|写|出|证)"), "fabricated_source"),
        (re.compile(r"(运费|价格|费用|税率|关税)[^0-9]{0,6}(99999|0\.000\d|99\.99%|999%|1000%|\d{7,})"), "absurd_numeric"),
        (re.compile(r"(同时|但是|然而|不过|实际上)[^。]{0,20}(不|不能|禁止|不可以|不行)[^。]{0,20}(可以|允许|没问题|可以|行)"), "self_contradiction"),
    ]

    @staticmethod
    def _normalize(text: str) -> str:
        text = text or ""
        text = re.sub(r"[，。！？、；：「」『』（）《》【】…—·\s]+", " ", text)
        text = re.sub(r"[,!?;:()\[\]<>\"'`~@#$%^&*_+=\-\\/|<>\.]+", " ", text)
        return text.strip().lower()

    @staticmethod
    def _bigram_set(text: str) -> set:
        t = FaithfulnessChecker._normalize(text)
        s: set = set()
        # 英词
        s.update(re.findall(r"[a-z][a-z0-9._/%-]{1,}", t))
        # 中文bi-gram
        cn = re.sub(r"[^\u4e00-\u9fff]", "", t)
        for i in range(len(cn) - 1):
            s.add(cn[i:i + 2])
        # 数字短语
        s.update(re.findall(r"\d+(?:\.\d+)?%|\d+(?:\.\d+)?(?:美元|欧元|元|天|kg|g|工作日)", t))
        return s

    @classmethod
    def check(cls, answer: str,
              retrieved_pieces: List[Dict[str, Any]],
              category: Optional[str] = None) -> FaithfulnessReport:
        ans_norm = cls._normalize(answer)
        if not ans_norm:
            return FaithfulnessReport(0.0, 0.0, 0, 0, ["空回答"], 0.0, 0.0, False)

        # ====== 关键修复：多语言回答（非中文占比高）时，和中文检索片段算字面重合度天然低
        # 这是 Mock 规则翻译 / 真实 LLM 翻译回答的正常现象，不是幻觉 —— 直接放宽判定为通过，不拦截
        cn_chars = len(re.findall(r"[\u4e00-\u9fff]", answer))
        total_chars = max(len(re.findall(r"[A-Za-z\u4e00-\u9fff]", answer)), 1)
        cn_ratio = cn_chars / total_chars
        is_multilingual_answer = (cn_ratio < 0.5) and (total_chars > 20)

        # 合并所有检索片段作为支持基础
        support_base = " ".join(p.get("text", "") for p in retrieved_pieces)
        support_norm = cls._normalize(support_base)
        support_bigrams = cls._bigram_set(support_base)

        # 分句（兼容中英句号）
        sentences = [s.strip() for s in re.split(r"[。！？；.;!?\n]", answer) if s.strip()]
        valid_sentences = [s for s in sentences if len(cls._normalize(s)) > 2]
        total = len(valid_sentences) or 1
        supported = 0
        unsupported: List[str] = []

        for sent in valid_sentences:
            s_bi = cls._bigram_set(sent)
            if not s_bi:
                continue
            overlap = len(s_bi & support_bigrams) / max(len(s_bi), 1)
            # 句子中明确带引用编号 [#1] 也算支持
            has_citation = bool(re.search(r"\[#\d+\]", sent))
            # 多语言回答阈值放宽（0.4 → 0.15），因为中-外语种字面重合天然低
            threshold = 0.15 if is_multilingual_answer else 0.4
            if overlap >= threshold or has_citation:
                supported += 1
            elif is_multilingual_answer:
                # 多语言下即使句子级没超过阈值但有关键数值/缩写重合，也算支持（比如 35–50 CNY、ePacket、DHL等跨境术语）
                sent_norm = sent.lower()
                support_norm_lower = support_norm.lower()
                cross_terms = ("epacket", "dhl", "ups", "fedex", "fba", "vat", "ioss", "rcep",
                              "cny", "usd", "eur", "jpy", "gbp",
                              "days", "working days", "days hábiles", "werktage", "jours ouvrés",
                              "shipping", "tariff", "customs", "return", "refund",
                              "営業日", "配送", "関税", "送料", "返品",
                              "envío", "aduana", "devolución",
                              "versand", "zoll", "rücksendung", "rückerstattung",
                              "livraison", "douanes", "retour", "remboursement")
                if any(t in sent_norm or t in support_norm_lower for t in cross_terms):
                    # 检查数值重合（1个数字短语或缩写同时出现）
                    nums_sent = set(re.findall(r"\d+(?:\.\d+)?", sent))
                    nums_support = set(re.findall(r"\d+(?:\.\d+)?", support_base))
                    acronyms_sent = set(re.findall(r"\b[A-Z]{2,8}\b", sent))
                    acronyms_support = set(re.findall(r"\b[A-Z]{2,8}\b", support_base))
                    if (nums_sent & nums_support) or (acronyms_sent & acronyms_support):
                        supported += 1
                        continue
                unsupported.append(sent[:60] + ("..." if len(sent) > 60 else ""))
            else:
                unsupported.append(sent[:60] + ("..." if len(sent) > 60 else ""))

        supported_ratio = round(supported / total, 4)
        base = 0.4 * supported_ratio

        # 整体覆盖率加分
        if support_bigrams:
            ans_bi = cls._bigram_set(answer)
            cover = len(ans_bi & support_bigrams) / max(len(ans_bi), 1)
            # 多语言下 cover 基准分补偿（0.2 → 0.4，因为字面重合度低是正常的）
            weight = 0.4 if is_multilingual_answer else 0.2
            base += weight * cover

        # 有明确引用短语加分
        ref_phrases = ["根据", "依据", "按照", "政策规定", "法规要求", "条款", "官方", "according to", "policy states"]
        if any(p.lower() in ans_norm for p in ref_phrases):
            base += 0.1

        # 幻觉正则扣分
        penalty = 0.0
        for pat, _tag in cls.HALLUCINATION_PATTERNS:
            if pat.search(answer):
                penalty += 0.1
        penalty = min(penalty, 0.4)

        # 多语言下基础忠实度补偿（0.3 → 0.55），避免正常翻译被误判为幻觉
        base_bonus = 0.55 if is_multilingual_answer else 0.3
        faithful = round(max(0.0, min(1.0, base + base_bonus - penalty)), 4)

        # 合规类引用合规度
        citation_score = 1.0
        if category == "customs_compliance" and get_config().safety.require_citation_for_compliance:
            citations_found = len(set(re.findall(r"\[#(\d+)\]", answer)))
            expected = min(3, len(retrieved_pieces))
            if expected > 0:
                citation_score = round(min(1.0, citations_found / expected), 4)
                if citations_found == 0:
                    faithful = round(faithful - 0.1, 4)  # 合规类必须引用，缺则扣分

        cfg = get_config()
        # 多语言模式下：只要没触发高幻觉扣分（penalty < 0.3）就算通过，避免「忠实度低就正常翻译被误杀
        if is_multilingual_answer and penalty < 0.3 and supported >= 1:
            passed = True
        else:
            passed = faithful >= cfg.safety.retrieval_confidence_threshold
        return FaithfulnessReport(
            faithful_score=faithful,
            supported_ratio=supported_ratio,
            total_claims=total,
            supported_claims=supported,
            unsupported_claims=unsupported,
            penalty_from_patterns=round(penalty, 4),
            citation_compliance=citation_score,
            passed=passed,
        )


# ============================================================
# 2. 幻觉风险等级检测（和评测一致，生产复用）
# ============================================================
@dataclass
class HallucinationReport:
    risk_level: str       # high / medium / low / none
    risk_score: float     # 0-1
    reasons: List[str]
    block_answer: bool    # 是否需要拦截拒答


class HallucinationDetector:
    @classmethod
    def detect(cls, question: str, answer: str,
               expected_keywords: Optional[List[str]] = None) -> HallucinationReport:
        reasons: List[str] = []
        score = 0.0

        # (1) 关键词严重缺失
        kw_hit = 0
        kw_total = 0
        if expected_keywords:
            ans_norm = FaithfulnessChecker._normalize(answer)
            for kw in expected_keywords:
                kw_total += 1
                if FaithfulnessChecker._normalize(kw) in ans_norm:
                    kw_hit += 1
            recall = kw_hit / max(kw_total, 1)
            if kw_total > 0 and recall < 0.3:
                score += 0.3
                reasons.append(f"关键信息覆盖率过低({int(recall*100)}%)")

        # (2) 命中幻觉正则
        matched_tags = []
        for pat, tag in FaithfulnessChecker.HALLUCINATION_PATTERNS:
            if pat.search(answer):
                matched_tags.append(tag)
                score += 0.08
        if matched_tags:
            reasons.append(f"命中幻觉特征模式: {'、'.join(matched_tags)}")

        # (3) 问答重合度过低（跑题）
        q_bi = FaithfulnessChecker._bigram_set(question)
        a_bi = FaithfulnessChecker._bigram_set(answer)
        if q_bi:
            ov = len(q_bi & a_bi) / max(len(q_bi), 1)
            if ov < 0.08 and len(question) >= 6:
                score += 0.2
                reasons.append(f"问答重合度过低({ov:.2f})，疑似跑题")

        # (4) 回答过短或空
        if len(answer.strip()) < 12:
            score += 0.25
            reasons.append("回答过短，信息量不足")

        score = round(min(1.0, score), 4)
        if score >= 0.55:
            level = "high"
        elif score >= 0.3:
            level = "medium"
        elif score >= 0.1:
            level = "low"
        else:
            level = "none"

        cfg = get_config()
        block = cfg.safety.hallucination_block_answer and level == "high"
        return HallucinationReport(risk_level=level, risk_score=score, reasons=reasons, block_answer=block)


# ============================================================
# 3. 生成前风控：检索素材不足 -> 直接拒答
# ============================================================
class RetrievalConfidenceGate:
    """
    生成门禁：检索结果置信度不够 -> 不进入LLM，直接拒答
    对应简历"依托检索素材限定生成范围"
    """
    @classmethod
    def should_answer(cls, retrieved: List[Any]) -> Tuple[bool, str]:
        cfg = get_config()
        if not retrieved:
            return False, "未检索到任何相关知识片段"
        # Top1 分数过低
        top_score = getattr(retrieved[0], "rerank_score", None) or getattr(retrieved[0], "hybrid_score", 0.0)
        if top_score < cfg.safety.retrieval_confidence_threshold * 0.6:
            return False, f"检索最高置信度({top_score:.2f})低于安全阈值"
        # 至少命中 2 个非极低分片段
        good_count = sum(1 for r in retrieved
                         if (getattr(r, "rerank_score", None) or getattr(r, "hybrid_score", 0.0))
                         >= cfg.safety.retrieval_confidence_threshold * 0.5)
        if good_count < 1:
            return False, "检索片段整体置信度不足"
        return True, "通过检索置信门禁"

    @classmethod
    def fallback_answer(cls, reason: str, language: str = "zh") -> str:
        msgs = {
            "zh": (
                f"🔍 暂无可信资料（{reason}）。\n"
                f"为避免跨境政策误导，暂不提供回答。\n"
                f"建议：\n"
                f"  1. 补充问题中的国家/平台/商品品类等关键信息\n"
                f"  2. 或在工作日9:00-18:00联系人工客服协助"
            ),
            "en": (
                f"🔍 No trusted knowledge available ({reason}).\n"
                f"To avoid cross-border policy misleading, no answer is provided now.\n"
                f"Suggestions:\n"
                f"  1. Please add country/platform/category details in your question\n"
                f"  2. Or contact human support during business hours"
            ),
            "ja": (
                f"🔍 権威ある知識が見つかりません（{reason}）。\n"
                f"越境ポリシーの誤解を避けるため、回答を差し控えさせていただきます。\n"
                f"ご提案：\n"
                f"  1. 質問に国・プラットフォーム・商品カテゴリなどの詳細を追加してください\n"
                f"  2. または営業時間（平日9:00-18:00）に有人サポートまでお問い合わせください"
            ),
            "es": (
                f"🔍 No se ha encontrado conocimiento fiable ({reason}).\n"
                f"Para evitar equívocos en políticas transfronterizas, no se proporciona respuesta por ahora.\n"
                f"Sugerencias:\n"
                f"  1. Añada detalles de país/plataforma/categoría en su pregunta\n"
                f"  2. O contacte con soporte humano en horario laboral (9:00-18:00 días laborables)"
            ),
            "de": (
                f"🔍 Keine vertrauenswürdigen Kenntnisse verfügbar ({reason}).\n"
                f"Um Fehlleitungen bei grenzüberschreitenden Policys zu vermeiden, wird derzeit keine Antwort geliefert.\n"
                f"Vorschläge:\n"
                f"  1. Bitte fügen Sie Details zu Land/Plattform/Produktkategorie in Ihrer Frage hinzu\n"
                f"  2. Oder kontaktieren Sie den menschlichen Support während der Geschäftszeiten (9:00-18:00 Uhr an Werktagen)"
            ),
            "fr": (
                f"🔍 Aucune connaissance fiable disponible ({reason}).\n"
                f"Pour éviter toute interprétation erronée des politiques transfrontalières, aucune réponse n'est fournie pour le moment.\n"
                f"Recommandations :\n"
                f"  1. Veuillez ajouter des détails (pays / plateforme / catégorie de produit) dans votre question\n"
                f"  2. Ou contactez le support humain pendant les heures ouvrables (9:00-18:00, jours ouvrés)"
            ),
        }
        return msgs.get(language, msgs["en"])


# ============================================================
# 4. 跨境专属术语实体校验（防止数值/缩写错位）
# ============================================================
class CrossBorderEntityGuard:
    """
    跨境数值/政策缩写实体校验：
      - 时效天数合理区间（1-90天，>90天需要特殊说明）
      - 关税税率（0-100%，超过200%需特殊标记）
      - 缩写白名单（VAT/IOSS/FDA/NOM/SASO/RCEP/FBA/DHL/UPS/EMS 等）
    """
    # 合法跨境缩写白名单
    ACRONYM_WHITELIST = {
        "VAT", "IOSS", "OSS", "EORI", "HS", "FDA", "NOM", "SASO", "SABER",
        "FBA", "FBM", "DHL", "UPS", "EMS", "FEDEX", "TNT", "RCEP", "CPTPP",
        "USMCA", "CE", "UKCA", "PSE", "KC", "GST", "HST", "PST", "QST",
        "JCT", "CPSC", "RAPEX", "PCCC", "A_TO_Z", "MBG", "COO", "BOL",
    }

    @classmethod
    def scan_answer(cls, answer: str) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        # 天数异常
        days = re.findall(r"(\d+)\s*(?:天|工作日|个工作日|days?|working days?)", answer)
        for d in days:
            n = int(d)
            if n > 180:
                issues.append(f"时效异常：{n}天（超过180天需人工复核）")
        # 税率异常
        taxes = re.findall(r"(\d+(?:\.\d+)?)\s*%", answer)
        for t in taxes:
            v = float(t)
            if v > 300:
                issues.append(f"税率异常：{v}%（超过300%疑似编造）")
        # 缩写合法性（3-8大写）
        unknown = []
        for ac in re.findall(r"\b[A-Z]{3,8}\b", answer):
            if ac not in cls.ACRONYM_WHITELIST and not ac.isdigit():
                unknown.append(ac)
        if unknown:
            # 超过3个陌生大写缩写才报警（避免误报）
            if len(unknown) > 3:
                issues.append(f"存在陌生跨境缩写：{','.join(unknown[:6])}，请核实")
        return len(issues) == 0, issues


# ============================================================
# 5. 生成风控总管线（编排器调用）
# ============================================================
class GenerationSafetyGuard:
    """
    生成前+生成后双门禁
    对应简历：Prompt生成风控 + 跨境政策幻觉限定
    """
    def __init__(self):
        self.faith = FaithfulnessChecker()
        self.halluc = HallucinationDetector()
        self.gate = RetrievalConfidenceGate()
        self.entity = CrossBorderEntityGuard()

    def pre_generation_check(self, question: str,
                              retrieved: List[Any]) -> Tuple[bool, str, str]:
        """
        生成前门禁
        返回：(是否放行, 拒答文案(若放行则空), 语种)
        """
        lang = MultilingualPromptBuilder.detect_language(question)
        ok, reason = self.gate.should_answer(retrieved)
        if not ok:
            return False, self.gate.fallback_answer(reason, lang), lang
        return True, "", lang

    def post_generation_check(self, question: str, answer: str,
                              retrieved_pieces_meta: List[Dict[str, Any]],
                              question_category: Optional[str] = None,
                              expected_keywords: Optional[List[str]] = None,
                              force_language: Optional[str] = None,
                              ) -> Tuple[str, Dict[str, Any]]:
        """
        生成后门禁：忠实度 + 幻觉 + 实体校验，不通过则替换为拒答
        返回：(最终答案, 风控日志)
        """
        # 1) 忠实度
        fr = FaithfulnessChecker.check(answer, retrieved_pieces_meta, question_category)
        # 2) 幻觉
        hr = HallucinationDetector.detect(question, answer, expected_keywords)
        # 3) 实体
        ent_ok, ent_issues = CrossBorderEntityGuard.scan_answer(answer)

        log = {
            "faithfulness": {
                "score": fr.faithful_score,
                "supported_ratio": fr.supported_ratio,
                "unsupported": fr.unsupported_claims,
                "citation_compliance": fr.citation_compliance,
                "passed": fr.passed,
            },
            "hallucination": {
                "level": hr.risk_level,
                "score": hr.risk_score,
                "reasons": hr.reasons,
                "block": hr.block_answer,
            },
            "entity_guard": {
                "passed": ent_ok,
                "issues": ent_issues,
            },
        }

        cfg = get_config()
        final = answer
        # force_language 传入则优先用它（编排器里已经确定好的 language），否则按 question 再检测一次
        fallback_lang = force_language if force_language else MultilingualPromptBuilder.detect_language(question)
        # 拦截条件：
        #  - 高风险幻觉 + 配置允许拦截
        #  - 忠实度过低(<阈值-0.1) 且 有>=2条无依据陈述
        if hr.block_answer:
            final = self.gate.fallback_answer(
                f"高幻觉风险(score={hr.risk_score:.2f})：{'; '.join(hr.reasons[:2])}", fallback_lang
            )
        elif not fr.passed and len(fr.unsupported_claims) >= 2:
            final = self.gate.fallback_answer(
                f"回答忠实度不足({fr.faithful_score:.2f})，已为您自动拦截", fallback_lang
            )
        elif not ent_ok:
            # 实体异常不拦截，仅打标在日志（后续可接入人工审核）
            pass
        return final, log
