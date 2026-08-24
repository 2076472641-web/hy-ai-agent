# -*- coding: utf-8 -*-
"""
星途环球·跨境售后RAG智能问答系统 - 网关编排器
串联 检索 -> 重排序 -> Prompt构建 -> LLM生成 -> 生成后风控 全链路
对应简历：模型训练与参数调优（全链路落地）
"""
from __future__ import annotations

import os
import json
import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from config import get_config, KNOWLEDGE_CATEGORIES
from data_processing import build_chunks_from_json_cases, KnowledgeChunk
from retrieval import HybridRetriever, VectorSearchResult
from generation import (
    LLMFactory,
    RerankerFactory,
    MultilingualPromptBuilder,
    BaseLLM,
    BaseReranker,
)
from safety import GenerationSafetyGuard


# ============================================================
# 编排器内部状态结构
# ============================================================
@dataclass
class PipelineTrace:
    """全链路Trace记录（用于问题排查/面试演示可观测性）"""
    question: str = ""
    language: str = "zh"
    category: Optional[str] = None
    retrieval_top_k: int = 0
    retrieval_results_count: int = 0
    rerank_top_k: int = 0
    rerank_results_count: int = 0
    retrieval_scores: Dict[str, float] = field(default_factory=dict)
    prompt_build_ok: bool = True
    llm_latency_ms: float = 0.0
    safety_passed: bool = True
    safety_log: Dict[str, Any] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    answer_truncated: str = ""


class AgentOrchestrator:
    """
    RAG 全链路编排器
    使用示例：
        orch = AgentOrchestrator()
        result = await orch.handle_user_request("发美国500g包裹最便宜的物流？", session_id="test1")
        print(result["response"])
    """
    _singleton_instance: Optional["AgentOrchestrator"] = None

    def __init__(self, auto_bootstrap: bool = True):
        self.cfg = get_config()
        self.retriever: HybridRetriever = HybridRetriever()
        self.reranker: BaseReranker = RerankerFactory.create()
        self.llm: BaseLLM = LLMFactory.create()
        self.safety: GenerationSafetyGuard = GenerationSafetyGuard()
        self._bootstrapped = False
        self._bootstrap_in_progress = False
        if auto_bootstrap:
            self._bootstrap_if_needed()

    @classmethod
    def get_instance(cls, auto_bootstrap: bool = True) -> "AgentOrchestrator":
        if cls._singleton_instance is None:
            cls._singleton_instance = AgentOrchestrator(auto_bootstrap=auto_bootstrap)
        return cls._singleton_instance

    async def bootstrap_async(self) -> int:
        """
        异步版本的知识库初始化（跑在后台线程，不阻塞 FastAPI 事件循环）
        启动快：服务先响应浏览器请求 -> 后台慢慢灌51组种子知识
        返回：本次新增注入的chunk数
        """
        if self._bootstrapped or self._bootstrap_in_progress:
            return 0
        self._bootstrap_in_progress = True
        try:
            loop = asyncio.get_running_loop()
            # 把同步阻塞的注入逻辑丢到线程池里跑，不卡事件循环
            added = await loop.run_in_executor(None, self._bootstrap_if_needed)
            return added or 0
        except Exception as exc:
            print(f"[Orchestrator] 异步初始化异常(忽略): {exc}")
            return 0
        finally:
            self._bootstrap_in_progress = False

    # -------- 知识库初始化（首次运行自动从 test_cases.json 灌种子）--------
    def _bootstrap_if_needed(self):
        if self.retriever.total_indexed > 0:
            self._bootstrapped = True
            return 0
        added_total = 0
        try:  # 🔥 最外层兜底：任何致命错误(MemoryError/UnicodeError)都不能阻止 bootstrapped 置位
            # 从 test_cases.json 初始化 51 组跨境知识
            cases_json = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "test_cases.json",
            )
            cases_json = os.path.normpath(cases_json)
            if os.path.exists(cases_json):
                try:
                    all_chunks = build_chunks_from_json_cases(cases_json)
                except Exception as exc:
                    print(f"[Orchestrator] ⚠️  种子知识库解析异常(忽略，继续尝试单条导入): {type(exc).__name__}: {exc}")
                    all_chunks = []
                # 分批注入，避免一次性encode/写入过多
                batch_size = 15
                for b_start in range(0, len(all_chunks), batch_size):
                    batch = all_chunks[b_start:b_start + batch_size]
                    try:
                        added_total += self.retriever.ingest_chunks(batch)
                    except Exception as exc:
                        print(f"[Orchestrator] ⚠️  种子注入batch异常(忽略): {type(exc).__name__}: {exc}")
                if added_total:
                    print(f"[Orchestrator] ✅ 种子知识库初始化：加载 {added_total}/{len(all_chunks)} 个知识分块")
                elif all_chunks:
                    print("[Orchestrator] ⚠️  种子知识分块全部已存在或注入失败")
                else:
                    print("[Orchestrator] ⚠️  种子解析结果 0 条（文本切分异常，请检查 test_cases.json）")
            # 再尝试扫描 data/knowledge 目录
            try:
                from data_processing import DataPreprocessingPipeline
                pipe = DataPreprocessingPipeline()
                extra = pipe.scan_directory()
                total_extra = 0
                batch_size = 15
                for cat, cks in extra.items():
                    if cks:
                        for b_start in range(0, len(cks), batch_size):
                            try:
                                total_extra += self.retriever.ingest_chunks(cks[b_start:b_start + batch_size])
                            except Exception as exc:
                                print(f"[Orchestrator] ⚠️  本地文档[{cat}]batch注入异常: {type(exc).__name__}: {exc}")
                if total_extra > 0:
                    print(f"[Orchestrator] ✅ 本地文档追加入库：{total_extra} 个分块")
                added_total += total_extra
            except Exception as exc:
                print(f"[Orchestrator] 本地文档扫描跳过（非关键）：{type(exc).__name__}: {exc}")
        except BaseException as bexc:  # BaseException 包含 MemoryError / KeyboardInterrupt 等非 Exception
            print(f"[Orchestrator] ❌ 知识库初始化致命错误(已忽略，服务仍可用但知识可能为空): {type(bexc).__name__}: {bexc}")
        finally:
            # 🔥 关键：无论注入成功/失败/抛异常，bootstrapped 必须置 True
            # 否则前端永远显示「加载中」，用户永远得不到回答
            self._bootstrapped = True
            print(f"[Orchestrator] 🏁 初始化流程结束: bootstrapped=True, 索引分块总数={self.retriever.total_indexed}")
        return added_total

    # -------- 类目预判（用于展示 + 风控加权）--------
    @staticmethod
    def _predict_category(question: str) -> Optional[str]:
        q = question.lower()
        logi = ["物流", "快递", "运费", "时效", "包裹", "清关", "派送", "fba", "dhl", "ems", "ups", "体积重", "丢件"]
        asp = ["退货", "退款", "售后", "质保", "差评", "争议", "退换", "索赔", "拒付", "chargeback", "paypal", "召回"]
        cus = ["关税", "vat", "海关", "hs编码", "申报", "ioss", "认证", "fda", "合规", "税率", "rcep", "起征点", "nom", "saso"]
        if any(k in q for k in cus):
            return "customs_compliance"
        if any(k in q for k in asp):
            return "after_sales_policy"
        if any(k in q for k in logi):
            return "cross_border_logistics"
        return None

    # -------- 对外主入口 --------
    async def handle_user_request(self,
                                   user_input: str,
                                   session_id: str = "default",
                                   force_language: Optional[str] = None,
                                   category_filter: Optional[str] = None,
                                   ) -> Dict[str, Any]:
        """
        对外问答主入口（async 以便 FastAPI 高并发场景）
        返回字段：
          response: 最终回答
          contexts: 多路召回知识片段列表（用于评测/test_rag_qa.py读取）
          trace: 全链路Trace
          retrieved_contexts: 同上（兼容字段名）
        """
        start_ts = time.perf_counter()
        question = (user_input or "").strip()
        if not question:
            return {
                "response": "请输入您的跨境售后/物流/合规相关问题。",
                "contexts": [],
                "retrieved_contexts": [],
                "trace": {},
            }

        category = category_filter or self._predict_category(question)

        # ========== 新增：知识库加载中友好提示 ==========
        if (not self._bootstrapped) or (self._bootstrap_in_progress):
            loading_lang = force_language or "zh"
            if loading_lang == "en":
                loading_msg = (
                    "⏳ Knowledge base is loading in the background, please wait 10-30s…\n"
                    "First run requires 51 cross-border knowledge chunks.\n"
                    "Please retry after you see 'bootstrap done' in terminal."
                )
            elif loading_lang == "ja":
                loading_msg = "⏳ ナレッジベースはバックグラウンドで読込中です。10～30秒お待ちください。"
            elif loading_lang == "es":
                loading_msg = "⏳ La base de conocimientos se está cargando en segundo plano. Espere 10-30s."
            elif loading_lang == "de":
                loading_msg = "⏳ Wissensdatenbank wird im Hintergrund geladen. Bitte 10-30s warten."
            elif loading_lang == "fr":
                loading_msg = "⏳ Base de connaissances en cours de chargement (10-30s)."
            else:
                loading_msg = (
                    "⏳ 星途环球知识库正在后台加载中，请耐心等待 10-30 秒……\n"
                    "首次启动需初始化 51 组跨境标准知识分块。\n"
                    "您可以稍后再试，或观察终端日志「种子知识库初始化」完成后再提问。"
                )
            return {
                "response": loading_msg,
                "contexts": [],
                "retrieved_contexts": [],
                "trace": {
                    "stage": "bootstrap_loading",
                    "language": loading_lang,
                    "bootstrapped": self._bootstrapped,
                    "bootstrap_in_progress": self._bootstrap_in_progress,
                    "indexed_chunks": self.retriever.total_indexed,
                },
            }

        # Step1: 多路检索（稠密+稀疏，父子块MMR去冗余）
        retrieved: List[VectorSearchResult] = self.retriever.retrieve(
            query=question,
            top_k=self.cfg.retrieval.top_k,
            category_filter=category_filter,
        )

        # 生成前门禁：检索置信度不足直接拒答
        ok_enter, fallback_ans, lang = self.safety.pre_generation_check(question, retrieved)
        # 强制指定语种优先（覆盖 pre_generation_check 里自动检测的 lang）
        if force_language:
            lang = force_language
            # 把拒答文案也翻译成 force_language（pre_generation_check 生成时可能用了检测出的 lang）
            if not ok_enter:
                try:
                    from safety.guard import RetrievalConfidenceGate
                    ok_check, reason = self.safety.gate.should_answer(retrieved)
                    if not ok_check:
                        fallback_ans = RetrievalConfidenceGate.fallback_answer(reason, force_language)
                except Exception:
                    pass

        if not ok_enter:
            return {
                "response": fallback_ans,
                "contexts": [r.chunk.text for r in retrieved[:5]],
                "retrieved_contexts": [r.chunk.text for r in retrieved[:5]],
                "language": lang,
                "category": category,
                "trace": {
                    "stage": "pre_gate_blocked",
                    "language": lang,
                    "total_latency_ms": round((time.perf_counter() - start_ts) * 1000, 2),
                },
                "expected_keywords_hint": self._hint_keywords(category),
            }

        # Step2: 重排序模型优化候选集
        reranked = self.reranker.rerank(
            query=question,
            candidates=retrieved,
            top_k=self.cfg.retrieval.rerank_top_k,
        )
        pieces_texts = [r.chunk.text for r in reranked]

        # Step3: 多语种结构化Prompt构建
        messages, prompt_meta = MultilingualPromptBuilder.build(
            query=question,
            retrieved=reranked,
            force_language=force_language or lang,
        )
        pieces_meta = prompt_meta["pieces"]

        # Step4: LLM生成
        llm_start = time.perf_counter()
        # asyncio 下把同步LLM调用放到线程池，避免阻塞事件循环
        try:
            raw_answer = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.chat(messages),
            )
        except Exception:
            try:
                raw_answer = self.llm.chat(messages)
            except Exception:
                raw_answer = ""
        # 🔥 关键 fallback：真实 LLM 调用失败(返回空/抛异常) → 自动回退 MockLLM
        # 解决 系统代理/网络不可用/API Key 问题 导致用户看到 AI 回答为空的问题
        if not raw_answer or not raw_answer.strip():
            try:
                from generation.llm_prompt import MockLLM
                mock_llm = MockLLM()
                raw_answer = mock_llm.chat(messages)
                print(f"[Orchestrator] 真实LLM无响应，已自动回退 MockLLM 拼答案 (len={len(raw_answer)})")
            except Exception as mock_exc:
                print(f"[Orchestrator] MockLLM 回退也失败: {type(mock_exc).__name__}: {mock_exc}")
                raw_answer = "抱歉，AI 服务暂时不可用，请稍后再试。\n（网络或API Key异常，已触发兜底模式但仍失败）"
        llm_ms = round((time.perf_counter() - llm_start) * 1000, 2)

        # Step5: 生成后风控（忠实度+幻觉+跨境实体）
        # 最终 language 优先用 prompt_meta 里 build 出来的，fallback 到前面 force_language 优先的 lang
        final_language = prompt_meta.get("language", lang)
        final_answer, safety_log = self.safety.post_generation_check(
            question=question,
            answer=raw_answer,
            retrieved_pieces_meta=pieces_meta,
            question_category=category,
            force_language=final_language,
        )

        total_ms = round((time.perf_counter() - start_ts) * 1000, 2)

        # Trace记录（便于演示和调试）
        trace = PipelineTrace(
            question=question,
            language=prompt_meta.get("language", lang),
            category=category,
            retrieval_top_k=self.cfg.retrieval.top_k,
            retrieval_results_count=len(retrieved),
            retrieval_scores={
                r.chunk.chunk_id: r.hybrid_score for r in retrieved[:10]
            },
            rerank_top_k=self.cfg.retrieval.rerank_top_k,
            rerank_results_count=len(reranked),
            llm_latency_ms=llm_ms,
            safety_passed=safety_log["hallucination"].get("block", False) is False,
            safety_log=safety_log,
            total_latency_ms=total_ms,
            answer_truncated=final_answer[:120],
        )

        return {
            "response": final_answer,
            "contexts": pieces_texts,
            "retrieved_contexts": pieces_texts,  # 兼容test_rag_qa.py字段
            "pieces_meta": pieces_meta,
            "trace": trace.__dict__,
            "category": category,
            "language": prompt_meta.get("language", lang),
            "expected_keywords_hint": self._hint_keywords(category),
        }

    @staticmethod
    def _hint_keywords(category: Optional[str]) -> List[str]:
        """用于演示/调试：按类目给期望关键词参考（非强制）"""
        if not category:
            return []
        if category == "customs_compliance":
            return ["关税", "VAT", "IOSS", "申报", "认证", "合规"]
        if category == "after_sales_policy":
            return ["退货", "退款", "售后", "质保", "平台规则"]
        if category == "cross_border_logistics":
            return ["物流", "时效", "运费", "清关", "派送"]
        return []

    # -------- 工具方法：知识库状态 --------
    def status(self) -> Dict[str, Any]:
        return {
            "bootstrapped": self._bootstrapped,
            "bootstrap_in_progress": self._bootstrap_in_progress,
            "total_indexed_chunks": self.retriever.total_indexed,
            "categories": {
                k: {
                    "name": v["name"],
                    "chunk_size": v["chunk_size"],
                    "retrieval_weight": self.cfg.retrieval.category_weights.get(k, 1.0),
                }
                for k, v in KNOWLEDGE_CATEGORIES.items()
            },
            "llm_provider": self.cfg.llm.provider,
            "rerank_provider": self.cfg.retrieval.rerank_provider,
            "embedding_provider": self.cfg.embedding.dense_provider,
            "vector_store": getattr(self.retriever.store, "store_label", type(self.retriever.store).__name__),
            "hybrid_weights": {
                "dense": self.cfg.milvus.hybrid_search_weight_dense,
                "sparse": self.cfg.milvus.hybrid_search_weight_sparse,
            },
            "safety": {
                "confidence_threshold": self.cfg.safety.retrieval_confidence_threshold,
                "hallucination_block": self.cfg.safety.hallucination_block_answer,
            },
        }

    def ingest_document_chunks(self, chunks: List[KnowledgeChunk]) -> int:
        """对外暴露接口：增量导入知识分块"""
        n = self.retriever.ingest_chunks(chunks)
        return n
