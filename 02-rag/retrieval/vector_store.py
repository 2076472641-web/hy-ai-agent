# -*- coding: utf-8 -*-
"""
星途环球·跨境售后RAG智能问答系统 - 特征工程与向量检索模块
对应简历模块2：
  - 针对跨境外贸专属术语，搭建 Milvus 稠密向量 + 稀疏向量双表征架构
  - 配置父子块多级索引筛除冗余向量，降低检索时延
  - 适配跨境商品、各国海关政策术语精准召回需求
"""
from __future__ import annotations

import math
import re
import json
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

from config import get_config, KNOWLEDGE_CATEGORIES
from data_processing import KnowledgeChunk


# ============================================================
# 1. Embedding接口：稠密向量表征（Mock / 真实HuggingFace / OpenAI / DashScope）
# ============================================================
class BaseEmbedding(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """批量编码为稠密向量"""
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError


class MockEmbedding(BaseEmbedding):
    """
    Mock版：基于字符n-gram哈希，生成稳定伪向量
    - 优点：零依赖离线可跑，相同文本生成相同向量，语义相近文本向量余弦接近
    - 用途：开发调试、面试演示跑通全链路
    """
    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _ngram_features(text: str, n_gram: int = 3) -> Counter:
        text = re.sub(r"\s+", "", text).lower()
        pure_cn = re.sub(r"[^\u4e00-\u9fff]", "", text)
        feats: Counter = Counter()
        # 中文bi/tri gram
        for i in range(len(pure_cn) - n_gram + 1):
            feats[pure_cn[i:i + n_gram]] += 1
        for i in range(len(pure_cn) - 1):
            feats[pure_cn[i:i + 2]] += 1
        # 英文整词
        for w in re.findall(r"[A-Za-z][A-Za-z0-9._/%-]{1,}", text):
            feats[w.lower()] += 2
        # 数字组合（跨境时效税率）
        for m in re.findall(r"\d+(?:\.\d+)?%|\d+(?:\.\d+)?(?:美元|欧元|元|天|kg|g)", text):
            feats[m] += 3
        return feats

    def encode(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            feats = self._ngram_features(text)
            vec = [0.0] * self._dim
            norm_sq = 0.0
            for tok, w in feats.items():
                # 稳定哈希
                h = 0
                for ch in tok:
                    h = (h * 131 + ord(ch)) & 0xFFFFFFFF
                idx = h % self._dim
                sign = 1.0 if (h >> 16) & 1 else -1.0
                val = sign * (1.0 + 0.5 * math.log(1 + w))
                vec[idx] += val
                norm_sq += val * val
            # L2归一化
            norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
            vec = [v / norm for v in vec]
            out.append(vec)
        return out


class EmbeddingFactory:
    @classmethod
    def create(cls) -> BaseEmbedding:
        cfg = get_config()
        provider = cfg.embedding.dense_provider.lower()
        if provider == "mock":
            return MockEmbedding(dim=cfg.embedding.mock_dim)
        # 真实场景可扩展 HuggingFaceBgeEmbedding / OpenAIEmbedding / DashScopeEmbedding
        print(f"[EmbeddingFactory] provider='{provider}' 未实现，回退MockEmbedding")
        return MockEmbedding(dim=cfg.embedding.mock_dim)


# ============================================================
# 2. 稀疏向量表征（BM25/关键词匹配层）+ 稀疏检索
# ============================================================
class SparseIndex:
    """
    稀疏向量索引：基于倒排索引 + BM25-like打分
    对应简历：稀疏向量双表征架构
    """
    def __init__(self):
        self.docs: Dict[str, str] = {}  # chunk_id -> text
        self.avgdl: float = 0.0
        self.total_docs: int = 0
        self.inverted: Dict[str, List[str]] = {}  # term -> [chunk_ids]
        self.df: Dict[str, int] = {}  # term -> doc_freq
        self.k1: float = 1.5
        self.b: float = 0.75
        self.dl: Dict[str, int] = {}  # chunk_id -> doc_length

    def _tokenize(self, text: str) -> List[str]:
        text = (text or "").lower()
        toks: List[str] = []
        # 英文/数字整词
        toks.extend(re.findall(r"[a-z][a-z0-9._/%-]{1,}", text))
        # 数字时效税率短语
        toks.extend(re.findall(r"\d+(?:\.\d+)?%|\d+(?:\.\d+)?(?:美元|欧元|元|天|kg|g|工作日)", text))
        # 中文bi-gram
        cn = re.sub(r"[^\u4e00-\u9fff]", "", text)
        for i in range(len(cn) - 1):
            toks.append(cn[i:i + 2])
        return toks

    def add_documents(self, chunks: List[KnowledgeChunk]):
        for c in chunks:
            if c.chunk_id in self.docs:
                continue
            text = c.text + " " + " ".join(c.sparse_keywords)
            self.docs[c.chunk_id] = text
            self.dl[c.chunk_id] = len(text)
            toks = self._tokenize(text)
            seen = set()
            for t in toks:
                if t in seen:
                    continue
                seen.add(t)
                self.inverted.setdefault(t, []).append(c.chunk_id)
                self.df[t] = self.df.get(t, 0) + 1
        self.total_docs = len(self.docs)
        self.avgdl = (sum(self.dl.values()) / self.total_docs) if self.total_docs > 0 else 0.0

    def search(self, query: str, top_k: int = 10,
               category_filter: Optional[str] = None,
               chunk_id_to_category: Optional[Dict[str, str]] = None) -> List[Tuple[str, float]]:
        """BM25检索，返回 [(chunk_id, bm25_score)]"""
        q_toks = self._tokenize(query)
        scores: Dict[str, float] = {}
        N = self.total_docs or 1
        for t in q_toks:
            if t not in self.df:
                continue
            idf = math.log(1 + (N - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for cid in self.inverted[t]:
                if category_filter and chunk_id_to_category and chunk_id_to_category.get(cid) != category_filter:
                    continue
                dl = self.dl.get(cid, 0)
                tf = self.docs[cid].count(t)  # 简化：字符串出现次数近似词频
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl > 0 else 1.0))
                bm25 = idf * (tf * (self.k1 + 1)) / (denom if denom > 0 else 1.0)
                scores[cid] = scores.get(cid, 0.0) + bm25
        # 稀疏关键词强匹配（跨境术语/缩写直接加权）
        for sk in set(re.findall(r"[A-Z]{2,8}", query)):
            for cid, doc_text in self.docs.items():
                if sk in doc_text.upper():
                    scores[cid] = scores.get(cid, 0.0) + 2.0
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        # 归一化到0-1
        if ranked:
            max_s = max(r[1] for r in ranked)
            if max_s > 0:
                ranked = [(cid, s / max_s) for cid, s in ranked]
        return ranked


# ============================================================
# 3. 向量库抽象层（真实Milvus + 本地Mock双实现）
# ============================================================
@dataclass
class VectorSearchResult:
    chunk: KnowledgeChunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0


class BaseVectorStore(ABC):
    store_label: str = "Base"

    @abstractmethod
    def add_chunks(self, chunks: List[KnowledgeChunk]) -> int:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: List[float], query_text: str,
               top_k: int = 10,
               category_filter: Optional[str] = None) -> List[VectorSearchResult]:
        raise NotImplementedError

    @property
    @abstractmethod
    def total_chunks(self) -> int:
        raise NotImplementedError


class MockVectorStore(BaseVectorStore):
    """
    本地Mock版向量库（用Python dict + 余弦 + BM25稀疏实现双表征）
    可随时替换为真实 Milvus / Chroma / FAISS，接口一致
    """
    store_label = "MockLocal"
    AUTO_PERSIST = os.environ.get("MOCK_VS_AUTOSAVE", "0") == "1"

    def __init__(self):
        cfg = get_config()
        self._embed = EmbeddingFactory.create()
        self._sparse = SparseIndex()
        self._chunks: Dict[str, KnowledgeChunk] = {}
        self._vectors: Dict[str, List[float]] = {}
        self._cid_to_cat: Dict[str, str] = {}
        self._parent_cache: Dict[str, str] = {}  # child -> parent id
        self._w_dense = cfg.milvus.hybrid_search_weight_dense
        self._w_sparse = cfg.milvus.hybrid_search_weight_sparse
        self._mmr_enabled = cfg.retrieval.mmr_enabled
        self._mmr_lambda = cfg.retrieval.mmr_lambda
        self._persist_path = os.path.join(cfg.data_dir, "vector_store_mock.pkl")
        self._load_from_disk()

    # ---- 持久化 ----
    def _load_from_disk(self):
        if not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "rb") as f:
                data = pickle.load(f)
            self._chunks = data["chunks"]
            self._vectors = data["vectors"]
            self._cid_to_cat = data["cid_to_cat"]
            self._parent_cache = data["parent_cache"]
            if self._chunks:
                # 分批添加稀疏索引，避免超大列表一次灌入
                all_chunks = list(self._chunks.values())
                for i in range(0, len(all_chunks), 100):
                    self._sparse.add_documents(all_chunks[i:i+100])
                print(f"[MockVectorStore] 从磁盘加载 {len(self._chunks)} 个向量")
        except Exception as exc:
            print(f"[MockVectorStore] 磁盘加载失败(忽略): {exc}")

    def save_to_disk(self):
        """手动保存（生产环境才需要）"""
        try:
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            with open(self._persist_path, "wb") as f:
                pickle.dump({
                    "chunks": self._chunks,
                    "vectors": self._vectors,
                    "cid_to_cat": self._cid_to_cat,
                    "parent_cache": self._parent_cache,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            print(f"[MockVectorStore] 持久化失败(忽略): {exc}")

    # ---- 写入（分批编码，避免Windows下栈溢出）----
    def add_chunks(self, chunks: List[KnowledgeChunk]) -> int:
        if not chunks:
            return 0
        # 先构建父子索引（筛除冗余向量用）
        for c in chunks:
            if c.parent_chunk_id:
                self._parent_cache[c.chunk_id] = c.parent_chunk_id
        # 分批编码 + 入库（每批20个，安全）
        batch_size = 20
        added_total = 0
        for b_start in range(0, len(chunks), batch_size):
            batch = chunks[b_start:b_start + batch_size]
            texts = [c.text for c in batch]
            try:
                vecs = self._embed.encode(texts)
            except Exception as exc:
                print(f"[MockVectorStore] encode batch失败，跳过本批: {exc}")
                continue
            for c, vec in zip(batch, vecs):
                if c.chunk_id in self._chunks:
                    continue
                self._chunks[c.chunk_id] = c
                self._vectors[c.chunk_id] = vec
                self._cid_to_cat[c.chunk_id] = c.category
                added_total += 1
            # 稀疏索引分批灌入
            self._sparse.add_documents(batch)
        if added_total and self.AUTO_PERSIST:
            self.save_to_disk()
        return added_total

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    # ---- 检索 ----
    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (na * nb)

    def search(self, query_vector: List[float], query_text: str,
               top_k: int = 10,
               category_filter: Optional[str] = None) -> List[VectorSearchResult]:
        if not self._chunks:
            return []
        # (A) 稠密向量：余弦 Top-2K
        dense_raw: List[Tuple[str, float]] = []
        for cid, vec in self._vectors.items():
            if category_filter and self._cid_to_cat.get(cid) != category_filter:
                continue
            sim = self._cosine(query_vector, vec)
            dense_raw.append((cid, sim))
        dense_raw.sort(key=lambda x: -x[1])
        dense_top = {cid: sim for cid, sim in dense_raw[:top_k * 2]}

        # (B) 稀疏 BM25 Top-2K
        sparse_res = self._sparse.search(query_text, top_k=top_k * 2,
                                         category_filter=category_filter,
                                         chunk_id_to_category=self._cid_to_cat)
        sparse_top = {cid: sc for cid, sc in sparse_res}

        # (C) 并集混合打分（加权Fusion，对应简历"多路检索权重调试"）
        cid_set = set(dense_top.keys()) | set(sparse_top.keys())
        fused: List[VectorSearchResult] = []
        for cid in cid_set:
            ds = dense_top.get(cid, 0.0)
            ss = sparse_top.get(cid, 0.0)
            hs = self._w_dense * ds + self._w_sparse * ss
            if cid not in self._chunks:
                continue
            fused.append(VectorSearchResult(
                chunk=self._chunks[cid],
                dense_score=round(ds, 4),
                sparse_score=round(ss, 4),
                hybrid_score=round(hs, 4),
            ))
        fused.sort(key=lambda r: -r.hybrid_score)

        # (D) 父子块去重：同一父块的多个子块只保留score最高的1-2个
        # 对应简历"父子块多级索引筛除冗余向量"
        selected = self._parent_dedup_and_mmr(fused, top_k, query_vector)
        return selected

    def _parent_dedup_and_mmr(self, candidates: List[VectorSearchResult],
                              top_k: int,
                              query_vector: List[float]) -> List[VectorSearchResult]:
        # 父子块去重
        parent_seen: Dict[str, int] = {}
        deduped: List[VectorSearchResult] = []
        for r in candidates:
            pid = r.chunk.parent_chunk_id
            if pid:
                if parent_seen.get(pid, 0) >= 2:  # 同父块最多2个子块
                    continue
                parent_seen[pid] = parent_seen.get(pid, 0) + 1
            deduped.append(r)
            if len(deduped) >= top_k * 2:
                break
        # MMR 去冗余（相似度高的重复块剔除）
        if not self._mmr_enabled or len(deduped) <= top_k:
            return deduped[:top_k]
        selected_ids: List[str] = []
        final: List[VectorSearchResult] = []
        while len(final) < top_k and deduped:
            best_idx = -1
            best_mmr = -1e9
            for i, r in enumerate(deduped):
                rvec = self._vectors[r.chunk.chunk_id]
                # 与query相似度
                rel = r.hybrid_score
                # 与已选最大相似度
                red = 0.0
                for sid in selected_ids:
                    svec = self._vectors[sid]
                    sim = self._cosine(rvec, svec)
                    if sim > red:
                        red = sim
                mmr = self._mmr_lambda * rel - (1 - self._mmr_lambda) * red
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            if best_idx < 0:
                break
            picked = deduped.pop(best_idx)
            final.append(picked)
            selected_ids.append(picked.chunk.chunk_id)
        return final


class MilvusVectorStore(BaseVectorStore):
    """
    🔥 真实Milvus实现（接口与Mock 100%一致，生产环境直接替换Mock）
    对应简历：Milvus稠密 + 稀疏双表征 + 父子块多级索引
    🧩 增强点（共7项关键健壮性）：
      1️⃣ 初始化时**自动从Milvus拉取已有chunks** → 重建稀疏索引+chunks缓存（Milvus重启后不丢失）
      2️⃣ search时**优先从hit.entity字段构建KnowledgeChunk** → 不依赖内存缓存，重启后立即生效
      3️⃣ 所有Milvus操作包try/except → 异常时返回空，不抛异常打爆上层服务
      4️⃣ pymilvus import/connect失败 → VectorStoreFactory正确回退Mock
      5️⃣ 🔥 NEW: 向量维度不兼容自动检测 → 旧集合自动备份+重建（embedding模型切换不炸）
      6️⃣ 🔥 NEW: 兼容pymilvus 2.3/2.4/2.5各种hit.entity格式 → 老版本新版本都能正确取字段
      7️⃣ 🔥 NEW: add_chunks前做维度校验 → 插错维度的向量直接跳过不插坏集合
    """
    store_label = "Milvus"

    def __init__(self):
        from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
        cfg = get_config()
        self.cfg = cfg
        self._embed = EmbeddingFactory.create()
        dim = self._embed.dim
        self._expected_dim = dim  # 期望维度，用于后续校验
        self._w_dense = cfg.milvus.hybrid_search_weight_dense
        self._w_sparse = cfg.milvus.hybrid_search_weight_sparse

        # ① 连接Milvus（多次连接安全：alias=default已存在则先断开）
        try:
            connections.disconnect("default")
        except Exception:
            pass
        try:
            connections.connect(host=cfg.milvus.host, port=cfg.milvus.port, alias="default", timeout=10)
        except Exception as exc:
            # 连接失败：抛给工厂层，自动回退Mock
            raise RuntimeError(f"Milvus连接失败 host={cfg.milvus.host}:{cfg.milvus.port} → {exc}")

        # ② 创建/获取Collection（带维度校验，不兼容自动重建）
        col_name = cfg.milvus.collection_prefix
        self.col = self._init_collection_with_dim_check(
            col_name=col_name,
            expected_dim=dim,
        )

        # 预加载
        try:
            self.col.load()
        except Exception:
            pass

        # 稀疏索引 + chunks缓存（Milvus重启后也能重建）
        self._sparse = SparseIndex()
        self._chunks_cache: Dict[str, KnowledgeChunk] = {}
        self._cid_to_cat: Dict[str, str] = {}  # 稀疏过滤用

        # ③ 关键增强：从Milvus拉取已有实体，重建内存索引&缓存
        self._rebuild_cache_from_milvus()

    # ---------------------------
    # 🔥 NEW: 带维度校验的集合初始化（embedding模型切换时自动重建）
    # ---------------------------
    def _init_collection_with_dim_check(self, col_name: str, expected_dim: int):
        from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility

        def _build_fields(dim: int):
            return [
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="parent_chunk_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            ]

        def _build_schema_and_index(col: "Collection", dim: int):
            try:
                col.create_index(
                    field_name="dense_vector",
                    index_params={
                        "index_type": "IVF_FLAT",
                        "metric_type": "COSINE",
                        "params": {"nlist": min(1024, max(64, dim // 2))},
                    },
                )
            except Exception as idx_exc:
                print(f"[MilvusVectorStore] 建索引提示（可忽略）: {idx_exc}")

        # --- 情况A：集合不存在 → 直接新建 ---
        try:
            has_col = bool(utility.has_collection(col_name))
        except Exception:
            has_col = False

        if not has_col:
            print(f"[MilvusVectorStore] 集合不存在，创建新集合: {col_name} (dim={expected_dim})")
            schema = CollectionSchema(
                fields=_build_fields(expected_dim),
                description=f"xingtu_kb_dim{expected_dim}",  # dimension写进description做校验
            )
            new_col = Collection(col_name, schema=schema)
            _build_schema_and_index(new_col, expected_dim)
            return new_col

        # --- 情况B：集合已存在 → 检查维度是否匹配 ---
        existing_col = Collection(col_name)
        actual_dim = None
        try:
            # 从schema中读取向量字段的dim
            for f in existing_col.schema.fields:
                try:
                    if f.name == "dense_vector":
                        actual_dim = int(getattr(f, "dim", 0))
                        break
                except Exception:
                    pass
        except Exception:
            actual_dim = None

        # 如果读不到dim，尝试从description解析（兜底）
        if not actual_dim:
            try:
                desc = str(existing_col.schema.description or "")
                import re as _re
                m = _re.search(r"dim(\d+)", desc)
                if m:
                    actual_dim = int(m.group(1))
            except Exception:
                pass

        # 维度匹配 → 直接用
        if actual_dim and actual_dim == expected_dim:
            try:
                existing_col.load()
            except Exception:
                pass
            return existing_col

        # --- 情况C：维度不匹配 → 备份旧集合+重建新集合 ---
        print("=" * 60)
        print(f"[MilvusVectorStore] ⚠️  向量维度不兼容！")
        print(f"    集合现有维度: dim={actual_dim or '未知'}")
        print(f"    当前Embedding期望维度: dim={expected_dim}")
        print(f"    原因: 您切换了 EMBED_PROVIDER/MODEL，向量维度发生变化")
        print("=" * 60)

        backup_name = f"{col_name}_backup_dim{actual_dim or '0'}_{int(__import__('time').time())}"
        try:
            # 尝试备份（有些pymilvus版本不支持rename，失败就直接删）
            try:
                utility.rename_collection(col_name, backup_name)
                print(f"[MilvusVectorStore] 旧集合已备份为: {backup_name}（保留30天后可手动删除）")
            except Exception:
                # rename不支持 → 直接删旧集合
                print(f"[MilvusVectorStore] 当前pymilvus不支持rename，直接重建集合…")
                utility.drop_collection(col_name)
        except Exception as exc:
            print(f"[MilvusVectorStore] 备份旧集合失败，强制重建: {exc}")
            try:
                utility.drop_collection(col_name)
            except Exception:
                pass

        # 建全新集合
        print(f"[MilvusVectorStore] 重建集合: {col_name} (dim={expected_dim})")
        schema = CollectionSchema(
            fields=_build_fields(expected_dim),
            description=f"xingtu_kb_dim{expected_dim}",
        )
        new_col = Collection(col_name, schema=schema)
        _build_schema_and_index(new_col, expected_dim)
        return new_col

    # ---------------------------
    # 🔥 增强：启动时从 Milvus 重建稀疏索引和 chunks 缓存
    # ---------------------------
    def _rebuild_cache_from_milvus(self):
        try:
            total = int(self.col.num_entities)
        except Exception:
            total = 0
        if total <= 0:
            return
        print(f"[MilvusVectorStore] 集合中已有 {total} 个分块，开始重建稀疏索引&缓存…")
        try:
            batch_size = 500
            # 迭代拉取（避免一次拉太多爆内存）
            all_entities: List[Any] = []
            offset = 0
            while offset < total:
                try:
                    batch = self.col.query(
                        expr="",
                        output_fields=["chunk_id", "doc_id", "category", "parent_chunk_id", "text"],
                        offset=offset,
                        limit=batch_size,
                    )
                except Exception:
                    # 老版pymilvus不支持offset+limit，回退全量
                    batch = self.col.query(
                        expr="",
                        output_fields=["chunk_id", "doc_id", "category", "parent_chunk_id", "text"],
                    )
                    all_entities = batch
                    break
                if not batch:
                    break
                all_entities.extend(batch)
                offset += len(batch)
                if len(batch) < batch_size:
                    break

            # 转成 KnowledgeChunk，构建稀疏索引+缓存
            chunks_to_add: List[KnowledgeChunk] = []
            for ent in all_entities:
                try:
                    cid = ent.get("chunk_id") or ""
                    if not cid:
                        continue
                    # 从 text 中推断语言（中文字符占比>20%认为是中文，否则英文）
                    txt = ent.get("text") or ""
                    cn_chars = len(re.findall(r"[\u4e00-\u9fff]", txt))
                    total_chars = max(len(re.findall(r"[A-Za-z\u4e00-\u9fff]", txt)), 1)
                    inferred_lang = "zh" if (cn_chars / total_chars) > 0.2 else "en"
                    chunk = KnowledgeChunk(
                        chunk_id=cid,
                        doc_id=ent.get("doc_id") or "",
                        category=ent.get("category") or "cross_border_logistics",
                        text=txt,
                        language=inferred_lang,
                        parent_chunk_id=ent.get("parent_chunk_id") or None,
                        sparse_keywords=[],  # 后面重建稀疏索引时从text分词
                    )
                    self._chunks_cache[cid] = chunk
                    self._cid_to_cat[cid] = chunk.category
                    chunks_to_add.append(chunk)
                except Exception as rebuild_exc:
                    print(f"[MilvusVectorStore] 重建单条失败(忽略): {rebuild_exc}")
                    continue
            if chunks_to_add:
                # 每批100个重建稀疏索引，避免卡顿
                for i in range(0, len(chunks_to_add), 100):
                    self._sparse.add_documents(chunks_to_add[i:i+100])
            print(f"[MilvusVectorStore] 重建完成: 缓存 {len(self._chunks_cache)} 个chunk, 稀疏文档{self._sparse.total_docs} 条")
        except Exception as exc:
            print(f"[MilvusVectorStore] 缓存重建失败(忽略，不影响主流程): {exc}")

    # ---------------------------
    # 🔥 NEW: 写入（含维度校验 + 分批 + try/except）
    # ---------------------------
    def add_chunks(self, chunks: List[KnowledgeChunk]) -> int:
        if not chunks:
            return 0
        # 先过滤已存在的chunk（避免重复插入主键冲突）
        new_chunks: List[KnowledgeChunk] = []
        for c in chunks:
            if c.chunk_id not in self._chunks_cache:
                new_chunks.append(c)
        if not new_chunks:
            return 0
        try:
            # 分批encode+insert（每批30个）
            batch_size = 30
            inserted = 0
            for b_start in range(0, len(new_chunks), batch_size):
                batch = new_chunks[b_start:b_start + batch_size]
                texts = [c.text for c in batch]
                try:
                    vecs = self._embed.encode(texts)
                except Exception as exc:
                    print(f"[MilvusVectorStore] encode batch失败: {exc}")
                    continue
                # 🔥 NEW: 维度校验（防止插错维度的向量搞坏集合）
                dim_ok = True
                for i, v in enumerate(vecs):
                    if len(v) != self._expected_dim:
                        print(f"[MilvusVectorStore] ⚠️  向量维度异常，跳过第{i}条: expect={self._expected_dim} actual={len(v)}")
                        dim_ok = False
                        break
                if not dim_ok:
                    continue
                # 兼容两种insert方式（pymilvus不同版本API略有差异）
                try:
                    # 方式1: 按列传（最常用）
                    entities = [
                        [c.chunk_id for c in batch],
                        [c.doc_id for c in batch],
                        [c.category for c in batch],
                        [c.parent_chunk_id or "" for c in batch],
                        list(vecs),
                        [c.text[:8190] for c in batch],
                    ]
                    self.col.insert(entities)
                except Exception as exc1:
                    # 方式2: 按行传dict（有些版本要求）
                    try:
                        rows = []
                        for c, v in zip(batch, vecs):
                            rows.append({
                                "chunk_id": c.chunk_id,
                                "doc_id": c.doc_id,
                                "category": c.category,
                                "parent_chunk_id": c.parent_chunk_id or "",
                                "dense_vector": v,
                                "text": c.text[:8190],
                            })
                        self.col.insert(rows)
                    except Exception as exc2:
                        print(f"[MilvusVectorStore] insert batch失败(两种格式均失败): {exc1} / {exc2}")
                        continue
                try:
                    self.col.flush()
                except Exception:
                    pass  # flush失败不阻塞
                # 更新缓存+稀疏索引
                for c in batch:
                    self._chunks_cache[c.chunk_id] = c
                    self._cid_to_cat[c.chunk_id] = c.category
                self._sparse.add_documents(batch)
                inserted += len(batch)
            return inserted
        except Exception as exc:
            print(f"[MilvusVectorStore] add_chunks异常: {exc}，已写入 {len(self._chunks_cache)} 条")
            return 0

    @property
    def total_chunks(self) -> int:
        try:
            return int(self.col.num_entities)
        except Exception:
            return len(self._chunks_cache)

    # ---------------------------
    # 🔥 检索增强：直接从 Milvus hit.entity 构建 chunk，不依赖缓存
    #    兼容 pymilvus 2.3 / 2.4 / 2.5 各种返回格式
    # ---------------------------
    @staticmethod
    def _entity_to_dict(entity: Any) -> Dict[str, Any]:
        """🔥 NEW: 多版本兼容：把各种pymilvus hit.entity格式转成普通dict"""
        out: Dict[str, Any] = {}
        if entity is None:
            return out
        # 方案1: to_dict() 方法（新版pymilvus）
        try:
            d = entity.to_dict()
            if isinstance(d, dict):
                # to_dict结果可能嵌套在 entity 字段下，也可能直接平铺
                if "entity" in d and isinstance(d["entity"], dict):
                    out.update(d["entity"])
                # 取id和score字段
                for k in ["id", "chunk_id", "score", "distance"]:
                    if k in d and k not in out:
                        out[k] = d[k]
                out.update({k: v for k, v in d.items() if k not in out})
        except Exception:
            pass
        # 方案2: entity是对象，有 get() 方法和 fields 属性
        if not out:
            try:
                for f_name in ["chunk_id", "doc_id", "category", "parent_chunk_id", "text"]:
                    try:
                        val = entity.get(f_name)
                        if val is not None:
                            out[f_name] = val
                    except Exception:
                        pass
            except Exception:
                pass
        # 方案3: 遍历 entity.fields 或 entity 的动态属性（兜底）
        if not out:
            try:
                fields = getattr(entity, "fields", None) or getattr(entity, "keys", None)
                if callable(fields):
                    fields = fields()
                if fields:
                    for f in fields:
                        try:
                            out[str(f)] = entity.get(f)
                        except Exception:
                            pass
            except Exception:
                pass
        # 方案4: dir()暴力取值（终极兜底，永远能捞到字段）
        if not out:
            for attr in ["chunk_id", "doc_id", "category", "parent_chunk_id", "text", "pk", "id"]:
                try:
                    val = getattr(entity, attr, None)
                    if val is not None and not callable(val):
                        out[attr] = val
                except Exception:
                    pass
        # 兼容 chunk_id <-> id <-> pk 三种主键名
        for k in ("id", "pk"):
            if (not out.get("chunk_id")) and out.get(k):
                out["chunk_id"] = out[k]
        return out

    @staticmethod
    def _chunk_from_milvus_hit(entity: Dict[str, Any]) -> Optional[KnowledgeChunk]:
        """从Milvus返回的entity字段构建KnowledgeChunk对象（零缓存依赖）"""
        try:
            cid = entity.get("chunk_id") or entity.get("id") or entity.get("pk")
            if not cid:
                return None
            parent_val = entity.get("parent_chunk_id") or None
            if parent_val == "":
                parent_val = None
            txt = str(entity.get("text") or "")
            # 推断语言
            cn_chars = len(re.findall(r"[\u4e00-\u9fff]", txt))
            total_chars = max(len(re.findall(r"[A-Za-z\u4e00-\u9fff]", txt)), 1)
            inferred_lang = "zh" if (cn_chars / total_chars) > 0.2 else "en"
            return KnowledgeChunk(
                chunk_id=str(cid),
                doc_id=str(entity.get("doc_id") or ""),
                category=str(entity.get("category") or "cross_border_logistics"),
                text=txt,
                language=inferred_lang,
                parent_chunk_id=parent_val,
                sparse_keywords=[],
            )
        except Exception as chunk_exc:
            print(f"[MilvusVectorStore] _chunk_from_milvus_hit 异常(跳过): {chunk_exc}")
            return None

    def search(self, query_vector: List[float], query_text: str,
               top_k: int = 10,
               category_filter: Optional[str] = None) -> List[VectorSearchResult]:
        try:
            # 🔥 query_vector 维度校验（防止传错维度直接报错）
            if len(query_vector) != self._expected_dim:
                print(f"[MilvusVectorStore] ⚠️  query向量维度不匹配: expect={self._expected_dim} actual={len(query_vector)}，跳过稠密检索，只走稀疏")
                dense_top = {}
                hit_to_entity = {}
            else:
                expr = f'category == "{category_filter}"' if category_filter else None
                search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}
                # (A) 稠密向量：Milvus 原生余弦检索
                try:
                    self.col.load()  # 保证已加载（第一次search必走）
                except Exception:
                    pass
                res = self.col.search(
                    data=[query_vector],
                    anns_field="dense_vector",
                    param=search_params,
                    limit=top_k * 2,
                    expr=expr,
                    output_fields=["chunk_id", "doc_id", "category", "parent_chunk_id", "text"],
                )
                dense_top: Dict[str, float] = {}
                hit_to_entity: Dict[str, Dict[str, Any]] = {}
                for hits in res:
                    for h in hits:
                        try:
                            # 🔥 用多版本兼容函数解析entity
                            try:
                                raw_ent = h.entity
                            except Exception:
                                raw_ent = getattr(h, "entity", None)
                            ent_dict = self._entity_to_dict(raw_ent)
                            # 兜底：从 h.id / h.pk 拿主键
                            if not ent_dict.get("chunk_id"):
                                for attr in ("id", "pk"):
                                    try:
                                        v = getattr(h, attr, None)
                                        if v is not None:
                                            ent_dict["chunk_id"] = v
                                            break
                                    except Exception:
                                        pass
                            cid = ent_dict.get("chunk_id")
                            if not cid:
                                continue
                            cid = str(cid)
                            # COSINE distance: pymilvus返回1-cosine，所以score = 1 - distance
                            try:
                                dist = float(h.distance)
                            except Exception:
                                dist = 1.0  # 拿不到distance就算最低分
                            dense_top[cid] = float(1.0 - dist)
                            hit_to_entity[cid] = ent_dict
                        except Exception as h_exc:
                            print(f"[MilvusVectorStore] 解析hit失败(跳过): {h_exc}")
                            continue

            # (B) 稀疏 BM25：从内存稀疏索引取
            sparse_top: Dict[str, float] = {}
            try:
                sparse_top = {
                    cid: sc for cid, sc in self._sparse.search(
                        query_text, top_k=top_k * 2,
                        category_filter=category_filter,
                        chunk_id_to_category=self._cid_to_cat,
                    )
                }
            except Exception:
                pass  # 稀疏失败不影响稠密

            all_ids = set(dense_top.keys()) | set(sparse_top.keys())
            out: List[VectorSearchResult] = []
            for cid in all_ids:
                # 🔥 优先：从Milvus hit实体构建chunk（避免缓存空导致漏结果）
                chunk = None
                if cid in hit_to_entity:
                    chunk = self._chunk_from_milvus_hit(hit_to_entity[cid])
                if chunk is None and cid in self._chunks_cache:
                    chunk = self._chunks_cache[cid]
                if chunk is None:
                    continue
                ds = dense_top.get(cid, 0.0)
                ss = sparse_top.get(cid, 0.0)
                hs = self._w_dense * ds + self._w_sparse * ss
                out.append(VectorSearchResult(
                    chunk=chunk,
                    dense_score=round(ds, 4),
                    sparse_score=round(ss, 4),
                    hybrid_score=round(hs, 4),
                ))
            out.sort(key=lambda r: -r.hybrid_score)
            return out[:top_k]
        except Exception as exc:
            print(f"[MilvusVectorStore] search异常: {exc}（返回空结果，上层服务继续工作）")
            return []


class VectorStoreFactory:
    @classmethod
    def create(cls) -> BaseVectorStore:
        """
        优先级：
          ① 装了pymilvus + Milvus服务连接成功 → 🔥 MilvusVectorStore（真实生产方案）
          ② 任何一环失败 → 自动回退 MockVectorStore（本地模拟，接口完全一致）
        🔥 新手用户无需担心：装不上pymilvus/忘开Docker Milvus，服务照样能跑，知识存本地pickle文件
        """
        cfg = get_config()
        # --- 第1关：pymilvus是否安装 ---
        try:
            import pymilvus  # noqa: F401
        except ImportError as exc:
            print("=" * 70)
            print("⚠️  [VectorStoreFactory] 未安装 pymilvus，自动回退Mock本地向量库")
            print("💡 安装命令（PyCharm Terminal运行，先关代理）：")
            print("  步骤1: 关闭HTTP代理环境变量")
            print('         $env:HTTP_PROXY=$null; $env:HTTPS_PROXY=$null; $env:http_proxy=$null; $env:https_proxy=$null')
            print("  步骤2: 用阿里云镜像装 pymilvus + grpcio + protobuf")
            print("         pip install pymilvus grpcio protobuf -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com")
            print(f"\n原始错误: {exc}")
            print("=" * 70)
            return MockVectorStore()

        # --- 第2关：Milvus服务是否可连接（创建实例测试，含集合初始化+缓存重建） ---
        try:
            store = MilvusVectorStore()
            # 初始化完，total_chunks是合法int就说明真的连上了
            _ = store.total_chunks  # noqa: F841
            col_name = cfg.milvus.collection_prefix
            host = cfg.milvus.host
            port = cfg.milvus.port
            print("=" * 70)
            print(f"✅ [VectorStoreFactory] 使用【Milvus真实向量库】")
            print(f"   地址: {host}:{port}  集合名: {col_name}")
            print(f"   已索引分块: {store.total_chunks} 条")
            print("=" * 70)
            return store
        except Exception as exc:
            print("=" * 70)
            print(f"⚠️  [VectorStoreFactory] Milvus连接/初始化失败 → {exc}")
            print("💡 排查清单:")
            print("  ① Docker里milvus-standalone容器是否在运行？（您截图里容器是running就没问题）")
            print(f"  ② 端口 {cfg.milvus.port} 是否监听？→ Docker Desktop看PORTS列是否为 {cfg.milvus.port}:{cfg.milvus.port}")
            print(f"  ③ .env中 MILVUS_HOST 是否 = {cfg.milvus.host} ? MILVUS_PORT = {cfg.milvus.port} ?")
            print("→ 自动回退Mock本地向量库，服务继续可用，知识存 data/vector_store_mock.pkl")
            print("=" * 70)
            return MockVectorStore()


# ============================================================
# 4. 多路检索器（对外统一入口）
# ============================================================
class HybridRetriever:
    """
    多路检索统一封装：
      - Query分类预判 -> 对应知识库加权
      - 稠密(Embedding) + 稀疏(BM25) 混合召回
      - 父子块 + MMR 去冗余
    """
    def __init__(self):
        cfg = get_config()
        self.cfg = cfg
        self.embed = EmbeddingFactory.create()
        self.store = VectorStoreFactory.create()
        self._category_weights = cfg.retrieval.category_weights
        self._top_k = cfg.retrieval.top_k

    # ---- 知识库写入 ----
    def ingest_chunks(self, chunks: List[KnowledgeChunk]) -> int:
        return self.store.add_chunks(chunks)

    # ---- 预判问题类别（提升召回精度）----
    def _predict_category(self, query: str) -> Optional[str]:
        q_lower = query.lower()
        # 物流关键词
        logi_kws = ["物流", "快递", "运费", "时效", "包裹", "清关", "派送", "fba", "dhl", "ems", "ups", "体积重", "丢件", "单号", "头程"]
        if any(k in q_lower for k in logi_kws):
            return "cross_border_logistics"
        # 售后关键词
        asp_kws = ["退货", "退款", "售后", "质保", "差评", "争议", "退换", "索赔", "拒付", "chargeback", "paypal", "召回"]
        if any(k in q_lower for k in asp_kws):
            return "after_sales_policy"
        # 合规关键词
        cus_kws = ["关税", "vat", "海关", "hs编码", "申报", "ioss", "认证", "fda", "合规", "税率", "rcep", "起征点", "nom", "saso"]
        if any(k in q_lower for k in cus_kws):
            return "customs_compliance"
        return None

    # ---- 对外检索 ----
    def retrieve(self, query: str,
                 top_k: Optional[int] = None,
                 category_filter: Optional[str] = None) -> List[VectorSearchResult]:
        """
        返回混合检索结果列表
        category_filter 可选：指定只搜某类知识库
        """
        top_k = top_k or self._top_k
        # 1) 预判类别
        predicted = self._predict_category(query)
        effective_filter = category_filter or predicted
        # 2) 编码
        q_vec = self.embed.encode([query])[0]
        # 3) 混合检索
        results = self.store.search(q_vec, query, top_k=top_k, category_filter=effective_filter)
        # 4) 分类加权（简历：多路检索权重调试）
        for r in results:
            cat_w = self._category_weights.get(r.chunk.category, 1.0)
            r.hybrid_score = round(r.hybrid_score * cat_w, 4)
        results.sort(key=lambda r: -r.hybrid_score)
        return results[:top_k]

    @property
    def total_indexed(self) -> int:
        return self.store.total_chunks
