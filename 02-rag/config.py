# -*- coding: utf-8 -*-
"""
星途环球·跨境售后RAG智能问答系统 - 全局配置
对应简历模块5：模型工程化部署（配置运维）
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _env(name: str, default: Any = None) -> Any:
    return os.environ.get(name, default)


# ============================================================
# 1. 三大知识库分类（对应简历：跨境物流/海外售后/各国合规）
# ============================================================
KNOWLEDGE_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "cross_border_logistics": {
        "name": "跨境物流咨询",
        "icon": "📦",
        "chunk_size": 300,
        "chunk_overlap": 50,
        "description": "国际物流渠道、时效报价、清关异常、FBA头程、危险品运输规则等",
    },
    "after_sales_policy": {
        "name": "海外售后政策",
        "icon": "🛒",
        "chunk_size": 350,
        "chunk_overlap": 60,
        "description": "亚马逊/eBay/Wish/独立站退换货规则、PayPal拒付、差评处理、召回SOP等",
    },
    "customs_compliance": {
        "name": "各国海关合规政策",
        "icon": "📜",
        "chunk_size": 400,
        "chunk_overlap": 70,
        "description": "各国关税起征点、VAT/IOSS、HS编码合规、RCEP优惠、FDA/NOM认证等",
    },
}


# ============================================================
# 2. 支持的多语种（对应简历：多语种结构化Prompt）
# ============================================================
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "zh": "中文",
    "en": "English",
    "ja": "日本語",
    "es": "Español",
    "de": "Deutsch",
    "fr": "Français",
}


# ============================================================
# 3. 大模型 & 向量配置（对应简历：Milvus稠密+稀疏双表征）
# ============================================================
@dataclass
class LLMConfig:
    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai_compatible"))
    base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.deepseek.com/v1"))
    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "sk-demo-key"))
    model: str = field(default_factory=lambda: _env("LLM_MODEL", "deepseek-chat"))
    temperature: float = 0.1
    max_tokens: int = 1500
    timeout: int = 60


@dataclass
class EmbeddingConfig:
    dense_provider: str = field(default_factory=lambda: _env("EMBED_PROVIDER", "mock"))
    dense_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "bge-m3"))
    dense_dim: int = 1024
    sparse_enabled: bool = True  # 稀疏向量 BM25/TF-IDF 表征
    mock_dim: int = 384  # Mock模式下的向量维度，方便离线运行


@dataclass
class MilvusConfig:
    host: str = field(default_factory=lambda: _env("MILVUS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(_env("MILVUS_PORT", "19530")))
    collection_prefix: str = "xingtu_crossborder"
    parent_child_enabled: bool = True  # 父子块多级索引
    hybrid_search_weight_dense: float = 0.6
    hybrid_search_weight_sparse: float = 0.4


# ============================================================
# 4. 检索 & 重排序配置（对应简历：多路检索权重、重排序模型）
# ============================================================
@dataclass
class RetrievalConfig:
    top_k: int = 10  # 多路检索初筛TopK
    rerank_top_k: int = 5  # 重排序后TopK进入Prompt
    rerank_provider: str = field(default_factory=lambda: _env("RERANK_PROVIDER", "mock"))  # mock / cross_encoder
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "bge-reranker-v2-m3"))
    # 三类知识差异化权重（可动态调优）
    category_weights: Dict[str, float] = field(default_factory=lambda: {
        "cross_border_logistics": 1.0,
        "after_sales_policy": 1.0,
        "customs_compliance": 1.05,  # 合规类加权重，降低幻觉
    })
    mmr_enabled: bool = True  # MMR多样性，筛除冗余向量
    mmr_lambda: float = 0.7


# ============================================================
# 5. 生成风控配置（对应简历：Prompt生成风控、跨境政策幻觉限定）
# ============================================================
@dataclass
class SafetyConfig:
    # 生成范围约束：严格引用检索内容
    strictly_use_retrieval: bool = True
    # 拒答条件：检索置信度低于阈值
    retrieval_confidence_threshold: float = 0.35
    # 幻觉检测正则（与test_evaluate保持一致，用于生产环节同步拦截）
    hallucination_block_answer: bool = True  # 高幻觉风险直接返回拒答
    hallucination_high_risk_threshold: float = 0.55
    # 合规类必须引用政策条款编号
    require_citation_for_compliance: bool = True


# ============================================================
# 6. FastAPI 服务配置
# ============================================================
@dataclass
class AgentsConfig:
    gateway_port: int = field(default_factory=lambda: int(_env("GATEWAY_PORT", "8000")))
    gateway_host: str = field(default_factory=lambda: _env("GATEWAY_HOST", "0.0.0.0"))
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 60
    frontend_dir: str = "frontend"


# ============================================================
# 7. 根配置类
# ============================================================
@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    data_dir: str = field(default_factory=lambda: _env("DATA_DIR", "data"))
    knowledge_dir: str = field(default_factory=lambda: _env("KNOWLEDGE_DIR", "data/knowledge"))
    ocr_languages: List[str] = field(default_factory=lambda: ["en", "ch", "ja", "es", "de", "fr"])


_CONFIG_INSTANCE: AppConfig | None = None


def get_config() -> AppConfig:
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None:
        _CONFIG_INSTANCE = AppConfig()
        os.makedirs(_CONFIG_INSTANCE.data_dir, exist_ok=True)
        os.makedirs(_CONFIG_INSTANCE.knowledge_dir, exist_ok=True)
    return _CONFIG_INSTANCE
