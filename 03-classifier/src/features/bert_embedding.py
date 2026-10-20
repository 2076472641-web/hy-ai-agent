# -*- coding: utf-8 -*-
"""
BERT嵌入提取模块
功能：
1. BERT嵌入提取（bert-base-chinese / 其他中文预训练模型）
2. CLS池化 / 均值池化 / 最大池化 / 多层融合池化
3. Prompt增强嵌入（手工设计分类提示模板）
4. 特征降维（PCA、TruncatedSVD）
5. 批量处理与磁盘缓存
"""

import os
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Literal

import numpy as np
from loguru import logger
from tqdm import tqdm

# 机器学习降维
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

from src import MODEL_DIR
from .tokenizer import ChineseTokenizer, CATEGORY_KEYWORDS


PoolingStrategy = Literal["cls", "mean", "max", "mean_max", "layer_mean"]


# 十二大类商品分类的Prompt模板，用于Prompt增强嵌入
PROMPT_TEMPLATES: Dict[str, List[str]] = {
    # 单标签直接分类模板
    "single": [
        "这是一件【{text}】商品，它属于{CATEGORY}类目。",
    ],
    # 前缀提示模板（用于 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>  token 学习增强语义）
    "prefix": [
        "下面是商品标题：{text}。该商品的类目是：",
        "商品信息：{text}。请判断它的类别：",
        "这是【{text}】，它属于什么类目？",
    ],
    # 对比式模板（配合CLS+MLM头使用可获取更丰富信息）
    "contrast": [
        "【{text}】是不是服装鞋帽类商品？",
        "【{text}】是不是数码家电类商品？",
        "【{text}】是不是美妆护肤类商品？",
        "【{text}】是不是食品生鲜类商品？",
        "【{text}】是不是家居家装类商品？",
        "【{text}】是不是母婴用品类商品？",
        "【{text}】是不是运动户外类商品？",
        "【{text}】是不是图书文娱类商品？",
        "【{text}】是不是珠宝饰品类商品？",
        "【{text}】是不是汽车用品类商品？",
        "【{text}】是不是宠物用品类商品？",
        "【{text}】是不是医药保健类商品？",
    ],
    # 详细属性模板
    "attributes": [
        "商品名称：{text}。商品类型、用途、属性：",
    ],
}


class BertEmbedder:
    """
    BERT嵌入提取器
    支持 bert-base-chinese 等中文预训练模型，提供多种池化策略与Prompt增强
    """

    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        max_seq_length: int = 128,
        pooling_strategy: PoolingStrategy = "cls",
        batch_size: int = 32,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        truncate_strategy: str = "head_tail",
        tokenizer: Optional[ChineseTokenizer] = None,
    ):
        """
        初始化BERT嵌入提取器

        Args:
            model_name: HuggingFace模型名或本地路径，默认 bert-base-chinese
            max_seq_length: 最大序列长度（含特殊token）
            pooling_strategy: 池化策略
                - "cls": 使用 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token表示
                - "mean": 最后一层隐藏层token均值（去掉PAD）
                - "max": 最后一层隐藏层token最大池化
                - "mean_max": mean与max拼接
                - "layer_mean": 最后4层隐藏层的 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 平均
            batch_size: 推理批次大小
            device: "cuda" / "cpu" / "auto"；None 时自动选择
            cache_dir: 嵌入缓存目录，默认 data/models/bert_cache
            use_cache: 是否启用磁盘缓存（相同文本+参数直接读取缓存）
            truncate_strategy: 超长文本截断策略（传递给ChineseTokenizer）
            tokenizer: 复用的中文分词器（用于长文本截断）
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.pooling_strategy = pooling_strategy
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.truncate_strategy = truncate_strategy
        self.text_tokenizer = tokenizer or ChineseTokenizer(
            max_seq_length=max_seq_length, truncate_strategy=truncate_strategy
        )

        # 缓存目录
        self.cache_dir = Path(cache_dir) if cache_dir else MODEL_DIR / "bert_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 降维模型
        self._reducer = None
        self._reducer_type: Optional[str] = None
        self._target_dim: Optional[int] = None
        self._scaler: Optional[StandardScaler] = None

        # 自动选择设备
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device

        # 懒加载HF模型和分词器
        self._hf_tokenizer = None
        self._hf_model = None
        self._embedding_dim: Optional[int] = None

        logger.info(
            f"BertEmbedder初始化 | 模型: {model_name} | 池化: {pooling_strategy} | "
            f"最大长度: {max_seq_length} | 设备: {device} | 缓存: {use_cache}"
        )

    # ==================== 懒加载 HuggingFace 模型 ====================

    def _load_hf(self) -> None:
        """懒加载 HuggingFace tokenizer与model"""
        if self._hf_tokenizer is not None and self._hf_model is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except ImportError as e:
            raise ImportError(
                "需要 transformers 和 torch 才能使用 BertEmbedder。\n"
                "请安装: pip install torch transformers"
            ) from e

        logger.info(f"正在加载 HuggingFace 模型: {self.model_name}")
        try:
            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, model_max_length=self.max_seq_length
            )
        except Exception as e:
            raise RuntimeError(f"加载分词器失败: {e}")

        try:
            self._hf_model = AutoModel.from_pretrained(self.model_name)
            self._hf_model.to(self.device)
            self._hf_model.eval()
        except Exception as e:
            raise RuntimeError(f"加载BERT模型失败: {e}")

        # 获取嵌入维度
        with torch.no_grad():
            dummy = self._hf_tokenizer(
                ["测试"], return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            out = self._hf_model(**dummy, output_hidden_states=True)
            last_hidden = out.last_hidden_state
            self._embedding_dim = last_hidden.shape[-1]

        logger.info(
            f"模型加载完成 | 嵌入维度: {self._embedding_dim} | "
            f"参数量: {sum(p.numel() for p in self._hf_model.parameters()) / 1e6:.1f}M"
        )

    @property
    def embedding_dim(self) -> int:
        """返回嵌入维度（未加载模型时触发加载）"""
        if self._embedding_dim is None:
            self._load_hf()
        return self._embedding_dim

    # ==================== 缓存键 ====================

    def _cache_key(
        self,
        texts: List[str],
        prompt_template: Optional[str] = None,
        pooling: Optional[str] = None,
    ) -> str:
        """生成缓存键（基于文本、模型、池化、模板等唯一标识）"""
        pooling = pooling or self.pooling_strategy
        key_parts = [
            self.model_name,
            str(self.max_seq_length),
            pooling,
            str(prompt_template),
            str(self.truncate_strategy),
            "|||".join(sorted(set(texts))[:200]),  # 防止键过长
            str(len(texts)),
        ]
        raw = "###".join(key_parts)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _save_cache(self, cache_key: str, embeddings: np.ndarray) -> None:
        """保存嵌入到磁盘缓存"""
        path = self.cache_dir / f"{cache_key}.npy"
        try:
            np.save(str(path), embeddings)
            meta_path = self.cache_dir / f"{cache_key}.meta.pkl"
            with open(meta_path, "wb") as f:
                pickle.dump(
                    {
                        "model_name": self.model_name,
                        "pooling": self.pooling_strategy,
                        "max_seq_length": self.max_seq_length,
                        "shape": embeddings.shape,
                    },
                    f,
                )
        except Exception as e:
            logger.warning(f"嵌入缓存保存失败: {e}")

    def _load_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """从磁盘缓存加载嵌入"""
        path = self.cache_dir / f"{cache_key}.npy"
        if not path.exists():
            return None
        try:
            emb = np.load(str(path))
            logger.debug(f"缓存命中: {cache_key} -> shape={emb.shape}")
            return emb
        except Exception as e:
            logger.warning(f"缓存读取失败: {e}")
            return None

    # ==================== 核心嵌入计算 ====================

    def _encode_batch(
        self,
        texts: List[str],
        pooling: PoolingStrategy,
    ) -> np.ndarray:
        """
        对一批文本进行编码，返回嵌入矩阵

        Args:
            texts: 一批文本
            pooling: 池化策略

        Returns:
            (batch_size, hidden_dim) 或 (batch_size, 2*hidden_dim) 的嵌入
        """
        import torch

        self._load_hf()
        assert self._hf_tokenizer is not None and self._hf_model is not None

        tokens = self._hf_tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        token_type_ids = tokens.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        with torch.no_grad():
            outputs = self._hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=(pooling == "layer_mean"),
            )

        last_hidden = outputs.last_hidden_state  # [B, L, D]
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]

        if pooling == "cls":
            emb = last_hidden[:, 0, :]

        elif pooling == "mean":
            sum_hidden = (last_hidden * mask).sum(dim=1)  # [B, D]
            count = mask.sum(dim=1).clamp(min=1e-9)
            emb = sum_hidden / count

        elif pooling == "max":
            # PAD位置置为极小值
            masked_hidden = last_hidden.masked_fill(mask == 0, -1e9)
            emb = masked_hidden.max(dim=1).values

        elif pooling == "mean_max":
            sum_hidden = (last_hidden * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-9)
            mean_emb = sum_hidden / count

            masked_hidden = last_hidden.masked_fill(mask == 0, -1e9)
            max_emb = masked_hidden.max(dim=1).values
            emb = torch.cat([mean_emb, max_emb], dim=-1)

        elif pooling == "layer_mean":
            # 取最后4层的 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token，并求平均
            hidden_states = outputs.hidden_states  # Tuple of [B, L, D]
            last4 = torch.stack(hidden_states[-4:], dim=0)  # [4, B, L, D]
            last4_cls = last4[:, :, 0, :]  # [4, B, D]
            emb = last4_cls.mean(dim=0)  # [B, D]

        else:
            raise ValueError(f"未知池化策略: {pooling}")

        return emb.cpu().numpy().astype(np.float32)

    def encode(
        self,
        texts: Union[str, List[str]],
        pooling: Optional[PoolingStrategy] = None,
        prompt_template: Optional[str] = None,
        batch_size: Optional[int] = None,
        use_cache: Optional[bool] = None,
    ) -> np.ndarray:
        """
        编码文本为BERT嵌入向量（主入口）

        Args:
            texts: 单条或多条文本
            pooling: 池化策略，None则使用初始化时的设定
            prompt_template: Prompt模板字符串，如 "商品：{text} 属于什么类目？"
                也可直接传入 PROMPT_TEMPLATES 中的键 "prefix" / "single" 等
            batch_size: 批次大小
            use_cache: 是否使用缓存

        Returns:
            嵌入矩阵 (n_samples, dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        pooling = pooling or self.pooling_strategy
        batch_size = batch_size or self.batch_size
        use_cache = self.use_cache if use_cache is None else use_cache

        # Prompt模板处理
        final_prompt_template = None
        if prompt_template is not None:
            if prompt_template in PROMPT_TEMPLATES:
                # 取该模板组中的第一条
                final_prompt_template = PROMPT_TEMPLATES[prompt_template][0]
            else:
                final_prompt_template = prompt_template

        # 对超长文本进行截断（保持与分类任务的分词策略一致）
        processed_texts = []
        for t in texts:
            chunks = self.text_tokenizer.truncate_for_transformer(
                t, max_length=self.max_seq_length - 10
            )
            processed_texts.append(chunks[0])

        # 应用Prompt模板
        if final_prompt_template is not None:
            try:
                texts_for_model = [
                    final_prompt_template.format(text=t) for t in processed_texts
                ]
            except Exception as e:
                logger.warning(f"Prompt模板渲染失败，回退原始文本: {e}")
                texts_for_model = processed_texts
        else:
            texts_for_model = processed_texts

        # 查缓存
        cache_key = None
        if use_cache:
            cache_key = self._cache_key(processed_texts, final_prompt_template, pooling)
            cached = self._load_cache(cache_key)
            if cached is not None and len(cached) == len(texts):
                return cached

        # 批量推理
        n = len(texts_for_model)
        embeddings = []
        iterator = tqdm(
            range(0, n, batch_size),
            desc=f"BERT编码 [{pooling}]",
            ncols=80,
            leave=False,
        )
        for start in iterator:
            batch = texts_for_model[start:start + batch_size]
            try:
                batch_emb = self._encode_batch(batch, pooling)
            except Exception as e:
                logger.error(f"批次编码失败 start={start}: {e}")
                raise
            embeddings.append(batch_emb)

        if not embeddings:
            dim = self.embedding_dim
            if pooling == "mean_max":
                dim *= 2
            return np.zeros((0, dim), dtype=np.float32)

        result = np.vstack(embeddings)

        # 保存缓存
        if use_cache and cache_key is not None:
            self._save_cache(cache_key, result)

        return result

    # ==================== Prompt 增强嵌入 ====================

    def encode_with_prompt_ensemble(
        self,
        texts: Union[str, List[str]],
        template_keys: Optional[List[str]] = None,
        pooling: Optional[PoolingStrategy] = None,
        reduction: str = "mean",
    ) -> np.ndarray:
        """
        使用多个Prompt模板编码并集成，生成更鲁棒的嵌入

        Args:
            texts: 文本或文本列表
            template_keys: 模板组名列表，默认为 ["single", "prefix", "attributes"]
            pooling: 池化策略
            reduction: 多模板结果的集成方式
                - "mean": 逐元素平均
                - "concat": 拼接

        Returns:
            集成后的嵌入矩阵
        """
        if template_keys is None:
            template_keys = ["single", "prefix", "attributes"]

        embs_list = []
        for key in template_keys:
            if key not in PROMPT_TEMPLATES:
                logger.warning(f"未知Prompt模板: {key}，跳过")
                continue
            # 取该模板组中的第一条
            tmpl = PROMPT_TEMPLATES[key][0]
            embs_list.append(self.encode(texts, pooling=pooling, prompt_template=tmpl))

        if not embs_list:
            return self.encode(texts, pooling=pooling)

        if reduction == "mean":
            stacked = np.stack(embs_list, axis=0)
            return stacked.mean(axis=0).astype(np.float32)
        elif reduction == "concat":
            return np.hstack(embs_list).astype(np.float32)
        else:
            raise ValueError(f"未知reduction方式: {reduction}")

    def encode_with_category_prompts(
        self,
        texts: Union[str, List[str]],
        pooling: Optional[PoolingStrategy] = None,
    ) -> np.ndarray:
        """
        使用十二大分类的对比式Prompt编码，生成 (n, 12, dim) 的类别感知嵌入

        Args:
            texts: 文本或文本列表
            pooling: 池化策略

        Returns:
            嵌入矩阵 (n_samples, 12, dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        contrast_templates = PROMPT_TEMPLATES["contrast"]  # 12条
        n = len(texts)
        per_emb = self.encode(
            [tmpl.format(text=t) for t in texts for tmpl in contrast_templates],
            pooling=pooling,
        )
        dim = per_emb.shape[-1]
        return per_emb.reshape(n, 12, dim).astype(np.float32)

    # ==================== 特征降维 ====================

    def fit_reducer(
        self,
        X: np.ndarray,
        method: Literal["pca", "svd"] = "pca",
        n_components: int = 128,
        scale: bool = True,
    ) -> "BertEmbedder":
        """
        拟合降维模型（在训练集上拟合，推理集上复用）

        Args:
            X: 训练集嵌入 (n_train, dim)
            method: "pca" 稠密矩阵 / "svd" 支持稀疏
            n_components: 目标维度
            scale: 是否标准化（仅PCA）

        Returns:
            self
        """
        n_components = min(n_components, X.shape[0], X.shape[1])
        logger.info(
            f"拟合降维模型 | 方法: {method} | {X.shape[1]} -> {n_components}"
        )

        if scale and method == "pca":
            self._scaler = StandardScaler()
            X_in = self._scaler.fit_transform(X)
        else:
            self._scaler = None
            X_in = X

        if method == "pca":
            self._reducer = PCA(n_components=n_components, random_state=42)
        elif method == "svd":
            self._reducer = TruncatedSVD(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"未知降维方法: {method}")

        self._reducer.fit(X_in)
        self._reducer_type = method
        self._target_dim = n_components

        if hasattr(self._reducer, "explained_variance_ratio_"):
            total_var = self._reducer.explained_variance_ratio_.sum()
            logger.info(f"累计可解释方差: {total_var:.4f}")

        return self

    def reduce_dim(self, X: np.ndarray) -> np.ndarray:
        """使用训练好的降维模型进行降维"""
        if self._reducer is None:
            raise RuntimeError("降维模型未拟合，请先调用 fit_reducer()")

        if self._scaler is not None:
            X_in = self._scaler.transform(X)
        else:
            X_in = X

        return self._reducer.transform(X_in).astype(np.float32)

    # ==================== 保存/加载 ====================

    def save(self, name: str = "bert_embedder") -> str:
        """保存配置与降维模型到磁盘（不保存HF大模型权重）"""
        save_dir = MODEL_DIR / "bert_configs"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{name}.pkl"

        data = {
            "config": {
                "model_name": self.model_name,
                "max_seq_length": self.max_seq_length,
                "pooling_strategy": self.pooling_strategy,
                "batch_size": self.batch_size,
                "truncate_strategy": self.truncate_strategy,
                "device": self.device,
            },
            "reducer": self._reducer,
            "reducer_type": self._reducer_type,
            "target_dim": self._target_dim,
            "scaler": self._scaler,
            "embedding_dim": self._embedding_dim,
        }
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"BertEmbedder配置已保存: {save_path}")
        return str(save_path)

    @classmethod
    def load(cls, path: str) -> "BertEmbedder":
        """从磁盘加载 BertEmbedder 配置与降维模型"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        cfg = data["config"]
        inst = cls(
            model_name=cfg["model_name"],
            max_seq_length=cfg["max_seq_length"],
            pooling_strategy=cfg["pooling_strategy"],
            batch_size=cfg["batch_size"],
            device=cfg["device"],
            truncate_strategy=cfg["truncate_strategy"],
        )
        inst._reducer = data.get("reducer")
        inst._reducer_type = data.get("reducer_type")
        inst._target_dim = data.get("target_dim")
        inst._scaler = data.get("scaler")
        inst._embedding_dim = data.get("embedding_dim")

        logger.info(f"BertEmbedder配置已加载: {path}")
        return inst

    def __repr__(self) -> str:
        return (
            f"BertEmbedder(model={self.model_name}, pooling={self.pooling_strategy}, "
            f"max_len={self.max_seq_length}, device={self.device}, "
            f"dim={self._embedding_dim or '未加载'})"
        )
