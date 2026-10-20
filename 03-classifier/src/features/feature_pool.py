# -*- coding: utf-8 -*-
"""
特征池模块
功能：
1. 无监督/有监督双路径特征池构建
   - 无监督：TF-IDF词级、TF-IDF字符级、Ngram词频、Word2Vec、BERT嵌入（CLS/Mean/LayerMean/Prompt）
   - 有监督：卡方选、互信息、树模型特征重要性、Wrapper搜索
2. 冗余特征剔除
   - 方差过滤（低方差特征）
   - 相关性过滤（高相关性冗余特征）
   - 多重共线性检测（VIF）
3. 特征选择
   - 过滤式（Filter）：卡方、互信息、F值
   - 包裹式（Wrapper）：RFE、前向/后向搜索
   - 嵌入式（Embedded）：L1正则、树模型
4. 特征压缩
   - PCA / TruncatedSVD
   - 特征哈希（FeatureHasher）
   - 基于互信息的特征聚合
5. 模块组合与自动筛选最优特征子集
"""

import os
import copy
import pickle
import time
from itertools import combinations
from pathlib import Path
from typing import (
    List, Dict, Optional, Tuple, Union, Any, Literal, Callable
)

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from loguru import logger
from tqdm import tqdm

# Scikit-learn 特征选择与降维
from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif, mutual_info_classif,
    RFE, RFECV, SelectFromModel, VarianceThreshold,
)
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_macro

# 本地模块
from src import MODEL_DIR
from .tokenizer import ChineseTokenizer
from .text_vectorizer import TextVectorizer
from .bert_embedding import BertEmbedder, PoolingStrategy


# 十二大类类目名称
CATEGORY_NAMES = [
    "服装鞋帽", "数码家电", "美妆护肤", "食品生鲜",
    "家居家装", "母婴用品", "运动户外", "图书文娱",
    "珠宝饰品", "汽车用品", "宠物用品", "医药保健",
]


class FeaturePool:
    """
    特征池：统一管理无监督/有监督双路径特征，支持冗余剔除、选择、压缩与组合搜索
    """

    def __init__(
        self,
        tokenizer: Optional[ChineseTokenizer] = None,
        vectorizer: Optional[TextVectorizer] = None,
        bert_embedder: Optional[BertEmbedder] = None,
        save_dir: Optional[str] = None,
        random_state: int = 42,
    ):
        """
        初始化特征池

        Args:
            tokenizer: 复用的中文分词器
            vectorizer: 复用的文本向量化器
            bert_embedder: 复用的BERT嵌入器
            save_dir: 保存目录，默认 data/models/feature_pool
            random_state: 随机种子
        """
        self.tokenizer = tokenizer or ChineseTokenizer()
        self.vectorizer = vectorizer or TextVectorizer(tokenizer=self.tokenizer)
        self.bert_embedder = bert_embedder
        self.random_state = random_state
        self.save_dir = Path(save_dir) if save_dir else MODEL_DIR / "feature_pool"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 特征存储字典：name -> (matrix, metadata_dict)
        self._feature_store: Dict[str, Tuple[Any, Dict]] = {}

        # 冗余剔除记录
        self._low_variance_mask: Optional[np.ndarray] = None
        self._correlation_mask: Optional[np.ndarray] = None

        # 特征选择器
        self._feature_selectors: Dict[str, Any] = {}

        # 特征压缩/降维模型
        self._compressor: Any = None
        self._compressor_name: Optional[str] = None

        # 组合搜索记录
        self._combination_log: List[Dict] = []

        logger.info(
            f"FeaturePool初始化完成 | 复用组件: "
            f"Tokenizer={self.tokenizer is not None}, "
            f"Vectorizer={self.vectorizer is not None}, "
            f"BertEmbedder={self.bert_embedder is not None}"
        )

    # ==================== 基础工具 ====================

    @staticmethod
    def _to_dense(X: Any) -> np.ndarray:
        """将稀疏矩阵转为稠密，必要时copy"""
        if sparse.issparse(X):
            return X.toarray() if hasattr(X, "toarray") else np.asarray(X.todense())
        return np.asarray(X, dtype=np.float32)

    @staticmethod
    def _to_sparse(X: Any) -> sparse.csr_matrix:
        """转为CSR稀疏矩阵"""
        if sparse.issparse(X):
            return X.tocsr()
        return sparse.csr_matrix(np.asarray(X, dtype=np.float32))

    def register_feature(
        self,
        name: str,
        X: Any,
        module: str = "custom",
        description: str = "",
    ) -> "FeaturePool":
        """
        注册自定义特征到池中

        Args:
            name: 特征名（唯一）
            X: 特征矩阵 (n_samples, n_features)
            module: 来源模块名
            description: 描述

        Returns:
            self
        """
        if name in self._feature_store:
            logger.warning(f"特征 {name} 已存在，将被覆盖")
        self._feature_store[name] = (
            X,
            {
                "module": module,
                "description": description,
                "n_samples": X.shape[0] if hasattr(X, "shape") else len(X),
                "n_features": X.shape[1] if hasattr(X, "shape") and len(X.shape) > 1 else 1,
                "is_sparse": sparse.issparse(X),
                "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        logger.info(f"注册特征 {name} | 维度={self._feature_store[name][1]['n_features']} | 来源={module}")
        return self

    def list_features(self) -> pd.DataFrame:
        """列出池中所有特征的元信息"""
        rows = []
        for name, (_, meta) in self._feature_store.items():
            rows.append({"name": name, **meta})
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["module", "n_features"], ascending=[True, False])
        return df.reset_index(drop=True)

    def clear(self) -> None:
        """清空特征池"""
        self._feature_store.clear()
        self._low_variance_mask = None
        self._correlation_mask = None
        self._feature_selectors.clear()
        self._compressor = None
        logger.info("特征池已清空")

    # ==================== 路径一：无监督特征构建 ====================

    def build_unsupervised_features(
        self,
        texts: List[str],
        enable_tfidf_word: bool = True,
        enable_tfidf_char: bool = True,
        enable_ngram_count: bool = False,
        enable_word2vec: bool = False,
        word2vec_pooling: str = "mean_max",
        enable_bert: bool = True,
        bert_pooling_list: Optional[List[PoolingStrategy]] = None,
        bert_prompt_ensemble: bool = False,
    ) -> "FeaturePool":
        """
        构建无监督特征池（在训练集上拟合并存储）

        Args:
            texts: 原始文本列表
            enable_tfidf_word: 启用词级TF-IDF
            enable_tfidf_char: 启用字符级TF-IDF
            enable_ngram_count: 启用Ngram词频
            enable_word2vec: 启用Word2Vec
            word2vec_pooling: Word2Vec池化方式
            enable_bert: 启用BERT嵌入
            bert_pooling_list: BERT池化策略列表，默认 ["cls", "mean_max"]
            bert_prompt_ensemble: 是否额外包含Prompt集成嵌入

        Returns:
            self
        """
        logger.info("========== 开始构建无监督特征池 ==========")
        start_time = time.time()

        # TF-IDF + Ngram 特征
        self.vectorizer.fit_all(
            texts,
            enable_word_tfidf=enable_tfidf_word,
            enable_char_tfidf=enable_tfidf_char,
            enable_ngram_count=enable_ngram_count,
            enable_word2vec=enable_word2vec,
        )

        if enable_tfidf_word and self.vectorizer.tfidf_word is not None:
            X = self.vectorizer.transform_tfidf_word(texts)
            self.register_feature("tfidf_word", X, module="text_vectorizer",
                                  description="词级TF-IDF，1~2gram")

        if enable_tfidf_char and self.vectorizer.tfidf_char is not None:
            X = self.vectorizer.transform_tfidf_char(texts)
            self.register_feature("tfidf_char", X, module="text_vectorizer",
                                  description="字符级TF-IDF，2~4gram")

        if enable_ngram_count and self.vectorizer.count_ngram is not None:
            X = self.vectorizer.transform_ngram_count(texts)
            self.register_feature("ngram_count", X, module="text_vectorizer",
                                  description="Ngram词频，L2归一化")

        if enable_word2vec and self.vectorizer.word2vec_model is not None:
            X = self.vectorizer.transform_word2vec(texts, pooling=word2vec_pooling)
            self.register_feature(
                f"word2vec_{word2vec_pooling}", X, module="text_vectorizer",
                description=f"Word2Vec嵌入，{word2vec_pooling}池化"
            )

        # BERT 嵌入特征
        if enable_bert:
            if self.bert_embedder is None:
                logger.info("自动初始化 BertEmbedder (bert-base-chinese)")
                self.bert_embedder = BertEmbedder(
                    model_name="bert-base-chinese",
                    pooling_strategy="cls",
                )

            if bert_pooling_list is None:
                bert_pooling_list = ["cls", "mean_max"]

            for pooling in bert_pooling_list:
                try:
                    X = self.bert_embedder.encode(texts, pooling=pooling)
                    self.register_feature(
                        f"bert_{pooling}", X, module="bert_embedding",
                        description=f"BERT嵌入，{pooling}池化"
                    )
                except Exception as e:
                    logger.warning(f"BERT池化 {pooling} 失败: {e}")

            if bert_prompt_ensemble:
                try:
                    X = self.bert_embedder.encode_with_prompt_ensemble(
                        texts, reduction="mean"
                    )
                    self.register_feature(
                        "bert_prompt_ensemble", X, module="bert_embedding",
                        description="多Prompt模板BERT集成嵌入"
                    )
                except Exception as e:
                    logger.warning(f"BERT Prompt集成失败: {e}")

        elapsed = time.time() - start_time
        total_feat = sum(m[1]["n_features"] for m in self._feature_store.values())
        logger.info(
            f"========== 无监督特征池构建完成 | 用时 {elapsed:.1f}s | "
            f"共 {len(self._feature_store)} 组特征，{total_feat} 维 =========="
        )
        return self

    # ==================== 路径二：有监督特征构建 ====================

    def build_supervised_features(
        self,
        texts: Optional[List[str]],
        labels: Union[List[int], np.ndarray],
        use_chi2: bool = True,
        use_mutual_info: bool = False,
        use_tree_importance: bool = True,
        k_per_module: int = 1000,
    ) -> "FeaturePool":
        """
        构建有监督特征：在无监督特征基础上进一步做特征选择派生

        Args:
            texts: 原始文本（需要时用于评估，但通常可以不传用已注册特征）
            labels: 标签数组
            use_chi2: 启用卡方筛选特征
            use_mutual_info: 启用互信息筛选特征
            use_tree_importance: 启用树模型重要性筛选特征
            k_per_module: 每个无监督模块保留的Top-K

        Returns:
            self
        """
        logger.info("========== 开始构建有监督特征池 ==========")
        start_time = time.time()
        y = np.asarray(labels)

        unsupervised_items = list(self._feature_store.items())
        if not unsupervised_items:
            raise RuntimeError("请先调用 build_unsupervised_features() 或手动 register_feature()")

        for fname, (X, meta) in unsupervised_items:
            module = meta["module"]
            n_feat = meta["n_features"]
            k = min(k_per_module, n_feat)

            if use_chi2:
                try:
                    X_chi2, selector = self.vectorizer.select_k_best(
                        X, y, k=k, score_func="chi2"
                    )
                    key = f"{fname}_chi2_top{k}"
                    self.register_feature(
                        key, X_chi2, module=f"{module}_supervised",
                        description=f"卡方 Top-{k} 特征 (源自 {fname})"
                    )
                    self._feature_selectors[key] = selector
                except Exception as e:
                    logger.warning(f"{fname} 卡方筛选失败: {e}")

            if use_mutual_info:
                try:
                    X_mi, selector = self.vectorizer.select_k_best(
                        X, y, k=k, score_func="mutual_info"
                    )
                    key = f"{fname}_mi_top{k}"
                    self.register_feature(
                        key, X_mi, module=f"{module}_supervised",
                        description=f"互信息 Top-{k} 特征 (源自 {fname})"
                    )
                    self._feature_selectors[key] = selector
                except Exception as e:
                    logger.warning(f"{fname} 互信息筛选失败: {e}")

            if use_tree_importance:
                try:
                    rf = RandomForestClassifier(
                        n_estimators=200, max_depth=30,
                        random_state=self.random_state, n_jobs=-1
                    )
                    X_dense = self._to_dense(X) if not sparse.issparse(X) else X
                    selector = SelectFromModel(rf, threshold="mean", max_features=k)
                    X_sel = selector.fit_transform(X_dense, y)
                    key = f"{fname}_tree_sel"
                    self.register_feature(
                        key, X_sel, module=f"{module}_supervised",
                        description=f"树模型特征选择 (源自 {fname}, 平均阈值)"
                    )
                    self._feature_selectors[key] = selector
                except Exception as e:
                    logger.warning(f"{fname} 树模型选择失败: {e}")

        elapsed = time.time() - start_time
        logger.info(
            f"========== 有监督特征池构建完成 | 用时 {elapsed:.1f}s | "
            f"当前共 {len(self._feature_store)} 组特征 =========="
        )
        return self

    # ==================== 冗余特征剔除 ====================

    def remove_low_variance(
        self,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.0,
        apply_all: bool = True,
    ) -> "FeaturePool":
        """
        去除低方差特征

        Args:
            feature_names: 要处理的特征名列表，None 表示全部
            threshold: 方差阈值，默认 0.0 去除恒值特征
            apply_all: 是否对池中所有匹配特征原地更新

        Returns:
            self
        """
        target_names = feature_names or list(self._feature_store.keys())
        logger.info(
            f"开始低方差过滤 | 阈值: {threshold} | 待处理: {len(target_names)} 组"
        )

        for fname in target_names:
            if fname not in self._feature_store:
                continue
            X, meta = self._feature_store[fname]
            n_feat = meta["n_features"]

            try:
                # 对稠密矩阵用numpy计算方差，稀疏矩阵取列方差
                if sparse.issparse(X):
                    X_csc = X.tocsc()
                    variances = np.zeros(X_csc.shape[1], dtype=np.float32)
                    # 逐列计算
                    for i in range(X_csc.shape[1]):
                        col = X_csc.getcol(i).data
                        variances[i] = col.var() if len(col) > 0 else 0.0
                else:
                    X_dense = np.asarray(X, dtype=np.float32)
                    variances = X_dense.var(axis=0)

                mask = variances > threshold
                n_kept = mask.sum()

                if n_kept < n_feat:
                    if apply_all:
                        if sparse.issparse(X):
                            X_new = X.tocsc()[:, mask].tocsr()
                        else:
                            X_new = np.asarray(X)[:, mask]
                        self._feature_store[fname] = (X_new, {
                            **meta,
                            "n_features": int(n_kept),
                            "low_variance_filtered": int(n_feat - n_kept),
                        })
                    logger.info(
                        f"  {fname}: {n_feat} -> {n_kept} 维 (剔除 {n_feat - n_kept})"
                    )
            except Exception as e:
                logger.warning(f"  {fname} 低方差过滤失败: {e}")

        return self

    def remove_highly_correlated(
        self,
        feature_names: Optional[List[str]] = None,
        corr_threshold: float = 0.95,
        method: Literal["pearson", "spearman"] = "pearson",
        sample_size: int = 10000,
    ) -> "FeaturePool":
        """
        去除高度相关（冗余）特征，每次剔除相关特征对中方差较小者

        Args:
            feature_names: 处理的特征名列表
            corr_threshold: 相关系数阈值，默认 0.95
            method: 相关系数计算方式
            sample_size: 用于计算相关矩阵的样本数（大样本下随机抽样加速）

        Returns:
            self
        """
        target_names = feature_names or list(self._feature_store.keys())
        logger.info(
            f"开始高相关特征过滤 | 阈值: {corr_threshold} ({method}) | "
            f"待处理: {len(target_names)} 组"
        )

        for fname in target_names:
            if fname not in self._feature_store:
                continue
            X, meta = self._feature_store[fname]
            n_feat = meta["n_features"]

            # 高维稀疏不适合直接相关矩阵（内存爆），跳过或降维再做
            if n_feat > 3000 and sparse.issparse(X):
                logger.info(f"  {fname}: 维度 {n_feat} > 3000，跳过高相关过滤")
                continue

            try:
                X_dense = self._to_dense(X)
                n = X_dense.shape[0]

                # 抽样
                if n > sample_size:
                    rng = np.random.default_rng(self.random_state)
                    idx = rng.choice(n, size=sample_size, replace=False)
                    X_sample = X_dense[idx]
                else:
                    X_sample = X_dense

                # 方差
                variances = X_sample.var(axis=0)

                # 相关矩阵（上三角）
                if method == "pearson":
                    corr_mat = np.corrcoef(X_sample, rowvar=False)
                else:
                    # Spearman 用排序 + Pearson
                    rank = np.argsort(np.argsort(X_sample, axis=0), axis=0).astype(np.float32)
                    corr_mat = np.corrcoef(rank, rowvar=False)

                if corr_mat.ndim < 2:
                    continue

                n_vars = corr_mat.shape[0]
                keep_mask = np.ones(n_vars, dtype=bool)

                # 贪心剔除：遍历上三角，若超阈值则剔除方差小的列
                for i in range(n_vars - 1):
                    if not keep_mask[i]:
                        continue
                    for j in range(i + 1, n_vars):
                        if not keep_mask[j]:
                            continue
                        if abs(corr_mat[i, j]) > corr_threshold:
                            if variances[i] >= variances[j]:
                                keep_mask[j] = False
                            else:
                                keep_mask[i] = False
                                break

                n_kept = keep_mask.sum()
                if n_kept < n_feat:
                    X_new = np.asarray(X_dense)[:, keep_mask]
                    if sparse.issparse(self._feature_store[fname][0]):
                        X_new = sparse.csr_matrix(X_new)
                    self._feature_store[fname] = (X_new, {
                        **meta,
                        "n_features": int(n_kept),
                        "high_corr_filtered": int(n_feat - n_kept),
                    })
                    logger.info(
                        f"  {fname}: {n_feat} -> {n_kept} 维 (剔除高相关 {n_feat - n_kept})"
                    )
            except Exception as e:
                logger.warning(f"  {fname} 高相关过滤失败: {e}")

        return self

    def prune_redundant(
        self,
        feature_names: Optional[List[str]] = None,
        var_threshold: float = 1e-8,
        corr_threshold: float = 0.95,
    ) -> "FeaturePool":
        """
        一站式冗余剔除：低方差 -> 高相关

        Args:
            feature_names: 目标特征名列表
            var_threshold: 方差阈值
            corr_threshold: 相关系数阈值

        Returns:
            self
        """
        logger.info("========== 开始冗余特征剔除 ==========")
        self.remove_low_variance(feature_names, threshold=var_threshold)
        self.remove_highly_correlated(feature_names, corr_threshold=corr_threshold)
        logger.info("========== 冗余特征剔除完成 ==========")
        return self

    # ==================== 特征选择（基于标签的进一步筛选） ====================

    def select_features(
        self,
        y: Union[List[int], np.ndarray],
        method: Literal["chi2", "mutual_info", "f_classif", "rfe", "sfm_l1", "sfm_tree"] = "chi2",
        feature_names: Optional[List[str]] = None,
        n_features: int = 1500,
        overwrite: bool = True,
        cv: int = 3,
    ) -> "FeaturePool":
        """
        对现有特征（按模块分别）进行有监督特征选择

        Args:
            y: 类目标签
            method: 选择方法
                - "chi2" / "mutual_info" / "f_classif": 过滤式
                - "rfe": 递归特征消除（基于LogReg）
                - "sfm_l1": L1正则嵌入式 (SelectFromModel + LogisticRegression)
                - "sfm_tree": 树模型嵌入式 (SelectFromModel + ExtraTrees)
            feature_names: 目标特征列表，None 表示全部
            n_features: 目标保留特征数
            overwrite: True=覆盖原特征；False=新建 *_selected 特征
            cv: RFE/RFECV用的折数

        Returns:
            self
        """
        y_arr = np.asarray(y)
        target_names = feature_names or list(self._feature_store.keys())
        logger.info(
            f"开始有监督特征选择 | 方法: {method} | 目标维度: {n_features} | "
            f"待处理: {len(target_names)} 组"
        )

        for fname in target_names:
            if fname not in self._feature_store:
                continue
            X, meta = self._feature_store[fname]
            n_feat = meta["n_features"]
            k = min(n_features, n_feat)
            try:
                if method in ("chi2", "mutual_info", "f_classif"):
                    func = {
                        "chi2": chi2,
                        "mutual_info": mutual_info_classif,
                        "f_classif": f_classif,
                    }[method]
                    # 过滤方法不允许负值
                    if sparse.issparse(X):
                        X_input = X
                    else:
                        X_input = np.asarray(X)
                        if (X_input < 0).any() and method in ("chi2",):
                            X_input = X_input - X_input.min(axis=0)
                    selector = SelectKBest(score_func=func, k=k)
                    X_new = selector.fit_transform(X_input, y_arr)

                elif method == "rfe":
                    # RFE 用LogReg + l2，适合中等维度
                    est = LogisticRegression(
                        penalty="l2", solver="saga", max_iter=2000,
                        multi_class="multinomial", random_state=self.random_state, n_jobs=-1,
                    )
                    # 高维稀疏只取前 n_features 做RFE避免过慢
                    if n_feat > 5000:
                        pre = SelectKBest(chi2, k=min(3000, n_feat))
                        X_in = pre.fit_transform(
                            (X - X.min(axis=0) if sparse.issparse(X) else np.clip(X, 0, None)),
                            y_arr
                        )
                    else:
                        X_in = self._to_dense(X)
                    selector = RFE(estimator=est, n_features_to_select=k, step=0.1)
                    X_new = selector.fit_transform(X_in, y_arr)

                elif method == "sfm_l1":
                    est = LogisticRegression(
                        penalty="l1", solver="saga", max_iter=2000,
                        multi_class="multinomial", C=1.0,
                        random_state=self.random_state, n_jobs=-1,
                    )
                    X_in = self._to_dense(X)
                    selector = SelectFromModel(est, max_features=k, threshold=-np.inf)
                    X_new = selector.fit_transform(X_in, y_arr)

                elif method == "sfm_tree":
                    est = ExtraTreesClassifier(
                        n_estimators=300, random_state=self.random_state, n_jobs=-1
                    )
                    X_in = self._to_dense(X)
                    selector = SelectFromModel(est, max_features=k, threshold=-np.inf)
                    X_new = selector.fit_transform(X_in, y_arr)

                else:
                    raise ValueError(f"未知方法: {method}")

                # 新特征名
                if overwrite:
                    new_name = fname
                else:
                    new_name = f"{fname}_{method}_k{k}"

                # 保留稀疏性
                if sparse.issparse(X) and not sparse.issparse(X_new):
                    X_new = sparse.csr_matrix(X_new)

                if overwrite:
                    self._feature_store[new_name] = (X_new, {
                        **meta,
                        "n_features": int(X_new.shape[1]),
                        f"selected_by_{method}": True,
                    })
                else:
                    self.register_feature(
                        new_name, X_new, module=meta["module"] + "_selected",
                        description=f"{method}特征选择, k={k} (源自 {fname})"
                    )
                self._feature_selectors[new_name] = selector

                logger.info(f"  {fname} -> {new_name}: {n_feat} -> {X_new.shape[1]} 维")
            except Exception as e:
                logger.warning(f"  {fname} 特征选择失败: {e}")

        return self

    # ==================== 特征压缩 ====================

    def compress_features(
        self,
        feature_names: Optional[List[str]] = None,
        method: Literal["pca", "svd"] = "svd",
        n_components: int = 256,
        concat: bool = True,
        y_for_supervised: Optional[Union[List[int], np.ndarray]] = None,
    ) -> Tuple[str, Any]:
        """
        对多组特征拼接后进行整体压缩（降维）

        Args:
            feature_names: 要组合压缩的特征组，None 表示全部
            method: "pca"（稠密） / "svd"（兼容稀疏）
            n_components: 目标维度
            concat: True=返回单个压缩特征；False=仅返回压缩矩阵并注册
            y_for_supervised: 预留接口（如 LDA 之类的有监督降维）

        Returns:
            (新注册的特征名, 压缩矩阵)
        """
        target_names = feature_names or list(self._feature_store.keys())
        if not target_names:
            raise RuntimeError("没有可用的特征组")

        # 水平拼接
        matrices = [self._feature_store[n][0] for n in target_names]
        X_combined = sparse.hstack(matrices, format="csr") if any(
            sparse.issparse(m) for m in matrices
        ) else np.hstack([self._to_dense(m) for m in matrices])

        total_dim = X_combined.shape[1]
        actual_components = min(n_components, total_dim, X_combined.shape[0])
        logger.info(
            f"开始特征压缩 | 方法: {method} | 维度: {total_dim} -> {actual_components} | "
            f"拼接特征: {target_names}"
        )

        if method == "svd":
            # 支持稀疏
            compressor = TruncatedSVD(n_components=actual_components, random_state=self.random_state)
            X_compressed = compressor.fit_transform(X_combined)
        elif method == "pca":
            X_dense = self._to_dense(X_combined)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_dense)
            compressor = PCA(n_components=actual_components, random_state=self.random_state)
            X_compressed = compressor.fit_transform(X_scaled)
            self._compressor = (compressor, scaler)
        else:
            raise ValueError(f"未知压缩方法: {method}")

        if method != "pca":
            self._compressor = compressor
        self._compressor_name = method

        comp_name = f"compressed_{method}_{actual_components}"
        self.register_feature(
            comp_name, X_compressed.astype(np.float32), module="compressed",
            description=f"{method}压缩，由 {len(target_names)} 组特征合并"
        )
        return comp_name, X_compressed

    # ==================== 组合 + 自动筛选最优子集 ====================

    def combine_features(
        self,
        feature_names: List[str],
        output_name: Optional[str] = None,
    ) -> Tuple[str, Any]:
        """
        按模块名组合多组特征（水平拼接）

        Args:
            feature_names: 要组合的特征名列表
            output_name: 输出特征名

        Returns:
            (组合特征名, 组合矩阵)
        """
        for fname in feature_names:
            if fname not in self._feature_store:
                raise KeyError(f"特征 {fname} 不在池中")

        matrices = [self._feature_store[n][0] for n in feature_names]
        any_sparse = any(sparse.issparse(m) for m in matrices)

        if any_sparse:
            combined = sparse.hstack([
                m if sparse.issparse(m) else sparse.csr_matrix(m)
                for m in matrices
            ], format="csr", dtype=np.float32)
        else:
            combined = np.hstack([self._to_dense(m) for m in matrices]).astype(np.float32)

        output_name = output_name or f"combo_{'__'.join(feature_names)}"
        output_name = output_name[:80]  # 截断防过长

        self.register_feature(
            output_name, combined, module="combined",
            description=f"手动组合: {' + '.join(feature_names)}"
        )
        return output_name, combined

    def auto_search_best_combination(
        self,
        y: Union[List[int], np.ndarray],
        candidates: Optional[List[str]] = None,
        max_combined: int = 5,
        cv: int = 3,
        estimator: Optional[Any] = None,
        scoring: str = "f1_macro",
        use_compressed_svd: bool = True,
        compressed_dim: int = 300,
    ) -> Dict[str, Any]:
        """
        自动搜索最优特征子集（暴力组合 + 交叉验证评分）

        Args:
            y: 标签数组
            candidates: 候选特征名列表，None=全部
            max_combined: 最多组合多少组特征
            cv: 交叉验证折数
            estimator: 评估用的分类器，默认 LightGBM（快）
            scoring: 评分函数名 sklearn 风格，默认 f1_macro
            use_compressed_svd: 组合后维度过高时先SVD压缩再评估
            compressed_dim: SVD压缩到多少维评估

        Returns:
            {
                "best_name": 最优组合名,
                "best_score": 最优分数,
                "best_features": 组合的特征名列表,
                "log": 完整搜索日志（DataFrame）,
                "X_best": 最优组合特征矩阵,
            }
        """
        y_arr = np.asarray(y)
        if candidates is None:
            candidates = list(self._feature_store.keys())

        # 过滤：保证候选存在，且样本数一致
        n_samples = self._feature_store[candidates[0]][1]["n_samples"]
        candidates = [c for c in candidates if c in self._feature_store
                      and self._feature_store[c][1]["n_samples"] == n_samples]

        if estimator is None:
            try:
                from lightgbm import LGBMClassifier
                estimator = LGBMClassifier(
                    n_estimators=200, learning_rate=0.05, num_leaves=63,
                    random_state=self.random_state, n_jobs=-1, verbose=-1,
                )
            except Exception:
                estimator = RandomForestClassifier(
                    n_estimators=200, random_state=self.random_state, n_jobs=-1
                )

        logger.info(
            f"========== 开始自动特征组合搜索 ==========\n"
            f"候选特征数: {len(candidates)} | 最大组合数: {max_combined} | "
            f"CV={cv} | 评分={scoring}"
        )

        # 生成所有非空子集（限制组合大小）
        subsets: List[List[str]] = []
        for r in range(1, min(max_combined, len(candidates)) + 1):
            for subset in combinations(candidates, r):
                subsets.append(list(subset))

        logger.info(f"共需评估 {len(subsets)} 个特征组合")

        self._combination_log.clear()
        best_score = -1.0
        best_subset: Optional[List[str]] = None
        best_name: Optional[str] = None
        best_X: Optional[Any] = None

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        for subset in tqdm(subsets, desc="特征组合搜索", ncols=80):
            try:
                name, X = self.combine_features(subset)
                dim = X.shape[1]

                # 过高维度先SVD压缩（保持相对性能同时大幅提速）
                if use_compressed_svd and dim > compressed_dim:
                    try:
                        n_comp = min(compressed_dim, dim, X.shape[0] - cv)
                        svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
                        X_eval = svd.fit_transform(X)
                    except Exception:
                        X_eval = X
                else:
                    X_eval = X

                # 为防止稀疏/稠密问题，必要时转稠密
                if sparse.issparse(X_eval):
                    try:
                        scores = cross_val_score(
                            estimator, X_eval, y_arr, cv=skf,
                            scoring=scoring, n_jobs=-1,
                        )
                    except Exception:
                        X_eval = self._to_dense(X_eval)
                        scores = cross_val_score(
                            estimator, X_eval, y_arr, cv=skf,
                            scoring=scoring, n_jobs=-1,
                        )
                else:
                    scores = cross_val_score(
                        estimator, X_eval, y_arr, cv=skf,
                        scoring=scoring, n_jobs=-1,
                    )

                mean_score = float(scores.mean())
                std_score = float(scores.std())
                entry = {
                    "features": " + ".join(subset),
                    "n_features": len(subset),
                    "dim": dim,
                    f"{scoring}_mean": mean_score,
                    f"{scoring}_std": std_score,
                }
                self._combination_log.append(entry)

                if mean_score > best_score:
                    best_score = mean_score
                    best_subset = subset
                    best_name = name
                    best_X = X
                    logger.info(
                        f"  新最优组合: {name} | dim={dim} | "
                        f"{scoring}={mean_score:.4f}±{std_score:.4f}"
                    )
            except Exception as e:
                logger.debug(f"组合 {subset} 评估失败: {e}")
                continue

        log_df = pd.DataFrame(self._combination_log)
        if not log_df.empty:
            log_df = log_df.sort_values(f"{scoring}_mean", ascending=False).reset_index(drop=True)

        logger.info(
            f"========== 搜索完成 ==========\n"
            f"最优组合: {best_name} | 分数: {best_score:.4f} | 特征: {best_subset}"
        )

        return {
            "best_name": best_name,
            "best_score": best_score,
            "best_features": best_subset,
            "log": log_df,
            "X_best": best_X,
        }

    # ==================== 获取/导出 ====================

    def get_feature(self, name: str) -> Any:
        """获取单个特征矩阵"""
        if name not in self._feature_store:
            raise KeyError(f"特征 {name} 不存在")
        return self._feature_store[name][0]

    def export_dataset(
        self,
        feature_names: List[str],
        y: Optional[Union[List[int], np.ndarray]] = None,
        return_df: bool = False,
        feature_prefix: bool = True,
    ) -> Union[Any, pd.DataFrame]:
        """
        导出为训练数据（矩阵或DataFrame）

        Args:
            feature_names: 要使用的特征名列表
            y: 可选，附加在DataFrame最后一列 label
            return_df: True 返回DataFrame（会将稀疏转为稠密！）；False 返回拼接矩阵
            feature_prefix: DataFrame列名加前缀区分模块

        Returns:
            拼接矩阵或DataFrame
        """
        name, X = self.combine_features(feature_names, output_name="export_temp")

        if not return_df:
            return X

        X_dense = self._to_dense(X)
        col_names = []
        offset = 0
        for fname in feature_names:
            meta = self._feature_store[fname][1]
            d = meta["n_features"]
            prefix = f"{fname}_" if feature_prefix else "f"
            col_names.extend([f"{prefix}{i}" for i in range(offset, offset + d)])
            offset += d

        df = pd.DataFrame(X_dense, columns=col_names)
        if y is not None:
            df["label"] = np.asarray(y)
            df["category"] = df["label"].map(
                lambda i: CATEGORY_NAMES[i] if 0 <= i < len(CATEGORY_NAMES) else "未知"
            )
        return df

    # ==================== 保存/加载 ====================

    def save(self, name: str = "feature_pool") -> str:
        """
        保存FeaturePool到磁盘（含注册特征的元数据、选择器、压缩模型）
        注意：稀疏/稠密特征矩阵默认也会保存（注意文件大小，必要时清理再保存）

        Args:
            name: 保存名称

        Returns:
            保存路径
        """
        save_path = self.save_dir / f"{name}.pkl"
        data = {
            "feature_names": list(self._feature_store.keys()),
            "features": {
                name: (matrix, meta)
                for name, (matrix, meta) in self._feature_store.items()
            },
            "feature_selectors": self._feature_selectors,
            "compressor": self._compressor,
            "compressor_name": self._compressor_name,
            "low_variance_mask": self._low_variance_mask,
            "correlation_mask": self._correlation_mask,
            "combination_log": self._combination_log,
            "random_state": self.random_state,
        }
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        # 保存子组件
        try:
            self.vectorizer.save(str(self.save_dir / f"{name}_vectorizer"))
        except Exception as e:
            logger.warning(f"保存子组件 vectorizer 失败: {e}")

        try:
            if self.bert_embedder is not None:
                self.bert_embedder.save(f"{name}_bert")
        except Exception as e:
            logger.warning(f"保存子组件 bert_embedder 失败: {e}")

        # 估算大小
        size_mb = save_path.stat().st_size / (1024 * 1024)
        logger.info(f"FeaturePool已保存: {save_path} (约 {size_mb:.1f} MB)")
        return str(save_path)

    @classmethod
    def load(cls, path: str) -> "FeaturePool":
        """从磁盘加载 FeaturePool"""
        load_path = Path(path)
        with open(load_path, "rb") as f:
            data = pickle.load(f)

        pool = cls(random_state=data.get("random_state", 42))
        pool._feature_store = data.get("features", {})
        pool._feature_selectors = data.get("feature_selectors", {})
        pool._compressor = data.get("compressor")
        pool._compressor_name = data.get("compressor_name")
        pool._low_variance_mask = data.get("low_variance_mask")
        pool._correlation_mask = data.get("correlation_mask")
        pool._combination_log = data.get("combination_log", [])

        # 尝试加载子组件（若存在）
        vec_path = load_path.parent / f"{load_path.stem}_vectorizer.pkl"
        if vec_path.exists():
            try:
                pool.vectorizer = TextVectorizer.load(str(vec_path), tokenizer=pool.tokenizer)
            except Exception as e:
                logger.warning(f"加载子组件 vectorizer 失败: {e}")

        bert_path = load_path.parent.parent / "bert_configs" / f"{load_path.stem}_bert.pkl"
        if bert_path.exists():
            try:
                pool.bert_embedder = BertEmbedder.load(str(bert_path))
            except Exception as e:
                logger.warning(f"加载子组件 bert_embedder 失败: {e}")

        logger.info(
            f"FeaturePool已加载: {load_path} | "
            f"特征组数: {len(pool._feature_store)}"
        )
        return pool

    def __repr__(self) -> str:
        total_dim = sum(m[1]["n_features"] for m in self._feature_store.values())
        return (
            f"FeaturePool(registered={len(self._feature_store)}, "
            f"total_dim={total_dim}, compressor={self._compressor_name})"
        )
