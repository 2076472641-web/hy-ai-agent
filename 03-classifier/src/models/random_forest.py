# -*- coding: utf-8 -*-
"""
TF-IDF + 随机森林 分类基线模型
采用经典的两阶段流水线：
1. 文本特征提取：使用 Jieba 分词 + TF-IDF 向量化
2. 分类器：随机森林（Random Forest）集成学习

特点：
- 训练速度快，可解释性较好
- 内置网格搜索和随机搜索超参调优
- 类别不平衡处理：balanced class_weight
- 特征重要性可输出分析
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import pickle
import copy

import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier as SKRandomForest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_macro
from scipy.stats import randint, uniform
from loguru import logger

from .base import BaseClassifier


class RandomForestClassifier(BaseClassifier):
    """
    TF-IDF + 随机森林 分类器
    
    该模型作为基线模型（Baseline），提供快速可靠的分类性能。
    内部集成了 TF-IDF 特征提取器和随机森林分类器，支持超参调优。

    Args:
        model_name: 模型名称标识
        n_estimators: 决策树数量，默认 200
        max_depth: 树的最大深度，None 表示不限制
        min_samples_split: 内部节点再划分所需最小样本数
        min_samples_leaf: 叶子节点最少样本数
        class_weight: 类别权重，'balanced' 自动平衡
        random_state: 随机种子
        tfidf_max_features: TF-IDF 最大特征数
        tfidf_ngram_range: n-gram 范围
        tfidf_min_df: 最小文档频率
        tfidf_max_df: 最大文档频率
        use_jieba: 是否使用 Jieba 分词
    """

    def __init__(
        self,
        model_name: str = "random_forest",
        n_estimators: int = 200,
        max_depth: Optional[int] = 30,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        tfidf_max_features: int = 5000,
        tfidf_ngram_range: Tuple[int, int] = (1, 2),
        tfidf_min_df: int = 2,
        tfidf_max_df: float = 0.95,
        use_jieba: bool = True,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)

        # 随机森林分类器超参数
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state

        # TF-IDF 向量化器超参数
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_df = tfidf_max_df
        self.use_jieba = use_jieba

        # 初始化组件（延迟初始化 sklearn 对象）
        self._tfidf: Optional[TfidfVectorizer] = None
        self._classifier: Optional[SKRandomForest] = None

        # 保存超参数到 params 字典
        self.params.update({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "class_weight": class_weight,
            "random_state": random_state,
            "tfidf_max_features": tfidf_max_features,
            "tfidf_ngram_range": list(tfidf_ngram_range),
            "tfidf_min_df": tfidf_min_df,
            "tfidf_max_df": tfidf_max_df,
            "use_jieba": use_jieba,
        })

    def _jieba_tokenize(self, text: str) -> List[str]:
        """
        使用 Jieba 进行中文分词

        Args:
            text: 输入文本

        Returns:
            List[str]: 分词结果列表
        """
        if not isinstance(text, str) or not text.strip():
            return []
        return list(jieba.cut(text.strip()))

    def _build_tfidf(self) -> TfidfVectorizer:
        """构建 TF-IDF 向量化器"""
        tokenizer = self._jieba_tokenize if self.use_jieba else None
        return TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=self.tfidf_ngram_range,
            min_df=self.tfidf_min_df,
            max_df=self.tfidf_max_df,
            tokenizer=tokenizer if self.use_jieba else None,
            token_pattern=None if self.use_jieba else r"(?u)\b\w+\b",
        )

    def _build_classifier(self) -> SKRandomForest:
        """构建随机森林分类器"""
        return SKRandomForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0,
        )

    def fit(
        self,
        X: Union[List[str], np.ndarray, pd.Series],
        y: Union[List[int], np.ndarray, pd.Series],
        X_val: Optional[Union[List[str], np.ndarray, pd.Series]] = None,
        y_val: Optional[Union[List[int], np.ndarray, pd.Series]] = None,
        tune_hyperparams: bool = False,
        search_method: str = "grid",
        param_grid: Optional[Dict[str, Any]] = None,
        n_iter: int = 20,
        cv: int = 3,
        **kwargs,
    ) -> "RandomForestClassifier":
        """
        训练 TF-IDF + 随机森林分类模型

        Args:
            X: 训练文本数据
            y: 训练标签 0-11
            X_val: 验证集文本（可选，用于调优时的早停参考）
            y_val: 验证集标签（可选）
            tune_hyperparams: 是否启用超参搜索调优
            search_method: 搜索方法 'grid'（网格搜索）或 'random'（随机搜索）
            param_grid: 超参搜索空间，None 则使用默认搜索空间
            n_iter: 随机搜索时的迭代次数
            cv: 交叉验证折数
            **kwargs: 额外参数

        Returns:
            self: 训练完成的模型
        """
        X_array, y_array = self._validate_inputs(X, y)
        X_list = X_array.tolist()

        logger.info(
            f"[{self.model_name}] 开始训练 - 样本数: {len(X_list)}, "
            f"TF-IDF特征: {self.tfidf_max_features}, 树数量: {self.n_estimators}"
        )

        # Step 1: 构建并拟合 TF-IDF 向量化器
        self._tfidf = self._build_tfidf()
        X_tfidf = self._tfidf.fit_transform(X_list)
        logger.info(
            f"[{self.model_name}] TF-IDF 向量化完成 - 形状: {X_tfidf.shape}, "
            f"词汇量: {len(self._tfidf.vocabulary_)}"
        )

        # Step 2: 超参搜索或直接训练
        if tune_hyperparams:
            if param_grid is None:
                param_grid = self._default_search_space(search_method)
            self._classifier = self._run_hyperparam_search(
                X_tfidf, y_array, param_grid, search_method, n_iter, cv
            )
            # 从搜索结果中提取最佳参数
            best_params = self._classifier.best_params_
            logger.info(f"[{self.model_name}] 超参搜索完成 - 最佳参数: {best_params}")
            self.set_params(**best_params)
            self._classifier = self._classifier.best_estimator_
        else:
            self._classifier = self._build_classifier()
            self._classifier.fit(X_tfidf, y_array)
            logger.info(f"[{self.model_name}] 随机森林训练完成")

        # Step 3: 验证集评估
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            val_f1 = f1_macro(y_val, val_pred)
            logger.info(f"[{self.model_name}] 验证集 Macro-F1: {val_f1:.4f}")

        self.is_trained = True
        return self

    def _default_search_space(self, method: str) -> Dict[str, Any]:
        """
        生成默认的超参搜索空间

        Args:
            method: 'grid' 或 'random'

        Returns:
            Dict: 搜索空间字典
        """
        if method == "grid":
            return {
                "n_estimators": [100, 200, 300],
                "max_depth": [20, 30, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2, 4],
            }
        else:  # random
            return {
                "n_estimators": randint(100, 500),
                "max_depth": randint(10, 60),
                "min_samples_split": randint(2, 10),
                "min_samples_leaf": randint(1, 8),
                "max_features": uniform(0.3, 0.7),
            }

    def _run_hyperparam_search(
        self,
        X_tfidf: Any,
        y: np.ndarray,
        param_grid: Dict[str, Any],
        method: str,
        n_iter: int,
        cv: int,
    ) -> Union[GridSearchCV, RandomizedSearchCV]:
        """
        执行超参搜索

        Args:
            X_tfidf: TF-IDF 特征矩阵
            y: 标签
            param_grid: 搜索空间
            method: 'grid' 或 'random'
            n_iter: 随机搜索迭代次数
            cv: 交叉验证折数

        Returns:
            训练完成的搜索对象
        """
        base_clf = self._build_classifier()
        scoring = "f1_macro"

        logger.info(
            f"[{self.model_name}] 启动超参搜索 - 方法: {method}, "
            f"CV: {cv}, 指标: {scoring}"
        )

        if method == "grid":
            search = GridSearchCV(
                base_clf,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
            )
        elif method == "random":
            search = RandomizedSearchCV(
                base_clf,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"未知搜索方法: {method}，可选 'grid' 或 'random'")

        search.fit(X_tfidf, y)
        logger.info(
            f"[{self.model_name}] 搜索最佳得分: {search.best_score_:.4f}"
        )
        return search

    def predict(
        self, X: Union[List[str], np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        预测类别ID

        Args:
            X: 待预测文本列表

        Returns:
            np.ndarray: 预测类别ID数组
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用 fit() 方法")
        X_tfidf = self._tfidf.transform(np.asarray(X).tolist())
        return self._classifier.predict(X_tfidf)

    def predict_proba(
        self, X: Union[List[str], np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        预测类别概率分布

        Args:
            X: 待预测文本列表

        Returns:
            np.ndarray: 概率矩阵 (n_samples, 12)
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用 fit() 方法")
        X_tfidf = self._tfidf.transform(np.asarray(X).tolist())
        proba = self._classifier.predict_proba(X_tfidf)

        # 确保概率矩阵包含所有12个类别（即使某些类别训练时未出现）
        seen_classes = self._classifier.classes_
        full_proba = np.zeros((proba.shape[0], self.num_classes))
        for i, cls in enumerate(seen_classes):
            full_proba[:, cls] = proba[:, i]
        return full_proba

    def get_feature_importance(self, top_k: int = 50) -> pd.DataFrame:
        """
        获取随机森林的特征重要性

        Args:
            top_k: 返回前 K 个重要特征

        Returns:
            pd.DataFrame: 特征重要性表格，包含 feature, importance 列
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用 fit() 方法")

        feature_names = self._tfidf.get_feature_names_out()
        importances = self._classifier.feature_importances_

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)

        logger.info(f"[{self.model_name}] 输出 Top-{top_k} 重要特征")
        return df

    def save(self, save_dir: Union[str, Path]) -> str:
        """
        保存模型（TF-IDF + 随机森林）到磁盘

        Args:
            save_dir: 保存目录

        Returns:
            str: 保存路径
        """
        save_path = Path(save_dir) / self.model_name
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存元数据
        self._save_metadata(save_path)

        # 保存 TF-IDF 向量化器
        tfidf_path = save_path / "tfidf.pkl"
        with open(tfidf_path, "wb") as f:
            pickle.dump(self._tfidf, f)

        # 保存随机森林分类器
        clf_path = save_path / "classifier.pkl"
        with open(clf_path, "wb") as f:
            pickle.dump(self._classifier, f)

        logger.info(f"[{self.model_name}] 模型已保存到: {save_path}")
        return str(save_path)

    def load(self, load_dir: Union[str, Path]) -> "RandomForestClassifier":
        """
        从磁盘加载模型

        Args:
            load_dir: 加载目录

        Returns:
            self: 加载完成的模型
        """
        load_path = Path(load_dir)

        # 加载元数据
        self._load_metadata(load_path)

        # 恢复超参数属性
        for key, val in self.params.items():
            if hasattr(self, key):
                setattr(self, key, val)

        # 加载 TF-IDF
        tfidf_path = load_path / "tfidf.pkl"
        with open(tfidf_path, "rb") as f:
            self._tfidf = pickle.load(f)

        # 加载分类器
        clf_path = load_path / "classifier.pkl"
        with open(clf_path, "rb") as f:
            self._classifier = pickle.load(f)

        logger.info(f"[{self.model_name}] 模型已从 {load_path} 加载")
        return self
