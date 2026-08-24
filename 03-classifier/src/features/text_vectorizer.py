# -*- coding: utf-8 -*-
"""
文本向量化模块
功能：
1. TF-IDF特征提取（支持字符级/词级 ngram）
2. Word2Vec词嵌入（可选，基于gensim）
3. Ngram特征构建
4. 特征重要性评估（基于卡方、互信息、树模型）
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import normalize
from loguru import logger
from tqdm import tqdm

from src import PROCESSED_DATA_DIR, MODEL_DIR
from .tokenizer import ChineseTokenizer


class TextVectorizer:
    """文本向量化器，集成TF-IDF、Ngram、Word2Vec和特征重要性评估"""

    def __init__(
        self,
        tokenizer: Optional[ChineseTokenizer] = None,
        tfidf_max_features: int = 5000,
        tfidf_ngram_range: Tuple[int, int] = (1, 2),
        tfidf_min_df: int = 2,
        tfidf_max_df: float = 0.95,
        tfidf_analyzer: str = "word",
        use_word2vec: bool = False,
        word2vec_dim: int = 100,
        word2vec_window: int = 5,
        word2vec_min_count: int = 2,
        char_ngram_range: Tuple[int, int] = (2, 4),
        char_max_features: int = 3000,
        save_dir: Optional[str] = None,
    ):
        """
        初始化文本向量化器

        Args:
            tokenizer: 中文分词器实例，不传则自动创建默认分词器
            tfidf_max_features: TF-IDF词级最大特征数
            tfidf_ngram_range: TF-IDF词级ngram范围
            tfidf_min_df: 最小文档频次
            tfidf_max_df: 最大文档频率比例
            tfidf_analyzer: TF-IDF分析器，"word"词级 / "char"字符级 / "char_wb"
            use_word2vec: 是否启用Word2Vec嵌入
            word2vec_dim: Word2Vec向量维度
            word2vec_window: Word2Vec上下文窗口大小
            word2vec_min_count: Word2Vec最小词频
            char_ngram_range: 字符级ngram范围
            char_max_features: 字符级最大特征数
            save_dir: 模型保存目录，默认 data/models/vectorizers
        """
        self.tokenizer = tokenizer or ChineseTokenizer()
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_df = tfidf_max_df
        self.tfidf_analyzer = tfidf_analyzer
        self.use_word2vec = use_word2vec
        self.word2vec_dim = word2vec_dim
        self.word2vec_window = word2vec_window
        self.word2vec_min_count = word2vec_min_count
        self.char_ngram_range = char_ngram_range
        self.char_max_features = char_max_features

        # 保存目录
        self.save_dir = Path(save_dir) if save_dir else MODEL_DIR / "vectorizers"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 子向量化器
        self.tfidf_word: Optional[TfidfVectorizer] = None
        self.tfidf_char: Optional[TfidfVectorizer] = None
        self.count_ngram: Optional[CountVectorizer] = None
        self.word2vec_model = None

        # 特征重要性缓存
        self._feature_importance_cache: Dict[str, pd.DataFrame] = {}

        logger.info(
            f"TextVectorizer初始化完成 | TF-IDF(词): {tfidf_max_features}维 | "
            f"TF-IDF(字符): {char_max_features}维 | Word2Vec: {use_word2vec}"
        )

    # ==================== TF-IDF 向量化 ====================

    def _tokenize_for_sklearn(self, text: str) -> List[str]:
        """供sklearn调用的分词函数（不带词性筛选，快速分词）"""
        return self.tokenizer.tokenize(text)

    def fit_tfidf_word(
        self,
        texts: List[str],
    ) -> "TextVectorizer":
        """
        拟合词级TF-IDF向量化器

        Args:
            texts: 原始文本列表

        Returns:
            self
        """
        logger.info(f"开始拟合词级TF-IDF，样本数: {len(texts)}")
        try:
            self.tfidf_word = TfidfVectorizer(
                tokenizer=self._tokenize_for_sklearn,
                ngram_range=self.tfidf_ngram_range,
                max_features=self.tfidf_max_features,
                min_df=self.tfidf_min_df,
                max_df=self.tfidf_max_df,
                sublinear_tf=True,
                norm="l2",
                dtype=np.float32,
            )
            self.tfidf_word.fit(texts)
            logger.info(
                f"词级TF-IDF拟合完成 | 词汇表大小: {len(self.tfidf_word.vocabulary_)}"
            )
        except Exception as e:
            logger.error(f"词级TF-IDF拟合失败: {e}")
            raise
        return self

    def transform_tfidf_word(
        self,
        texts: List[str],
    ) -> sparse.csr_matrix:
        """
        使用词级TF-IDF转换文本

        Args:
            texts: 原始文本列表

        Returns:
            TF-IDF稀疏矩阵
        """
        if self.tfidf_word is None:
            raise RuntimeError("词级TF-IDF未拟合，请先调用 fit_tfidf_word()")
        return self.tfidf_word.transform(texts)

    def fit_transform_tfidf_word(
        self,
        texts: List[str],
    ) -> sparse.csr_matrix:
        """拟合并转换词级TF-IDF"""
        self.fit_tfidf_word(texts)
        return self.transform_tfidf_word(texts)

    def fit_tfidf_char(
        self,
        texts: List[str],
    ) -> "TextVectorizer":
        """
        拟合字符级TF-IDF向量化器（捕捉汉字字形子串特征）

        Args:
            texts: 原始文本列表

        Returns:
            self
        """
        logger.info(f"开始拟合字符级TF-IDF，样本数: {len(texts)}")
        try:
            self.tfidf_char = TfidfVectorizer(
                analyzer="char",
                ngram_range=self.char_ngram_range,
                max_features=self.char_max_features,
                min_df=self.tfidf_min_df,
                max_df=self.tfidf_max_df,
                sublinear_tf=True,
                norm="l2",
                dtype=np.float32,
            )
            self.tfidf_char.fit(texts)
            logger.info(
                f"字符级TF-IDF拟合完成 | 字符ngram数: {len(self.tfidf_char.vocabulary_)}"
            )
        except Exception as e:
            logger.error(f"字符级TF-IDF拟合失败: {e}")
            raise
        return self

    def transform_tfidf_char(
        self,
        texts: List[str],
    ) -> sparse.csr_matrix:
        """使用字符级TF-IDF转换文本"""
        if self.tfidf_char is None:
            raise RuntimeError("字符级TF-IDF未拟合，请先调用 fit_tfidf_char()")
        return self.tfidf_char.transform(texts)

    def fit_transform_tfidf_char(
        self,
        texts: List[str],
    ) -> sparse.csr_matrix:
        """拟合并转换字符级TF-IDF"""
        self.fit_tfidf_char(texts)
        return self.transform_tfidf_char(texts)

    # ==================== Ngram 频率特征 ====================

    def fit_ngram_count(
        self,
        texts: List[str],
        ngram_range: Tuple[int, int] = (1, 3),
        max_features: int = 2000,
    ) -> "TextVectorizer":
        """
        拟合Ngram词频特征（用于捕捉高频短语）

        Args:
            texts: 原始文本列表
            ngram_range: ngram范围
            max_features: 最大特征数

        Returns:
            self
        """
        logger.info(f"开始拟合Ngram词频特征，样本数: {len(texts)}")
        try:
            self.count_ngram = CountVectorizer(
                tokenizer=self._tokenize_for_sklearn,
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=self.tfidf_min_df,
                max_df=self.tfidf_max_df,
                dtype=np.float32,
            )
            self.count_ngram.fit(texts)
            logger.info(
                f"Ngram词频拟合完成 | 短语数: {len(self.count_ngram.vocabulary_)}"
            )
        except Exception as e:
            logger.error(f"Ngram词频拟合失败: {e}")
            raise
        return self

    def transform_ngram_count(
        self,
        texts: List[str],
    ) -> sparse.csr_matrix:
        """转换Ngram词频特征"""
        if self.count_ngram is None:
            raise RuntimeError("Ngram词频未拟合，请先调用 fit_ngram_count()")
        return normalize(self.count_ngram.transform(texts), norm="l2", axis=1)

    # ==================== Word2Vec 嵌入 ====================

    def fit_word2vec(
        self,
        texts: List[str],
        epochs: int = 20,
        sg: int = 1,
    ) -> "TextVectorizer":
        """
        拟合Word2Vec词向量模型

        Args:
            texts: 原始文本列表
            epochs: 训练轮数
            sg: 1=Skip-gram, 0=CBOW

        Returns:
            self
        """
        if not self.use_word2vec:
            logger.warning("use_word2vec=False，跳过Word2Vec训练")
            return self

        try:
            from gensim.models import Word2Vec
        except ImportError:
            logger.error("gensim未安装，Word2Vec功能不可用。请安装: pip install gensim")
            self.use_word2vec = False
            return self

        logger.info(f"开始训练Word2Vec，样本数: {len(texts)}，维度: {self.word2vec_dim}")

        # 对语料分词
        tokenized_corpus = []
        for text in tqdm(texts, desc="Word2Vec语料分词", ncols=80):
            tokens = self.tokenizer.tokenize(text)
            if tokens:
                tokenized_corpus.append(tokens)

        logger.info(f"有效语料数: {len(tokenized_corpus)}")

        try:
            self.word2vec_model = Word2Vec(
                sentences=tokenized_corpus,
                vector_size=self.word2vec_dim,
                window=self.word2vec_window,
                min_count=self.word2vec_min_count,
                workers=os.cpu_count() or 4,
                sg=sg,
                epochs=epochs,
                seed=42,
            )
            logger.info(
                f"Word2Vec训练完成 | 词表大小: {len(self.word2vec_model.wv)}"
            )
        except Exception as e:
            logger.error(f"Word2Vec训练失败: {e}")
            raise
        return self

    def transform_word2vec(
        self,
        texts: List[str],
        pooling: str = "mean",
    ) -> np.ndarray:
        """
        使用Word2Vec转换文本为向量（词向量池化）

        Args:
            texts: 原始文本列表
            pooling: 池化方式，"mean"平均池化 / "max"最大池化 / "mean_max"拼接

        Returns:
            文本嵌入矩阵 (n_samples, dim) 或 (n_samples, 2*dim)
        """
        if self.word2vec_model is None:
            raise RuntimeError("Word2Vec未拟合，请先调用 fit_word2vec()")

        wv = self.word2vec_model.wv
        dim = self.word2vec_dim
        n = len(texts)

        if pooling == "mean_max":
            result = np.zeros((n, 2 * dim), dtype=np.float32)
        else:
            result = np.zeros((n, dim), dtype=np.float32)

        for i, text in enumerate(tqdm(texts, desc="Word2Vec转换", ncols=80, leave=False)):
            tokens = self.tokenizer.tokenize(text)
            vectors = [wv[tok] for tok in tokens if tok in wv]
            if not vectors:
                continue
            vec_mat = np.vstack(vectors)

            if pooling == "mean":
                result[i] = vec_mat.mean(axis=0)
            elif pooling == "max":
                result[i] = vec_mat.max(axis=0)
            elif pooling == "mean_max":
                result[i, :dim] = vec_mat.mean(axis=0)
                result[i, dim:] = vec_mat.max(axis=0)
            else:
                raise ValueError(f"未知池化方式: {pooling}")

        return result

    # ==================== 组合特征 ====================

    def fit_all(
        self,
        texts: List[str],
        enable_word_tfidf: bool = True,
        enable_char_tfidf: bool = True,
        enable_ngram_count: bool = False,
        enable_word2vec: bool = False,
    ) -> "TextVectorizer":
        """
        拟合所有指定的文本特征

        Args:
            texts: 文本列表
            enable_word_tfidf: 是否启用词级TF-IDF
            enable_char_tfidf: 是否启用字符级TF-IDF
            enable_ngram_count: 是否启用Ngram词频
            enable_word2vec: 是否启用Word2Vec

        Returns:
            self
        """
        if enable_word_tfidf:
            self.fit_tfidf_word(texts)
        if enable_char_tfidf:
            self.fit_tfidf_char(texts)
        if enable_ngram_count:
            self.fit_ngram_count(texts)
        if enable_word2vec and self.use_word2vec:
            self.fit_word2vec(texts)
        return self

    def transform_all(
        self,
        texts: List[str],
        enable_word_tfidf: bool = True,
        enable_char_tfidf: bool = True,
        enable_ngram_count: bool = False,
        enable_word2vec: bool = False,
        word2vec_pooling: str = "mean",
        return_parts: bool = False,
    ) -> Union[sparse.csr_matrix, Tuple[sparse.csr_matrix, Dict[str, sparse.csr_matrix]]]:
        """
        转换文本为所有已拟合的特征，并水平拼接

        Args:
            texts: 文本列表
            enable_word_tfidf: 是否使用词级TF-IDF
            enable_char_tfidf: 是否使用字符级TF-IDF
            enable_ngram_count: 是否使用Ngram词频
            enable_word2vec: 是否使用Word2Vec
            word2vec_pooling: Word2Vec池化方式
            return_parts: 是否同时返回各部分特征字典

        Returns:
            拼接后的特征矩阵；return_parts=True 时返回 (拼接矩阵, 各部分字典)
        """
        parts: Dict[str, sparse.csr_matrix] = {}

        if enable_word_tfidf and self.tfidf_word is not None:
            parts["tfidf_word"] = self.transform_tfidf_word(texts)
        if enable_char_tfidf and self.tfidf_char is not None:
            parts["tfidf_char"] = self.transform_tfidf_char(texts)
        if enable_ngram_count and self.count_ngram is not None:
            parts["ngram_count"] = self.transform_ngram_count(texts)
        if enable_word2vec and self.word2vec_model is not None:
            dense_w2v = self.transform_word2vec(texts, pooling=word2vec_pooling)
            parts["word2vec"] = sparse.csr_matrix(dense_w2v)

        if not parts:
            raise RuntimeError("没有可用的特征，请先拟合或启用至少一种向量化方式")

        combined = sparse.hstack(list(parts.values()), format="csr", dtype=np.float32)

        if return_parts:
            return combined, parts
        return combined

    def fit_transform_all(self, texts: List[str], **kwargs) -> Union[sparse.csr_matrix, Tuple]:
        """拟合并转换所有特征"""
        self.fit_all(texts, **{k: v for k, v in kwargs.items() if k.startswith("enable_")})
        return self.transform_all(texts, **kwargs)

    # ==================== 特征重要性评估 ====================

    def evaluate_feature_importance(
        self,
        texts: List[str],
        labels: Union[List[int], np.ndarray],
        feature_type: str = "tfidf_word",
        top_k: int = 50,
        method: str = "chi2",
    ) -> pd.DataFrame:
        """
        评估特征重要性，返回DataFrame含特征名、分数、对应类目

        Args:
            texts: 文本列表
            labels: 类目标签数组
            feature_type: 评估的特征类型 "tfidf_word" / "tfidf_char" / "ngram_count"
            top_k: 每类返回Top-K
            method: 评估方法
                - "chi2": 卡方检验
                - "mutual_info": 互信息
                - "random_forest": 随机森林特征重要性
                - "gradient_boosting": GBDT特征重要性

        Returns:
            DataFrame列: [feature, category, category_id, score, rank]
        """
        labels = np.asarray(labels)
        X = None
        vocab = None
        feature_names = None

        if feature_type == "tfidf_word":
            if self.tfidf_word is None:
                self.fit_tfidf_word(texts)
            X = self.transform_tfidf_word(texts)
            vocab = self.tfidf_word.vocabulary_
            feature_names = self.tfidf_word.get_feature_names_out()
        elif feature_type == "tfidf_char":
            if self.tfidf_char is None:
                self.fit_tfidf_char(texts)
            X = self.transform_tfidf_char(texts)
            vocab = self.tfidf_char.vocabulary_
            feature_names = self.tfidf_char.get_feature_names_out()
        elif feature_type == "ngram_count":
            if self.count_ngram is None:
                self.fit_ngram_count(texts)
            X = self.transform_ngram_count(texts)
            vocab = self.count_ngram.vocabulary_
            feature_names = self.count_ngram.get_feature_names_out()
        else:
            raise ValueError(f"未知特征类型: {feature_type}")

        cache_key = f"{feature_type}_{method}"
        if cache_key in self._feature_importance_cache:
            return self._feature_importance_cache[cache_key]

        logger.info(f"开始评估 {feature_type} 特征重要性 | 方法: {method}")

        rows = []
        num_classes = len(np.unique(labels))

        if method == "chi2":
            # 对每个类别做 One-vs-Rest 卡方
            for c in range(num_classes):
                y_bin = (labels == c).astype(int)
                chi2_scores, _ = chi2(X, y_bin)
                top_idx = np.argsort(chi2_scores)[::-1][:top_k]
                for rank, idx in enumerate(top_idx, 1):
                    rows.append({
                        "feature": feature_names[idx],
                        "category_id": c,
                        "score": float(chi2_scores[idx]),
                        "rank": rank,
                    })

        elif method == "mutual_info":
            for c in range(num_classes):
                y_bin = (labels == c).astype(int)
                mi_scores = mutual_info_classif(
                    X, y_bin, discrete_features=True, random_state=42
                )
                top_idx = np.argsort(mi_scores)[::-1][:top_k]
                for rank, idx in enumerate(top_idx, 1):
                    rows.append({
                        "feature": feature_names[idx],
                        "category_id": c,
                        "score": float(mi_scores[idx]),
                        "rank": rank,
                    })

        elif method in ("random_forest", "gradient_boosting"):
            if method == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=200, max_depth=30, random_state=42, n_jobs=-1
                )
            else:
                model = GradientBoostingClassifier(
                    n_estimators=200, max_depth=5, random_state=42
                )
            model.fit(X, labels)
            global_importance = model.feature_importances_

            # 整体Top-K
            top_idx_global = np.argsort(global_importance)[::-1][:top_k]
            for rank, idx in enumerate(top_idx_global, 1):
                rows.append({
                    "feature": feature_names[idx],
                    "category_id": -1,
                    "score": float(global_importance[idx]),
                    "rank": rank,
                })

            # 各类特征重要性（基于各类样本的平均特征值 * 全局重要性）
            for c in range(num_classes):
                mask = labels == c
                if mask.sum() == 0:
                    continue
                class_mean = X[mask].mean(axis=0).A1
                class_importance = class_mean * global_importance
                top_idx = np.argsort(class_importance)[::-1][:top_k]
                for rank, idx in enumerate(top_idx, 1):
                    rows.append({
                        "feature": feature_names[idx],
                        "category_id": c,
                        "score": float(class_importance[idx]),
                        "rank": rank,
                    })
        else:
            raise ValueError(f"未知评估方法: {method}")

        df = pd.DataFrame(rows)

        # 映射类目名称
        from .tokenizer import CATEGORY_KEYWORDS
        id_to_cat = {i: name for i, name in enumerate(CATEGORY_KEYWORDS.keys())}
        df["category"] = df["category_id"].map(
            lambda x: id_to_cat.get(x, "整体")
        )
        df = df[["feature", "category", "category_id", "score", "rank"]]
        df = df.sort_values(["category_id", "rank"]).reset_index(drop=True)

        self._feature_importance_cache[cache_key] = df

        logger.info(
            f"特征重要性评估完成 | 共 {len(df)} 条记录 | "
            f"已缓存结果: {list(self._feature_importance_cache.keys())}"
        )
        return df

    def select_k_best(
        self,
        X: sparse.csr_matrix,
        y: Union[List[int], np.ndarray],
        k: int = 1000,
        score_func: str = "chi2",
    ) -> Tuple[sparse.csr_matrix, SelectKBest]:
        """
        基于统计检验进行特征选择

        Args:
            X: 特征矩阵
            y: 标签
            k: 保留特征数
            score_func: 评分函数 "chi2" / "mutual_info"

        Returns:
            (筛选后的特征矩阵, SelectKBest对象)
        """
        func_map = {
            "chi2": chi2,
            "mutual_info": mutual_info_classif,
        }
        if score_func not in func_map:
            raise ValueError(f"未知评分函数: {score_func}")

        selector = SelectKBest(score_func=func_map[score_func], k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        logger.info(
            f"特征选择完成: {X.shape[1]} -> {X_selected.shape[1]} | "
            f"方法: {score_func}"
        )
        return X_selected, selector

    # ==================== 保存/加载 ====================

    def save(self, name: str = "text_vectorizer") -> str:
        """
        保存向量化器到磁盘

        Args:
            name: 保存名称

        Returns:
            保存路径
        """
        save_path = self.save_dir / f"{name}.pkl"
        data = {
            "tfidf_word": self.tfidf_word,
            "tfidf_char": self.tfidf_char,
            "count_ngram": self.count_ngram,
            "config": {
                "tfidf_max_features": self.tfidf_max_features,
                "tfidf_ngram_range": self.tfidf_ngram_range,
                "tfidf_min_df": self.tfidf_min_df,
                "tfidf_max_df": self.tfidf_max_df,
                "use_word2vec": self.use_word2vec,
                "word2vec_dim": self.word2vec_dim,
            },
            "feature_importance": self._feature_importance_cache,
        }
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        if self.word2vec_model is not None:
            w2v_path = self.save_dir / f"{name}_word2vec.model"
            try:
                self.word2vec_model.save(str(w2v_path))
                logger.info(f"Word2Vec模型已保存: {w2v_path}")
            except Exception as e:
                logger.warning(f"Word2Vec保存失败: {e}")

        logger.info(f"向量化器已保存: {save_path}")
        return str(save_path)

    @classmethod
    def load(cls, path: str, tokenizer: Optional[ChineseTokenizer] = None) -> "TextVectorizer":
        """
        从磁盘加载向量化器

        Args:
            path: 保存路径 (.pkl)
            tokenizer: 可选，复用现有分词器

        Returns:
            TextVectorizer实例
        """
        load_path = Path(path)
        with open(load_path, "rb") as f:
            data = pickle.load(f)

        cfg = data.get("config", {})
        vec = cls(
            tokenizer=tokenizer,
            tfidf_max_features=cfg.get("tfidf_max_features", 5000),
            tfidf_ngram_range=cfg.get("tfidf_ngram_range", (1, 2)),
            tfidf_min_df=cfg.get("tfidf_min_df", 2),
            tfidf_max_df=cfg.get("tfidf_max_df", 0.95),
            use_word2vec=cfg.get("use_word2vec", False),
            word2vec_dim=cfg.get("word2vec_dim", 100),
        )
        vec.tfidf_word = data.get("tfidf_word")
        vec.tfidf_char = data.get("tfidf_char")
        vec.count_ngram = data.get("count_ngram")
        vec._feature_importance_cache = data.get("feature_importance", {})

        # 尝试加载Word2Vec
        w2v_path = load_path.parent / f"{load_path.stem}_word2vec.model"
        if w2v_path.exists():
            try:
                from gensim.models import Word2Vec
                vec.word2vec_model = Word2Vec.load(str(w2v_path))
                logger.info(f"Word2Vec模型已加载: {w2v_path}")
            except Exception as e:
                logger.warning(f"Word2Vec加载失败: {e}")

        logger.info(f"向量化器已加载: {load_path}")
        return vec

    def __repr__(self) -> str:
        parts = []
        if self.tfidf_word is not None:
            parts.append(f"TF-IDF词:{len(self.tfidf_word.vocabulary_)}维")
        if self.tfidf_char is not None:
            parts.append(f"TF-IDF字符:{len(self.tfidf_char.vocabulary_)}维")
        if self.count_ngram is not None:
            parts.append(f"Ngram:{len(self.count_ngram.vocabulary_)}维")
        if self.word2vec_model is not None:
            parts.append(f"Word2Vec:{self.word2vec_dim}维")
        return f"TextVectorizer({', '.join(parts) if parts else '未拟合'})"
