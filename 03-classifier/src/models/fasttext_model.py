# -*- coding: utf-8 -*-
"""
FastText 分类模型
Facebook 开源的高效文本分类库，基于字符级 n-gram 特征，适合高并发场景。

核心特性：
1. 训练/推理速度极快，适合海量数据和高并发在线服务
2. 支持软标签优化：
   - 标签平滑 (Label Smoothing)：通过降低硬标签置信度处理冷门类目
   - 知识蒸馏 (Knowledge Distillation)：用教师模型（如BERT）的软标签指导训练
3. 字符级 n-gram 子词信息增强，对中文短文本鲁棒性好
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import tempfile
import os
import copy

import numpy as np
import pandas as pd
import jieba
from loguru import logger

try:
    import fasttext
except ImportError:
    fasttext = None
    logger.warning("fasttext 未安装，FastTextClassifier 将无法使用")

from .base import BaseClassifier


class FastTextClassifier(BaseClassifier):
    """
    FastText 文本分类器
    
    高性能轻量级分类模型，适配高并发在线推理场景。
    支持标签平滑和知识蒸馏两种软标签优化策略。

    Args:
        model_name: 模型名称标识
        lr: 学习率，默认 0.1
        epoch: 训练轮数，默认 50
        wordNgrams: n-gram 范围，默认 2（考虑2-gram短语）
        dim: 词向量维度，默认 128
        loss: 损失函数，'softmax' 或 'hs'（层次softmax，超大数据集更快）
        bucket: 字符n-gram哈希桶大小
        minCount: 最小词频
        label_smoothing: 标签平滑系数 epsilon，0表示不启用
        use_jieba: 是否使用 Jieba 预处理分词
        random_state: 随机种子
    """

    def __init__(
        self,
        model_name: str = "fasttext",
        lr: float = 0.1,
        epoch: int = 50,
        wordNgrams: int = 2,
        dim: int = 128,
        loss: str = "softmax",
        bucket: int = 200000,
        minCount: int = 2,
        label_smoothing: float = 0.0,
        use_jieba: bool = True,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)

        if fasttext is None:
            raise ImportError(
                "fasttext 库未安装，请先执行: pip install fasttext"
            )

        # FastText 核心超参数
        self.lr = lr
        self.epoch = epoch
        self.wordNgrams = wordNgrams
        self.dim = dim
        self.loss = loss
        self.bucket = bucket
        self.minCount = minCount
        self.label_smoothing = label_smoothing
        self.use_jieba = use_jieba
        self.random_state = random_state

        # 模型对象
        self._model: Optional[Any] = None

        # 保存超参数
        self.params.update({
            "lr": lr,
            "epoch": epoch,
            "wordNgrams": wordNgrams,
            "dim": dim,
            "loss": loss,
            "bucket": bucket,
            "minCount": minCount,
            "label_smoothing": label_smoothing,
            "use_jieba": use_jieba,
            "random_state": random_state,
        })

    def _preprocess_text(self, text: str) -> str:
        """
        文本预处理：分词并用空格连接（FastText要求空格分隔）

        Args:
            text: 原始文本

        Returns:
            str: 预处理后的文本
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.strip()
        if self.use_jieba:
            return " ".join(jieba.cut(text))
        else:
            return " ".join(list(text))  # 字粒度

    def _label_str(self, label_id: int) -> str:
        """
        构造 FastText 要求的标签格式：__label__{id}

        Args:
            label_id: 类别ID 0-11

        Returns:
            str: FastText 格式标签字符串
        """
        return f"__label__{label_id}"

    def _parse_label(self, label_str: str) -> int:
        """
        解析 FastText 标签字符串为类别ID

        Args:
            label_str: FastText 格式标签，如 '__label__3'

        Returns:
            int: 类别ID
        """
        return int(label_str.replace("__label__", ""))

    def _write_train_file(
        self,
        X: List[str],
        y: Union[np.ndarray, List[int]],
        file_path: Union[str, Path],
        soft_labels: Optional[np.ndarray] = None,
    ) -> None:
        """
        将数据写入 FastText 训练文件格式

        普通格式（硬标签）：__label__0 我 爱 中国
        软标签格式（知识蒸馏）：__label__0 0.9 __label__1 0.05 ... 我 爱 中国

        Args:
            X: 文本列表
            y: 硬标签列表
            file_path: 输出文件路径
            soft_labels: 软标签概率矩阵 (n_samples, num_classes)，用于知识蒸馏
        """
        with open(file_path, "w", encoding="utf-8") as f:
            for idx, (text, label_id) in enumerate(zip(X, y)):
                processed_text = self._preprocess_text(text)
                if not processed_text:
                    continue

                line_parts = []

                if soft_labels is not None:
                    # 知识蒸馏：输出所有类别的软标签
                    probs = soft_labels[idx]
                    for cls_id, prob in enumerate(probs):
                        line_parts.append(f"{self._label_str(cls_id)} {prob:.6f}")
                elif self.label_smoothing > 0:
                    # 标签平滑：对正确标签分配 (1-eps)，其余均分 eps/(K-1)
                    eps = self.label_smoothing
                    K = self.num_classes
                    for cls_id in range(K):
                        if cls_id == int(label_id):
                            p = 1.0 - eps
                        else:
                            p = eps / (K - 1)
                        line_parts.append(f"{self._label_str(cls_id)} {p:.6f}")
                else:
                    # 普通硬标签
                    line_parts.append(self._label_str(int(label_id)))

                line_parts.append(processed_text)
                f.write(" ".join(line_parts) + "\n")

    def fit(
        self,
        X: Union[List[str], np.ndarray, pd.Series],
        y: Union[List[int], np.ndarray, pd.Series],
        X_val: Optional[Union[List[str], np.ndarray, pd.Series]] = None,
        y_val: Optional[Union[List[int], np.ndarray, pd.Series]] = None,
        teacher_probs: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "FastTextClassifier":
        """
        训练 FastText 分类模型

        Args:
            X: 训练文本
            y: 训练标签 0-11
            X_val: 验证集文本（可选，用于日志）
            y_val: 验证集标签（可选）
            teacher_probs: 教师模型的软标签概率矩阵 (n_samples, 12)，
                          用于知识蒸馏优化冷门类目。若提供则优先使用蒸馏。
            **kwargs: 额外参数

        Returns:
            self: 训练完成的模型
        """
        X_array, y_array = self._validate_inputs(X, y)
        X_list = X_array.tolist()
        y_list = y_array.tolist()

        # 知识蒸馏检测
        use_distillation = teacher_probs is not None
        if use_distillation:
            teacher_probs = np.asarray(teacher_probs)
            if teacher_probs.shape != (len(X_list), self.num_classes):
                raise ValueError(
                    f"teacher_probs 形状错误: {teacher_probs.shape}，"
                    f"期望: ({len(X_list)}, {self.num_classes})"
                )
            logger.info(
                f"[{self.model_name}] 启用知识蒸馏训练 - 使用教师软标签"
            )
        elif self.label_smoothing > 0:
            logger.info(
                f"[{self.model_name}] 启用标签平滑训练 - eps={self.label_smoothing}"
            )
        else:
            logger.info(f"[{self.model_name}] 使用标准硬标签训练")

        # 写入临时训练文件
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".train", delete=False, encoding="utf-8"
        ) as tmp:
            train_file = tmp.name

        try:
            self._write_train_file(X_list, y_list, train_file, soft_labels=teacher_probs)
            logger.info(
                f"[{self.model_name}] 训练文件已生成: {train_file}, 样本数: {len(X_list)}"
            )

            # 调用 fasttext.train_supervised
            train_kwargs = dict(
                input=train_file,
                lr=self.lr,
                epoch=self.epoch,
                wordNgrams=self.wordNgrams,
                dim=self.dim,
                loss=self.loss,
                bucket=self.bucket,
                minCount=self.minCount,
                seed=self.random_state,
                verbose=2,
            )

            if use_distillation or self.label_smoothing > 0:
                # 软标签训练使用回归损失变体
                train_kwargs["loss"] = "softmax"
                logger.info(
                    f"[{self.model_name}] 启动软标签训练 - lr={self.lr}, epoch={self.epoch}"
                )
            else:
                logger.info(
                    f"[{self.model_name}] 启动标准训练 - lr={self.lr}, epoch={self.epoch}, "
                    f"ngram={self.wordNgrams}, dim={self.dim}"
                )

            self._model = fasttext.train_supervised(**train_kwargs)
            logger.info(f"[{self.model_name}] FastText 训练完成")

        finally:
            # 清理临时文件
            if os.path.exists(train_file):
                os.remove(train_file)

        # 验证集评估
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            correct = np.sum(val_pred == np.asarray(y_val))
            acc = correct / len(y_val)
            logger.info(f"[{self.model_name}] 验证集 Accuracy: {acc:.4f}")

        self.is_trained = True
        return self

    def predict(
        self, X: Union[List[str], np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        预测类别ID

        Args:
            X: 待预测文本

        Returns:
            np.ndarray: 预测类别ID数组
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用 fit() 方法")

        X_list = np.asarray(X).tolist()
        if not isinstance(X_list, list):
            X_list = [X_list]

        processed = [self._preprocess_text(x) for x in X_list]
        labels, _ = self._model.predict(processed, k=1)

        # labels 是 list of list，如 [['__label__3'], ['__label__5']]
        pred_ids = np.array([
            self._parse_label(labels[i][0]) if len(labels[i]) > 0 else -1
            for i in range(len(processed))
        ])
        return pred_ids

    def predict_proba(
        self, X: Union[List[str], np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        预测类别概率分布

        Args:
            X: 待预测文本

        Returns:
            np.ndarray: 概率矩阵 (n_samples, 12)
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用 fit() 方法")

        X_list = np.asarray(X).tolist()
        if not isinstance(X_list, list):
            X_list = [X_list]

        processed = [self._preprocess_text(x) for x in X_list]
        labels_list, probs_list = self._model.predict(
            processed, k=self.num_classes
        )

        n = len(processed)
        full_proba = np.zeros((n, self.num_classes), dtype=np.float32)

        for i in range(n):
            labels = labels_list[i]
            probs = probs_list[i]
            for label_str, prob in zip(labels, probs):
                cls_id = self._parse_label(label_str)
                if 0 <= cls_id < self.num_classes:
                    full_proba[i, cls_id] = prob

        # 归一化（由于浮点数精度问题，概率和可能略有偏差）
        row_sums = full_proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        full_proba = full_proba / row_sums

        return full_proba

    def tune_hyperparams(
        self,
        X: Union[List[str], np.ndarray, pd.Series],
        y: Union[List[int], np.ndarray, pd.Series],
        param_grid: Optional[Dict[str, List[Any]]] = None,
        cv: int = 3,
    ) -> Dict[str, Any]:
        """
        简易超参搜索（交叉验证手动实现，FastText无原生CV）

        Args:
            X: 训练文本
            y: 训练标签
            param_grid: 搜索空间字典，如 {'lr': [0.05, 0.1, 0.5], 'epoch': [20, 50]}
            cv: 交叉验证折数

        Returns:
            Dict: 最佳超参数组合
        """
        from sklearn.model_selection import KFold
        from sklearn.metrics import f1_macro

        X_array, y_array = self._validate_inputs(X, y)
        X_list = X_array.tolist()
        y_list = y_array.tolist()

        if param_grid is None:
            param_grid = {
                "lr": [0.05, 0.1, 0.2],
                "epoch": [30, 50],
                "wordNgrams": [1, 2],
                "dim": [64, 128],
            }

        logger.info(
            f"[{self.model_name}] 启动超参搜索 - CV={cv}, 搜索空间: {param_grid}"
        )

        # 生成所有参数组合
        import itertools
        keys = list(param_grid.keys())
        values_list = list(param_grid.values())
        combinations = list(itertools.product(*values_list))

        best_score = -1.0
        best_params = None

        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        for combo_idx, combo in enumerate(combinations):
            combo_params = dict(zip(keys, combo))
            fold_scores = []

            logger.info(
                f"[{self.model_name}] 参数组合 {combo_idx+1}/{len(combinations)}: {combo_params}"
            )

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_list)):
                X_train_fold = [X_list[i] for i in train_idx]
                y_train_fold = [y_list[i] for i in train_idx]
                X_val_fold = [X_list[i] for i in val_idx]
                y_val_fold = np.array([y_list[i] for i in val_idx])

                fold_model = FastTextClassifier(
                    model_name=f"fold_{fold_idx}",
                    **combo_params,
                    use_jieba=self.use_jieba,
                    label_smoothing=self.label_smoothing,
                )
                fold_model.fit(X_train_fold, y_train_fold)
                val_pred = fold_model.predict(X_val_fold)
                score = f1_macro(y_val_fold, val_pred)
                fold_scores.append(score)
                logger.debug(
                    f"[{self.model_name}]   Fold {fold_idx+1} Macro-F1: {score:.4f}"
                )

            avg_score = np.mean(fold_scores)
            logger.info(
                f"[{self.model_name}] 组合平均 Macro-F1: {avg_score:.4f}"
            )

            if avg_score > best_score:
                best_score = avg_score
                best_params = combo_params

        logger.info(
            f"[{self.model_name}] 超参搜索完成 - 最佳得分: {best_score:.4f}, "
            f"最佳参数: {best_params}"
        )
        return best_params

    def save(self, save_dir: Union[str, Path]) -> str:
        """
        保存 FastText 模型到磁盘

        Args:
            save_dir: 保存目录

        Returns:
            str: 保存路径
        """
        if not self.is_trained or self._model is None:
            raise RuntimeError("模型尚未训练，无法保存")

        save_path = Path(save_dir) / self.model_name
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存元数据
        self._save_metadata(save_path)

        # FastText 使用原生 .bin 格式保存
        model_path = save_path / "model.bin"
        self._model.save_model(str(model_path))

        logger.info(f"[{self.model_name}] 模型已保存到: {save_path}")
        return str(save_path)

    def load(self, load_dir: Union[str, Path]) -> "FastTextClassifier":
        """
        从磁盘加载 FastText 模型

        Args:
            load_dir: 加载目录

        Returns:
            self: 加载完成的模型
        """
        load_path = Path(load_dir)

        # 加载元数据
        self._load_metadata(load_path)

        # 恢复属性
        for key, val in self.params.items():
            if hasattr(self, key):
                setattr(self, key, val)

        # 加载 FastText 模型
        model_path = load_path / "model.bin"
        if not model_path.exists():
            raise FileNotFoundError(f"FastText 模型文件不存在: {model_path}")
        self._model = fasttext.load_model(str(model_path))

        self.is_trained = True
        logger.info(f"[{self.model_name}] 模型已从 {load_path} 加载")
        return self
