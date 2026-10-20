# -*- coding: utf-8 -*-
"""
模型基类模块
定义所有分类模型的通用接口规范，确保不同模型的使用方式保持一致。
所有具体分类器都必须继承 BaseClassifier 并实现其抽象方法。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


class BaseClassifier(ABC):
    """
    分类器基类（抽象类）
    
    定义了所有分类模型必须实现的统一接口：
    - fit(): 模型训练
    - predict(): 预测类别ID
    - predict_proba(): 预测类别概率分布
    - save(): 保存模型到磁盘
    - load(): 从磁盘加载模型
    - get_params(): 获取模型超参数
    - set_params(): 设置模型超参数
    
    属性:
        num_classes (int): 分类类别数量，固定为12（十二大类商品）
        model_name (str): 模型名称标识
        is_trained (bool): 模型是否已训练标志
        params (Dict): 模型超参数字典
    """

    NUM_CLASSES = 12  # 十二大类商品，类目ID 0-11

    def __init__(self, model_name: str = "base_classifier", **kwargs):
        """
        初始化基类分类器

        Args:
            model_name: 模型名称标识，用于日志和保存路径
            **kwargs: 模型超参数，具体参数由子类定义
        """
        self.num_classes: int = self.NUM_CLASSES
        self.model_name: str = model_name
        self.is_trained: bool = False
        self.params: Dict[str, Any] = kwargs
        self._class_names: Optional[List[str]] = None

        logger.info(f"初始化模型: {self.model_name}, 类别数: {self.num_classes}")

    @abstractmethod
    def fit(
        self,
        X: Union[List[str], np.ndarray, pd.Series],
        y: Union[List[int], np.ndarray, pd.Series],
        X_val: Optional[Union[List[str], np.ndarray, pd.Series]] = None,
        y_val: Optional[Union[List[int], np.ndarray, pd.Series]] = None,
        **kwargs,
    ) -> "BaseClassifier":
        """
        训练模型（抽象方法，子类必须实现）

        Args:
            X: 训练特征数据，可以是文本列表、TF-IDF矩阵或其他特征格式
            y: 训练标签，类目ID 0-11
            X_val: 验证集特征数据（可选）
            y_val: 验证集标签（可选）
            **kwargs: 额外训练参数

        Returns:
            self: 训练完成的模型实例
        """
        pass

    @abstractmethod
    def predict(
        self, X: Union[List[str], np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        预测样本的类别ID（抽象方法，子类必须实现）

        Args:
            X: 待预测的特征数据

        Returns:
            np.ndarray: 预测的类别ID数组，形状 (n_samples,)，取值 0-11
        """
        pass

    @abstractmethod
    def predict_proba(
        self, X: Union[List[str], np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        预测样本属于各类别的概率分布（抽象方法，子类必须实现）

        Args:
            X: 待预测的特征数据

        Returns:
            np.ndarray: 类别概率矩阵，形状 (n_samples, num_classes)，每行和为1
        """
        pass

    @abstractmethod
    def save(self, save_dir: Union[str, Path]) -> str:
        """
        保存模型到磁盘（抽象方法，子类必须实现）

        Args:
            save_dir: 保存目录路径

        Returns:
            str: 实际保存的路径
        """
        pass

    @abstractmethod
    def load(self, load_dir: Union[str, Path]) -> "BaseClassifier":
        """
        从磁盘加载模型（抽象方法，子类必须实现）

        Args:
            load_dir: 模型加载目录路径

        Returns:
            self: 加载完成的模型实例
        """
        pass

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        获取模型超参数

        Args:
            deep: 是否深度复制参数（兼容scikit-learn接口）

        Returns:
            Dict[str, Any]: 模型超参数字典
        """
        params = self.params.copy()
        params["model_name"] = self.model_name
        return params

    def set_params(self, **params) -> "BaseClassifier":
        """
        设置模型超参数

        Args:
            **params: 要更新的超参数键值对

        Returns:
            self: 更新参数后的模型实例
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            self.params[key] = value
        logger.debug(f"模型 {self.model_name} 更新参数: {params}")
        return self

    def set_class_names(self, class_names: List[str]) -> None:
        """
        设置类别名称映射

        Args:
            class_names: 类别名称列表，索引对应类别ID
        """
        if len(class_names) != self.num_classes:
            raise ValueError(
                f"类别名称数量 {len(class_names)} 与类别数 {self.num_classes} 不匹配"
            )
        self._class_names = class_names

    def get_class_names(self) -> List[str]:
        """
        获取类别名称列表

        Returns:
            List[str]: 类别名称列表，若未设置则返回默认名称
        """
        if self._class_names is None:
            self._class_names = [f"类别_{i}" for i in range(self.num_classes)]
        return self._class_names

    def predict_with_names(
        self, X: Union[List[str], np.ndarray, pd.Series]
    ) -> List[Dict[str, Any]]:
        """
        预测并返回包含类别名称和置信度的详细结果

        Args:
            X: 待预测的特征数据

        Returns:
            List[Dict]: 每个样本的预测结果，包含:
                - label: 预测类别ID
                - name: 预测类别名称
                - confidence: 预测置信度
                - probabilities: 各类别概率
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用 fit() 方法")

        probas = self.predict_proba(X)
        labels = np.argmax(probas, axis=1)
        class_names = self.get_class_names()

        results = []
        for idx, (label, proba) in enumerate(zip(labels, probas)):
            results.append({
                "label": int(label),
                "name": class_names[label],
                "confidence": float(proba[label]),
                "probabilities": {
                    class_names[i]: float(p) for i, p in enumerate(proba)
                },
            })
        return results

    def _validate_inputs(
        self,
        X: Union[List[str], np.ndarray, pd.Series],
        y: Union[List[int], np.ndarray, pd.Series],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        验证训练输入数据的合法性

        Args:
            X: 训练特征
            y: 训练标签

        Returns:
            Tuple[np.ndarray, np.ndarray]: 转换后的 (X_array, y_array)

        Raises:
            ValueError: 当数据格式或内容不合法时
        """
        X_array = np.asarray(X)
        y_array = np.asarray(y)

        if len(X_array) != len(y_array):
            raise ValueError(
                f"X 样本数 ({len(X_array)}) 与 y 标签数 ({len(y_array)}) 不匹配"
            )

        if len(X_array) == 0:
            raise ValueError("训练数据不能为空")

        unique_labels = np.unique(y_array)
        if not np.all(np.isin(unique_labels, np.arange(self.num_classes))):
            invalid = unique_labels[~np.isin(unique_labels, np.arange(self.num_classes))]
            raise ValueError(
                f"标签包含超出范围的值: {invalid.tolist()}，有效范围 0-{self.num_classes - 1}"
            )

        return X_array, y_array

    def _save_metadata(self, save_dir: Union[str, Path]) -> None:
        """
        保存模型元数据（通用辅助方法）

        Args:
            save_dir: 保存目录
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "is_trained": self.is_trained,
            "params": self.params,
            "class_names": self._class_names,
        }

        metadata_path = save_path / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"模型元数据已保存: {metadata_path}")

    def _load_metadata(self, load_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        加载模型元数据（通用辅助方法）

        Args:
            load_dir: 加载目录

        Returns:
            Dict[str, Any]: 元数据字典
        """
        load_path = Path(load_dir)
        metadata_path = load_path / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.model_name = metadata["model_name"]
        self.num_classes = metadata["num_classes"]
        self.is_trained = metadata["is_trained"]
        self.params = metadata["params"]
        self._class_names = metadata["class_names"]

        logger.info(f"模型元数据已加载: {metadata_path}")
        return metadata

    def __repr__(self) -> str:
        """模型字符串表示"""
        status = "已训练" if self.is_trained else "未训练"
        return (
            f"<{self.__class__.__name__}(name={self.model_name}, "
            f"classes={self.num_classes}, status={status})>"
        )
