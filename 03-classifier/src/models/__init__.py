# -*- coding: utf-8 -*-
"""
模型训练模块
提供多种分类模型的统一接口，包括：
- 基类：BaseClassifier
- 传统机器学习：RandomForestClassifier（TF-IDF + 随机森林）
- 轻量级高效模型：FastTextClassifier
- 深度学习：BERTClassifier（BERT微调）
- 统一训练器：Trainer
"""

from .base import BaseClassifier
from .random_forest import RandomForestClassifier
from .fasttext_model import FastTextClassifier
from .bert_classifier import BERTClassifier
from .trainer import Trainer, TrainingConfig, SearchConfig

__all__ = [
    "BaseClassifier",
    "RandomForestClassifier",
    "FastTextClassifier",
    "BERTClassifier",
    "Trainer",
    "TrainingConfig",
    "SearchConfig",
]
