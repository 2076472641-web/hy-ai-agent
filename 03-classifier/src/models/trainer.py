# -*- coding: utf-8 -*-
"""
统一训练器模块
提供模型训练全流程编排能力，包括：
- 训练流程控制：数据切分、验证评估、最佳权重保存、早停
- 超参搜索：网格搜索 (Grid Search)、随机搜索 (Random Search)
- 交叉验证：K-Fold CV 评估模型稳定性
- 训练日志：详细记录每次训练的参数、指标、耗时，支持 JSON 导出
- 学习率调度：支持 ReduceLROnPlateau 动态调整 LR（BERT模型）
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
import copy
import json
import time
import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    ParameterGrid,
    ParameterSampler,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from loguru import logger

from .base import BaseClassifier
from .random_forest import RandomForestClassifier
from .fasttext_model import FastTextClassifier
from .bert_classifier import BERTClassifier


# ============================================================
# 配置数据类
# ============================================================

@dataclass
class TrainingConfig:
    """
    统一训练器配置

    Attributes:
        output_dir: 输出根目录（模型、日志、检查点）
        random_seed: 全局随机种子
        train_ratio: 训练集比例 (0-1)
        val_ratio: 验证集比例 (0-1)；测试集自动为 1 - train - val
        use_stratified: 分层抽样（保持各类比例）
        early_stopping_patience: 早停耐心值（BERT等分epoch模型）
        save_best_model: 是否保存最佳权重
        primary_metric: 主指标，用于最佳模型判定和早停，如 'f1_macro'
        additional_metrics: 额外指标列表
        enable_logging: 是否保存训练日志
        cv_folds: 交叉验证折数（cross_validate() 用）
        log_classification_report: 是否保存分类报告
        save_confusion_matrix: 是否保存混淆矩阵
    """

    output_dir: str = "data/models"
    random_seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    use_stratified: bool = True
    early_stopping_patience: int = 3
    save_best_model: bool = True
    primary_metric: str = "f1_macro"
    additional_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision_macro", "recall_macro", "f1_micro",
    ])
    enable_logging: bool = True
    cv_folds: int = 5
    log_classification_report: bool = True
    save_confusion_matrix: bool = True


@dataclass
class SearchConfig:
    """
    超参搜索配置

    Attributes:
        method: 'grid'（网格）或 'random'（随机）
        param_grid: 超参搜索空间字典
        n_iter: 随机搜索迭代次数
        cv: 搜索阶段交叉验证折数
        maximize: 是否最大化主指标（分类任务通常 True）
        verbose: 日志详细级别
    """

    method: str = "grid"  # 'grid' or 'random'
    param_grid: Dict[str, Any] = field(default_factory=dict)
    n_iter: int = 20
    cv: int = 3
    maximize: bool = True
    verbose: int = 1


# ============================================================
# 指标计算工具
# ============================================================

ALL_METRICS = {
    "accuracy", "precision_macro", "precision_micro",
    "recall_macro", "recall_micro", "f1_macro", "f1_micro",
}


def compute_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_prob: Optional[np.ndarray] = None,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    计算分类指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（可选，暂用于扩展）
        metrics: 指标名列表，None 则计算所有标准指标

    Returns:
        Dict: {metric_name: value}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if metrics is None:
        metrics = [
            "accuracy", "precision_macro", "recall_macro",
            "f1_macro", "f1_micro",
        ]

    result: Dict[str, float] = {}
    for m in metrics:
        try:
            if m == "accuracy":
                result[m] = float(accuracy_score(y_true, y_pred))
            elif m == "precision_macro":
                result[m] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
            elif m == "precision_micro":
                result[m] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
            elif m == "recall_macro":
                result[m] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
            elif m == "recall_micro":
                result[m] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
            elif m == "f1_macro":
                result[m] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            elif m == "f1_micro":
                result[m] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
            else:
                logger.warning(f"未知指标: {m}，跳过")
        except Exception as e:
            logger.warning(f"计算指标 {m} 失败: {e}")
            result[m] = 0.0

    return result


# ============================================================
# 统一训练器
# ============================================================

class Trainer:
    """
    统一训练器

    对任何继承 BaseClassifier 的模型提供一致的训练体验：
    - train(): 简单训练 + 验证 + 测试评估 + 保存最佳模型
    - cross_validate(): K-Fold 交叉验证，输出各折统计量
    - hyperparam_search(): 网格/随机超参搜索，返回最佳参数与模型
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Args:
            config: TrainingConfig 训练配置
            class_names: 类别名称列表，长度应为12
        """
        self.config = config or TrainingConfig()
        self.class_names = class_names or [f"类别_{i}" for i in range(12)]
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 日志目录
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # 训练历史记录
        self.runs: List[Dict[str, Any]] = []

        logger.info(
            f"[Trainer] 初始化完成 - 输出目录: {self.output_dir}, "
            f"主指标: {self.config.primary_metric}"
        )

    # --------------------------------------------------------
    # 数据划分
    # --------------------------------------------------------

    def split_data(
        self,
        X: Union[List[str], np.ndarray, pd.Series],
        y: Union[List[int], np.ndarray, pd.Series],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        划分训练/验证/测试集

        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if self.config.train_ratio + self.config.val_ratio >= 1.0:
            raise ValueError(
                f"train_ratio({self.config.train_ratio}) + val_ratio({self.config.val_ratio}) 必须 < 1.0"
            )

        test_ratio = 1.0 - self.config.train_ratio - self.config.val_ratio

        # Step 1: (train+val) vs test
        first_split_test = test_ratio / (1.0 - self.config.train_ratio) \
            if (1.0 - self.config.train_ratio) > 0 else test_ratio

        if self.config.use_stratified:
            X_tv, X_test, y_tv, y_test = train_test_split(
                X_arr, y_arr,
                test_size=test_ratio,
                random_state=self.config.random_seed,
                stratify=y_arr,
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_tv, y_tv,
                test_size=self.config.val_ratio / (self.config.train_ratio + self.config.val_ratio),
                random_state=self.config.random_seed,
                stratify=y_tv,
            )
        else:
            X_tv, X_test, y_tv, y_test = train_test_split(
                X_arr, y_arr,
                test_size=test_ratio,
                random_state=self.config.random_seed,
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_tv, y_tv,
                test_size=self.config.val_ratio / (self.config.train_ratio + self.config.val_ratio),
                random_state=self.config.random_seed,
            )

        logger.info(
            f"[Trainer] 数据划分完成 - Train:{len(X_train)}, Val:{len(X_val)}, Test:{len(X_test)}"
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    # --------------------------------------------------------
    # 主训练流程
    # --------------------------------------------------------

    def train(
        self,
        model: BaseClassifier,
        X: Union[List[str], np.ndarray, pd.Series],
        y: Union[List[int], np.ndarray, pd.Series],
        X_val: Optional[Union[List[str], np.ndarray, pd.Series]] = None,
        y_val: Optional[Union[List[int], np.ndarray, pd.Series]] = None,
        X_test: Optional[Union[List[str], np.ndarray, pd.Series]] = None,
        y_test: Optional[Union[List[int], np.ndarray, pd.Series]] = None,
        auto_split: bool = True,
        run_name: Optional[str] = None,
        **fit_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        训练模型并执行完整评估流程

        Args:
            model: BaseClassifier 实例
            X: 完整训练特征（auto_split=True时）或训练集特征
            y: 完整标签或训练集标签
            X_val: 验证集（auto_split=False时使用）
            y_val: 验证集标签
            X_test: 测试集（auto_split=False时使用）
            y_test: 测试集标签
            auto_split: 是否自动划分 train/val/test
            run_name: 本次训练的名字，用于日志命名
            **fit_kwargs: 传递给 model.fit() 的额外参数

        Returns:
            Dict: 本次训练的完整结果（指标、参数、耗时等）
        """
        start_time = time.time()
        run_name = run_name or f"{model.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_output_dir = self.output_dir / run_name
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # 设置类别名称
        model.set_class_names(self.class_names)

        # 数据划分
        if auto_split:
            X_train, y_train, X_val, y_val, X_test, y_test = self.split_data(X, y)
        else:
            if X_val is None or y_val is None:
                raise ValueError("auto_split=False 时必须提供 X_val 和 y_val")
            X_train = np.asarray(X)
            y_train = np.asarray(y)
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val)
            X_test = np.asarray(X_test) if X_test is not None else None
            y_test = np.asarray(y_test) if y_test is not None else None

        logger.info(f"[Trainer] 启动训练 Run: {run_name} (模型: {model.model_name})")

        # 训练
        train_start = time.time()
        model.fit(
            X_train, y_train,
            X_val=X_val.tolist() if X_val is not None else None,
            y_val=y_val.tolist() if y_val is not None else None,
            output_dir=str(run_output_dir),
            early_stopping_patience=self.config.early_stopping_patience,
            save_best=self.config.save_best_model,
            **fit_kwargs,
        )
        train_duration = time.time() - train_start

        # 验证集评估
        val_metrics: Dict[str, float] = {}
        if X_val is not None and y_val is not None:
            val_pred = model.predict(X_val)
            val_metrics = compute_metrics(
                y_val, val_pred,
                metrics=[self.config.primary_metric] + self.config.additional_metrics,
            )
            logger.info(
                f"[Trainer] Val 指标 - "
                + ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            )

        # 测试集评估
        test_metrics: Dict[str, float] = {}
        test_report: Optional[Dict] = None
        test_cm: Optional[List[List[int]]] = None
        if X_test is not None and y_test is not None:
            test_pred = model.predict(X_test)
            test_proba = model.predict_proba(X_test)
            test_metrics = compute_metrics(
                y_test, test_pred,
                metrics=[self.config.primary_metric] + self.config.additional_metrics,
            )
            logger.info(
                f"[Trainer] Test 指标 - "
                + ", ".join(f"{k}={v:.4f}" for k, v in test_metrics.items())
            )
            if self.config.log_classification_report:
                test_report = classification_report(
                    y_test, test_pred,
                    target_names=self.class_names,
                    output_dict=True,
                    zero_division=0,
                )
            if self.config.save_confusion_matrix:
                test_cm = confusion_matrix(y_test, test_pred, labels=list(range(12))).tolist()

        # 保存模型
        if self.config.save_best_model:
            model.save(str(run_output_dir))

        # 组装结果
        result: Dict[str, Any] = {
            "run_name": run_name,
            "model_name": model.model_name,
            "model_class": type(model).__name__,
            "params": model.get_params(),
            "dataset": {
                "train_size": len(X_train),
                "val_size": len(X_val) if X_val is not None else 0,
                "test_size": len(X_test) if X_test is not None else 0,
            },
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "classification_report": test_report,
            "confusion_matrix": test_cm,
            "train_duration_sec": round(train_duration, 2),
            "total_duration_sec": round(time.time() - start_time, 2),
            "timestamp": datetime.now().isoformat(),
        }

        # 保存日志
        if self.config.enable_logging:
            log_path = run_output_dir / "run_result.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"[Trainer] 训练结果已保存: {log_path}")

        # 汇总 runs
        self.runs.append(result)
        self._save_summary()

        return result

    # --------------------------------------------------------
    # 交叉验证
    # --------------------------------------------------------

    def cross_validate(
        self,
        model_cls: type,
        X: Union[List[str], np.ndarray, pd.Series],
        y: Union[List[int], np.ndarray, pd.Series],
        model_params: Optional[Dict[str, Any]] = None,
        folds: Optional[int] = None,
        **fit_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        执行 K-Fold 交叉验证（使用 StratifiedKFold）

        Args:
            model_cls: BaseClassifier 的子类（Class，不是实例）
            X: 特征
            y: 标签
            model_params: 构造模型的参数字典
            folds: 折数，None 则使用 config.cv_folds
            **fit_kwargs: 传递给 fit() 的额外参数

        Returns:
            Dict: {
                'fold_metrics': [...每折指标...],
                'mean': {平均指标},
                'std': {标准差},
            }
        """
        folds = folds or self.config.cv_folds
        model_params = model_params or {}
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if self.config.use_stratified:
            splitter = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=self.config.random_seed,
            )
        else:
            splitter = KFold(
                n_splits=folds,
                shuffle=True,
                random_state=self.config.random_seed,
            )

        fold_metrics: List[Dict[str, float]] = []
        all_metric_names = [self.config.primary_metric] + self.config.additional_metrics

        logger.info(f"[Trainer] 启动 {folds}-Fold 交叉验证 (模型: {model_cls.__name__})")

        for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X_arr, y_arr), 1):
            X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
            y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]

            logger.info(f"[Trainer] Fold {fold_idx}/{folds} - Train:{len(X_tr)}, Val:{len(X_va)}")

            fold_model = model_cls(**model_params)
            fold_model.set_class_names(self.class_names)

            fold_model.fit(
                X_tr.tolist(), y_tr.tolist(),
                X_val=X_va.tolist(), y_val=y_va.tolist(),
                **fit_kwargs,
            )
            va_pred = fold_model.predict(X_va.tolist())
            metrics = compute_metrics(y_va, va_pred, metrics=all_metric_names)
            fold_metrics.append(metrics)
            logger.info(
                f"[Trainer] Fold {fold_idx} 指标 - "
                + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )

        # 统计均值和标准差
        metric_keys = fold_metrics[0].keys()
        mean_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in metric_keys}
        std_metrics = {k: float(np.std([m[k] for m in fold_metrics])) for k in metric_keys}

        result = {
            "model_class": model_cls.__name__,
            "model_params": model_params,
            "folds": folds,
            "fold_metrics": fold_metrics,
            "mean": mean_metrics,
            "std": std_metrics,
        }

        logger.info(
            f"[Trainer] 交叉验证完成 - 平均 {self.config.primary_metric}: "
            f"{mean_metrics[self.config.primary_metric]:.4f} "
            f"(±{std_metrics[self.config.primary_metric]:.4f})"
        )

        # 保存 CV 结果
        if self.config.enable_logging:
            cv_log_path = self.log_dir / f"cv_{model_cls.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(cv_log_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    # --------------------------------------------------------
    # 超参搜索
    # --------------------------------------------------------

    def hyperparam_search(
        self,
        model_cls: type,
        X: Union[List[str], np.ndarray, pd.Series],
        y: Union[List[int], np.ndarray, pd.Series],
        search_config: Optional[SearchConfig] = None,
        X_val: Optional[Union[List[str], np.ndarray, pd.Series]] = None,
        y_val: Optional[Union[List[int], np.ndarray, pd.Series]] = None,
        **fit_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        超参搜索：网格搜索或随机搜索

        Args:
            model_cls: BaseClassifier 的子类
            X: 训练特征
            y: 训练标签
            search_config: SearchConfig 搜索配置
            X_val: 外部验证集（不提供则在 X 内做 CV）
            y_val: 外部验证集标签
            **fit_kwargs: 额外训练参数

        Returns:
            Dict: {
                'best_params': 最佳超参数,
                'best_score': 最佳主指标得分,
                'best_model': 以最佳参数训练好的模型实例,
                'search_results': [...所有实验的结果...]
            }
        """
        search_config = search_config or SearchConfig()
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if not search_config.param_grid:
            raise ValueError("param_grid 不能为空")

        # 生成参数候选列表
        if search_config.method == "grid":
            param_list = list(ParameterGrid(search_config.param_grid))
        elif search_config.method == "random":
            param_list = list(ParameterSampler(
                search_config.param_grid,
                n_iter=search_config.n_iter,
                random_state=self.config.random_seed,
            ))
        else:
            raise ValueError(f"未知搜索方法: {search_config.method}")

        logger.info(
            f"[Trainer] 启动超参搜索 - 方法: {search_config.method}, "
            f"实验数: {len(param_list)}, CV: {search_config.cv}"
        )

        all_results: List[Dict[str, Any]] = []
        primary = self.config.primary_metric

        for idx, trial_params in enumerate(param_list, 1):
            trial_start = time.time()
            logger.info(
                f"[Trainer] Trial {idx}/{len(param_list)} - 参数: {trial_params}"
            )

            if X_val is not None and y_val is not None:
                # 外部验证集：训练一次
                trial_model = model_cls(**trial_params)
                trial_model.set_class_names(self.class_names)
                trial_model.fit(
                    X_arr, y_arr,
                    X_val=np.asarray(X_val).tolist(),
                    y_val=np.asarray(y_val).tolist(),
                    **fit_kwargs,
                )
                va_pred = trial_model.predict(np.asarray(X_val).tolist())
                metrics = compute_metrics(
                    np.asarray(y_val).tolist(), va_pred,
                    metrics=[primary] + self.config.additional_metrics,
                )
                score = metrics[primary]
            else:
                # CV 内部划分
                if self.config.use_stratified:
                    splitter = StratifiedKFold(
                        n_splits=search_config.cv,
                        shuffle=True,
                        random_state=self.config.random_seed,
                    )
                else:
                    splitter = KFold(
                        n_splits=search_config.cv,
                        shuffle=True,
                        random_state=self.config.random_seed,
                    )

                fold_scores = []
                for tr_idx, va_idx in splitter.split(X_arr, y_arr):
                    X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
                    y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]
                    fold_model = model_cls(**trial_params)
                    fold_model.set_class_names(self.class_names)
                    fold_model.fit(
                        X_tr.tolist(), y_tr.tolist(),
                        **fit_kwargs,
                    )
                    va_pred = fold_model.predict(X_va.tolist())
                    fm = compute_metrics(y_va, va_pred, metrics=[primary])
                    fold_scores.append(fm[primary])
                score = float(np.mean(fold_scores))
                metrics = {primary: score}

            elapsed = round(time.time() - trial_start, 2)
            all_results.append({
                "trial": idx,
                "params": trial_params,
                "score": score,
                "metrics": metrics,
                "duration_sec": elapsed,
            })
            logger.info(
                f"[Trainer] Trial {idx} 完成 - {primary}: {score:.4f} ({elapsed}s)"
            )

        # 选最优
        if search_config.maximize:
            best_idx = int(np.argmax([r["score"] for r in all_results]))
        else:
            best_idx = int(np.argmin([r["score"] for r in all_results]))

        best_result = all_results[best_idx]
        best_params = best_result["params"]
        best_score = best_result["score"]

        logger.info(
            f"[Trainer] 搜索完成 - 最佳 #{best_result['trial']}: {primary}={best_score:.4f}, "
            f"参数: {best_params}"
        )

        # 用最佳参数重训完整数据
        best_model = model_cls(**best_params)
        best_model.set_class_names(self.class_names)
        best_model.fit(X_arr, y_arr, X_val=X_val, y_val=y_val, **fit_kwargs)

        result = {
            "search_method": search_config.method,
            "num_trials": len(param_list),
            "cv_folds": search_config.cv,
            "best_params": best_params,
            "best_score": best_score,
            "best_model": best_model,
            "search_results": all_results,
        }

        # 保存搜索日志
        if self.config.enable_logging:
            search_log = {k: v for k, v in result.items() if k != "best_model"}
            log_path = self.log_dir / f"search_{model_cls.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(search_log, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"[Trainer] 超参搜索结果已保存: {log_path}")

        return result

    # --------------------------------------------------------
    # 训练记录汇总
    # --------------------------------------------------------

    def _save_summary(self) -> None:
        """保存全部 run 的汇总 CSV/JSON"""
        if not self.runs:
            return
        try:
            summary_path = self.log_dir / "runs_summary.json"
            serializable = []
            for r in self.runs:
                r_copy = {}
                for k, v in r.items():
                    if k == "confusion_matrix" and v is not None:
                        r_copy[k] = v
                    elif isinstance(v, (np.ndarray, np.floating, np.integer)):
                        r_copy[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v.tolist()
                    elif isinstance(v, Path):
                        r_copy[k] = str(v)
                    else:
                        r_copy[k] = v
                serializable.append(r_copy)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)

            # 简化 CSV：只保留扁平字段
            rows = []
            for r in serializable:
                row = {
                    "run_name": r.get("run_name"),
                    "model_class": r.get("model_class"),
                    "train_duration_sec": r.get("train_duration_sec"),
                }
                for mk, mv in r.get("test_metrics", {}).items():
                    row[f"test_{mk}"] = mv
                for mk, mv in r.get("val_metrics", {}).items():
                    row[f"val_{mk}"] = mv
                rows.append(row)
            if rows:
                pd.DataFrame(rows).to_csv(self.log_dir / "runs_summary.csv", index=False, encoding="utf-8-sig")
        except Exception as e:
            logger.warning(f"[Trainer] 保存汇总日志失败: {e}")

    # --------------------------------------------------------
    # 便捷方法：批量跑多种模型对比
    # --------------------------------------------------------

    def benchmark(
        self,
        X: Union[List[str], np.ndarray, pd.Series],
        y: Union[List[int], np.ndarray, pd.Series],
        models_config: Optional[List[Tuple[type, Dict[str, Any]]]] = None,
        auto_split: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        便捷基准测试：在同一数据集上依次训练多个模型并输出对比

        Args:
            X: 特征
            y: 标签
            models_config: [(模型类, 参数字典), ...]，
                          None 则使用默认三个模型的默认参数
            auto_split: 是否自动划分数据

        Returns:
            List[Dict]: 每个模型的训练结果字典
        """
        if models_config is None:
            models_config = [
                (RandomForestClassifier, {"n_estimators": 200, "max_depth": 30}),
                (FastTextClassifier, {"epoch": 30, "lr": 0.1}),
                (BERTClassifier, {"num_train_epochs": 3, "batch_size": 32}),
            ]

        results = []
        for idx, (model_cls, params) in enumerate(models_config, 1):
            logger.info(f"[Trainer] Benchmark {idx}/{len(models_config)}: {model_cls.__name__}")
            try:
                model = model_cls(**params)
                res = self.train(model, X, y, auto_split=auto_split)
                results.append(res)
            except Exception as e:
                logger.exception(f"[Trainer] Benchmark 模型 {model_cls.__name__} 失败: {e}")
                results.append({"model_class": model_cls.__name__, "error": str(e)})
        return results
