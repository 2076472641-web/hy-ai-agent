# -*- coding: utf-8 -*-
"""
模型评估器模块
以F1-score为核心指标，提供完整的分类模型评估能力：
- 核心分类指标：accuracy / precision / recall / F1_micro / F1_macro
- 逐类分类报告：每个类别的precision/recall/f1/support
- 混淆矩阵热力图：12x12类目可视化PNG输出
- ROC曲线：多分类One-vs-Rest ROC曲线（可选，需概率预测）
- sklearn风格API，与BaseClassifier模型完全兼容
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize
from loguru import logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# 兼容无中文字体环境
try:
    plt.rcParams["font.sans-serif"] = [
        "SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

from ..models.base import BaseClassifier


# ============================================================
# 默认类目名称（十二大类商品）
# ============================================================

DEFAULT_CLASS_NAMES: List[str] = [
    "服装鞋帽", "数码家电", "美妆护肤", "食品生鲜",
    "家居家装", "母婴用品", "运动户外", "图书文娱",
    "珠宝饰品", "汽车用品", "宠物用品", "医药保健",
]

NUM_CLASSES: int = len(DEFAULT_CLASS_NAMES)


# ============================================================
# 评估结果数据结构
# ============================================================

@dataclass
class EvaluationResult:
    """
    模型评估结果数据类

    Attributes:
        model_name: 模型名称标识
        accuracy: 总体准确率
        precision_macro: 宏平均精确率
        precision_micro: 微平均精确率
        recall_macro: 宏平均召回率
        recall_micro: 微平均召回率
        f1_macro: 宏平均F1分数（核心指标）
        f1_micro: 微平均F1分数
        classification_report: sklearn分类报告字典（逐类）
        confusion_matrix: 混淆矩阵numpy数组 (12, 12)
        class_names: 类目名称列表
        num_samples: 评估样本总数
        roc_auc_macro: 宏平均ROC-AUC（可选）
        roc_auc_per_class: 各类别ROC-AUC（可选）
        pr_auc_per_class: 各类别PR-AUC（可选）
        extra_metrics: 额外自定义指标
    """
    model_name: str = "unknown"
    accuracy: float = 0.0
    precision_macro: float = 0.0
    precision_micro: float = 0.0
    recall_macro: float = 0.0
    recall_micro: float = 0.0
    f1_macro: float = 0.0
    f1_micro: float = 0.0
    classification_report: Optional[Dict[str, Any]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    class_names: List[str] = field(default_factory=lambda: DEFAULT_CLASS_NAMES.copy())
    num_samples: int = 0
    roc_auc_macro: Optional[float] = None
    roc_auc_per_class: Optional[Dict[str, float]] = None
    pr_auc_per_class: Optional[Dict[str, float]] = None
    extra_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典"""
        d = asdict(self)
        if self.confusion_matrix is not None and isinstance(self.confusion_matrix, np.ndarray):
            d["confusion_matrix"] = self.confusion_matrix.tolist()
        return d

    def to_dataframe(self) -> pd.DataFrame:
        """将核心指标转换为单行列DataFrame（便于汇总对比）"""
        core = {
            "model_name": [self.model_name],
            "accuracy": [self.accuracy],
            "precision_macro": [self.precision_macro],
            "precision_micro": [self.precision_micro],
            "recall_macro": [self.recall_macro],
            "recall_micro": [self.recall_micro],
            "f1_macro": [self.f1_macro],
            "f1_micro": [self.f1_micro],
            "num_samples": [self.num_samples],
        }
        if self.roc_auc_macro is not None:
            core["roc_auc_macro"] = [self.roc_auc_macro]
        for k, v in self.extra_metrics.items():
            core[k] = [v]
        return pd.DataFrame(core)

    def classification_report_df(self) -> Optional[pd.DataFrame]:
        """逐类分类报告转DataFrame"""
        if self.classification_report is None:
            return None
        rows = []
        for cls_name in self.class_names:
            if cls_name in self.classification_report:
                info = self.classification_report[cls_name]
                rows.append({
                    "class": cls_name,
                    "precision": info.get("precision", 0.0),
                    "recall": info.get("recall", 0.0),
                    "f1-score": info.get("f1-score", 0.0),
                    "support": info.get("support", 0),
                })
        for key in ["macro avg", "weighted avg", "micro avg"]:
            if key in self.classification_report:
                info = self.classification_report[key]
                rows.append({
                    "class": key,
                    "precision": info.get("precision", 0.0),
                    "recall": info.get("recall", 0.0),
                    "f1-score": info.get("f1-score", 0.0),
                    "support": info.get("support", 0),
                })
        return pd.DataFrame(rows)

    def save(self, save_dir: Union[str, Path], prefix: str = "") -> Dict[str, str]:
        """
        保存评估结果到目录（JSON + CSV + 可视化）

        Args:
            save_dir: 保存目录
            prefix: 文件名前缀（如模型名）

        Returns:
            Dict: {文件类型: 保存路径}
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        prefix = f"{prefix}_" if prefix else ""
        saved: Dict[str, str] = {}

        json_path = save_path / f"{prefix}evaluation_result.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        saved["json"] = str(json_path)

        metrics_csv = save_path / f"{prefix}metrics_summary.csv"
        self.to_dataframe().to_csv(metrics_csv, index=False, encoding="utf-8-sig")
        saved["metrics_csv"] = str(metrics_csv)

        report_df = self.classification_report_df()
        if report_df is not None:
            report_csv = save_path / f"{prefix}classification_report.csv"
            report_df.to_csv(report_csv, index=False, encoding="utf-8-sig")
            saved["report_csv"] = str(report_csv)

        logger.info(f"评估结果已保存到目录: {save_path}")
        return saved

    def summary(self) -> str:
        """生成人类可读的评估摘要字符串"""
        lines = [
            f"===== 模型评估结果: {self.model_name} =====",
            f"样本数: {self.num_samples}",
            f"- Accuracy       : {self.accuracy:.4f}",
            f"- Precision(Macro): {self.precision_macro:.4f}",
            f"- Precision(Micro): {self.precision_micro:.4f}",
            f"- Recall(Macro)  : {self.recall_macro:.4f}",
            f"- Recall(Micro)  : {self.recall_micro:.4f}",
            f"- F1(Macro) ★    : {self.f1_macro:.4f}",
            f"- F1(Micro)      : {self.f1_micro:.4f}",
        ]
        if self.roc_auc_macro is not None:
            lines.append(f"- ROC-AUC(Macro) : {self.roc_auc_macro:.4f}")
        if self.extra_metrics:
            for k, v in self.extra_metrics.items():
                lines.append(f"- {k:<15s}: {v:.4f}")
        return "\n".join(lines)


# ============================================================
# 可视化工具函数
# ============================================================

def plot_confusion_matrix_heatmap(
    cm: Union[np.ndarray, List[List[int]]],
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "混淆矩阵热力图",
    figsize: Tuple[int, int] = (14, 12),
    normalize: bool = False,
    cmap: str = "YlGnBu",
    show_values: bool = True,
    dpi: int = 150,
) -> Optional[plt.Figure]:
    """
    绘制混淆矩阵热力图（12x12类目）

    Args:
        cm: 混淆矩阵 (num_classes, num_classes)
        class_names: 类目名称，默认使用DEFAULT_CLASS_NAMES
        save_path: 保存为PNG的路径，None则只返回Figure对象
        title: 图表标题
        figsize: 图像尺寸
        normalize: 是否按行归一化（显示召回率比例）
        cmap: 颜色映射
        show_values: 是否在格子中显示数值
        dpi: 图像DPI

    Returns:
        plt.Figure: 保存路径不为None时返回None，否则返回Figure对象
    """
    cm_arr = np.asarray(cm)
    class_names = class_names or DEFAULT_CLASS_NAMES[:cm_arr.shape[0]]

    if len(class_names) != cm_arr.shape[0]:
        warnings.warn(
            f"类目名称数量({len(class_names)})与混淆矩阵维度({cm_arr.shape[0]})不匹配，"
            f"将使用默认名称"
        )
        class_names = [f"类别_{i}" for i in range(cm_arr.shape[0])]

    display_cm = cm_arr.astype(np.float64)
    fmt = "d"
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        display_cm = display_cm / row_sums
        fmt = ".2f"
        title = f"{title}（行归一化）"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        display_cm,
        annot=show_values,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "样本数" if not normalize else "比例"},
        linewidths=0.5,
        linecolor="lightgray",
    )
    ax.set_xlabel("预测类别", fontsize=12)
    ax.set_ylabel("真实类别", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(sp), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"混淆矩阵热力图已保存: {sp}")
        return None
    return fig


def plot_roc_curves(
    y_true: Union[List[int], np.ndarray],
    y_prob: Union[np.ndarray, List[List[float]]],
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "多分类ROC曲线 (One-vs-Rest)",
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 150,
) -> Tuple[Optional[plt.Figure], Dict[str, float]]:
    """
    绘制多分类One-vs-Rest ROC曲线

    Args:
        y_true: 真实标签 (n_samples,)
        y_prob: 预测概率矩阵 (n_samples, num_classes)
        class_names: 类目名称
        save_path: 保存PNG路径
        title: 图表标题
        figsize: 图像尺寸
        dpi: 图像DPI

    Returns:
        (Figure或None, 各类别AUC字典)
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    n_classes = y_prob_arr.shape[1]
    class_names = class_names or DEFAULT_CLASS_NAMES[:n_classes]

    if len(class_names) != n_classes:
        class_names = [f"类别_{i}" for i in range(n_classes)]

    classes = np.arange(n_classes)
    y_true_bin = label_binarize(y_true_arr, classes=classes)
    if y_true_bin.shape[1] == 1 and n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fpr_dict: Dict[str, np.ndarray] = {}
    tpr_dict: Dict[str, np.ndarray] = {}
    roc_auc_dict: Dict[str, float] = {}

    for i, cls_name in enumerate(class_names):
        if n_classes == 2:
            y_true_cls = y_true_arr == i
        else:
            y_true_cls = y_true_bin[:, i]
        fpr, tpr, _ = roc_curve(y_true_cls, y_prob_arr[:, i])
        roc_auc = auc(fpr, tpr)
        fpr_dict[cls_name] = fpr
        tpr_dict[cls_name] = tpr
        roc_auc_dict[cls_name] = float(roc_auc)

    all_fpr = np.unique(np.concatenate([fpr for fpr in fpr_dict.values()]))
    mean_tpr = np.zeros_like(all_fpr)
    for cls_name in class_names:
        mean_tpr += np.interp(all_fpr, fpr_dict[cls_name], tpr_dict[cls_name])
    mean_tpr /= len(class_names)
    macro_auc = auc(all_fpr, mean_tpr)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

    for idx, cls_name in enumerate(class_names):
        ax.plot(
            fpr_dict[cls_name], tpr_dict[cls_name],
            color=colors[idx], lw=1.5, alpha=0.7,
            label=f"{cls_name} (AUC={roc_auc_dict[cls_name]:.3f})",
        )

    ax.plot(
        all_fpr, mean_tpr,
        color="navy", lw=2.5, linestyle="--",
        label=f"Macro-Avg (AUC={macro_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    roc_auc_dict["__macro_avg__"] = float(macro_auc)

    if save_path is not None:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(sp), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"ROC曲线图已保存: {sp}")
        return None, roc_auc_dict
    return fig, roc_auc_dict


# ============================================================
# 核心评估器
# ============================================================

class Evaluator:
    """
    分类模型评估器（sklearn风格API）

    使用示例：
        >>> from src.evaluation import Evaluator
        >>> evaluator = Evaluator(class_names=DEFAULT_CLASS_NAMES)
        >>> result = evaluator.evaluate(model, X_test, y_test)
        >>> print(result.summary())
        >>> evaluator.save_visualizations(result, output_dir="results")
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        compute_roc: bool = True,
        compute_pr: bool = False,
        normalize_cm: bool = False,
    ):
        """
        初始化评估器

        Args:
            class_names: 类目名称列表，默认使用DEFAULT_CLASS_NAMES
            output_dir: 默认输出目录（保存结果/可视化时使用）
            compute_roc: 是否计算并绘制ROC曲线（需predict_proba）
            compute_pr: 是否计算PR-AUC
            normalize_cm: 混淆矩阵热力图是否行归一化
        """
        self.class_names: List[str] = class_names or DEFAULT_CLASS_NAMES.copy()
        self.output_dir: Optional[Path] = Path(output_dir) if output_dir else None
        self.compute_roc: bool = compute_roc
        self.compute_pr: bool = compute_pr
        self.normalize_cm: bool = normalize_cm

        if len(self.class_names) != NUM_CLASSES:
            warnings.warn(
                f"类目名称数量({len(self.class_names)})与默认类目数({NUM_CLASSES})不符，"
                f"请确认一致性"
            )

    # --------------------------------------------------------
    # 核心评估方法
    # --------------------------------------------------------

    def evaluate(
        self,
        model: Any,
        X: Union[List[str], np.ndarray, pd.Series],
        y_true: Union[List[int], np.ndarray, pd.Series],
        model_name: Optional[str] = None,
        y_prob: Optional[np.ndarray] = None,
    ) -> EvaluationResult:
        """
        评估模型在数据集上的性能（sklearn风格）

        Args:
            model: BaseClassifier实例或具有predict/predict_proba方法的模型
            X: 评估特征（文本列表/矩阵等）
            y_true: 真实标签 (n_samples,)
            model_name: 模型名称，默认从model.model_name获取
            y_prob: 预计算的概率矩阵，若为None且compute_roc=True则调用predict_proba

        Returns:
            EvaluationResult: 评估结果对象
        """
        y_true_arr = np.asarray(y_true).astype(int)
        n_samples = len(y_true_arr)

        if model_name is None:
            model_name = getattr(model, "model_name", type(model).__name__)

        logger.info(
            f"[{model_name}] 开始评估 - 样本数={n_samples}, "
            f"类别数={len(self.class_names)}"
        )

        # 预测标签
        y_pred = self._safe_predict(model, X)

        # 核心分类指标
        accuracy = float(accuracy_score(y_true_arr, y_pred))
        precision_macro = float(precision_score(
            y_true_arr, y_pred, average="macro", zero_division=0,
        ))
        precision_micro = float(precision_score(
            y_true_arr, y_pred, average="micro", zero_division=0,
        ))
        recall_macro = float(recall_score(
            y_true_arr, y_pred, average="macro", zero_division=0,
        ))
        recall_micro = float(recall_score(
            y_true_arr, y_pred, average="micro", zero_division=0,
        ))
        f1_macro = float(f1_score(
            y_true_arr, y_pred, average="macro", zero_division=0,
        ))
        f1_micro = float(f1_score(
            y_true_arr, y_pred, average="micro", zero_division=0,
        ))

        # 逐类分类报告
        try:
            cls_report = classification_report(
                y_true_arr, y_pred,
                target_names=self.class_names,
                output_dict=True,
                zero_division=0,
                labels=list(range(len(self.class_names))),
            )
        except Exception as e:
            logger.warning(f"生成分类报告失败，使用labels参数重试: {e}")
            cls_report = classification_report(
                y_true_arr, y_pred,
                target_names=self.class_names,
                output_dict=True,
                zero_division=0,
            )

        # 混淆矩阵
        labels_range = list(range(len(self.class_names)))
        cm = confusion_matrix(y_true_arr, y_pred, labels=labels_range)

        result = EvaluationResult(
            model_name=model_name,
            accuracy=accuracy,
            precision_macro=precision_macro,
            precision_micro=precision_micro,
            recall_macro=recall_macro,
            recall_micro=recall_micro,
            f1_macro=f1_macro,
            f1_micro=f1_micro,
            classification_report=cls_report,
            confusion_matrix=cm.tolist(),
            class_names=self.class_names.copy(),
            num_samples=n_samples,
        )

        # ROC / PR 曲线（需概率）
        if self.compute_roc or self.compute_pr:
            if y_prob is None and hasattr(model, "predict_proba"):
                try:
                    y_prob = self._safe_predict_proba(model, X)
                except Exception as e:
                    logger.warning(f"获取预测概率失败，跳过ROC/PR计算: {e}")

            if y_prob is not None:
                self._compute_roc_pr(result, y_true_arr, y_prob)

        logger.info(
            f"[{model_name}] 评估完成 - "
            f"F1_Macro={f1_macro:.4f}, Accuracy={accuracy:.4f}"
        )
        return result

    # --------------------------------------------------------
    # 便捷方法：直接评估预测结果（无需模型）
    # --------------------------------------------------------

    def evaluate_predictions(
        self,
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
        model_name: str = "predictions",
        y_prob: Optional[np.ndarray] = None,
    ) -> EvaluationResult:
        """
        直接评估预测结果（无需模型对象，sklearn风格）

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            model_name: 名称标识
            y_prob: 预测概率（可选，用于ROC）

        Returns:
            EvaluationResult
        """
        proxy_model = type("ProxyModel", (), {
            "predict": lambda self, X: np.asarray(y_pred),
            "predict_proba": (lambda self, X: y_prob) if y_prob is not None else None,
            "model_name": model_name,
        })()
        X_dummy = np.zeros(len(y_true))
        return self.evaluate(proxy_model, X_dummy, y_true, model_name=model_name, y_prob=y_prob)

    # --------------------------------------------------------
    # 可视化保存
    # --------------------------------------------------------

    def save_visualizations(
        self,
        result: EvaluationResult,
        output_dir: Optional[Union[str, Path]] = None,
        save_cm: bool = True,
        save_roc: bool = True,
        y_true: Optional[Union[List[int], np.ndarray]] = None,
        y_prob: Optional[np.ndarray] = None,
        dpi: int = 150,
    ) -> Dict[str, str]:
        """
        保存评估可视化图表（混淆矩阵+ROC）

        Args:
            result: EvaluationResult对象
            output_dir: 输出目录，默认使用self.output_dir
            save_cm: 是否保存混淆矩阵热力图
            save_roc: 是否保存ROC曲线（需要y_true和y_prob）
            y_true: ROC所需真实标签
            y_prob: ROC所需预测概率
            dpi: 图像DPI

        Returns:
            Dict: {图表类型: 保存路径}
        """
        out_dir = Path(output_dir) if output_dir else self.output_dir
        if out_dir is None:
            raise ValueError("未指定output_dir，且Evaluator无默认输出目录")
        out_dir.mkdir(parents=True, exist_ok=True)

        prefix = result.model_name
        saved: Dict[str, str] = {}

        if save_cm and result.confusion_matrix is not None:
            cm_path = out_dir / f"{prefix}_confusion_matrix.png"
            plot_confusion_matrix_heatmap(
                result.confusion_matrix,
                class_names=result.class_names,
                save_path=str(cm_path),
                normalize=self.normalize_cm,
                dpi=dpi,
                title=f"{result.model_name} - 混淆矩阵热力图",
            )
            saved["confusion_matrix"] = str(cm_path)

            cm_norm_path = out_dir / f"{prefix}_confusion_matrix_normalized.png"
            plot_confusion_matrix_heatmap(
                result.confusion_matrix,
                class_names=result.class_names,
                save_path=str(cm_norm_path),
                normalize=True,
                dpi=dpi,
                title=f"{result.model_name} - 混淆矩阵（行归一化）",
            )
            saved["confusion_matrix_norm"] = str(cm_norm_path)

        if save_roc and y_true is not None and y_prob is not None:
            roc_path = out_dir / f"{prefix}_roc_curves.png"
            plot_roc_curves(
                y_true, y_prob,
                class_names=result.class_names,
                save_path=str(roc_path),
                dpi=dpi,
                title=f"{result.model_name} - ROC曲线 (One-vs-Rest)",
            )
            saved["roc_curves"] = str(roc_path)

        return saved

    # --------------------------------------------------------
    # 多模型对比汇总
    # --------------------------------------------------------

    def compare_models(
        self,
        results: List[EvaluationResult],
        output_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        多模型评估结果横向对比（输出汇总CSV）

        Args:
            results: 多个EvaluationResult
            output_path: 可选，保存汇总CSV路径

        Returns:
            pd.DataFrame: 对比表（每行一个模型）
        """
        if not results:
            return pd.DataFrame()

        frames = [r.to_dataframe() for r in results]
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("f1_macro", ascending=False).reset_index(drop=True)

        if output_path is not None:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            logger.info(f"模型对比汇总已保存: {out_path}")

        return df

    # --------------------------------------------------------
    # 内部辅助
    # --------------------------------------------------------

    @staticmethod
    def _safe_predict(model: Any, X: Any) -> np.ndarray:
        """安全调用predict，兼容BaseClassifier和sklearn风格"""
        preds = model.predict(X)
        return np.asarray(preds).astype(int)

    @staticmethod
    def _safe_predict_proba(model: Any, X: Any) -> np.ndarray:
        """安全调用predict_proba"""
        probas = model.predict_proba(X)
        arr = np.asarray(probas)
        if arr.ndim == 1:
            arr = np.column_stack([1 - arr, arr])
        return arr.astype(np.float64)

    def _compute_roc_pr(
        self,
        result: EvaluationResult,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> None:
        """在result中填充ROC/PR相关指标（原地修改）"""
        y_prob_arr = np.asarray(y_prob)
        n_classes = len(self.class_names)

        try:
            _, roc_auc_per_class = plot_roc_curves(
                y_true, y_prob_arr,
                class_names=self.class_names,
                save_path=None,
            )
            result.roc_auc_macro = roc_auc_per_class.get("__macro_avg__")
            result.roc_auc_per_class = {
                k: v for k, v in roc_auc_per_class.items() if not k.startswith("__")
            }
        except Exception as e:
            logger.warning(f"计算ROC-AUC失败: {e}")

        if self.compute_pr:
            try:
                pr_auc = {}
                y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
                if y_true_bin.shape[1] == 1 and n_classes == 2:
                    y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
                for i, cls_name in enumerate(self.class_names):
                    yt = y_true_bin[:, i] if n_classes > 2 else (y_true == i).astype(int)
                    precision_curve, recall_curve, _ = precision_recall_curve(
                        yt, y_prob_arr[:, i],
                    )
                    pr_auc[cls_name] = float(auc(recall_curve, precision_curve))
                result.pr_auc_per_class = pr_auc
            except Exception as e:
                logger.warning(f"计算PR-AUC失败: {e}")
