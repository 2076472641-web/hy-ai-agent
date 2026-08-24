# -*- coding: utf-8 -*-
"""
模型评估与压缩模块
提供模型性能评估、量化压缩、知识蒸馏、ONNX导出、基准测试等功能。

子模块：
- evaluator: 模型评估器，计算分类指标、混淆矩阵、分类报告、ROC曲线
- model_compressor: 模型压缩器，INT8量化、知识蒸馏、ONNX格式导出
- benchmark: 基准测试，吞吐量、延迟、内存占用、CPU/GPU利用率
"""

from .evaluator import (
    Evaluator,
    EvaluationResult,
    DEFAULT_CLASS_NAMES,
    plot_confusion_matrix_heatmap,
    plot_roc_curves,
)
from .model_compressor import (
    ModelCompressor,
    QuantizationConfig,
    DistillationConfig,
    DistillationTrainer,
    QuantizationType,
    CompressionReport,
)
from .benchmark import (
    Benchmarker,
    BenchmarkConfig,
    BenchmarkResult,
    run_benchmark_suite,
)

__all__ = [
    "Evaluator",
    "EvaluationResult",
    "DEFAULT_CLASS_NAMES",
    "plot_confusion_matrix_heatmap",
    "plot_roc_curves",
    "ModelCompressor",
    "QuantizationConfig",
    "DistillationConfig",
    "DistillationTrainer",
    "QuantizationType",
    "CompressionReport",
    "Benchmarker",
    "BenchmarkConfig",
    "BenchmarkResult",
    "run_benchmark_suite",
]
