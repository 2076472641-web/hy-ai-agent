# -*- coding: utf-8 -*-
"""
模型基准测试模块
提供全面的模型性能基准测试能力，输出标准化对比报告CSV：

测试维度：
1. 批量推理吞吐量（Throughput）：不同batch大小下的每秒处理样本数
2. 单样本延迟（Latency）：P50/P90/P99延迟统计
3. 内存占用（Memory Footprint）：峰值内存使用、平均内存使用
4. CPU/GPU利用率（Utilization）：推理过程中的资源占用率
5. 不同输入长度性能：不同max_length下的性能表现

支持模型类型：
- BaseClassifier子类（RandomForest / FastText / BERT等）
- PyTorch nn.Module（如量化后的模型）
- ONNX模型（onnxruntime推理）
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
import json
import time
import platform
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
from loguru import logger

# ============================================================
# 可选依赖导入（惰性加载，缺失时给出警告而非报错）
# ============================================================

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from ..models.base import BaseClassifier
from ..models.bert_classifier import BERTClassifier, TextDataset
from .evaluator import DEFAULT_CLASS_NAMES


# ============================================================
# 配置与结果数据类
# ============================================================

@dataclass
class BenchmarkConfig:
    """
    基准测试配置

    Attributes:
        batch_sizes: 测试的batch大小列表
        num_warmup_runs: 预热轮数（避免JIT/缓存影响）
        num_test_runs: 每轮测试重复次数（统计稳定性）
        max_lengths: BERT类模型测试的序列长度列表
        measure_cpu: 是否测量CPU使用率（需psutil）
        measure_gpu: 是否测量GPU使用率（需pynvml/torch.cuda）
        measure_memory: 是否测量内存占用
        latency_percentiles: 延迟统计的分位点（如P50/P90/P99）
        device: 推理设备（cpu/cuda），None为自动
        output_dir: 输出报告目录
    """
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32, 64])
    num_warmup_runs: int = 10
    num_test_runs: int = 50
    max_lengths: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    measure_cpu: bool = True
    measure_gpu: bool = True
    measure_memory: bool = True
    latency_percentiles: List[float] = field(default_factory=lambda: [50, 90, 95, 99])
    device: Optional[str] = None
    output_dir: Optional[str] = None
    csv_encoding: str = "utf-8-sig"


@dataclass
class BenchmarkResult:
    """
    单模型基准测试结果

    Attributes:
        model_name: 模型名称
        model_type: 模型类型（BaseClassifier / TorchModel / ONNX）
        device: 推理设备
        env_info: 运行环境信息（系统、CPU、内存、Python版本等）
        batch_results: 各batch大小的测试结果
        length_results: 各序列长度的测试结果（BERT类）
        summary_metrics: 汇总指标（最佳吞吐、平均延迟等）
        resource_usage: 资源占用统计（CPU/GPU/内存峰值）
    """
    model_name: str = ""
    model_type: str = ""
    device: str = "cpu"
    env_info: Dict[str, Any] = field(default_factory=dict)
    batch_results: List[Dict[str, Any]] = field(default_factory=list)
    length_results: List[Dict[str, Any]] = field(default_factory=list)
    summary_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_dataframe(self) -> pd.DataFrame:
        """汇总指标转DataFrame"""
        row = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            **self.summary_metrics,
            **self.resource_usage,
        }
        return pd.DataFrame([row])

    def batch_dataframe(self) -> pd.DataFrame:
        """Batch维度测试结果DataFrame"""
        if not self.batch_results:
            return pd.DataFrame()
        df = pd.DataFrame(self.batch_results)
        df.insert(0, "model_name", self.model_name)
        return df

    def length_dataframe(self) -> pd.DataFrame:
        """序列长度维度测试结果DataFrame"""
        if not self.length_results:
            return pd.DataFrame()
        df = pd.DataFrame(self.length_results)
        df.insert(0, "model_name", self.model_name)
        return df

    def save(self, output_dir: Union[str, Path], prefix: str = "") -> Dict[str, str]:
        """
        保存基准测试结果

        Args:
            output_dir: 输出目录
            prefix: 文件名前缀

        Returns:
            Dict: {文件类型: 保存路径}
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{prefix}_" if prefix else ""
        saved: Dict[str, str] = {}

        json_path = out_dir / f"{prefix}benchmark_result.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        saved["json"] = str(json_path)

        summary_df = self.summary_dataframe()
        if not summary_df.empty:
            summary_csv = out_dir / f"{prefix}benchmark_summary.csv"
            summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
            saved["summary_csv"] = str(summary_csv)

        batch_df = self.batch_dataframe()
        if not batch_df.empty:
            batch_csv = out_dir / f"{prefix}benchmark_batch.csv"
            batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")
            saved["batch_csv"] = str(batch_csv)

        length_df = self.length_dataframe()
        if not length_df.empty:
            length_csv = out_dir / f"{prefix}benchmark_length.csv"
            length_df.to_csv(length_csv, index=False, encoding="utf-8-sig")
            saved["length_csv"] = str(length_csv)

        logger.info(f"基准测试结果已保存到: {out_dir}")
        return saved

    def summary(self) -> str:
        """人类可读摘要"""
        lines = [
            f"===== 基准测试结果: {self.model_name} =====",
            f"模型类型: {self.model_type}, 设备: {self.device}",
            "",
            f"[环境]",
        ]
        for k, v in self.env_info.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append(f"[汇总指标]")
        for k, v in self.summary_metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append(f"[资源占用]")
        for k, v in self.resource_usage.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.2f}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ============================================================
# 资源监控工具
# ============================================================

class ResourceMonitor:
    """系统资源监控（CPU / 内存 / GPU）"""

    def __init__(self, measure_cpu: bool = True, measure_gpu: bool = True):
        self.measure_cpu = measure_cpu and PSUTIL_AVAILABLE
        self.measure_gpu = measure_gpu and TORCH_AVAILABLE

        self._nvml_handle = None
        if self.measure_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._pynvml = pynvml
            except Exception:
                self._nvml_handle = None

        if not PSUTIL_AVAILABLE and measure_cpu:
            logger.warning("psutil未安装，CPU/内存监控将不可用")
        if not TORCH_AVAILABLE and measure_gpu:
            logger.warning("torch未安装，GPU监控将不可用")

    def snapshot(self) -> Dict[str, float]:
        """采样一次当前资源状态"""
        snap: Dict[str, float] = {}
        try:
            if self.measure_cpu:
                snap["cpu_percent"] = float(psutil.cpu_percent(interval=None))
                mem = psutil.virtual_memory()
                snap["memory_used_gb"] = float(mem.used / (1024 ** 3))
                snap["memory_percent"] = float(mem.percent)
                process = psutil.Process()
                snap["process_rss_mb"] = float(process.memory_info().rss / (1024 * 1024))

            if self.measure_gpu:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    snap["gpu_memory_allocated_mb"] = float(
                        torch.cuda.memory_allocated() / (1024 * 1024)
                    )
                    snap["gpu_memory_reserved_mb"] = float(
                        torch.cuda.memory_reserved() / (1024 * 1024)
                    )
                if self._nvml_handle is not None:
                    try:
                        util = self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                        snap["gpu_util_percent"] = float(util.gpu)
                        snap["gpu_memory_util_percent"] = float(util.memory)
                        mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                        snap["gpu_memory_total_gb"] = float(mem_info.total / (1024 ** 3))
                        snap["gpu_memory_used_gb"] = float(mem_info.used / (1024 ** 3))
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"资源采样异常（可忽略）: {e}")
        return snap

    @staticmethod
    def aggregate(snapshots: List[Dict[str, float]]) -> Dict[str, float]:
        """聚合一列采样：均值/峰值"""
        if not snapshots:
            return {}
        keys = set(k for s in snapshots for k in s.keys())
        agg: Dict[str, float] = {}
        for key in keys:
            vals = [s.get(key, 0.0) for s in snapshots if key in s]
            if vals:
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_max"] = float(np.max(vals))
                agg[f"{key}_min"] = float(np.min(vals))
        return agg


def get_environment_info() -> Dict[str, Any]:
    """获取运行环境信息"""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor() or "unknown",
    }
    if PSUTIL_AVAILABLE:
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        mem = psutil.virtual_memory()
        info["total_memory_gb"] = round(mem.total / (1024 ** 3), 2)
    if TORCH_AVAILABLE:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name_0"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
    if ONNX_AVAILABLE:
        info["onnxruntime_version"] = ort.__version__
    return info


# ============================================================
# 基准测试器
# ============================================================

class Benchmarker:
    """
    模型基准测试器

    使用示例：
        >>> benchmarker = Benchmarker()
        >>> result = benchmarker.benchmark(model, sample_texts, name="my_model")
        >>> print(result.summary())
        >>> benchmarker.save_report([result1, result2], "output/benchmark.csv")
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Args:
            config: 基准测试配置，默认使用BenchmarkConfig默认值
        """
        self.config: BenchmarkConfig = config or BenchmarkConfig()
        self.env_info: Dict[str, Any] = get_environment_info()

        if self.config.device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available() and self.config.measure_gpu:
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device

        self.monitor = ResourceMonitor(
            measure_cpu=self.config.measure_cpu,
            measure_gpu=self.config.measure_gpu,
        )
        logger.info(
            f"[Benchmarker] 初始化完成 - 设备={self.device}, "
            f"Batch={self.config.batch_sizes}, Runs={self.config.num_test_runs}"
        )

    # ========================================================
    # 单模型完整基准测试入口
    # ========================================================

    def benchmark(
        self,
        model: Any,
        sample_texts: List[str],
        name: Optional[str] = None,
        is_onnx: bool = False,
    ) -> BenchmarkResult:
        """
        对单个模型执行完整基准测试

        Args:
            model: 待测试模型（BaseClassifier / nn.Module / ONNX路径）
            sample_texts: 样本文本列表（用于构造推理输入）
            name: 模型名称（默认从model属性推断）
            is_onnx: 是否为ONNX模型（当model是str路径时自动识别）

        Returns:
            BenchmarkResult: 完整测试结果
        """
        if not sample_texts:
            raise ValueError("sample_texts不能为空")

        model_type, model_name, infer_fn, aux = self._prepare_model(
            model, sample_texts, is_onnx,
        )
        if name is not None:
            model_name = name

        logger.info(
            f"[Benchmarker] 开始基准测试 - 模型={model_name}, "
            f"类型={model_type}, 样本量={len(sample_texts)}"
        )

        result = BenchmarkResult(
            model_name=model_name,
            model_type=model_type,
            device=self.device,
            env_info=self.env_info,
        )

        resource_snapshots: List[Dict[str, float]] = []

        # 1. 不同Batch大小的性能测试
        for bs in self.config.batch_sizes:
            if bs > len(sample_texts) * 4:
                sample_batch = (sample_texts * ((bs // len(sample_texts)) + 1))[:bs]
            else:
                sample_batch = sample_texts[:bs] if len(sample_texts) >= bs else sample_texts * (bs // len(sample_texts) + 1)
                sample_batch = sample_batch[:bs]

            batch_metrics, snaps = self._benchmark_batch(
                infer_fn, sample_batch, bs, aux,
            )
            result.batch_results.append({"batch_size": bs, **batch_metrics})
            resource_snapshots.extend(snaps)
            logger.info(
                f"  Batch[{bs:3d}] 吞吐={batch_metrics.get('throughput_samples_sec', 0):.1f}样本/s, "
                f"平均延迟={batch_metrics.get('latency_avg_ms', 0):.2f}ms/样本"
            )

        # 2. 不同序列长度测试（仅支持文本tokenizer的模型）
        if aux.get("type") in ("bert_classifier", "torch_bert") and aux.get("tokenizer") is not None:
            tokenizer = aux["tokenizer"]
            base_bs = 8
            base_texts = sample_texts[:base_bs] if len(sample_texts) >= base_bs else sample_texts * (base_bs // len(sample_texts) + 1)
            base_texts = base_texts[:base_bs]
            for ml in self.config.max_lengths:
                length_metrics, snaps = self._benchmark_seq_length(
                    infer_fn, base_texts, ml, tokenizer, aux,
                )
                result.length_results.append({"max_length": ml, **length_metrics})
                resource_snapshots.extend(snaps)
                logger.info(
                    f"  SeqLen[{ml:4d}] 吞吐={length_metrics.get('throughput_samples_sec', 0):.1f}样本/s, "
                    f"平均延迟={length_metrics.get('latency_avg_ms', 0):.2f}ms/样本"
                )

        # 3. 汇总指标
        result.summary_metrics = self._compute_summary(result.batch_results)
        result.resource_usage = ResourceMonitor.aggregate(resource_snapshots)

        # 4. 保存
        if self.config.output_dir:
            result.save(self.config.output_dir, prefix=model_name)

        logger.info(f"[Benchmarker] {model_name} 基准测试完成")
        return result

    # ========================================================
    # 内部实现
    # ========================================================

    def _prepare_model(
        self, model: Any, sample_texts: List[str], is_onnx: bool,
    ) -> Tuple[str, str, Callable, Dict[str, Any]]:
        """
        标准化模型：返回 (类型名, 模型名, 推理函数, 辅助信息字典)

        推理函数签名：infer_fn(texts: List[str], **kwargs) -> Any
        """
        aux: Dict[str, Any] = {}

        # ONNX模型
        if isinstance(model, (str, Path)) or is_onnx:
            model_path = str(model)
            if not ONNX_AVAILABLE:
                raise ImportError("onnxruntime未安装，无法测试ONNX模型")
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = []
            if self.device == "cuda":
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            session = ort.InferenceSession(model_path, sess_options, providers=providers)
            input_names = [inp.name for inp in session.get_inputs()]
            model_name = Path(model_path).stem
            aux.update({
                "type": "onnx",
                "session": session,
                "input_names": input_names,
                "onnx_max_length": 128,
            })
            return (
                "ONNX",
                model_name,
                self._onnx_inference_fn,
                aux,
            )

        # BaseClassifier子类（BERTClassifier / RandomForest / FastText 等）
        if isinstance(model, BaseClassifier):
            model_name = getattr(model, "model_name", type(model).__name__)
            aux["type"] = "base_classifier"

            if isinstance(model, BERTClassifier):
                aux["type"] = "bert_classifier"
                if model._tokenizer is not None:
                    aux["tokenizer"] = model._tokenizer
                    aux["max_length"] = model.max_length
                return (
                    "BERTClassifier",
                    model_name,
                    self._baseclassifier_inference_fn,
                    {"model": model, **aux},
                )

            return (
                type(model).__name__,
                model_name,
                self._baseclassifier_inference_fn,
                {"model": model, **aux},
            )

        # PyTorch nn.Module
        if TORCH_AVAILABLE and isinstance(model, nn.Module):
            model_name = type(model).__name__
            aux.update({
                "type": "torch_bert",
                "torch_device": self.device,
                "tokenizer": aux.get("tokenizer"),
                "max_length": aux.get("max_length", 128),
            })
            return (
                "TorchModel",
                model_name,
                self._torch_module_inference_fn,
                {"model": model, **aux},
            )

        model_name = type(model).__name__
        if hasattr(model, "predict"):
            return (
                "GenericWithPredict",
                model_name,
                self._generic_predict_fn,
                {"model": model, **aux},
            )

        raise TypeError(f"不支持的模型类型: {type(model)}")

    # --------------------------------------------------------
    # 各类模型的推理函数
    # --------------------------------------------------------

    @staticmethod
    def _baseclassifier_inference_fn(
        texts: List[str], aux: Dict[str, Any], **kwargs,
    ) -> Any:
        model: BaseClassifier = aux["model"]
        return model.predict(texts)

    @staticmethod
    def _torch_module_inference_fn(
        texts: List[str], aux: Dict[str, Any], **kwargs,
    ) -> Any:
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch不可用")
        model: nn.Module = aux["model"]
        tokenizer = aux.get("tokenizer")
        device = aux.get("torch_device", "cpu")
        max_length = aux.get("max_length", 128)

        if tokenizer is None:
            raise ValueError("Torch模型缺少tokenizer，无法编码文本")

        encoded = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        model.eval()
        with torch.no_grad():
            inputs = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
            if "token_type_ids" in encoded and not isinstance(
                model,
                type.__class__,
            ):
                try:
                    from transformers import DistilBertForSequenceClassification
                    if not isinstance(model, DistilBertForSequenceClassification):
                        inputs["token_type_ids"] = encoded["token_type_ids"]
                except Exception:
                    pass
            outputs = model(**inputs)
        return outputs.logits.cpu().numpy()

    @staticmethod
    def _onnx_inference_fn(
        texts: List[str], aux: Dict[str, Any], **kwargs,
    ) -> Any:
        session: "ort.InferenceSession" = aux["session"]
        input_names = aux["input_names"]
        max_length = aux.get("onnx_max_length", 128)

        tokenizer = kwargs.get("tokenizer")
        if tokenizer is None:
            try:
                tokenizer = aux["tokenizer"]
            except KeyError:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
                aux["tokenizer"] = tokenizer

        encoded = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        feed = {}
        for name in input_names:
            if name in encoded:
                feed[name] = encoded[name].astype(np.int64)
        outputs = session.run(None, feed)
        return outputs[0]

    @staticmethod
    def _generic_predict_fn(
        texts: List[str], aux: Dict[str, Any], **kwargs,
    ) -> Any:
        model = aux["model"]
        return model.predict(texts)

    # --------------------------------------------------------
    # 测试执行
    # --------------------------------------------------------

    def _benchmark_batch(
        self,
        infer_fn: Callable,
        texts: List[str],
        batch_size: int,
        aux: Dict[str, Any],
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """测试指定batch大小的性能"""
        tokens = {}
        if aux.get("tokenizer") is not None and aux.get("type") in (
            "bert_classifier", "torch_bert", "onnx",
        ):
            tokens = {"tokenizer": aux["tokenizer"]}

        snapshots: List[Dict[str, float]] = []
        for _ in range(self.config.num_warmup_runs):
            try:
                infer_fn(texts, aux, **tokens)
            except Exception:
                pass

        run_times: List[float] = []
        for i in range(self.config.num_test_runs):
            if self.config.measure_memory or self.config.measure_cpu:
                if i % 3 == 0:
                    snapshots.append(self.monitor.snapshot())
            t0 = time.perf_counter()
            infer_fn(texts, aux, **tokens)
            elapsed = time.perf_counter() - t0
            run_times.append(elapsed)

        if snapshots and (self.config.measure_memory or self.config.measure_cpu):
            snapshots.append(self.monitor.snapshot())

        run_times_arr = np.asarray(run_times)
        total_samples = batch_size
        per_sample = run_times_arr / total_samples * 1000

        metrics: Dict[str, float] = {
            "total_time_avg_ms": float(run_times_arr.mean() * 1000),
            "total_time_std_ms": float(run_times_arr.std() * 1000),
            "latency_avg_ms": float(per_sample.mean()),
            "latency_std_ms": float(per_sample.std()),
            "throughput_samples_sec": float(total_samples / run_times_arr.mean()),
        }

        for p in self.config.latency_percentiles:
            metrics[f"latency_p{int(p)}_ms"] = float(np.percentile(per_sample, p))

        return metrics, snapshots

    def _benchmark_seq_length(
        self,
        infer_fn: Callable,
        texts: List[str],
        max_length: int,
        tokenizer: Any,
        aux: Dict[str, Any],
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """测试不同序列长度的性能"""
        aux_override = dict(aux)
        aux_override["max_length"] = max_length
        if "onnx_max_length" in aux_override:
            aux_override["onnx_max_length"] = max_length

        bs = len(texts)
        for _ in range(self.config.num_warmup_runs // 2):
            try:
                infer_fn(texts, aux_override, tokenizer=tokenizer)
            except Exception:
                pass

        snapshots: List[Dict[str, float]] = []
        run_times: List[float] = []
        for i in range(max(self.config.num_test_runs // 2, 5)):
            if i % 3 == 0:
                snapshots.append(self.monitor.snapshot())
            t0 = time.perf_counter()
            infer_fn(texts, aux_override, tokenizer=tokenizer)
            elapsed = time.perf_counter() - t0
            run_times.append(elapsed)

        run_times_arr = np.asarray(run_times)
        per_sample = run_times_arr / bs * 1000

        metrics: Dict[str, float] = {
            "batch_size": bs,
            "total_time_avg_ms": float(run_times_arr.mean() * 1000),
            "latency_avg_ms": float(per_sample.mean()),
            "throughput_samples_sec": float(bs / run_times_arr.mean()),
        }
        for p in self.config.latency_percentiles:
            metrics[f"latency_p{int(p)}_ms"] = float(np.percentile(per_sample, p))

        return metrics, snapshots

    @staticmethod
    def _compute_summary(batch_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """从batch结果计算汇总指标"""
        if not batch_results:
            return {}
        throughputs = [
            r.get("throughput_samples_sec", 0.0) for r in batch_results
        ]
        latencies_avg = [
            r.get("latency_avg_ms", float("inf")) for r in batch_results
        ]
        best_bs_idx = int(np.argmax(throughputs))
        summary: Dict[str, float] = {
            "best_batch_size": float(batch_results[best_bs_idx]["batch_size"]),
            "max_throughput_samples_sec": float(np.max(throughputs)),
            "avg_throughput_samples_sec": float(np.mean(throughputs)),
            "min_latency_avg_ms": float(np.min(latencies_avg)),
            "avg_latency_avg_ms": float(np.mean(latencies_avg)),
            "batch1_latency_avg_ms": 0.0,
        }
        for r in batch_results:
            if r["batch_size"] == 1:
                summary["batch1_latency_avg_ms"] = float(r.get("latency_avg_ms", 0.0))
                for p in (50, 90, 99):
                    k = f"latency_p{p}_ms"
                    if k in r:
                        summary[f"batch1_{k}"] = float(r[k])
                break
        return summary

    # ========================================================
    # 批量对比与报告导出
    # ========================================================

    def compare(
        self,
        results: List[BenchmarkResult],
        output_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        多模型基准测试对比报告（CSV输出）

        Args:
            results: 多个BenchmarkResult
            output_path: 可选，保存对比CSV路径

        Returns:
            pd.DataFrame: 对比汇总表（每行一个模型）
        """
        if not results:
            return pd.DataFrame()

        summary_frames = [r.summary_dataframe() for r in results]
        summary_df = pd.concat(summary_frames, ignore_index=True)

        batch_frames = [r.batch_dataframe() for r in results if not r.batch_dataframe().empty]
        length_frames = [r.length_dataframe() for r in results if not r.length_dataframe().empty]

        if output_path is not None:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(out_path, index=False, encoding=self.config.csv_encoding)
            logger.info(f"基准对比汇总已保存: {out_path}")

            base_name = out_path.stem
            parent = out_path.parent
            if batch_frames:
                pd.concat(batch_frames, ignore_index=True).to_csv(
                    parent / f"{base_name}_batch_details.csv",
                    index=False, encoding=self.config.csv_encoding,
                )
            if length_frames:
                pd.concat(length_frames, ignore_index=True).to_csv(
                    parent / f"{base_name}_length_details.csv",
                    index=False, encoding=self.config.csv_encoding,
                )
        return summary_df

    def save_report(
        self,
        results: List[BenchmarkResult],
        output_dir: Union[str, Path],
    ) -> Dict[str, str]:
        """
        保存完整基准测试报告（summary + batch + length + JSON）

        Args:
            results: 多模型结果
            output_dir: 输出目录

        Returns:
            Dict: {报告类型: 保存路径}
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: Dict[str, str] = {}

        compare_csv = out_dir / "benchmark_comparison.csv"
        self.compare(results, compare_csv)
        saved["comparison_csv"] = str(compare_csv)

        env_path = out_dir / "environment_info.json"
        with open(env_path, "w", encoding="utf-8") as f:
            json.dump(self.env_info, f, ensure_ascii=False, indent=2)
        saved["environment"] = str(env_path)

        for r in results:
            sub_saved = r.save(out_dir, prefix=r.model_name)
            saved.update({f"{r.model_name}_{k}": v for k, v in sub_saved.items()})

        return saved


# ============================================================
# 便捷函数
# ============================================================

def run_benchmark_suite(
    models: List[Tuple[Any, str]],
    sample_texts: List[str],
    config: Optional[BenchmarkConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[List[BenchmarkResult], pd.DataFrame]:
    """
    便捷函数：批量跑多模型基准测试套件

    Args:
        models: [(模型实例, 模型名称), ...]
        sample_texts: 样本文本列表
        config: 基准测试配置
        output_dir: 可选输出目录

    Returns:
        (结果列表, 对比汇总DataFrame)
    """
    if config is None:
        config = BenchmarkConfig()
    if output_dir is not None:
        config.output_dir = str(output_dir)

    benchmarker = Benchmarker(config)
    results: List[BenchmarkResult] = []
    for idx, (model, name) in enumerate(models, 1):
        logger.info(f"基准测试套件 [{idx}/{len(models)}]: {name}")
        try:
            is_onnx = isinstance(model, (str, Path))
            res = benchmarker.benchmark(model, sample_texts, name=name, is_onnx=is_onnx)
            results.append(res)
        except Exception as e:
            logger.exception(f"模型 {name} 基准测试失败: {e}")

    compare_df = benchmarker.compare(results)
    if output_dir is not None:
        benchmarker.save_report(results, output_dir)
    return results, compare_df
