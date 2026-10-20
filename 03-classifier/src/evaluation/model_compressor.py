# -*- coding: utf-8 -*-
"""
模型压缩模块
提供三种主流的模型压缩与加速方案：

1. INT8量化（torch.quantization）
   - 动态量化：仅量化权重（Linear/Embedding），无需校准数据，推理时动态量化激活
   - 静态量化：权重+激活均量化，需校准数据集，精度更高
   - 量化后可导出ONNX格式，支持batch推理

2. 知识蒸馏（Knowledge Distillation）
   - DistillationTrainer：教师模型(bert-base-chinese)→学生模型(distilbert-base-chinese)
   - 损失函数：L = alpha * KD_Loss(T) + (1-alpha) * CE_Loss(y_true)
   - 温度系数T=4，alpha=0.5（默认值，可自定义）

3. 轻量化对比报告
   - 压缩率：原模型 vs 压缩模型的参数数量与磁盘体积
   - 推理速度对比：单样本延迟、批量吞吐量
   - 精度损失：各指标相对原始模型的下降幅度
"""

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import time
import copy
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from loguru import logger

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        DistilBertForSequenceClassification,
        BertForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
except ImportError:
    raise ImportError("transformers未安装，请先安装 transformers>=4.30.0")

from ..models.base import BaseClassifier
from ..models.bert_classifier import BERTClassifier, TextDataset
from .evaluator import Evaluator, EvaluationResult, DEFAULT_CLASS_NAMES


# ============================================================
# 枚举与配置数据类
# ============================================================

class QuantizationType(str, Enum):
    """量化类型枚举"""
    DYNAMIC = "dynamic"       # 动态量化：仅权重量化，推理时激活动态量化
    STATIC = "static"         # 静态量化：权重+激活均量化，需校准数据
    QAT = "qat"               # 量化感知训练（Quantization Aware Training）


@dataclass
class QuantizationConfig:
    """
    量化配置

    Attributes:
        quant_type: 量化类型（动态/静态/QAT）
        dtype: 目标量化数据类型，默认qint8
        calibration_batches: 静态量化校准批次数
        calibration_batch_size: 校准batch大小
        qconfig_backend: 量化后端（'fbgemm' for CPU / 'qnnpack' for mobile）
        onnx_opset_version: 导出ONNX的opset版本
        export_onnx: 量化后是否导出ONNX
        onnx_dynamic_batch: ONNX是否支持动态batch维度
    """
    quant_type: QuantizationType = QuantizationType.DYNAMIC
    dtype: torch.dtype = field(default_factory=lambda: torch.qint8)
    calibration_batches: int = 10
    calibration_batch_size: int = 16
    qconfig_backend: str = "fbgemm"
    onnx_opset_version: int = 14
    export_onnx: bool = True
    onnx_dynamic_batch: bool = True


@dataclass
class DistillationConfig:
    """
    知识蒸馏配置

    Attributes:
        teacher_model: 教师模型名称或已训练的BERTClassifier实例
        student_model: 学生模型名称，默认 distilbert-base-chinese
        temperature: 蒸馏温度系数T（默认4），软标签平滑程度
        alpha: 蒸馏损失权重（默认0.5），总损失 = alpha*KD + (1-alpha)*CE
        num_train_epochs: 蒸馏训练轮数
        batch_size: 训练batch大小
        learning_rate: 学习率
        warmup_ratio: warmup步数比例
        weight_decay: 权重衰减
        max_length: 最大序列长度
        device: 训练设备
        fp16: 是否混合精度
        gradient_accumulation_steps: 梯度累积步数
        early_stopping_patience: 早停耐心值
    """
    teacher_model: Optional[Union[str, BERTClassifier]] = None
    student_model: str = "distilbert-base-chinese"
    temperature: float = 4.0
    alpha: float = 0.5
    num_train_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 3.0e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_length: int = 128
    device: Optional[str] = None
    fp16: bool = False
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 3


@dataclass
class CompressionReport:
    """
    轻量化对比报告

    Attributes:
        original_model_name: 原始模型名称
        compressed_model_name: 压缩后模型名称
        method: 压缩方法（quantization / distillation / onnx）
        original_params_million: 原始参数量（百万）
        compressed_params_million: 压缩后参数量（百万）
        param_compression_ratio: 参数压缩率（>1表示压缩）
        original_size_mb: 原始模型大小（MB）
        compressed_size_mb: 压缩后模型大小（MB）
        size_compression_ratio: 体积压缩率
        original_latency_ms: 原始模型单样本推理延迟（ms）
        compressed_latency_ms: 压缩后单样本延迟（ms）
        latency_speedup: 延迟加速比（>1表示更快）
        original_throughput: 原始模型吞吐量（样本/秒）
        compressed_throughput: 压缩后吞吐量（样本/秒）
        throughput_speedup: 吞吐量加速比
        original_metrics: 原始模型评估指标
        compressed_metrics: 压缩后模型评估指标
        accuracy_drop: 准确率下降（绝对值）
        f1_macro_drop: F1_macro下降（绝对值）
        export_path: 压缩后模型导出路径
    """
    original_model_name: str = ""
    compressed_model_name: str = ""
    method: str = ""
    original_params_million: float = 0.0
    compressed_params_million: float = 0.0
    param_compression_ratio: float = 0.0
    original_size_mb: float = 0.0
    compressed_size_mb: float = 0.0
    size_compression_ratio: float = 0.0
    original_latency_ms: float = 0.0
    compressed_latency_ms: float = 0.0
    latency_speedup: float = 0.0
    original_throughput: float = 0.0
    compressed_throughput: float = 0.0
    throughput_speedup: float = 0.0
    original_metrics: Dict[str, float] = field(default_factory=dict)
    compressed_metrics: Dict[str, float] = field(default_factory=dict)
    accuracy_drop: float = 0.0
    f1_macro_drop: float = 0.0
    export_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_dataframe(self) -> pd.DataFrame:
        d = self.to_dict()
        flat = {}
        for k, v in d.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}_{kk}"] = vv
            else:
                flat[k] = v
        return pd.DataFrame([flat])

    def summary(self) -> str:
        lines = [
            f"===== 压缩对比报告: {self.original_model_name} → {self.compressed_model_name} =====",
            f"压缩方法: {self.method}",
            "",
            f"[参数量]",
            f"  原始: {self.original_params_million:.2f}M",
            f"  压缩: {self.compressed_params_million:.2f}M",
            f"  压缩率: {self.param_compression_ratio:.2f}x",
            "",
            f"[模型体积]",
            f"  原始: {self.original_size_mb:.2f} MB",
            f"  压缩: {self.compressed_size_mb:.2f} MB",
            f"  压缩率: {self.size_compression_ratio:.2f}x",
            "",
            f"[推理速度]",
            f"  单样本延迟 - 原始: {self.original_latency_ms:.2f}ms, 压缩: {self.compressed_latency_ms:.2f}ms",
            f"  延迟加速比: {self.latency_speedup:.2f}x",
            f"  吞吐量 - 原始: {self.original_throughput:.1f}样本/s, 压缩: {self.compressed_throughput:.1f}样本/s",
            f"  吞吐加速比: {self.throughput_speedup:.2f}x",
            "",
            f"[精度损失]",
            f"  Accuracy下降: {self.accuracy_drop*100:.2f}个百分点",
            f"  F1_Macro下降: {self.f1_macro_drop*100:.2f}个百分点",
            "",
            f"[导出路径]",
            f"  {self.export_path}",
        ]
        return "\n".join(lines)

    def save(self, save_path: Union[str, Path]) -> str:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"压缩报告已保存: {sp}")
        return str(sp)


# ============================================================
# 知识蒸馏训练器
# ============================================================

class DistillationTrainer:
    """
    知识蒸馏训练器

    将教师模型（如bert-base-chinese）的知识迁移到学生模型（如distilbert-base-chinese）：
    Loss = alpha * KL(softmax(z_s/T), softmax(z_t/T)) * T^2
         + (1 - alpha) * CrossEntropy(z_s, y_true)

    使用示例：
        >>> config = DistillationConfig(temperature=4.0, alpha=0.5)
        >>> trainer = DistillationTrainer(teacher, config)
        >>> student = trainer.fit(X_train, y_train, X_val, y_val)
    """

    def __init__(
        self,
        teacher_model: BERTClassifier,
        config: Optional[DistillationConfig] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Args:
            teacher_model: 已训练好的教师模型（BERTClassifier实例）
            config: 蒸馏配置，默认使用DistillationConfig默认值
            class_names: 类目名称列表
        """
        if not teacher_model.is_trained or teacher_model._model is None:
            raise ValueError("教师模型必须已训练（is_trained=True且_model非空）")

        self.teacher: BERTClassifier = teacher_model
        self.config: DistillationConfig = config or DistillationConfig()
        self.class_names: List[str] = class_names or DEFAULT_CLASS_NAMES.copy()

        if self.config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        self.student: Optional[BERTClassifier] = None
        self._student_tokenizer: Optional[Any] = None
        self._student_model: Optional[nn.Module] = None

        logger.info(
            f"[DistillationTrainer] 初始化完成 - "
            f"教师: {teacher_model.model_name}, "
            f"学生: {self.config.student_model}, "
            f"T={self.config.temperature}, alpha={self.config.alpha}, "
            f"设备={self.device}"
        )

    # --------------------------------------------------------
    # 损失函数
    # --------------------------------------------------------

    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        蒸馏损失 = alpha * KD_Loss + (1-alpha) * CE_Loss

        Args:
            student_logits: (batch, num_classes) 学生模型输出logits
            teacher_logits: (batch, num_classes) 教师模型输出logits
            labels: (batch,) 真实标签

        Returns:
            标量损失
        """
        T = self.config.temperature
        alpha = self.config.alpha

        hard_loss = F.cross_entropy(student_logits, labels)

        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        kd_loss = kd_loss * (T * T)

        total_loss = alpha * kd_loss + (1.0 - alpha) * hard_loss
        return total_loss

    # --------------------------------------------------------
    # 学生模型构建
    # --------------------------------------------------------

    def _build_student(self) -> None:
        """构建学生模型（默认DistilBERT）"""
        logger.info(f"构建学生模型: {self.config.student_model}")
        self._student_tokenizer = AutoTokenizer.from_pretrained(self.config.student_model)
        self._student_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.student_model,
            num_labels=self.teacher.num_labels,
        )
        self._student_model.to(self.device)

        self.student = BERTClassifier(
            model_name=f"distilled_{self.config.student_model.replace('/', '_')}",
            base_model=self.config.student_model,
            num_labels=self.teacher.num_labels,
            max_length=self.config.max_length,
            device=str(self.device),
        )
        self.student._tokenizer = self._student_tokenizer
        self.student._model = self._student_model
        self.student.set_class_names(self.class_names)

    # --------------------------------------------------------
    # 训练
    # --------------------------------------------------------

    def fit(
        self,
        X_train: Union[List[str], np.ndarray, pd.Series],
        y_train: Union[List[int], np.ndarray, pd.Series],
        X_val: Optional[Union[List[str], np.ndarray, pd.Series]] = None,
        y_val: Optional[Union[List[int], np.ndarray, pd.Series]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> BERTClassifier:
        """
        执行知识蒸馏训练

        Args:
            X_train: 训练文本
            y_train: 训练标签
            X_val: 验证集文本（用于早停）
            y_val: 验证集标签
            output_dir: 输出目录（模型+日志）

        Returns:
            BERTClassifier: 蒸馏训练完成的学生模型
        """
        if self.student is None:
            self._build_student()

        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "distilled" / self.student.model_name
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        X_train_list = np.asarray(X_train).tolist()
        y_train_list = np.asarray(y_train).astype(int).tolist()

        train_dataset = TextDataset(
            X_train_list, y_train_list,
            self._student_tokenizer, self.config.max_length,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_list = np.asarray(X_val).tolist()
            y_val_list = np.asarray(y_val).astype(int).tolist()
            val_dataset = TextDataset(
                X_val_list, y_val_list,
                self._student_tokenizer, self.config.max_length,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.config.batch_size, shuffle=False,
            )

        teacher_model = self.teacher._model
        teacher_tokenizer = self.teacher._tokenizer
        teacher_device = self.teacher.device
        teacher_model.eval()

        student_model = self._student_model
        student_model.train()

        no_decay = ["bias", "LayerNorm.weight"]
        param_groups = [
            {
                "params": [p for n, p in student_model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in student_model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=self.config.learning_rate)

        total_steps = len(train_loader) * self.config.num_train_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )

        scaler = torch.cuda.amp.GradScaler() if self.config.fp16 and self.device.type == "cuda" else None

        best_val_f1 = -1.0
        patience_counter = 0
        best_ckpt_path = None

        logger.info(
            f"[DistillationTrainer] 开始蒸馏训练 - "
            f"样本数={len(X_train_list)}, Epochs={self.config.num_train_epochs}, "
            f"Batch={self.config.batch_size}, LR={self.config.learning_rate}"
        )

        for epoch in range(1, self.config.num_train_epochs + 1):
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step_idx, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch["labels"]

                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                token_type_ids = batch.get("token_type_ids")

                if self.config.fp16 and scaler is not None:
                    with torch.cuda.amp.autocast():
                        student_outputs = student_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=None if isinstance(student_model, DistilBertForSequenceClassification) else token_type_ids,
                        )
                        student_logits = student_outputs.logits

                        with torch.no_grad():
                            teacher_batch = self._prepare_teacher_batch(
                                input_ids, attention_mask, teacher_device,
                            )
                            teacher_outputs = teacher_model(**teacher_batch)
                            teacher_logits = teacher_outputs.logits.to(self.device)

                        loss = self._distillation_loss(student_logits, teacher_logits, labels)
                        loss = loss / self.config.gradient_accumulation_steps

                    scaler.scale(loss).backward()
                else:
                    student_outputs = student_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=None if isinstance(student_model, DistilBertForSequenceClassification) else token_type_ids,
                    )
                    student_logits = student_outputs.logits

                    with torch.no_grad():
                        teacher_batch = self._prepare_teacher_batch(
                            input_ids, attention_mask, teacher_device,
                        )
                        teacher_outputs = teacher_model(**teacher_batch)
                        teacher_logits = teacher_outputs.logits.to(self.device)

                    loss = self._distillation_loss(student_logits, teacher_logits, labels)
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()

                epoch_loss += loss.item() * self.config.gradient_accumulation_steps

                if (step_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            avg_train_loss = epoch_loss / len(train_loader)

            val_f1 = None
            if val_loader is not None:
                val_metrics = self._validate(student_model, val_loader)
                val_f1 = val_metrics["f1_macro"]
                logger.info(
                    f"[DistillationTrainer] Epoch {epoch}/{self.config.num_train_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, Val F1_Macro: {val_f1:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    ckpt_path = output_dir / f"best_epoch{epoch}_f1{val_f1:.4f}"
                    self._save_student_checkpoint(ckpt_path)
                    if best_ckpt_path and best_ckpt_path.exists() and best_ckpt_path != ckpt_path:
                        shutil.rmtree(best_ckpt_path, ignore_errors=True)
                    best_ckpt_path = ckpt_path
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"[DistillationTrainer] 触发早停 (best F1={best_val_f1:.4f})")
                        break
            else:
                logger.info(
                    f"[DistillationTrainer] Epoch {epoch}/{self.config.num_train_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}"
                )

        if best_ckpt_path is not None:
            self._load_student_checkpoint(best_ckpt_path)
            logger.info(f"[DistillationTrainer] 已加载最佳检查点: {best_ckpt_path}")

        self.student.is_trained = True
        save_path = output_dir / "final_model"
        self.student.save(str(output_dir))
        logger.info(f"[DistillationTrainer] 蒸馏完成，学生模型保存至: {save_path}")

        return self.student

    # --------------------------------------------------------
    # 辅助方法
    # --------------------------------------------------------

    def _prepare_teacher_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """构造教师模型输入（使用教师tokenizer重新编码以确保一致性）"""
        texts = self._student_tokenizer.batch_decode(
            input_ids, skip_special_tokens=True,
        )
        encoded = self.teacher._tokenizer(
            texts,
            max_length=self.teacher.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].to(target_device),
            "attention_mask": encoded["attention_mask"].to(target_device),
            **(
                {"token_type_ids": encoded["token_type_ids"].to(target_device)}
                if "token_type_ids" in encoded else {}
            ),
        }

    @torch.no_grad()
    def _validate(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score
        model.eval()
        all_preds = []
        all_labels = []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch["labels"]
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=(
                    None if isinstance(model, DistilBertForSequenceClassification)
                    else batch.get("token_type_ids")
                ),
            )
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
        acc = accuracy_score(all_labels, all_preds)
        f1_ma = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        f1_mi = f1_score(all_labels, all_preds, average="micro", zero_division=0)
        model.train()
        return {"accuracy": float(acc), "f1_macro": float(f1_ma), "f1_micro": float(f1_mi)}

    def _save_student_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._student_tokenizer.save_pretrained(str(path))
        state = {
            "model_state_dict": self._student_model.state_dict(),
            "class_names": self.class_names,
        }
        torch.save(state, path / "pytorch_model.bin")
        if hasattr(self._student_model, "config"):
            self._student_model.config.save_pretrained(str(path))

    def _load_student_checkpoint(self, path: Path) -> None:
        state = torch.load(path / "pytorch_model.bin", map_location=self.device)
        self._student_model.load_state_dict(state["model_state_dict"])
        if "class_names" in state:
            self.class_names = state["class_names"]
            self.student.set_class_names(self.class_names)


# ============================================================
# 主模型压缩器
# ============================================================

class ModelCompressor:
    """
    模型压缩器：集成量化、知识蒸馏、ONNX导出功能

    使用示例：
        >>> compressor = ModelCompressor()
        >>> quantized, report = compressor.quantize(
        ...     model, calibration_data=X_calib, output_dir="outputs/quant"
        ... )
        >>> distilled, report = compressor.distill(
        ...     teacher_model, X_train, y_train, X_val, y_val,
        ...     output_dir="outputs/distill"
        ... )
        >>> onnx_path = compressor.export_onnx(model, "outputs/model.onnx")
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        default_output_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            class_names: 类目名称列表
            default_output_dir: 默认输出根目录
        """
        self.class_names: List[str] = class_names or DEFAULT_CLASS_NAMES.copy()
        self.default_output_dir: Optional[Path] = (
            Path(default_output_dir) if default_output_dir else None
        )
        self.evaluator = Evaluator(class_names=self.class_names)

    # ========================================================
    # 1. INT8 量化
    # ========================================================

    def quantize(
        self,
        model: BERTClassifier,
        calibration_data: Optional[Union[List[str], np.ndarray]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[QuantizationConfig] = None,
        X_eval: Optional[Union[List[str], np.ndarray]] = None,
        y_eval: Optional[Union[List[int], np.ndarray]] = None,
    ) -> Tuple[nn.Module, CompressionReport]:
        """
        对BERT分类器执行INT8量化

        Args:
            model: 已训练的BERTClassifier实例
            calibration_data: 静态量化校准数据（文本列表）
            output_dir: 输出目录
            config: 量化配置
            X_eval: 评估集（用于生成压缩报告的精度对比）
            y_eval: 评估标签

        Returns:
            (量化后模型, 压缩对比报告)
        """
        if not model.is_trained or model._model is None:
            raise ValueError("模型必须已训练才能量化")

        config = config or QuantizationConfig()
        output_dir = self._resolve_output_dir(output_dir, f"quant_{model.model_name}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        original_model = model._model
        original_model.eval()
        original_model_cpu = copy.deepcopy(original_model).cpu()

        logger.info(
            f"[ModelCompressor] 开始量化 - 类型={config.quant_type.value}, "
            f"模型={model.model_name}"
        )

        torch.backends.quantized.engine = config.qconfig_backend

        if config.quant_type == QuantizationType.DYNAMIC:
            quantized_model = self._dynamic_quantize(original_model_cpu, config)
        elif config.quant_type == QuantizationType.STATIC:
            if calibration_data is None:
                raise ValueError("静态量化必须提供calibration_data校准数据")
            quantized_model = self._static_quantize(
                original_model_cpu, model._tokenizer, calibration_data, config,
            )
        elif config.quant_type == QuantizationType.QAT:
            if calibration_data is None:
                raise ValueError("QAT量化感知训练需要校准/训练数据")
            quantized_model = self._qat_quantize(
                original_model_cpu, model._tokenizer, calibration_data, config,
            )
        else:
            raise ValueError(f"不支持的量化类型: {config.quant_type}")

        quantized_model.eval()
        logger.info("[ModelCompressor] 量化完成")

        export_path = str(output_dir / "quantized_model")
        self._save_quantized_model(quantized_model, model._tokenizer, export_path)

        onnx_path = None
        if config.export_onnx:
            try:
                onnx_path = self.export_onnx(
                    model,
                    output_path=str(output_dir / f"{model.model_name}.onnx"),
                    config=config,
                )
                export_path = onnx_path
            except Exception as e:
                logger.warning(f"导出ONNX失败（可忽略）: {e}")

        report = self._generate_compression_report(
            original=model,
            compressed=quantized_model,
            method=f"quantization_{config.quant_type.value}",
            output_dir=output_dir,
            X_eval=X_eval,
            y_eval=y_eval,
            compressed_name=f"quantized_{model.model_name}",
        )
        report.export_path = export_path or str(output_dir)
        report.save(output_dir / "compression_report.json")

        return quantized_model, report

    # --------------------------------------------------------
    # 量化内部实现
    # --------------------------------------------------------

    def _dynamic_quantize(
        self, model: nn.Module, config: QuantizationConfig,
    ) -> nn.Module:
        """动态量化：量化Linear和Embedding的权重"""
        quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Embedding},
            dtype=config.dtype,
        )
        logger.info("[ModelCompressor] 动态量化完成")
        return quantized

    def _static_quantize(
        self,
        model: nn.Module,
        tokenizer: Any,
        calibration_data: Union[List[str], np.ndarray],
        config: QuantizationConfig,
    ) -> nn.Module:
        """静态量化：校准激活范围，量化权重+激活"""
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig(config.qconfig_backend)
        prepared = torch.quantization.prepare(model, inplace=False)

        calib_texts = np.asarray(calibration_data).tolist()
        calib_dataset = TextDataset(
            calib_texts, None, tokenizer, config.calibration_batch_size,
        )
        calib_loader = DataLoader(
            calib_dataset, batch_size=config.calibration_batch_size, shuffle=False,
        )

        logger.info(f"[ModelCompressor] 静态量化校准中... 批次数={config.calibration_batches}")
        with torch.no_grad():
            for i, batch in enumerate(calib_loader):
                if i >= config.calibration_batches:
                    break
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                token_type_ids = batch.get("token_type_ids")
                try:
                    prepared(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **(
                            {"token_type_ids": token_type_ids}
                            if token_type_ids is not None else {}
                        ),
                    )
                except Exception:
                    prepared(input_ids=input_ids, attention_mask=attention_mask)

        quantized = torch.quantization.convert(prepared, inplace=False)
        logger.info("[ModelCompressor] 静态量化完成")
        return quantized

    def _qat_quantize(
        self,
        model: nn.Module,
        tokenizer: Any,
        train_data: Union[List[str], np.ndarray],
        config: QuantizationConfig,
    ) -> nn.Module:
        """量化感知训练（简易版）"""
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig(config.qconfig_backend)
        prepared = torch.quantization.prepare_qat(model, inplace=False)

        texts = np.asarray(train_data).tolist()
        dataset = TextDataset(texts, [0]*len(texts), tokenizer, config.calibration_batch_size)
        loader = DataLoader(dataset, batch_size=config.calibration_batch_size, shuffle=True)

        optimizer = torch.optim.Adam(prepared.parameters(), lr=1e-5)
        logger.info(f"[ModelCompressor] QAT训练中... 批次数={config.calibration_batches}")
        for i, batch in enumerate(loader):
            if i >= config.calibration_batches:
                break
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch.get("token_type_ids")
            optimizer.zero_grad()
            try:
                outputs = prepared(
                    input_ids=input_ids, attention_mask=attention_mask,
                    **(
                        {"token_type_ids": token_type_ids, "labels": batch["labels"]}
                        if token_type_ids is not None else {"labels": batch["labels"]}
                    ),
                )
            except Exception:
                outputs = prepared(
                    input_ids=input_ids, attention_mask=attention_mask,
                    labels=batch["labels"],
                )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        prepared.eval()
        quantized = torch.quantization.convert(prepared.eval(), inplace=False)
        logger.info("[ModelCompressor] QAT量化完成")
        return quantized

    # ========================================================
    # 2. 知识蒸馏
    # ========================================================

    def distill(
        self,
        teacher_model: BERTClassifier,
        X_train: Union[List[str], np.ndarray, pd.Series],
        y_train: Union[List[int], np.ndarray, pd.Series],
        X_val: Optional[Union[List[str], np.ndarray, pd.Series]] = None,
        y_val: Optional[Union[List[int], np.ndarray, pd.Series]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[DistillationConfig] = None,
        X_eval: Optional[Union[List[str], np.ndarray]] = None,
        y_eval: Optional[Union[List[int], np.ndarray]] = None,
    ) -> Tuple[BERTClassifier, CompressionReport]:
        """
        执行知识蒸馏：教师→学生

        Args:
            teacher_model: 已训练的教师BERTClassifier
            X_train: 训练文本
            y_train: 训练标签
            X_val: 验证文本（早停用）
            y_val: 验证标签
            output_dir: 输出目录
            config: 蒸馏配置
            X_eval: 评估集（精度对比）
            y_eval: 评估标签

        Returns:
            (学生模型, 压缩对比报告)
        """
        config = config or DistillationConfig()
        output_dir = self._resolve_output_dir(
            output_dir, f"distilled_{teacher_model.model_name}"
        )

        distillation_trainer = DistillationTrainer(
            teacher_model=teacher_model,
            config=config,
            class_names=self.class_names,
        )
        student = distillation_trainer.fit(
            X_train, y_train, X_val, y_val, output_dir=output_dir,
        )

        report = self._generate_compression_report(
            original=teacher_model,
            compressed=student,
            method=f"distillation_T{config.temperature}_a{config.alpha}",
            output_dir=Path(output_dir),
            X_eval=X_eval,
            y_eval=y_eval,
            compressed_name=student.model_name,
        )
        report.export_path = str(output_dir)
        report.save(Path(output_dir) / "compression_report.json")

        return student, report

    # ========================================================
    # 3. ONNX 导出
    # ========================================================

    def export_onnx(
        self,
        model: BERTClassifier,
        output_path: Union[str, Path],
        config: Optional[QuantizationConfig] = None,
        batch_size: int = 1,
        opset_version: int = 14,
        dynamic_batch: bool = True,
    ) -> str:
        """
        将BERT分类器导出为ONNX格式（支持batch推理）

        Args:
            model: 已训练的BERTClassifier
            output_path: ONNX输出路径
            config: QuantizationConfig（读取opset_version/dynamic_batch）
            batch_size: 导出时的样例batch大小
            opset_version: ONNX opset版本（优先级高于config）
            dynamic_batch: 是否启用动态batch维度（优先级高于config）

        Returns:
            str: ONNX文件保存路径
        """
        if not model.is_trained or model._model is None:
            raise ValueError("模型必须已训练才能导出ONNX")

        if config is not None:
            opset_version = config.onnx_opset_version
            dynamic_batch = config.onnx_dynamic_batch

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        pt_model = model._model
        pt_model.eval()
        pt_model_cpu = copy.deepcopy(pt_model).cpu()

        tokenizer = model._tokenizer
        dummy_text = ["测试文本"] * batch_size
        encoded = tokenizer(
            dummy_text,
            max_length=model.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded.get("token_type_ids")

        input_names = ["input_ids", "attention_mask"]
        inputs = (input_ids, attention_mask)
        if token_type_ids is not None:
            input_names.append("token_type_ids")
            inputs = (input_ids, attention_mask, token_type_ids)

        with_attention_mask = token_type_ids is not None

        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "seq_length"},
                "attention_mask": {0: "batch_size", 1: "seq_length"},
                "logits": {0: "batch_size"},
            }
            if with_attention_mask:
                dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "seq_length"}

        output_names = ["logits"]

        try:
            with torch.no_grad():
                symbolic_inputs = inputs
                torch.onnx.export(
                    pt_model_cpu,
                    symbolic_inputs,
                    str(out_path),
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    verbose=False,
                )
            logger.info(f"[ModelCompressor] ONNX导出成功: {out_path}")
        except Exception as e:
            logger.warning(f"标准ONNX导出失败，尝试简化模式: {e}")
            try:
                with torch.no_grad():
                    torch.onnx.export(
                        pt_model_cpu,
                        (input_ids, attention_mask),
                        str(out_path),
                        export_params=True,
                        opset_version=opset_version,
                        do_constant_folding=True,
                        input_names=["input_ids", "attention_mask"],
                        output_names=["logits"],
                        dynamic_axes={
                            "input_ids": {0: "batch_size"},
                            "attention_mask": {0: "batch_size"},
                            "logits": {0: "batch_size"},
                        },
                        verbose=False,
                    )
                logger.info(f"[ModelCompressor] ONNX简化导出成功: {out_path}")
            except Exception as e2:
                raise RuntimeError(f"ONNX导出失败: {e2}") from e2

        try:
            import onnx
            onnx_model = onnx.load(str(out_path))
            onnx.checker.check_model(onnx_model)
            logger.info("[ModelCompressor] ONNX模型验证通过")
        except ImportError:
            logger.warning("onnx未安装，跳过ONNX模型校验")
        except Exception as e:
            logger.warning(f"ONNX模型校验警告（不影响使用）: {e}")

        return str(out_path)

    # ========================================================
    # 辅助方法
    # ========================================================

    def _resolve_output_dir(
        self, user_dir: Optional[Union[str, Path]], default_name: str,
    ) -> Path:
        if user_dir is not None:
            return Path(user_dir)
        if self.default_output_dir is not None:
            return self.default_output_dir / default_name
        return Path(__file__).resolve().parent.parent.parent / "data" / "compressed" / default_name

    @staticmethod
    def _save_quantized_model(model: nn.Module, tokenizer: Any, path: Union[str, Path]) -> str:
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(save_path))
        state_path = save_path / "quantized_model.pt"
        torch.save(model.state_dict(), state_path)
        wrapper_path = save_path / "model_wrapper.pt"
        torch.save(model, wrapper_path)
        logger.info(f"量化模型已保存: {save_path}")
        return str(save_path)

    def _count_params(self, model: Any) -> float:
        """计算模型参数量（百万M）"""
        if isinstance(model, BERTClassifier):
            if model._model is None:
                return 0.0
            m = model._model
        elif isinstance(model, nn.Module):
            m = model
        else:
            return 0.0
        total = sum(p.numel() for p in m.parameters())
        return total / 1_000_000.0

    @staticmethod
    def _measure_size(paths: List[Union[str, Path]]) -> float:
        """计算文件/目录总大小（MB）"""
        total = 0.0
        for p in paths:
            pp = Path(p)
            if pp.is_file():
                total += pp.stat().st_size
            elif pp.is_dir():
                for fp in pp.rglob("*"):
                    if fp.is_file():
                        total += fp.stat().st_size
        return total / (1024 * 1024)

    def _measure_inference_speed(
        self, model: Any, sample_texts: List[str], num_runs: int = 50,
    ) -> Tuple[float, float]:
        """测量推理速度：(单样本延迟ms, 吞吐量样本/s)"""
        device = torch.device("cpu")
        is_torch_model = False

        if isinstance(model, BERTClassifier):
            if model._model is not None:
                m = copy.deepcopy(model._model).to(device)
                tok = model._tokenizer
                ml = model.max_length
                is_torch_model = True
        elif isinstance(model, nn.Module):
            m = copy.deepcopy(model).to(device)
            m.eval()
            is_torch_model = True
        else:
            m = model

        if is_torch_model and isinstance(model, BERTClassifier):
            m.eval()
            encoded = tok(
                sample_texts, max_length=ml, padding="max_length",
                truncation=True, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                _ = m(**encoded)
            latencies = []
            with torch.no_grad():
                for _ in range(num_runs):
                    t0 = time.perf_counter()
                    _ = m(**encoded)
                    latencies.append(time.perf_counter() - t0)
            avg_latency = float(np.mean(latencies)) * 1000 / len(sample_texts)
            throughput = len(sample_texts) / (float(np.mean(latencies)))
            return avg_latency, throughput

        latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            try:
                _ = model.predict(sample_texts)
            except Exception:
                pass
            latencies.append(time.perf_counter() - t0)
        avg_latency = float(np.mean(latencies)) * 1000 / len(sample_texts)
        throughput = len(sample_texts) / float(np.mean(latencies))
        return avg_latency, throughput

    def _generate_compression_report(
        self,
        original: BERTClassifier,
        compressed: Any,
        method: str,
        output_dir: Path,
        X_eval: Optional[Union[List[str], np.ndarray]],
        y_eval: Optional[Union[List[int], np.ndarray]],
        compressed_name: str,
    ) -> CompressionReport:
        """生成压缩对比报告"""
        report = CompressionReport(
            original_model_name=original.model_name,
            compressed_model_name=compressed_name,
            method=method,
        )

        report.original_params_million = self._count_params(original)
        report.compressed_params_million = self._count_params(compressed)
        if report.compressed_params_million > 0:
            report.param_compression_ratio = (
                report.original_params_million / report.compressed_params_million
            )

        tmp_orig = output_dir.parent / f"_tmp_orig_{original.model_name}"
        orig_path = None
        try:
            orig_path = original.save(str(tmp_orig))
        except Exception:
            pass
        if orig_path:
            report.original_size_mb = self._measure_size([orig_path])
        report.compressed_size_mb = self._measure_size([output_dir])
        if report.compressed_size_mb > 0:
            report.size_compression_ratio = (
                report.original_size_mb / report.compressed_size_mb
            )
        if orig_path and Path(orig_path).exists():
            shutil.rmtree(Path(orig_path), ignore_errors=True)

        sample_texts = ["测试样本文本" for _ in range(8)]
        try:
            orig_lat, orig_tp = self._measure_inference_speed(original, sample_texts, num_runs=30)
            report.original_latency_ms = orig_lat
            report.original_throughput = orig_tp
        except Exception as e:
            logger.warning(f"测量原始模型推理速度失败: {e}")
        try:
            comp_lat, comp_tp = self._measure_inference_speed(compressed, sample_texts, num_runs=30)
            report.compressed_latency_ms = comp_lat
            report.compressed_throughput = comp_tp
        except Exception as e:
            logger.warning(f"测量压缩模型推理速度失败: {e}")
        if report.compressed_latency_ms > 0:
            report.latency_speedup = report.original_latency_ms / report.compressed_latency_ms
        if report.compressed_throughput > 0:
            report.throughput_speedup = report.compressed_throughput / report.original_throughput

        if X_eval is not None and y_eval is not None:
            try:
                orig_result = self.evaluator.evaluate(original, X_eval, y_eval)
                report.original_metrics = {
                    "accuracy": orig_result.accuracy,
                    "precision_macro": orig_result.precision_macro,
                    "recall_macro": orig_result.recall_macro,
                    "f1_macro": orig_result.f1_macro,
                    "f1_micro": orig_result.f1_micro,
                }
            except Exception as e:
                logger.warning(f"评估原始模型失败: {e}")

            try:
                if isinstance(compressed, BERTClassifier):
                    comp_result = self.evaluator.evaluate(compressed, X_eval, y_eval)
                else:
                    comp_result = self._evaluate_quantized(
                        compressed, original, X_eval, y_eval,
                    )
                report.compressed_metrics = {
                    "accuracy": comp_result.accuracy if isinstance(comp_result, EvaluationResult) else comp_result.get("accuracy", 0.0),
                    "precision_macro": comp_result.precision_macro if isinstance(comp_result, EvaluationResult) else comp_result.get("precision_macro", 0.0),
                    "recall_macro": comp_result.recall_macro if isinstance(comp_result, EvaluationResult) else comp_result.get("recall_macro", 0.0),
                    "f1_macro": comp_result.f1_macro if isinstance(comp_result, EvaluationResult) else comp_result.get("f1_macro", 0.0),
                    "f1_micro": comp_result.f1_micro if isinstance(comp_result, EvaluationResult) else comp_result.get("f1_micro", 0.0),
                }
            except Exception as e:
                logger.warning(f"评估压缩模型失败: {e}")

            report.accuracy_drop = (
                report.original_metrics.get("accuracy", 0.0)
                - report.compressed_metrics.get("accuracy", 0.0)
            )
            report.f1_macro_drop = (
                report.original_metrics.get("f1_macro", 0.0)
                - report.compressed_metrics.get("f1_macro", 0.0)
            )

        return report

    @torch.no_grad()
    def _evaluate_quantized(
        self,
        quant_model: nn.Module,
        original_model: BERTClassifier,
        X_eval: Union[List[str], np.ndarray],
        y_eval: Union[List[int], np.ndarray],
    ) -> Dict[str, float]:
        """评估量化后的torch模型"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        quant_model.eval()
        tokenizer = original_model._tokenizer
        max_length = original_model.max_length
        batch_size = original_model.batch_size

        texts = np.asarray(X_eval).tolist()
        dataset = TextDataset(texts, None, tokenizer, max_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_preds = []
        for batch in loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch.get("token_type_ids")
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                inputs["token_type_ids"] = token_type_ids
            try:
                outputs = quant_model(**inputs)
            except Exception:
                outputs = quant_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).numpy()
            all_preds.extend(preds.tolist())

        y_true_arr = np.asarray(y_eval).astype(int)
        y_pred_arr = np.asarray(all_preds).astype(int)
        return {
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "precision_macro": float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
            "precision_micro": float(precision_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)),
            "recall_macro": float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
            "recall_micro": float(recall_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
            "f1_micro": float(f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)),
        }
