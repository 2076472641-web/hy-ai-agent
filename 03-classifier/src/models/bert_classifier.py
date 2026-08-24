# -*- coding: utf-8 -*-
"""
BERT 微调分类器
基于 HuggingFace transformers 库，支持中文预训练模型 bert-base-chinese。

核心特性：
1. Prompt 模板学习：支持如"这是{mask}类商品"的提示词模板，将分类转化为完形填空任务
2. 矩阵分解优化：分类层使用低秩矩阵分解 (LoRA 风格)，压缩参数量加速推理
3. 多轮超参搜索：集成 Optuna 风格的超参优化框架（多轮贝叶斯搜索）
4. 最佳检查点保存：训练过程中自动保存验证集最佳权重
5. 标签平滑与知识蒸馏：支持冷门类目的软标签优化
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import math
import os
import copy
import json
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
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
        BertTokenizer,
        BertForSequenceClassification,
        get_linear_schedule_with_warmup,
        AdamW,
    )
except ImportError:
    raise ImportError(
        "transformers 未安装，请先安装: pip install transformers>=4.30.0"
    )

from .base import BaseClassifier


# ============================================================
# 辅助数据结构与工具函数
# ============================================================

class TextDataset(Dataset):
    """文本分类 PyTorch Dataset"""

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: Any = None,
        max_length: int = 128,
        prompt_template: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Args:
            texts: 文本列表
            labels: 标签列表（None 表示预测模式）
            tokenizer: HuggingFace Tokenizer
            max_length: 最大序列长度
            prompt_template: Prompt 模板，如 "这是{mask}类商品。{text}"
            class_names: 类别名称列表，用于 prompt 构建
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.class_names = class_names
        self.use_prompt = prompt_template is not None

        if self.use_prompt and class_names is None:
            raise ValueError("使用 prompt 模板时必须提供 class_names")

        # 计算 prompt 中 [MASK] token 的位置索引（用于取出 mask 表示）
        self._mask_pos_cache: Optional[int] = None

    def __len__(self) -> int:
        return len(self.texts)

    def _build_prompt_text(self, text: str) -> str:
        """用模板构建 prompt 文本"""
        return self.prompt_template.format(mask="[MASK]", text=text)

    def _get_mask_position(self, input_ids: torch.Tensor) -> int:
        """获取 [MASK] token 在序列中的位置索引（缓存第一个样本计算结果）"""
        if self._mask_pos_cache is not None:
            return self._mask_pos_cache
        mask_token_id = self.tokenizer.mask_token_id
        positions = torch.where(input_ids == mask_token_id)[0]
        if len(positions) == 0:
            raise ValueError("Prompt 模板中未找到 [MASK] token")
        self._mask_pos_cache = int(positions[0].item())
        return self._mask_pos_cache

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = str(self.texts[idx])
        if self.use_prompt:
            text = self._build_prompt_text(text)

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)

        if self.use_prompt:
            item["mask_position"] = self._get_mask_position(item["input_ids"])

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


# ============================================================
# 低秩矩阵分解分类头
# ============================================================

class LowRankClassifier(nn.Module):
    """
    低秩矩阵分解分类层
    
    将原始的 (hidden_size, num_classes) 权重矩阵分解为两个小矩阵相乘：
    W ≈ A @ B, 其中 A: (hidden_size, rank), B: (rank, num_classes)
    
    参数量从 H*K 降为 r*(H+K)，当 r << min(H,K) 时显著压缩。
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        rank: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.rank = rank

        self.dropout = nn.Dropout(dropout)
        self.matrix_a = nn.Linear(hidden_size, rank, bias=False)
        self.matrix_b = nn.Linear(rank, num_classes, bias=True)

        # 初始化：使用 SVD 风格的小值初始化
        nn.init.xavier_uniform_(self.matrix_a.weight)
        nn.init.xavier_uniform_(self.matrix_b.weight)
        nn.init.zeros_(self.matrix_b.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (batch, hidden_size) <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>向量

        Returns:
            logits: (batch, num_classes)
        """
        x = self.dropout(hidden)
        x = self.matrix_a(x)  # (batch, rank)
        logits = self.matrix_b(x)  # (batch, num_classes)
        return logits


# ============================================================
# BERT 分类器主类
# ============================================================

class BERTClassifier(BaseClassifier):
    """
    BERT 微调文本分类器
    
    支持三种训练模式：
    1. 标准分类 (default)：<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> + 全连接分类头
    2. Prompt 分类：使用 "[MASK]" 模板，将分类转化为完形填空
    3. 低秩分解模式：分类层矩阵分解压缩，加速推理

    Args:
        model_name: 模型名称标识
        base_model: HuggingFace 预训练模型名，默认 'bert-base-chinese'
        num_labels: 分类数量，默认 12
        max_length: 最大序列长度
        batch_size: 批大小
        learning_rate: 学习率
        num_train_epochs: 训练轮数
        warmup_ratio: warmup 步数比例
        weight_decay: 权重衰减
        gradient_accumulation_steps: 梯度累积步数
        fp16: 是否使用混合精度
        use_prompt: 是否使用 Prompt 模板
        prompt_template: Prompt 模板字符串，需包含 {mask} 和 {text} 占位符
        use_low_rank: 是否使用低秩矩阵分解分类层
        low_rank: 分解秩大小
        label_smoothing: 标签平滑系数 epsilon
        device: 训练设备，None 为自动选择
        random_state: 随机种子
    """

    DEFAULT_PROMPT = "这是{mask}类商品。{text}"

    def __init__(
        self,
        model_name: str = "bert_classifier",
        base_model: str = "bert-base-chinese",
        num_labels: int = 12,
        max_length: int = 128,
        batch_size: int = 32,
        learning_rate: float = 2.0e-5,
        num_train_epochs: int = 5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 2,
        fp16: bool = False,
        use_prompt: bool = False,
        prompt_template: Optional[str] = None,
        use_low_rank: bool = False,
        low_rank: int = 64,
        label_smoothing: float = 0.0,
        device: Optional[str] = None,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)

        # 模型结构超参数
        self.base_model = base_model
        self.num_labels = num_labels
        self.max_length = max_length
        self.use_prompt = use_prompt
        self.prompt_template = prompt_template or (
            self.DEFAULT_PROMPT if use_prompt else None
        )
        self.use_low_rank = use_low_rank
        self.low_rank = low_rank
        self.label_smoothing = label_smoothing

        # 训练超参数
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16

        # 设备与随机种子
        self.random_state = random_state
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 运行时对象
        self._tokenizer: Optional[Any] = None
        self._model: Optional[nn.Module] = None
        self._best_ckpt_path: Optional[str] = None  # 最佳检查点路径
        self._class_token_ids: Optional[List[int]] = None  # Prompt模式下类别名称的token id

        # 保存超参数
        self.params.update({
            "base_model": base_model,
            "num_labels": num_labels,
            "max_length": max_length,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "fp16": fp16,
            "use_prompt": use_prompt,
            "prompt_template": self.prompt_template,
            "use_low_rank": use_low_rank,
            "low_rank": low_rank,
            "label_smoothing": label_smoothing,
            "device": str(self.device),
            "random_state": random_state,
        })

        # 设置随机种子
        self._set_seed()

    # --------------------------------------------------------
    # 初始化与工具方法
    # --------------------------------------------------------

    def _set_seed(self) -> None:
        """设置 Python / NumPy / PyTorch 随机种子以保证可复现"""
        import random
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _build_model_and_tokenizer(self) -> None:
        """加载预训练 Tokenizer 和 模型，并替换分类头"""
        logger.info(
            f"[{self.model_name}] 加载预训练模型: {self.base_model}, "
            f"Prompt模式: {self.use_prompt}, 低秩分解: {self.use_low_rank}"
        )

        if self.use_prompt:
            # Prompt 模式使用 MaskedLM 模型
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            self._model = AutoModelForMaskedLM.from_pretrained(self.base_model)
            # 预计算类别名称对应的第一个子词 token id（用于从 vocab 概率中取出类别分数）
            self._build_class_token_map()
        else:
            # 标准分类模式
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            base = AutoModelForSequenceClassification.from_pretrained(
                self.base_model, num_labels=self.num_labels
            )

            if self.use_low_rank:
                # 替换原生分类头为低秩分解版本
                hidden_size = base.config.hidden_size
                base.classifier = LowRankClassifier(
                    hidden_size=hidden_size,
                    num_classes=self.num_labels,
                    rank=self.low_rank,
                    dropout=base.config.hidden_dropout_prob,
                )
                logger.info(
                    f"[{self.model_name}] 已替换低秩分类头 - rank={self.low_rank}, "
                    f"参数压缩率: {(self.low_rank * (hidden_size + self.num_labels)) / (hidden_size * self.num_labels):.2%}"
                )

            self._model = base

        self._model.to(self.device)

    def _build_class_token_map(self) -> None:
        """
        Prompt 模式下：为每个类别名称计算其第一个 token 的 id。
        后续从 MLM 输出的 vocab logits 中，取出这些 token 的概率作为类别分数。
        """
        class_names = self.get_class_names()
        self._class_token_ids = []
        for name in class_names:
            # 取类别名称的第一个子词（通常是第一个字）的 id
            tokens = self._tokenizer.tokenize(name)
            if len(tokens) == 0:
                tokens = [self._tokenizer.unk_token]
            token_id = self._tokenizer.convert_tokens_to_ids(tokens[0])
            self._class_token_ids.append(token_id)
        logger.debug(
            f"[{self.model_name}] 类别名称首 token id: {self._class_token_ids}"
        )

    # --------------------------------------------------------
    # 训练核心逻辑
    # --------------------------------------------------------

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算损失，支持标签平滑

        Args:
            logits: (batch, num_classes)
            labels: (batch,)

        Returns:
            标量 loss
        """
        if self.label_smoothing > 0:
            eps = self.label_smoothing
            K = self.num_labels
            log_probs = F.log_softmax(logits, dim=-1)
            # 平滑后的 one-hot：正确类 1-eps，其余 eps/(K-1)
            with torch.no_grad():
                smooth = torch.full_like(log_probs, eps / (K - 1))
                smooth.scatter_(1, labels.unsqueeze(1), 1.0 - eps)
            loss = -(smooth * log_probs).sum(dim=-1).mean()
        else:
            loss = F.cross_entropy(logits, labels)
        return loss

    def _forward_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播一个 batch，返回 logits 和 loss

        Returns:
            (logits, loss)
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        labels = batch.get("labels")

        if self.use_prompt:
            # MLM 模式
            mask_pos = batch["mask_position"]  # (batch,)
            outputs = self._model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            # logits: (batch, seq, vocab) -> (batch, vocab) at mask position
            batch_idx = torch.arange(outputs.logits.size(0), device=self.device)
            mask_logits = outputs.logits[batch_idx, mask_pos]  # (batch, vocab)
            # 取类别对应的 token 分数
            class_idx = torch.tensor(self._class_token_ids, device=self.device)
            logits = mask_logits[:, class_idx]  # (batch, num_classes)
        else:
            # 标准分类模式
            outputs = self._model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            logits = outputs.logits

        loss = self._compute_loss(logits, labels) if labels is not None else None
        return logits, loss

    def fit(
        self,
        X: Union[List[str], np.ndarray, pd.Series],
        y: Union[List[int], np.ndarray, pd.Series],
        X_val: Optional[Union[List[str], np.ndarray, pd.Series]] = None,
        y_val: Optional[Union[List[int], np.ndarray, pd.Series]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        early_stopping_patience: int = 3,
        save_best: bool = True,
        teacher_probs: Optional[np.ndarray] = None,
        distill_temperature: float = 4.0,
        distill_alpha: float = 0.5,
        **kwargs,
    ) -> "BERTClassifier":
        """
        训练 BERT 分类器

        Args:
            X: 训练文本
            y: 训练标签 0-11
            X_val: 验证集文本（用于早停与最佳权重保存）
            y_val: 验证集标签
            output_dir: 模型与检查点输出目录
            early_stopping_patience: 验证集性能未提升 patience 轮后早停
            save_best: 是否保存验证集最佳检查点
            teacher_probs: 教师模型软标签 (n, 12)，启用知识蒸馏
            distill_temperature: 蒸馏温度
            distill_alpha: 蒸馏损失权重（硬标签损失权重 1-alpha）
            **kwargs: 额外参数

        Returns:
            self
        """
        X_array, y_array = self._validate_inputs(X, y)
        X_train = X_array.tolist()
        y_train = y_array.tolist()

        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "models" / self.model_name
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        # 构建 tokenizer + model
        if self._model is None:
            self._build_model_and_tokenizer()

        # Dataset & DataLoader
        train_dataset = TextDataset(
            X_train, y_train, self._tokenizer, self.max_length,
            prompt_template=self.prompt_template if self.use_prompt else None,
            class_names=self.get_class_names() if self.use_prompt else None,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_arr = np.asarray(X_val).tolist()
            y_val_arr = np.asarray(y_val).tolist()
            val_dataset = TextDataset(
                X_val_arr, y_val_arr, self._tokenizer, self.max_length,
                prompt_template=self.prompt_template if self.use_prompt else None,
                class_names=self.get_class_names() if self.use_prompt else None,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        # 优化器与调度器
        no_decay = ["bias", "LayerNorm.weight"]
        param_groups = [
            {
                "params": [p for n, p in self._model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self._model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate)

        total_steps = len(train_loader) * self.num_train_epochs // max(1, self.gradient_accumulation_steps)
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # 混合精度
        scaler = torch.cuda.amp.GradScaler() if self.fp16 and self.device.type == "cuda" else None

        # 蒸馏教师标签处理
        use_distill = teacher_probs is not None
        if use_distill:
            teacher_probs = torch.tensor(np.asarray(teacher_probs), dtype=torch.float32)
            logger.info(
                f"[{self.model_name}] 启用知识蒸馏 - 温度={distill_temperature}, alpha={distill_alpha}"
            )

        # 训练循环
        best_val_f1 = -1.0
        patience_counter = 0
        global_step = 0
        train_history = []

        logger.info(
            f"[{self.model_name}] 开始训练 - 样本数={len(X_train)}, "
            f"Epochs={self.num_train_epochs}, Batch={self.batch_size}, "
            f"LR={self.learning_rate}, Device={self.device}"
        )

        for epoch in range(1, self.num_train_epochs + 1):
            self._model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step_idx, batch in enumerate(train_loader):
                labels = batch["labels"]

                if self.fp16 and scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits, task_loss = self._forward_batch(batch)
                else:
                    logits, task_loss = self._forward_batch(batch)

                loss = task_loss

                # 知识蒸馏损失
                if use_distill:
                    batch_indices = step_idx * self.batch_size + torch.arange(len(labels))
                    batch_indices = batch_indices.clamp(max=len(teacher_probs) - 1)
                    teacher_batch = teacher_probs[batch_indices].to(self.device)
                    student_log_probs = F.log_softmax(logits / distill_temperature, dim=-1)
                    teacher_probs_sm = F.softmax(teacher_batch / distill_temperature, dim=-1)
                    kd_loss = F.kl_div(student_log_probs, teacher_probs_sm, reduction="batchmean")
                    kd_loss = kd_loss * (distill_temperature ** 2)
                    loss = (1 - distill_alpha) * loss + distill_alpha * kd_loss

                loss = loss / self.gradient_accumulation_steps

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item() * self.gradient_accumulation_steps

                if (step_idx + 1) % self.gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            avg_train_loss = epoch_loss / len(train_loader)

            # 验证
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._evaluate(val_loader)
                val_f1 = val_metrics["f1_macro"]
                logger.info(
                    f"[{self.model_name}] Epoch {epoch}/{self.num_train_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, Val Macro-F1: {val_f1:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )

                # 早停 & 最佳检查点
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    if save_best:
                        best_path = ckpt_dir / f"best_epoch{epoch}_f1{val_f1:.4f}"
                        self._save_checkpoint(best_path)
                        # 清理旧的最佳
                        if self._best_ckpt_path and Path(self._best_ckpt_path).exists():
                            old = Path(self._best_ckpt_path)
                            if old != best_path and old.parent == best_path.parent:
                                shutil.rmtree(old, ignore_errors=True)
                        self._best_ckpt_path = str(best_path)
                        logger.info(
                            f"[{self.model_name}] 保存最佳检查点: {best_path.name}"
                        )
                else:
                    patience_counter += 1
                    logger.info(
                        f"[{self.model_name}] 验证集未提升 ({best_val_f1:.4f}) - "
                        f"Patience {patience_counter}/{early_stopping_patience}"
                    )
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"[{self.model_name}] 触发早停，终止训练")
                        break
            else:
                logger.info(
                    f"[{self.model_name}] Epoch {epoch}/{self.num_train_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}"
                )

            train_history.append({"epoch": epoch, "train_loss": avg_train_loss, **val_metrics})

        # 训练结束：加载最佳检查点
        if save_best and self._best_ckpt_path is not None:
            self._load_checkpoint(Path(self._best_ckpt_path))
            logger.info(f"[{self.model_name}] 已加载最佳检查点权重: {self._best_ckpt_path}")

        # 保存训练历史
        history_path = output_dir / "train_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(train_history, f, ensure_ascii=False, indent=2)

        self.is_trained = True
        return self

    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """在验证集上评估，返回 accuracy 和 macro-f1"""
        from sklearn.metrics import accuracy_score, f1_score

        self._model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                labels = batch["labels"]
                logits, _ = self._forward_batch(batch)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
        return {"accuracy": acc, "f1_macro": f1_macro, "f1_micro": f1_micro}

    # --------------------------------------------------------
    # 检查点保存/加载
    # --------------------------------------------------------

    def _save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save_pretrained(str(path))
        state = {
            "model_state_dict": self._model.state_dict(),
            "class_names": self.get_class_names(),
            "class_token_ids": self._class_token_ids,
        }
        torch.save(state, path / "pytorch_model.bin")
        # 保存 config 副本
        if hasattr(self._model, "config"):
            self._model.config.save_pretrained(str(path))

    def _load_checkpoint(self, path: Path) -> None:
        if self._model is None:
            self._build_model_and_tokenizer()
        state = torch.load(path / "pytorch_model.bin", map_location=self.device)
        self._model.load_state_dict(state["model_state_dict"])
        if "class_names" in state and state["class_names"] is not None:
            self.set_class_names(state["class_names"])
        if state.get("class_token_ids") is not None:
            self._class_token_ids = state["class_token_ids"]

    # --------------------------------------------------------
    # 超参搜索（多轮）
    # --------------------------------------------------------

    def run_hyperparam_search(
        self,
        X: Union[List[str], np.ndarray, pd.Series],
        y: Union[List[int], np.ndarray, pd.Series],
        X_val: Union[List[str], np.ndarray, pd.Series],
        y_val: Union[List[int], np.ndarray, pd.Series],
        search_rounds: int = 3,
        n_trials_per_round: int = 5,
        base_output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        多轮超参搜索（简易贝叶斯 + 贪心迭代）

        每轮从搜索空间随机采样，训练并评估，保留最佳参数；
        下一轮在最佳参数邻域内更精细搜索。

        Args:
            X: 训练文本
            y: 训练标签
            X_val: 验证文本
            y_val: 验证标签
            search_rounds: 搜索轮数
            n_trials_per_round: 每轮采样实验数
            base_output_dir: 搜索结果输出目录

        Returns:
            Dict: 最佳超参数
        """
        import random as pyrandom

        X_arr, y_arr = self._validate_inputs(X, y)

        # 初始搜索空间
        spaces = {
            "learning_rate": [1e-5, 3e-5, 5e-5],
            "batch_size": [16, 32],
            "num_train_epochs": [3, 5],
            "max_length": [64, 128],
            "warmup_ratio": [0.05, 0.1, 0.2],
        }
        if self.use_low_rank:
            spaces["low_rank"] = [32, 64, 128]

        best_params = None
        best_score = -1.0
        history = []

        pyrandom.seed(self.random_state)

        logger.info(
            f"[{self.model_name}] 启动多轮超参搜索 - 轮数={search_rounds}, "
            f"每轮实验={n_trials_per_round}"
        )

        for rnd in range(1, search_rounds + 1):
            logger.info(f"[{self.model_name}] ====== 第 {rnd}/{search_rounds} 轮搜索 ======")
            # 每轮采样 n_trials_per_round 组参数
            round_results = []

            for trial in range(1, n_trials_per_round + 1):
                trial_params = {}
                for key, candidates in spaces.items():
                    # 最佳参数存在则以 60% 概率选邻域
                    if best_params is not None and pyrandom.random() < 0.6:
                        idx = candidates.index(best_params[key]) if best_params[key] in candidates else 0
                        delta = pyrandom.choice([-1, 0, 1])
                        idx = max(0, min(len(candidates) - 1, idx + delta))
                        trial_params[key] = candidates[idx]
                    else:
                        trial_params[key] = pyrandom.choice(candidates)

                logger.info(
                    f"[{self.model_name}] Round{rnd}-Trial{trial}: {trial_params}"
                )

                # 训练临时模型
                trial_model = BERTClassifier(
                    model_name=f"search_r{rnd}_t{trial}",
                    base_model=self.base_model,
                    num_labels=self.num_labels,
                    use_prompt=self.use_prompt,
                    prompt_template=self.prompt_template,
                    use_low_rank=self.use_low_rank,
                    label_smoothing=self.label_smoothing,
                    device=str(self.device),
                    random_state=self.random_state,
                    **trial_params,
                )
                trial_out = (
                    Path(base_output_dir) / f"r{rnd}_t{trial}"
                    if base_output_dir else None
                )
                trial_model.fit(
                    X_arr, y_arr, X_val, y_val,
                    output_dir=trial_out,
                    early_stopping_patience=2,
                    save_best=False,
                )
                preds = trial_model.predict(X_val)
                from sklearn.metrics import f1_score
                score = f1_score(np.asarray(y_val).tolist(), preds, average="macro")
                round_results.append((score, trial_params))
                logger.info(
                    f"[{self.model_name}] Round{rnd}-Trial{trial} 验证 Macro-F1: {score:.4f}"
                )

            # 选出本轮最佳
            round_results.sort(key=lambda x: x[0], reverse=True)
            rnd_best_score, rnd_best_params = round_results[0]
            if rnd_best_score > best_score:
                best_score = rnd_best_score
                best_params = rnd_best_params
            history.append({"round": rnd, "best_score": rnd_best_score, "best_params": rnd_best_params})
            logger.info(
                f"[{self.model_name}] 第 {rnd} 轮完成 - 本轮最佳 F1={rnd_best_score:.4f}, "
                f"全局最佳 F1={best_score:.4f}"
            )

        logger.info(
            f"[{self.model_name}] 超参搜索完成 - 全局最佳 F1={best_score:.4f}, 最佳参数: {best_params}"
        )

        # 应用最佳参数
        if best_params:
            self.set_params(**best_params)

        return best_params or {}

    # --------------------------------------------------------
    # 预测接口
    # --------------------------------------------------------

    def predict(
        self, X: Union[List[str], np.ndarray, pd.Series]
    ) -> np.ndarray:
        """预测类别ID"""
        if not self.is_trained or self._model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit() 方法")

        X_list = np.asarray(X).tolist()
        dataset = TextDataset(
            X_list, None, self._tokenizer, self.max_length,
            prompt_template=self.prompt_template if self.use_prompt else None,
            class_names=self.get_class_names() if self.use_prompt else None,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self._model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                logits, _ = self._forward_batch(batch)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())

        return np.array(all_preds)

    def predict_proba(
        self, X: Union[List[str], np.ndarray, pd.Series]
    ) -> np.ndarray:
        """预测类别概率矩阵 (n, 12)"""
        if not self.is_trained or self._model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit() 方法")

        X_list = np.asarray(X).tolist()
        dataset = TextDataset(
            X_list, None, self._tokenizer, self.max_length,
            prompt_template=self.prompt_template if self.use_prompt else None,
            class_names=self.get_class_names() if self.use_prompt else None,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self._model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                logits, _ = self._forward_batch(batch)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)

    # --------------------------------------------------------
    # 模型保存与加载
    # --------------------------------------------------------

    def save(self, save_dir: Union[str, Path]) -> str:
        """保存整个模型和 tokenizer"""
        if not self.is_trained or self._model is None:
            raise RuntimeError("模型尚未训练，无法保存")

        save_path = Path(save_dir) / self.model_name
        save_path.mkdir(parents=True, exist_ok=True)

        # 元数据
        self._save_metadata(save_path)

        # Tokenizer
        self._tokenizer.save_pretrained(str(save_path))

        # 模型权重
        state = {
            "model_state_dict": self._model.state_dict(),
            "class_names": self.get_class_names(),
            "class_token_ids": self._class_token_ids,
            "best_ckpt_path": self._best_ckpt_path,
        }
        torch.save(state, save_path / "pytorch_model.bin")

        # Config（便于后续 from_pretrained）
        if hasattr(self._model, "config"):
            self._model.config.save_pretrained(str(save_path))

        logger.info(f"[{self.model_name}] 模型已保存到: {save_path}")
        return str(save_path)

    def load(self, load_dir: Union[str, Path]) -> "BERTClassifier":
        """从目录加载模型"""
        load_path = Path(load_dir)

        # 元数据
        self._load_metadata(load_path)
        for key, val in self.params.items():
            if hasattr(self, key):
                # device 参数单独处理
                if key == "device":
                    self.device = torch.device(val)
                    continue
                setattr(self, key, val)

        # 构建模型结构
        if self._model is None:
            self._build_model_and_tokenizer()

        # Tokenizer 可能覆盖原生 tokenizer（使用保存的）
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(str(load_path))
        except Exception as e:
            logger.warning(f"加载保存的 tokenizer 失败，使用默认: {e}")

        # 权重
        weight_path = load_path / "pytorch_model.bin"
        if not weight_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {weight_path}")
        state = torch.load(str(weight_path), map_location=self.device)
        self._model.load_state_dict(state["model_state_dict"])
        if state.get("class_names") is not None:
            self.set_class_names(state["class_names"])
        self._class_token_ids = state.get("class_token_ids")
        self._best_ckpt_path = state.get("best_ckpt_path")

        self.is_trained = True
        logger.info(f"[{self.model_name}] 模型已从 {load_path} 加载")
        return self
