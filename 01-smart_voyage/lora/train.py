# -*- coding: utf-8 -*-
"""
LoRA 微调训练脚本
用于文旅出行场景的领域微调
支持：单卡训练、LoRA 参数高效微调

使用方法：
    python -m lora.train                          # 使用默认配置
    python -m lora.train --model-path /path/to/llm # 指定模型路径
    python -m lora.train --epochs 5 --lr 1e-4     # 自定义参数
"""
from __future__ import annotations

import os
import sys
import json
import argparse

# 仅在需要训练时导入（避免无 GPU 环境下的导入错误）
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        prepare_model_for_kbit_training,
    )
    from datasets import Dataset
except ImportError as e:
    print(f"[LoRA] 缺少必要依赖: {e}")
    print("[LoRA] 请安装: pip install peft transformers datasets accelerate")
    sys.exit(1)


from .lora_config import LoRAConfig, TRAVEL_INSTRUCTION_TEMPLATE, SAMPLE_TRAIN_DATA
from .data_preprocess import preprocess_travel_data


def parse_args():
    parser = argparse.ArgumentParser(description="途景汇 LoRA 微调训练")
    parser.add_argument("--model-path", type=str, default="", help="本地模型路径")
    parser.add_argument("--base-model", type=str, default="qwen2.5-7b-instruct", help="基础模型名称")
    parser.add_argument("--train-data", type=str, default="", help="训练数据路径")
    parser.add_argument("--output-dir", type=str, default="./models/lora-output", help="输出目录")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--batch-size", type=int, default=2, help="批大小")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA 秩")
    parser.add_argument("--max-length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--skip-preprocess", action="store_true", help="跳过数据预处理")
    return parser.parse_args()


def load_and_prepare_data(train_data_path: str, tokenizer, max_length: int) -> Dataset:
    """加载并准备训练数据"""
    if not os.path.exists(train_data_path):
        print(f"[LoRA] 训练数据不存在，使用示例数据: {train_data_path}")
        preprocess_travel_data(output_path=train_data_path)

    with open(train_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 格式化为指令-回答
    def format_example(example):
        text = example.get("text", "")
        if not text:
            instruction = example.get("instruction", "")
            output = example.get("output", "")
            text = TRAVEL_INSTRUCTION_TEMPLATE.format(
                instruction=instruction, response=output
            )
        return tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    dataset = Dataset.from_list(raw_data)
    tokenized = dataset.map(format_example, remove_columns=dataset.column_names)
    return tokenized


def create_lora_model(model_name_or_path: str, config: LoRAConfig):
    """创建带 LoRA 的模型"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map="auto",
    )

    # 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(config: LoRAConfig):
    """执行训练"""
    # 确定模型路径
    model_path = config.model_path or config.base_model_name
    print(f"[LoRA] 加载模型: {model_path}")

    # 创建模型和 tokenizer
    model, tokenizer = create_lora_model(model_path, config)

    # 加载数据
    train_data_path = config.train_data_path
    if not os.path.exists(train_data_path):
        preprocess_travel_data(output_path=train_data_path)

    tokenized_dataset = load_and_prepare_data(
        train_data_path, tokenizer, config.max_seq_length
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
        report_to="none",
    )

    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("[LoRA] 开始训练...")
    trainer.train()

    # 保存
    os.makedirs(config.output_dir, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # 保存配置
    config_path = os.path.join(config.output_dir, "lora_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"[LoRA] 训练完成！模型保存至: {config.output_dir}")
    return config.output_dir


def merge_lora_with_base(lora_path: str, base_model_path: str, output_path: str):
    """将 LoRA 权重合并到基础模型"""
    from peft import PeftModel

    print(f"[LoRA] 加载 LoRA 适配器: {lora_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, lora_path)

    print("[LoRA] 合并权重...")
    merged_model = model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

    print(f"[LoRA] 合并完成！模型保存至: {output_path}")


def test_lora_model(model_path: str, test_prompt: str = ""):
    """测试微调后的模型"""
    from transformers import pipeline

    print(f"[LoRA] 加载微调模型: {model_path}")
    generator = pipeline(
        "text-generation",
        model=model_path,
        torch_dtype=torch.float16,
        device=0 if torch.cuda.is_available() else -1,
    )

    if not test_prompt:
        test_prompts = [
            "帮我查一下三亚明天的天气",
            "推荐成都3天游玩的景点",
            "从北京到上海怎么去最方便",
        ]
    else:
        test_prompts = [test_prompt]

    for prompt in test_prompts:
        formatted = TRAVEL_INSTRUCTION_TEMPLATE.format(
            instruction=prompt, response=""
        )
        result = generator(
            formatted,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
        )
        generated_text = result[0]["generated_text"]
        print(f"\n{'='*60}")
        print(f"问题: {prompt}")
        print(f"回答: {generated_text}")


if __name__ == "__main__":
    args = parse_args()

    config = LoRAConfig(
        base_model_name=args.base_model,
        model_path=args.model_path,
        train_data_path=args.train_data or "./data/train/train_data.json",
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        lora_r=args.lora_r,
        max_seq_length=args.max_length,
    )

    train(config)