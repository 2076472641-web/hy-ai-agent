# -*- coding: utf-8 -*-
"""
LoRA 微调数据预处理
将原始数据转换为训练所需格式
"""
from __future__ import annotations

import os
import json
import csv
from typing import List, Dict


def preprocess_travel_data(
    input_path: str = "",
    output_path: str = "./data/train/train_data.json",
    template: str = "",
) -> str:
    """
    预处理文旅训练数据

    Args:
        input_path: 原始数据路径（支持 JSON/CSV）
        output_path: 输出路径
        template: 指令模板

    Returns:
        输出文件路径
    """
    from .lora_config import SAMPLE_TRAIN_DATA, TRAVEL_INSTRUCTION_TEMPLATE

    if not template:
        template = TRAVEL_INSTRUCTION_TEMPLATE

    # 加载原始数据
    if input_path and os.path.exists(input_path):
        raw_data = _load_raw_data(input_path)
    else:
        raw_data = SAMPLE_TRAIN_DATA

    # 转换为训练格式
    train_data = []
    for item in raw_data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")

        # 构建训练样本
        if input_text:
            formatted = f"{template.format(instruction=instruction, response=output)}"
        else:
            formatted = template.format(instruction=instruction, response=output)

        train_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "text": formatted,
        })

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    print(f"[DataPreprocess] 处理完成: {len(train_data)} 条样本")
    print(f"[DataPreprocess] 输出文件: {output_path}")
    return output_path


def _load_raw_data(path: str) -> list:
    """加载原始数据文件"""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    elif ext == ".csv":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    "instruction": row.get("instruction", ""),
                    "input": row.get("input", ""),
                    "output": row.get("output", ""),
                })
        return data

    elif ext == ".txt":
        # 每行一条指令-回答对，用 \t 分隔
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "\t" in line:
                    parts = line.split("\t", 1)
                    data.append({
                        "instruction": parts[0],
                        "input": "",
                        "output": parts[1],
                    })
        return data

    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def augment_travel_data(data_path: str, output_path: str) -> str:
    """
    数据增强：生成更多的训练样本变体
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    augmented = []
    for item in data:
        augmented.append(item)

        # 变体1：改变问法
        instruction = item["instruction"]
        variants = _generate_instruction_variants(instruction)
        for variant in variants:
            augmented.append({
                "instruction": variant,
                "input": item.get("input", ""),
                "output": item["output"],
                "text": item.get("text", ""),
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)

    print(f"[DataAugment] 增强完成: {len(data)} → {len(augmented)} 条样本")
    return output_path


def _generate_instruction_variants(instruction: str) -> list:
    """生成指令变体"""
    prefixes = [
        "请问",
        "帮我",
        "能不能",
        "请帮我",
        "我想知道",
    ]
    variants = []
    for prefix in prefixes:
        variants.append(f"{prefix}{instruction}")
    return variants