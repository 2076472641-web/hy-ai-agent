# -*- coding: utf-8 -*-
"""
优店汇·商品类目智能研判分类系统
基于NLP与机器学习的小程序商品自动化分类系统
"""

__version__ = "1.0.0"
__author__ = "NLP工程团队"

# 项目路径常量
from pathlib import Path
import os

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

# 配置目录
CONFIG_DIR = PROJECT_ROOT / "configs"

# 日志目录
LOG_DIR = PROJECT_ROOT / "logs"

# 确保目录存在
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)
