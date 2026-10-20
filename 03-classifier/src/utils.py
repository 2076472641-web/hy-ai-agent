# -*- coding: utf-8 -*-
"""
项目通用工具模块
功能：
1. load_config()：加载configs/config.yaml（PyYAML）
2. setup_logger()：配置loguru日志器，输出到控制台+logs目录文件
3. seed_everything(seed=42)：固定随机种子（random/numpy/torch）
4. get_category_mappings()：返回十二大类的id<->name映射
5. Timer上下文管理器：用于统计代码块耗时
6. lazy_property装饰器：惰性计算属性
"""

import os
import sys
import time
import random
import functools
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional, Any

import yaml
from loguru import logger
import numpy as np

from src import PROJECT_ROOT, CONFIG_DIR, LOG_DIR


# ============================================================================
# 1. 配置加载
# ============================================================================

_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[Union[str, Path]] = None, force_reload: bool = False) -> Dict[str, Any]:
    """
    加载configs/config.yaml配置文件

    Args:
        config_path: 配置文件路径，为None时使用默认路径 configs/config.yaml
        force_reload: 是否强制重新加载（忽略缓存）

    Returns:
        Dict: 配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置文件格式错误
    """
    global _CONFIG_CACHE

    if not force_reload and _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    if config_path is None:
        config_path = CONFIG_DIR / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件YAML格式错误：{e}") from e

    if config is None:
        config = {}
        logger.warning("配置文件为空，使用默认配置")

    _CONFIG_CACHE = config
    logger.info(f"配置文件加载成功：{config_path}")
    return config


# ============================================================================
# 2. 日志配置
# ============================================================================

def setup_logger(
    log_dir: Optional[Union[str, Path]] = None,
    log_level: str = "INFO",
    console_output: bool = True,
    file_output: bool = True,
    rotation: str = "10 MB",
    retention: str = "7 days",
    log_format: Optional[str] = None,
) -> None:
    """
    配置loguru日志器，输出到控制台+logs目录文件

    Args:
        log_dir: 日志文件目录，为None时使用项目默认 logs/
        log_level: 日志级别，DEBUG/INFO/WARNING/ERROR/CRITICAL
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        rotation: 日志文件轮转规则（如"10 MB"、"00:00"、"1 week"）
        retention: 日志文件保留时长
        log_format: 自定义日志格式，为None时使用默认格式
    """
    # 移除loguru默认的handler
    logger.remove()

    if log_dir is None:
        log_dir = LOG_DIR
    else:
        log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 默认日志格式
    if log_format is None:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    # 控制台输出
    if console_output:
        logger.add(
            sys.stderr,
            level=log_level,
            format=log_format,
            colorize=True,
            enqueue=True,
        )

    # 文件输出（按日志级别分文件）
    if file_output:
        # 所有级别日志
        logger.add(
            str(log_dir / "all_{time:YYYY-MM-DD}.log"),
            level=log_level,
            format=log_format,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            enqueue=True,
        )
        # 仅ERROR及以上级别
        logger.add(
            str(log_dir / "error_{time:YYYY-MM-DD}.log"),
            level="ERROR",
            format=log_format,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            enqueue=True,
        )

    logger.info(f"日志器配置完成 | 级别: {log_level} | 控制台: {console_output} | 文件: {file_output}")
    logger.info(f"日志文件目录: {log_dir}")


# ============================================================================
# 3. 随机种子固定
# ============================================================================

def seed_everything(seed: int = 42) -> None:
    """
    固定所有随机种子，保证实验可复现

    固定以下库的随机种子：
    - Python内置 random
    - NumPy
    - PyTorch（如已安装）
    - Python hash 随机化

    Args:
        seed: 随机种子值，默认42
    """
    # Python内置随机
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Python hash随机化（PYTHONHASHSEED）
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch（如已安装）
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # CuDNN确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"PyTorch随机种子已固定: {seed}")
    except ImportError:
        logger.info("PyTorch未安装，跳过torch随机种子设置")

    logger.info(f"全局随机种子已固定: {seed}")


# ============================================================================
# 4. 十二大类id<->name映射
# ============================================================================

# 默认十二大类商品名称（顺序与id对应）
DEFAULT_CATEGORY_NAMES: List[str] = [
    "服装鞋帽",
    "数码家电",
    "美妆护肤",
    "食品生鲜",
    "家居家装",
    "母婴用品",
    "运动户外",
    "图书文娱",
    "珠宝饰品",
    "汽车用品",
    "宠物用品",
    "医药保健",
]


def get_category_mappings(
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[int, str], Dict[str, int], List[str]]:
    """
    返回十二大类的id<->name映射

    优先级：配置文件categories配置 > 默认DEFAULT_CATEGORY_NAMES

    Args:
        config: 已加载的配置字典，为None时自动调用load_config()

    Returns:
        Tuple: (id_to_name, name_to_id, category_names_list)
            - id_to_name: Dict[int, str]，类目ID->名称
            - name_to_id: Dict[str, int]，类目名称->ID
            - category_names_list: List[str]，按ID顺序排列的类目名称列表
    """
    if config is None:
        try:
            config = load_config()
        except (FileNotFoundError, ValueError):
            config = {}

    categories_cfg = config.get("categories", []) if config else []

    if categories_cfg:
        # 从配置文件构建
        id_to_name: Dict[int, str] = {}
        for cat in categories_cfg:
            cat_id = cat.get("id")
            cat_name = cat.get("name")
            if cat_id is not None and cat_name:
                id_to_name[int(cat_id)] = cat_name
        # 按ID排序
        sorted_ids = sorted(id_to_name.keys())
        category_names = [id_to_name[i] for i in sorted_ids]
    else:
        # 使用默认
        category_names = DEFAULT_CATEGORY_NAMES.copy()
        id_to_name = {i: name for i, name in enumerate(category_names)}

    name_to_id = {name: idx for idx, name in id_to_name.items()}

    return id_to_name, name_to_id, category_names


# ============================================================================
# 5. Timer上下文管理器
# ============================================================================

@contextmanager
def Timer(name: str = "代码块", verbose: bool = True, precision: int = 4):
    """
    上下文管理器：统计代码块耗时

    使用示例：
        >>> with Timer("数据处理"):
        ...     time.sleep(1)
        [Timer] 数据处理 耗时: 1.0002 秒

        >>> with Timer("计算") as t:
        ...     x = sum(range(1000))
        >>> print(t.elapsed)
        0.000123

    Args:
        name: 代码块名称，用于日志输出
        verbose: 是否在退出时打印日志
        precision: 秒数小数位数
    """
    start = time.perf_counter()
    result = {"elapsed": 0.0, "name": name}

    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result["elapsed"] = elapsed

        if verbose:
            logger.info(
                f"[Timer] {name} 耗时: {elapsed:.{precision}f} 秒"
            )


# ============================================================================
# 6. lazy_property装饰器
# ============================================================================

def lazy_property(func):
    """
    惰性计算属性装饰器

    被装饰的属性只会在第一次访问时计算，之后缓存结果。
    适用于计算开销大且值不变的属性。

    使用示例：
        >>> class MyClass:
        ...     @lazy_property
        ...     def expensive_data(self):
        ...         print("计算中...")
        ...         return sum(range(1000000))
        >>> obj = MyClass()
        >>> obj.expensive_data  # 第一次：计算中...
        499999500000
        >>> obj.expensive_data  # 第二次：直接返回缓存
        499999500000

    Args:
        func: 被装饰的方法（应为无参实例方法）

    Returns:
        装饰后的属性描述符
    """
    attr_name = f"_lazy_{func.__name__}"

    @property
    @functools.wraps(func)
    def wrapper(self):
        # 如果缓存不存在，计算并缓存
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return wrapper


# ============================================================================
# 便捷函数：确保项目基础目录存在
# ============================================================================

def ensure_project_dirs() -> None:
    """
    确保项目所需的所有数据目录存在
    包含：data/raw、data/processed、data/models、logs
    """
    from src import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR

    dirs = [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"项目目录已确保存在: {[str(d) for d in dirs]}")
