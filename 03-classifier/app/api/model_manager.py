# -*- coding: utf-8 -*-
"""
模型管理器模块
负责模型的懒加载、热切换、版本管理和内存管理。
在真实模型未加载时提供基于关键词匹配的模拟分类器（Fallback）兜底，
确保服务始终可用。
"""

import sys
import os
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import yaml
import numpy as np
from loguru import logger

# 添加项目根目录到系统路径，确保可以导入src模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.schemas import CATEGORY_NAMES, CATEGORY_ID_TO_NAME, get_category_name


# ============================================================================
# 配置加载
# ============================================================================

def load_config() -> Dict[str, Any]:
    """
    加载项目配置文件 configs/config.yaml

    Returns:
        Dict[str, Any]: 配置字典
    """
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config or {}
    except Exception as e:
        logger.warning(f"加载配置文件失败: {e}，使用默认配置")
        return {}


# ============================================================================
# 模型信息数据类
# ============================================================================

@dataclass
class ModelInfo:
    """
    模型元信息数据类

    Attributes:
        model_name: 模型名称标识
        display_name: 模型展示名称
        description: 模型描述
        accuracy: 模型准确率
        f1_macro: 宏平均F1分数
        size_mb: 模型大小(MB)
        inference_time_ms: 推理耗时(毫秒/条)
        is_loaded: 是否已加载到内存
        load_time: 加载时间戳
        last_used_time: 最后使用时间戳
        use_count: 使用次数
    """
    model_name: str
    display_name: str = ""
    description: str = ""
    accuracy: float = 0.0
    f1_macro: float = 0.0
    size_mb: float = 0.0
    inference_time_ms: float = 0.0
    is_loaded: bool = False
    load_time: float = 0.0
    last_used_time: float = 0.0
    use_count: int = 0


# ============================================================================
# 模拟分类器（Fallback）- 基于关键词匹配
# ============================================================================

class FallbackClassifier:
    """
    模拟分类器（兜底方案）

    当真实模型未加载或加载失败时，基于关键词匹配进行分类。
    使用config.yaml中定义的类目关键词进行匹配，保证服务始终可用。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化模拟分类器

        Args:
            config: 配置字典，包含categories关键词信息
        """
        self.model_name = "fallback_keyword"
        self.is_trained = True
        self.num_classes = 12

        # 从配置中加载类目关键词
        self.category_keywords: Dict[int, List[str]] = {}
        categories_cfg = config.get("categories", [])
        if categories_cfg:
            for cat in categories_cfg:
                cat_id = cat.get("id", -1)
                keywords = cat.get("keywords", [])
                if 0 <= cat_id < 12:
                    self.category_keywords[cat_id] = keywords
        else:
            # 使用默认关键词
            self._init_default_keywords()

        logger.info(f"[FallbackClassifier] 初始化完成，共加载 {len(self.category_keywords)} 个类目的关键词")

    def _init_default_keywords(self):
        """加载默认关键词（配置缺失时使用）"""
        self.category_keywords = {
            0: ["T恤", "衬衫", "外套", "裤子", "裙子", "鞋子", "帽子", "袜子", "羽绒服", "卫衣", "牛仔裤", "运动鞋"],
            1: ["手机", "电脑", "耳机", "相机", "电视", "冰箱", "洗衣机", "空调", "平板", "充电器", "充电宝"],
            2: ["面膜", "口红", "粉底", "面霜", "精华", "香水", "眉笔", "眼影", "睫毛膏", "卸妆", "洁面", "防晒霜"],
            3: ["零食", "饼干", "巧克力", "坚果", "水果", "蔬菜", "肉类", "海鲜", "牛奶", "饮料", "咖啡", "茶叶"],
            4: ["沙发", "床", "桌子", "椅子", "衣柜", "灯具", "窗帘", "地毯", "枕头", "被子", "收纳"],
            5: ["奶粉", "纸尿裤", "奶瓶", "婴儿车", "玩具", "童装", "孕装", "湿巾", "辅食", "安全座椅", "童床"],
            6: ["篮球", "足球", "羽毛球", "跑步机", "瑜伽垫", "哑铃", "自行车", "帐篷", "登山鞋", "运动服"],
            7: ["小说", "教材", "辅导书", "杂志", "文具", "乐器", "拼图", "桌游", "电子琴", "吉他", "画笔"],
            8: ["项链", "戒指", "耳环", "手链", "手镯", "手表", "胸针", "发饰", "眼镜", "黄金", "钻石"],
            9: ["汽车坐垫", "脚垫", "香水", "行车记录仪", "导航", "轮胎", "机油", "雨刷", "车蜡"],
            10: ["狗粮", "猫粮", "猫砂", "狗窝", "猫爬架", "玩具", "牵引绳", "衣服", "食盆", "航空箱"],
            11: ["维生素", "钙片", "蛋白粉", "感冒药", "创可贴", "体温计", "血压计", "血糖仪", "口罩"],
        }

    def _match_text(self, text: str) -> np.ndarray:
        """
        对单条文本进行关键词匹配，返回各类目得分

        Args:
            text: 待分类文本

        Returns:
            np.ndarray: 各类目得分数组 (12,)
        """
        scores = np.zeros(self.num_classes, dtype=np.float32)

        if not text:
            # 空文本，返回均匀分布
            scores[:] = 1.0 / self.num_classes
            return scores

        text_lower = text.lower()

        for cat_id in range(self.num_classes):
            keywords = self.category_keywords.get(cat_id, [])
            match_count = 0
            for kw in keywords:
                if kw.lower() in text_lower:
                    match_count += 1
            # 得分 = 匹配的关键词数量，加一个平滑项避免全零
            scores[cat_id] = match_count + 0.01

        # 归一化为概率
        total = scores.sum()
        if total > 0:
            scores = scores / total
        else:
            scores[:] = 1.0 / self.num_classes

        return scores

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        预测批次文本的类别概率分布

        Args:
            texts: 待分类文本列表

        Returns:
            np.ndarray: 概率矩阵 (n_samples, 12)
        """
        results = []
        for text in texts:
            # 合并title和detail
            combined = text if isinstance(text, str) else str(text)
            results.append(self._match_text(combined))
        return np.array(results)

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        预测批次文本的类别ID

        Args:
            texts: 待分类文本列表

        Returns:
            np.ndarray: 预测类别ID数组 (n_samples,)
        """
        probas = self.predict_proba(texts)
        return np.argmax(probas, axis=1)


# ============================================================================
# 模型管理器主类
# ============================================================================

class ModelManager:
    """
    模型管理器

    核心功能：
    1. 懒加载：模型仅在首次使用时加载
    2. 热切换：支持运行时切换当前使用的模型
    3. 内存管理：跟踪模型使用情况，支持卸载闲置模型
    4. 版本管理：记录模型元信息
    5. 兜底机制：真实模型不可用时自动切换到Fallback
    """

    # 支持的模型列表
    SUPPORTED_MODELS = {
        "random_forest": {
            "display_name": "TF-IDF + 随机森林",
            "description": "传统机器学习基线模型，TF-IDF特征 + 随机森林分类器，训练快，可解释性好",
            "accuracy": 0.82,
            "f1_macro": 0.81,
            "size_mb": 45.0,
            "inference_time_ms": 15.0,
        },
        "fasttext": {
            "display_name": "FastText",
            "description": "Facebook高效文本分类模型，基于字符n-gram，推理速度极快，适合高并发",
            "accuracy": 0.85,
            "f1_macro": 0.84,
            "size_mb": 120.0,
            "inference_time_ms": 3.0,
        },
        "bert": {
            "display_name": "BERT (bert-base-chinese)",
            "description": "谷歌BERT中文预训练模型微调，分类精度最高，但推理较慢",
            "accuracy": 0.93,
            "f1_macro": 0.925,
            "size_mb": 410.0,
            "inference_time_ms": 65.0,
        },
        "bert_distilled": {
            "display_name": "Distilled BERT",
            "description": "知识蒸馏后的轻量化BERT模型，精度接近原生BERT，推理速度提升3倍",
            "accuracy": 0.91,
            "f1_macro": 0.905,
            "size_mb": 180.0,
            "inference_time_ms": 22.0,
        },
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化模型管理器

        Args:
            config: 配置字典，None则从配置文件加载
        """
        self.config = config or load_config()
        self._lock = threading.RLock()  # 线程锁，确保并发安全

        # 模型存储
        self._models: Dict[str, Any] = {}  # model_name -> model实例
        self._model_infos: Dict[str, ModelInfo] = {}  # model_name -> ModelInfo

        # 当前活动模型
        service_cfg = self.config.get("service", {})
        default_model = service_cfg.get("default_model", "bert_distilled")
        if default_model not in self.SUPPORTED_MODELS:
            default_model = "bert_distilled"
        self._current_model_name: str = default_model

        # 模型路径配置
        paths_cfg = self.config.get("paths", {})
        self._models_dir = PROJECT_ROOT / paths_cfg.get("models", "data/models")
        self._models_dir.mkdir(parents=True, exist_ok=True)

        # 初始化模型信息
        self._init_model_infos()

        # 初始化Fallback分类器（始终可用）
        self._fallback = FallbackClassifier(self.config)
        self._fallback_info = ModelInfo(
            model_name="fallback_keyword",
            display_name="关键词匹配分类器",
            description="基于类目关键词匹配的兜底分类器，在真实模型未加载时使用",
            accuracy=0.55,
            f1_macro=0.52,
            size_mb=0.001,
            inference_time_ms=0.5,
            is_loaded=True,
            load_time=time.time(),
            last_used_time=time.time(),
        )

        # 启动时加载默认模型（失败则使用fallback）
        try:
            self.load_model(self._current_model_name)
            logger.info(f"[ModelManager] 默认模型 {self._current_model_name} 加载成功")
        except Exception as e:
            logger.warning(
                f"[ModelManager] 默认模型 {self._current_model_name} 加载失败: {e}，"
                f"将使用 Fallback 分类器兜底"
            )
            self._current_model_name = "fallback_keyword"

    def _init_model_infos(self):
        """初始化所有支持模型的元信息"""
        for model_name, meta in self.SUPPORTED_MODELS.items():
            info = ModelInfo(
                model_name=model_name,
                display_name=meta["display_name"],
                description=meta["description"],
                accuracy=meta["accuracy"],
                f1_macro=meta["f1_macro"],
                size_mb=meta["size_mb"],
                inference_time_ms=meta["inference_time_ms"],
            )
            self._model_infos[model_name] = info
        logger.info(f"[ModelManager] 已注册 {len(self._model_infos)} 个模型: {list(self._model_infos.keys())}")

    # ----------------------------------------------------------------
    # 模型加载与卸载
    # ----------------------------------------------------------------

    def _get_model_path(self, model_name: str) -> Path:
        """
        获取模型文件路径

        Args:
            model_name: 模型名称

        Returns:
            Path: 模型目录路径
        """
        # 不同模型可能存储在不同子目录
        model_dir_map = {
            "random_forest": "random_forest",
            "fasttext": "fasttext",
            "bert": "bert_classifier",
            "bert_distilled": "bert_distilled",
        }
        subdir = model_dir_map.get(model_name, model_name)
        return self._models_dir / subdir

    def load_model(self, model_name: str) -> bool:
        """
        加载指定模型到内存（懒加载，支持并发安全）

        Args:
            model_name: 模型名称

        Returns:
            bool: 是否加载成功

        Raises:
            ValueError: 模型名称不支持
        """
        if model_name == "fallback_keyword":
            return True  # fallback始终已加载

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"不支持的模型: {model_name}，可用模型: {list(self.SUPPORTED_MODELS.keys())}"
            )

        with self._lock:
            # 已加载则直接返回
            if model_name in self._models:
                logger.info(f"[ModelManager] 模型 {model_name} 已加载，跳过")
                return True

            model_path = self._get_model_path(model_name)
            logger.info(f"[ModelManager] 开始加载模型: {model_name}，路径: {model_path}")

            try:
                model = self._instantiate_model(model_name, model_path)

                # 加载成功
                self._models[model_name] = model
                info = self._model_infos[model_name]
                info.is_loaded = True
                info.load_time = time.time()
                info.last_used_time = time.time()

                # 更新实际模型大小
                actual_size = self._calculate_model_size(model_path)
                if actual_size > 0:
                    info.size_mb = actual_size

                logger.info(f"[ModelManager] 模型 {model_name} 加载成功")
                return True

            except Exception as e:
                logger.error(f"[ModelManager] 模型 {model_name} 加载失败: {e}")
                raise

    def _instantiate_model(self, model_name: str, model_path: Path):
        """
        根据模型类型实例化并加载模型

        Args:
            model_name: 模型名称
            model_path: 模型路径

        Returns:
            加载后的模型实例
        """
        if not model_path.exists():
            raise FileNotFoundError(
                f"模型目录不存在: {model_path}，请先训练并保存模型"
            )

        metadata_file = model_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"模型元数据不存在: {metadata_file}，模型可能未正确保存"
            )

        if model_name == "random_forest":
            from src.models.random_forest import RandomForestClassifier
            model = RandomForestClassifier()
            model.load(model_path)

        elif model_name == "fasttext":
            from src.models.fasttext_model import FastTextClassifier
            model = FastTextClassifier()
            model.load(model_path)

        elif model_name in ("bert", "bert_distilled"):
            from src.models.bert_classifier import BERTClassifier
            model = BERTClassifier(model_name=model_name)
            model.load(model_path)

        else:
            raise ValueError(f"未知模型类型: {model_name}")

        # 设置类目名称
        model.set_class_names(CATEGORY_NAMES)
        return model

    def _calculate_model_size(self, model_path: Path) -> float:
        """
        计算模型目录的实际大小(MB)

        Args:
            model_path: 模型目录路径

        Returns:
            float: 大小(MB)
        """
        try:
            total_bytes = 0
            for f in model_path.rglob("*"):
                if f.is_file():
                    total_bytes += f.stat().st_size
            return round(total_bytes / (1024 * 1024), 2)
        except Exception:
            return 0.0

    def unload_model(self, model_name: str) -> bool:
        """
        卸载指定模型，释放内存

        Args:
            model_name: 模型名称

        Returns:
            bool: 是否卸载成功
        """
        with self._lock:
            if model_name == "fallback_keyword":
                logger.warning("[ModelManager] Fallback分类器不允许卸载")
                return False

            if model_name not in self._models:
                logger.info(f"[ModelManager] 模型 {model_name} 未加载，无需卸载")
                return False

            # 不允许卸载当前正在使用的模型
            if model_name == self._current_model_name:
                logger.warning(
                    f"[ModelManager] 模型 {model_name} 是当前活动模型，无法卸载，"
                    f"请先切换到其他模型"
                )
                return False

            model = self._models.pop(model_name)
            info = self._model_infos.get(model_name)
            if info:
                info.is_loaded = False

            # 显式删除引用，帮助GC
            del model
            import gc
            gc.collect()

            logger.info(f"[ModelManager] 模型 {model_name} 已卸载，内存已释放")
            return True

    # ----------------------------------------------------------------
    # 模型切换与获取
    # ----------------------------------------------------------------

    def switch_model(self, model_name: str) -> bool:
        """
        热切换当前使用的模型

        Args:
            model_name: 目标模型名称

        Returns:
            bool: 是否切换成功
        """
        if model_name != "fallback_keyword" and model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"不支持的模型: {model_name}，可用模型: {list(self.SUPPORTED_MODELS.keys())}"
            )

        with self._lock:
            # 如果目标模型未加载，先加载
            if model_name != "fallback_keyword" and model_name not in self._models:
                logger.info(f"[ModelManager] 目标模型 {model_name} 未加载，先执行加载")
                try:
                    self.load_model(model_name)
                except Exception as e:
                    logger.error(
                        f"[ModelManager] 切换失败，模型 {model_name} 加载异常: {e}"
                    )
                    return False

            old_model = self._current_model_name
            self._current_model_name = model_name
            logger.info(f"[ModelManager] 模型切换: {old_model} -> {model_name}")
            return True

    def get_current_model(self):
        """
        获取当前活动模型实例

        Returns:
            模型实例（优先返回真实模型，否则返回fallback）
        """
        with self._lock:
            model_name = self._current_model_name

            if model_name == "fallback_keyword":
                return self._fallback

            if model_name in self._models:
                model = self._models[model_name]
                # 更新使用统计
                info = self._model_infos.get(model_name)
                if info:
                    info.last_used_time = time.time()
                    info.use_count += 1
                return model

            # 当前模型未加载，返回fallback并记录警告
            logger.warning(
                f"[ModelManager] 当前模型 {model_name} 未加载，临时使用Fallback"
            )
            return self._fallback

    def get_current_model_name(self) -> str:
        """获取当前活动模型名称"""
        with self._lock:
            return self._current_model_name

    def get_model(self, model_name: str):
        """
        获取指定模型（如果已加载），未加载则返回fallback

        Args:
            model_name: 模型名称

        Returns:
            模型实例
        """
        with self._lock:
            if model_name == "fallback_keyword":
                return self._fallback

            if model_name in self._models:
                model = self._models[model_name]
                info = self._model_infos.get(model_name)
                if info:
                    info.last_used_time = time.time()
                    info.use_count += 1
                return model

            # 尝试加载，失败则fallback
            try:
                self.load_model(model_name)
                return self._models.get(model_name, self._fallback)
            except Exception as e:
                logger.warning(
                    f"[ModelManager] 获取模型 {model_name} 失败: {e}，使用Fallback"
                )
                return self._fallback

    # ----------------------------------------------------------------
    # 模型信息查询
    # ----------------------------------------------------------------

    def get_all_model_infos(self) -> List[Dict[str, Any]]:
        """
        获取所有支持模型的详细信息列表

        Returns:
            List[Dict]: 模型信息列表
        """
        result = []
        # 包含fallback
        infos = dict(self._model_infos)
        infos["fallback_keyword"] = self._fallback_info

        for model_name, info in sorted(infos.items()):
            result.append({
                "model_name": info.model_name,
                "display_name": info.display_name,
                "description": info.description,
                "accuracy": info.accuracy,
                "f1_macro": info.f1_macro,
                "size_mb": info.size_mb,
                "inference_time_ms": info.inference_time_ms,
                "is_loaded": info.is_loaded,
                "is_current": (model_name == self._current_model_name),
                "use_count": info.use_count,
            })
        return result

    def get_loaded_models(self) -> List[str]:
        """获取当前已加载到内存的模型名称列表"""
        with self._lock:
            loaded = list(self._models.keys())
            loaded.append("fallback_keyword")  # fallback始终加载
            return loaded

    # ----------------------------------------------------------------
    # 内存管理：闲置模型清理
    # ----------------------------------------------------------------

    def cleanup_idle_models(self, idle_threshold_seconds: float = 3600.0) -> int:
        """
        清理闲置超过阈值的模型（Fallback和当前模型不会被清理）

        Args:
            idle_threshold_seconds: 闲置时间阈值(秒)，默认3600秒(1小时)

        Returns:
            int: 清理的模型数量
        """
        cleaned_count = 0
        now = time.time()

        with self._lock:
            model_names = list(self._models.keys())
            for model_name in model_names:
                if model_name == self._current_model_name:
                    continue  # 跳过当前模型

                info = self._model_infos.get(model_name)
                if info is None:
                    continue

                idle_time = now - info.last_used_time
                if idle_time >= idle_threshold_seconds:
                    logger.info(
                        f"[ModelManager] 清理闲置模型 {model_name}，"
                        f"闲置时间: {idle_time:.0f}秒"
                    )
                    model = self._models.pop(model_name)
                    info.is_loaded = False
                    del model
                    cleaned_count += 1

            if cleaned_count > 0:
                import gc
                gc.collect()

        return cleaned_count


# ============================================================================
# 全局单例
# ============================================================================

_model_manager_singleton: Optional[ModelManager] = None
_singleton_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    """
    获取模型管理器全局单例

    Returns:
        ModelManager: 模型管理器实例
    """
    global _model_manager_singleton
    if _model_manager_singleton is None:
        with _singleton_lock:
            if _model_manager_singleton is None:
                _model_manager_singleton = ModelManager()
    return _model_manager_singleton
