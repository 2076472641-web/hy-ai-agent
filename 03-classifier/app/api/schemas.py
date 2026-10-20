# -*- coding: utf-8 -*-
"""
Pydantic数据模型定义模块
定义API请求和响应的数据结构，确保类型安全和数据验证。
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# 十二大类商品类目映射
# ============================================================================
CATEGORY_NAMES = [
    "服装鞋帽",   # 0
    "数码家电",   # 1
    "美妆护肤",   # 2
    "食品生鲜",   # 3
    "家居家装",   # 4
    "母婴用品",   # 5
    "运动户外",   # 6
    "图书文娱",   # 7
    "珠宝饰品",   # 8
    "汽车用品",   # 9
    "宠物用品",   # 10
    "医药保健",   # 11
]

CATEGORY_ID_TO_NAME = {i: name for i, name in enumerate(CATEGORY_NAMES)}
CATEGORY_NAME_TO_ID = {name: i for i, name in enumerate(CATEGORY_NAMES)}


# ============================================================================
# 单条分类相关模型
# ============================================================================

class SingleClassifyRequest(BaseModel):
    """
    单商品分类请求模型

    Attributes:
        title: 商品标题（必填）
        detail: 商品详情描述（可选，默认为空字符串）
        top_k: 返回Top K个预测结果，默认3，范围1-12
        model_name: 指定使用的模型名称，默认 "bert_distilled"
    """
    title: str = Field(..., min_length=1, max_length=200, description="商品标题")
    detail: str = Field(default="", max_length=2000, description="商品详情描述")
    top_k: int = Field(default=3, ge=1, le=12, description="返回Top K预测结果")
    model_name: str = Field(default="bert_distilled", description="指定使用的模型名称")


class PredictionItem(BaseModel):
    """
    单个预测结果项

    Attributes:
        category_id: 预测类目ID (0-11)
        category_name: 预测类目名称
        confidence: 预测置信度 (0-1)
    """
    category_id: int = Field(..., ge=0, le=11, description="预测类目ID")
    category_name: str = Field(..., description="预测类目名称")
    confidence: float = Field(..., ge=0.0, le=1.0, description="预测置信度")


class SingleClassifyResponse(BaseModel):
    """
    单商品分类响应模型

    Attributes:
        category_id: 最优预测类目ID
        category_name: 最优预测类目名称
        confidence: 最优预测置信度
        top_predictions: Top K个预测结果列表
    """
    category_id: int = Field(..., ge=0, le=11, description="最优预测类目ID")
    category_name: str = Field(..., description="最优预测类目名称")
    confidence: float = Field(..., ge=0.0, le=1.0, description="最优预测置信度")
    top_predictions: List[PredictionItem] = Field(..., description="Top K预测结果列表")


# ============================================================================
# 批量分类相关模型
# ============================================================================

class BatchItem(BaseModel):
    """
    批量分类中的单条商品项

    Attributes:
        title: 商品标题
        detail: 商品详情描述（可选）
    """
    title: str = Field(..., min_length=1, max_length=200, description="商品标题")
    detail: str = Field(default="", max_length=2000, description="商品详情描述")


class BatchClassifyRequest(BaseModel):
    """
    批量商品分类请求模型

    Attributes:
        items: 待分类的商品列表（最多64条）
        top_k: 每条返回Top K个预测结果，默认3
        model_name: 指定使用的模型名称，默认 "bert_distilled"
    """
    items: List[BatchItem] = Field(..., description="待分类商品列表，最大64条")
    top_k: int = Field(default=3, ge=1, le=12, description="每条返回Top K预测结果")
    model_name: str = Field(default="bert_distilled", description="指定使用的模型名称")

    @field_validator("items")
    @classmethod
    def check_batch_size(cls, v: List[BatchItem]) -> List[BatchItem]:
        """校验批量大小不超过64条"""
        if len(v) > 64:
            raise ValueError(f"批量分类最大支持64条，当前提交了 {len(v)} 条")
        if len(v) == 0:
            raise ValueError("批量分类请求不能为空")
        return v


class BatchResultItem(BaseModel):
    """
    批量分类中的单条结果

    Attributes:
        index: 请求中对应的索引（从0开始）
        category_id: 最优预测类目ID
        category_name: 最优预测类目名称
        confidence: 最优预测置信度
        top_predictions: Top K个预测结果
    """
    index: int = Field(..., ge=0, description="请求中的索引位置")
    category_id: int = Field(..., ge=0, le=11, description="最优预测类目ID")
    category_name: str = Field(..., description="最优预测类目名称")
    confidence: float = Field(..., ge=0.0, le=1.0, description="最优预测置信度")
    top_predictions: List[PredictionItem] = Field(..., description="Top K预测结果列表")


class BatchClassifyResponse(BaseModel):
    """
    批量商品分类响应模型

    Attributes:
        results: 分类结果列表
        total_count: 处理的总条数
        processing_time_ms: 总处理耗时（毫秒）
    """
    results: List[BatchResultItem] = Field(..., description="分类结果列表")
    total_count: int = Field(..., ge=0, description="处理的总条数")
    processing_time_ms: float = Field(..., ge=0.0, description="总处理耗时(毫秒)")


# ============================================================================
# 健康检查与模型信息相关模型
# ============================================================================

class HealthResponse(BaseModel):
    """
    服务健康检查响应模型

    Attributes:
        status: 服务状态 ("healthy" / "degraded" / "unhealthy")
        model_name: 当前加载的模型名称
        version: 服务版本号
        uptime_seconds: 服务已运行秒数
    """
    status: str = Field(..., description="服务状态: healthy/degraded/unhealthy")
    model_name: str = Field(..., description="当前加载的模型名称")
    version: str = Field(..., description="服务版本号")
    uptime_seconds: float = Field(..., ge=0.0, description="服务已运行秒数")


class ModelInfoResponse(BaseModel):
    """
    模型详细信息响应模型

    Attributes:
        model_name: 模型名称标识
        accuracy: 模型准确率（0-1）
        f1_macro: 宏平均F1分数（0-1）
        size_mb: 模型文件大小（MB）
        inference_time_ms: 单样本平均推理耗时（毫秒）
        description: 模型描述说明
    """
    model_name: str = Field(..., description="模型名称标识")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="模型准确率")
    f1_macro: float = Field(..., ge=0.0, le=1.0, description="宏平均F1分数")
    size_mb: float = Field(..., ge=0.0, description="模型文件大小(MB)")
    inference_time_ms: float = Field(..., ge=0.0, description="单样本平均推理耗时(毫秒)")
    description: str = Field(..., description="模型描述说明")


# ============================================================================
# 纠错反馈相关模型
# ============================================================================

class FeedbackRequest(BaseModel):
    """
    运营纠错反馈请求模型

    Attributes:
        title: 商品标题
        detail: 商品详情描述（可选）
        predicted_category_id: 模型预测的类目ID
        correct_category_id: 运营纠正的正确类目ID
        model_name: 使用的模型名称（可选）
        remark: 备注说明（可选）
    """
    title: str = Field(..., min_length=1, max_length=200, description="商品标题")
    detail: str = Field(default="", max_length=2000, description="商品详情描述")
    predicted_category_id: int = Field(..., ge=0, le=11, description="模型预测的类目ID")
    correct_category_id: int = Field(..., ge=0, le=11, description="正确的类目ID")
    model_name: str = Field(default="", description="使用的模型名称")
    remark: str = Field(default="", max_length=500, description="备注说明")


class FeedbackRecord(BaseModel):
    """
    单条纠错反馈记录

    Attributes:
        id: 记录ID
        timestamp: 反馈时间戳（ISO格式）
        title: 商品标题
        detail: 商品详情
        predicted_category_id: 预测类目ID
        predicted_category_name: 预测类目名称
        correct_category_id: 正确类目ID
        correct_category_name: 正确类目名称
        model_name: 使用的模型名称
        remark: 备注
    """
    id: int = Field(..., description="记录ID")
    timestamp: str = Field(..., description="反馈时间戳")
    title: str = Field(..., description="商品标题")
    detail: str = Field(default="", description="商品详情")
    predicted_category_id: int = Field(..., description="预测类目ID")
    predicted_category_name: str = Field(..., description="预测类目名称")
    correct_category_id: int = Field(..., description="正确类目ID")
    correct_category_name: str = Field(..., description="正确类目名称")
    model_name: str = Field(default="", description="使用的模型名称")
    remark: str = Field(default="", description="备注")


class FeedbackListResponse(BaseModel):
    """
    纠错反馈记录列表响应模型

    Attributes:
        records: 反馈记录列表
        total_count: 总记录数
    """
    records: List[FeedbackRecord] = Field(..., description="反馈记录列表")
    total_count: int = Field(..., ge=0, description="总记录数")


# ============================================================================
# 工具函数
# ============================================================================

def get_category_name(category_id: int) -> str:
    """
    根据类目ID获取类目名称

    Args:
        category_id: 类目ID (0-11)

    Returns:
        str: 类目名称
    """
    return CATEGORY_ID_TO_NAME.get(category_id, f"未知类目_{category_id}")


def get_category_id(category_name: str) -> Optional[int]:
    """
    根据类目名称获取类目ID

    Args:
        category_name: 类目名称

    Returns:
        Optional[int]: 类目ID，不存在返回None
    """
    return CATEGORY_NAME_TO_ID.get(category_name)
