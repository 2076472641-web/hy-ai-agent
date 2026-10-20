# -*- coding: utf-8 -*-
"""
FastAPI主应用模块
提供商品类目智能分类的RESTful API服务，包括：
- 健康检查接口
- 模型管理接口
- 单条/批量分类接口
- 纠错反馈接口

使用方式：
    uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import os
import time
import csv
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import numpy as np

# 添加项目根目录到系统路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.schemas import (
    SingleClassifyRequest,
    SingleClassifyResponse,
    BatchClassifyRequest,
    BatchClassifyResponse,
    BatchResultItem,
    PredictionItem,
    HealthResponse,
    ModelInfoResponse,
    FeedbackRequest,
    FeedbackRecord,
    FeedbackListResponse,
    CATEGORY_NAMES,
    get_category_name,
)
from app.api.model_manager import ModelManager, get_model_manager, load_config


# ============================================================================
# 全局配置与初始化
# ============================================================================

# 加载配置
CONFIG = load_config()
SERVICE_CFG = CONFIG.get("service", {})
PROJECT_CFG = CONFIG.get("project", {})

# 服务版本
SERVICE_VERSION = PROJECT_CFG.get("version", "1.0.0")
SERVICE_NAME = PROJECT_CFG.get("name", "商品类目智能分类服务")

# 启动时间
START_TIME = time.time()

# 反馈数据文件路径
PATHS_CFG = CONFIG.get("paths", {})
PROCESSED_DATA_DIR = PROJECT_ROOT / PATHS_CFG.get("processed_data", "data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_FILE = PROCESSED_DATA_DIR / "feedback.csv"

# 批量大小限制
MAX_BATCH_SIZE = SERVICE_CFG.get("max_batch_size", 64)


# ============================================================================
# FastAPI应用实例创建
# ============================================================================

app = FastAPI(
    title=SERVICE_NAME,
    description="基于NLP与机器学习的小程序商品自动化分类系统 - 推理API服务",
    version=SERVICE_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================================
# 中间件配置
# ============================================================================

# CORS中间件 - 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 允许所有来源（生产环境应限制具体域名）
    allow_credentials=True,
    allow_methods=["*"],           # 允许所有HTTP方法
    allow_headers=["*"],           # 允许所有请求头
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    请求日志中间件
    记录每个请求的方法、路径、处理耗时和状态码
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    logger.info(
        f"[Request][{request_id}] {request.method} {request.url.path} "
        f"来自 {client_ip}"
    )

    try:
        response = await call_next(request)
        process_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[Response][{request_id}] {request.method} {request.url.path} "
            f"状态={response.status_code} 耗时={process_time_ms:.1f}ms"
        )
        # 在响应头中添加请求ID和耗时
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-MS"] = f"{process_time_ms:.1f}"
        return response
    except Exception as e:
        process_time_ms = (time.time() - start_time) * 1000
        logger.error(
            f"[Error][{request_id}] {request.method} {request.url.path} "
            f"异常={type(e).__name__}: {e} 耗时={process_time_ms:.1f}ms"
        )
        raise


# ============================================================================
# 全局异常处理
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    HTTP异常处理
    """
    logger.warning(f"HTTP异常 {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "code": exc.status_code,
            "message": exc.detail,
            "path": request.url.path,
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    全局未捕获异常处理
    """
    logger.exception(f"未捕获异常: {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "code": 500,
            "message": "服务器内部错误，请稍后重试",
            "detail": str(exc) if os.environ.get("DEBUG") else None,
            "path": request.url.path,
            "timestamp": datetime.now().isoformat(),
        },
    )


# ============================================================================
# 启动与关闭事件
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    应用启动事件
    初始化模型管理器，确保反馈CSV文件存在
    """
    logger.info("=" * 60)
    logger.info(f"{SERVICE_NAME} v{SERVICE_VERSION} 正在启动...")
    logger.info("=" * 60)

    # 初始化模型管理器（全局单例）
    mm = get_model_manager()
    current_model = mm.get_current_model_name()
    loaded_models = mm.get_loaded_models()
    logger.info(f"当前活动模型: {current_model}")
    logger.info(f"已加载模型: {loaded_models}")

    # 确保反馈CSV文件存在并写入表头
    if not FEEDBACK_FILE.exists():
        _init_feedback_file()
        logger.info(f"创建反馈记录文件: {FEEDBACK_FILE}")

    logger.info("=" * 60)
    logger.info(f"{SERVICE_NAME} 启动完成!")
    logger.info(f"API文档: http://localhost:{SERVICE_CFG.get('port', 8000)}/docs")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """
    应用关闭事件
    """
    uptime = time.time() - START_TIME
    logger.info("=" * 60)
    logger.info(f"{SERVICE_NAME} 正在关闭...")
    logger.info(f"服务运行时长: {uptime:.0f}秒 ({uptime/3600:.2f}小时)")
    logger.info("=" * 60)


# ============================================================================
# 辅助函数
# ============================================================================

def _init_feedback_file():
    """初始化反馈CSV文件，写入表头"""
    with open(FEEDBACK_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "timestamp", "title", "detail",
            "predicted_category_id", "predicted_category_name",
            "correct_category_id", "correct_category_name",
            "model_name", "remark"
        ])


def _ensure_feedback_file():
    """确保反馈文件存在，不存在则初始化"""
    if not FEEDBACK_FILE.exists():
        _init_feedback_file()


def _combine_text(title: str, detail: str = "") -> str:
    """
    合并商品标题和详情为单一文本

    Args:
        title: 商品标题
        detail: 商品详情

    Returns:
        str: 合并后的文本
    """
    title = title or ""
    detail = detail or ""
    if detail:
        return f"{title} {detail}"
    return title


def _get_top_predictions(
    proba: np.ndarray, top_k: int
) -> List[PredictionItem]:
    """
    从概率数组中提取Top K个预测结果

    Args:
        proba: 概率数组 (12,)
        top_k: 返回前K个

    Returns:
        List[PredictionItem]: Top K预测结果列表
    """
    top_k = min(top_k, len(CATEGORY_NAMES))
    # 获取Top K的索引（从大到小）
    top_indices = np.argsort(proba)[::-1][:top_k]

    predictions = []
    for idx in top_indices:
        predictions.append(PredictionItem(
            category_id=int(idx),
            category_name=get_category_name(int(idx)),
            confidence=float(proba[idx]),
        ))
    return predictions


def _classify_single_text(
    text: str, model, top_k: int
) -> Tuple[int, str, float, List[PredictionItem]]:
    """
    对单条文本执行分类

    Args:
        text: 待分类文本
        model: 模型实例
        top_k: 返回Top K结果

    Returns:
        Tuple: (category_id, category_name, confidence, top_predictions)
    """
    probas = model.predict_proba([text])
    proba = probas[0]

    # 获取最优预测
    best_idx = int(np.argmax(proba))
    best_confidence = float(proba[best_idx])
    best_name = get_category_name(best_idx)

    # 获取Top K
    top_predictions = _get_top_predictions(proba, top_k)

    return best_idx, best_name, best_confidence, top_predictions


# ============================================================================
# API接口
# ============================================================================

# ----------------------------------------------------------------------------
# 健康检查
# ----------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """
    检查服务健康状态，返回服务运行情况和当前模型信息。

    **返回字段说明：**
    - `status`: 服务状态，healthy=正常，degraded=降级模式(使用Fallback)，unhealthy=异常
    - `model_name`: 当前加载使用的模型名称
    - `version`: 服务版本号
    - `uptime_seconds`: 服务已运行秒数
    """
    mm = get_model_manager()
    current_model_name = mm.get_current_model_name()

    # 判断服务状态
    if current_model_name == "fallback_keyword":
        # 正在使用兜底分类器，标记为降级状态
        status = "degraded"
    else:
        status = "healthy"

    uptime = time.time() - START_TIME

    return HealthResponse(
        status=status,
        model_name=current_model_name,
        version=SERVICE_VERSION,
        uptime_seconds=round(uptime, 2),
    )


# ----------------------------------------------------------------------------
# 模型管理
# ----------------------------------------------------------------------------

@app.get("/models", response_model=List[ModelInfoResponse], summary="列出所有可用模型")
async def list_models():
    """
    列出系统支持的所有分类模型及其详细信息。

    **返回每个模型的信息：**
    - `model_name`: 模型名称标识
    - `accuracy`: 模型准确率 (0-1)
    - `f1_macro`: 宏平均F1分数 (0-1)
    - `size_mb`: 模型文件大小 (MB)
    - `inference_time_ms`: 单样本平均推理耗时 (毫秒)
    - `description`: 模型描述说明
    """
    mm = get_model_manager()
    model_infos = mm.get_all_model_infos()

    result = []
    for info in model_infos:
        result.append(ModelInfoResponse(
            model_name=info["model_name"],
            accuracy=info["accuracy"],
            f1_macro=info["f1_macro"],
            size_mb=info["size_mb"],
            inference_time_ms=info["inference_time_ms"],
            description=(
                f"[{'已加载' if info['is_loaded'] else '未加载'}]"
                f"[{'当前使用' if info['is_current'] else '未激活'}] "
                f"{info['display_name']} - {info['description']}"
            ),
        ))
    return result


@app.post("/models/switch/{model_name}", summary="切换当前使用的模型")
async def switch_model(model_name: str):
    """
    热切换当前使用的分类模型，无需重启服务。

    **支持的模型名称：**
    - `random_forest`: TF-IDF + 随机森林
    - `fasttext`: FastText
    - `bert`: BERT (bert-base-chinese)
    - `bert_distilled`: 蒸馏BERT (默认)
    - `fallback_keyword`: 关键词匹配兜底

    **路径参数：**
    - `model_name`: 目标模型名称
    """
    mm = get_model_manager()

    # 校验模型名称
    supported = list(mm.SUPPORTED_MODELS.keys()) + ["fallback_keyword"]
    if model_name not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的模型 '{model_name}'，可选: {supported}"
        )

    try:
        success = mm.switch_model(model_name)
        if success:
            return {
                "success": True,
                "message": f"已切换到模型: {model_name}",
                "current_model": mm.get_current_model_name(),
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"切换到模型 '{model_name}' 失败，请检查日志"
            )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"模型文件不存在: {e}，请先训练并保存模型"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"切换模型失败: {str(e)}"
        )


# ----------------------------------------------------------------------------
# 单条分类
# ----------------------------------------------------------------------------

@app.post("/classify", response_model=SingleClassifyResponse, summary="单商品分类")
async def classify_single(request: SingleClassifyRequest):
    """
    对单条商品进行类目分类。

    **请求参数：**
    - `title`: 商品标题（必填）
    - `detail`: 商品详情描述（可选，提供有助于提升分类准确率）
    - `top_k`: 返回Top K个预测结果，默认3，范围1-12
    - `model_name`: 指定使用的模型，默认使用当前活动模型

    **返回字段说明：**
    - `category_id`: 最优预测类目ID (0-11)
    - `category_name`: 最优预测类目名称
    - `confidence`: 最优预测置信度 (0-1)
    - `top_predictions`: Top K个候选预测结果列表
    """
    mm = get_model_manager()

    # 获取模型（优先使用请求指定的模型）
    if request.model_name and request.model_name != mm.get_current_model_name():
        model = mm.get_model(request.model_name)
        using_model = request.model_name
    else:
        model = mm.get_current_model()
        using_model = mm.get_current_model_name()

    # 合并标题和详情
    combined_text = _combine_text(request.title, request.detail)

    logger.debug(
        f"[Classify] 使用模型={using_model}, "
        f"title={request.title[:30]}..."
    )

    try:
        # 执行分类
        cat_id, cat_name, confidence, top_preds = _classify_single_text(
            combined_text, model, request.top_k
        )

        return SingleClassifyResponse(
            category_id=cat_id,
            category_name=cat_name,
            confidence=confidence,
            top_predictions=top_preds,
        )

    except Exception as e:
        logger.exception(f"分类失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"分类处理失败: {str(e)}"
        )


# ----------------------------------------------------------------------------
# 批量分类
# ----------------------------------------------------------------------------

@app.post("/classify/batch", response_model=BatchClassifyResponse, summary="批量商品分类")
async def classify_batch(request: BatchClassifyRequest):
    """
    对多条商品进行批量类目分类，最大支持64条。

    **请求参数：**
    - `items`: 待分类商品列表，最大64条，最少1条
        - `title`: 商品标题（必填）
        - `detail`: 商品详情（可选）
    - `top_k`: 每条返回Top K预测，默认3
    - `model_name`: 指定模型，默认当前活动模型

    **返回字段说明：**
    - `results`: 分类结果列表，每条包含index（对应请求中的顺序）、最优预测和Top K结果
    - `total_count`: 处理的总条数
    - `processing_time_ms`: 总处理耗时（毫秒）
    """
    start_time = time.time()
    mm = get_model_manager()

    # 批量大小校验（Pydantic已做，但双层保险）
    if len(request.items) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"批量分类最大支持 {MAX_BATCH_SIZE} 条，当前提交了 {len(request.items)} 条"
        )
    if len(request.items) == 0:
        raise HTTPException(
            status_code=400,
            detail="批量分类请求不能为空"
        )

    # 获取模型
    if request.model_name and request.model_name != mm.get_current_model_name():
        model = mm.get_model(request.model_name)
        using_model = request.model_name
    else:
        model = mm.get_current_model()
        using_model = mm.get_current_model_name()

    logger.info(
        f"[BatchClassify] 使用模型={using_model}, 批量大小={len(request.items)}"
    )

    try:
        # 批量准备文本
        texts = []
        for item in request.items:
            texts.append(_combine_text(item.title, item.detail))

        # 批量推理
        probas = model.predict_proba(texts)

        # 构造结果
        results = []
        for i, proba in enumerate(probas):
            best_idx = int(np.argmax(proba))
            best_conf = float(proba[best_idx])
            best_name = get_category_name(best_idx)
            top_preds = _get_top_predictions(proba, request.top_k)

            results.append(BatchResultItem(
                index=i,
                category_id=best_idx,
                category_name=best_name,
                confidence=best_conf,
                top_predictions=top_preds,
            ))

        processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"[BatchClassify] 完成: {len(results)}条, 总耗时={processing_time_ms:.1f}ms, "
            f"平均每条={processing_time_ms/max(1, len(results)):.1f}ms"
        )

        return BatchClassifyResponse(
            results=results,
            total_count=len(results),
            processing_time_ms=round(processing_time_ms, 2),
        )

    except Exception as e:
        logger.exception(f"批量分类失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"批量分类处理失败: {str(e)}"
        )


# ----------------------------------------------------------------------------
# 纠错反馈
# ----------------------------------------------------------------------------

@app.post("/feedback", summary="提交运营纠错反馈")
async def submit_feedback(request: FeedbackRequest):
    """
    提交运营纠错反馈，用于记录分类错误样本，供后续增量训练和模型优化。

    **请求参数：**
    - `title`: 商品标题（必填）
    - `detail`: 商品详情描述（可选）
    - `predicted_category_id`: 模型预测的类目ID (0-11)
    - `correct_category_id`: 运营纠正的正确类目ID (0-11)
    - `model_name`: 使用的模型名称（可选）
    - `remark`: 备注说明（可选）

    **反馈数据会保存到：** `data/processed/feedback.csv`
    """
    _ensure_feedback_file()

    # 校验类目ID合法性
    for field_name, field_val in [
        ("predicted_category_id", request.predicted_category_id),
        ("correct_category_id", request.correct_category_id),
    ]:
        if not (0 <= field_val <= 11):
            raise HTTPException(
                status_code=400,
                detail=f"{field_name}={field_val} 超出有效范围 0-11"
            )

    try:
        # 生成记录ID（使用当前时间戳+简单自增，保证单实例内唯一）
        record_id = int(time.time() * 1000) % 10000000000
        timestamp = datetime.now().isoformat()

        # 追加写入CSV
        with open(FEEDBACK_FILE, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                record_id,
                timestamp,
                request.title,
                request.detail,
                request.predicted_category_id,
                get_category_name(request.predicted_category_id),
                request.correct_category_id,
                get_category_name(request.correct_category_id),
                request.model_name or "",
                request.remark or "",
            ])

        logger.info(
            f"[Feedback] 收到纠错反馈 #{record_id}: "
            f"预测={request.predicted_category_id}({get_category_name(request.predicted_category_id)}) -> "
            f"正确={request.correct_category_id}({get_category_name(request.correct_category_id)})"
        )

        return {
            "success": True,
            "message": "反馈提交成功，感谢您的纠错！",
            "record_id": record_id,
            "timestamp": timestamp,
        }

    except Exception as e:
        logger.exception(f"反馈写入失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"反馈保存失败: {str(e)}"
        )


@app.get("/feedback/list", response_model=FeedbackListResponse, summary="查看纠错反馈记录")
async def list_feedback(
    limit: int = 100,
    offset: int = 0,
    category_id: Optional[int] = None,
):
    """
    查看已提交的纠错反馈记录列表。

    **查询参数：**
    - `limit`: 返回最大记录数，默认100，最大1000
    - `offset`: 跳过的记录数，用于分页，默认0
    - `category_id`: 按正确类目ID过滤（可选），范围0-11

    **返回：**
    - `records`: 反馈记录列表
    - `total_count`: 满足条件的总记录数
    """
    _ensure_feedback_file()

    # 参数校验
    limit = max(1, min(limit, 1000))
    offset = max(0, offset)
    if category_id is not None and not (0 <= category_id <= 11):
        raise HTTPException(
            status_code=400,
            detail=f"category_id={category_id} 超出有效范围 0-11"
        )

    try:
        records = []
        with open(FEEDBACK_FILE, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 类目过滤
                if category_id is not None:
                    correct_id = int(row.get("correct_category_id", -1))
                    if correct_id != category_id:
                        continue

                records.append(FeedbackRecord(
                    id=int(row.get("id", 0)),
                    timestamp=row.get("timestamp", ""),
                    title=row.get("title", ""),
                    detail=row.get("detail", ""),
                    predicted_category_id=int(row.get("predicted_category_id", 0)),
                    predicted_category_name=row.get("predicted_category_name", ""),
                    correct_category_id=int(row.get("correct_category_id", 0)),
                    correct_category_name=row.get("correct_category_name", ""),
                    model_name=row.get("model_name", ""),
                    remark=row.get("remark", ""),
                ))

        # 按时间倒序（最新在前）
        records.sort(key=lambda r: r.timestamp, reverse=True)
        total_count = len(records)

        # 分页
        paginated = records[offset: offset + limit]

        logger.debug(
            f"[FeedbackList] 查询: limit={limit}, offset={offset}, "
            f"category_id={category_id}, 返回={len(paginated)}/{total_count}条"
        )

        return FeedbackListResponse(
            records=paginated,
            total_count=total_count,
        )

    except Exception as e:
        logger.exception(f"查询反馈记录失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"查询反馈记录失败: {str(e)}"
        )


# ============================================================================
# 开发调试入口（直接运行此文件时启动服务）
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    host = SERVICE_CFG.get("host", "0.0.0.0")
    port = SERVICE_CFG.get("port", 8000)

    logger.info(f"直接启动服务: {host}:{port}")
    uvicorn.run(
        "app.api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
