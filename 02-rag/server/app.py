# -*- coding: utf-8 -*-
"""
星途环球·跨境售后RAG智能问答系统 - FastAPI 网关服务
对应简历模块5：
  - 通过 FastAPI 封装问答推理接口
  - 联动前端搭建面向海外消费者、跨境入驻商家的多语种可视化咨询页面
  - 承接日常及海外大促咨询流量，全链路正式上线投产
"""
from __future__ import annotations

import os
import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from config import get_config, KNOWLEDGE_CATEGORIES, SUPPORTED_LANGUAGES
from gateway import AgentOrchestrator
from data_processing import DataPreprocessingPipeline, KnowledgeChunk

# ============================================================
# 🔥 依赖兼容性检查（新手小白最容易崩的地方：缺包直接起不来）
#    策略：缺 python-multipart 也让服务能启动，只在调用 /api/ingest 时提示装包
# ============================================================
_MULTIPART_OK = False
try:
    import python_multipart  # noqa: F401
    _MULTIPART_OK = True
    from fastapi import UploadFile, File, Form  # 正常导入
except Exception:
    # 缺 python-multipart：做假导入让路由注册不崩（实际调用时返回友好提示）
    class _FakeForm:
        def __init__(self, *a, **kw): pass
        def __call__(self, v): return v
    class _FakeFile:
        def __init__(self, *a, **kw): pass
        def __call__(self, v): return v
    UploadFile = object  # noqa: F811 (type alias, never used in fallback path)
    Form = _FakeForm  # noqa: F811
    File = _FakeFile  # noqa: F811


# ============================================================
# 🔥 崩溃兜底：即使 lifespan 没跑（TestClient/异常启动/进程重启），
#    所有接口也能拿到 orchestrator 单例，永远不返回 500
# ============================================================
_bg_bootstrap_started = False

def _get_orch(request: Request) -> AgentOrchestrator:
    """所有路由统一用这个函数取编排器，避免 app.state 没初始化时崩溃"""
    global _bg_bootstrap_started
    try:
        orch = request.app.state.orchestrator
        if orch is not None:
            return orch
    except Exception:
        pass
    # ---------- 兜底：拿 AgentOrchestrator 单例 ----------
    orch = AgentOrchestrator.get_instance(auto_bootstrap=False)
    # 尝试挂回 app.state 方便下次复用
    try:
        request.app.state.orchestrator = orch
    except Exception:
        pass
    try:
        if not hasattr(request.app.state, "pipeline"):
            request.app.state.pipeline = DataPreprocessingPipeline()
    except Exception:
        pass
    # ---------- 懒启动：首次取 orchestrator 时异步后台加载知识库 ----------
    if (not _bg_bootstrap_started) and (not orch._bootstrapped) and (not orch._bootstrap_in_progress):
        _bg_bootstrap_started = True
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(orch.bootstrap_async())
        except Exception:
            pass
    return orch


def _get_pipeline(request: Request) -> DataPreprocessingPipeline:
    try:
        return request.app.state.pipeline
    except Exception:
        pipe = DataPreprocessingPipeline()
        try:
            request.app.state.pipeline = pipe
        except Exception:
            pass
        return pipe


# ============================================================
# 1. 生命周期：启动时初始化编排器
#    🔥 优化：知识库注入改为后台异步线程，不阻塞服务启动（解决新手"页面一直没响应"问题）
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 70)
    print("🚀 星途环球·跨境售后RAG智能问答系统 - FastAPI 网关启动中")
    print("=" * 70)
    # ① 快速创建编排器实例（不注入知识，<0.2s搞定）
    orch = AgentOrchestrator.get_instance(auto_bootstrap=False)
    app.state.orchestrator = orch
    app.state.pipeline = DataPreprocessingPipeline()
    status = orch.status()
    print(f"  📚 已索引知识分块: {status['total_indexed_chunks']}（后台异步加载中…）")
    print(f"  🧠 LLM: {status['llm_provider']} | Rerank: {status['rerank_provider']}")
    print(f"  📐 向量库: {status['vector_store']} | 混合权重 D{status['hybrid_weights']['dense']}:S{status['hybrid_weights']['sparse']}")
    print("=" * 70)

    # ② 后台异步注入 51 组跨境种子知识库（跑在线程池，不阻塞事件循环）
    async def _bg_bootstrap():
        try:
            added = await orch.bootstrap_async()
            if added:
                print(f"[Startup] ✅ 后台知识库加载完成，共新增 {added} 个知识分块")
            else:
                print("[Startup] ✅ 后台知识库加载完成（已存在或空）")
        except Exception as exc:
            print(f"[Startup] ⚠️ 后台知识库加载异常(忽略): {exc}")

    _bg_task = asyncio.create_task(_bg_bootstrap())
    app.state._bg_bootstrap_task = _bg_task

    yield  # ← 立刻让出控制权！浏览器现在就能打开页面 ✨

    print("\n🛑 服务停止")


app = FastAPI(
    title="XingTu Global Cross-border After-sales RAG System",
    description="星途环球·跨境售后RAG智能问答系统 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 跨域：允许独立站前端
_cfg = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cfg.agents.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 2. 请求 / 响应 Pydantic 模型
# ============================================================
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: str = Field(default="default", description="会话ID")
    language: Optional[str] = Field(default=None, description="强制输出语种: zh/en/ja/es/de/fr")
    category_filter: Optional[str] = Field(
        default=None,
        description="类目过滤: cross_border_logistics / after_sales_policy / customs_compliance",
    )
    stream: bool = Field(default=False, description="预留流式输出")


class ChatResponse(BaseModel):
    response: str
    session_id: str
    language: str
    category: Optional[str]
    retrieved_contexts: List[str]
    latency_ms: float
    trace: Dict[str, Any]


class IngestResponse(BaseModel):
    category: str
    added_chunks: int
    warnings: List[str]
    file_name: str


# ============================================================
# 3. 核心路由
# ============================================================

# -------- 3.1 健康检查 --------
@app.get("/api/health", tags=["System"])
async def health(request: Request):
    orch: AgentOrchestrator = _get_orch(request)
    status = orch.status()
    return {
        "status": "ok",
        "timestamp": time.time(),
        "service": "xingtu_crossborder_rag",
        "indexed_chunks": status["total_indexed_chunks"],
        "categories": list(KNOWLEDGE_CATEGORIES.keys()),
        "languages": SUPPORTED_LANGUAGES,
    }


# -------- 3.2 系统配置/类目/语种 元信息 --------
@app.get("/api/meta", tags=["System"])
async def get_meta():
    return {
        "categories": [
            {"key": k, "name": v["name"], "icon": v["icon"], "description": v["description"]}
            for k, v in KNOWLEDGE_CATEGORIES.items()
        ],
        "languages": [{"code": k, "name": v} for k, v in SUPPORTED_LANGUAGES.items()],
        "hybrid_weights": {
            "dense": _cfg.milvus.hybrid_search_weight_dense,
            "sparse": _cfg.milvus.hybrid_search_weight_sparse,
        },
        "safety_threshold": _cfg.safety.retrieval_confidence_threshold,
    }


# -------- 3.3 问答主接口（被 test_rag_qa.py 调用）--------
@app.post("/api/chat", response_model=ChatResponse, tags=["RAG"])
async def chat(req: ChatRequest, request: Request):
    start = time.perf_counter()
    if req.category_filter and req.category_filter not in KNOWLEDGE_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"category_filter 非法，有效值：{list(KNOWLEDGE_CATEGORIES.keys())}")
    if req.language and req.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"language 非法，有效值：{list(SUPPORTED_LANGUAGES.keys())}")

    orch: AgentOrchestrator = _get_orch(request)
    # 调用编排器
    result = await orch.handle_user_request(
        user_input=req.message,
        session_id=req.session_id,
        force_language=req.language,
        category_filter=req.category_filter,
    )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return ChatResponse(
        response=result.get("response", ""),
        session_id=req.session_id,
        language=result.get("language", "zh"),
        category=result.get("category"),
        retrieved_contexts=result.get("retrieved_contexts", [])[:5],
        latency_ms=latency_ms,
        trace=result.get("trace", {}),
    )


# -------- 3.4 文档入库接口（增量添加知识库）--------
#    ⚠️ 兼容处理：缺 python-multipart 时注册 fallback 路由提示装包，不让整个服务崩
def _register_ingest_route_ok():
    """正常路径：已装 python-multipart，启用真实的文件上传解析"""
    @app.post("/api/ingest", response_model=IngestResponse, tags=["Knowledge Base"])
    async def ingest_document(
        request: Request,
        category: str = Form(..., description="cross_border_logistics / after_sales_policy / customs_compliance"),
        title: Optional[str] = Form(None, description="文档标题"),
        file: UploadFile = File(..., description="PDF/Word/TXT/图片"),
    ):
        if category not in KNOWLEDGE_CATEGORIES:
            raise HTTPException(status_code=400, detail=f"category 非法")
        # 保存临时文件
        tmp_dir = os.path.join(_cfg.data_dir, "tmp_uploads")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, f"{int(time.time()*1000)}_{file.filename}")
        with open(tmp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        # 预处理 + 入库
        pipe: DataPreprocessingPipeline = _get_pipeline(request)
        orch: AgentOrchestrator = _get_orch(request)
        try:
            chunks, warns = pipe.process_file(tmp_path, category, doc_title=title or file.filename)
            added = orch.ingest_document_chunks(chunks)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
        return IngestResponse(
            category=category,
            added_chunks=added,
            warnings=warns,
            file_name=file.filename or "",
        )


def _register_ingest_route_fallback():
    """兜底路径：缺 python-multipart，提示用户安装（不让整个服务启动崩溃）"""
    @app.post("/api/ingest", tags=["Knowledge Base"])
    async def ingest_document_fallback(request: Request):
        msg = (
            "⚠️ 缺少依赖包 python-multipart，无法解析上传的文档文件。\n"
            "请在 PyCharm Terminal 运行以下命令后重试：\n\n"
            "  pip install python-multipart\n\n"
            "（💡 其他核心功能：聊天 / 问答 / 健康检查 / 前端页面 都可正常使用，无需此包）"
        )
        return JSONResponse(status_code=503, content={"detail": msg, "status": "missing_dependency", "install_command": "pip install python-multipart"})


# 👉 关键：先尝试注册真实路由；如果因为缺 multipart 崩了，就注册兜底提示路由
try:
    if not _MULTIPART_OK:
        raise RuntimeError("multipart not installed")
    _register_ingest_route_ok()
except Exception:
    _register_ingest_route_fallback()


# -------- 3.5 状态接口（运维监控）--------
@app.get("/api/status", tags=["System"])
async def status(request: Request):
    orch: AgentOrchestrator = _get_orch(request)
    return orch.status()


# ============================================================
# 4. 多语种可视化前端页面（🔥 100% 离线纯原生 JS + CSS，无CDN依赖）
# 对应简历：联动前端搭建面向海外消费者、跨境入驻商家的多语种可视化咨询页面
# ============================================================
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
FRONTEND_INDEX = os.path.join(FRONTEND_DIR, "index.html")


@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def frontend_index():
    """前端主页面入口"""
    if not os.path.exists(FRONTEND_INDEX):
        return HTMLResponse(content=_build_frontend_html(), status_code=200)
    with open(FRONTEND_INDEX, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


def _build_frontend_html() -> str:
    """动态构建前端HTML — 🔥 100% 离线可用（0 CDN依赖）
    针对小白用户：不管有没有网络/有没有代理，只要服务启动了就能用！
    - 样式：全部内嵌CSS，不依赖Tailwind CDN
    - 交互：纯原生JS，不依赖Vue3 CDN
    """
    return r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>星途环球 · 跨境售后智能客服 | XingTu Global AI Support</title>
<style>
/* ========== 基础样式（替代Tailwind） ========== */
*, *::before, *::after { box-sizing: border-box; }
body { margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
       background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%); min-height:100vh; color:#334155; }
a { color: #4f46e5; text-decoration: none; }
header { background: rgba(255,255,255,0.85); backdrop-filter: blur(10px); border-bottom:1px solid #e5e7eb; position:sticky; top:0; z-index:10; }
.hd-container { max-width: 1152px; margin:0 auto; padding:14px 20px; display:flex; align-items:center; justify-content: space-between; flex-wrap:wrap; gap:10px; }
.brand { display:flex; align-items:center; gap:10px; }
.brand-logo { width:40px; height:40px; border-radius:10px; background: linear-gradient(135deg, #6366f1, #8b5cf6); color:#fff;
              display:flex; align-items:center; justify-content:center; font-size:20px; }
.brand-name { font-weight:700; color:#1e293b; font-size:16px; }
.brand-sub { font-size:12px; color:#64748b; margin-top:2px; }
.lang-select { padding:7px 12px; border-radius:8px; border:1px solid #cbd5e1; background:#fff; color:#334155;
               font-size:14px; cursor:pointer; outline:none; }
.lang-select:hover { border-color:#818cf8; }
.btn-status { padding:7px 12px; border-radius:8px; background:#f1f5f9; border:none; color:#475569;
              font-size:12px; cursor:pointer; }
.btn-status:hover { background:#e2e8f0; }

main { max-width: 1152px; margin:0 auto; padding:24px 20px; display:grid; grid-template-columns: 1fr 340px; gap:24px; }
@media (max-width: 900px) { main { grid-template-columns:1fr; } }

/* ========== 左：聊天面板 ========== */
.chat-panel { background:#fff; border-radius:18px; border:1px solid #e5e7eb; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
              display:flex; flex-direction:column; height: calc(100vh - 170px); min-height:540px; }
.presets-bar { padding:14px 18px 10px; border-bottom:1px solid #f1f5f9; }
.presets-label { font-size:12px; color:#64748b; margin-bottom:8px; }
.presets-list { display:flex; flex-wrap:wrap; gap:8px; }
.preset-btn { font-size:12px; padding:7px 14px; border-radius:999px; background:#eef2ff; color:#4338ca;
              border:1px solid #e0e7ff; cursor:pointer; transition:all .15s; }
.preset-btn:hover { background:#e0e7ff; transform: translateY(-1px); }
.msg-list { flex:1; overflow-y:auto; padding:18px; display:flex; flex-direction:column; gap:16px; }
.msg-row { display:flex; gap:12px; }
.msg-row.user { justify-content:flex-end; }
.avatar { width:34px; height:34px; border-radius:50%; color:#fff; display:flex; align-items:center; justify-content:center;
          font-size:15px; flex-shrink:0; }
.avatar.ai { background: linear-gradient(135deg, #34d399, #14b8a6); }
.avatar.user { background: linear-gradient(135deg, #6366f1, #8b5cf6); }
.bubble { max-width:78%; padding:12px 16px; border-radius:16px; font-size:14px; line-height:1.65;
          white-space:pre-wrap; word-break: break-word; box-shadow:0 1px 2px rgba(0,0,0,0.05); }
.msg-row.user .bubble { background: linear-gradient(135deg, #6366f1, #8b5cf6); color:#fff; border-top-right-radius:4px; }
.msg-row.ai   .bubble { background:#ffffff; color:#334155; border:1px solid #e5e7eb; border-top-left-radius:4px; }
.bubble-meta { margin-top:8px; padding-top:8px; border-top:1px dashed #f1f5f9; font-size:11px; color:#94a3b8;
               display:flex; flex-wrap:wrap; gap:6px 14px; }
.bubble-meta .tag-faith { color:#059669; }
.bubble-meta .tag-halluc { color:#dc2626; font-weight:600; }
.loader { display:inline-block; width:14px; height:14px; border:2.5px solid #e5e7eb; border-top-color:#6366f1;
          border-radius:50%; animation: spin 0.8s linear infinite; vertical-align:-2px; margin-right:8px; }
@keyframes spin { to { transform: rotate(360deg); } }

/* ========== 输入框 ========== */
.input-area { padding:14px 18px; border-top:1px solid #f1f5f9; background:#f8fafc; border-bottom-left-radius:18px; border-bottom-right-radius:18px; }
.filter-row { display:flex; align-items:center; gap:8px; margin-bottom:10px; font-size:12px; color:#64748b; }
.filter-row select { padding:3px 8px; border-radius:6px; border:1px solid #cbd5e1; background:#fff; font-size:12px; }
.input-row { display:flex; gap:10px; }
.input-row input { flex:1; padding:12px 16px; border-radius:14px; border:1px solid #cbd5e1; background:#fff;
                   outline:none; font-size:14px; transition: all .15s; }
.input-row input:focus { border-color:#818cf8; box-shadow:0 0 0 3px rgba(99,102,241,0.12); }
.btn-send { padding:0 22px; border-radius:14px; background: linear-gradient(135deg, #6366f1, #8b5cf6); color:#fff;
            border:none; font-size:14px; font-weight:500; cursor:pointer; transition:opacity .15s; }
.btn-send:hover { opacity:0.92; }
.btn-send:disabled { opacity:0.4; cursor:not-allowed; }

/* ========== 右侧卡片 ========== */
.side { display:flex; flex-direction:column; gap:18px; }
.card { background:#fff; border-radius:18px; border:1px solid #e5e7eb; overflow:hidden; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.card-header { padding:13px 18px; background: linear-gradient(135deg, #6366f1, #8b5cf6); color:#fff; }
.card-header .title { font-size:14px; font-weight:600; }
.card-header .sub { font-size:12px; opacity:0.9; margin-top:3px; }
.card-body { padding:16px 18px; }
.cat-row { display:flex; align-items:center; gap:12px; padding:10px 0; }
.cat-row + .cat-row { border-top:1px solid #f8fafc; }
.cat-icon { width:40px; height:40px; border-radius:10px; background:#f8fafc; display:flex;
            align-items:center; justify-content:center; font-size:20px; flex-shrink:0; }
.cat-name { font-size:14px; font-weight:500; color:#1e293b; }
.cat-desc { font-size:12px; color:#64748b; margin-top:2px; line-height:1.5; }

.lang-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }
.lang-btn { padding:9px 0; text-align:center; border-radius:10px; background:#f8fafc; color:#475569;
            font-size:12px; cursor:pointer; transition:all .15s; border:none; }
.lang-btn .flag { font-size:17px; display:block; margin-bottom:3px; }
.lang-btn.active { background: linear-gradient(135deg, #6366f1, #8b5cf6); color:#fff; }
.lang-btn:hover:not(.active) { background:#eef2ff; }

/* 系统状态弹层 */
.status-panel { padding:14px 16px; font-size:12px; color:#475569; }
.status-panel .row-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; font-weight:600; color:#334155; }
.close-x { cursor:pointer; color:#94a3b8; font-size:15px; background:none; border:none; }
.status-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:12px; }
.status-box { padding:10px; text-align:center; border-radius:10px; }
.status-box.green { background:#ecfdf5; color:#047857; }
.status-box.blue  { background:#eef2ff; color:#4338ca; }
.status-num { font-size:20px; font-weight:700; line-height:1; }
.status-label { font-size:11px; opacity:0.8; margin-top:4px; }
.status-list { margin-top:8px; }
.status-list .sl-item { display:flex; justify-content:space-between; align-items:center; padding:7px 10px; background:#f8fafc;
                        border-radius:7px; margin-bottom:5px; font-size:11px; color:#64748b; }

.loading-text { color:#94a3b8; font-size:12px; }
</style>
</head>
<body>
<div id="app">
  <!-- 顶部导航 -->
  <header>
    <div class="hd-container">
      <div class="brand">
        <div class="brand-logo">🌍</div>
        <div>
          <div class="brand-name" id="ui_brand">星途环球 · 跨境智能客服</div>
          <div class="brand-sub" id="ui_subtitle">7×24 跨语种 · 物流/售后/合规 一站式咨询</div>
        </div>
      </div>
      <div style="display:flex; align-items:center; gap:10px;">
        <select class="lang-select" id="langSelect"></select>
        <button class="btn-status" id="btnStatus" onclick="toggleStatus()">⚙️ <span id="ui_systemStatus">系统状态</span></button>
      </div>
    </div>
  </header>

  <main>
    <!-- 左：聊天面板 -->
    <section class="chat-panel">
      <!-- 快捷问题 -->
      <div class="presets-bar">
        <div class="presets-label" id="ui_presetsLabel">🔥 热门咨询</div>
        <div class="presets-list" id="presetsList"></div>
      </div>
      <!-- 消息列表 -->
      <div class="msg-list" id="msgList"></div>
      <!-- 输入框 -->
      <div class="input-area">
        <div class="filter-row">
          <span id="ui_filterLabel">🎯 知识库过滤：</span>
          <select id="categoryFilter">
            <option value="" id="ui_filterAll">全部（自动识别）</option>
          </select>
        </div>
        <div class="input-row">
          <input type="text" id="inputText" onkeydown="if(event.key==='Enter')sendMessage()" />
          <button class="btn-send" id="btnSend" onclick="sendMessage()">发送</button>
        </div>
      </div>
    </section>

    <!-- 右：侧边栏 -->
    <aside class="side">
      <!-- 系统状态 -->
      <div class="card" id="statusCard" style="display:none;">
        <div class="status-panel">
          <div class="row-head">
            <span id="ui_systemStatus2">系统状态</span>
            <button class="close-x" onclick="toggleStatus()">✕</button>
          </div>
          <div id="statusLoading" class="loading-text">加载中…</div>
          <div id="statusContent" style="display:none;">
            <div class="status-grid">
              <div class="status-box green">
                <div class="status-num" id="st_indexed">0</div>
                <div class="status-label" id="ui_indexedChunks">知识分块</div>
              </div>
              <div class="status-box blue">
                <div class="status-num" id="st_latency">-</div>
                <div class="status-label" id="ui_avgLatency">平均响应</div>
              </div>
            </div>
            <div class="status-list" id="statusCats"></div>
          </div>
        </div>
      </div>

      <!-- 知识库卡片 -->
      <div class="card">
        <div class="card-header">
          <div class="title" id="ui_knowledgeBase">📚 三大知识库</div>
          <div class="sub" id="ui_kbDesc">跨境物流 · 海外售后 · 各国海关合规</div>
        </div>
        <div class="card-body" id="kbBody"></div>
      </div>

      <!-- 多语种卡片 -->
      <div class="card">
        <div class="card-body">
          <div style="font-weight:600; font-size:14px; color:#334155; margin-bottom:12px;">🌐 <span id="ui_langs">多语种支持</span></div>
          <div class="lang-grid" id="langGrid"></div>
        </div>
      </div>
    </aside>
  </main>
</div>

<script>
/* ============================================================
   100% 纯原生JS实现（替代Vue3 CDN，离线可用）
   ============================================================ */
const I18N = {
  zh: {
    brand:"星途环球 · 跨境智能客服", subtitle:"7×24 跨语种 · 物流/售后/合规 一站式咨询",
    systemStatus:"系统状态", loading:"加载中…",
    presetsLabel:"🔥 热门咨询",
    placeholder:"请输入您的跨境问题（物流/售后/关税VAT…），支持中英日韩德西",
    send:"发送", thinking:"RAG检索中 · 多路召回 + 风控校验…",
    filterLabel:"🎯 知识库过滤：", filterAll:"全部（自动识别）",
    knowledgeBase:"📚 三大知识库", kbDesc:"跨境物流 · 海外售后 · 各国海关合规",
    langs:"多语种支持", indexedChunks:"知识分块", avgLatency:"平均响应",
    presets:[
      {icon:"📦", text:"发美国500g包裹最便宜的物流？"},
      {icon:"🛒", text:"亚马逊美国站30天无理由退货规则？"},
      {icon:"📜", text:"美国个人包裹免税额度多少？"},
      {icon:"🌍", text:"欧盟VAT IOSS新政是什么？"},
    ],
    welcome:"您好！我是星途环球跨境售后AI助手，支持 中英日韩德西 6 种语言。\n我可以为您解答【跨境物流、海外售后政策、各国海关合规】相关问题，请随时提问。\n\n示例：\"发往英国的包裹为什么被收了VAT？\"",
  },
  en: {
    brand:"XingTu Global · Cross-border AI Support", subtitle:"24/7 Multilingual · Logistics/After-sales/Compliance",
    systemStatus:"System Status", loading:"Loading…",
    presetsLabel:"🔥 Popular Questions",
    placeholder:"Ask about shipping, returns, tariffs, VAT, etc.",
    send:"Send", thinking:"RAG retrieval in progress…",
    filterLabel:"🎯 Filter:", filterAll:"All (Auto)",
    knowledgeBase:"📚 Knowledge Domains", kbDesc:"Logistics · After-sales · Customs Compliance",
    langs:"Languages", indexedChunks:"Chunks", avgLatency:"Avg Latency",
    presets:[
      {icon:"📦", text:"Cheapest shipping for 500g to USA?"},
      {icon:"🛒", text:"Amazon US 30-day return policy?"},
      {icon:"📜", text:"USA personal parcel duty-free threshold?"},
      {icon:"🌍", text:"What is EU VAT IOSS new policy?"},
    ],
    welcome:"Hello! I'm XingTu Global Cross-border After-sales AI assistant, supporting 6 languages.\nI can answer questions on Cross-border Logistics, After-sales Policies, and Customs Compliance.\n\nExample: \"Why was my UK parcel charged VAT?\"",
  },
  ja: {
    brand:"星途グローバル · 越境AIカスタマー", subtitle:"24時間365日 · 物流/アフター/関税コンプラ",
    systemStatus:"システム状態", loading:"読込中…",
    presetsLabel:"🔥 よくある質問",
    placeholder:"越境問題を入力（物流・返品・VATなど）",
    send:"送信", thinking:"RAG検索中…",
    filterLabel:"🎯 フィルタ:", filterAll:"すべて（自動）",
    knowledgeBase:"📚 ナレッジ", kbDesc:"物流・アフター・通関コンプラ",
    langs:"言語", indexedChunks:"チャンク数", avgLatency:"平均応答",
    presets:[
      {icon:"📦", text:"米国へ500gの最安配送方法は？"},
      {icon:"🛒", text:"Amazon US 30日返品ポリシーは？"},
      {icon:"📜", text:"米国個人宅の免税額はいくら？"},
      {icon:"🌍", text:"EU VAT IOSS新政策とは？"},
    ],
    welcome:"こんにちは！星途グローバル越境AIサポートです。\n物流・アフター・通関について何でもご質問ください。",
  },
  es: {
    brand:"XingTu Global · Atención AI", subtitle:"24/7 · Logística / Postventa / Cumplimiento",
    systemStatus:"Estado", loading:"Cargando…",
    presetsLabel:"🔥 Preguntas Frecuentes",
    placeholder:"Pregunta sobre envíos, devoluciones, aranceles…",
    send:"Enviar", thinking:"Buscando en RAG…",
    filterLabel:"🎯 Filtrar:", filterAll:"Todo (Auto)",
    knowledgeBase:"📚 Bases de Conocimiento", kbDesc:"Logística / Postventa / Aduanas",
    langs:"Idiomas", indexedChunks:"Fragmentos", avgLatency:"Latencia Media",
    presets:[
      {icon:"📦", text:"Envío más barato 500g a USA?"},
      {icon:"🛒", text:"Política de devolución Amazon 30 días?"},
      {icon:"📜", text:"Umbral libre de impuestos USA?"},
      {icon:"🌍", text:"Qué es la nueva política EU VAT IOSS?"},
    ],
    welcome:"¡Hola! Soy el asistente de XingTu Global. Puedo ayudarle con envíos, postventa y aduanas.",
  },
  de: {
    brand:"XingTu Global · KI Kundenservice", subtitle:"24/7 · Logistik / After-Sales / Compliance",
    systemStatus:"Systemstatus", loading:"Laden…",
    presetsLabel:"🔥 Beliebte Fragen",
    placeholder:"Fragen zu Versand, Rücksendung, Zoll, VAT…",
    send:"Senden", thinking:"RAG-Suche läuft…",
    filterLabel:"🎯 Filter:", filterAll:"Alle (Auto)",
    knowledgeBase:"📚 Wissensdatenbanken", kbDesc:"Logistik / After-Sales / Zoll",
    langs:"Sprachen", indexedChunks:"Wissensbausteine", avgLatency:"Ø Latenz",
    presets:[
      {icon:"📦", text:"Günstigster Versand 500g in USA?"},
      {icon:"🛒", text:"Amazon US 30-Tage Rückgabe?"},
      {icon:"📜", text:"Zollfreibetrag USA für Pakete?"},
      {icon:"🌍", text:"Was ist die neue EU VAT IOSS Regel?"},
    ],
    welcome:"Hallo! Ich bin der XingTu Global KI Assistent für Logistik, After-Sales und Zoll.",
  },
  fr: {
    brand:"XingTu Global · Support IA Cross-border", subtitle:"24/7 · Logistique / SAV / Conformité",
    systemStatus:"État", loading:"Chargement…",
    presetsLabel:"🔥 Questions Populaires",
    placeholder:"Questions sur livraison, retours, TVA…",
    send:"Envoyer", thinking:"Recherche RAG en cours…",
    filterLabel:"🎯 Filtrer:", filterAll:"Tout (Auto)",
    knowledgeBase:"📚 Bases de connaissances", kbDesc:"Logistique / SAV / Douanes",
    langs:"Langues", indexedChunks:"Fragments", avgLatency:"Latence moy.",
    presets:[
      {icon:"📦", text:"Livraison 500g la moins chère vers USA?"},
      {icon:"🛒", text:"Politique de retour Amazon 30j?"},
      {icon:"📜", text:"Seuil franchise de douane USA?"},
      {icon:"🌍", text:"Quelle est la nouvelle réforme TVA IOSS UE?"},
    ],
    welcome:"Bonjour ! Je suis l'assistant IA XingTu Global pour la logistique, le SAV et les douanes.",
  },
};
const LANG_META = {
  zh:{code:"zh",flag:"🇨🇳",label:"中文"}, en:{code:"en",flag:"🇺🇸",label:"English"},
  ja:{code:"ja",flag:"🇯🇵",label:"日本語"}, es:{code:"es",flag:"🇪🇸",label:"Español"},
  de:{code:"de",flag:"🇩🇪",label:"Deutsch"}, fr:{code:"fr",flag:"🇫🇷",label:"Français"},
};

/* ---------- 全局状态 ---------- */
let language = "zh";
let categoryFilter = "";
let latencies = [];
let categoriesList = [
  {key:"cross_border_logistics", icon:"📦", name:"", description:""},
  {key:"after_sales_policy",   icon:"🛒", name:"", description:""},
  {key:"customs_compliance",   icon:"📜", name:"", description:""},
];
let openStatus = false;

/* ---------- 初始化 ---------- */
window.addEventListener("DOMContentLoaded", () => {
  renderLangOptions();
  renderLangGrid();
  renderCategoryFilter();
  loadMeta();
  applyI18N();
  pushWelcome();
});

function currentUI() { return I18N[language] || I18N.en; }

function applyI18N() {
  const ui = currentUI();
  byId("ui_brand").textContent = ui.brand;
  byId("ui_subtitle").textContent = ui.subtitle;
  byId("ui_systemStatus").textContent = ui.systemStatus;
  if (byId("ui_systemStatus2")) byId("ui_systemStatus2").textContent = ui.systemStatus;
  byId("ui_presetsLabel").textContent = ui.presetsLabel;
  byId("inputText").placeholder = ui.placeholder;
  byId("btnSend").textContent = ui.send;
  byId("ui_filterLabel").textContent = ui.filterLabel;
  byId("ui_filterAll").textContent = ui.filterAll;
  byId("ui_knowledgeBase").textContent = ui.knowledgeBase;
  byId("ui_kbDesc").textContent = ui.kbDesc;
  byId("ui_langs").textContent = ui.langs;
  byId("ui_indexedChunks").textContent = ui.indexedChunks;
  byId("ui_avgLatency").textContent = ui.avgLatency;
  byId("statusLoading").textContent = ui.loading;
  renderPresets();
  renderKB();
  // 如果没有消息，推一条欢迎语
  const list = byId("msgList");
  if (!list.children.length) pushWelcome();
}

function pushWelcome() {
  appendAI(currentUI().welcome, false);
}

/* ---------- 语言切换 ---------- */
function renderLangOptions() {
  const sel = byId("langSelect");
  sel.innerHTML = "";
  Object.values(LANG_META).forEach(l => {
    const opt = document.createElement("option");
    opt.value = l.code; opt.textContent = l.flag + " " + l.label;
    sel.appendChild(opt);
  });
  sel.value = language;
  sel.onchange = () => { language = sel.value; applyI18N(); renderLangGrid(); };
}
function renderLangGrid() {
  const g = byId("langGrid");
  g.innerHTML = "";
  Object.values(LANG_META).forEach(l => {
    const b = document.createElement("button");
    b.className = "lang-btn" + (language === l.code ? " active" : "");
    b.innerHTML = `<span class="flag">${l.flag}</span>${l.label}`;
    b.onclick = () => { language = l.code; byId("langSelect").value = l.code; applyI18N(); renderLangGrid(); };
    g.appendChild(b);
  });
}

/* ---------- 知识库 ---------- */
function renderCategoryFilter() {
  const sel = byId("categoryFilter");
  // 保留第一个（ui_filterAll）
  while (sel.options.length > 1) sel.remove(1);
  categoriesList.forEach(c => {
    const opt = document.createElement("option");
    opt.value = c.key; opt.textContent = c.icon + " " + (c.name || c.key);
    sel.appendChild(opt);
  });
  sel.value = categoryFilter;
  sel.onchange = () => { categoryFilter = sel.value; };
}
function renderKB() {
  const body = byId("kbBody");
  body.innerHTML = "";
  categoriesList.forEach(c => {
    const row = document.createElement("div");
    row.className = "cat-row";
    row.innerHTML = `
      <div class="cat-icon">${c.icon}</div>
      <div style="flex:1;">
        <div class="cat-name">${c.name || c.key}</div>
        <div class="cat-desc">${c.description || "-"}</div>
      </div>`;
    body.appendChild(row);
  });
}
function renderPresets() {
  const list = byId("presetsList");
  list.innerHTML = "";
  currentUI().presets.forEach(q => {
    const b = document.createElement("button");
    b.className = "preset-btn";
    b.textContent = q.icon + " " + q.text;
    b.onclick = () => { byId("inputText").value = q.text; sendMessage(); };
    list.appendChild(b);
  });
}
function renderStatusCats() {
  const box = byId("statusCats");
  box.innerHTML = "";
  categoriesList.forEach(c => {
    const d = document.createElement("div");
    d.className = "sl-item";
    d.innerHTML = `<span>${c.icon} ${c.name || c.key}</span><span>✓</span>`;
    box.appendChild(d);
  });
}

/* ---------- Meta加载 ---------- */
async function loadMeta() {
  try {
    const r = await fetch("/api/meta");
    const d = await r.json();
    const catMap = {};
    d.categories.forEach(c => catMap[c.key] = c);
    categoriesList.forEach(c => {
      const info = catMap[c.key] || {};
      c.name = info.name || c.key;
      c.description = info.description || "";
    });
    renderCategoryFilter();
    renderKB();
    renderStatusCats();
  } catch(e) { console.warn("loadMeta failed", e); }
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    byId("st_indexed").textContent = d.indexed_chunks || 0;
    byId("statusLoading").style.display = "none";
    byId("statusContent").style.display = "block";
  } catch(e) { console.warn("load health failed", e); }
}

/* ---------- 系统状态开关 ---------- */
function toggleStatus() {
  openStatus = !openStatus;
  byId("statusCard").style.display = openStatus ? "block" : "none";
}

/* ---------- 消息处理 ---------- */
function byId(id){ return document.getElementById(id); }
function avgLatency() {
  if (!latencies.length) return "-";
  const n = Math.min(5, latencies.length);
  return Math.round(latencies.slice(-n).reduce((a,b)=>a+b,0)/n);
}
function scrollBottom() {
  setTimeout(() => {
    const box = byId("msgList");
    if (box) box.scrollTop = box.scrollHeight;
  }, 20);
}

function appendUser(text) {
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `
    <div class="bubble"></div>
    <div class="avatar user">👤</div>`;
  row.querySelector(".bubble").textContent = text;
  byId("msgList").appendChild(row);
  scrollBottom();
}
function appendAI(text, loading=true, meta=null) {
  const row = document.createElement("div");
  row.className = "msg-row ai";
  row.innerHTML = `
    <div class="avatar ai">🤖</div>
    <div class="bubble"></div>`;
  const bubble = row.querySelector(".bubble");
  if (loading) {
    bubble.innerHTML = `<span class="loader"></span><span>${currentUI().thinking}</span>`;
  } else {
    bubble.textContent = text;
  }
  if (meta) {
    const faithHtml = meta.faith ? `<span class="tag-faith">🎯 Faith: ${meta.faith}</span>` : "";
    const hallucHtml = meta.halluc ? `<span class="tag-halluc">⚠️ Halluc: ${meta.halluc}</span>` : "";
    const metaHtml = `<div class="bubble-meta">
      <span>🏷️ ${meta.category || "-"}</span>
      <span>🌐 ${meta.language || "-"}</span>
      <span>⏱️ ${meta.latency}ms</span>
      ${faithHtml} ${hallucHtml}
    </div>`;
    bubble.insertAdjacentHTML("beforeend", metaHtml);
  }
  byId("msgList").appendChild(row);
  scrollBottom();
  return row;
}
function replaceLastAI(text, meta) {
  const list = byId("msgList");
  const rows = list.querySelectorAll(".msg-row.ai");
  if (!rows.length) return appendAI(text, false, meta);
  const last = rows[rows.length - 1];
  const bubble = last.querySelector(".bubble");
  bubble.textContent = text;
  if (meta) {
    const faithHtml = meta.faith ? `<span class="tag-faith">🎯 Faith: ${meta.faith}</span>` : "";
    const hallucHtml = meta.halluc ? `<span class="tag-halluc">⚠️ Halluc: ${meta.halluc}</span>` : "";
    const metaHtml = `<div class="bubble-meta">
      <span>🏷️ ${meta.category || "-"}</span>
      <span>🌐 ${meta.language || "-"}</span>
      <span>⏱️ ${meta.latency}ms</span>
      ${faithHtml} ${hallucHtml}
    </div>`;
    bubble.insertAdjacentHTML("beforeend", metaHtml);
  }
  scrollBottom();
}

/* ---------- 发送消息 ---------- */
async function sendMessage() {
  const input = byId("inputText");
  const text = input.value.trim();
  if (!text) return;
  appendUser(text);
  input.value = "";
  appendAI("", true); // loading气泡
  try {
    const resp = await fetch("/api/chat", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({
        message: text,
        session_id: "web_" + Date.now(),
        language: language === "zh" ? null : language,
        category_filter: categoryFilter || null,
      }),
    });
    const data = await resp.json();
    latencies.push(data.latency_ms || 0);
    byId("st_latency").textContent = avgLatency() + "ms";
    const faith = (data.trace && data.trace.safety_log && data.trace.safety_log.faithfulness)
      ? Math.round(data.trace.safety_log.faithfulness.score * 100) + "%" : null;
    const halluc = (data.trace && data.trace.safety_log && data.trace.safety_log.hallucination
      && data.trace.safety_log.hallucination.level !== "none") ? data.trace.safety_log.hallucination.level : null;
    const langName = LANG_META[data.language] ? LANG_META[data.language].label : (data.language || "zh");
    replaceLastAI(data.response || "", {
      category: data.category || "-",
      language: langName,
      latency: data.latency_ms || 0,
      faith, halluc,
    });
  } catch(err) {
    replaceLastAI("❌ Network error: " + err.message);
  }
}
</script>
</body>
</html>
"""
