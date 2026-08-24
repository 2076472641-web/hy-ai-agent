# -*- coding: utf-8 -*-
"""
Streamlit可视化运营后台 - 主应用
优店汇·商品类目智能研判分类系统

包含页面：
1. 单商品智能分类
2. 批量分类与核验
3. 运营数据看板
4. 模型管理与对比
"""

import os
import sys
import time
import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# 将项目根目录加入路径，便于导入utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    # 类目常量
    CATEGORY_NAMES, CATEGORY_ID_TO_NAME, CATEGORY_NAME_TO_ID,
    CATEGORY_COLORS, get_category_color, get_category_name, get_category_id,
    # API客户端
    api_client,
    # 可视化辅助
    format_confidence, confidence_to_color,
    pie_chart_data_from_counts, radar_chart_data,
    # 数据解析
    parse_uploaded_file, validate_dataframe, parse_pasted_text,
    dataframe_to_items, export_results_to_csv, validate_file_extension,
    DataValidationError,
    # 词云
    generate_wordcloud_image,
)

# ==================== 页面配置 ====================

st.set_page_config(
    page_title="优店汇·商品类目智能研判分类系统",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "优店汇·商品类目智能研判分类系统 v1.0.0\n基于NLP与机器学习的小程序商品自动化分类系统",
        "Get help": None,
        "Report a bug": None,
    },
)

# 自定义CSS样式
CUSTOM_CSS = """
<style>
    /* 全局字体优化 */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    }
    
    /* 主标题样式 */
    .main-title {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding: 0.5rem 0;
        margin-bottom: 1rem;
    }
    
    /* 卡片容器 */
    .card-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #f0f0f0;
        margin-bottom: 1rem;
    }
    
    /* 分类结果卡片 */
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border-left: 6px solid #667eea;
        margin: 1rem 0;
    }
    
    .result-card-success {
        border-left-color: #22c55e !important;
    }
    
    .result-card-warning {
        border-left-color: #eab308 !important;
    }
    
    .result-card-danger {
        border-left-color: #ef4444 !important;
    }
    
    /* 统计数字 */
    .stat-number {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        font-size: 0.95rem;
        color: #64748b;
        margin-top: 0.3rem;
    }
    
    /* 模型卡片 */
    .model-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }
    
    .model-card-default {
        border-color: #667eea !important;
        background: linear-gradient(135deg, #eef2ff 0%, #ffffff 100%);
    }
    
    /* 进度条容器 */
    .progress-wrapper {
        background: #e5e7eb;
        border-radius: 999px;
        height: 12px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 999px;
        transition: width 0.6s ease;
    }
    
    /* 徽章样式 */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .badge-success {
        background: #dcfce7;
        color: #166534;
    }
    
    .badge-warning {
        background: #fef3c7;
        color: #92400e;
    }
    
    .badge-danger {
        background: #fee2e2;
        color: #991b1b;
    }
    
    .badge-info {
        background: #dbeafe;
        color: #1e40af;
    }
    
    .badge-purple {
        background: #ede9fe;
        color: #6d28d9;
    }
    
    /* 隐藏Streamlit默认footer等 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 按钮样式增强 */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }
    
    /* TopK排行项 */
    .rank-item {
        display: flex;
        align-items: center;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        background: #f8fafc;
        transition: all 0.2s ease;
    }
    
    .rank-item:hover {
        background: #f1f5f9;
    }
    
    .rank-number {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.9rem;
        margin-right: 1rem;
        flex-shrink: 0;
    }
    
    .rank-1 { background: linear-gradient(135deg, #fbbf24, #f59e0b); color: white; }
    .rank-2 { background: linear-gradient(135deg, #94a3b8, #64748b); color: white; }
    .rank-3 { background: linear-gradient(135deg, #fb923c, #ea580c); color: white; }
    .rank-other { background: #e2e8f0; color: #475569; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==================== 通用组件 ====================

def render_status_banner():
    """渲染顶部状态横幅（显示API连接状态）"""
    is_mock = api_client.is_mock_mode
    if is_mock:
        st.warning(
            "⚠️ 当前运行于**模拟模式**：FastAPI后端 (http://localhost:8000) 不可达，"
            "正在使用内置模拟分类器兜底。启动后端服务可获得真实分类效果。",
            icon="🔌",
        )
    else:
        st.success(
            "✅ 已连接至 FastAPI 分类服务 (http://localhost:8000)",
            icon="🚀",
        )


def render_confidence_bar(confidence: float, show_label: bool = True, height: str = "20px"):
    """渲染置信度进度条（带颜色渐变）"""
    color = confidence_to_color(confidence)
    pct = confidence * 100
    bar_html = f"""
    <div style="margin: 0.3rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
            <span style="font-size: 0.85rem; color: #475569; font-weight: 500;">置信度</span>
            {f'<span style="font-size: 0.85rem; font-weight: 700; color: {color};">{pct:.2f}%</span>' if show_label else ''}
        </div>
        <div style="background: #e5e7eb; border-radius: 999px; height: {height}; overflow: hidden;">
            <div style="width: {pct}%; height: 100%; border-radius: 999px; 
                        background: linear-gradient(90deg, {color}dd, {color}); 
                        transition: width 0.6s ease;"></div>
        </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)


def render_stat_card(
    label: str,
    value: Any,
    suffix: str = "",
    delta: Optional[str] = None,
    color: str = "#667eea",
):
    """渲染统计数字卡片"""
    delta_html = ""
    if delta:
        delta_color = "#22c55e" if delta.startswith("+") else "#ef4444"
        delta_html = f'<div style="font-size: 0.85rem; color: {delta_color}; margin-top: 0.3rem; font-weight: 600;">{delta}</div>'
    
    html = f"""
    <div style="background: #ffffff; border-radius: 14px; padding: 1.2rem; 
                box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid #f0f0f0;">
        <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.5rem;">{label}</div>
        <div style="font-size: 2rem; font-weight: 700; 
                    background: linear-gradient(135deg, {color} 0%, #764ba2 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text;">
            {value}{suffix}
        </div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ==================== 页面一：单商品智能分类 ====================

def render_page_single_classification():
    """页面一：单商品智能分类"""
    st.markdown('<div class="main-title">🎯 单商品智能分类</div>', unsafe_allow_html=True)
    st.caption("输入商品信息，实时获取AI智能分类结果与置信度分析")
    
    render_status_banner()
    
    # 输入区域
    input_col1, input_col2 = st.columns([3, 2])
    
    with input_col1:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("📝 商品信息输入")
        
        title = st.text_input(
            "商品标题 *",
            placeholder="例如：夏季新款纯棉短袖T恤男士宽松半袖",
            max_chars=200,
            help="必填项，商品的主要标题",
        )
        
        detail = st.text_area(
            "商品详情（选填）",
            placeholder="例如：100%纯棉材质，舒适透气，修身版型，多色可选...",
            max_chars=1000,
            height=120,
            help="选填，商品的详细描述信息，有助于提高分类精度",
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with input_col2:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("⚙️ 分类参数设置")
        
        model = st.selectbox(
            "选择分类模型",
            options=["DistilledBERT", "BERT", "FastText", "RF"],
            index=0,
            format_func=lambda x: {
                "DistilledBERT": "🤖 蒸馏BERT（推荐 · 速度×精度平衡）",
                "BERT": "🧠 BERT全参微调（高精度 · 较慢）",
                "FastText": "⚡ FastText（极速 · 轻量）",
                "RF": "🌲 随机森林（传统ML · 最快）",
            }.get(x, x),
            help="不同模型在精度和速度间各有取舍",
        )
        
        top_k = st.slider(
            "Top-K 结果数量",
            min_value=1,
            max_value=12,
            value=5,
            step=1,
            help="返回置信度最高的前N个类目结果",
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        classify_btn = st.button(
            "🚀 开始智能分类",
            type="primary",
            use_container_width=True,
            disabled=not (title and title.strip()),
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 分类结果展示
    if classify_btn and title and title.strip():
        with st.spinner("🤖 AI正在分析商品信息..."):
            start_time = time.time()
            result = api_client.classify(
                title=title.strip(),
                detail=detail.strip(),
                model=model,
                top_k=top_k,
            )
            total_time = (time.time() - start_time) * 1000
        
        if not result.get("success"):
            st.error(f"分类失败：{result.get('error', '未知错误')}")
        else:
            # 主分类结果卡片
            pred = result["prediction"]
            conf = pred["confidence"]
            cat_id = pred["category_id"]
            cat_name = pred["category_name"]
            latency = result.get("latency_ms", total_time)
            
            # 置信度级别
            if conf >= 0.9:
                card_class = "result-card result-card-success"
                badge_class = "badge badge-success"
                badge_text = "高置信度"
            elif conf >= 0.7:
                card_class = "result-card result-card-warning"
                badge_class = "badge badge-warning"
                badge_text = "中置信度"
            else:
                card_class = "result-card result-card-danger"
                badge_class = "badge badge-danger"
                badge_text = "低置信度"
            
            st.markdown("---")
            st.subheader("🔍 分类结果")
            
            result_col1, result_col2 = st.columns([3, 2])
            
            with result_col1:
                st.markdown(
                    f'''
                    <div class="{card_class}">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
                            <div>
                                <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.3rem;">预测主分类</div>
                                <div style="font-size: 2rem; font-weight: 700; color: #1e293b;">{cat_name}</div>
                                <div style="font-size: 0.95rem; color: #64748b; margin-top: 0.3rem;">类目ID: {cat_id}</div>
                            </div>
                            <div>
                                <span class="{badge_class}">{badge_text}</span>
                            </div>
                        </div>
                    ''',
                    unsafe_allow_html=True,
                )
                render_confidence_bar(conf, height="24px")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 推理耗时
                time_col1, time_col2, time_col3 = st.columns(3)
                with time_col1:
                    render_stat_card("模型推理耗时", f"{latency:.1f}", " ms", color="#22c55e")
                with time_col2:
                    render_stat_card("端到端耗时", f"{total_time:.1f}", " ms", color="#667eea")
                with time_col3:
                    model_badge = {
                        "RF": "🌲 RF", "FastText": "⚡ FastText",
                        "BERT": "🧠 BERT", "DistilledBERT": "🤖 DistilBERT"
                    }.get(model, model)
                    st.markdown(
                        f'''
                        <div style="background: #ffffff; border-radius: 14px; padding: 1.2rem; 
                                    box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid #f0f0f0;">
                            <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.5rem;">使用模型</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: #764ba2;">{model_badge}</div>
                        </div>
                        ''',
                        unsafe_allow_html=True,
                    )
            
            with result_col2:
                # TopK分类排行榜
                st.markdown(
                    f'''
                    <div class="card-container" style="margin: 0;">
                        <div style="font-size: 1.1rem; font-weight: 700; color: #1e293b; margin-bottom: 1rem;">
                            🏆 Top-{top_k} 分类排行榜
                        </div>
                    ''',
                    unsafe_allow_html=True,
                )
                
                for idx, item in enumerate(result["top_k"]):
                    rank = idx + 1
                    rank_class = f"rank-{rank}" if rank <= 3 else "rank-other"
                    item_conf = item["confidence"]
                    item_color = confidence_to_color(item_conf)
                    
                    st.markdown(
                        f'''
                        <div class="rank-item">
                            <div class="rank-number {rank_class}">{rank}</div>
                            <div style="flex: 1; min-width: 0;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                                    <span style="font-weight: 600; color: #1e293b;">{item["category_name"]}</span>
                                    <span style="font-size: 0.85rem; font-weight: 700; color: {item_color};">
                                        {item_conf * 100:.1f}%
                                    </span>
                                </div>
                                <div style="background: #e5e7eb; border-radius: 999px; height: 6px; overflow: hidden;">
                                    <div style="width: {item_conf * 100}%; height: 100%; 
                                                background: {item_color}; border-radius: 999px;"></div>
                                </div>
                            </div>
                        </div>
                        ''',
                        unsafe_allow_html=True,
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 纠错反馈区域
            st.markdown("---")
            with st.expander("📝 发现分类错误？提交纠错反馈", expanded=False):
                st.info("💡 您的反馈将帮助我们持续优化分类模型")
                feedback_col1, feedback_col2 = st.columns([3, 2])
                with feedback_col1:
                    correct_cat = st.selectbox(
                        "请选择正确的类目",
                        options=CATEGORY_NAMES,
                        index=cat_id if 0 <= cat_id < len(CATEGORY_NAMES) else 0,
                        format_func=lambda x: f"{get_category_id(x)} - {x}",
                        key="single_feedback_cat",
                    )
                with feedback_col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    feedback_btn = st.button(
                        "✅ 提交纠错反馈",
                        type="secondary",
                        use_container_width=True,
                        key="single_feedback_btn",
                    )
                
                if feedback_btn:
                    correct_id = get_category_id(correct_cat)
                    if correct_id == cat_id:
                        st.warning("您选择的类目与预测结果相同，无需提交反馈")
                    else:
                        with st.spinner("正在提交反馈..."):
                            fb_result = api_client.submit_feedback(
                                title=title.strip(),
                                detail=detail.strip(),
                                predicted_id=cat_id,
                                correct_id=correct_id,
                                model=model,
                            )
                            if fb_result.get("success"):
                                mode_note = "（模拟模式）" if fb_result.get("mock_mode") else ""
                                st.success(f"✅ 反馈提交成功{mode_note}！感谢您的贡献")
                            else:
                                st.error("反馈提交失败，请稍后重试")


# ==================== 页面二：批量分类与核验 ====================

def render_page_batch_classification():
    """页面二：批量分类与核验"""
    st.markdown('<div class="main-title">📊 批量分类与核验</div>', unsafe_allow_html=True)
    st.caption("批量上传商品数据，快速完成分类标注与人工核验")
    
    render_status_banner()
    
    # 初始化session_state
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []
    if "batch_df" not in st.session_state:
        st.session_state.batch_df = pd.DataFrame()
    if "verified_items" not in st.session_state:
        st.session_state.verified_items = {}  # {index: "correct"/{"category_id": x}}
    
    # 数据输入区域
    st.subheader("📥 数据输入")
    input_tab1, input_tab2 = st.tabs(["📁 上传文件", "✏️ 手动粘贴"])
    
    with input_tab1:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "上传 CSV 或 Excel 文件",
            type=["csv", "xlsx", "xls"],
            help="文件需包含 title 列（必填）和 detail 列（选填），最大支持500条",
        )
        if uploaded_file:
            try:
                df = parse_uploaded_file(uploaded_file)
                is_valid, issues = validate_dataframe(df)
                if issues:
                    for issue in issues:
                        st.warning(f"⚠️ {issue}")
                st.info(f"📋 文件解析成功，共 {len(df)} 条记录，列：{', '.join(df.columns.tolist())}")
                st.dataframe(df.head(5), use_container_width=True, hide_index=True)
                
                if st.button("📥 确认导入此文件数据", use_container_width=True, key="file_import_btn"):
                    items = dataframe_to_items(df, max_items=500)
                    st.session_state.batch_df = df.head(500)
                    st.success(f"✅ 已导入 {len(items)} 条待分类记录")
            except DataValidationError as e:
                st.error(f"❌ 文件校验失败：{e}")
            except Exception as e:
                st.error(f"❌ 文件解析失败：{e}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with input_tab2:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.write("**格式说明**：每行一条商品，标题和详情用 **Tab** 键分隔（详情可选）")
        example_text = """夏季纯棉短袖T恤\t100%纯棉 修身版型 透气舒适
华为Mate60 Pro 手机\t5G 256GB 昆仑玻璃
补水面膜玻尿酸精华\t深层补水 敏感肌可用
进口婴儿纸尿裤\t超薄透气 L码 50片
便携式户外帐篷\t3-4人 双层防雨 自动速开"""
        
        pasted_text = st.text_area(
            "粘贴商品数据（每行一条）",
            placeholder=example_text,
            height=200,
            help="每行一条记录，用Tab分隔标题和详情",
        )
        col_p1, col_p2 = st.columns([3, 1])
        with col_p2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📋 解析粘贴内容", use_container_width=True, key="parse_paste_btn"):
                if not pasted_text.strip():
                    st.warning("请先粘贴商品数据")
                else:
                    df = parse_pasted_text(pasted_text)
                    items = dataframe_to_items(df, max_items=500)
                    st.session_state.batch_df = df
                    st.success(f"✅ 已解析 {len(items)} 条记录")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 显示已导入数据预览
    if not st.session_state.batch_df.empty:
        st.markdown("---")
        
        # 模型选择与执行
        st.subheader("⚙️ 批量分类设置")
        batch_col1, batch_col2 = st.columns([2, 1])
        with batch_col1:
            batch_model = st.selectbox(
                "选择分类模型",
                options=["DistilledBERT", "BERT", "FastText", "RF"],
                index=0,
                key="batch_model_select",
            )
        with batch_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_batch_btn = st.button(
                "🚀 执行批量分类",
                type="primary",
                use_container_width=True,
                key="run_batch_btn",
            )
        
        if run_batch_btn:
            items = dataframe_to_items(st.session_state.batch_df, max_items=500)
            progress_bar = st.progress(0, text=f"准备处理 {len(items)} 条商品...")
            
            # 分批处理并显示进度
            batch_size = 50
            all_results = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_results = api_client.batch_classify(batch, model=batch_model)
                all_results.extend(batch_results)
                progress = min(1.0, (i + len(batch)) / len(items))
                progress_bar.progress(
                    progress,
                    text=f"已处理 {min(i + len(batch), len(items))} / {len(items)} 条",
                )
                time.sleep(0.1)
            
            st.session_state.batch_results = all_results
            st.session_state.verified_items = {}
            progress_bar.empty()
            st.success(f"✅ 批量分类完成！共处理 {len(all_results)} 条商品")
    
    # 结果展示与核验
    if st.session_state.batch_results:
        results = st.session_state.batch_results
        
        # 统计面板
        st.markdown("---")
        st.subheader("📈 分类统计概览")
        
        total_count = len(results)
        confidences = [r.get("prediction", {}).get("confidence", 0) for r in results]
        high_conf_count = sum(1 for c in confidences if c > 0.9)
        high_conf_ratio = high_conf_count / total_count if total_count > 0 else 0
        
        # 类目分布
        category_counts: Dict[str, int] = {}
        for r in results:
            cat_name = r.get("prediction", {}).get("category_name", "未知")
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            render_stat_card("总数量", total_count, " 条", color="#667eea")
        with stat_col2:
            render_stat_card(
                "高置信度(>0.9)",
                high_conf_count,
                " 条",
                f"+{high_conf_ratio * 100:.1f}%",
                color="#22c55e",
            )
        with stat_col3:
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            render_stat_card("平均置信度", f"{avg_conf * 100:.1f}", "%", color="#f59e0b")
        with stat_col4:
            verified_count = len(st.session_state.verified_items)
            render_stat_card(
                "已核验",
                verified_count,
                f" / {total_count}",
                f"{verified_count / total_count * 100:.1f}% 完成",
                color="#764ba2",
            )
        
        # 类目分布饼图
        pie_col1, pie_col2 = st.columns([2, 1])
        with pie_col1:
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.markdown("#### 🥧 各类目分布")
            pie_data = pie_chart_data_from_counts(category_counts)
            pie_option = {
                "tooltip": {"trigger": "item", "formatter": "{b}: {c}条 ({d}%)"},
                "legend": {
                    "orient": "vertical",
                    "left": "left",
                    "top": "middle",
                    "textStyle": {"fontSize": 12},
                },
                "series": [
                    {
                        "name": "类目分布",
                        "type": "pie",
                        "radius": ["40%", "70%"],
                        "center": ["65%", "50%"],
                        "avoidLabelOverlap": True,
                        "itemStyle": {
                            "borderRadius": 8,
                            "borderColor": "#fff",
                            "borderWidth": 2,
                        },
                        "label": {"show": True, "formatter": "{b}\n{d}%"},
                        "labelLine": {"show": True},
                        "data": pie_data,
                    }
                ],
            }
            st_echarts(pie_option, height="380px")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with pie_col2:
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.markdown("#### 📊 分布明细")
            dist_df = pd.DataFrame(
                [
                    {
                        "类目名称": name,
                        "数量": cnt,
                        "占比": f"{cnt / total_count * 100:.1f}%",
                    }
                    for name, cnt in sorted(category_counts.items(), key=lambda x: -x[1])
                ]
            )
            st.dataframe(
                dist_df,
                use_container_width=True,
                hide_index=True,
                height=360,
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 结果核验表格
        st.markdown("---")
        st.subheader("✅ 分类结果核验")
        st.caption("表格中可标记每条结果是否正确，对错误结果选择正确类目")
        
        # 构造表格数据
        table_rows = []
        for idx, r in enumerate(results):
            pred = r.get("prediction", {})
            conf = pred.get("confidence", 0)
            verified = st.session_state.verified_items.get(idx)
            if verified == "correct":
                verify_status = "✅ 已标记正确"
            elif isinstance(verified, dict) and "category_id" in verified:
                verify_status = f"🔧 已纠错 → {get_category_name(verified['category_id'])}"
            else:
                verify_status = "⏳ 待核验"
            
            table_rows.append({
                "序号": idx + 1,
                "商品标题": r.get("title", ""),
                "预测类目": pred.get("category_name", ""),
                "置信度": f"{conf * 100:.1f}%",
                "置信值": round(conf, 4),
                "核验状态": verify_status,
            })
        
        display_df = pd.DataFrame(table_rows)
        
        # 操作栏
        action_col1, action_col2, action_col3 = st.columns([2, 2, 3])
        with action_col1:
            verify_idx = st.number_input(
                "选择记录序号进行核验",
                min_value=1,
                max_value=total_count,
                value=1,
                step=1,
                key="verify_idx_input",
            )
        with action_col2:
            verify_action = st.radio(
                "标记操作",
                options=["标记正确", "标记错误并选择正确类目"],
                horizontal=True,
                key="verify_action_radio",
            )
        with action_col3:
            if verify_action == "标记错误并选择正确类目":
                correct_category = st.selectbox(
                    "选择正确类目",
                    options=CATEGORY_NAMES,
                    key="batch_correct_cat",
                )
            else:
                correct_category = None
            
            verify_btn = st.button(
                "📝 提交核验结果",
                type="primary",
                use_container_width=True,
                key="batch_verify_btn",
            )
        
        if verify_btn:
            idx = verify_idx - 1
            if verify_action == "标记正确":
                st.session_state.verified_items[idx] = "correct"
                st.success(f"✅ 第 {verify_idx} 条已标记为正确")
            else:
                cat_id = get_category_id(correct_category)
                st.session_state.verified_items[idx] = {"category_id": cat_id}
                st.success(f"🔧 第 {verify_idx} 条已纠错为：{correct_category}")
            st.rerun()
        
        # 用AgGrid展示可编辑表格
        gb = GridOptionsBuilder.from_dataframe(display_df)
        gb.configure_column("序号", width=60, pinned="left")
        gb.configure_column("商品标题", width=280, wrapText=True, autoHeight=True)
        gb.configure_column("预测类目", width=120)
        gb.configure_column("置信度", width=90)
        gb.configure_column("置信值", hide=True)
        gb.configure_column("核验状态", width=200)
        gb.configure_default_column(filter=True, sortable=True, resizable=True)
        gb.configure_grid_options(
            domLayout="normal",
            rowHeight=56,
            enableRangeSelection=True,
        )
        grid_options = gb.build()
        
        AgGrid(
            display_df,
            gridOptions=grid_options,
            height=480,
            use_container_width=True,
            update_mode=GridUpdateMode.NO_UPDATE,
            data_return_mode=DataReturnMode.FILTERED,
            fit_columns_on_grid_load=False,
            allow_unsafe_jscode=True,
        )
        
        # 导出操作
        st.markdown("---")
        export_col1, export_col2 = st.columns([1, 4])
        with export_col1:
            csv_bytes = export_results_to_csv(results)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📥 导出分类结果 CSV",
                data=csv_bytes,
                file_name=f"batch_classification_result_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )
        with export_col2:
            if st.session_state.verified_items:
                st.success(f"📊 已核验 {len(st.session_state.verified_items)} 条数据，可提交反馈用于模型迭代")
            else:
                st.info("💡 提示：请先对分类结果进行人工核验")


# ==================== 页面三：运营数据看板 ====================

def render_page_dashboard():
    """页面三：运营数据看板"""
    st.markdown('<div class="main-title">📈 运营数据看板</div>', unsafe_allow_html=True)
    st.caption("多维度展示分类服务运营数据，洞察模型效果与用户行为")
    
    render_status_banner()
    
    # 时间范围选择
    period_col1, _ = st.columns([1, 4])
    with period_col1:
        period = st.selectbox(
            "📅 选择时间范围",
            options=["today", "7d", "30d"],
            index=1,
            format_func=lambda x: {
                "today": "📆 今日数据",
                "7d": "📊 近7天",
                "30d": "📉 近30天",
            }.get(x, x),
            key="dashboard_period",
        )
    
    # 获取看板数据
    with st.spinner("加载看板数据..."):
        stats = api_client.get_dashboard_stats(period)
    
    # 顶部核心指标
    st.markdown("---")
    total_requests = stats.get("total_requests", 0)
    avg_accuracy = stats.get("avg_accuracy", 0)
    
    core_col1, core_col2, core_col3, core_col4 = st.columns(4)
    with core_col1:
        render_stat_card("分类请求总量", f"{total_requests:,}", " 次", "+12.5%", color="#667eea")
    with core_col2:
        render_stat_card("平均准确率", f"{avg_accuracy * 100:.1f}", "%", "+0.8%", color="#22c55e")
    with core_col3:
        render_stat_card("接入模型数", 4, " 个", color="#f59e0b")
    with core_col4:
        render_stat_card("覆盖类目数", 12, " 类", color="#764ba2")
    
    # 图表区域
    st.markdown("---")
    
    # 分类请求量折线图
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.subheader("📊 分类请求量趋势")
    line_option = {
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "cross", "label": {"backgroundColor": "#6a7985"}},
        },
        "legend": {"data": ["分类请求量"], "top": 0},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "top": "15%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": stats.get("dates", []),
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
        },
        "yAxis": {
            "type": "value",
            "axisLine": {"show": False},
            "splitLine": {"lineStyle": {"color": "#f1f5f9"}},
        },
        "series": [
            {
                "name": "分类请求量",
                "type": "line",
                "smooth": True,
                "symbol": "circle",
                "symbolSize": 8,
                "itemStyle": {"color": "#667eea"},
                "areaStyle": {
                    "color": {
                        "type": "linear",
                        "x": 0,
                        "y": 0,
                        "x2": 0,
                        "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": "rgba(102, 126, 234, 0.4)"},
                            {"offset": 1, "color": "rgba(102, 126, 234, 0.05)"},
                        ],
                    }
                },
                "lineStyle": {"width": 3, "color": "#667eea"},
                "data": stats.get("request_counts", []),
            }
        ],
    }
    st_echarts(line_option, height="320px")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 第二行：模型使用占比 + 准确率趋势
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("🤖 各模型使用占比")
        model_usage = stats.get("model_usage", {})
        model_colors = ["#5470c6", "#91cc75", "#fac858", "#ee6666"]
        donut_data = []
        for i, (model_name, pct) in enumerate(model_usage.items()):
            donut_data.append({
                "value": pct,
                "name": model_name,
                "itemStyle": {"color": model_colors[i % len(model_colors)]},
            })
        donut_option = {
            "tooltip": {"trigger": "item", "formatter": "{b}: {c}%"},
            "legend": {"bottom": "0%", "left": "center"},
            "series": [
                {
                    "name": "模型使用占比",
                    "type": "pie",
                    "radius": ["45%", "70%"],
                    "center": ["50%", "45%"],
                    "avoidLabelOverlap": True,
                    "itemStyle": {
                        "borderRadius": 10,
                        "borderColor": "#fff",
                        "borderWidth": 3,
                    },
                    "label": {"show": True, "formatter": "{b}\n{c}%", "fontSize": 12},
                    "labelLine": {"show": True, "length": 15, "length2": 10},
                    "data": donut_data,
                }
            ],
        }
        st_echarts(donut_option, height="340px")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with row2_col2:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("🎯 分类准确率趋势")
        acc_data = [round(x * 100, 2) for x in stats.get("accuracy_trend", [])]
        acc_line_option = {
            "tooltip": {
                "trigger": "axis",
                "formatter": "{b}<br/>准确率: {c}%",
                "axisPointer": {"type": "shadow"},
            },
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "top": "10%", "containLabel": True},
            "xAxis": {
                "type": "category",
                "data": stats.get("dates", []),
                "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            },
            "yAxis": {
                "type": "value",
                "min": 70,
                "max": 100,
                "axisLabel": {"formatter": "{value}%"},
                "splitLine": {"lineStyle": {"color": "#f1f5f9"}},
            },
            "series": [
                {
                    "name": "准确率",
                    "type": "line",
                    "smooth": True,
                    "symbol": "circle",
                    "symbolSize": 8,
                    "itemStyle": {"color": "#22c55e"},
                    "areaStyle": {
                        "color": {
                            "type": "linear",
                            "x": 0, "y": 0, "x2": 0, "y2": 1,
                            "colorStops": [
                                {"offset": 0, "color": "rgba(34, 197, 94, 0.35)"},
                                {"offset": 1, "color": "rgba(34, 197, 94, 0.05)"},
                            ],
                        }
                    },
                    "lineStyle": {"width": 3, "color": "#22c55e"},
                    "markLine": {
                        "data": [{"type": "average", "name": "平均值"}],
                        "lineStyle": {"color": "#eab308"},
                    },
                    "data": acc_data,
                }
            ],
        }
        st_echarts(acc_line_option, height="340px")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 第三行：各类目准确率 + 高频纠错词云
    row3_col1, row3_col2 = st.columns(2)
    
    with row3_col1:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("📊 各类目分类准确率")
        cat_acc = stats.get("category_accuracy", {})
        cat_names = list(cat_acc.keys())
        cat_acc_values = [round(v * 100, 2) for v in cat_acc.values()]
        cat_colors_list = [get_category_color(get_category_id(name)) for name in cat_names]
        
        bar_option = {
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "shadow"},
                "formatter": "{b}<br/>准确率: {c}%",
            },
            "grid": {"left": "3%", "right": "4%", "bottom": "15%", "top": "5%", "containLabel": True},
            "xAxis": {
                "type": "category",
                "data": cat_names,
                "axisLabel": {
                    "interval": 0,
                    "rotate": 35,
                    "fontSize": 11,
                },
                "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            },
            "yAxis": {
                "type": "value",
                "min": 60,
                "max": 100,
                "axisLabel": {"formatter": "{value}%"},
                "splitLine": {"lineStyle": {"color": "#f1f5f9"}},
            },
            "series": [
                {
                    "name": "准确率",
                    "type": "bar",
                    "barWidth": "60%",
                    "itemStyle": {
                        "borderRadius": [8, 8, 0, 0],
                        "color": {
                            "type": "linear",
                            "x": 0, "y": 0, "x2": 0, "y2": 1,
                            "colorStops": [
                                {"offset": 0, "color": "#764ba2"},
                                {"offset": 1, "color": "#667eea"},
                            ],
                        },
                    },
                    "label": {
                        "show": True,
                        "position": "top",
                        "formatter": "{c}%",
                        "fontSize": 10,
                    },
                    "data": [
                        {"value": v, "itemStyle": {"color": cat_colors_list[i]}}
                        for i, v in enumerate(cat_acc_values)
                    ],
                }
            ],
        }
        st_echarts(bar_option, height="360px")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with row3_col2:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("☁️ 高频纠错词云")
        st.caption("从用户反馈的纠错样本中提取的高频关键词")
        
        keyword_freq = stats.get("feedback_keywords", {})
        wordcloud_img = generate_wordcloud_image(keyword_freq, width=700, height=300)
        
        if wordcloud_img:
            st.image(wordcloud_img, use_container_width=True, caption="图中词越大表示该词出现的纠错反馈越多")
        else:
            # 词云生成失败时用Table展示高频词
            st.info("词云组件未安装，以下展示高频词列表")
            kw_df = pd.DataFrame(
                [
                    {"关键词": k, "频次": v}
                    for k, v in sorted(keyword_freq.items(), key=lambda x: -x[1])[:20]
                ]
            )
            st.dataframe(kw_df, use_container_width=True, hide_index=True, height=320)
        st.markdown('</div>', unsafe_allow_html=True)


# ==================== 页面四：模型管理与对比 ====================

def render_page_model_management():
    """页面四：模型管理与对比"""
    st.markdown('<div class="main-title">🧠 模型管理与对比</div>', unsafe_allow_html=True)
    st.caption("查看各模型详情指标，对比性能差异，管理默认模型")
    
    render_status_banner()
    
    # 获取模型信息
    with st.spinner("加载模型信息..."):
        model_info = api_client.get_model_info()
    
    models_data = model_info.get("models", {})
    default_model = model_info.get("default_model", "DistilledBERT")
    
    # 模型卡片网格
    st.subheader("📋 模型列表")
    
    model_names = ["RF", "FastText", "BERT", "DistilledBERT"]
    card_col1, card_col2, card_col3, card_col4 = st.columns(4)
    card_cols = [card_col1, card_col2, card_col3, card_col4]
    
    model_icons = {
        "RF": "🌲",
        "FastText": "⚡",
        "BERT": "🧠",
        "DistilledBERT": "🤖",
    }
    
    for idx, (col, model_name) in enumerate(zip(card_cols, model_names)):
        m = models_data.get(model_name, {})
        is_default = (model_name == default_model)
        card_class = "model-card model-card-default" if is_default else "model-card"
        
        status = m.get("status", "loaded")
        status_color = "#22c55e" if status == "loaded" else "#f59e0b"
        status_text = "🟢 已加载" if status == "loaded" else "🟡 加载中"
        
        with col:
            st.markdown(
                f'''
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.8rem;">
                        <div>
                            <div style="font-size: 1.5rem;">{model_icons.get(model_name, "📦")}</div>
                            <div style="font-size: 1.1rem; font-weight: 700; color: #1e293b; margin-top: 0.3rem;">
                                {model_name}
                            </div>
                            <div style="font-size: 0.75rem; color: #64748b;">{m.get("version", "1.0.0")}</div>
                        </div>
                        {f'<span class="badge badge-purple" style="font-size: 0.7rem;">⭐ 默认</span>' if is_default else ''}
                    </div>
                    <div style="font-size: 0.85rem; color: #475569; margin-bottom: 1rem; line-height: 1.5;">
                        {m.get("name", "")}
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; font-size: 0.85rem;">
                        <div>
                            <span style="color: #64748b;">F1-macro</span>
                            <div style="font-weight: 700; color: #667eea;">{m.get("f1_macro", 0) * 100:.2f}%</div>
                        </div>
                        <div>
                            <span style="color: #64748b;">参数量</span>
                            <div style="font-weight: 700; color: #764ba2;">{m.get("params_count", "-")}</div>
                        </div>
                        <div>
                            <span style="color: #64748b;">模型大小</span>
                            <div style="font-weight: 700; color: #f59e0b;">{m.get("model_size", "-")}</div>
                        </div>
                        <div>
                            <span style="color: #64748b;">平均耗时</span>
                            <div style="font-weight: 700; color: #22c55e;">{m.get("avg_latency_ms", "-")} ms</div>
                        </div>
                    </div>
                    <div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid #f0f0f0; display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.8rem; color: {status_color};">{status_text}</span>
                    </div>
                </div>
                ''',
                unsafe_allow_html=True,
            )
    
    # 模型管理操作
    st.markdown("---")
    st.subheader("⚙️ 模型管理")
    
    mgmt_col1, mgmt_col2, mgmt_col3 = st.columns(3)
    with mgmt_col1:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown("#### 🎯 设置默认模型")
        new_default = st.selectbox(
            "选择默认分类模型",
            options=model_names,
            index=model_names.index(default_model) if default_model in model_names else 0,
            key="new_default_model",
            help="未指定模型时将使用此默认模型进行分类",
        )
        if st.button("💾 保存默认模型设置", use_container_width=True, type="primary"):
            with st.spinner("正在更新设置..."):
                result = api_client.set_default_model(new_default)
                if result.get("success"):
                    mode_note = "（模拟模式）" if result.get("mock_mode") else ""
                    st.success(f"✅ {result.get('message')}{mode_note}")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("设置失败")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with mgmt_col2:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown("#### 🔄 模型操作")
        action_model = st.selectbox(
            "选择模型",
            options=model_names,
            key="action_model_select",
        )
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            st.button("🔄 重新加载", use_container_width=True, key=f"reload_{action_model}")
        with act_col2:
            st.button("📊 查看日志", use_container_width=True, key=f"log_{action_model}")
        st.caption("支持模型热加载与运行状态监控")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with mgmt_col3:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown("#### 📊 加载状态总览")
        loaded_count = sum(
            1 for m in models_data.values() if m.get("status") == "loaded"
        )
        st.markdown(
            f'''
            <div style="text-align: center; padding: 1rem 0;">
                <div style="font-size: 2.5rem; font-weight: 700; 
                            background: linear-gradient(135deg, #22c55e, #667eea);
                            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {loaded_count} / 4
                </div>
                <div style="font-size: 0.9rem; color: #64748b; margin-top: 0.3rem;">
                    模型已加载就绪
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        for mn in model_names:
            ms = models_data.get(mn, {}).get("status", "unknown")
            dot = "🟢" if ms == "loaded" else "🟡"
            st.markdown(f"<div style='font-size: 0.85rem;'>{dot} {mn}: {'已加载' if ms == 'loaded' else '加载中'}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 雷达图对比
    st.markdown("---")
    st.subheader("📡 多维度性能对比（雷达图）")
    st.caption("6大维度全面对比各模型的综合表现，分值越高越好")
    
    # 构建雷达图数据
    model_metrics_for_radar = {}
    for mn in model_names:
        m = models_data.get(mn, {})
        model_metrics_for_radar[mn] = {
            "accuracy": m.get("accuracy", 0) * 100,
            "precision": m.get("precision", 0) * 100,
            "recall": m.get("recall", 0) * 100,
            "f1_macro": m.get("f1_macro", 0) * 100,
            "speed_score": m.get("speed_score", 50),
            "lightweight_score": m.get("lightweight_score", 50),
        }
    
    radar_data = radar_chart_data(model_metrics_for_radar)
    
    # 图例开关
    legend_models = st.multiselect(
        "选择要对比的模型（至少选择1个）",
        options=model_names,
        default=model_names,
        key="radar_models",
    )
    
    if legend_models:
        filtered_series = [
            s for s in radar_data["series_data"] if s["name"] in legend_models
        ]
        radar_option = {
            "tooltip": {"trigger": "item"},
            "legend": {
                "data": [s["name"] for s in filtered_series],
                "top": 0,
                "bottom": None,
            },
            "radar": {
                "indicator": radar_data["indicators"],
                "shape": "polygon",
                "splitNumber": 5,
                "axisName": {
                    "color": "#475569",
                    "fontSize": 13,
                    "fontWeight": 600,
                },
                "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
                "splitArea": {
                    "areaStyle": {
                        "color": ["#ffffff", "#f8fafc", "#f1f5f9", "#e2e8f0", "#ffffff"]
                    }
                },
                "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            },
            "series": [
                {
                    "type": "radar",
                    "data": filtered_series,
                    "emphasis": {"lineStyle": {"width": 4}},
                }
            ],
        }
        
        radar_col1, radar_col2 = st.columns([3, 2])
        with radar_col1:
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st_echarts(radar_option, height="500px")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with radar_col2:
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.markdown("#### 📖 维度说明")
            st.markdown(
                """
                - **准确率 (Accuracy)**：整体分类正确率
                - **精确率 (Precision)**：预测为正的样本中实际为正的比例
                - **召回率 (Recall)**：实际为正的样本中被预测正确的比例
                - **F1宏平均 (F1-macro)**：各类目F1的平均值，综合指标
                - **推理速度**：单样本平均耗时，越快分值越高
                - **轻量度**：模型大小和资源占用，越小越轻量分值越高
                """
            )
            st.markdown("---")
            
            st.markdown("#### 🥇 各维度冠军模型")
            dims = [
                ("准确率", "accuracy", "#667eea"),
                ("精确率", "precision", "#764ba2"),
                ("召回率", "recall", "#f59e0b"),
                ("F1宏平均", "f1_macro", "#22c55e"),
                ("推理速度", "speed_score", "#06b6d4"),
                ("轻量度", "lightweight_score", "#ec4899"),
            ]
            for dim_name, dim_key, dim_color in dims:
                best_model = max(
                    model_names,
                    key=lambda x: model_metrics_for_radar[x][dim_key],
                )
                best_val = model_metrics_for_radar[best_model][dim_key]
                st.markdown(
                    f'''
                    <div style="display: flex; justify-content: space-between; align-items: center; 
                                padding: 0.5rem 0.8rem; background: #f8fafc; border-radius: 8px; 
                                margin-bottom: 0.4rem; border-left: 4px solid {dim_color};">
                        <span style="font-size: 0.9rem; color: #475569;">{dim_name}</span>
                        <div style="text-align: right;">
                            <span style="font-weight: 700; color: #1e293b;">{best_model}</span>
                            <span style="font-size: 0.8rem; color: #64748b; margin-left: 0.5rem;">
                                ({best_val:.1f}分)
                            </span>
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("请至少选择一个模型进行对比")
    
    # 详细指标对比表
    st.markdown("---")
    st.subheader("📋 详细指标对比表")
    
    comparison_rows = []
    for mn in model_names:
        m = models_data.get(mn, {})
        comparison_rows.append({
            "模型": mn,
            "版本": m.get("version", "-"),
            "准确率": f"{m.get('accuracy', 0) * 100:.2f}%",
            "精确率": f"{m.get('precision', 0) * 100:.2f}%",
            "召回率": f"{m.get('recall', 0) * 100:.2f}%",
            "F1-macro": f"{m.get('f1_macro', 0) * 100:.2f}%",
            "参数量": m.get("params_count", "-"),
            "模型大小": m.get("model_size", "-"),
            "平均耗时(ms)": m.get("avg_latency_ms", "-"),
            "加载状态": "✅ 已加载" if m.get("status") == "loaded" else "⏳ 加载中",
            "是否默认": "⭐ 是" if mn == default_model else "-",
        })
    
    comp_df = pd.DataFrame(comparison_rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True, height=200)


# ==================== 侧边栏与主入口 ====================

def render_sidebar():
    """渲染侧边栏导航"""
    with st.sidebar:
        # Logo区域
        st.markdown(
            '''
            <div style="text-align: center; padding: 1rem 0.5rem 1.5rem;">
                <div style="font-size: 2.5rem;">🏷️</div>
                <div style="font-size: 1.3rem; font-weight: 700; 
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                            background-clip: text; margin-top: 0.5rem;">
                    优店汇
                </div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.3rem;">
                    商品类目智能研判分类系统
                </div>
                <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 0.2rem;">
                    v1.0.0
                </div>
            </div>
            <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 0.5rem 0 1.5rem;">
            ''',
            unsafe_allow_html=True,
        )
        
        # 页面导航
        page = st.radio(
            "📌 功能导航",
            options=[
                "单商品智能分类",
                "批量分类与核验",
                "运营数据看板",
                "模型管理与对比",
            ],
            index=0,
            key="page_radio",
        )
        
        st.markdown(
            '''
            <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0 1rem;">
            ''',
            unsafe_allow_html=True,
        )
        
        # 系统状态
        st.markdown("#### 🖥️ 系统状态")
        is_mock = api_client.is_mock_mode
        status_emoji = "🟡" if is_mock else "🟢"
        status_text = "模拟模式" if is_mock else "API已连接"
        st.markdown(
            f'''
            <div style="background: white; border-radius: 10px; padding: 0.8rem 1rem; 
                        box-shadow: 0 1px 4px rgba(0,0,0,0.05); margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.85rem; color: #64748b;">后端服务</span>
                    <span style="font-size: 0.8rem; font-weight: 600;">{status_emoji} {status_text}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem;">
                    <span style="font-size: 0.85rem; color: #64748b;">可用模型</span>
                    <span style="font-size: 0.8rem; font-weight: 600;">4 / 4 ✅</span>
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        
        # 快捷操作
        st.markdown("#### 🔗 快捷操作")
        st.link_button("📖 项目文档", "https://github.com/", use_container_width=True)
        st.button("💬 问题反馈", use_container_width=True)
        
        # 版权
        st.markdown(
            '''
            <div style="position: fixed; bottom: 1rem; left: 1rem; right: 1rem; 
                        text-align: center; font-size: 0.7rem; color: #94a3b8;">
                © 2024 NLP工程团队 · 优店汇
            </div>
            ''',
            unsafe_allow_html=True,
        )
        
        return page


def main():
    """主应用入口"""
    page = render_sidebar()
    
    if page == "单商品智能分类":
        render_page_single_classification()
    elif page == "批量分类与核验":
        render_page_batch_classification()
    elif page == "运营数据看板":
        render_page_dashboard()
    elif page == "模型管理与对比":
        render_page_model_management()


if __name__ == "__main__":
    main()
