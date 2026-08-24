# -*- coding: utf-8 -*-
"""
EDA探索性数据分析模块
功能：
1. 类目分布分析 - 统计各类目样本数量，输出柱状图/饼图
2. 文本长度统计 - 标题/详情的长度分布，输出直方图/箱线图
3. 词频统计 - 高频词汇统计，输出柱状图
4. 词云生成 - 生成全局和分类目词云图
5. 相关性分析 - 类目与文本特征的相关性热力图
"""

import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from wordcloud import WordCloud
from scipy.stats import pearsonr, spearmanr

from loguru import logger

# 导入项目路径配置
from src import LOG_DIR, PROJECT_ROOT

# 导入分词器用于词频统计
from .preprocessor import ChineseTokenizer, DEFAULT_STOPWORDS


# 设置中文字体（支持Windows/Linux/Mac）
def _setup_chinese_font():
    """配置matplotlib中文字体"""
    # Windows常用中文字体
    chinese_fonts = [
        "SimHei",  # 黑体
        "Microsoft YaHei",  # 微软雅黑
        "SimSun",  # 宋体
        "KaiTi",  # 楷体
        "FangSong",  # 仿宋
        "Arial Unicode MS",  # Mac
        "PingFang SC",  # Mac
        "Heiti SC",  # Mac
        "WenQuanYi Micro Hei",  # Linux
        "WenQuanYi Zen Hei",  # Linux
        "Noto Sans CJK SC",  # Linux
    ]

    # 尝试找到可用的中文字体
    font_found = False
    for font in chinese_fonts:
        try:
            matplotlib.font_manager.findfont(font, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font]
            font_found = True
            logger.info(f"使用中文字体：{font}")
            break
        except Exception:
            continue

    if not font_found:
        logger.warning("未找到合适的中文字体，图表中文可能显示为方框")

    # 解决负号显示问题
    plt.rcParams["axes.unicode_minus"] = False


# 初始化字体
_setup_chinese_font()


# ============================================================================
# EDA分析主类
# ============================================================================
class EDAnalyzer:
    """
    EDA探索性数据分析器

    整合多种分析方法，提供一键完成所有EDA分析并输出图表的能力。
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        category_col: str = "category_name",
        category_id_col: str = "category_id",
        text_columns: Optional[List[str]] = None,
        tokenizer: Optional[ChineseTokenizer] = None,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 150,
        save_format: str = "png",
    ):
        """
        初始化EDA分析器

        Args:
            output_dir: 图表输出目录，默认 logs/eda
            category_col: 类目名称列名
            category_id_col: 类目ID列名
            text_columns: 要分析的文本列，默认 ['title', 'detail']
            tokenizer: 自定义分词器，None则使用默认配置
            figsize: 图表尺寸 (宽, 高)
            dpi: 图表分辨率
            save_format: 图表保存格式 (png, jpg, svg, pdf)
        """
        # 输出目录
        if output_dir is None:
            self.output_dir = LOG_DIR / "eda"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.category_col = category_col
        self.category_id_col = category_id_col
        self.text_columns = text_columns or ["title", "detail"]
        self.tokenizer = tokenizer or ChineseTokenizer()
        self.figsize = figsize
        self.dpi = dpi
        self.save_format = save_format

        # 颜色配置
        self.category_colors = sns.color_palette("husl", 12)

        logger.info(f"EDA分析器初始化完成，图表输出目录：{self.output_dir}")

    def _save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        tight_layout: bool = True,
    ) -> Path:
        """
        保存图表到文件

        Args:
            fig: matplotlib Figure对象
            filename: 文件名（不含扩展名）
            tight_layout: 是否自动调整布局

        Returns:
            Path: 保存的文件路径
        """
        if tight_layout:
            fig.tight_layout()

        file_path = self.output_dir / f"{filename}.{self.save_format}"
        fig.savefig(file_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"图表已保存：{file_path}")
        return file_path

    # ========================================================================
    # 1. 类目分布分析
    # ========================================================================
    def analyze_category_distribution(
        self,
        df: pd.DataFrame,
        plot_types: Optional[List[str]] = None,
    ) -> Dict:
        """
        分析类目分布并输出可视化图表

        Args:
            df: 数据DataFrame
            plot_types: 图表类型列表，可选 ['bar', 'pie', 'both']，默认两者都生成

        Returns:
            Dict: 包含统计信息和图表路径的字典
        """
        logger.info("开始分析类目分布...")

        if self.category_col not in df.columns:
            logger.error(f"类目列 {self.category_col} 不存在")
            return {}

        plot_types = plot_types or ["bar", "pie"]

        # 统计各类目数量
        category_counts = df[self.category_col].value_counts().sort_values(ascending=False)
        category_percents = category_counts / len(df) * 100

        stats = {
            "total_samples": len(df),
            "num_categories": len(category_counts),
            "category_counts": category_counts.to_dict(),
            "category_percents": category_percents.to_dict(),
            "max_category": category_counts.index[0],
            "max_count": category_counts.iloc[0],
            "min_category": category_counts.index[-1],
            "min_count": category_counts.iloc[-1],
            "imbalance_ratio": category_counts.iloc[0] / category_counts.iloc[-1],
        }

        logger.info(
            f"类目分布统计：共 {stats['num_categories']} 个类目，"
            f"{stats['total_samples']} 条样本，"
            f"不平衡比：{stats['imbalance_ratio']:.2f}"
        )

        result = {"stats": stats, "plots": {}}

        # 生成柱状图
        if "bar" in plot_types or "both" in plot_types:
            result["plots"]["bar"] = plot_category_distribution(
                category_counts=category_counts,
                output_path=self.output_dir / f"category_distribution_bar.{self.save_format}",
                figsize=self.figsize,
                dpi=self.dpi,
            )

        # 生成饼图
        if "pie" in plot_types or "both" in plot_types:
            result["plots"]["pie"] = plot_category_distribution_pie(
                category_counts=category_counts,
                output_path=self.output_dir / f"category_distribution_pie.{self.save_format}",
                figsize=self.figsize,
                dpi=self.dpi,
            )

        return result

    # ========================================================================
    # 2. 文本长度统计分析
    # ========================================================================
    def analyze_text_length(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> Dict:
        """
        分析文本长度分布并输出可视化图表

        Args:
            df: 数据DataFrame
            columns: 要分析的文本列，默认使用初始化时的text_columns

        Returns:
            Dict: 包含统计信息和图表路径的字典
        """
        logger.info("开始分析文本长度分布...")

        columns = columns or self.text_columns
        valid_columns = [col for col in columns if col in df.columns]

        if not valid_columns:
            logger.error("没有可分析的文本列")
            return {}

        stats = {}
        plots = {}

        for col in valid_columns:
            # 计算长度（字符数）
            lengths = df[col].astype(str).apply(len)

            # 统计描述
            desc = lengths.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            stats[col] = {
                "min": int(desc["min"]),
                "max": int(desc["max"]),
                "mean": float(desc["mean"]),
                "median": float(desc["50%"]),
                "std": float(desc["std"]),
                "q25": float(desc["25%"]),
                "q75": float(desc["75%"]),
                "q90": float(desc["90%"]),
                "q95": float(desc["95%"]),
                "q99": float(desc["99%"]),
                "null_count": int(df[col].isna().sum()),
                "empty_count": int((df[col].astype(str) == "").sum()),
            }

            logger.info(
                f"[{col}] 长度统计：均值={stats[col]['mean']:.1f}，"
                f"中位数={stats[col]['median']:.1f}，"
                f"范围=[{stats[col]['min']}, {stats[col]['max']}]"
            )

        # 生成直方图（所有列）
        plots["histogram"] = plot_text_length_distribution(
            df=df,
            columns=valid_columns,
            output_path=self.output_dir / f"text_length_histogram.{self.save_format}",
            figsize=self.figsize,
            dpi=self.dpi,
        )

        # 生成箱线图（按类目分组）
        if self.category_col in df.columns:
            for col in valid_columns:
                plots[f"boxplot_{col}"] = plot_text_length_boxplot(
                    df=df,
                    text_col=col,
                    category_col=self.category_col,
                    output_path=self.output_dir / f"text_length_boxplot_{col}.{self.save_format}",
                    figsize=(16, 8),
                    dpi=self.dpi,
                )

        return {"stats": stats, "plots": plots}

    # ========================================================================
    # 3. 词频统计分析
    # ========================================================================
    def analyze_word_frequency(
        self,
        df: pd.DataFrame,
        column: str = "combined_text",
        top_n: int = 30,
        per_category: bool = True,
        max_categories: int = 12,
    ) -> Dict:
        """
        分析词频分布并输出可视化图表

        Args:
            df: 数据DataFrame
            column: 要分析的文本列（建议使用分词后的列或combined_text）
            top_n: 显示前N个高频词
            per_category: 是否按类目分别统计词频
            max_categories: 最多分析的类目数

        Returns:
            Dict: 包含词频统计和图表路径的字典
        """
        logger.info(f"开始分析词频（列：{column}）...")

        if column not in df.columns:
            logger.warning(f"列 {column} 不存在，尝试使用 combined_tokens 或 text_columns")
            if "combined_tokens" in df.columns:
                column = "combined_tokens"
            else:
                valid_text_cols = [c for c in self.text_columns if c in df.columns]
                if not valid_text_cols:
                    logger.error("没有可分析的文本列")
                    return {}
                column = valid_text_cols[0]

        # 全局词频
        texts = df[column].astype(str).tolist()
        global_freq = self.tokenizer.get_word_frequency(texts, top_n=top_n * 2)

        logger.info(f"全局高频词Top5：{[f'{w}({c})' for w, c in global_freq.most_common(5)]}")

        result = {
            "global_top_words": dict(global_freq.most_common(top_n)),
            "plots": {},
        }

        # 生成全局词频柱状图
        result["plots"]["global"] = plot_word_frequency(
            word_freq=global_freq,
            top_n=top_n,
            title="全局高频词 Top{}".format(top_n),
            output_path=self.output_dir / f"word_frequency_global.{self.save_format}",
            figsize=self.figsize,
            dpi=self.dpi,
        )

        # 按类目统计词频
        if per_category and self.category_col in df.columns:
            category_freqs = {}
            categories = df[self.category_col].unique()[:max_categories]

            for cat in categories:
                cat_texts = df[df[self.category_col] == cat][column].astype(str).tolist()
                cat_freq = self.tokenizer.get_word_frequency(cat_texts, top_n=top_n)
                category_freqs[cat] = dict(cat_freq.most_common(top_n))

            result["category_top_words"] = category_freqs

            # 生成分类目词频图（2x3子图或单个大子图）
            num_cats = len(categories)
            if num_cats <= 6:
                cols = 3
                rows = (num_cats + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
                axes = axes.flatten() if num_cats > 1 else [axes]

                for idx, cat in enumerate(categories):
                    ax = axes[idx]
                    cat_freq_counter = Counter(category_freqs[cat])
                    words = [w for w, _ in cat_freq_counter.most_common(top_n)][::-1]
                    freqs = [c for _, c in cat_freq_counter.most_common(top_n)][::-1]
                    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(words)))
                    ax.barh(words, freqs, color=colors)
                    ax.set_title(f"【{cat}】高频词", fontsize=11)
                    ax.set_xlabel("出现次数")
                    ax.tick_params(axis="y", labelsize=9)

                # 隐藏多余的子图
                for idx in range(num_cats, len(axes)):
                    axes[idx].set_visible(False)

                fig.suptitle("各分类目高频词 Top{}".format(top_n), fontsize=14, y=1.01)
                result["plots"]["per_category"] = self._save_figure(
                    fig, "word_frequency_per_category", tight_layout=False
                )
            else:
                # 类目过多时，每个类目单独生成图
                for cat in categories:
                    cat_freq_counter = Counter(category_freqs[cat])
                    safe_name = str(cat).replace("/", "_").replace("\\", "_")
                    plot_word_frequency(
                        word_freq=cat_freq_counter,
                        top_n=top_n,
                        title=f"【{cat}】高频词 Top{top_n}",
                        output_path=self.output_dir / f"word_frequency_{safe_name}.{self.save_format}",
                        figsize=self.figsize,
                        dpi=self.dpi,
                    )

        return result

    # ========================================================================
    # 4. 词云生成
    # ========================================================================
    def generate_wordclouds(
        self,
        df: pd.DataFrame,
        column: str = "combined_tokens",
        per_category: bool = True,
        max_words: int = 200,
        width: int = 1200,
        height: int = 800,
        background_color: str = "white",
        max_categories: int = 12,
    ) -> Dict:
        """
        生成全局和分类目词云

        Args:
            df: 数据DataFrame
            column: 要分析的文本列（建议使用分词后的列）
            per_category: 是否按类目生成词云
            max_words: 词云显示的最大词数
            width: 词云宽度
            height: 词云高度
            background_color: 背景颜色
            max_categories: 最多生成的类目词云数

        Returns:
            Dict: 包含图表路径的字典
        """
        logger.info(f"开始生成词云（列：{column}）...")

        if column not in df.columns:
            logger.warning(f"列 {column} 不存在，尝试使用分词后的列")
            for alt_col in ["combined_tokens", "title_tokens", "detail_tokens"]:
                if alt_col in df.columns:
                    column = alt_col
                    break
            if column not in df.columns:
                logger.error("没有可分析的分词列")
                return {}

        result = {"plots": {}}

        # 全局词云
        all_text = " ".join(df[column].astype(str).tolist())
        if all_text.strip():
            result["plots"]["global"] = generate_wordcloud(
                text=all_text,
                output_path=self.output_dir / f"wordcloud_global.{self.save_format}",
                max_words=max_words,
                width=width,
                height=height,
                background_color=background_color,
                figsize=self.figsize,
                dpi=self.dpi,
                title="全局词云",
            )

        # 分类目词云
        if per_category and self.category_col in df.columns:
            categories = df[self.category_col].unique()[:max_categories]

            # 如果类目数 <= 6，合并为一个大图
            if len(categories) <= 6:
                cols = 3
                rows = (len(categories) + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
                axes = axes.flatten() if len(categories) > 1 else [axes]

                for idx, cat in enumerate(categories):
                    ax = axes[idx]
                    cat_text = " ".join(
                        df[df[self.category_col] == cat][column].astype(str).tolist()
                    )
                    if cat_text.strip():
                        wc = WordCloud(
                            font_path=_get_font_path(),
                            width=600,
                            height=400,
                            background_color=background_color,
                            max_words=max_words // 2,
                            colormap="viridis",
                            collocations=False,
                        ).generate(cat_text)
                        ax.imshow(wc, interpolation="bilinear")
                        ax.set_title(f"【{cat}】词云", fontsize=11)
                    ax.axis("off")

                # 隐藏多余的子图
                for idx in range(len(categories), len(axes)):
                    axes[idx].set_visible(False)

                fig.suptitle("各分类目词云", fontsize=14, y=1.01)
                result["plots"]["per_category_combined"] = self._save_figure(
                    fig, "wordcloud_per_category_combined", tight_layout=False
                )
            else:
                # 每个类目单独生成
                for cat in categories:
                    cat_text = " ".join(
                        df[df[self.category_col] == cat][column].astype(str).tolist()
                    )
                    if cat_text.strip():
                        safe_name = str(cat).replace("/", "_").replace("\\", "_")
                        generate_wordcloud(
                            text=cat_text,
                            output_path=self.output_dir / f"wordcloud_{safe_name}.{self.save_format}",
                            max_words=max_words,
                            width=width,
                            height=height,
                            background_color=background_color,
                            figsize=self.figsize,
                            dpi=self.dpi,
                            title=f"【{cat}】词云",
                        )

        return result

    # ========================================================================
    # 5. 相关性分析
    # ========================================================================
    def analyze_correlation(
        self,
        df: pd.DataFrame,
        text_columns: Optional[List[str]] = None,
    ) -> Dict:
        """
        分析文本特征与类目之间的相关性

        Args:
            df: 数据DataFrame
            text_columns: 要分析的文本列

        Returns:
            Dict: 包含统计信息和图表路径的字典
        """
        logger.info("开始相关性分析...")

        text_columns = text_columns or self.text_columns
        valid_columns = [col for col in text_columns if col in df.columns]

        if not valid_columns:
            logger.error("没有可分析的文本列")
            return {}

        # 构建特征DataFrame
        feature_df = pd.DataFrame()

        # 添加文本长度特征
        for col in valid_columns:
            feature_df[f"{col}_len"] = df[col].astype(str).apply(len)
            feature_df[f"{col}_token_count"] = df[col].astype(str).apply(
                lambda x: len(self.tokenizer.tokenize(x, return_list=True))
            )

        # 添加类目ID
        if self.category_id_col in df.columns:
            feature_df["category_id"] = df[self.category_id_col].astype(int)
        elif self.category_col in df.columns:
            # 将类目名称编码为数字
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            feature_df["category_id"] = le.fit_transform(df[self.category_col])

        # 计算相关性矩阵
        corr_matrix = feature_df.corr(method="spearman")

        result = {
            "correlation_matrix": corr_matrix.to_dict(),
            "plots": {},
        }

        # 生成相关性热力图
        result["plots"]["heatmap"] = plot_correlation_heatmap(
            corr_matrix=corr_matrix,
            output_path=self.output_dir / f"correlation_heatmap.{self.save_format}",
            figsize=self.figsize,
            dpi=self.dpi,
            title="特征相关性热力图",
        )

        # 文本长度 vs 类目 散点图/箱线图
        if self.category_col in df.columns and len(feature_df.columns) > 1:
            combined_len = feature_df[[f"{col}_len" for col in valid_columns]].sum(axis=1)
            plot_df = pd.DataFrame(
                {
                    "category": df[self.category_col],
                    "total_text_length": combined_len,
                }
            )
            fig, ax = plt.subplots(figsize=self.figsize)
            sns.boxplot(data=plot_df, x="category", y="total_text_length", ax=ax)
            ax.set_title("各分类目文本总长度分布", fontsize=14)
            ax.set_xlabel("类目")
            ax.set_ylabel("总文本长度（字符数）")
            ax.tick_params(axis="x", rotation=45)
            result["plots"]["length_vs_category"] = self._save_figure(
                fig, "text_length_vs_category"
            )

        return result

    # ========================================================================
    # 一键执行完整EDA
    # ========================================================================
    def run_full_analysis(
        self,
        df: pd.DataFrame,
        do_category: bool = True,
        do_text_length: bool = True,
        do_word_frequency: bool = True,
        do_wordcloud: bool = True,
        do_correlation: bool = True,
    ) -> Dict:
        """
        一键执行完整的EDA分析流程

        Args:
            df: 预处理后的DataFrame
            do_category: 是否分析类目分布
            do_text_length: 是否分析文本长度
            do_word_frequency: 是否分析词频
            do_wordcloud: 是否生成词云
            do_correlation: 是否做相关性分析

        Returns:
            Dict: 所有分析结果，包含统计信息和图表路径
        """
        logger.info("=" * 60)
        logger.info("开始执行完整EDA分析")
        logger.info("=" * 60)

        results = {}

        # 1. 类目分布
        if do_category:
            results["category_distribution"] = self.analyze_category_distribution(df)

        # 2. 文本长度
        if do_text_length:
            results["text_length"] = self.analyze_text_length(df)

        # 3. 词频分析
        if do_word_frequency:
            results["word_frequency"] = self.analyze_word_frequency(df)

        # 4. 词云
        if do_wordcloud:
            results["wordcloud"] = self.generate_wordclouds(df)

        # 5. 相关性分析
        if do_correlation:
            results["correlation"] = self.analyze_correlation(df)

        logger.info("=" * 60)
        logger.info(f"EDA分析完成！所有图表已保存到：{self.output_dir}")
        logger.info("=" * 60)

        return results


# ============================================================================
# 辅助函数：查找可用中文字体文件路径
# ============================================================================
def _get_font_path() -> Optional[str]:
    """
    查找系统中可用的中文字体文件路径（供WordCloud使用）

    Returns:
        Optional[str]: 字体文件路径，找不到返回None
    """
    # Windows常见字体路径
    font_candidates = [
        # Windows
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        # Linux
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]

    for font_path in font_candidates:
        if os.path.exists(font_path):
            return font_path

    # 尝试从matplotlib字体管理器查找
    try:
        import matplotlib.font_manager as fm
        for font in fm.fontManager.ttflist:
            if any(keyword in font.name.lower() for keyword in [
                "simhei", "yahei", "simsun", "kaiti", "fangsong",
                "pingfang", "heiti", "wenquanyi", "noto sans cjk",
            ]):
                return font.fname
    except Exception:
        pass

    logger.warning("未找到中文字体文件，词云中的中文可能显示为方框")
    return None


# ============================================================================
# 独立的绘图函数（可单独调用）
# ============================================================================
def plot_category_distribution(
    category_counts: pd.Series,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 150,
    title: str = "类目分布柱状图",
) -> Optional[Path]:
    """
    绘制类目分布柱状图

    Args:
        category_counts: 类目-数量Series（已排序）
        output_path: 输出路径，None则显示而不保存
        figsize: 图表尺寸
        dpi: 分辨率
        title: 图表标题

    Returns:
        Optional[Path]: 保存的文件路径（如果output_path不为None）
    """
    fig, ax = plt.subplots(figsize=figsize)

    categories = category_counts.index.tolist()
    counts = category_counts.values.tolist()
    colors = sns.color_palette("husl", len(categories))

    bars = ax.bar(categories, counts, color=colors, edgecolor="white", linewidth=0.8)

    # 在柱子上添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(counts) * 0.005,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # 添加百分比标签
    total = sum(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height / 2,
            f"{count / total:.1%}",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("商品类目", fontsize=12)
    ax.set_ylabel("样本数量", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, max(counts) * 1.1)

    sns.despine(ax=ax, top=True, right=True)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"类目分布柱状图已保存：{output_path}")
        return output_path
    else:
        plt.show()
        return None


def plot_category_distribution_pie(
    category_counts: pd.Series,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 150,
    title: str = "类目分布饼图",
) -> Optional[Path]:
    """
    绘制类目分布饼图

    Args:
        category_counts: 类目-数量Series
        output_path: 输出路径
        figsize: 图表尺寸
        dpi: 分辨率
        title: 图表标题

    Returns:
        Optional[Path]: 保存的文件路径
    """
    fig, ax = plt.subplots(figsize=figsize)

    categories = category_counts.index.tolist()
    counts = category_counts.values.tolist()
    colors = sns.color_palette("husl", len(categories))

    # 计算百分比
    total = sum(counts)
    autopct = lambda pct: f"{pct:.1f}%" if pct > 2 else ""

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=categories,
        colors=colors,
        autopct=autopct,
        startangle=90,
        pctdistance=0.85,
        labeldistance=1.1,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10),
    )

    # 美化百分比文本
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight("bold")
        autotext.set_color("white")

    # 添加图例
    ax.legend(
        wedges,
        [f"{cat}: {cnt} ({cnt/total:.1%})" for cat, cnt in zip(categories, counts)],
        title="类目详情",
        loc="center left",
        bbox_to_anchor=(1.0, 0, 0.5, 1),
        fontsize=9,
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"类目分布饼图已保存：{output_path}")
        return output_path
    else:
        plt.show()
        return None


def plot_text_length_distribution(
    df: pd.DataFrame,
    columns: List[str],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 150,
    bins: int = 50,
) -> Optional[Path]:
    """
    绘制多列文本长度分布直方图

    Args:
        df: 数据DataFrame
        columns: 要绘制的文本列列表
        output_path: 输出路径
        figsize: 图表尺寸
        dpi: 分辨率
        bins: 直方图箱数

    Returns:
        Optional[Path]: 保存的文件路径
    """
    num_cols = len(columns)
    cols_per_row = min(2, num_cols)
    rows = (num_cols + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(rows, cols_per_row, figsize=figsize)
    if num_cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    for idx, col in enumerate(columns):
        ax = axes[idx]
        lengths = df[col].astype(str).apply(len)
        color = colors[idx % len(colors)]

        # 绘制直方图
        ax.hist(
            lengths,
            bins=bins,
            color=color,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )

        # 添加统计信息
        mean_len = lengths.mean()
        median_len = lengths.median()
        ax.axvline(mean_len, color="red", linestyle="--", linewidth=2, label=f"均值: {mean_len:.1f}")
        ax.axvline(median_len, color="green", linestyle="--", linewidth=2, label=f"中位数: {median_len:.1f}")

        ax.set_title(f"【{col}】长度分布", fontsize=13, fontweight="bold")
        ax.set_xlabel("文本长度（字符数）", fontsize=11)
        ax.set_ylabel("频次", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")
        sns.despine(ax=ax, top=True, right=True)

    # 隐藏多余的子图
    for idx in range(num_cols, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("文本长度分布分析", fontsize=16, fontweight="bold", y=1.01)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"文本长度分布图已保存：{output_path}")
        return output_path
    else:
        plt.show()
        return None


def plot_text_length_boxplot(
    df: pd.DataFrame,
    text_col: str,
    category_col: str,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 8),
    dpi: int = 150,
) -> Optional[Path]:
    """
    绘制各分类目文本长度箱线图

    Args:
        df: 数据DataFrame
        text_col: 文本列名
        category_col: 类目列名
        output_path: 输出路径
        figsize: 图表尺寸
        dpi: 分辨率

    Returns:
        Optional[Path]: 保存的文件路径
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 计算长度
    plot_df = df[[category_col, text_col]].copy()
    plot_df["length"] = plot_df[text_col].astype(str).apply(len)

    # 按中位数排序类目
    order = plot_df.groupby(category_col)["length"].median().sort_values(ascending=False).index

    # 绘制箱线图
    colors = sns.color_palette("husl", len(order))
    sns.boxplot(
        data=plot_df,
        x=category_col,
        y="length",
        order=order,
        ax=ax,
        palette=colors,
        showfliers=False,
        width=0.7,
    )

    # 添加散点（采样以避免过密）
    if len(plot_df) <= 5000:
        sample_df = plot_df
    else:
        sample_df = plot_df.sample(5000, random_state=42)
    sns.stripplot(
        data=sample_df,
        x=category_col,
        y="length",
        order=order,
        ax=ax,
        size=2,
        alpha=0.3,
        color="black",
    )

    ax.set_title(f"各分类目【{text_col}】长度分布箱线图", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("商品类目", fontsize=12)
    ax.set_ylabel(f"文本长度（{text_col}字符数）", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    sns.despine(ax=ax, top=True, right=True)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"文本长度箱线图已保存：{output_path}")
        return output_path
    else:
        plt.show()
        return None


def plot_word_frequency(
    word_freq: Counter,
    top_n: int = 30,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 150,
    title: str = "高频词统计",
) -> Optional[Path]:
    """
    绘制词频水平柱状图

    Args:
        word_freq: 词频Counter对象
        top_n: 显示前N个词
        output_path: 输出路径
        figsize: 图表尺寸
        dpi: 分辨率
        title: 图表标题

    Returns:
        Optional[Path]: 保存的文件路径
    """
    # 获取TopN词
    top_words = word_freq.most_common(top_n)
    words = [w for w, _ in top_words][::-1]
    freqs = [c for _, c in top_words][::-1]

    fig, ax = plt.subplots(figsize=figsize)

    # 使用渐变色
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(words)))
    bars = ax.barh(words, freqs, color=colors, edgecolor="white", linewidth=0.5)

    # 添加数值标签
    for bar, freq in zip(bars, freqs):
        ax.text(
            bar.get_width() + max(freqs) * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{freq}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("出现次数", fontsize=12)
    ax.set_ylabel("词语", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_xlim(0, max(freqs) * 1.1)
    sns.despine(ax=ax, top=True, right=True)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"词频统计图已保存：{output_path}")
        return output_path
    else:
        plt.show()
        return None


def generate_wordcloud(
    text: str,
    output_path: Optional[Path] = None,
    max_words: int = 200,
    width: int = 1200,
    height: int = 800,
    background_color: str = "white",
    colormap: str = "viridis",
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 150,
    title: str = "词云",
) -> Optional[Path]:
    """
    生成并保存/显示词云

    Args:
        text: 输入文本（空格分隔的词语）
        output_path: 输出路径
        max_words: 最大词数
        width: 词云宽度
        height: 词云高度
        background_color: 背景色
        colormap: 颜色映射
        figsize: 图表尺寸
        dpi: 分辨率
        title: 图表标题

    Returns:
        Optional[Path]: 保存的文件路径
    """
    font_path = _get_font_path()

    # 创建WordCloud对象
    wc_kwargs = {
        "width": width,
        "height": height,
        "background_color": background_color,
        "max_words": max_words,
        "colormap": colormap,
        "collocations": False,
        "prefer_horizontal": 0.7,
        "relative_scaling": 0.5,
    }
    if font_path:
        wc_kwargs["font_path"] = font_path

    wc = WordCloud(**wc_kwargs).generate(text)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"词云图已保存：{output_path}")
        return output_path
    else:
        plt.show()
        return None


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 150,
    title: str = "相关性热力图",
    annot: bool = True,
    fmt: str = ".2f",
    cmap: str = "coolwarm",
) -> Optional[Path]:
    """
    绘制相关性热力图

    Args:
        corr_matrix: 相关性矩阵DataFrame
        output_path: 输出路径
        figsize: 图表尺寸
        dpi: 分辨率
        title: 图表标题
        annot: 是否显示数值标注
        fmt: 数值格式
        cmap: 颜色映射

    Returns:
        Optional[Path]: 保存的文件路径
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制热力图
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={
            "shrink": 0.8,
            "label": "相关系数",
        },
        vmin=-1,
        vmax=1,
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"相关性热力图已保存：{output_path}")
        return output_path
    else:
        plt.show()
        return None
