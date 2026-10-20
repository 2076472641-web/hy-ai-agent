# -*- coding: utf-8 -*-
"""
数据处理模块
包含数据加载、预处理、EDA分析等功能
"""

from .data_loader import (
    DataLoader,
    MockDataGenerator,
    split_dataset,
    load_csv,
    save_csv,
)

from .preprocessor import (
    DataPreprocessor,
    TextCleaner,
    remove_duplicates,
    remove_nulls,
)

from .eda import (
    EDAnalyzer,
    plot_category_distribution,
    plot_text_length_distribution,
    plot_word_frequency,
    generate_wordcloud,
    plot_correlation_heatmap,
)

__all__ = [
    # data_loader
    "DataLoader",
    "MockDataGenerator",
    "split_dataset",
    "load_csv",
    "save_csv",
    # preprocessor
    "DataPreprocessor",
    "TextCleaner",
    "remove_duplicates",
    "remove_nulls",
    # eda
    "EDAnalyzer",
    "plot_category_distribution",
    "plot_text_length_distribution",
    "plot_word_frequency",
    "generate_wordcloud",
    "plot_correlation_heatmap",
]
