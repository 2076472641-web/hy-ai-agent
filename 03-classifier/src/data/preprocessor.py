# -*- coding: utf-8 -*-
"""
数据预处理模块
功能：
1. 数据清洗 - 去重、去空值、去特殊字符、HTML标签、表情符号等
2. 文本规整 - 繁简转换、全角半角转换、大小写归一化等
3. 中文分词与停用词过滤
"""

import re
import html
from typing import List, Optional, Tuple, Union
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
from loguru import logger

# NLP相关库
import jieba
import jieba_fast as jieba_fast  # 高性能jieba分词
from hanziconv import HanziConv  # 繁简转换
import emoji  # 表情符号处理

# 导入项目路径配置
from src import PROJECT_ROOT


# ============================================================================
# 中文停用词表
# ============================================================================
# 常用中文停用词（包含标点、助词、连词等）
DEFAULT_STOPWORDS = set([
    # 标点符号
    "，", "。", "！", "？", "、", "；", "：", "“", "”", "‘", "’",
    "（", "）", "【", "】", "《", "》", "—", "…", "·", "～",
    ",", ".", "!", "?", ";", ":", "\"", "'", "(", ")", "[", "]",
    "{", "}", "<", ">", "-", "_", "+", "=", "*", "/", "\\", "|",
    "@", "#", "$", "%", "^", "&", "~", "`",
    # 助词
    "的", "了", "和", "是", "就", "都", "而", "及", "与", "着",
    "或", "一个", "没有", "我们", "你们", "他们", "她们", "它们",
    "自己", "这", "那", "这个", "那个", "这些", "那些", "这样",
    "那样", "之", "其", "此", "该", "各", "每", "某", "等",
    "等等", "之类", "等等",
    # 连词
    "但是", "然而", "虽然", "如果", "因为", "所以", "因此", "于是",
    "而且", "并且", "或者", "还是", "不是", "而是", "无论", "不管",
    "只要", "只有", "除非", "以便", "以免", "进而", "从而",
    # 介词
    "在", "于", "从", "到", "向", "往", "由", "以", "用", "通过",
    "根据", "按照", "关于", "对于", "为了", "由于", "除了", "作为",
    # 量词
    "个", "些", "把", "条", "只", "件", "张", "本", "台", "部",
    "套", "双", "副", "对", "顿", "场", "次", "遍", "趟", "阵",
    # 代词
    "我", "你", "他", "她", "它", "咱", "俺", "这", "那", "哪",
    "谁", "什么", "怎么", "怎样", "如何", "哪里", "哪儿", "多少",
    "几", "多", "多么", "那么", "这么", "怎样", "如何",
    # 副词
    "很", "非常", "特别", "十分", "极其", "相当", "比较", "更",
    "最", "太", "真", "真的", "确实", "实在", "的确", "当然",
    "已经", "曾经", "刚刚", "刚才", "正在", "现在", "立刻",
    "马上", "将", "将要", "会", "能", "可以", "可能", "应该",
    "必须", "需要", "得", "不得不",
    # 语气词
    "啊", "呀", "吧", "呢", "嘛", "哦", "嗯", "哈", "唉", "咦",
    "哇", "嗨", "嘿", "哼", "哟", "喽", "呗", "呐",
    # 数字相关
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "百", "千", "万", "亿", "零", "两",
    # 其他常见停用词
    "不", "没", "有", "也", "还", "又", "再", "才", "就", "都",
    "要", "会", "能", "去", "来", "上", "下", "里", "外", "中",
    "前", "后", "左", "右", "东", "西", "南", "北",
    "请", "您", "好", "您好", "谢谢", "感谢", "抱歉", "对不起",
    "欢迎", "新", "老", "大", "小", "高", "低", "长", "短",
    "更多", "了解", "联系", "客服", "咨询", "点击", "查看",
    "购买", "下单", "付款", "发货", "收货", "售后", "退换",
    "服务", "支持", "保证", "品质", "质量", "正品", "品牌",
    "商品", "产品", "宝贝", "详情", "介绍", "说明", "图片",
    "规格", "参数", "型号", "颜色", "尺寸", "重量", "数量",
    "价格", "优惠", "特价", "折扣", "促销", "活动", "包邮",
    "赠送", "礼品", "赠品", "礼盒", "包装",
])


# ============================================================================
# 文本清洗核心类
# ============================================================================
class TextCleaner:
    """
    文本清洗器

    提供多种文本清洗功能：去HTML、去特殊字符、去表情、去URL等
    """

    # HTML标签正则
    HTML_PATTERN = re.compile(r"<[^>]+>", re.IGNORECASE)
    # URL正则
    URL_PATTERN = re.compile(
        r"(https?://|www\.)[^\s]+",
        re.IGNORECASE,
    )
    # 邮箱正则
    EMAIL_PATTERN = re.compile(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        re.IGNORECASE,
    )
    # 电话号码正则（中国手机号/固话）
    PHONE_PATTERN = re.compile(
        r"(?:\+?86)?1[3-9]\d{9}"  # 手机号
        r"|0\d{2,3}[-\s]?\d{7,8}",  # 固话
        re.IGNORECASE,
    )
    # 多余空白字符正则（包含换行、制表符等）
    WHITESPACE_PATTERN = re.compile(r"\s+")
    # 连续重复标点正则
    REPEAT_PUNCT_PATTERN = re.compile(r"([^\w\s])\1{2,}")
    # 中文标点与英文标点映射
    PUNCT_MAPPING = {
        "，": ",", "。": ".", "！": "!", "？": "?",
        "；": ";", "：": ":", "（": "(", "）": ")",
        "【": "[", "】": "]", "《": "<", "》": ">",
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "、": ",", "…": "...", "—": "-",
    }

    def __init__(
        self,
        remove_html: bool = True,
        remove_url: bool = True,
        remove_email: bool = True,
        remove_phone: bool = False,
        remove_emoji: bool = True,
        remove_special_chars: bool = True,
        traditional_to_simplified: bool = True,
        fullwidth_to_halfwidth: bool = True,
        normalize_whitespace: bool = True,
        remove_repeat_punct: bool = True,
        min_text_length: int = 1,
    ):
        """
        初始化文本清洗器

        Args:
            remove_html: 是否移除HTML标签
            remove_url: 是否移除URL链接
            remove_email: 是否移除邮箱地址
            remove_phone: 是否移除电话号码（谨慎使用，可能包含有用信息）
            remove_emoji: 是否移除emoji表情符号
            remove_special_chars: 是否移除特殊字符
            traditional_to_simplified: 是否进行繁体转简体
            fullwidth_to_halfwidth: 是否进行全角转半角
            normalize_whitespace: 是否规范化空白字符
            remove_repeat_punct: 是否移除连续重复标点
            min_text_length: 清洗后文本最小长度（小于此值返回空字符串）
        """
        self.remove_html = remove_html
        self.remove_url = remove_url
        self.remove_email = remove_email
        self.remove_phone = remove_phone
        self.remove_emoji = remove_emoji
        self.remove_special_chars = remove_special_chars
        self.traditional_to_simplified = traditional_to_simplified
        self.fullwidth_to_halfwidth = fullwidth_to_halfwidth
        self.normalize_whitespace = normalize_whitespace
        self.remove_repeat_punct = remove_repeat_punct
        self.min_text_length = min_text_length

        logger.info("文本清洗器初始化完成")

    def clean(self, text: str) -> str:
        """
        对文本执行完整的清洗流程

        Args:
            text: 待清洗的原始文本

        Returns:
            str: 清洗后的文本
        """
        if not text or not isinstance(text, str):
            return ""

        result = text

        # 1. HTML实体解码（如 &amp; -> &）
        result = html.unescape(result)

        # 2. 移除HTML标签
        if self.remove_html:
            result = self._remove_html_tags(result)

        # 3. 移除URL
        if self.remove_url:
            result = self.URL_PATTERN.sub(" ", result)

        # 4. 移除邮箱
        if self.remove_email:
            result = self.EMAIL_PATTERN.sub(" ", result)

        # 5. 移除电话号码
        if self.remove_phone:
            result = self.PHONE_PATTERN.sub(" ", result)

        # 6. 移除emoji表情
        if self.remove_emoji:
            result = emoji.replace_emoji(result, replace=" ")

        # 7. 繁体转简体
        if self.traditional_to_simplified:
            result = HanziConv.toSimplified(result)

        # 8. 全角转半角
        if self.fullwidth_to_halfwidth:
            result = self._fullwidth_to_halfwidth(result)

        # 9. 移除特殊字符（保留中文、英文、数字、基本标点）
        if self.remove_special_chars:
            result = self._remove_special_characters(result)

        # 10. 移除连续重复标点
        if self.remove_repeat_punct:
            result = self.REPEAT_PUNCT_PATTERN.sub(r"\1\1", result)

        # 11. 规范化空白字符
        if self.normalize_whitespace:
            result = self.WHITESPACE_PATTERN.sub(" ", result).strip()

        # 12. 检查最小长度
        if len(result) < self.min_text_length:
            result = ""

        return result

    def _remove_html_tags(self, text: str) -> str:
        """移除HTML标签"""
        # 先移除script和style标签及其内容
        text = re.sub(
            r"<(script|style)[^>]*>.*?</\1>",
            " ",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # 再移除普通HTML标签
        return self.HTML_PATTERN.sub(" ", text)

    def _fullwidth_to_halfwidth(self, text: str) -> str:
        """
        全角字符转半角字符

        全角字符Unicode范围：FF01-FF5E
        与半角字符偏移量：0xFEE0
        """
        result = []
        for char in text:
            code = ord(char)
            # 处理全角空格
            if code == 0x3000:
                result.append(" ")
            # 处理其他全角字符
            elif 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            else:
                result.append(char)
        return "".join(result)

    def _remove_special_characters(self, text: str) -> str:
        """
        移除特殊字符

        保留：中文（\u4e00-\u9fff）、英文大小写、数字、
              基本标点（逗号、句号、感叹号、问号、分号、冒号、括号等）
        """
        # 定义保留的字符模式
        pattern = re.compile(
            r"[^\u4e00-\u9fa5"  # 中文
            r"a-zA-Z0-9"  # 英文数字
            r"\s"  # 空白字符
            r",.!?;:()\[\]<>\"'\-_+/\\&@#$%=]"  # 基本标点和特殊符号
            r"]",
            flags=re.UNICODE,
        )
        return pattern.sub(" ", text)

    def __call__(self, text: str) -> str:
        """支持将清洗器作为函数调用"""
        return self.clean(text)


# ============================================================================
# 中文分词器
# ============================================================================
class ChineseTokenizer:
    """
    中文分词器

    基于jieba分词，支持停用词过滤、自定义词典等功能
    """

    def __init__(
        self,
        stopwords: Optional[Union[set, List[str], str, Path]] = None,
        use_fast_tokenizer: bool = True,
        user_dict: Optional[Union[str, Path]] = None,
        min_token_length: int = 1,
        filter_non_chinese: bool = True,
    ):
        """
        初始化中文分词器

        Args:
            stopwords: 停用词集合/列表，或停用词文件路径；None则使用默认停用词
            use_fast_tokenizer: 是否使用高速分词器（jieba_fast）
            user_dict: 用户自定义词典路径
            min_token_length: 词语最小长度（小于此值会被过滤）
            filter_non_chinese: 是否过滤非中文词语（纯英文/数字等）
        """
        # 选择分词后端
        self.tokenizer = jieba_fast if use_fast_tokenizer else jieba
        logger.info(f"使用分词器：{'jieba_fast' if use_fast_tokenizer else 'jieba'}")

        # 加载自定义词典
        if user_dict:
            user_dict = Path(user_dict)
            if user_dict.exists():
                self.tokenizer.load_userdict(str(user_dict))
                logger.info(f"已加载自定义词典：{user_dict}")
            else:
                logger.warning(f"自定义词典不存在：{user_dict}")

        # 加载停用词
        self.stopwords = self._load_stopwords(stopwords)
        logger.info(f"停用词数量：{len(self.stopwords)}")

        self.min_token_length = min_token_length
        self.filter_non_chinese = filter_non_chinese

        # 中文正则（用于过滤非中文词）
        self.chinese_pattern = re.compile(r"[\u4e00-\u9fa5]")

    def _load_stopwords(
        self,
        stopwords: Optional[Union[set, List[str], str, Path]],
    ) -> set:
        """
        加载停用词

        Args:
            stopwords: 停用词源（集合、列表、文件路径）

        Returns:
            set: 停用词集合
        """
        if stopwords is None:
            return DEFAULT_STOPWORDS.copy()

        if isinstance(stopwords, set):
            return stopwords

        if isinstance(stopwords, list):
            return set(stopwords)

        # 从文件加载
        stopwords_path = Path(stopwords)
        if stopwords_path.exists():
            words = set()
            with open(stopwords_path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith("#"):
                        words.add(word)
            logger.info(f"从文件加载停用词 {len(words)} 个：{stopwords_path}")
            return words
        else:
            logger.warning(f"停用词文件不存在，使用默认停用词：{stopwords_path}")
            return DEFAULT_STOPWORDS.copy()

    def tokenize(
        self,
        text: str,
        return_list: bool = True,
    ) -> Union[List[str], str]:
        """
        对文本进行分词

        Args:
            text: 待分词的文本（应为已清洗的文本）
            return_list: True返回词语列表，False返回空格分隔的字符串

        Returns:
            List[str] 或 str: 分词结果
        """
        if not text or not isinstance(text, str):
            return [] if return_list else ""

        # 精确模式分词
        tokens = self.tokenizer.lcut(text)

        # 过滤处理
        filtered_tokens = []
        for token in tokens:
            token = token.strip()
            # 跳过空字符串
            if not token:
                continue
            # 跳过停用词
            if token in self.stopwords:
                continue
            # 跳过过短的词
            if len(token) < self.min_token_length:
                continue
            # 过滤非中文词（纯英文/数字/标点）
            if self.filter_non_chinese and not self.chinese_pattern.search(token):
                continue
            filtered_tokens.append(token)

        return filtered_tokens if return_list else " ".join(filtered_tokens)

    def get_word_frequency(
        self,
        texts: List[str],
        top_n: Optional[int] = None,
    ) -> Counter:
        """
        统计词频

        Args:
            texts: 文本列表
            top_n: 返回前N个高频词，None返回全部

        Returns:
            Counter: 词频计数器
        """
        all_tokens = []
        for text in texts:
            tokens = self.tokenize(text, return_list=True)
            all_tokens.extend(tokens)

        freq = Counter(all_tokens)
        if top_n:
            return Counter(dict(freq.most_common(top_n)))
        return freq

    def __call__(self, text: str, return_list: bool = True):
        """支持将分词器作为函数调用"""
        return self.tokenize(text, return_list)


# ============================================================================
# 数据预处理主类
# ============================================================================
class DataPreprocessor:
    """
    数据预处理器

    整合文本清洗、分词、去重、去空值等功能，提供端到端的数据处理能力。
    """

    def __init__(
        self,
        text_cleaner: Optional[TextCleaner] = None,
        tokenizer: Optional[ChineseTokenizer] = None,
        text_columns: Optional[List[str]] = None,
        subset_columns: Optional[List[str]] = None,
    ):
        """
        初始化数据预处理器

        Args:
            text_cleaner: 自定义文本清洗器，None则使用默认配置
            tokenizer: 自定义分词器，None则使用默认配置
            text_columns: 需要进行文本清洗/分词的列名，默认 ['title', 'detail']
            subset_columns: 去重和去空值时参考的列名，默认与text_columns相同
        """
        self.text_cleaner = text_cleaner or TextCleaner()
        self.tokenizer = tokenizer or ChineseTokenizer()
        self.text_columns = text_columns or ["title", "detail"]
        self.subset_columns = subset_columns or self.text_columns

        logger.info(f"数据预处理器初始化完成，处理列：{self.text_columns}")

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = "first",
    ) -> pd.DataFrame:
        """
        去除重复数据

        Args:
            df: 输入DataFrame
            subset: 判断重复的参考列，None使用默认subset_columns
            keep: 保留方式 {'first', 'last', False}

        Returns:
            pd.DataFrame: 去重后的DataFrame
        """
        subset = subset or self.subset_columns
        before_count = len(df)

        # 确保列存在
        valid_subset = [col for col in subset if col in df.columns]
        if not valid_subset:
            logger.warning("去重参考列均不存在，跳过去重")
            return df

        df = df.drop_duplicates(subset=valid_subset, keep=keep).reset_index(drop=True)
        after_count = len(df)
        removed_count = before_count - after_count

        logger.info(f"去重完成：移除 {removed_count} 条重复数据，剩余 {after_count} 条")
        return df

    def remove_nulls(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        how: str = "any",
    ) -> pd.DataFrame:
        """
        去除空值数据

        Args:
            df: 输入DataFrame
            subset: 检查空值的列，None使用默认subset_columns
            how: {'any', 'all'}，any表示任一列为空则删除，all表示全部为空才删除

        Returns:
            pd.DataFrame: 去除空值后的DataFrame
        """
        subset = subset or self.subset_columns
        before_count = len(df)

        # 确保列存在
        valid_subset = [col for col in subset if col in df.columns]
        if not valid_subset:
            logger.warning("空值检查列均不存在，跳过去空值")
            return df

        # 先将空字符串视为NaN
        for col in valid_subset:
            if df[col].dtype == object:
                df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

        df = df.dropna(subset=valid_subset, how=how).reset_index(drop=True)
        after_count = len(df)
        removed_count = before_count - after_count

        logger.info(f"去空值完成：移除 {removed_count} 条空值数据，剩余 {after_count} 条")
        return df

    def clean_texts(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        对DataFrame中的文本列进行清洗

        Args:
            df: 输入DataFrame
            columns: 需要清洗的列，None使用默认text_columns
            inplace: 是否原地修改

        Returns:
            pd.DataFrame: 清洗后的DataFrame
        """
        if not inplace:
            df = df.copy()

        columns = columns or self.text_columns

        for col in columns:
            if col not in df.columns:
                logger.warning(f"列 {col} 不存在，跳过文本清洗")
                continue

            logger.info(f"正在清洗列：{col}")
            df[col] = df[col].apply(self.text_cleaner.clean)

            # 统计清洗后为空的数量
            empty_count = (df[col] == "").sum()
            if empty_count > 0:
                logger.warning(f"列 {col} 清洗后有 {empty_count} 条空文本")

        logger.info("文本清洗完成")
        return df

    def tokenize_texts(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        output_suffix: str = "_tokens",
        join_str: Optional[str] = " ",
    ) -> pd.DataFrame:
        """
        对DataFrame中的文本列进行分词

        Args:
            df: 输入DataFrame（应已完成文本清洗）
            columns: 需要分词的列，None使用默认text_columns
            output_suffix: 输出列后缀，例如title -> title_tokens
            join_str: 词语连接符，为None则输出列表类型

        Returns:
            pd.DataFrame: 添加了分词列的DataFrame
        """
        df = df.copy()
        columns = columns or self.text_columns

        for col in columns:
            if col not in df.columns:
                logger.warning(f"列 {col} 不存在，跳过分词")
                continue

            output_col = f"{col}{output_suffix}"
            logger.info(f"正在分词列：{col} -> {output_col}")

            if join_str is None:
                df[output_col] = df[col].apply(
                    lambda x: self.tokenizer.tokenize(x, return_list=True)
                )
            else:
                df[output_col] = df[col].apply(
                    lambda x: self.tokenizer.tokenize(x, return_list=False)
                )

        logger.info("文本分词完成")
        return df

    def combine_texts(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        output_col: str = "combined_text",
        separator: str = " ",
    ) -> pd.DataFrame:
        """
        合并多个文本列

        Args:
            df: 输入DataFrame
            columns: 需要合并的列，None使用默认text_columns
            output_col: 输出列名
            separator: 列之间的分隔符

        Returns:
            pd.DataFrame: 添加了合并列的DataFrame
        """
        df = df.copy()
        columns = columns or self.text_columns

        # 检查列是否存在
        valid_columns = [col for col in columns if col in df.columns]
        if not valid_columns:
            logger.warning("没有可合并的文本列")
            return df

        logger.info(f"合并文本列 {valid_columns} -> {output_col}")
        df[output_col] = df[valid_columns].apply(
            lambda row: separator.join([str(val) for val in row if val]),
            axis=1,
        )
        return df

    def transform(
        self,
        df: pd.DataFrame,
        do_dedup: bool = True,
        do_dropna: bool = True,
        do_clean: bool = True,
        do_tokenize: bool = True,
        do_combine: bool = True,
    ) -> pd.DataFrame:
        """
        执行完整的数据预处理流水线

        Args:
            df: 原始DataFrame
            do_dedup: 是否去重
            do_dropna: 是否去空值
            do_clean: 是否文本清洗
            do_tokenize: 是否分词
            do_combine: 是否合并文本列

        Returns:
            pd.DataFrame: 预处理后的DataFrame
        """
        logger.info("=" * 50)
        logger.info(f"开始数据预处理，原始数据量：{len(df)}")
        logger.info("=" * 50)

        result = df

        # 1. 去重
        if do_dedup:
            result = self.remove_duplicates(result)

        # 2. 去空值
        if do_dropna:
            result = self.remove_nulls(result)

        # 3. 文本清洗
        if do_clean:
            result = self.clean_texts(result)

        # 清洗后再做一次去空值（可能清洗后产生空值）
        if do_clean and do_dropna:
            result = self.remove_nulls(result)

        # 4. 分词
        if do_tokenize:
            result = self.tokenize_texts(result)

        # 5. 合并文本
        if do_combine:
            # 先合并原始清洗后的文本
            result = self.combine_texts(
                result,
                columns=self.text_columns,
                output_col="combined_text",
            )
            # 再合并分词后的文本
            token_columns = [f"{col}_tokens" for col in self.text_columns if f"{col}_tokens" in result.columns]
            if token_columns:
                result = self.combine_texts(
                    result,
                    columns=token_columns,
                    output_col="combined_tokens",
                )

        logger.info("=" * 50)
        logger.info(f"数据预处理完成，最终数据量：{len(result)}")
        logger.info("=" * 50)

        return result


# ============================================================================
# 便捷函数
# ============================================================================
def remove_duplicates(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """便捷函数：去重"""
    preprocessor = DataPreprocessor()
    return preprocessor.remove_duplicates(df, **kwargs)


def remove_nulls(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """便捷函数：去空值"""
    preprocessor = DataPreprocessor()
    return preprocessor.remove_nulls(df, **kwargs)
