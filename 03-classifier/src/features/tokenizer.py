# -*- coding: utf-8 -*-
"""
中文分词模块
功能：
1. 基于jieba的中文分词
2. 自定义电商领域词典加载
3. 停用词过滤
4. 词性筛选
5. Transformer截断策略（滑动窗口 + 首段尾段优先）
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import jieba
import jieba.posseg as pseg
from loguru import logger

from src import PROCESSED_DATA_DIR


# 十二大类商品类目关键词，用于构建电商领域词典
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "服装鞋帽": ["T恤", "衬衫", "外套", "裤子", "裙子", "鞋子", "帽子", "袜子",
                "羽绒服", "卫衣", "牛仔裤", "运动鞋", "高跟鞋", "西装", "毛衣",
                "连衣裙", "夹克", "风衣", "马甲", "打底衫", "休闲裤", "短裤",
                "半身裙", "凉鞋", "拖鞋", "靴子", "棒球帽", "针织衫"],
    "数码家电": ["手机", "电脑", "耳机", "相机", "电视", "冰箱", "洗衣机", "空调",
                "平板", "充电器", "数据线", "充电宝", "音箱", "路由器", "智能手表",
                "笔记本", "显示器", "键盘", "鼠标", "显卡", "主板", "固态硬盘",
                "投影仪", "吸尘器", "电饭煲", "微波炉", "电磁炉", "吹风机"],
    "美妆护肤": ["面膜", "口红", "粉底", "面霜", "精华", "香水", "眉笔", "眼影",
                "睫毛膏", "卸妆", "洁面", "爽肤水", "乳液", "防晒霜", "唇膏",
                "隔离霜", "遮瑕膏", "散粉", "腮红", "高光", "眼线笔", "美甲",
                "护发素", "洗发水", "沐浴露", "身体乳", "护手霜", "磨砂膏"],
    "食品生鲜": ["零食", "饼干", "巧克力", "坚果", "水果", "蔬菜", "肉类", "海鲜",
                "牛奶", "饮料", "咖啡", "茶叶", "方便面", "火腿肠", "蛋糕",
                "面包", "糖果", "薯片", "果冻", "蜂蜜", "麦片", "芝麻糊",
                "牛排", "三文鱼", "小龙虾", "大闸蟹", "车厘子", "榴莲"],
    "家居家装": ["沙发", "床", "桌子", "椅子", "衣柜", "灯具", "窗帘", "地毯",
                "枕头", "被子", "收纳", "装饰画", "花瓶", "厨房用具", "卫浴",
                "茶几", "电视柜", "书柜", "餐桌", "餐椅", "床垫", "床头柜",
                "吊灯", "落地灯", "抱枕", "四件套", "置物架", "晾衣架"],
    "母婴用品": ["奶粉", "纸尿裤", "奶瓶", "婴儿车", "玩具", "童装", "孕装", "湿巾",
                "辅食", "安全座椅", "童床", "书包", "文具", "绘本", "积木",
                "奶嘴", "吸奶器", "待产包", "爬爬垫", "学步车", "儿童手表",
                "橡皮泥", "乐高", "拼图", "遥控车", "芭比娃娃", "奥特曼"],
    "运动户外": ["篮球", "足球", "羽毛球", "跑步机", "瑜伽垫", "哑铃", "自行车",
                "帐篷", "睡袋", "登山鞋", "运动服", "健身器材", "滑板", "钓鱼",
                "游泳圈", "乒乓球", "网球", "高尔夫", "滑雪板", "冲浪板",
                "烧烤架", "望远镜", "指南针", "登山杖", "头盔", "护膝", "护腕"],
    "图书文娱": ["小说", "教材", "辅导书", "杂志", "文具", "乐器", "拼图", "桌游",
                "电子琴", "吉他", "画笔", "宣纸", "印章", "书签", "字帖",
                "漫画", "诗集", "传记", "历史书", "科普书", "素描本", "水彩",
                "小提琴", "架子鼓", "萨克斯", "围棋", "象棋", "扑克牌"],
    "珠宝饰品": ["项链", "戒指", "耳环", "手链", "手镯", "手表", "胸针", "发饰",
                "眼镜", "墨镜", "黄金", "钻石", "银饰", "珍珠", "琥珀",
                "翡翠", "玉石", "铂金", "钯金", "宝石", "锆石", "水晶",
                "发夹", "发箍", "丝巾扣", "皮带扣", "袖扣", "领带夹"],
    "汽车用品": ["汽车坐垫", "脚垫", "香水", "行车记录仪", "导航", "轮胎", "机油",
                "雨刷", "车蜡", "洗车工具", "钥匙包", "手机支架", "后备箱垫",
                "靠枕", "挂件", "座套", "方向盘套", "车衣", "遮阳挡", "掸子",
                "充气泵", "胎压计", "搭电线", "拖车绳", "灭火器", "安全锤"],
    "宠物用品": ["狗粮", "猫粮", "猫砂", "狗窝", "猫爬架", "玩具", "牵引绳", "衣服",
                "食盆", "航空箱", "浴液", "指甲剪", "梳子", "项圈", "零食",
                "自动喂食器", "饮水机", "猫包", "狗绳", "嘴套", "拾便袋",
                "鱼油", "营养膏", "化毛膏", "驱虫药", "疫苗", "体检套餐"],
    "医药保健": ["维生素", "钙片", "蛋白粉", "感冒药", "创可贴", "体温计", "血压计",
                "血糖仪", "眼罩", "口罩", "消毒水", "止痛药", "肠胃药", "过敏药",
                "安神补脑", "鱼油", "卵磷脂", "螺旋藻", "褪黑素", "益生菌",
                "按摩仪", "颈椎枕", "护眼仪", "足浴盆", "艾灸盒", "拔罐器"],
}

# 默认保留的词性（名词、动词、形容词、副词等实义词）
DEFAULT_ALLOWED_POS = {
    "n", "nr", "ns", "nt", "nz",    # 名词类
    "v", "vd", "vn",                # 动词类
    "a", "ad", "an",                # 形容词类
    "d", "z",                       # 副词、状态词
    "eng", "x",                     # 英文、非语素字
}


class ChineseTokenizer:
    """中文分词器，集成电商领域词典、停用词、词性筛选和长文本截断策略"""

    def __init__(
        self,
        custom_dict_path: Optional[str] = None,
        stopwords_path: Optional[str] = None,
        use_stopwords: bool = True,
        min_token_length: int = 1,
        enable_pos_filter: bool = False,
        allowed_pos: Optional[set] = None,
        max_seq_length: int = 128,
        truncate_strategy: str = "head_tail",
    ):
        """
        初始化中文分词器

        Args:
            custom_dict_path: 自定义电商词典路径，默认使用 data/processed/ecommerce_dict.txt
            stopwords_path: 停用词表路径，默认使用 data/processed/stopwords.txt
            use_stopwords: 是否启用停用词过滤
            min_token_length: 最小词长度，短于此值的词会被过滤
            enable_pos_filter: 是否启用词性筛选
            allowed_pos: 允许的词性集合，默认使用 DEFAULT_ALLOWED_POS
            max_seq_length: 最大序列长度（用于Transformer截断）
            truncate_strategy: 截断策略，可选 "head"（首部）、"tail"（尾部）、
                              "head_tail"（首尾优先）、"sliding_window"（滑动窗口）
        """
        self.use_stopwords = use_stopwords
        self.min_token_length = min_token_length
        self.enable_pos_filter = enable_pos_filter
        self.allowed_pos = allowed_pos or DEFAULT_ALLOWED_POS
        self.max_seq_length = max_seq_length
        self.truncate_strategy = truncate_strategy

        # 确保目录存在
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # 加载自定义电商词典
        self.custom_dict_path = custom_dict_path or str(PROCESSED_DATA_DIR / "ecommerce_dict.txt")
        self._load_custom_dict()

        # 加载停用词
        self.stopwords_path = stopwords_path or str(PROCESSED_DATA_DIR / "stopwords.txt")
        self.stopwords: set = set()
        if self.use_stopwords:
            self._load_stopwords()

        logger.info(
            f"中文分词器初始化完成 | 停用词数: {len(self.stopwords)} | "
            f"词性筛选: {self.enable_pos_filter} | 截断策略: {self.truncate_strategy}"
        )

    def _load_custom_dict(self) -> None:
        """加载自定义电商领域词典，若不存在则自动生成"""
        dict_path = Path(self.custom_dict_path)

        if not dict_path.exists():
            logger.info(f"电商词典不存在，自动生成: {dict_path}")
            self._generate_ecommerce_dict()

        try:
            jieba.load_userdict(str(dict_path))
            logger.info(f"电商词典加载成功: {dict_path}")
        except Exception as e:
            logger.warning(f"电商词典加载失败: {e}，使用默认词典")

    def _generate_ecommerce_dict(self) -> None:
        """生成电商领域词典文件，包含十二大类目的关键词"""
        dict_path = Path(self.custom_dict_path)
        dict_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dict_path, "w", encoding="utf-8") as f:
            for category, keywords in CATEGORY_KEYWORDS.items():
                # 格式：词语 词频 词性
                for kw in keywords:
                    f.write(f"{kw} 1000 n\n")

            # 补充常见电商词缀
            extra_terms = [
                ("包邮", 800, "d"), ("秒杀", 800, "v"), ("新品", 800, "n"),
                ("特价", 800, "a"), ("正品", 800, "a"), ("旗舰店", 800, "n"),
                ("官方", 800, "a"), ("专柜", 700, "n"), ("代购", 700, "v"),
                ("现货", 700, "n"), ("预售", 700, "n"), ("限量", 700, "a"),
                ("赠品", 700, "n"), ("套餐", 700, "n"), ("尺码", 700, "n"),
                ("颜色", 700, "n"), ("规格", 700, "n"), ("型号", 700, "n"),
            ]
            for term, freq, pos in extra_terms:
                f.write(f"{term} {freq} {pos}\n")

        logger.info(f"电商词典生成完毕，共 {len(CATEGORY_KEYWORDS)} 个类目关键词 + 补充电商词缀")

    def _load_stopwords(self) -> None:
        """加载停用词表，若不存在则生成默认停用词表"""
        sw_path = Path(self.stopwords_path)

        if not sw_path.exists():
            logger.info(f"停用词表不存在，自动生成: {sw_path}")
            self._generate_default_stopwords()

        try:
            with open(sw_path, "r", encoding="utf-8") as f:
                self.stopwords = {line.strip() for line in f if line.strip()}
            logger.info(f"停用词加载成功，共 {len(self.stopwords)} 个停用词")
        except Exception as e:
            logger.warning(f"停用词加载失败: {e}")
            self.stopwords = set()

    def _generate_default_stopwords(self) -> None:
        """生成默认中文停用词表"""
        sw_path = Path(self.stopwords_path)
        sw_path.parent.mkdir(parents=True, exist_ok=True)

        default_stopwords = [
            # 语气助词
            "的", "了", "啊", "呢", "吧", "吗", "哦", "嗯", "哈", "呀",
            "啦", "呗", "哇", "嘛", "哩", "喽", "咯", "呕", "耶", "呜",
            # 结构助词
            "地", "得", "所", "之", "于", "以", "及", "与", "或", "而",
            "且", "则", "但", "然", "虽", "若", "如", "因", "故", "所以",
            # 代词
            "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们",
            "这", "那", "这个", "那个", "这些", "那些", "这里", "那里",
            "谁", "什么", "怎么", "哪", "哪里", "为什么", "如何",
            # 介词连词
            "在", "对", "为", "从", "到", "把", "被", "让", "给", "向",
            "往", "由", "沿", "通过", "经过", "根据", "按照", "关于",
            "和", "跟", "同", "与", "及", "或", "还是", "或者",
            # 数量相关
            "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
            "百", "千", "万", "亿", "个", "只", "件", "条", "双", "套",
            "款", "种", "类", "些", "每", "各", "某", "该", "此",
            # 电商无效词
            "包邮", "秒杀", "特价", "正品", "官方", "现货", "预售",
            "新品", "热卖", "爆款", "促销", "打折", "优惠", "活动",
            "链接", "详情", "客服", "咨询", "下单", "购买", "购物",
            # 标点符号
            ",", ".", "，", "。", "!", "！", "?", "？", ";", "；",
            ":", "：", "(", ")", "（", "）", "[", "]", "【", "】",
            "/", "\\", "-", "_", "=", "+", "*", "&", "^", "%", "$",
            "#", "@", "~", "`", "<", ">", "《", "》", "\"", "'",
        ]

        with open(sw_path, "w", encoding="utf-8") as f:
            for sw in default_stopwords:
                f.write(f"{sw}\n")

        logger.info(f"默认停用词表生成完毕，共 {len(default_stopwords)} 个停用词")

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        文本清洗：去除特殊字符、多余空格，统一半角/全角

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not isinstance(text, str):
            return ""

        # 去除HTML标签
        text = re.sub(r"<[^>]+>", "", text)
        # 去除URL
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        # 统一全角空格为半角
        text = text.replace("\u3000", " ")
        # 去除多余空白
        text = re.sub(r"\s+", " ", text).strip()
        # 去除控制字符
        text = re.sub(r"[\x00-\x1f\x7f]", "", text)

        return text

    def tokenize(
        self,
        text: str,
        return_pos: bool = False,
    ) -> List:
        """
        对文本进行分词

        Args:
            text: 输入文本
            return_pos: 是否返回词性 (token, pos) 对

        Returns:
            分词结果列表；return_pos=True 时返回 List[Tuple[str, str]]
        """
        text = self._clean_text(text)
        if not text:
            return []

        if self.enable_pos_filter:
            # 带词性分词
            tokens_with_pos = []
            for word, pos in pseg.cut(text):
                word = word.strip()
                if not word:
                    continue
                # 词性筛选
                if pos not in self.allowed_pos:
                    continue
                # 停用词过滤
                if self.use_stopwords and word in self.stopwords:
                    continue
                # 最小长度过滤
                if len(word) < self.min_token_length:
                    continue
                tokens_with_pos.append((word, pos))

            if return_pos:
                return tokens_with_pos
            return [t[0] for t in tokens_with_pos]
        else:
            # 普通分词
            tokens = []
            for word in jieba.cut(text):
                word = word.strip()
                if not word:
                    continue
                if self.use_stopwords and word in self.stopwords:
                    continue
                if len(word) < self.min_token_length:
                    continue
                tokens.append(word)

            if return_pos:
                # 不带词性筛选时补全词性为 n
                return [(t, "n") for t in tokens]
            return tokens

    def tokenize_batch(
        self,
        texts: List[str],
        return_pos: bool = False,
    ) -> List[List]:
        """
        批量分词

        Args:
            texts: 文本列表
            return_pos: 是否返回词性

        Returns:
            批量分词结果
        """
        return [self.tokenize(text, return_pos=return_pos) for text in texts]

    def truncate_for_transformer(
        self,
        text: str,
        max_length: Optional[int] = None,
        strategy: Optional[str] = None,
        overlap: int = 20,
    ) -> List[str]:
        """
        针对Transformer的长文本截断策略

        支持四种策略：
        - head: 仅保留首部 max_length 个字符
        - tail: 仅保留尾部 max_length 个字符
        - head_tail: 首段 + 尾段拼接（商品标题通常首尾信息更重要）
        - sliding_window: 滑动窗口切分，重叠overlap个字符

        Args:
            text: 原始文本
            max_length: 最大长度，默认使用 self.max_seq_length
            strategy: 截断策略，默认使用 self.truncate_strategy
            overlap: 滑动窗口重叠字符数

        Returns:
            截断后的文本片段列表
        """
        max_length = max_length or self.max_seq_length
        strategy = strategy or self.truncate_strategy

        text = self._clean_text(text)
        if len(text) <= max_length:
            return [text]

        if strategy == "head":
            return [text[:max_length]]

        elif strategy == "tail":
            return [text[-max_length:]]

        elif strategy == "head_tail":
            # 首段取前 60%，尾段取后 40%；商品标题通常品牌属性在前、规格促销在后
            head_len = int(max_length * 0.6)
            tail_len = max_length - head_len
            head_part = text[:head_len]
            tail_part = text[-tail_len:] if tail_len > 0 else ""
            combined = head_part + tail_part
            return [combined[:max_length]]

        elif strategy == "sliding_window":
            # 滑动窗口切分，窗口间有 overlap 字符重叠
            step = max_length - overlap
            chunks = []
            start = 0
            while start < len(text):
                chunk = text[start:start + max_length]
                if len(chunk) >= self.min_token_length * 2:
                    chunks.append(chunk)
                if start + max_length >= len(text):
                    break
                start += step
            return chunks if chunks else [text[:max_length]]

        else:
            logger.warning(f"未知截断策略: {strategy}，使用默认 head 策略")
            return [text[:max_length]]

    def extract_keywords_with_category(
        self,
        text: str,
        top_k: int = 10,
    ) -> Dict[str, List[str]]:
        """
        从文本中提取各分类关键词命中情况

        Args:
            text: 输入文本
            top_k: 每类最多返回命中数

        Returns:
            {类目名: [命中关键词列表]} 字典
        """
        tokens = set(self.tokenize(text))
        result = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            hits = [kw for kw in keywords if kw in tokens or kw in text]
            if hits:
                result[category] = hits[:top_k]
        return result

    def __repr__(self) -> str:
        return (
            f"ChineseTokenizer(use_stopwords={self.use_stopwords}, "
            f"min_token_length={self.min_token_length}, "
            f"enable_pos_filter={self.enable_pos_filter}, "
            f"truncate_strategy='{self.truncate_strategy}', "
            f"max_seq_length={self.max_seq_length})"
        )
