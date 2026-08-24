# -*- coding: utf-8 -*-
"""
数据加载模块
功能：
1. 模拟数据生成器 - 为十二大类商品生成模拟数据（标题+详情）
2. 数据加载器 - 从CSV/Excel等文件加载数据
3. 数据集划分 - 将数据集划分为训练集、验证集、测试集
"""

import os
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from loguru import logger

# 导入项目路径配置
from src import RAW_DATA_DIR, PROCESSED_DATA_DIR, CONFIG_DIR
import yaml


# ============================================================================
# 十二大类商品配置
# ============================================================================
# 从配置文件加载类目信息
def load_categories_config() -> List[Dict]:
    """
    从配置文件加载十二大类商品配置

    Returns:
        List[Dict]: 类目配置列表，每个元素包含 id、name、keywords
    """
    config_path = CONFIG_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("categories", [])


# 默认类目配置（配置文件不存在时使用）
DEFAULT_CATEGORIES = [
    {
        "id": 0,
        "name": "服装鞋帽",
        "keywords": ["T恤", "衬衫", "外套", "裤子", "裙子", "鞋子", "帽子", "袜子", "羽绒服", "卫衣", "牛仔裤", "运动鞋", "高跟鞋", "西装", "毛衣"],
        "adjectives": ["时尚", "新款", "百搭", "修身", "舒适", "透气", "韩版", "潮流", "简约", "个性", "休闲", "商务", "复古", "运动", "可爱"],
        "detail_templates": [
            "采用优质面料制作，手感柔软舒适，透气性好。设计简约时尚，百搭款式，适合多种场合穿着。精细做工，品质保证。",
            "精选高品质材料，舒适亲肤，透气性极佳。潮流版型设计，修饰身形，展现完美身材。多色可选，满足不同搭配需求。",
            "经典款式与现代元素完美结合，时尚又不失品位。舒适面料，长时间穿着也不会感到不适。细节处理到位，彰显品质。"
        ]
    },
    {
        "id": 1,
        "name": "数码家电",
        "keywords": ["手机", "电脑", "耳机", "相机", "电视", "冰箱", "洗衣机", "空调", "平板", "充电器", "数据线", "充电宝", "音箱", "路由器", "智能手表"],
        "adjectives": ["高性能", "智能", "高清", "超薄", "便携", "长续航", "快速充电", "无线", "蓝牙", "4K", "5G", "旗舰", "游戏", "商务", "家用"],
        "detail_templates": [
            "采用最新技术，性能强劲，运行流畅。高清显示屏，色彩还原真实。大容量电池，续航持久。多功能一体化设计，满足您的各种需求。",
            "高品质芯片，性能卓越，处理速度快。智能系统，操作简单便捷。精致外观设计，彰显科技感。完善的售后服务，让您购物无忧。",
            "旗舰级配置，带来极致体验。智能降噪技术，享受纯净音质。节能环保设计，为您节省开支。结实耐用，品质值得信赖。"
        ]
    },
    {
        "id": 2,
        "name": "美妆护肤",
        "keywords": ["面膜", "口红", "粉底", "面霜", "精华", "香水", "眉笔", "眼影", "睫毛膏", "卸妆", "洁面", "爽肤水", "乳液", "防晒霜", "唇膏"],
        "adjectives": ["保湿", "美白", "补水", "抗皱", "淡斑", "控油", "收缩毛孔", "防晒", "修复", "滋养", "提亮", "紧致", "舒缓", "清爽", "天然"],
        "detail_templates": [
            "萃取天然植物精华，温和不刺激，适合各种肤质使用。深层滋养肌肤，改善肌肤问题。长期使用，让肌肤焕发自然光彩。",
            "科学配方，有效成分直达肌肤底层，快速改善肌肤状态。质地清爽，易吸收，不油腻。经过严格质检，安全放心使用。",
            "蕴含多种珍贵护肤成分，全方位呵护您的肌肤。清新淡雅的香味，带来愉悦的使用体验。持续使用，肌肤日渐水润透亮。"
        ]
    },
    {
        "id": 3,
        "name": "食品生鲜",
        "keywords": ["零食", "饼干", "巧克力", "坚果", "水果", "蔬菜", "肉类", "海鲜", "牛奶", "饮料", "咖啡", "茶叶", "方便面", "火腿肠", "蛋糕"],
        "adjectives": ["新鲜", "有机", "进口", "特产", "手工", "无糖", "低糖", "营养", "美味", "香脆", "麻辣", "鲜甜", "软糯", "酥脆", "正宗"],
        "detail_templates": [
            "精选优质原料，新鲜直供，保证品质和口感。传统工艺制作，保留原汁原味。营养丰富，老少皆宜。独立包装，方便携带和储存。",
            "源自优质产地，自然生长，无添加，健康又美味。严格品控，层层把关，确保每一份都是精品。多种口味可选，满足不同味蕾。",
            "新鲜采摘/生产，冷链配送，保证新鲜度。口感纯正，回味无穷。富含多种营养成分，是居家旅行必备佳品。"
        ]
    },
    {
        "id": 4,
        "name": "家居家装",
        "keywords": ["沙发", "床", "桌子", "椅子", "衣柜", "灯具", "窗帘", "地毯", "枕头", "被子", "收纳", "装饰画", "花瓶", "厨房用具", "卫浴"],
        "adjectives": ["北欧风", "现代简约", "中式", "欧式", "轻奢", "田园", "日式", "美式", "地中海", "工业风", "环保", "实木", "多功能", "智能", "创意"],
        "detail_templates": [
            "精选环保材质，结实耐用，安全放心。简约时尚的设计风格，完美融入各种家居风格。人性化设计，使用更加便捷舒适。",
            "高品质原材料，经过多道工序精心制作。多功能设计，一物多用，节省空间。精湛工艺，细节之处彰显品质。",
            "创意设计，为您的家居增添独特魅力。易于安装和清洁，省心省力。多种尺寸和颜色可选，满足不同空间需求。"
        ]
    },
    {
        "id": 5,
        "name": "母婴用品",
        "keywords": ["奶粉", "纸尿裤", "奶瓶", "婴儿车", "玩具", "童装", "孕装", "湿巾", "辅食", "安全座椅", "童床", "书包", "文具", "绘本", "积木"],
        "adjectives": ["安全", "无毒", "环保", "舒适", "柔软", "透气", "益智", "早教", "轻便", "折叠", "防摔", "恒温", "抗菌", "亲肤", "科学"],
        "detail_templates": [
            "严格按照母婴用品标准生产，安全无毒，妈妈放心。舒适亲肤材质，呵护宝宝娇嫩肌肤。科学设计，助力宝宝健康成长。",
            "精选优质环保材料，通过多项安全认证。人性化设计，方便妈妈使用。可爱造型，宝宝喜欢。高品质，值得信赖。",
            "专为宝宝设计，考虑每一个细节。透气舒适，不闷热。易于清洗和消毒，保持卫生。陪伴宝宝快乐成长每一天。"
        ]
    },
    {
        "id": 6,
        "name": "运动户外",
        "keywords": ["篮球", "足球", "羽毛球", "跑步机", "瑜伽垫", "哑铃", "自行车", "帐篷", "睡袋", "登山鞋", "运动服", "健身器材", "滑板", "钓鱼", "游泳圈"],
        "adjectives": ["专业", "轻便", "耐用", "防滑", "防水", "透气", "速干", "高弹", "减震", "折叠", "便携", "户外", "健身", "竞技", "休闲"],
        "detail_templates": [
            "专业运动级品质，为您的运动保驾护航。采用高科技材料，轻便耐用，性能卓越。人体工学设计，使用更加舒适高效。",
            "精选高强度材料，经久耐用，安全可靠。防滑防水设计，应对各种复杂环境。多功能用途，满足不同运动需求。",
            "专为运动爱好者设计，品质过硬。透气速干面料，运动时保持干爽舒适。便于携带和收纳，随时随地享受运动乐趣。"
        ]
    },
    {
        "id": 7,
        "name": "图书文娱",
        "keywords": ["小说", "教材", "辅导书", "杂志", "文具", "乐器", "拼图", "桌游", "电子琴", "吉他", "画笔", "宣纸", "印章", "书签", "字帖"],
        "adjectives": ["正版", "精装", "畅销", "经典", "收藏", "启蒙", "益智", "专业", "入门", "进阶", "古风", "创意", "可爱", "简约", "高品质"],
        "detail_templates": [
            "精选优质内容，正版保证，值得阅读和收藏。印刷精美，纸张品质好，阅读体验佳。丰富知识，开拓视野，是学习和休闲的好选择。",
            "高品质材料制作，手感舒适，使用顺畅。创意设计，激发灵感和创造力。适合各年龄段人群，是送礼自用的佳品。",
            "经典著作/畅销作品，内容优质，深受好评。精美装帧，彰显品位。配套齐全，满足学习和娱乐的各种需求。"
        ]
    },
    {
        "id": 8,
        "name": "珠宝饰品",
        "keywords": ["项链", "戒指", "耳环", "手链", "手镯", "手表", "胸针", "发饰", "眼镜", "墨镜", "黄金", "钻石", "银饰", "珍珠", "琥珀"],
        "adjectives": ["精致", "优雅", "高贵", "时尚", "经典", "简约", "个性", "复古", "甜美", "奢华", "925银", "18K金", "天然", "定制", "限量"],
        "detail_templates": [
            "精心设计，匠心制作，每一件都是艺术品。精选优质材质，光泽持久，不易褪色。简约优雅的款式，适合日常佩戴和各种场合。",
            "高品质材料，工艺精湛，细节完美。时尚设计，彰显个人品味。精美包装，是送礼的绝佳选择。附品质保证，让您购买无忧。",
            "经典与时尚的完美融合，展现独特魅力。舒适佩戴体验，长时间佩戴也不会不适。多种款式可选，总有一款适合您。"
        ]
    },
    {
        "id": 9,
        "name": "汽车用品",
        "keywords": ["汽车坐垫", "脚垫", "香水", "行车记录仪", "导航", "轮胎", "机油", "雨刷", "车蜡", "洗车工具", "钥匙包", "手机支架", "后备箱垫", "靠枕", "挂件"],
        "adjectives": ["专用", "通用", "四季", "防水", "防滑", "耐磨", "环保", "多功能", "高清", "夜视", "智能", "便捷", "舒适", "定制", "高端"],
        "detail_templates": [
            "专为爱车设计，完美贴合，安装简单。采用优质材料，耐磨耐用，易于清洁。人性化设计，提升驾驶体验，让您的爱车更出众。",
            "高品质原材料，环保无异味，呵护您和家人的健康。多功能设计，一物多用，实用方便。精细做工，彰显对爱车的品味。",
            "精选汽车专用材质，防水防滑，经久耐用。安装便捷，无需专业工具。多种款式颜色可选，打造专属于您的爱车空间。"
        ]
    },
    {
        "id": 10,
        "name": "宠物用品",
        "keywords": ["狗粮", "猫粮", "猫砂", "狗窝", "猫爬架", "玩具", "牵引绳", "衣服", "食盆", "航空箱", "浴液", "指甲剪", "梳子", "项圈", "零食"],
        "adjectives": ["营养", "天然", "有机", "除臭", "抗菌", "耐用", "舒适", "可爱", "趣味", "训练", "便携", "保暖", "降温", "美毛", "补钙"],
        "detail_templates": [
            "专为宠物设计，选用优质安全材料。舒适耐用，给您的爱宠最好的呵护。科学配方/设计，满足宠物的各种需求。",
            "精选天然原料，营养均衡，促进宠物健康成长。严格品控，安全无毒，放心使用。可爱外观，宠物喜欢，主人满意。",
            "高品质宠物用品，结实耐用，使用长久。人性化设计，使用方便清洁简单。陪伴您的爱宠健康快乐成长。"
        ]
    },
    {
        "id": 11,
        "name": "医药保健",
        "keywords": ["维生素", "钙片", "蛋白粉", "感冒药", "创可贴", "体温计", "血压计", "血糖仪", "眼罩", "口罩", "消毒水", "止痛药", "肠胃药", "过敏药", "安神补脑"],
        "adjectives": ["正品", "官方", "天然", "草本", "进口", "中老年", "儿童", "孕妇", "增强免疫", "补钙", "补血", "调理", "养生", "家用", "便携"],
        "detail_templates": [
            "正品保证，经过严格质量检测。科学配方，功效确切。方便使用，是家庭常备的健康好帮手。关爱家人健康，从选择优质产品开始。",
            "精选优质原料，天然成分，安全无副作用。专业品牌，值得信赖。多种规格可选，满足不同人群的健康需求。",
            "严格按照国家标准生产，品质有保障。使用方便，效果显著。关爱健康，预防为主。为您和家人的健康保驾护航。"
        ]
    },
]


# ============================================================================
# 模拟数据生成器
# ============================================================================
class MockDataGenerator:
    """
    模拟商品数据生成器

    为十二大类商品生成包含标题和详情的模拟数据，
    每个类目至少生成500条数据。
    """

    def __init__(
        self,
        categories: Optional[List[Dict]] = None,
        random_seed: int = 42,
        samples_per_category: int = 500,
    ):
        """
        初始化模拟数据生成器

        Args:
            categories: 类目配置列表，为None时使用默认配置
            random_seed: 随机种子，保证数据可复现
            samples_per_category: 每个类目生成的数据条数（至少500）
        """
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)

        # 加载类目配置
        if categories is None:
            self.categories = DEFAULT_CATEGORIES
        else:
            # 合并配置文件中的类目和默认配置中的adjectives和detail_templates
            self.categories = []
            for i, cat in enumerate(categories):
                default_cat = DEFAULT_CATEGORIES[i] if i < len(DEFAULT_CATEGORIES) else DEFAULT_CATEGORIES[0]
                merged_cat = {
                    "id": cat.get("id", i),
                    "name": cat.get("name", f"类目{i}"),
                    "keywords": cat.get("keywords", default_cat["keywords"]),
                    "adjectives": default_cat.get("adjectives", []),
                    "detail_templates": default_cat.get("detail_templates", []),
                }
                self.categories.append(merged_cat)

        # 确保每个类目至少生成500条
        self.samples_per_category = max(500, samples_per_category)

        logger.info(f"模拟数据生成器初始化完成，共 {len(self.categories)} 个类目，每类生成 {self.samples_per_category} 条数据")

    def _generate_title(self, category: Dict) -> str:
        """
        为指定类目生成商品标题

        Args:
            category: 类目配置字典

        Returns:
            str: 生成的商品标题
        """
        keywords = category["keywords"]
        adjectives = category["adjectives"]

        # 随机选择1-2个关键词
        num_keywords = random.randint(1, 2)
        selected_keywords = random.sample(keywords, min(num_keywords, len(keywords)))

        # 随机选择1-3个形容词
        num_adjectives = random.randint(1, 3)
        selected_adjectives = random.sample(adjectives, min(num_adjectives, len(adjectives)))

        # 组合标题的各个部分
        parts = []

        # 添加形容词前缀
        parts.extend(selected_adjectives)

        # 品牌前缀（随机）
        brand_prefixes = ["", "[新品]", "【热销】", "【爆款】", "限时特惠", "买一送一", "品质保证", "专柜同款"]
        prefix = random.choice(brand_prefixes)
        if prefix:
            parts.insert(0, prefix)

        # 添加关键词
        parts.extend(selected_keywords)

        # 后缀（规格/促销等）
        suffixes = [
            "",
            "包邮",
            "特价",
            "促销中",
            "多色可选",
            "多规格可选",
            "送赠品",
            "礼盒装",
            "家用/商用",
            "男女通用",
            f"{random.randint(1, 10)}件装",
        ]
        suffix = random.choice(suffixes)
        if suffix:
            parts.append(suffix)

        # 用空格或连接符组合
        title = "".join(parts) if random.random() < 0.3 else " ".join(parts)

        # 随机添加一些特殊字符模拟真实数据噪声
        if random.random() < 0.15:
            noises = ["🔥", "✨", "💯", "❤️", "⭐", "🎉", "✅", "👉", "【】", "[]", "（）", "~", "·", "★", "☆"]
            noise = random.choice(noises)
            title = f"{noise}{title}{noise}" if random.random() < 0.5 else f"{title}{noise}"

        return title.strip()

    def _generate_detail(self, category: Dict, title: str) -> str:
        """
        为指定类目生成商品详情

        Args:
            category: 类目配置字典
            title: 已生成的商品标题（用于详情中引用）

        Returns:
            str: 生成的商品详情文本
        """
        templates = category["detail_templates"]
        keywords = category["keywords"]

        # 基础详情：选择1-2个模板组合
        num_templates = random.randint(1, 2)
        selected_templates = random.sample(templates, min(num_templates, len(templates)))
        detail = "\n".join(selected_templates)

        # 添加规格参数部分（50%概率）
        if random.random() < 0.5:
            params = []
            sample_keywords = random.sample(keywords, min(3, len(keywords)))
            for kw in sample_keywords:
                param_values = [
                    f"优质{kw}",
                    f"高品质{kw}",
                    f"{random.choice(['S', 'M', 'L', 'XL', 'XXL'])}码",
                    f"{random.randint(10, 1000)}g/ml",
                    f"{random.randint(1, 100)}cm",
                    f"{random.choice(['红色', '蓝色', '黑色', '白色', '粉色', '绿色', '灰色', '紫色'])}",
                ]
                params.append(f"- {kw}：{random.choice(param_values)}")

            detail += f"\n\n【规格参数】\n" + "\n".join(params)

        # 添加使用说明部分（40%概率）
        if random.random() < 0.4:
            instructions = [
                "使用前请仔细阅读说明书。",
                "请放置于阴凉干燥处保存。",
                "如有不适，请停止使用并咨询专业人士。",
                "建议定期清洁保养，延长使用寿命。",
                "本品仅供外用，请勿内服。",
            ]
            detail += f"\n\n【使用说明】\n{random.choice(instructions)}"

        # 添加售后服务部分（60%概率）
        if random.random() < 0.6:
            services = [
                "本店商品均为正品，假一赔十。",
                "支持7天无理由退换货，请放心购买。",
                "如有任何问题，请随时联系客服。",
                "全国联保，售后无忧。",
                "品质保证，质保一年。",
            ]
            detail += f"\n\n【售后服务】\n{random.choice(services)}"

        # 随机添加一些HTML标签模拟真实数据噪声（20%概率）
        if random.random() < 0.2:
            html_noises = [
                f"<p>{detail}</p>",
                f"<div>{detail}</div>",
                f"{detail}<br/><br/>更多详情请咨询客服~",
                f"<strong>商品特点：</strong>\n{detail}",
                f"<span style='color:red'>特惠中！</span>\n{detail}",
            ]
            detail = random.choice(html_noises)

        # 随机添加emoji表情（15%概率）
        if random.random() < 0.15:
            emojis = ["😊", "😍", "🥰", "😘", "👍", "👏", "💪", "🌟", "💝", "🎁"]
            emoji_str = "".join(random.sample(emojis, random.randint(1, 3)))
            detail = f"{emoji_str}\n{detail}\n{emoji_str}"

        return detail.strip()

    def generate(self) -> pd.DataFrame:
        """
        生成所有类目的模拟数据

        Returns:
            pd.DataFrame: 包含 id, category_id, category_name, title, detail 的DataFrame
        """
        logger.info("开始生成模拟商品数据...")

        records = []
        global_id = 0

        for category in self.categories:
            cat_id = category["id"]
            cat_name = category["name"]
            logger.info(f"正在生成类目【{cat_name}】的 {self.samples_per_category} 条数据...")

            for i in range(self.samples_per_category):
                title = self._generate_title(category)
                detail = self._generate_detail(category, title)

                records.append({
                    "id": global_id,
                    "category_id": cat_id,
                    "category_name": cat_name,
                    "title": title,
                    "detail": detail,
                })
                global_id += 1

        df = pd.DataFrame(records)

        # 统计数据概况
        logger.info(f"模拟数据生成完成！共 {len(df)} 条数据")
        logger.info(f"类目分布：\n{df['category_name'].value_counts().to_string()}")

        return df

    def generate_and_save(self, output_path: Optional[Path] = None) -> Path:
        """
        生成模拟数据并保存到CSV文件

        Args:
            output_path: 输出文件路径，为None时保存到默认位置

        Returns:
            Path: 保存的文件路径
        """
        if output_path is None:
            output_path = RAW_DATA_DIR / f"mock_products_{len(self.categories)}cats_{self.samples_per_category}per.csv"

        # 确保目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 生成并保存数据
        df = self.generate()
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        logger.info(f"模拟数据已保存到：{output_path}")

        return output_path


# ============================================================================
# 数据加载器
# ============================================================================
class DataLoader:
    """
    数据加载器

    支持从CSV、Excel等格式加载商品数据，并提供基本的数据校验功能。
    """

    REQUIRED_COLUMNS = ["title", "detail"]
    OPTIONAL_COLUMNS = ["category_id", "category_name"]

    def __init__(self):
        """初始化数据加载器"""
        logger.info("数据加载器初始化完成")

    def load_csv(
        self,
        file_path: Path,
        encoding: str = "utf-8-sig",
        **kwargs,
    ) -> pd.DataFrame:
        """
        从CSV文件加载数据

        Args:
            file_path: CSV文件路径
            encoding: 文件编码，默认utf-8-sig（兼容Excel）
            **kwargs: 传递给pd.read_csv的额外参数

        Returns:
            pd.DataFrame: 加载的数据

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 缺少必需的列
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在：{file_path}")

        logger.info(f"正在从CSV加载数据：{file_path}")

        df = pd.read_csv(file_path, encoding=encoding, **kwargs)

        # 数据校验
        self._validate_dataframe(df)

        logger.info(f"成功加载 {len(df)} 条数据")

        return df

    def load_excel(
        self,
        file_path: Path,
        sheet_name: str = "Sheet1",
        **kwargs,
    ) -> pd.DataFrame:
        """
        从Excel文件加载数据

        Args:
            file_path: Excel文件路径
            sheet_name: 工作表名称
            **kwargs: 传递给pd.read_excel的额外参数

        Returns:
            pd.DataFrame: 加载的数据
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在：{file_path}")

        logger.info(f"正在从Excel加载数据：{file_path}, sheet={sheet_name}")

        df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)

        # 数据校验
        self._validate_dataframe(df)

        logger.info(f"成功加载 {len(df)} 条数据")

        return df

    def load_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        直接从DataFrame加载数据（带校验）

        Args:
            df: 输入的DataFrame

        Returns:
            pd.DataFrame: 校验通过的DataFrame
        """
        self._validate_dataframe(df)
        logger.info(f"成功加载 {len(df)} 条数据（从DataFrame）")
        return df

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        校验DataFrame是否包含必需的列

        Args:
            df: 待校验的DataFrame

        Raises:
            ValueError: 缺少必需的列
        """
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]

        if missing_cols:
            raise ValueError(
                f"数据缺少必需的列：{missing_cols}。"
                f"必需列：{self.REQUIRED_COLUMNS}，"
                f"可选列：{self.OPTIONAL_COLUMNS}"
            )

        # 输出警告（如果缺少可选列）
        missing_optional = [col for col in self.OPTIONAL_COLUMNS if col not in df.columns]
        if missing_optional:
            logger.warning(f"数据缺少可选列：{missing_optional}（训练模型时需要category_id）")

    def load_mock_data(
        self,
        regenerate: bool = False,
        samples_per_category: int = 500,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        加载模拟数据（如果不存在则生成）

        Args:
            regenerate: 是否强制重新生成数据
            samples_per_category: 每个类目生成的样本数
            random_seed: 随机种子

        Returns:
            pd.DataFrame: 模拟数据DataFrame
        """
        mock_file = RAW_DATA_DIR / f"mock_products_12cats_{samples_per_category}per.csv"

        if mock_file.exists() and not regenerate:
            logger.info(f"找到已存在的模拟数据文件，直接加载：{mock_file}")
            return self.load_csv(mock_file)

        logger.info("未找到模拟数据文件或要求重新生成，开始生成...")
        generator = MockDataGenerator(
            samples_per_category=samples_per_category,
            random_seed=random_seed,
        )
        generator.generate_and_save(mock_file)

        return self.load_csv(mock_file)


# ============================================================================
# 数据集划分函数
# ============================================================================
def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify_col: str = "category_id",
    random_seed: int = 42,
    save_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    将数据集划分为训练集、验证集、测试集

    采用分层抽样，保持各子集的类目分布与整体一致。

    Args:
        df: 待划分的DataFrame
        train_ratio: 训练集比例（默认0.8）
        val_ratio: 验证集比例（默认0.1）
        test_ratio: 测试集比例（默认0.1）
        stratify_col: 分层抽样的列名（默认按category_id分层）
        random_seed: 随机种子
        save_dir: 保存划分结果的目录（为None则不保存）

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)

    Raises:
        ValueError: 比例之和不为1或分层列不存在
    """
    # 校验比例
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"数据集划分比例之和必须为1，当前为 {ratio_sum}")

    # 校验分层列
    if stratify_col not in df.columns:
        raise ValueError(f"分层列 {stratify_col} 不存在于数据中")

    logger.info(
        f"开始划分数据集：总样本数={len(df)}，"
        f"训练集={train_ratio:.0%}，验证集={val_ratio:.0%}，测试集={test_ratio:.0%}"
    )

    # 获取分层标签
    stratify_labels = df[stratify_col]

    # 第一步：划分训练集 + 临时集（验证+测试）
    test_val_ratio = 1.0 - train_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=test_val_ratio,
        random_state=random_seed,
        stratify=stratify_labels,
        shuffle=True,
    )

    # 第二步：从临时集中划分验证集和测试集
    if test_ratio > 0 and val_ratio > 0:
        val_size_in_temp = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1.0 - val_size_in_temp),
            random_state=random_seed,
            stratify=temp_df[stratify_col],
            shuffle=True,
        )
    elif val_ratio == 0:
        val_df = pd.DataFrame(columns=df.columns)
        test_df = temp_df
    else:  # test_ratio == 0
        val_df = temp_df
        test_df = pd.DataFrame(columns=df.columns)

    # 输出划分结果统计
    logger.info("数据集划分完成：")
    logger.info(f"  训练集: {len(train_df)} 条 ({len(train_df) / len(df):.1%})")
    logger.info(f"  验证集: {len(val_df)} 条 ({len(val_df) / len(df):.1%})")
    logger.info(f"  测试集: {len(test_df)} 条 ({len(test_df) / len(df):.1%})")

    # 保存划分结果（如果指定了目录）
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        train_path = save_dir / "train.csv"
        val_path = save_dir / "val.csv"
        test_path = save_dir / "test.csv"

        train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
        val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
        test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

        logger.info(f"划分结果已保存到：{save_dir}")

    return train_df, val_df, test_df


# ============================================================================
# 便捷函数
# ============================================================================
def load_csv(file_path, **kwargs) -> pd.DataFrame:
    """便捷函数：加载CSV数据"""
    loader = DataLoader()
    return loader.load_csv(file_path, **kwargs)


def save_csv(df: pd.DataFrame, file_path, **kwargs) -> None:
    """便捷函数：保存DataFrame到CSV"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False, encoding="utf-8-sig", **kwargs)
    logger.info(f"数据已保存到：{file_path}")
