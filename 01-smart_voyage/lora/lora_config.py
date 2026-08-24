# -*- coding: utf-8 -*-
"""
LoRA 配置 - 文旅出行场景微调参数
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoRAConfig:
    """LoRA 微调配置"""

    # 基础模型
    base_model_name: str = "qwen2.5-7b-instruct"
    model_path: str = ""  # 本地模型路径

    # LoRA 参数
    lora_r: int = 8              # LoRA 秩
    lora_alpha: int = 16          # LoRA alpha 参数
    lora_dropout: float = 0.05    # LoRA dropout
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # 训练参数
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_seq_length: int = 2048
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

    # 数据路径
    train_data_path: str = "./data/train"
    output_dir: str = "./models/lora-output"

    # 其他
    seed: int = 42
    fp16: bool = True
    gradient_checkpointing: bool = True

    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model_name,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "batch_size": self.per_device_train_batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.num_train_epochs,
            "max_seq_length": self.max_seq_length,
        }


# 文旅场景的指令模板
TRAVEL_INSTRUCTION_TEMPLATE = """你是途景汇文旅智能助手。根据用户的需求，提供专业的旅游建议。

用户：{instruction}
助手：{response}"""

# 训练数据示例
SAMPLE_TRAIN_DATA = [
    {
        "instruction": "帮我查一下成都明天的天气",
        "input": "",
        "output": "成都明天多云转晴，温度18-26℃，空气质量良，适合出行。建议穿着长袖衬衫，注意防晒。"
    },
    {
        "instruction": "推荐北京3天游玩的景点",
        "input": "",
        "output": "北京3天行程推荐：\n第1天：天安门广场→故宫博物院→景山公园\n第2天：八达岭长城一日游\n第3天：颐和园→圆明园→南锣鼓巷\n建议优先预约故宫门票，交通以地铁为主。"
    },
    {
        "instruction": "从上海到杭州怎么去最方便",
        "input": "",
        "output": "从上海到杭州推荐乘坐高铁：\n1. 高铁：上海虹桥→杭州东，约1小时，票价73元，班次密集\n2. 自驾：约3小时，距离约180公里\n推荐选择高铁，准点率高、舒适度好。"
    },
    {
        "instruction": "我想下个月去三亚旅游，需要准备什么",
        "input": "",
        "output": "三亚旅游准备清单：\n1. 证件：身份证、护照（如需出境）\n2. 衣物：夏季轻薄衣物、泳衣、遮阳帽、太阳镜\n3. 防护：高倍防晒霜（SPF50+）、驱蚊水\n4. 药品：晕车药、肠胃药、止痛药\n5. 其他：防水手机袋、充电宝、少量现金\n最佳旅游时间：10月至次年4月"
    },
    {
        "instruction": "杭州有哪些值得去的景点",
        "input": "",
        "output": "杭州推荐景点：\n1. 西湖（免费）- 人间天堂，环湖骑行\n2. 灵隐寺（75元）- 千年古刹，禅意深厚\n3. 千岛湖（150元）- 湖光山色，千岛星罗\n4. 宋城（320元）- 给我一天，还你千年\n5. 河坊街（免费）- 南宋御街，美食聚集\n春秋季是最佳旅游季节。"
    },
    {
        "instruction": "帮我规划一个西安5天的行程",
        "input": "",
        "output": "西安5天行程：\n第1天：钟楼→回民街→大雁塔夜景\n第2天：秦始皇兵马俑→华清池\n第3天：陕西历史博物馆→大雁塔→大唐不夜城\n第4天：华山一日游\n第5天：古城墙骑行→书院门→碑林博物馆\n美食推荐：肉夹馍、羊肉泡馍、biangbiang面、甑糕"
    },
    {
        "instruction": "成都有什么好吃的美食推荐",
        "input": "",
        "output": "成都必尝美食：\n1. 火锅：麻辣鲜香，推荐小龙翻大江、蜀大侠\n2. 串串香：街头巷尾的美味\n3. 龙抄手：皮薄馅大\n4. 钟水饺：甜辣可口\n5. 担担面：经典面食\n6. 蛋烘糕：成都特色小吃\n建议去宽窄巷子、锦里、春熙路品尝美食。"
    },
    {
        "instruction": "冬天去北京旅游合适吗",
        "input": "",
        "output": "冬天去北京旅游有独特魅力：\n优点：\n1. 雪景美：颐和园、故宫雪景如画\n2. 年味浓：庙会、年货市场热闹\n3. 人流量少：景点不拥挤\n注意：\n1. 气温低：零下5-10℃，需穿羽绒服\n2. 天气干燥：注意补水保湿\n3. 部分室外景点游览时间缩短\n建议搭配室内景点（博物馆、剧院）一起安排。"
    },
]