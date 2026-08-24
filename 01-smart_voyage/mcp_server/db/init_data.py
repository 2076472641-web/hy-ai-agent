# -*- coding: utf-8 -*-
"""
数据库初始化脚本
将示例数据插入 MySQL 数据库
生产环境使用时先执行 schema.sql 创建表结构
"""
from __future__ import annotations

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymysql
from config import get_config


def init_database():
    """初始化数据库示例数据"""
    config = get_config()
    mysql = config.mysql

    try:
        conn = pymysql.connect(
            host=mysql.host,
            port=mysql.port,
            user=mysql.user,
            password=mysql.password,
            database=mysql.database,
            charset="utf8mb4",
        )
        cursor = conn.cursor()

        # 插入目的地数据
        destinations = [
            ("北京", "一线", "北京", "中国首都，历史文化名城", "秋季(9-11月)最佳",
             json.dumps(["历史文化", "政治中心", "四季分明"], ensure_ascii=False)),
            ("上海", "一线", "上海", "国际金融中心，东方魔都", "春秋季(3-5月,9-11月)最佳",
             json.dumps(["国际化", "都市风光", "购物"], ensure_ascii=False)),
            ("广州", "一线", "广东", "千年商都，美食之都", "冬季(11-2月)最佳",
             json.dumps(["美食", "商业", "亚热带"], ensure_ascii=False)),
            ("成都", "新一线", "四川", "天府之国，熊猫故乡", "春秋季最佳",
             json.dumps(["美食", "熊猫", "慢生活"], ensure_ascii=False)),
            ("杭州", "新一线", "浙江", "人间天堂，西湖美景", "春季(3-5月)最佳",
             json.dumps(["自然景观", "历史文化", "电商"], ensure_ascii=False)),
            ("西安", "新一线", "陕西", "十三朝古都，千年历史", "春秋季最佳",
             json.dumps(["历史文化", "美食", "古都"], ensure_ascii=False)),
            ("三亚", "二线", "海南", "热带海滨度假胜地", "冬季(10-次年4月)最佳",
             json.dumps(["海滨", "度假", "热带"], ensure_ascii=False)),
            ("大理", "二线", "云南", "风花雪月，苍山洱海", "四季皆宜",
             json.dumps(["自然景观", "民族风情", "文艺"], ensure_ascii=False)),
        ]

        for d in destinations:
            cursor.execute(
                "INSERT IGNORE INTO destinations (name, level, province, description, best_season, tags) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                d,
            )

        # 插入景点数据
        attractions = [
            # 北京
            ("故宫博物院", 1, "历史文化", 4.9, "60元", "半天",
             "世界文化遗产，明清两代皇宫，收藏大量珍贵文物", "08:30-17:00",
             "北京市东城区景山前街4号",
             json.dumps(["提前网上预约", "周一闭馆", "建议租借讲解器"], ensure_ascii=False)),
            ("长城(八达岭)", 1, "历史文化", 4.8, "40元", "半天",
             "万里长城最著名的段落之一，依山而建，气势雄伟", "07:30-17:30",
             "北京市延庆区八达岭镇",
             json.dumps(["建议穿舒适的运动鞋", "冬季注意保暖", "可乘坐索道上山"], ensure_ascii=False)),
            ("颐和园", 1, "历史文化", 4.7, "30元", "半天",
             "中国清朝时期皇家园林，保存最完整的皇家园林", "08:00-18:00",
             "北京市海淀区新建宫门路19号",
             json.dumps(["建议划船游昆明湖", "参观长廊和佛香阁"], ensure_ascii=False)),
            # 成都
            ("大熊猫繁育研究基地", 4, "自然风光", 4.8, "55元", "半天",
             "近距离观赏国宝大熊猫的最佳地点，模拟野生环境", "07:30-18:00",
             "四川省成都市成华区熊猫大道1375号",
             json.dumps(["建议早上前往，熊猫最活跃", "可观看熊猫喂食", "有熊猫医院和繁育中心"], ensure_ascii=False)),
            ("宽窄巷子", 4, "历史文化", 4.5, "免费", "2小时",
             "成都慢生活的代表，感受老成都韵味", "全天开放",
             "四川省成都市青羊区金河路口",
             json.dumps(["品尝盖碗茶", "体验掏耳朵", "购买特色文创"], ensure_ascii=False)),
            ("都江堰景区", 4, "自然风光", 4.8, "80元", "一天",
             "世界文化遗产，古代水利工程奇迹，至今仍在使用", "08:30-18:00",
             "四川省成都市都江堰市",
             json.dumps(["建议请导游讲解", "参观二王庙", "体验安澜索桥"], ensure_ascii=False)),
            # 杭州
            ("西湖", 5, "自然风光", 4.9, "免费", "半天",
             "人间天堂，淡妆浓抹总相宜，三面云山一面城", "全天开放",
             "浙江省杭州市西湖区",
             json.dumps(["租自行车环湖", "乘坐游船", "参观断桥残雪"], ensure_ascii=False)),
            ("灵隐寺", 5, "历史文化", 4.6, "75元", "3小时",
             "江南禅宗五山之一，千年古刹", "07:00-18:00",
             "浙江省杭州市西湖区法云弄1号",
             json.dumps(["参观飞来峰", "品尝素斋", "避开节假日人流"], ensure_ascii=False)),
        ]

        for a in attractions:
            cursor.execute(
                "INSERT IGNORE INTO attractions (name, city_id, category, rating, ticket_price, "
                "duration, description, opening_hours, address, tips) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                a,
            )

        # 插入交通线路数据
        routes = [
            (2, 4, "高铁", 60, 73.00, 65.00, "每小时一班", "上海虹桥站", "成都东站",
             "高铁准点率高，建议提前30分钟到站"),
            (2, 4, "飞机", 120, 1280.00, 640.00, "每日5班", "上海虹桥机场", "成都双流机场",
             "飞行时间约2小时，建议提前2小时到达机场"),
            (4, 5, "高铁", 60, 73.00, 65.00, "每30分钟一班", "成都东站", "杭州东站",
             "高铁便捷，车程约1小时"),
            (1, 2, "高铁", 300, 553.00, 498.00, "每日多班", "北京南站", "上海虹桥站",
             "京沪高铁全长1318公里，约5小时可达"),
            (1, 7, "飞机", 240, 2500.00, 1200.00, "每日3班", "首都机场", "三亚凤凰机场",
             "冬季是三亚旅游旺季，机票紧张建议提前预订"),
        ]

        for r in routes:
            cursor.execute(
                "INSERT IGNORE INTO transport_routes "
                "(origin_city_id, dest_city_id, transport_type, duration_min, "
                "price_standard, price_discount, frequency, departure, arrival, tips) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                r,
            )

        # 插入套餐数据
        packages = [
            ("成都3天2晚精华游", "成都", 3, 2, 1200.00, 1800.00, 3500.00,
             json.dumps(["大熊猫基地", "宽窄巷子", "都江堰"], ensure_ascii=False),
             json.dumps([
                 {"day": 1, "itinerary": ["大熊猫繁育研究基地", "宽窄巷子"]},
                 {"day": 2, "itinerary": ["都江堰景区", "南桥"]},
                 {"day": 3, "itinerary": ["锦里古街", "武侯祠"]},
             ], ensure_ascii=False),
             json.dumps(["酒店住宿", "早餐", "交通接送", "导游服务"], ensure_ascii=False),
             json.dumps(["门票费", "个人消费"], ensure_ascii=False)),
            ("北京5天4晚深度游", "北京", 5, 4, 2500.00, 3800.00, 6800.00,
             json.dumps(["故宫", "长城", "颐和园", "天坛"], ensure_ascii=False),
             json.dumps([
                 {"day": 1, "itinerary": ["天安门广场", "故宫博物院"]},
                 {"day": 2, "itinerary": ["八达岭长城", "奥林匹克公园"]},
                 {"day": 3, "itinerary": ["颐和园", "圆明园"]},
                 {"day": 4, "itinerary": ["天坛", "南锣鼓巷"]},
                 {"day": 5, "itinerary": ["王府井", "798艺术区"]},
             ], ensure_ascii=False),
             json.dumps(["酒店住宿", "早餐", "交通接送", "导游服务", "景点门票"], ensure_ascii=False),
             json.dumps(["个人消费", "午餐晚餐"], ensure_ascii=False)),
        ]

        for p in packages:
            cursor.execute(
                "INSERT IGNORE INTO travel_packages "
                "(name, destination, days, nights, price_economy, price_standard, price_luxury, "
                "highlights, itinerary, included, excluded) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                p,
            )

        conn.commit()
        print("[DB] 数据初始化完成！")
        print(f"  - 插入 {len(destinations)} 条目的地数据")
        print(f"  - 插入 {len(attractions)} 条景点数据")
        print(f"  - 插入 {len(routes)} 条交通线路数据")
        print(f"  - 插入 {len(packages)} 条套餐数据")

    except Exception as e:
        print(f"[DB] 初始化失败: {e}")
        raise
    finally:
        if 'conn' in locals() and conn.open:
            conn.close()


if __name__ == "__main__":
    init_database()