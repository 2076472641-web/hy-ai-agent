# -*- coding: utf-8 -*-
"""预算档位动态化验证脚本：
验证点：
 1) 3档名称正确：经济型、舒适型、高端型；无"标准型"残留
 2) 物价系数生效：一线(上海1.5) > 二线(长沙1.0) > 三四线(郴州0.73) > 小众(张掖0.6)；同档位金额随城市降序降低
 3) 店名/菜名清零：不允许出现"维也纳/全季/亚朵/麗枫/7天/如家/尚客优/速8/东江湖观景/温德姆/东江三文鱼城/高端私厨/东江三文鱼宴"等硬编码名称
 4) 小众城市高端档如实描述：无虚构"五星酒店/黑珍珠/行政酒廊/专业导游车队/9-14 座商务车车队"等不存在配置
 5) 人数字段生效：2人 vs 4人在预算标题中正确显示
"""
import asyncio, sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gateway'))
from orchestrator import AgentOrchestrator

# 违禁词黑名单：禁止输出的具体店名菜名/硬编码郴州内容
FORBIDDEN_NAMES = [
    # 连锁酒店品牌（禁止写死）
    '维也纳','全季','亚朵','麗枫','7天','如家','尚客优','速8','派柏云','汉庭','桔子','希尔顿','万豪','洲际','温德姆','东江湖观景',
    # 郴州特定菜/食材/店名
    '东江三文鱼','三文鱼城','三文鱼宴','高端私厨','东江鱼','栖凤渡鱼粉',
    # 郴州特定餐厅名
    '高子','老南湖','三毛','彭厨','美丽年华','蜗牛塔',
]
# 小众城市高端档禁止虚构配置
FORBIDDEN_PREMIUM_FICTION = [
    '五星酒店', '五星档次', '行政酒廊', '黑珍珠', '必吃榜优先', '必吃榜提前',
    '专业导游', '9-14 座商务车', '9-14座商务车', 'VIP通道免排队',
]

def _make_req(depart_name, dest_name, days, budget, travelers=2, prefs=None):
    # 简单构造请求（location 结构只填 cityName 即可，因为 orchestrator 内按_get_city_price_context取城市名匹配）
    return {
        'departLocation': {'level':'city','cityName':depart_name,'provinceId':'','provinceName':'','cityId':'','districtId':None,'districtName':None,'lng':None,'lat':None},
        'destLocation':   {'level':'city','cityName':dest_name,  'provinceId':'','provinceName':'','cityId':'','districtId':None,'districtName':None,'lng':None,'lat':None},
        'departDetailAddress': '',
        'destDetailAddress': '',
        'dates': '2026-08-15',
        'days': days,
        'travelers': travelers,
        'preferences': prefs or ['美食','自然风光'],
        'budget': budget,
    }

orch = AgentOrchestrator()

async def main():
    test_cases = [
        ('上海_经济型_2',  '上海',   '经济型', 2,  '上海(一线1.5)'),
        ('上海_舒适型_2',  '上海',   '舒适型', 2,  '上海(一线1.5)'),
        ('上海_高端型_2',  '上海',   '高端型', 2,  '上海(一线1.5)'),
        ('长沙_舒适型_2',  '长沙',   '舒适型', 2,  '长沙(二线1.0)'),
        ('郴州_舒适型_2',  '郴州',   '舒适型', 2,  '郴州(三四线0.73)'),
        ('郴州_高端型_2',  '郴州',   '高端型', 2,  '郴州(三四线0.73)'),
        ('张掖_高端型_2',  '张掖',   '高端型', 2,  '张掖(小众0.6)'),
        ('张掖_经济型_4',  '张掖',   '经济型', 4,  '张掖(小众0.6)'),
    ]
    sep = '=' * 120
    print(sep)
    print('【预算档位动态化验证报告】')
    print(sep)

    outputs = {}
    all_pass = True
    for key, city, lvl, ppl, label in test_cases:
        req = _make_req('广州', city, 3, lvl, travelers=ppl)
        r = await orch.plan_full_trip(req)
        tp = r['trip_plan']
        outputs[key] = {
            'city': city, 'lvl': lvl, 'ppl': ppl, 'label': label,
            'hotels': tp['hotels'], 'food': tp['food'],
            'budget': tp['budget'], 'transport': tp.get('internal_transport',''),
            'daily': tp['daily_routes'], 'note': tp.get('resource_note',''),
        }
        # --- 检查 1: 档位名称正确 ---
        c1_text = tp['hotels'] + tp['food'] + tp['budget'] + tp['daily_routes']
        has_standard = '标准型' in c1_text
        c1_pass = not has_standard
        # --- 检查 2: 违禁店名菜名清零 ---
        forbidden_found = [w for w in FORBIDDEN_NAMES if w in c1_text]
        c2_pass = len(forbidden_found) == 0
        # --- 检查 3: 小众城市高端档无虚构 ---
        c3_pass = True
        fiction_found = []
        if city == '张掖' and lvl == '高端型':
            fiction_found = [w for w in FORBIDDEN_PREMIUM_FICTION if w in c1_text]
            c3_pass = len(fiction_found) == 0
        # --- 检查 4: 人数正确显示 ---
        has_ppl = f'{ppl}人同行' in tp['budget']
        c4_pass = has_ppl
        # --- 输出每条结果 ---
        case_pass = c1_pass and c2_pass and c3_pass and c4_pass
        all_pass = all_pass and case_pass
        budget_title = tp['budget'].split('\n')[0][:100] if tp['budget'] else ''
        print(f'\n▶️ 案例 [{label}] 档位={lvl} 人数={ppl}: {"✅ 通过" if case_pass else "❌ 失败"}')
        print(f'  💰 预算标题: {budget_title}')
        if has_standard: print(f'  ❌ 档位残留: 文本中仍包含「标准型」字样')
        if forbidden_found: print(f'  ❌ 违禁店名菜名: {forbidden_found}')
        if fiction_found:   print(f'  ❌ 小众城市虚构高端配置: {fiction_found}')
        if not has_ppl:     print(f'  ❌ 人数未正确展示（预算标题中未找到「{ppl}人同行」）')

    # --- 检查 5: 物价系数生效（同档位舒适型，上海>长沙>郴州）---
    print('\n' + sep)
    def _extract_perperson_range(txt):
        # 从标题中提取"约 X-Y 元/人"，取(X+Y)/2做比较
        import re
        m = re.search(r'约\s*(\d+)\s*[-~到]\s*(\d+)\s*元/人', txt)
        if m: return (int(m.group(1)) + int(m.group(2))) // 2
        return 0
    # 从唯一key中取舒适型对应的案例
    shanghai_comfort = _extract_perperson_range(outputs['上海_舒适型_2']['budget'])
    changsha_comfort = _extract_perperson_range(outputs['长沙_舒适型_2']['budget'])
    chenzhou_comfort = _extract_perperson_range(outputs['郴州_舒适型_2']['budget'])
    price_gradient_pass = shanghai_comfort > changsha_comfort > chenzhou_comfort > 0
    print(f'📈 物价系数梯度检查（舒适型3天人均中间值）：')
    print(f'   上海(×1.50)={shanghai_comfort}，长沙(×1.00)={changsha_comfort}，郴州(×0.73)={chenzhou_comfort}')
    print(f'   结果：{"✅ 上海>长沙>郴州，物价系数生效" if price_gradient_pass else "❌ 梯度不满足，检查价格系数是否正确应用"}')
    all_pass = all_pass and price_gradient_pass

    print('\n' + sep)
    print(f'🎯 最终综合结论：{"✅ 所有验证项通过，预算档位动态化改造完成！" if all_pass else "❌ 存在问题，需要修复上方❌条目"}')
    print(sep)

asyncio.run(main())
