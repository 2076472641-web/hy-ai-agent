# -*- coding: utf-8 -*-
"""预算档位 3 档差异化验证脚本：长沙→郴州 3 天，分别选经济型/标准型/高端型，对比返回结果是否真正不同"""
import asyncio, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gateway'))
from orchestrator import AgentOrchestrator

orch = AgentOrchestrator()
base_req = {
    'departLocation': {'level':'city','provinceId':'43','provinceName':'湖南省','cityId':'430100','cityName':'长沙','districtId':None,'districtName':None,'lng':None,'lat':None},
    'destLocation':   {'level':'city','provinceId':'43','provinceName':'湖南省','cityId':'431000','cityName':'郴州','districtId':None,'districtName':None,'lng':None,'lat':None},
    'departDetailAddress': '',
    'destDetailAddress': '',
    'dates': '2026-08-15',
    'days': 3,
    'preferences': ['美食','自然风光'],
}

async def main():
    results = {}
    for lvl in ['经济型','标准型','高端型']:
        req = dict(base_req)
        req['budget'] = lvl
        r = await orch.plan_full_trip(req)
        tp = r['trip_plan']
        results[lvl] = {
            'hotels_title': tp['hotels'].split('\n')[0][:80] if tp['hotels'] else '',
            'food_title':   tp['food'].split('\n')[0][:80]   if tp['food']   else '',
            'budget_title': tp['budget'].split('\n')[0][:80] if tp['budget'] else '',
            'budget_total_line': next((l for l in tp['budget'].split('\n') if '总计' in l), '')[:100],
            'daily_budget_line': next((l for l in tp['daily_routes'].split('\n') if '预算档位' in l), '')[:100],
            'H_经济': ('100-180' in tp['hotels'] or '经济型住宿' in tp['hotels']),
            'H_高端': ('280-450' in tp['hotels'] or '高端舒适' in tp['hotels']),
            'F_经济': ('省钱吃饱' in tp['food'] or '人均 15-60' in tp['food']),
            'F_高端': ('品质私藏' in tp['food'] or '东江三文鱼城' in tp['food'] or '正餐人均 100-200' in tp['food']),
        }
    sep = '=' * 110
    print(sep)
    print('【预算档位 3 档差异化验证报告】长沙→郴州·3天·2人同行')
    print(sep)
    for lvl in ['经济型','标准型','高端型']:
        d = results[lvl]
        print(f'\n▶️ {lvl}:')
        print(f'  🏨 酒店模块标题:  {d["hotels_title"]}')
        print(f'  🍜 美食模块标题:  {d["food_title"]}')
        print(f'  💰 预算模块标题:  {d["budget_title"]}')
        print(f'  💰 预算总计行:    {d["budget_total_line"]}')
        print(f'  📅 每日档位提示:  {d["daily_budget_line"]}')
        eco_pass = d['H_经济'] and d['F_经济']
        hig_pass = d['H_高端'] and d['F_高端']
        if lvl == '经济型':
            print(f'  ✅ 经济特色命中：酒店={d["H_经济"]}，美食={d["F_经济"]}，整体={"✅ 通过" if eco_pass else "❌ 失败"}')
        elif lvl == '高端型':
            print(f'  ✅ 高端特色命中：酒店={d["H_高端"]}，美食={d["F_高端"]}，整体={"✅ 通过" if hig_pass else "❌ 失败"}')
        else:
            std_pass = (not d['H_经济']) and (not d['H_高端'])
            print(f'  ✅ 标准型（非经济/非高端）：酒店非经济={not d["H_经济"]}，酒店非高端={not d["H_高端"]}，整体={"✅ 通过" if std_pass else "❌ 失败"}')
    print('\n' + sep)
    totals = [results[lvl]['budget_total_line'] for lvl in ['经济型','标准型','高端型']]
    all_diff = len(set(totals)) == 3
    titles = {lvl: (results[lvl]['hotels_title'], results[lvl]['food_title'], results[lvl]['budget_title']) for lvl in ['经济型','标准型','高端型']}
    titles_all_diff = len(set(titles.values())) == 3
    print(f'🎯 综合结论：')
    print(f'  · 3档预算总计数字是否全部不同？  {"✅ 是，3档金额有差" if all_diff else "❌ 否，重复了"}')
    print(f'  · 3档酒店+美食+预算标题均不同？ {"✅ 是，3档内容完全独立" if titles_all_diff else "❌ 否，存在重复标题"}')
    print(f'  · 最终结果：{"✅ 预算3档差异化功能正常生效！用户选择不同档位，生成的方案内容会明显不一样" if (all_diff and titles_all_diff) else "❌ 仍有问题，需要继续检查"}')
    print(sep)

asyncio.run(main())
