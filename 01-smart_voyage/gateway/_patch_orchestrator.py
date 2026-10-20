# -*- coding: utf-8 -*-
"""orchestrator.py 行号级别精准替换：消除郴州特化优待，所有城市用通用模板平等输出。
同时：新增 resource_note 透传到 trip_plan 字段。
"""
import re

PATH = r"d:\my_work\01-smart_voyage\gateway\orchestrator.py"

with open(PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()
orig_len = len(lines)
print(f"[INFO] 原文件总行数: {orig_len}")

# ============================================================
# 1) 区间替换：(start_line_inclusive, end_line_inclusive, new_text)
#    行号从1开始
# ============================================================
REPLACEMENTS = []

# --- Step3.5 酒店 (L1128 ~ L1338) ---
hotel_new = '''        # ========== Step 3.5: 🏨 住宿推荐（全域通用：所有城市同结构3大片区+4家酒店+入住提示，3档预算完全不同）==========
        hotels_resp = ""
        try:
            first_day_spot = ""
            if attractions:
                first_day_spot = str(attractions[0].get("name") or "")
            hotels_resp = _generate_hotels_universal(destination, budget_level, first_day_spot_hint=first_day_spot)
        except Exception as eh:
            print(f"[Orchestrator] 酒店推荐模块异常（不影响主流程）: {eh}")
            hotels_resp = f"【酒店推荐 · {destination}】{budget_level}住宿：建议订「火车站/市中心/第1天核心景区」三大片区的对应档次酒店，订房选可免费取消的，夏天带驱蚊液和薄外套。"

'''
REPLACEMENTS.append((1128, 1338, hotel_new))

# --- Step3.6 美食 (L1340 ~ L1615) ---
food_new = '''        # ========== Step 3.6: 🍜 吃喝玩乐推荐（全域通用：所有城市5大板块，预算+偏好双维度生效）==========
        food_resp = ""
        try:
            food_resp = _generate_food_universal(destination, budget_level, preferences)
        except Exception as ef:
            print(f"[Orchestrator] 吃喝玩乐推荐模块异常（不影响主流程）: {ef}")
            food_resp = f"【美食玩乐 · {destination}】吃喝攻略：优先吃城市特色正餐+夜市街边小吃，夏天喝冰粉/绿豆沙解暑，景区门口别吃饭、景区内别买特产，找本地人多的店基本不会踩雷。"

'''
REPLACEMENTS.append((1340, 1615, food_new))

# --- Step3.7 市内交通 (L1617 ~ L1668) ---
it_new = '''        # ========== Step 3.7: 🚗 市内交通体系（全域通用：所有城市按人数+预算+偏好推荐，统一黑车避雷+公交短板）==========
        internal_transport_resp = ""
        try:
            internal_transport_resp = _generate_internal_transport_universal(destination, days, budget_level, preferences)
        except Exception as eit:
            print(f"[Orchestrator] 市内交通模块异常（不影响主流程）: {eit}")
            internal_transport_resp = f"【市内交通 · {destination}】出行建议：市区用平台网约车/公交，远景区拼车或包车；车站/景区门口黑车私车别上，出租车推荐饭馆要警惕。"

'''
REPLACEMENTS.append((1617, 1668, it_new))

# --- Step3.8 预算 (L1670 ~ L1808) ---
budget_new = '''        # ========== Step 3.8: 💰 出行总预算（全域通用：所有城市6行明细表，按天数缩放，当前档位高亮）==========
        budget_resp = ""
        try:
            intercity_train_cost = 260 if (origin and destination == "郴州" and origin == "长沙") else 300
            budget_resp = _generate_budget_universal(destination, days, budget_level, intercity_train=intercity_train_cost)
        except Exception as ebg:
            print(f"[Orchestrator] 总预算模块异常（不影响主流程）: {ebg}")
            fallback_total = {"经济型": "1200-1500", "标准型": "1600-2200", "高端型": "2800-4000"}.get(budget_level, "1600-2200")
            budget_resp = f"【三日总预算 · {destination}】当前档位：{budget_level}（约{fallback_total}元/人，2人同行，含住宿/门票/吃喝/交通）"

'''
REPLACEMENTS.append((1670, 1808, budget_new))

# --- Step4 每日行程 (L1809 ~ L1959，到_correct_cities之前) ---
# 先找 _correct_cities 的起始行号（应该在L1960附近）
correct_line_idx = None
for i, ln in enumerate(lines):
    if "_correct_cities(" in ln and "internal_transport_resp" in ln:
        correct_line_idx = i
        break
print(f"[INFO] _correct_cities 起始行号（0-based）: {correct_line_idx}")
daily_end = correct_line_idx if correct_line_idx is not None else 1959
# Step4 start 是 L1809 (1-based) = 1808 (0-based)
# Step4 end 是 daily_end - 1（correct_line_idx的前一行是except Exception: pass结尾），所以1-based是 daily_end
REPLACEMENTS.append((1809, daily_end + 1 - 0 if False else daily_end + 0, '''        # ========== Step 4: 📅 每日行程安排（全域通用：所有城市上午→中午→下午→晚上顺路，景点按偏好轮询分配）==========
        daily_resp = ""
        try:
            daily_resp = _generate_daily_routes_universal(destination, days, attractions, preferences, budget_level)
        except Exception as edr:
            print(f"[Orchestrator] 每日行程模块异常，用 fallback 兜底: {edr}")
            try:
                fb = _generate_daily_routes(destination, days, "", budget_level or "标准")
                daily_resp = fb.get("response") or fb.get("tips") or f"【每日行程 · {destination}】建议{days}天分片区游玩：市中心1天+近郊1天+老城1天，避免折返。"
            except Exception:
                daily_resp = f"【📅 每日行程安排 · {destination}】共{days}天：Day1 市中心经典景点+步行街；Day2 近郊自然/人文景区；Day3 老城深度+博物馆+采购返程特产。"

'''.rstrip() + "\n\n"))

# ============================================================
# 2) 按从大到小行号排序，这样替换不会影响小区间的行号
# ============================================================
REPLACEMENTS.sort(key=lambda x: x[0], reverse=True)

for (s1, e1, new_txt) in REPLACEMENTS:
    s0, e0 = s1 - 1, e1 - 1
    print(f"[REPLACE] L{s1}-L{e1} (0-based {s0}-{e0}) 共 {e1-s1+1} 行 → 新内容 {len(new_txt.splitlines())} 行")
    if 0 <= s0 < len(lines) and 0 <= e0 < len(lines) and s0 <= e0:
        lines[s0:e0 + 1] = [new_txt]
    else:
        print(f"  [WARNING] 行号越界！跳过 s1={s1} e1={e1} len(lines)={len(lines)}")

# ============================================================
# 3) 在 trip_plan dict 最顶部插入 resource_note 字段（资源匮乏备注）
#    找 "trip_plan": { 所在行，找到后在 { 的下一行插入
# ============================================================
insert_done = False
for i, ln in enumerate(lines):
    if '"trip_plan": {' in ln and not insert_done:
        # 找 trip_plan dict 的下一行
        j = i + 1
        pad = ""
        # 获取当前行的缩进前缀
        m = re.match(r"^(\s*)", ln)
        if m: pad = m.group(1)
        indent = pad + " " * 4 * 2  # 再加两级
        # 在 trip_plan dict 开头插入一行 resource_note
        note_line = f'{indent}"resource_note": resource_note or "",  # 资源匮乏城市方案顶部提示备注\n'
        lines.insert(j, note_line)
        print(f"[INSERT] L{i+1} (trip_plan dict顶部) 插入 resource_note 字段 → 新总行数 {len(lines)}")
        insert_done = True
        break

if not insert_done:
    print("[WARNING] 未找到 trip_plan dict，resource_note 字段未插入！")

# ============================================================
# 4) 写回文件
# ============================================================
with open(PATH, "w", encoding="utf-8") as f:
    f.writelines(lines)

new_len = len(lines)
print(f"[DONE] ✅ 替换完成：原 {orig_len} 行 → 新 {new_len} 行（{'缩短' if new_len < orig_len else '增加'} {abs(orig_len - new_len)} 行）")
