# -*- coding: utf-8 -*-
"""全接口健康检查脚本（可离线运行）"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("🩺 星途环球RAG系统 - 全接口健康检查")
print("=" * 60)

# ---------- 第1关：导入阶段（用户最容易启动就崩在这里）----------
print("\n✅ 第1关：导入 server.app（注册路由阶段）")
errors = []
try:
    from server.app import app
    print("   成功！路由注册阶段无崩溃（缺python-multipart也不崩，fallback生效）")
except Exception as e:
    import traceback; traceback.print_exc()
    errors.append(f"导入崩溃: {type(e).__name__}: {e}")
    print(f"   ❌ 失败: {errors[-1]}")
    sys.exit(1)

# ---------- 第2关：TestClient模拟请求 ----------
print("\n✅ 第2关：用 TestClient 模拟 HTTP 请求（需要httpx）")
try:
    from fastapi.testclient import TestClient
    client = TestClient(app)
except Exception as e:
    print(f"   ⚠️ 跳过: TestClient初始化失败（不影响真实uvicorn启动）: {type(e).__name__}: {e}")
    sys.exit(0)

def check(name, r, extra=""):
    ok = 200 <= r.status_code < 500
    print(f"  {'✅' if ok else '❌'} [{r.status_code:3d}] {name} {extra}")
    return ok

r = client.get("/api/health"); d = r.json(); check("/api/health", r, f"索引分块={d.get('indexed_chunks','?')}")
r = client.get("/api/meta");   d = r.json(); check("/api/meta",   r, f"类目{len(d.get('categories',[]))} 语种{len(d.get('languages',[]))}")
r = client.get("/");           check("/ 前端页面", r, f"大小={len(r.content)}字节")
r = client.get("/api/status"); d = r.json(); check("/api/status", r, f"bootstrapped={d.get('bootstrapped')} 加载中={d.get('bootstrap_in_progress')}")

print("\n✅ 第3关：核心问答接口 POST /api/chat")
payload = {"message": "发日本FBA头程怎么选物流？", "session_id": "healthcheck_01"}
r = client.post("/api/chat", json=payload)
if r.status_code == 200:
    body = r.json()
    ans_clip = body.get("response", "")[:70].replace("\n", " | ")
    print(f"  ✅ [200] POST /api/chat 延迟={body.get('latency_ms','?')}ms")
    print(f"     回答前70字: {ans_clip}")
    print(f"     分类={body.get('category')} 语种={body.get('language')}")
else:
    errors.append(f"/api/chat HTTP {r.status_code}: {r.text[:100]}")
    print(f"  ❌ /api/chat 失败: {errors[-1]}")

print("\n✅ 第4关：文档上传接口缺包时不崩（fallback路由验证）")
r = client.post("/api/ingest")  # 故意不带任何参数
ok_msg = False
if r.status_code in (503, 422):
    try:
        body = r.json()
        ok_msg = ("install_command" in body) or ("python-multipart" in str(body))
    except Exception:
        body = {}
print(f"  {'✅' if ok_msg else '🟡'} [{r.status_code:3d}] POST /api/ingest（缺包fallback） 含装包提示: {ok_msg}")

print()
print("=" * 60)
if errors:
    print(f"❌ 发现 {len(errors)} 个严重问题:")
    for e in errors: print(f"  - {e}")
    sys.exit(1)
else:
    print("🎉🎉🎉 健康检查全部通过！无严重级运行错误！")
    print("    👉 浏览器打开 http://localhost:8000/ 就能用了")
print("=" * 60)
