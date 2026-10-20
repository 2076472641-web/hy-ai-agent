"""
AI 文档脱敏工具 - 后端
双击"启动.bat"即可使用，自动打开浏览器。
"""
import os
import json
import urllib.request
import urllib.error
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static", static_url_path="")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(BASE_DIR, "api_key.txt")


def load_api_key():
    if not os.path.exists(KEY_FILE):
        return None
    with open(KEY_FILE, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line.startswith("sk-"):
                return line
    return None


SYSTEM_PROMPT = "你是严谨的技术文档工程师，逐字段提取技术需求，绝不编造原文没有的数字与指标。"

USER_PROMPT_TEMPLATE = """你是技术文档脱敏工程师。请把下面这份商业/产品文档，转成一份纯技术需求清单。
【硬性要求】
1. 逐字段提取所有接口定义，列成 Markdown 表格
2. 完整保留所有业务规则原文逻辑
3. 列出所有第三方系统/外部 API 依赖
4. 删除：公司名、产品名、人名、金额、审批流程、商业话术
5. 严禁编造原文没有的数字、性能指标
6. 用结构化 Markdown
7. 直接输出清单本身

【原文开始】
{doc}
【原文结束】
"""


def call_minimax(api_key: str, doc: str) -> str:
    payload = {
        "model": "MiniMax-M3",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(doc=doc)},
        ],
        "temperature": 0.1,
        "max_tokens": 8000,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.minimax.chat/v1/text/chatcompletion_v2",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"MiniMax 返回异常: {json.dumps(data, ensure_ascii=False)[:800]}")
    return data["choices"][0]["message"]["content"]


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/desensitize", methods=["POST"])
def desensitize():
    if "file" not in request.files:
        return jsonify({"error": "没收到文件"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "文件名为空"}), 400

    doc = f.read().decode("utf-8", errors="replace")
    if len(doc.strip()) == 0:
        return jsonify({"error": "文件内容为空"}), 400

    api_key = load_api_key()
    if not api_key:
        return jsonify({"error": "api_key.txt 里没找到 sk- 开头的 key"}), 500

    try:
        result = call_minimax(api_key, doc)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        return jsonify({"error": f"MiniMax API 错误: {detail}"}), 502
    except Exception as e:
        return jsonify({"error": f"调用失败: {e}"}), 500

    return jsonify({
        "filename": os.path.splitext(f.filename)[0] + "_脱敏后.md",
        "content": result,
        "chars": len(result),
    })


if __name__ == "__main__":
    import webbrowser
    import threading

    port = 5099
    url = f"http://127.0.0.1:{port}"

    def open_browser():
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    open_browser()
    print(f"脱敏工具已启动: {url}")
    print("关闭本窗口即可退出。")
    app.run(host="127.0.0.1", port=port, debug=False)
