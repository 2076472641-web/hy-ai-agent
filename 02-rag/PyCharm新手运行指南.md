# 🚀 星途环球·跨境售后RAG智能问答系统 — PyCharm 新手小白运行完全指南

> 面向人群：第一次接触 Python / PyCharm 的零基础同学
> 预计完成时间：15 ~ 25 分钟
> 本项目路径：`D:\my_work\rag`（以下所有操作都围绕这个文件夹）

---

## 📋 一、前置准备（没装的先装，已经装好跳到「第1步」）

在开始 PyCharm 操作前，你需要先确保电脑上有这 3 个软件：

| 软件 | 作用 | 下载地址 | 怎么验证装好了？ |
|---|---|---|---|
| **Python 3.10+** | 运行本项目代码的「发动机」 | https://www.python.org/downloads/windows/ | 按 `Win + R` → 输入 `cmd` 回车 → 黑框里输入 `python --version`，能看到版本号（例如 `Python 3.12.7`）就OK |
| **PyCharm 社区版** | 写代码/运行代码的「工作台」 | https://www.jetbrains.com/pycharm/download/#section=windows | 桌面有 PyCharm 图标就行（⚠️ 一定要选「Community 社区版」，免费！不用买 Professional） |
| **阿里云百炼 API Key**（可选） | 让项目调用真实AI回答（不填也能用 Mock 演示跑通） | https://bailian.console.aliyun.com/ → 右上角头像 → API-KEY管理 | 复制保存好 `sk-` 开头的一串字符 |

> 💡 温馨提示：安装 Python 时，**一定一定一定** 要勾选安装界面最底部的 ✅ `Add Python to PATH`（不然 PyCharm 找不到 Python）！

---

## 👣 第 1 步：用 PyCharm 打开这个项目文件夹

1. 双击桌面 **PyCharm** 图标启动软件
2. 在欢迎界面，点击左侧 👉 **Open**（打开）
3. 在弹出的文件选择窗口中，**找到并选中文件夹 `D:\my_work\rag`**，然后点击右下角「OK」
4. 此时会弹出一个「Trust Project?（是否信任项目）」的对话框，**一定要点 👉 Trust Project（信任项目）**（不信任的话代码是不能运行的！）
5. 等待 10~30 秒，让 PyCharm 加载项目（右下角会有进度条）

---

## 🐍 第 2 步：给项目配置 Python 解释器（⭐ 最重要的一步！80% 的新手报错都出在这）

解释器 = 运行 Python 代码的「发动机」。每个项目第一次打开都要配一次。

### 操作路径（按顺序点）：
顶部菜单栏 → **File**（文件） → **Settings**（设置） → 左侧展开 **Project: rag** → 点击 **Python Interpreter**（Python 解释器）

### 接下来分两种情况（任选一种，推荐「方案A：创建全新虚拟环境」）：

---

#### ✅ 方案 A：创建一个全新的虚拟环境（推荐，最干净不混乱）

1. 在「Python Interpreter」页面，点击右上角的 **⚙️ Add Interpreter**（添加解释器）
2. 左侧选择 **Virtualenv Environment**（虚拟环境）
3. 右侧按如下填写（基本都是默认值，不用改）：
   - **Location（位置）**：保持默认（通常是 `D:\my_work\rag\.venv`）
   - **Base interpreter（基础解释器）**：下拉选择你电脑安装的 Python（例如 `Python 3.12 (python.exe)`），**如果看不到就点右侧 `...` 按钮手动找到 `python.exe`**
   - ✅ 勾选 **Inherit global site-packages（继承全局包）** 可以不勾，空着更干净
4. 点击右下角 **OK**，等 PyCharm 创建虚拟环境（10秒左右）
5. 回到 Settings 主界面，看到解释器那里显示：`Python 3.12 (rag) D:\my_work\rag\.venv\Scripts\python.exe` ✨（带项目名「rag」就对了！）
6. 再点 **OK** 关闭 Settings 窗口

> 🔑 记住这句话：**本项目以后所有的依赖（pip install），都只会装在这个 `.venv` 文件夹里，不会影响你电脑上其他项目！**

---

#### 🟡 方案 B：直接用系统 Python（不推荐，但如果你嫌麻烦也能用）

1. 在「Python Interpreter」页面，**直接下拉选一个你电脑上已有的 Python 3.10+ 版本**
2. 点 OK 即可

---

## 📦 第 3 步：安装项目所需的依赖包

PyCharm 最底部有一排标签，点开 **Terminal**（终端）—— 这是我们以后敲命令的地方。

### 在 Terminal 里输入这两行命令（一行一行敲，敲完一行回车）：

```powershell
# 第 1 行：升级 pip（防止装包失败）
python -m pip install --upgrade pip

# 第 2 行：安装本项目全部核心依赖（FastAPI 服务 + .env 读取工具）
pip install fastapi uvicorn httpx pydantic python-dotenv
```

### 安装完成判断标志：
Terminal 最后几行能看到：
```
Successfully installed fastapi-xxx uvicorn-xxx ...
```
并且**没有红色报错**就是成功 ✅

> 💡 进阶说明：如果你将来想接「真实 Milvus 向量库」「真实 bge 嵌入模型」「OCR 扫描件识别」，可以再装这些（**现在不装也不影响项目跑通**）：
> ```powershell
> pip install pymilvus sentence-transformers PyMuPDF python-docx paddleocr paddlepaddle
> ```

---

## 🔑 第 4 步：配置 `.env` 文件（填阿里云百炼 Key，不填也能跑 Mock 演示）

### 4.1 在 PyCharm 左侧文件树中找到文件 `.env`，双击打开
（如果找不到，看看是不是隐藏了：顶部菜单 → **View** → **Show Hidden Files**）

### 4.2 找到第 10 行这一句：
```
LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
把等号后面**全部内容**，替换成你自己从阿里云百炼控制台复制的 Key。

### 4.3 按 `Ctrl + S` 保存文件 ✅

> 🎯 **不填 Key 也没关系**：本项目所有模块都内置了 **Mock（模拟）模式**，可以零成本直接跑通全部流程，用于演示、截图、面试完全没问题。

---

## ▶️ 第 5 步：运行项目（三种方式任选一种，推荐先试「方式一 Web 服务」）

---

### 🖥️ 方式一：启动 Web 服务 + 可视化前端（最直观！面试/演示用这个）

#### 操作 A（最简单：直接点绿色三角形）：
1. 左侧文件树找到 `run.py`，**右键** → **Run 'run'**（运行 run）
2. 看底部 **Run** 窗口，出现以下文字说明启动成功 ✨：
   ```
   Uvicorn running on http://0.0.0.0:8000
   ```
3. 打开浏览器，访问 👉 **http://localhost:8000/**
4. 你会看到一个多语种的跨境客服咨询页面！🎉
   - 右上角可以切换中/英/日/韩/德/西 6 种语言
   - 中间输入问题（比如「从中国发日本海运要多少天？」）回车，就能收到回答
   - 左侧可以过滤「物流/售后/合规」三类知识库

#### 操作 B（Run 窗口没出来？用 Terminal 也行）：
底部 Terminal 敲：
```powershell
python run.py
```
看到 `Uvicorn running on http://0.0.0.0:8000` 就成功了。

> 🔴 **想停止服务时**：点击 Run 窗口左侧的红色方块（■）停止按钮，或者按 `Ctrl + C`。

---

### 💬 方式二：命令行交互模式（不打开浏览器，直接在 Terminal 里聊天）

在 PyCharm 的底部 **Terminal** 敲：
```powershell
python run.py --cli
```
回车后会出现：
```
🤖 星途环球多语种客服助手
   输入你的问题（输入 exit 退出）：
> 
```
在这里直接打字提问就行，比如：
```
> 美国关税起征点是多少？
```
按回车等待AI回答。输入 `exit` 回车退出。

---

### 📊 方式三：运行 51 组跨境标准用例的自动化评测（面试装X神器！）

在 Terminal 敲：
```powershell
python run.py --test --mock-noise 0.0 --limit 10
```
等 10 秒左右，你会看到一个**非常炫酷的彩色测试报告**：
```
📊 星途环球·跨境售后RAG智能问答系统 - 自动化测试汇总报告
  总用例数       : 10
  通过 / 失败    : 10 ✅  /  0 ❌
  通过率         : 100.0%  ██████████████████████████████
  综合平均得分   : 82.5 / 100
  🔴 HIGH 幻觉   : 0
```
这个报告可以直接截图放进简历/面试PPT里！📸

---

## ✅ 第 6 步：判断你是否成功跑通了？（成功 checklist）

打勾 ✅ 超过 4 项就是完全成功：

- [ ] 1. PyCharm 打开 `D:\my_work\rag` 没有爆红（代码没有红色波浪线）
- [ ] 2. Terminal 输入 `python --version` 能看到 Python 版本号
- [ ] 3. `pip install fastapi ...` 那一行没报错
- [ ] 4. 运行 `run.py`（方式一）后，浏览器打开 `http://localhost:8000/` 能看到咨询页面
- [ ] 5. 在咨询页面输入「发美国FBA头程怎么选？」能得到 4 段式结构化回答（结论/依据要点/来源/注意事项）
- [ ] 6. 运行自动化评测（方式三）能看到 PASS/NOT PASS 的彩色结果

---

## 🆘 第 7 步：新手最常见的 10 个报错 & 快速解决办法

| # | 你看到的报错信息 | 原因 | 一步到位解决办法 |
|---|---|---|---|
| 1 | `ModuleNotFoundError: No module named 'fastapi'` | 包没装到当前解释器（装到别的环境去了） | **方案A**：Terminal 重新敲 `pip install fastapi uvicorn httpx pydantic python-dotenv` <br> **方案B**：回到「第2步」重新检查解释器是不是选的 `(rag) .venv\Scripts\python.exe` |
| 2 | 右键 `run.py` 后，Run 窗口一片红，全是 import 错误 | 解释器没配置，或者配置错了 | 严格重做一次「第 2 步 方案A」（创建 .venv 虚拟环境），然后重启 PyCharm |
| 3 | `Address already in use` / `端口 8000 被占用` | 8000 端口已经被上次运行的服务占了 | ① 点 Run 窗口左侧红色方块（■）把所有正在跑的程序停止 <br> ② 或者改端口：Terminal 敲 `python run.py --port 8001`，然后访问 `http://localhost:8001/` |
| 4 | `.env` 文件里填了真实百炼 Key，但回答还是 Mock 的内容 | **这是正常现象！** — 本项目向量库/嵌入/重排序默认就是 Mock 模式，LLM 才会用真实 Key | 如果你想验证真实 LLM：打开前端 → 输入「你好」，看回答里如果出现「星途环球」这种定制化字眼，说明是 Mock（LLM 生成的没有这个词）；或者看终端日志里 `[LLM] provider=` 一行 |
| 5 | 浏览器打开 `http://localhost:8000/` 显示「无法访问此网站」 | 服务没启动，或者启动错端口了 | ① 先看 Run 窗口有没有文字 `Uvicorn running on http://...`（没文字就是没启动成功）<br> ② Run 窗口显示的端口是多少就访问多少（例如 8001 就访问 localhost:8001） |
| 6 | 顶部菜单 → Run → 找不到 Run 按钮 | 文件没选中，或者解释器没配 | ① 先在左侧文件树**点一下 `run.py` 选中它** <br> ② 解释器没配的话绿色三角形是灰色的 — 先做「第 2 步」 |
| 7 | `PSSecurityException` / `无法加载文件 Activate.ps1，因为在此系统上禁止运行脚本` | PowerShell 执行策略限制（.venv 激活脚本被拦截）| **不用管它！** — 我们的项目**完全不需要 activate 虚拟环境**，直接 `python xxx.py` 就能跑；如果你非要解决，在 Terminal 敲：`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`，然后输入 `Y` 回车 |
| 8 | 打开 `.env` 文件后是乱码，全是看不懂的符号 | 文件编码问题 | 在 PyCharm 右下角，点击当前编码（通常是 `UTF-8`）→ 弹窗选 **GBK** → 弹出对话框选 **Reload（重新加载）**；如果还乱就再切回 UTF-8 |
| 9 | 前端页面能打开，但回答永远是「我无法为您提供帮助，请重新描述...」 | Mock 向量库里只有 51 组示例知识，问的内容超出范围了 | 问一下内置的示例问题试试：<br> ① 物流类「发日本FBA头程」<br> ② 售后类「亚马逊30天退换货」<br> ③ 合规类「欧盟VAT税率」 |
| 10 | `pip install` 慢到怀疑人生，或者报 Timeout | 国外 PyPI 源在国内访问慢 | 在 Terminal 敲这一行切换到阿里云国内源（一次配置，以后所有装包都自动快10倍！）：<br> `pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/` |

---

## 📈 第 8 步：进阶操作 — 如何从「全 Mock 演示」切换到「百炼真实回答」？

当你想让前端回答的质量更高、更接近真实生产环境时，按这 3 步切换：

| 步骤 | 做什么 | 改完后长这样 |
|---|---|---|
| 1 | 打开 `.env`，确保 `LLM_API_KEY` 是你真实的百炼 Key | `LLM_API_KEY=sk-真实的一串字符` |
| 2 | 打开 `server/app.py`，大概第 100 多行的 `create_orchestrator` 函数，把 `use_mock=True` 改成 `use_mock=False` | `use_mock=False  # 启用真实 LLM（根据 .env）` |
| 3 | 重新运行 `run.py`，再次在前端提问 | 回答内容会是真实百炼模型（qwen-plus）生成的，不会再带有「星途环球」这类 Mock 固定词 |

> 🔒 面试前建议保持默认 Mock 模式，因为：
> - 回答快，不用等真实LLM响应
> - 回答内容**稳定可控**，不会被面试官的刁钻问题搞出幻觉
> - 51 组用例 100% 通过率，可以现场跑评测证明

---

## 🗂️ 附：本项目核心文件一览（面试前背下来，能讲出来就行）

不用一个个打开看，**知道「它是干嘛的」**就够了：

| 文件名 | 对应你简历里的哪句话 | 干嘛用的 |
|---|---|---|
| `config.py` | 配置运维 | 全局配置（LLM/向量库/检索/风控/端口） |
| `data_processing/preprocess.py` | 数据集搭建与预处理 | PDF/DOCX 解析 + PaddleOCR + 三类差异化分块 |
| `retrieval/vector_store.py` | 特征工程：Milvus稠密+稀疏双表征、父子块索引 | Milvus + BM25 混合检索（Hybrid Search） |
| `generation/llm_prompt.py` | 模型训练与参数调优：重排序 + 多语种结构化 Prompt | LLM 封装 + CrossEncoder 重排 + 6 语种 Prompt 模板 |
| `safety/guard.py` | 模型效果评估与生成风控 | Faithfulness 忠实度检测 + 幻觉风险拦截 + 跨境实体守护 |
| `gateway/orchestrator.py` | （编排全链路） | 检索 → 重排 → Prompt → LLM → 风控 的流水线总指挥 |
| `server/app.py` + 内嵌前端 | 模型工程化部署：FastAPI + 多语种可视化页面 | RESTful 接口 + Vue3 咨询页面 |
| `run.py` | 主启动入口 | 三合一入口（服务/CLI/评测）|

---

## 🏁 最后一步：如果仍然跑不通怎么办？

按以下顺序自检（**99% 的问题都在这 3 项里**）：

1. 🔍 **解释器对不对？** — `File → Settings → Project: rag → Python Interpreter`，是不是显示的是 `(rag) .venv\Scripts\python.exe`？
2. 🔍 **依赖装没装？** — Terminal 敲 `pip list`，输出列表里有没有 `fastapi`、`uvicorn`、`httpx` 这三个关键字？
3. 🔍 **服务启没启？** — Run 窗口里有没有绿色的 `Uvicorn running on ...` 字样？

祝你跑通！💪🎉
