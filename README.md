# AI 文档脱敏工具

一个本地跑的小工具：把商业 PRD / 产品文档拖进来，自动调用 MiniMax API，剔除公司名、人名、金额等敏感信息，输出一份纯技术需求清单。

## 功能

- 拖文件上传，自动调 LLM 脱敏
- 强制模型逐字段提取接口定义，禁止编造性能数字
- 结果预览 + 一键下载 .md
- 本地启动，API key 不离开本机

## 快速开始

1. 安装 Python 3.8+
2. 装依赖：`pip install -r requirements.txt`
3. 在 `api_key.txt` 填入你的 MiniMax API key（sk- 开头）
4. 双击 `启动.bat`（Windows），或手动 `python app.py`
5. 浏览器自动打开 http://127.0.0.1:5099

## 技术栈

- 后端：Flask
- 前端：原生 HTML/JS（无框架）
- LLM：MiniMax-Text-01
