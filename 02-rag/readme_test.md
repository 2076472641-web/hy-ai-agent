# 星途环球·跨境售后RAG智能问答系统 - 自动化测试模块使用说明

> 所在目录：`D:\my_work\smart_voyage\tests\`

---

## 一、模块文件清单

本目录新增4个核心文件，**不要移出 `tests/` 目录外：

| 文件 | 作用 |
|------|------|
| `test_cases.json | 内置51组跨境售后标准测试用例（3大知识库场景，各17组） |
| `test_evaluate.py` | 评测指标计算 + 幻觉检测工具函数库 |
| `test_rag_qa.py` | **主自动化测试执行脚本**（入口） |
| `readme_test.md` | 本说明文档 |

**配套根目录额外新增：
- `test_run.bat`（项目根目录下）`：Windows 双击一键启动测试

输出目录：`tests/test_output/`（自动生成，存放每次运行的明细JSON+汇总TXT）

---

## 二、51组测试用例覆盖范围（面试直接背这张表）

```
三大知识库场景 × 17组/场景 = 51组
├─ 📦 跨境物流咨询（LOG-001 ~ LOG-017）
│    · 物流方式/时效/报价对比
│    · 清关异常、丢件、签收异常
│    · FBA头程、海外仓、体积重
│    · 特殊地区（巴西/乌克兰/沙特/澳大利亚）
│    · 危险品电池、英国脱欧、俄乌战事物流
│
├─ 🛒 海外售后政策（ASP-001 ~ ASP-017）
│    · 亚马逊/eBay/Wish/TikTok Shop 各平台售后
│    · 30天退换、超期质量、PayPal/Stripe拒付
│    · 差评急救SOP、召回、过保维修、定制类不退
│    · 海外仓本地退换费用、欧盟14天冷静期
│
└─ 📜 各国海关合规政策（CUS-001 ~ CUS-017）
     · 美/欧/日/韩/加/墨/俄/阿联酋 进口规则
     · 欧盟VAT IOSS/OSS、RCEP关税优惠
     · HS编码风险、美国301关税、低报风险
     · FDA/NOM/EAC/AEO/FDA合规认证
     · OFAC制裁合规（SDN名单风控）
```

---

## 三、3种运行模式（面试演示灵活切换）

### ▶️ 方式1：双击运行（最简单，推荐面试用这个）

直接**双击项目根目录 `test_run.bat`**，选择对应模式即可。

---

### ▶️ 方式2：命令行运行（灵活）

打开 CMD / PowerShell，**进入项目根目录**后运行：

#### ① Mock模式（默认，离线可跑，零Token，51组全跑）
```bat
cd /d D:\my_work\smart_voyage
venv\Scripts\activate.bat
python -m tests.test_rag_qa
```

#### ② Local模式（复用项目现有Orchestrator编排能力，真实逻辑）
```bat
python -m tests.test_rag_qa --mode local
```

#### ③ API模式（调已经启动的FastAPI网关RAG接口）
```bat
:: 先启动网关（另开一个CMD窗口）
python -m gateway.gateway

:: 回到测试窗口运行（默认8000端口）
python -m tests.test_rag_qa --mode api
```

---

### 常用命令组合速查表

| 目标 | 命令 |
|------|------|
| 只跑【跨境物流】17组 | `python -m tests.test_rag_qa --category cross_border_logistics` |
| 只跑【海外售后政策】17组 | `python -m tests.test_rag_qa --category after_sales_policy` |
| 只跑【海关合规】17组 | `python -m tests.test_rag_qa --category customs_compliance` |
| 快速调试跑前5条 | `python -m tests.test_rag_qa --limit 5` |
| **面试演示幻觉检测效果 | `python -m tests.test_rag_qa --mock-noise 0.8 --limit 5` |
| 评测工具自测 | `python -m tests.test_evaluate` |

---

## 四、6项评测指标说明（面试官问"怎么判断RAG系统好不好？直接念这6条）

| 指标（权重） | 是干什么的？ | 怎么算出来的？|
|---|---|---|
| ① **关键词召回率**（25%） | RAG多路召回有没有命中了多少个要求的要点 | 回答里命中 `expected_keywords` 的比例，0~100% |
| ② **语义相似度**（20%） | 回答和参考答案是不是一个意思 | n-gram Jaccard + Overlap + 余弦相似度综合分 |
| ③ **回答忠实度**（25%） | **防幻觉第一道防线：回答有没有依据支持 | 分句检查每句话n-gram覆盖参考资料+命中引用短语的比例，扣幻觉正则 |
| ④ **实体准确率**（20%） | 关键槽位（国家/重量/金额/税号）对不对 | `expected_entities` 字典逐槽比对 |
| ⑤ **幻觉风险**（倒扣10%） | 综合判定 HIGH/MEDIUM/LOW/NONE 四级 | 关键信息<30%+实体错位+绝对化断言+跑题+参考差异大 |
| ⑥ **接口响应耗时** | P50/P95/MAX 毫秒级统计 | `perf_counter` 精确打点，统计报告自动算分位数 |

综合分 = ①0.25 + ②0.20 + ③0.25 + ④0.20 + (1-幻觉分)*0.10
≥ 0.6 且 非 HIGH 幻觉 → Case 通过（PASS）

---

## 五、输出产物查看

每次运行会在 `tests/test_output/` 下生成**2份文件（时间戳区分）：

### 1. 汇总报告 `rag_test_summary_YYYYMMDD_HHMMSS.txt`
控制台也会同步打印，内容：
```
· 总用例 / 通过 / 失败 / 通过率
· 4大核心指标均值（带进度条）
· 幻觉分布 HIGH/MEDIUM/LOW/NONE
· P50/P95 响应耗时
· 3大场景分类通过率
· 🚫 失败用例清单（含原因）
```

### 2. 明细数据 `rag_test_detail_YYYYMMDD_HHMMSS.json`
含每条用例：
- 原始问题 / LLM输出 / 参考答案 / 多路召回Top3
- 6项评测完整结果（含每一条幻觉原因、缺失关键词明细）
- 接口耗时 / 报错栈（如有）

---

## 六、面试现场演示路线推荐（5分钟讲清楚）

```
第1分钟：开场双击 test_run.bat → 选1（Mock全量51组），然后打开输出目录
第2分钟：展示控制台进度条一条条过（51条 ✅ PASS
第3分钟：展示汇总报告：
         通过率100%、4项指标全>80%、幻觉 HIGH=0
第4分钟：执行演示评测工具抗幻觉能力：
         python -m tests.test_rag_qa --mock-noise 0.8 --limit 5
         展示5条 ❌，逐条解释幻觉原因
第5分钟：打开 test_evaluate.py 讲设计思路：
         正则特征+实体对齐+忠实度分句+加权综合
```

---

## 七、依赖安装（首次用一次就行）

项目现有 venv 已含 pytest/httpx/FastAPI，直接用就行，**不用装新库**。如果缺：
```bat
cd /d D:\my_work\smart_voyage
venv\Scripts\activate.bat
pip install -r requirements.txt
```

---

## 八、环境变量复用说明

测试脚本通过项目根目录 `config.py` → 自动 `load_dotenv()` 加载 `.env` 中的：
- DASHSCOPE_API_KEY （LLM API Key，local/API模式用
- GATEWAY_PORT （网关端口，默认8000，API模式用
- 其他 MySQL/MCP配置（本模块**不新增**任何环境变量要求

完全复用现有项目 `.env` 规范，零新增配置。

---

## 九、常见问题 FAQ

**Q1：我改了RAG接口，要重新测51组，怎么跑最方便？**
A：直接双击根目录 test_run.bat，选3（API模式）即可，或者命令行 `python -m tests.test_rag_qa --mode api`

**Q2：test_output 目录文件太多，怎么清？**
A：直接删整个 test_output 文件夹即可，下次运行会自动重建。

**Q3：想扩展更多Case怎么加？**
A：打开 `test_cases.json`，按现有格式往 `test_cases` 数组里加对象即可，必填字段：id/category/question/expected_keywords/reference_answer，其他选填。

**Q4：跑Local模式报 orchestrator 连接远端智能体连接不上？**
A：没关系，Orchestrator自动走 _local_fallback，51组照样能全跑通（这就是项目本地回退的设计）。

---
> 祝面试顺利！
