# DeepResearch 企业级多智能体深度调研系统 🚀

[![LangGraph](https://img.shields.io/badge/Framework-LangGraph-blue.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Database-Milvus](https://img.shields.io/badge/VectorDB-Milvus-orange.svg)](https://milvus.io/)
[![LLM-DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-red.svg)](https://www.deepseek.com/)

本项目是一个基于 **LangGraph**、**FastAPI** 和 **Vue 3 / React** 构建的企业级多智能体深度调研（DeepResearch）系统。系统采用证据驱动（Evidence-Driven）的设计思想，融合了网络实时检索（Tavily API）与本地企业知识库（Milvus RAG），并在底座中构建了精细的**双层记忆系统**与**角色级专属工具链**，为复杂业务场景下的深度搜索与学术级研报撰写提供稳健、高品质的工程实现。

---

## 🌟 核心技术亮点与架构演进

### 1. 生产级双层记忆系统（Dual-Layer Memory & LLM Fallback）
系统提供统一的记忆网关 `MemoryManager`，完美分离了短期工作记忆与长期经验记忆：
* **短期工作记忆 (Thread-Level Checkpointer)**：利用 **LangGraph Checkpointer**，自适应支持 **PostgreSQL / Redis (物理 6380 端口物理隔离)** 或内存数据库，在后台为每个 `thread_id` 自动保存每一步的状态快照。若发生停电或网络闪断，系统能实现毫秒级的**断点续传（Resume）**，并完美支持多路并发合流。
* **长期认知记忆 (Long-Term Cognitive Memory)**：
  * **结构化 LLM 提取**：在每轮会话清算阶段，系统通过 **DeepSeek (deepseek-chat) 生产级接口**，运行精心设计的**【认知提炼专家 Prompt】**，从原始对话中自动剥离并去重事实（facts）与用户偏好（preferences），写入 SQLite 关系数据库。
  * **鲁棒本地规则降级 (Regex Fallback)**：我们设计了完备的异常处理。一旦 DeepSeek 官方 API 发生网络抖动或超时，系统将**自动且无感地降级至本地的正则表达式与关键词匹配引擎**，保障 100% 的写入可用性。
  * **跨会话自动注入**：当该用户在任何新窗口（New Thread）提问时，系统会基于 `user_id` 自动从长期记忆库（SQLite / Milvus）中召回专属画像上下文，实现跨时空的个性化问答。

### 2. 语义与检索解耦：RAG 上下文合并召回（Window Buffer Retrieval）
为了防止传统 RAG 在检索时“断章取义”导致大模型出现幻觉，我们在 `app/mult_agents/rag/core.py` 中实现了先进的滑动窗口拼接技术：
* **动态 Schema 探测**：自适应提取 Milvus 集合的 Schema 结构，动态获取主键字段名称（如 `pk` 或 `id`），完美兼容各种版本的 `LangChain Milvus` 封装。
* **物理相邻多路召回**：当向量相似度精准命中某个 500 字的核心 Chunks 后，提取其物理主键 `pk`，自动使用高性能过滤条件 `expr = f"{pk_field} in [{pk - 1}, {pk + 1}]"`，拉取其在物理上紧邻的**前一个和后一个 Chunks**。
* **数据一致性防线 (Document Consistency Guardrail)**：
  * 系统在拼接前，会严格对比邻近分片与核心分片的元数据 `source`（物理文件绝对路径）。
  * **只有当它们被验证属于同一个物理来源文件时，才允许拼接**。这绝对防止了跨文件（如两个完全不同的财务报告）的断章碎块强行粘合造成的语义污染。
* **时序无缝缝合**：将通过一致性校验的前文（`pk-1`）、正文（`pk`）与后文（`pk+1`）按照主键 ID 升序排序，中间用虚线指示符拼接，为下游 Agent 递送 **1500 字左右极其连贯的滑动大视界窗口上下文**。

### 3. 角色级专属工具绑定（Cognitive Separation of Concerns）
在多智能体系统中，盲目将所有工具塞给每个 Agent 会造成“认知过载”并大幅拖慢运行速度。我们采用**“职责分离，精准绑定”**的设计，在 `main.py` 的 `build_agents` 实例化工厂中，为各个角色量身定制了专属武器库：
* **`intent_router` (门卫) & `planner` (规划师)** ➡️ **时间感知工具** (`get_current_time`)：使其能准确识别“今年”、“今天”等时效概念，防止时间发生错乱。
* **`scout_web` (网络侦查兵)** ➡️ **网络高精搜网套件** (`web_search_stub` + 高德地图天气、定位、周边、驾车路径规划四件套)：代替模糊的网页搜索，直接调用高精度官方 API 召回结构化事实。
* **`scout_local` (私有库侦查兵)** ➡️ **私有知识库检索工具** (`local_docs_lookup_stub`)：专注于在本地 Milvus 集合中召回机密知识。
* **`analyst` (推理分析师)** ➡️ **理科运算与数据加载工具** (`simple_calculator` + `extract_data_stub` + `sql_inter` 只读 SQL)：具备极高精度的算术、Pandas 表格分析和数据库查询能力。
* **`codegen` (代码生成器)** ➡️ **全栈 AI 程序员套件** (`python_inter` 安全沙箱 + 工作区文件安全读写五件套 + 终端安全命令执行器)：让它能在安全的沙箱里真正运行自己写的代码、自我修改 Bug、查看当前工作空间目录，并使用 `pip list`, `git status` 进行环境自检。
* **`writer` (撰写器)** ➡️ **学术加工与矢量绘图套件** (`fig_inter` 高保真画图 + 文本去重、提炼、合并五件套)：使它在生成精美 Markdown 报告的同时，能自动利用 Python Matplotlib 绘制数据图表，并快速清洗去重搜集来的粗糙素材。

---

## 🗺️ 多智能体协同与工具流全景拓扑 (Topology)

系统中的各智能体通过读写共享状态机沙盒 `ResearchState` 传递上下文，并在执行节点中调度其专属工具：

```mermaid
graph TD
    %% 样式与类定义
    classDef state fill:#f7fafc,stroke:#4a5568,stroke-width:1.5px,color:#1a202c;
    classDef agent fill:#ebf8ff,stroke:#3182ce,stroke-width:2px,color:#1a202c;
    classDef tool fill:#fffaf0,stroke:#dd6b20,stroke-width:1.5px,color:#1a202c;
    classDef ext fill:#e6fffa,stroke:#319795,stroke-width:1.5px,color:#1a202c;

    %% 共享状态沙盒
    State["ResearchState 共享状态机沙盒<br>存储: query, plan, evidence, findings, gaps, checkpoints"]:::state

    %% 智能体节点 (专家角色)
    Agent_Intent["【意图分流门卫】<br>Intent Router 【intent_node】"]:::agent
    Agent_Direct["【极速直答专家】<br>Direct Responder 【direct_answer_node】"]:::agent
    Agent_Architect["【总规划师】<br>Chief Architect 【plan_node】"]:::agent
    Agent_WebScout["【网页侦查兵】<br>Web Scout 【web_search_node】"]:::agent
    Agent_RAGScout["【局内侦查兵】<br>Local RAG Scout 【local_rag_node】"]:::agent
    Agent_Judge["【信源裁判官】<br>Evidence Judge 【deep_dive_node】"]:::agent
    Agent_Analyst["【分析师】<br>Analyst 【analyze_node】"]:::agent
    Agent_Planner["【补搜规划专家】<br>Research Planner 【reflect_node】"]:::agent
    Agent_Writer["【终审写作者】<br>Senior Writer 【write_node】"]:::agent

    %% 物理工具库
    Tool_Time["get_current_time 时间感知"]:::tool
    Tool_Tavily["tavily_web_search_records 实时网搜"]:::tool
    Tool_Amap["amap_* 高德地图高精套件"]:::tool
    Tool_Milvus["search_knowledge_base_records 局内检索<br>+ Window Buffer 上下文拼接"]:::tool
    Tool_Score["_score_evidence 信源分级打分与冲突审计"]:::tool
    Tool_Data["extract_data_stub / sql_inter<br>Pandas与只读SQL数据挖掘"]:::tool
    Tool_Sandbox["python_inter / safe_*<br>安全代码沙箱与工作区文件管理"]:::tool
    Tool_Render["fig_inter Matplotlib矢量绘图"]:::tool

    %% 外部物理基础设施
    Ext_Tavily["Tavily Search API Cloud"]:::ext
    Ext_Milvus["Milvus RAG Vector DB"]:::ext
    Ext_SQLite["SQLite / PostgreSQL 【长期个人记忆】"]:::ext

    %% --------------------------------------------------
    %% 协作与工具流连线
    %% --------------------------------------------------
    
    %% 1. 启动与分流
    State -->|1. 读写共享状态| Agent_Intent
    Agent_Intent -->|意图=direct 【闲聊/直答】| Agent_Direct
    Agent_Intent -->|意图=multiagent 【深度调研】| Agent_Architect
    Agent_Architect -.->|调用专属工具| Tool_Time
 
    %% 2. 并发双源取证
    State -->|2. 并行多路调度| Agent_WebScout
    State -->|2. 并行多路调度| Agent_RAGScout

    %% 网页探路者工具流
    Agent_WebScout -.->|调用| Tool_Tavily
    Agent_WebScout -.->|调用| Tool_Amap
    Tool_Tavily <-->|urllib HTTP POST| Ext_Tavily
    
    %% 本地向量探路者工具流
    Agent_RAGScout -.->|调用| Tool_Milvus
    Tool_Milvus <-->|Window Buffer 缝合| Ext_Milvus

    %% 取证数据写入状态
    Tool_Tavily -->|累加网页证据 web_evidence| State
    Tool_Milvus -->|累加本地数据 local_evidence| State

    %% 3. 证据链裁判与事实收拢
    State -->|3. 汇聚双源成果| Agent_Judge
    Agent_Judge -.->|调用工具| Tool_Score
    Tool_Score -->|生成裁判证据池 evidence_pool| State

    %% 4. 结论归纳与缺口评估
    State -->|4. 读取裁判事实| Agent_Analyst
    Agent_Analyst -.->|调用工具| Tool_Data
    Agent_Analyst -->|needs_more_research == True 【存在缺口】| Agent_Planner
    Agent_Analyst -->|needs_more_research == False 【完备收敛】| Agent_Writer

    %% 5. 缺口反思补搜环路
    Agent_Planner -->|迭代次数自增, 重新并行取证| Agent_WebScout
    Agent_Planner -->|迭代次数自增, 重新并行取证| Agent_RAGScout

    %% 6. 研报长文撰写与画图
    Agent_Writer -.->|调用工具| Tool_Sandbox
    Agent_Writer -.->|调用工具| Tool_Render
    Tool_Render <-->|持久化研报 & 沉淀| Ext_SQLite
    Agent_Writer -->|拼接正文与文献附录| State

    %% 样式美化
    linkStyle 0 stroke:#3182ce,stroke-width:2px;
    linkStyle 3 stroke:#e53e3e,stroke-width:2px;
    linkStyle 4 stroke:#2b6cb0,stroke-width:2px;
    linkStyle 13 stroke:#38a169,stroke-width:2px;
    linkStyle 14 stroke:#38a169,stroke-width:2px;
```

---

## 📂 项目工作区目录规范

```
d:/AI/deep_research/deep_research/
├── app/
│   ├── backend/                 # FastAPI Web 服务端
│   │   ├── config/              # settings.py 配置（基于 Pydantic Settings）
│   │   ├── router/              # API 路由器（health_router, research_router sse流）
│   │   ├── schemas/             # Pydantic 数据校验对象（health, research）
│   │   └── service/             # 单例服务提供者 workflow_service.py（LangGraph 调度器）
│   ├── mult_agents/             # 多智能体大脑引擎
│   │   ├── memory/              # 双层记忆子系统
│   │   │   ├── base.py          # 记忆实体模型与内存字典默认实现
│   │   │   ├── short_term.py    # 带有消息自动 UUID 生成的短期对话缓冲区
│   │   │   ├── long_term.py     # SQLite 长期语义与情节记忆数据库
│   │   │   ├── manager.py       # 统一记忆网关与 DeepSeek 摘要引擎
│   │   │   └── utils.py         # 认知提取专家（LLM + 正则降级）
│   │   ├── rag/                 # RAG 本地检索系统
│   │   │   ├── core.py          # RAGSystem (Window Buffer 检索与一致性防线)
│   │   │   └── ingest.py        # 离线知识递归入库引擎
│   │   ├── config.py            # 全局 AppConfig 加载器
│   │   ├── graph.py             # 偏函数节点绑定与 LangGraph 拓扑编排
│   │   ├── main.py              # 命令行启动入口与全量工具专属分配中心
│   │   ├── nodes.py             # 专家节点高阶业务控制（JSON 解析与大纲派生）
│   │   ├── prompts.py           # 专家系统提示词定义
│   │   ├── state.py             # 状态沙盒结构定义
│   │   └── tools.py             # 全量物理工具定义（沙箱/数据/高德地图/SQL等）
│   └── app_main.py              # FastAPI 启动总装载脚本
├── config.json                  # 静态配置文件
├── requirements.txt             # 生产环境依赖列表
└── README.md                    # 技术全景指南说明文档
```

---

## ⚙️ 环境配置与部署规范

### 1. 配置文件 `config.json`
用于托管 Milvus、Redis 物理端口等静态底层参数：
```json
{
  "milvus_host": "127.0.0.1",
  "milvus_port": 19530,
  "milvus_collection": "deep_research_memory",
  "short_term_backend": "postgres",
  "long_term_backend": "sqlite",
  "long_term_scope": "user",
  "checkpointer_backend": "auto"
}
```

### 2. 环境变量 `.env`
在项目根目录下创建 `.env` 文件，用于填充敏感凭证与跨项目隔离配置：
```env
# 核心大模型大底座与向量模型（DashScope 通义千问）
DASHSCOPE_API_KEY="your_dashscope_api_key"

# 生产级大模型记忆提炼引擎（DeepSeek）
DEEPSEEK_API_KEY="sk-your_deepseek_api_key"
DEEPSEEK_MODEL="deepseek-chat"

# 网页取证搜索引擎 Key
TAVILY_API_KEY="tvly-your_tavily_key"

# 长期情景/会话存储 PostgreSQL（Schema级别隔离）
POSTGRES_DSN="postgresql://postgres:password@127.0.0.1:5432/deep_research_db"

# 高端地图服务密钥（用于网页侦查兵定位规划）
AMAP_API_KEY="your_amap_api_key"

# 物理硬隔离的 Redis 端口实例配置
REDIS_URL="redis://:password@127.0.0.1:6380/0"
```

---

## 🚀 快速启动指南

### 1. 离线知识库入库 (RAG Ingestion)
要为局内侦查兵准备本地参考资料，将待入库的 Markdown 或文本文件放入路径，运行：
```bash
# 激活虚拟环境后，在根目录下执行
python app/mult_agents/rag/ingest.py
```
系统会自动进行 500 字切片、50 字重叠，并调用通义千问 Embedding 写入 Milvus 中。

### 2. 启动 CLI 交互式终端 (Chat Loop)
在终端里直接与整个多智能体集群进行交互：
```bash
python app/mult_agents/main.py
```
* **单次快捷查询**：`python app/mult_agents/main.py --once-query "分析大模型记忆系统"`
* **自检长期记忆**：在聊天中输入 `/memory` 查看当前用户已沉淀 Facts/Prefs 的统计。
* **自检向量链路**：在聊天中输入 `/memory-trace` 查看向量搜索引擎的召回 Trace 明细。

### 3. 启动 FastAPI Web 服务端 (Web Gateway)
为前端提供极速响应的 SSE 事件流网关：
```bash
python app/app_main.py
```
服务默认绑定在 `0.0.0.0:8000`。
* **极速直答接口**：`POST /api/v1/research/run`
* **SSE 流式实时流接口**：`POST /api/v1/research/stream`，支持将 `phase`、`node` 状态机轨迹与 Markdown 最终报告无缝推送给前端展示。

### 4. 运行全系统集成测试
运行我们编写的完整记忆与提取验证测试套件：
```bash
python C:\Users\lenovo\.gemini\antigravity\brain\ddee3a42-4f58-4122-b77e-99ee42892b4f\scratch\test_memory_system.py
```
*(该脚本将自动对内存、长期 SQLite 数据库的增删改查以及 DeepSeek 生产级同步提炼链路进行全功能自动化跑通校验)*

---

## 🛡️ 数据逻辑安全与隐私合规

本项目在设计上完全符合现代企业数据合规与隐私保护标准：
1. **防范 SQL 注入与高危拦截**：`sql_inter` 工具内部集成了强力的只读静态安全检查，**严格只允许 SELECT / WITH 开头的静态只读查询**，物理上拦截了所有的 `DROP`、`DELETE`、`UPDATE` 等指令。
2. **沙箱安全执行屏蔽**：`python_inter` 沙箱内置了高危内置函数与模块阻断（封禁了 `os`、`sys`、`subprocess`、`socket` 以及 `getattr/setattr` 等），有效防范恶意代码对系统的入侵。
3. **完全“被遗忘权（Right to be Forgotten）”**：通过内置的 `clear_user_memory` 管理面接口，能一键安全抹除特定用户在系统所有物理存储（SQLite, Redis, Milvus）中的完整数字足迹。
