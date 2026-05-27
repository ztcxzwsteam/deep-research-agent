"""节点执行模块：实现意图识别、检索、证据裁判、分析与写作等节点逻辑。"""

import json
import logging
import os
import re
from functools import partial

from langchain_core.messages import HumanMessage

from .state import ResearchState
from .tools import tavily_web_search_records, search_knowledge_base_records

logger = logging.getLogger("mult_agents")

ANSI = {
    "reset": "\033[0m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "red": "\033[31m",
}


# ==============================================================================
# 模块一：基础辅助与大模型容错函数
# ==============================================================================

def colorize(text: str, color: str) -> str:
    """
    终端输出带色文字工具函数。
    
    参数:
        text (str): 待着色的原始文本。
        color (str): 目标颜色名称（cyan, magenta, yellow, green, red）。
        
    返回:
        str: 带有 ANSI 颜色控制字符的格式化文本。
    """
    if os.getenv("NO_COLOR"):
        return text
    code = ANSI.get(color, "")
    if not code:
        return text
    return f"{code}{text}{ANSI['reset']}"


def emit(node: str, content: str):
    """
    流式进度控制台输出器。
    
    截取大模型或节点的前 400 个字符进行单行打印，用于在多智能体异步后台运行时，
    向开发人员或系统日志提供实时的进度指示。
    
    参数:
        node (str): 正在运行的智能体节点名称。
        content (str): 待输出的明文内容。
    """
    preview = content.replace("\n", " ")
    if len(preview) > 400:
        preview = preview[:400] + "..."
    logger.info("%s 输出: %s", colorize(f"[{node}]", "yellow"), preview)


def collect_tool_calls(messages) -> tuple[list, list]:
    """
    工具调用轨迹搜集器。
    
    遍历历史消息，自动提取大模型调用过的所有 Tool 名称以及对应的工具输出，
    用作检索明细统计和系统链路轨迹的沉淀分析。
    
    参数:
        messages (list): LangChain/LangGraph 的消息历史。
        
    返回:
        tuple[list, list]: (调用过的工具名称列表, 对应的工具输出信息列表)。
    """
    tools = []
    tool_outputs = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else None
                if name:
                    tools.append(name)
        name = getattr(msg, "name", None)
        msg_type = getattr(msg, "type", None)
        if msg_type == "tool" and name:
            tools.append(name)
            output = getattr(msg, "content", "")
            if output:
                tool_outputs.append(f"{name}: {output}")
    return tools, tool_outputs


def with_memory_context(state: ResearchState, user_prompt: str) -> str:
    """
    跨会话记忆动态注入处理器。
    
    检测状态沙盒中是否存在已被 MemoryManager 装载的长期语义记忆，
    如果存在，以独立的规范结构拼接在当前用户提示词头部。
    
    参数:
        state (ResearchState): 共享上下文状态。
        user_prompt (str): 当前节点的原始指令或提示词。
        
    返回:
        str: 拼接了记忆上下文后的鲁棒提示词。
    """
    memory_context = state.get("memory_context", "").strip()
    if not memory_context:
        return user_prompt
    return f"{user_prompt}\n\n[跨会话记忆]\n{memory_context}"


def log_inputs(node: str, agent_name: str, payload: dict):
    """
    节点输入标准追踪日志记录器。
    
    将节点入参（截取主要字符串）格式化记录进系统 INFO 日志，实现透明、可追溯的链路排查。
    
    参数:
        node (str): 图节点名称。
        agent_name (str): 内部绑定执行的智能体代号。
        payload (dict): 输入参数字典。
    """
    preview = {
        key: (value[:200] + "..." if isinstance(value, str) and len(value) > 200 else value)
        for key, value in payload.items()
    }
    logger.info("%s 输入 | agent=%s | data=%s", colorize(f"[{node}]", "cyan"), colorize(agent_name, "magenta"), preview)


def detect_intent(query: str) -> str:
    """
    规则判定引擎。
    
    通过敏感关键词和年份正则，极速判断用户 Query 是需要进入深度多轮研究，
    还是仅仅是日常打招呼、自我介绍等简单问答。
    
    参数:
        query (str): 用户输入的原始问题。
        
    返回:
        str: 意图初判路由方向 ('direct' 或 'multiagent')。
    """
    normalized_query = query.strip()
    force_multiagent_keywords = [
        "调查",
        "调研",
        "来源",
        "证据",
        "检索统计",
        "来源清单",
        "重大新闻",
        "热门项目",
        "趋势",
        "新闻",
        "最新",
        "盘点",
    ]
    if re.search(r"20\d{2}年", normalized_query) and any(word in normalized_query for word in ["趋势", "新闻", "调研", "调查", "盘点"]):
        return "multiagent"
    if any(word in query for word in force_multiagent_keywords):
        return "multiagent"
    keywords = [
        "调研",
        "研究",
        "调查",
        "盘点",
        "热门",
        "趋势",
        "榜单",
        "分析",
        "方案",
        "架构",
        "设计",
        "对比",
        "报告",
        "代码",
        "实现",
        "落地",
        "检索",
        "知识库",
        "证据",
        "来源",
        "溯源",
        "资料",
        "手册",
        "验证",
        "数据",
        "模型",
    ]
    return "multiagent" if any(word in query for word in keywords) else "direct"


def bind_agent(node_func, agent, agent_name: str):
    """
    偏函数适配器。
    
    用于将底座 Qwen/Tongyi 模型实例以及具体角色代号直接绑定到节点行为上，
    使 LangGraph 能以通用的单入参 `node(state)` 签名执行不同的多智能体。
    
    参数:
        node_func (function): 目标执行节点函数。
        agent (any): 大模型实例。
        agent_name (str): 智能体中文代号。
        
    返回:
        function: 包装绑定后的偏函数。
    """
    return partial(node_func, agent=agent, agent_name=agent_name)


def _last_content(result) -> str:
    """
    消息格式降级提取器。
    
    鲁棒处理大模型调用返回的消息格式，支持处理标准的纯文本字符串
    以及多模态、复杂列表等异构消息。
    
    参数:
        result (any): 大模型返回的原始消息包装。
        
    返回:
        str: 降级还原出的纯文本字符串。
    """
    content = result["messages"][-1].content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
    return str(content)


def _extract_json_block(text: str) -> str:
    """
    JSON 块边界拦截算法。
    
    大模型由于贪婪性极易输出带有 ```json 格式包裹的 Markdown 字符或额外自然语言。
    本算法使用最外层大括号进行贪婪正向/反向雷达扫描匹配，实施完美截取。
    
    参数:
        text (str): 模型返回的夹带杂质的文本。
        
    返回:
        str: 拦截出的干净、可直接 json.loads 的 JSON 字符串。
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _load_json(text: str, fallback: dict) -> dict:
    """
    防崩溃解析器。
    
    调用边界拦截算法对 JSON 进行解析，若大模型发生格式灾难，
    则静默退避返回给定的默认字典（fallback），阻断整个工作流的崩溃。
    
    参数:
        text (str): 待解析的文本。
        fallback (dict): 兜底默认字典。
        
    返回:
        dict: 反序列化出的结果字典。
    """
    try:
        value = json.loads(_extract_json_block(text))
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    return fallback


def _invoke_json_agent(state: ResearchState, prompt: str, agent, agent_name: str, node: str, fallback: dict) -> tuple[dict, str, list]:
    """
    大模型结构化交互总网关。
    
    1. 注入跨轮次记忆；
    2. 【极致优化点】断开之前的 messages 历史，单指令请求，极大控制 Token 消耗并强化指令遵循；
    3. 获取原始回复文本，提取并退避解析结构化 JSON，归档工具轨迹并返回三元组。
    
    参数:
        state (ResearchState): 共享状态。
        prompt (str): 执行指令。
        agent (any): 模型对象。
        agent_name (str): 智能体名称。
        node (str): 节点名称。
        fallback (dict): 退避解析的兜底字典。
        
    返回:
        tuple[dict, str, list]: (结构化字典, 原始文本, 本轮产生的消息对)。
    """
    human = HumanMessage(content=with_memory_context(state, prompt))
    result = agent.invoke({"messages": [human]})
    tools, tool_outputs = collect_tool_calls(result["messages"])
    logger.info("%s 工具: %s", colorize(f"[{node}]", "green"), ", ".join(tools) if tools else "无")
    for item in tool_outputs[:5]:
        logger.info("%s 工具输出: %s", colorize(f"[{node}]", "green"), item[:400])
    logger.info("%s LLM调用: 是 | 思考: 不可见", colorize(f"[{node}]", "yellow"))
    content = _last_content(result)
    emit(node, content)
    return _load_json(content, fallback), content, [human, result["messages"][-1]]


# ==============================================================================
# 模块二：规划与检索计划派生函数
# ==============================================================================

def _default_plan(state: ResearchState) -> dict:
    """
    Planner 规划兜底函数。
    
    如果总规划师大模型由于接口超时或解析坏账无法产出规划，
    算法将自动返回本函数定义的包含基础轮次大纲的兜底规划字典，确保流程收敛。
    
    参数:
        state (ResearchState): 当前状态。
        
    返回:
        dict: 兜底的研究规划字典。
    """
    return {
        "objective": state["query"],
        "sub_questions": [state["query"]],
        "outline": [
            {
                "id": "sec_1",
                "title": "默认大纲",
                "description": "默认生成的大纲",
                "section_type": "mixed",
                "requires_data": False,
                "requires_chart": False,
                "priority": 1,
                "search_queries": [state["query"]],
                "status": "pending",
            }
        ],
        "research_questions": [state["query"]],
        "budget": {"max_rounds": 2, "max_sources": 12, "max_tokens": 12000, "max_seconds": 45},
    }


def _guess_primary_entity(query: str) -> str:
    """
    主实体猜测器。
    
    使用中英文匹配正则表达式，猜测用户原始诉求中的核心实体（如“Milvus”或“企业知识库”），
    剔除常用的日常口语辅助字眼，为搜索改写提供强力语义定位锚。
    
    参数:
        query (str): 原始查询。
        
    返回:
        str: 猜测出的核心实体名。
    """
    lowered = query.lower()
    ascii_terms = re.findall(r"[a-z][a-z0-9_-]{2,}", lowered)
    for term in ascii_terms:
        if term not in {"latest", "trend", "news", "agent", "open", "using"}:
            return term
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", query)
    for term in chinese_terms:
        if term not in {"帮我", "调查", "最新", "使用趋势", "是什么", "多少", "情况"}:
            return term
    return ""


def _derive_direct_search_queries(query: str) -> list[str]:
    """
    直接检索词派生算法。
    
    在原始问题的基础上，配合猜测的核心主实体，衍生出 3-5 个高相关、
    有利于搜索引擎召回官方文档与 GitHub 地址的直接关联搜索 Query 候选。
    
    参数:
        query (str): 用户 Query。
        
    返回:
        list[str]: 派生出的自然语言检索词列表。
    """
    base_query = query.strip()
    if not base_query:
        return []
    entity = _guess_primary_entity(base_query)
    candidates = [base_query]
    if entity:
        candidates.extend(
            [
                f"{entity}是什么",
                f"{entity} GitHub",
                f"{entity} 官方文档",
                f"{entity} 使用趋势",
                f"{entity} AI Agent",
            ]
        )
    else:
        candidates.extend(
            [
                f"{base_query} 是什么",
                f"{base_query} GitHub",
                f"{base_query} 官方文档",
            ]
        )
    deduped: list[str] = []
    for item in candidates:
        text = item.strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped[:6]


def _is_query_grounded(candidate: str, user_query: str) -> bool:
    """
    检索词安全校验。
    
    对比派生词与原始 Query 中的核心词表，当派生词与原词在关键概念上有至少一处重合，
    或者包含原主实体时判定为合法派生，防止大模型发生“语义漂移”瞎乱搜。
    
    参数:
        candidate (str): 待校验的派生检索词。
        user_query (str): 用户原始 Query。
        
    返回:
        bool: 是否合法。
    """
    candidate_terms = set(_extract_query_terms(candidate))
    user_terms = set(_extract_query_terms(user_query))
    if not candidate_terms or not user_terms:
        return False
    if _guess_primary_entity(user_query) and _guess_primary_entity(user_query) in candidate.lower():
        return True
    overlap = candidate_terms & user_terms
    return len(overlap) >= 1


def _derive_search_plan(outline: list[dict], sub_questions: list[str], _research_questions: list[str], query: str) -> list[dict]:
    """
    研究检索大图景派生器。
    
    将用户直接派生词与大纲中每个章节分配的检索 Query 进行汇聚，
    去重后构建出一个完整的包含章节归属和检索偏好的检索大图景计划（search_plan）。
    
    参数:
        outline (list): 大纲章节。
        sub_questions (list): 衍生子问题。
        query (str): 原始提问。
        
    返回:
        list[dict]: 结构化的全盘检索计划。
    """
    plan: list[dict] = []
    for direct_query in _derive_direct_search_queries(query):
        plan.append(
            {
                "section_id": "user_query",
                "query": direct_query,
                "source_preference": "hybrid",
                "reason": "围绕用户原始问题生成的直接检索词",
            }
        )
    for section in outline:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("id") or "sec")
        for item in section.get("search_queries", []) or []:
            text = str(item).strip()
            if text and _is_query_grounded(text, query):
                plan.append(
                    {
                        "section_id": section_id,
                        "query": text,
                        "source_preference": "hybrid",
                        "reason": f"来自大纲章节 {section_id}",
                    }
                )
    if not plan:
        plan.append({"section_id": "sec_1", "query": query, "source_preference": "hybrid", "reason": "fallback"})
    deduped = _dedupe_sources(plan, ["query"])
    return deduped[:6]


def _build_queries(state: ResearchState, source_preference: str) -> list[dict]:
    """
    检索词分发调度器。
    
    智能识别图流转所处的阶段。如果是首次执行（iteration=0），从 `search_plan` 分发检索词；
    如果是补搜迭代（iteration>0），则提取 Reflection 派生的 `supplementary_queries` 检索词。
    同时，根据节点的偏好属性（web 或 local）执行针对性过滤。
    
    参数:
        state (ResearchState): 共享状态。
        source_preference (str): 当前节点偏好 ('web' 或 'local')。
        
    返回:
        list[dict]: 本轮分配执行的检索词包。
    """
    queries: list[dict] = []
    iteration = state.get("iteration", 0)
    if iteration > 0 and state.get("supplementary_queries"):
        base_plan = state.get("supplementary_queries", [])
    else:
        base_plan = state.get("search_plan", [])
        
    for item in base_plan:
        if not isinstance(item, dict):
            continue
        pref = item.get("source_preference", "hybrid")
        if pref in (source_preference, "hybrid"):
            query = str(item.get("query", "")).strip()
            if query:
                queries.append(item)
    if not queries:
        queries.append({"section_id": "sec_1", "query": state["query"], "source_preference": source_preference, "reason": "fallback"})
    return queries[:6]


# ==============================================================================
# 模块三：数据取证与清洗过滤函数
# ==============================================================================

def _extract_query_terms(query: str) -> list[str]:
    """
    检索词核心实体分词器。
    
    剥离中文口语化停用词，只提取具备高实意价值的中英文词表（2字以上），
    为后置的粗筛算法计算关键词碰撞打下基础。
    
    参数:
        query (str): 自然语言搜索 Query。
        
    返回:
        list[str]: 提取出的核心概念/词组。
    """
    parts = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_-]{3,}", query.lower())
    terms = []
    stopwords = {"什么", "如何", "以及", "一个", "关于", "这个", "那个", "进行", "基于", "附带", "来源", "清单"}
    for part in parts:
        if part in stopwords:
            continue
        terms.append(part)
    return terms[:12]


def _estimate_relevance(query: str, text: str) -> float:
    """
    文本相关性粗筛算法。
    
    计算检索词核心概念列表在被检索文本 haystack 中的词命中碰撞率，
    以 `0.0 ~ 1.0` 之间的浮点数衡量相关性。
    
    参数:
        query (str): 当前检索词。
        text (str): 网页摘要或 RAG 片段文本。
        
    返回:
        float: 粗略计算出的命中相关得分。
    """
    terms = _extract_query_terms(query)
    if not terms:
        return 0.0
    haystack = text.lower()
    hits = sum(1 for term in terms if term in haystack)
    return hits / max(len(terms), 1)


def _is_bad_web_domain(domain: str) -> bool:
    """
    低置信域名黑名单过滤。
    
    检测域名是否包含说明书、下载、推广广告站（如 datasheet, elecfans），
    若是，从取证源直接丢弃，拦截低品质互联网杂质。
    
    参数:
        domain (str): 二级/顶级域名。
        
    返回:
        bool: 是否属于黑名单低置信站点。
    """
    value = domain.lower()
    blocked = ["datasheet", "bdtic", "doc88", "elecfans", "down"]
    return any(item in value for item in blocked)


def _filter_web_records(query: str, records: list[dict]) -> tuple[list[dict], dict]:
    """
    Tavily 网页召回粗筛处理器。
    
    对 Tavily 返回的数据进行实体碰撞评估。静默物理删除空数据、
    黑名单站点，以及相关性估算得分低于 `0.2` 的边缘不相关数据。
    
    参数:
        query (str): 当前检索词。
        records (list): 原始 Tavily API 召回的数据。
        
    返回:
        tuple[list[dict], dict]: (粗筛保留后的记录列表, 本次过滤明细统计字典)。
    """
    kept = []
    stats = {"raw_count": len(records), "kept_count": 0, "dropped_irrelevant": 0, "dropped_domain": 0, "dropped_empty": 0}
    for record in records:
        title = str(record.get("title", ""))
        snippet = str(record.get("snippet", ""))
        domain = str(record.get("domain", ""))
        if not title and not snippet:
            stats["dropped_empty"] += 1
            continue
        if _is_bad_web_domain(domain):
            stats["dropped_domain"] += 1
            continue
        relevance = _estimate_relevance(query, f"{title}\n{snippet}")
        record["relevance_score"] = relevance
        if relevance < 0.2 and not _is_official_domain(domain):
            stats["dropped_irrelevant"] += 1
            continue
        kept.append(record)
    stats["kept_count"] = len(kept)
    return kept, stats


def _filter_local_records(query: str, records: list[dict]) -> tuple[list[dict], dict]:
    """
    Milvus 召回粗筛处理器。
    
    由于 RAG 检索在向量空间召回时极易出现“强行召回”非语义相关片段，
    本函数根据实体碰撞过滤不相关的 RAG 切片。
    
    参数:
        query (str): 检索词。
        records (list): MilvusVectorStore 召回切片列表。
        
    返回:
        tuple[list[dict], dict]: (粗筛保留的 RAG 列表, 本次过滤明细字典)。
    """
    kept = []
    stats = {"raw_count": len(records), "kept_count": 0, "dropped_irrelevant": 0, "dropped_missing_doc": 0, "dropped_empty": 0}
    for record in records:
        title = str(record.get("title", ""))
        snippet = str(record.get("snippet", ""))
        doc_id = str(record.get("doc_id", "")).strip()
        if not snippet:
            stats["dropped_empty"] += 1
            continue
        relevance = _estimate_relevance(query, f"{title}\n{snippet}")
        record["relevance_score"] = relevance
        if not doc_id and relevance < 0.35:
            stats["dropped_missing_doc"] += 1
            continue
        if relevance < 0.2:
            stats["dropped_irrelevant"] += 1
            continue
        kept.append(record)
    stats["kept_count"] = len(kept)
    return kept, stats


def _format_raw_records(records: list[dict], source_type: str) -> str:
    """
    精炼序列化输出器。
    
    将粗筛保留的证据转化为极其紧凑的 JSON-Lines（每行一个 JSON）传给大模型。
    丢弃一切无用元数据，最大限度压缩 LLM 节点的 Context Token 消耗，防止爆 Token。
    
    参数:
        records (list): 证据列表。
        source_type (str): 'web' 或 'local'。
        
    返回:
        str: 紧凑的 JSON 格式多行字符串。
    """
    if not records:
        return "[]"
    lines = []
    for record in records[:40]:
        locator = record.get("url") or record.get("doc_id") or ""
        lines.append(
            json.dumps(
                {
                    "source_id": record.get("source_id"),
                    "title": record.get("title"),
                    "url": record.get("url", ""),
                    "doc_id": record.get("doc_id", ""),
                    "snippet": str(record.get("snippet", ""))[:500],
                    "source_type": source_type,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def _minimal_record_filter(records: list[dict], required_any: list[str]) -> list[dict]:
    """
    基础非空验证筛查器。
    
    确保召回的记录字典中，必须有定位符（URL / Path）、摘要等关键字段之一，
    防止空指针数据混入。
    
    参数:
        records (list): 记录列表。
        required_any (list): 检查的字段列表。
        
    返回:
        list[dict]: 保留的非空记录。
    """
    kept: list[dict] = []
    for record in records:
        if any(str(record.get(field, "")).strip() for field in required_any):
            kept.append(record)
    return kept


def _assign_source_ids(records: list[dict], prefix: str) -> list[dict]:
    """
    时间线引用 ID 注入器。
    
    在首轮取证和 Reflection 补搜迭代中，为搜集到的海量文献打上完全唯一且具备可追溯性的
    学术标识物理 ID（如网页为 `WEB1_1-1`、本地为 `LOC2_3-2`），作为后续引用定位基础。
    
    参数:
        records (list): 记录列表。
        prefix (str): ID 头部标识（如 WEB1_1）。
        
    返回:
        list[dict]: 标记了 source_id 后的证据列表。
    """
    assigned: list[dict] = []
    for index, record in enumerate(records, 1):
        item = dict(record)
        item["source_id"] = f"{prefix}-{index}"
        assigned.append(item)
    return assigned


def _enrich_evidence_from_raw(evidence: list[dict], raw_records: list[dict]) -> list[dict]:
    """
    【核心健壮性】原始字段智能回补算法。
    
    大模型精筛输出 JSON 时，经常会贪婪或遗漏丢掉 `url`、`domain`、`title` 字段。
    本算法拿大模型判定高相关的 `source_id` 去反向对照原始 API 召回的数据，
    自动将丢失的大字段强行回补，坚决杜绝“幻觉空链接”和“引用断线”。
    
    参数:
        evidence (list): 大模型精筛返回的证据。
        raw_records (list): 原始 API 请求回来的数据记录。
        
    返回:
        list[dict]: 彻底补全了定位地址和标题的证据列表。
    """
    raw_lookup = {str(r.get("source_id", "")).strip(): r for r in raw_records if r.get("source_id")}
    enriched = []
    for ev in evidence:
        item = dict(ev)
        sid = str(item.get("source_id", "")).strip()
        raw = raw_lookup.get(sid, {})
        if not item.get("url") and raw.get("url"):
            item["url"] = raw["url"]
        if not item.get("domain") and raw.get("domain"):
            item["domain"] = raw["domain"]
        if not item.get("title") and raw.get("title"):
            item["title"] = raw["title"]
        enriched.append(item)
    return enriched


def _prune_evidence_to_allowed_sources(evidence: list[dict], allowed_source_ids: set[str]) -> list[dict]:
    """
    【防幻觉引用】非法引用 ID 裁剪算法。
    
    强力删除大模型自己凭空捏造出来的、根本不存在于原始输入中的虚拟引用 ID，
    捍卫证据链的物理真实性。
    
    参数:
        evidence (list): 大模型拟采纳的证据。
        allowed_source_ids (set): 本轮合法检索出的 ID 集合。
        
    返回:
        list[dict]: 剔除了虚拟捏造 ID 后的可靠证据列表。
    """
    kept: list[dict] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id", "")).strip()
        if source_id and source_id in allowed_source_ids:
            kept.append(item)
    return kept


def _summarize_records(records: list[dict]) -> list[dict]:
    """
    检索轨迹记录缩略生成器。
    
    提取前 5 条记录的标题、定位符和短摘要组成简短的 Trace，
    用于更新 `web_search_trace` 轨迹追踪，极大方便调试。
    
    参数:
        records (list): 记录列表。
        
    返回:
        list[dict]: 缩略轨迹列表。
    """
    summary: list[dict] = []
    for record in records[:5]:
        summary.append(
            {
                "source_id": record.get("source_id"),
                "title": record.get("title", ""),
                "locator": record.get("url") or record.get("doc_id") or "",
                "snippet": str(record.get("snippet", ""))[:160],
            }
        )
    return summary


def _finalize_query_traces(query_traces: list[dict], kept_ids: set[str], rejected_ids: list[str], reject_reason: str) -> list[dict]:
    """
    检索全生命周期明细沉淀归档器。
    
    在每一轮检索节点执行收尾时被激活。清晰归档当前搜索词、
    本轮搜出来的所有 ID、本轮采纳的 ID、不采纳的 ID 以及不采纳的具体原因。
    生成极高可读性的调试 Trace。
    
    参数:
        query_traces (list): 当前追踪缓存。
        kept_ids (set): 本轮被精筛保留的合法 ID。
        rejected_ids (list): 模型认为不相关排除的 ID。
        reject_reason (str): 排除的文本解释。
        
    返回:
        list[dict]: 完成最终生命周期归档的 Trace 列表。
    """
    normalized_rejected = set(_normalize_source_ids(rejected_ids))
    finalized: list[dict] = []
    for trace in query_traces:
        raw_items = [item for item in trace.get("raw_records", []) if isinstance(item, dict)]
        kept_records = [item for item in raw_items if str(item.get("source_id", "")).strip() in kept_ids]
        rejected_records = [
            item
            for item in raw_items
            if str(item.get("source_id", "")).strip() in normalized_rejected or str(item.get("source_id", "")).strip() not in kept_ids
        ]
        trace_item = dict(trace)
        trace_item["raw_source_ids"] = _normalize_source_ids(item.get("source_id") for item in raw_items)
        trace_item["kept_source_ids"] = _normalize_source_ids(item.get("source_id") for item in kept_records)
        trace_item["rejected_source_ids"] = _normalize_source_ids(item.get("source_id") for item in rejected_records)
        trace_item["kept_count"] = len(trace_item["kept_source_ids"])
        trace_item["rejected_count"] = len(trace_item["rejected_source_ids"])
        trace_item["kept_records"] = kept_records[:3]
        trace_item["rejected_records"] = rejected_records[:3]
        if reject_reason:
            trace_item["reject_reason"] = reject_reason
        finalized.append(trace_item)
    return finalized


# ==============================================================================
# 模块四：证据裁判与文献引用函数
# ==============================================================================

def _is_official_domain(domain: str) -> bool:
    """
    官方渠道识别判定器。
    
    检测域名是否以 `.gov`, `.gov.cn`, `.edu`, `.edu.cn` 结尾，或者包含官方字眼。
    该结果是后续信源高保真打分算法的重要依据。
    
    参数:
        domain (str): 顶级/二级域名。
        
    返回:
        bool: 是否属于官方或高校教育机构。
    """
    value = domain.lower()
    return value.endswith(".gov.cn") or value.endswith(".gov") or value.endswith(".edu") or value.endswith(".edu.cn") or "gov" in value or "official" in value


def _score_evidence(record: dict) -> tuple[float, str]:
    """
    信源信用可信度分级判定评级算法。
    
    基于可追溯的域名，对合并后的证据池实施极其精准的客观判分：
    - 企业内部知识库 RAG 召回数据（'local'）：打 0.92 分（默认高可信）；
    - .gov / .edu 官方权威渠道：打 0.88 分；
    - 主流专业与科技财经媒体：打 0.72 分；
    - 普通互联网来源（普通域名）：打 0.58 分（需要后置分析交叉验证）；
    - 来源信息缺失的孤立证据：打 0.45 分。
    
    参数:
        record (dict): 证据字典。
        
    返回:
        tuple[float, str]: (基准可信度得分, 打分依据文本解释)。
    """
    source_type = record.get("source_type")
    if source_type == "local":
        return 0.92, "企业内部知识库证据，默认高可信"
    domain = str(record.get("domain", "")).lower()
    if _is_official_domain(domain):
        return 0.88, "官方或权威机构域名"
    if any(word in domain for word in ["news", "finance", "reuters", "bloomberg", "people", "xinhuanet"]):
        return 0.72, "主流媒体域名"
    if domain:
        return 0.58, "普通互联网来源，需要交叉验证"
    return 0.45, "来源信息不完整"


def _dedupe_sources(items: list[dict], key_fields: list[str]) -> list[dict]:
    """
    字段级去重辅助函数。
    
    参数:
        items (list): 待去重的字典列表。
        key_fields (list): 参与哈希唯一键拼接的字段。
        
    返回:
        list[dict]: 去重后的干净列表。
    """
    seen = set()
    results = []
    for item in items:
        key = tuple(str(item.get(field, "")).strip() for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        results.append(item)
    return results


def _fallback_audit(state: ResearchState) -> dict:
    """
    证据裁判兜底函数。
    
    如果 EvidenceJudge 智能体运行异常或发生严重的 LLM 接口调用灾难，
    由本算法在后端直接接管。基于纯规则自动对状态中的 web/local 证据进行信源判分，
    并对低于 0.6 分的边缘低可靠源生成 `low_confidence` 的 `audit_flags` 审计红牌。
    
    参数:
        state (ResearchState): 共享状态。
        
    返回:
        dict: 兜底的证据裁判成果字典。
    """
    evidence_pool = []
    source_index = []
    audit_flags = []
    for record in state.get("web_evidence", []) + state.get("local_evidence", []):
        score, reason = _score_evidence(record)
        normalized = dict(record)
        normalized["reliability_score"] = score
        normalized["reliability_reason"] = reason
        normalized["source_label"] = record.get("title") or record.get("doc_id") or record.get("url") or record.get("source_id")
        normalized.setdefault("supports", [])
        normalized.setdefault("refutes", [])
        evidence_pool.append(normalized)
        locator = record.get("url") or record.get("doc_id") or ""
        if score < 0.6:
            audit_flags.append({"type": "low_confidence", "target": record.get("source_id"), "reason": reason})
        else:
            source_index.append(
                {
                    "source_id": record.get("source_id"),
                    "label": normalized["source_label"],
                    "locator": locator or "未提供定位信息",
                    "source_type": record.get("source_type", "source"),
                }
            )
    for hypo in state.get("hypotheses", []):
        hypo_id = hypo.get("id")
        related = [item for item in evidence_pool if hypo_id in item.get("supports", []) or hypo_id in item.get("refutes", [])]
        if not related:
            audit_flags.append({"type": "missing_evidence", "target": hypo_id, "reason": "缺少直接关联证据"})
    return {
        "summary": "完成证据评分与审计。",
        "evidence_pool": evidence_pool,
        "audit_flags": audit_flags,
        "source_index": _dedupe_sources(source_index, ["source_id"]),
    }


def _fallback_analysis(state: ResearchState) -> dict:
    """
    分析师节点兜底函数。
    
    如果 Analyst 节点大模型调用出现坏账，由该规则在内存中接管。
    抓取当前已通过裁判的证据池中评分最高的前三个 WEB/LOC 源，生成默认论断 claims，
    并自动与这些文献 ID 进行关系映射，保障研报生成的高追溯品质。
    
    参数:
        state (ResearchState): 当前状态。
        
    返回:
        dict: 兜底的分析结论字典。
    """
    source_ids = [item.get("source_id") for item in state.get("evidence_pool", [])[:3] if item.get("source_id")]
    findings = [
        {
            "claim_id": "c_1",
            "claim": f"围绕“{state['query']}”已完成多源检索，初步证据表明问题可以从网络与本地知识库双侧支撑。",
            "confidence": "medium" if source_ids else "low",
            "source_ids": source_ids,
        }
    ]
    hypothesis_status = []
    for hypo in state.get("hypotheses", []):
        hypothesis_status.append(
            {
                "id": hypo.get("id"),
                "status": "verified" if source_ids else "uncertain",
                "reason": "已有可用证据池" if source_ids else "证据不足",
                "source_ids": source_ids,
            }
        )
    return {
        "analysis_summary": "完成结论归纳与假设状态整理。",
        "hypothesis_status": hypothesis_status,
        "findings": findings,
        "claim_map": [{"claim_id": item["claim_id"], "source_ids": item["source_ids"]} for item in findings],
        "next_actions": [] if source_ids else ["补充更多高质量来源"],
    }


def _render_fallback_report(state: ResearchState) -> str:
    """
    报告草稿自动渲染器。
    
    当写作节点大模型在生成 Markdown 时发生异常时执行。
    直接通过算法读取并归纳 Findings，自动输出结构完整、包含各项检索统计指标
    和完整引用文献溯源说明的默认报告。
    
    参数:
        state (ResearchState): 当前状态。
        
    返回:
        str: 兜底渲染出的纯 Markdown 大报告。
    """
    lines = ["# 调研结果", "", "## 执行摘要", state.get("analysis", "暂无分析结果"), ""]
    lines.append("## 任务规划与假设状态")
    for hypo in state.get("hypotheses", []):
        status = hypo.get("status", "unverified")
        lines.append(f"- {hypo.get('id', 'h')}: {hypo.get('content', '')} | 状态: {status}")
    lines.append("")
    lines.append("## 核心结论")
    for finding in state.get("findings", []):
        refs = "".join(f"[{source_id}]" for source_id in finding.get("source_ids", []))
        lines.append(f"- {finding.get('claim', '')} {refs}".rstrip())
    lines.append("")
    lines.append("## 风险与不确定性")
    if state.get("audit_flags"):
        for flag in state["audit_flags"]:
            lines.append(f"- {flag.get('type')}: {flag.get('reason')} ({flag.get('target')})")
    else:
        lines.append("- 当前未发现明显冲突。")
    lines.append("")
    lines.append("## 检索统计")
    web_stats = state.get("web_retrieval_stats", {})
    local_stats = state.get("local_retrieval_stats", {})
    if web_stats or local_stats:
        lines.append(f"- 网络检索：queries={web_stats.get('query_count', 0)} raw={web_stats.get('raw_count', 0)} kept={web_stats.get('kept_count', 0)} dropped={web_stats.get('dropped_count', 0)}")
        lines.append(f"- 本地检索：queries={local_stats.get('query_count', 0)} raw={local_stats.get('raw_count', 0)} kept={local_stats.get('kept_count', 0)} dropped={local_stats.get('dropped_count', 0)}")
    else:
        lines.append("- 未记录检索统计。")
    lines.append("")
    lines.append("## 引用列表")
    for source in state.get("source_index", []):
        source_type = source.get("source_type", "source")
        lines.append(f"- {source.get('source_id')} [{source_type}]: {source.get('label')} | {source.get('locator')}")
    return "\n".join(lines)


def _build_source_lookup(state: ResearchState) -> dict[str, dict]:
    """
    全会话大一统文献主键 Lookup 提取器。
    
    读取状态中散落在各个局部的证据、索引，将其聚类汇聚成以 source_id 强关联唯一
    物理地址的 Lookup 字典，用作终稿渲染时的快速检索和上标替换。
    
    参数:
        state (ResearchState): 共享状态。
        
    返回:
        dict[str, dict]: 映射唯一文献详情的 lookup 表。
    """
    lookup: dict[str, dict] = {}

    def _put(source_id: str, source_type: str, label: str, locator: str):
        if not source_id:
            return
        item = lookup.get(source_id)
        if not item:
            lookup[source_id] = {
                "source_id": source_id,
                "source_type": source_type or "source",
                "label": label or source_id,
                "locator": locator or "",
            }
            return
        if (not item.get("locator")) and locator:
            item["locator"] = locator
        if (not item.get("label")) and label:
            item["label"] = label
        if item.get("source_type") in {"source", ""} and source_type:
            item["source_type"] = source_type

    for source in state.get("source_index", []):
        _put(
            str(source.get("source_id", "")).strip(),
            str(source.get("source_type", "source")).strip(),
            str(source.get("label", "")).strip(),
            str(source.get("locator", "")).strip(),
        )
    for ev in state.get("evidence_pool", []):
        _put(
            str(ev.get("source_id", "")).strip(),
            str(ev.get("source_type", "source")).strip(),
            str(ev.get("title") or ev.get("source_label") or "").strip(),
            str(ev.get("url") or ev.get("doc_id") or "").strip(),
        )
    for ev in state.get("web_evidence", []):
        _put(
            str(ev.get("source_id", "")).strip(),
            "web",
            str(ev.get("title", "")).strip(),
            str(ev.get("url") or "").strip(),
        )
    for ev in state.get("local_evidence", []):
        _put(
            str(ev.get("source_id", "")).strip(),
            "local",
            str(ev.get("title") or ev.get("doc_id") or "").strip(),
            str(ev.get("doc_id") or "").strip(),
        )
    for key, item in lookup.items():
        if key.startswith("LOC"):
            item["source_type"] = "local"
        elif key.startswith("WEB"):
            item["source_type"] = "web"
    return lookup


def _extract_citation_ids(content: str) -> list[str]:
    """
    正文引用上标 ID 正则解析器。
    
    使用正则表达式精确匹配符合 `[WEB1_1-1]` 或 `[LOC2_3-2]` 的标准常态化物理引用 ID 列表。
    
    参数:
        content (str): 研报 Draft 的文本内容。
        
    返回:
        list[str]: 提取出的去重且保留出现次序的引用 ID 列表。
    """
    pattern = r'\[([A-Z]+\d+_\d+-\d+)\]'
    matches = re.findall(pattern, content)
    return list(dict.fromkeys(matches))


def _validate_and_fix_citations(content: str, valid_source_ids: set[str]) -> tuple[str, list[str]]:
    """
    【防御性洗引用】正文防虚构引用过滤与清洗算法。
    
    逐一匹配并校验大模型正文自己打的所有引用标识。如果引用的 ID 在当前文献库
    （`valid_source_ids`）中无此实存记录（属于大模型胡说八道脑造的引用），
    **算法执行静默物理删除**，彻底斩断了“幻觉死链”，仅在正文保留百分之百真实的文献上标。
    
    参数:
        content (str): 研报 Draft。
        valid_source_ids (set): 本轮真实取证保留的 ID 集合。
        
    返回:
        tuple[str, list[str]]: (过滤并纠偏后的正文 Markdown, 正文真正采纳过的合法文献 ID)。
    """
    pattern = r'\[([A-Z]+\d+_\d+-\d+)\]'
    
    def replace_citation(match):
        citation_id = match.group(1)
        if citation_id in valid_source_ids:
            return f"[{citation_id}]"
        else:
            return ""
    
    fixed_content = re.sub(pattern, replace_citation, content)
    used_ids = [cid for cid in _extract_citation_ids(fixed_content) if cid in valid_source_ids]
    return fixed_content, used_ids


def _render_reference_list(state: ResearchState) -> str:
    """
    【论文级】学术参考资料列表生成渲染器。
    
    1. 动态读取被 `_validate_and_fix_citations` 过滤后的正文真实引用，并按正文出现顺序降级排序；
    2. **物理定位符（Locator）去重**：针对同一个本地物理文件（如同一 `ingest.txt`），即便大模型引用了 5 个不同切片（LOC1_1-1 至 LOC1_1-5），文末的参考资料列表中**也仅优雅展示一条关于该物理文件路径的完美记录**。
    3. 规范拼接最终 Markdown 文献清单。
    
    参数:
        state (ResearchState): 共享状态。
        
    返回:
        str: 极具学术感排版的 Markdown 格式参考资料列表。
    """
    lines = ["## 参考资料"]
    lookup = _build_source_lookup(state)
    
    draft_content = state.get("draft", "") or state.get("final", "")
    cited_ids = []
    if draft_content:
        for sid in _extract_citation_ids(draft_content):
            if sid in lookup and sid not in cited_ids:
                cited_ids.append(sid)
    
    if not cited_ids:
        for finding in state.get("findings", []):
            for sid in finding.get("source_ids", []):
                text = str(sid).strip()
                if text and text not in cited_ids and text in lookup:
                    cited_ids.append(text)
    
    if not cited_ids:
        cited_ids = list(lookup.keys())
    
    seen_locators = set()
    display_ids = []
    web_ids = []
    local_ids = []
    
    for sid in cited_ids:
        source = lookup.get(sid)
        if not source:
            continue
        source_type = source.get("source_type", "")
        locator = source.get("locator", "").strip()
        
        if source_type == "local":
            dedup_key = locator or sid
            if dedup_key in seen_locators:
                continue
            seen_locators.add(dedup_key)
            local_ids.append(sid)
        else:
            web_ids.append(sid)
    
    display_ids = web_ids + local_ids
    
    if not display_ids:
        display_ids = cited_ids[:15]
    
    for sid in display_ids:
        source = lookup.get(sid)
        if not source:
            continue
        locator = source.get("locator", "").strip()
        label = source.get("label", "").strip()
        source_type = source.get("source_type", "source")
        source_id = source.get("source_id", sid)
        
        if not locator:
            locator = "链接暂不可用" if source_type == "web" else "本地知识库"
        
        lines.append(f"- [{source_id}] [{source_type}]: {label} | {locator}")
    
    if len(lines) == 1:
        lines.append("- 暂无参考资料")
    return "\n".join(lines)


def _render_execution_appendix(state: ResearchState) -> str:
    """
    研究明细与执行轨迹附录渲染器。
    
    在研报终稿尾部，自动将整个调研任务过程中 Planner 的拆解、Scouts 执行过
    的详细 Tavily/Milvus 检索数据、以及完整的 Kept/Dropped 过滤明细和排除理由，
    以规范的结构大一统拼接进附录中，实现极致透明和科学可信。
    
    参数:
        state (ResearchState): 共享状态。
        
    返回:
        str: Markdown 格式的丰富执行明细附录。
    """
    lines = ["## 规划与检索明细", "", "### 执行概览"]
    search_plan = state.get("search_plan", [])
    web_stats = state.get("web_retrieval_stats", {})
    local_stats = state.get("local_retrieval_stats", {})
    lines.append(f"- 规划生成研究问题数: {len(state.get('research_questions', []))}")
    lines.append(f"- 规划生成搜索步骤数: {len(search_plan)}")
    
    iteration = state.get("iteration", 0)
    lines.append(f"- 经过 {iteration + 1} 轮检索迭代")
    if state.get("needs_more_research"):
        lines.append(f"- 信息缺口: {state.get('missing_gaps', [])}")
        
    lines.append(
        f"- 实际执行网页检索问题数: {web_stats.get('query_count', 0)} | 原始命中: {web_stats.get('raw_count', 0)} | 保留证据: {web_stats.get('kept_count', 0)} | 丢弃: {web_stats.get('dropped_count', 0)}"
    )
    lines.append(
        f"- 实际执行本地检索问题数: {local_stats.get('query_count', 0)} | 原始命中: {local_stats.get('raw_count', 0)} | 保留证据: {local_stats.get('kept_count', 0)} | 丢弃: {local_stats.get('dropped_count', 0)}"
    )
    lines.append("")
    lines.append("### 问题拆解明细")
    for sq in state.get("sub_questions", []):
        lines.append(f"- {sq}")
    if not state.get("sub_questions"):
        lines.append("- 无")
    lines.append("")
    lines.append("### 规划输出")
    outline = state.get("outline", [])
    if outline:
        for section in outline:
            lines.append(
                f"- {section.get('id')}: {section.get('title')} | {section.get('description')} | search_queries={section.get('search_queries', [])}"
            )
    else:
        lines.append("- 无")
    lines.append("")
    lines.append("### 研究问题")
    for index, question in enumerate(state.get("research_questions", []), 1):
        lines.append(f"- Q{index}: {question}")
    if not state.get("research_questions"):
        lines.append("- 无")
    lines.append("")
    lines.append("### 搜索计划")
    for index, item in enumerate(state.get("search_plan", []), 1):
        lines.append(
            f"- S{index}: section={item.get('section_id')} | query={item.get('query')} | source={item.get('source_preference')} | reason={item.get('reason')}"
        )
    if not state.get("search_plan"):
        lines.append("- 无")
    lines.append("")
    if state.get("supplementary_queries"):
        lines.append("### 补搜计划")
        for index, item in enumerate(state.get("supplementary_queries", []), 1):
            lines.append(f"- S{index} (补搜): query={item.get('query')} | reason={item.get('reason')}")
        lines.append("")
    lines.append("### 网页检索明细")
    for index, trace in enumerate(state.get("web_search_trace", []), 1):
        lines.append(
            f"- WQ{index}: section={trace.get('section_id')} | query={trace.get('query')} | reason={trace.get('reason')} | raw={trace.get('raw_count', 0)} | kept={trace.get('kept_count', 0)} | rejected={trace.get('rejected_count', 0)}"
        )
        lines.append(f"  - raw_ids={trace.get('raw_source_ids', [])}")
        lines.append(f"  - kept_ids={trace.get('kept_source_ids', [])}")
        lines.append(f"  - rejected_ids={trace.get('rejected_source_ids', [])}")
        if trace.get("reject_reason"):
            lines.append(f"  - reject_reason={trace.get('reject_reason')}")
        lines.append("  - raw_samples:")
        for item in trace.get("raw_records", [])[:3]:
            lines.append(f"    - {item.get('source_id')}: {item.get('title')} | {item.get('locator')}")
        if trace.get("kept_records"):
            lines.append("  - kept_samples:")
            for item in trace.get("kept_records", [])[:3]:
                lines.append(f"    - {item.get('source_id')}: {item.get('title')} | {item.get('locator')}")
        if trace.get("rejected_records"):
            lines.append("  - rejected_samples:")
            for item in trace.get("rejected_records", [])[:3]:
                lines.append(f"    - {item.get('source_id')}: {item.get('title')} | {item.get('locator')}")
    if not state.get("web_search_trace"):
        lines.append("- 无")
    lines.append("")
    lines.append("### 本地检索明细")
    for index, trace in enumerate(state.get("local_rag_trace", []), 1):
        lines.append(
            f"- LQ{index}: section={trace.get('section_id')} | query={trace.get('query')} | reason={trace.get('reason')} | raw={trace.get('raw_count', 0)} | kept={trace.get('kept_count', 0)} | rejected={trace.get('rejected_count', 0)}"
        )
        lines.append(f"  - raw_ids={trace.get('raw_source_ids', [])}")
        lines.append(f"  - kept_ids={trace.get('kept_source_ids', [])}")
        lines.append(f"  - rejected_ids={trace.get('rejected_source_ids', [])}")
        if trace.get("reject_reason"):
            lines.append(f"  - reject_reason={trace.get('reject_reason')}")
        lines.append("  - raw_samples:")
        for item in trace.get("raw_records", [])[:3]:
            lines.append(f"    - {item.get('source_id')}: {item.get('title')} | {item.get('locator')}")
        if trace.get("kept_records"):
            lines.append("  - kept_samples:")
            for item in trace.get("kept_records", [])[:3]:
                lines.append(f"    - {item.get('source_id')}: {item.get('title')} | {item.get('locator')}")
        if trace.get("rejected_records"):
            lines.append("  - rejected_samples:")
            for item in trace.get("rejected_records", [])[:3]:
                lines.append(f"    - {item.get('source_id')}: {item.get('title')} | {item.get('locator')}")
    if not state.get("local_rag_trace"):
        lines.append("- 无")
    return "\n".join(lines)


def _ensure_reference_section(content: str, state: ResearchState) -> str:
    """
    参考资料尾部无缝安全拼接器。
    
    检测大模型返回的正文草稿尾部是否包含参考资料模块说明。
    若无，调用 `_render_reference_list` 优雅物理拼接。
    
    参数:
        content (str): 正文草稿。
        state (ResearchState): 共享状态。
        
    返回:
        str: 完美拼接了参考资料后的报告。
    """
    base = content.rstrip()
    references = _render_reference_list(state)
    if "## 引用列表" in base or "## 来源清单" in base or "## 参考资料" in base:
        return base
    return f"{base}\n\n{references}"


def _normalize_source_ids(values) -> list[str]:
    """
    引用 ID 文本规范去重处理器。
    
    参数:
        values (list): 输入引用 ID。
        
    返回:
        list[str]: 去重规范处理后的引用 ID 列表。
    """
    normalized = []
    for value in values or []:
        text = str(value).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


# ==============================================================================
# 模块五：LangGraph 智能体节点函数
# ==============================================================================

def intent_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    """
    【1. 意图分流器节点】 (Intent Router)
    
    状态机的首发节点。结合规则初判（detect_intent）和大模型意图理解，
    精炼研判用户 Query 属于直接快速回答，还是深度多轮多源调研，
    往 ResearchState 写入 'direct' 或 'multiagent' 状态，直接驱动图条件分支路由走向。
    
    参数:
        state (ResearchState): 图运行时共享状态沙盒。
        agent (any): 绑定的意图识别大语言模型。
        agent_name (str): 智能体中文代号。
        
    返回:
        ResearchState: 包含更新后的 'intent'、'draft' 和历史消息的更新字典。
    """
    logger.info("%s 开始 | agent=%s", colorize("[intent]", "cyan"), colorize(agent_name, "magenta"))
    rule_route = detect_intent(state["query"])
    prompt = (
        f"用户问题：{state['query']}\n"
        f"规则引擎初判：{rule_route}\n"
        "请输出 JSON：{\"route\":\"direct|multiagent\",\"reason\":\"...\"}"
    )
    payload, content, messages = _invoke_json_agent(
        state,
        prompt,
        agent,
        agent_name,
        "intent",
        {"route": rule_route, "reason": "rule"},
    )
    route = str(payload.get("route", rule_route)).strip().lower()
    if route not in {"direct", "multiagent"}:
        route = rule_route
    logger.info("%s 路由: %s", colorize("[intent]", "green"), route)
    return {"intent": route, "draft": content, "messages": messages}


def direct_answer_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    """
    【2. 快速直答节点】 (Direct Responder)
    
    当 Intent Router 判定提问属于日常闲聊或极简常识时唤醒。
    直接给用户一个精炼、口语化的答复，绕过繁琐的研究大纲规划和双源并行检索，
    极速收敛，直接流转至 [END]。
    
    参数:
        state (ResearchState): 共享状态。
        agent (any): 大语言模型实例。
        agent_name (str): 智能体中文代号。
        
    返回:
        ResearchState: 包含 final 终答和消息流的更新字典。
    """
    logger.info("%s 开始 | agent=%s", colorize("[direct_answer]", "cyan"), colorize(agent_name, "magenta"))
    prompt = f"用户问题：{state['query']}"
    human = HumanMessage(content=with_memory_context(state, prompt))
    result = agent.invoke({"messages": [human]})
    content = _last_content(result).strip()
    emit("direct_answer", content)
    return {
        "intent": "direct",
        "final": content,
        "draft": content,
        "analysis_summary": content,
        "needs_more_research": False,
        "messages": [human, result["messages"][-1]],
    }


def plan_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    """
    【3. 总规划师节点】 (ChiefArchitect)
    
    深度调研的“领航员”。当确定进入 multiagent 研究后被执行。
    根据核心 Query，驱动模型生成结构化研究大纲（outline）、衍生子问题（sub_questions）
    与资源预算约束。随后，算法基于大纲和衍生问题在后台自动计算、
    派生出初始章节级结构化搜索计划 `search_plan` 并写入状态沙盒。
    
    参数:
        state (ResearchState): 共享状态。
        agent (any): 绑定的规划大语言模型。
        agent_name (str): 智能体中文代号。
        
    返回:
        ResearchState: 包含 outline, sub_questions, search_plan 等规划核心因子的更新字典。
    """
    logger.info("%s 开始 | agent=%s", colorize("[plan]", "cyan"), colorize(agent_name, "magenta"))
    log_inputs("plan", agent_name, {"query": state["query"]})
    fallback = _default_plan(state)
    payload, content, messages = _invoke_json_agent(
        state,
        f"用户需求：{state['query']}\n请先做大纲与问题拆解，再输出规划 JSON。",
        agent,
        agent_name,
        "plan",
        fallback,
    )
    outline = payload.get("outline") if isinstance(payload.get("outline"), list) else fallback["outline"]
    sub_questions = payload.get("sub_questions") if isinstance(payload.get("sub_questions"), list) else fallback["sub_questions"]
    research_questions = payload.get("research_questions") if isinstance(payload.get("research_questions"), list) else fallback["research_questions"]
    budget = payload.get("budget") if isinstance(payload.get("budget"), dict) else fallback["budget"]
    search_plan = _derive_search_plan(outline, sub_questions, research_questions, state["query"])
    plan_summary = payload.get("objective") or state["query"]
    return {
        "phase": "planning completed",
        "plan": plan_summary,
        "outline": outline,
        "sub_questions": sub_questions,
        "research_questions": research_questions,
        "search_plan": search_plan,
        "budget": budget,
        "messages": messages,
        "draft": content,
        "iteration": 0,
    }


def web_search_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    """
    【4. 网页实时取证与精筛节点】 (WebScout)
    
    网络侧核心取证节点。
    1. 智能调度 `_build_queries`，从 search_plan 或补搜计划中分发检索 Query；
    2. 高并发并行调取 **Tavily 搜索引擎 REST 端点** 召回网络信息；
    3. 调用域名黑名单和相关性粗筛，丢弃低可靠网站及不相关记录；
    4. 用紧凑 JSON 格式（_format_raw_records）喂给 LLM 精筛，产出精审证据；
    5. 【高健壮性防幻觉】执行非法引用 ID 强力裁剪（_prune_evidence_to_allowed_sources）
       与原始大字段智能回补（_enrich_evidence_from_raw），更新轨迹并追加至 `web_evidence`。
    
    参数:
        state (ResearchState): 共享状态。
        agent (any): 网页过滤大语言模型。
        agent_name (str): 智能体中文代号。
        
    返回:
        ResearchState: 包含更新后的 web_evidence、检索统计及 WQ-Trace 轨迹的更新字典。
    """
    logger.info("%s 开始 | agent=%s", colorize("[web_search]", "cyan"), colorize(agent_name, "magenta"))
    queries = _build_queries(state, "web")
    logger.info("[web_search_node] 构建查询 | 查询数量=%s | queries=%s", len(queries), [q.get("query", "") for q in queries])
    
    raw_records = []
    query_traces = state.get("web_search_trace", [])
    
    iteration = state.get("iteration", 0)
    prefix = f"WEB{iteration+1}"
    logger.info("[web_search_node] 迭代信息 | iteration=%s | prefix=%s", iteration, prefix)
    
    for query_index, item in enumerate(queries, 1):
        query_text = str(item.get("query", ""))
        logger.info("[web_search_node] 执行第 %s/%s 个查询 | query=%s | section_id=%s", query_index, len(queries), query_text, item.get("section_id"))
        records = tavily_web_search_records(query_text, count=4)
        logger.info("[web_search_node] 查询 %s 返回 | 记录数=%s", query_index, len(records))
        records = _assign_source_ids(records, f"{prefix}_{query_index}")
        for record in records:
            record["section_id"] = item.get("section_id")
            record["search_query"] = item.get("query")
        raw_records.extend(records)
        query_traces.append(
            {
                "iteration": iteration,
                "plan_step": query_index,
                "query": str(item.get("query", "")),
                "section_id": item.get("section_id"),
                "reason": item.get("reason", ""),
                "source_preference": item.get("source_preference", "web"),
                "raw_count": len(records),
                "raw_records": _summarize_records(records),
            }
        )
    raw_records = _dedupe_sources(raw_records, ["url", "title"])
    raw_records = _minimal_record_filter(raw_records, ["title", "snippet", "url"])
    logger.info("[web_search_node] 数据清洗后 | 去重过滤后记录数=%s", len(raw_records))
    
    web_retrieval_stats = state.get("web_retrieval_stats", {})
    web_retrieval_stats["query_count"] = web_retrieval_stats.get("query_count", 0) + len(queries)
    web_retrieval_stats["raw_count"] = web_retrieval_stats.get("raw_count", 0) + len(raw_records)
    
    log_inputs("web_search", agent_name, {"query_count": str(len(queries)), "raw_count": str(len(raw_records))})
    if not raw_records:
        logger.warning("[web_search_node] 无可用网页证据，跳过网页上下文注入 | 查询数=%s", len(queries))
        logger.info("%s 无可用网页证据，跳过网页上下文注入", colorize("[web_search]", "yellow"))
        return {
            "web_search": "未检索到可用网页证据，已跳过网页上下文注入。",
            "web_evidence": state.get("web_evidence", []),
            "web_retrieval_stats": web_retrieval_stats,
            "web_search_trace": query_traces,
        }
    logger.info("[web_search_node] 调用 LLM 整理证据 | raw_records=%s", len(raw_records))
    fallback = _fallback_web_evidence(raw_records)
    payload, content, messages = _invoke_json_agent(
        state,
        "请基于以下网页证据整理结构化 JSON。\n"
        f"原问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"原始网页证据：\n{_format_raw_records(raw_records, 'web')}",
        agent,
        agent_name,
        "web_search",
        fallback,
    )
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), list) else fallback["evidence"]
    logger.info("[web_search_node] LLM 返回证据 | evidence数量=%s", len(evidence))
    allowed_source_ids = {str(item.get("source_id")) for item in raw_records if item.get("source_id")}
    evidence = _prune_evidence_to_allowed_sources(evidence, allowed_source_ids)
    evidence = _enrich_evidence_from_raw(evidence, raw_records)
    
    web_retrieval_stats["kept_count"] = web_retrieval_stats.get("kept_count", 0) + len(evidence)
    web_retrieval_stats["dropped_count"] = web_retrieval_stats.get("dropped_count", 0) + max(len(raw_records) - len(evidence), 0)
    
    kept_ids = {str(item.get("source_id")) for item in evidence if item.get("source_id")}
    query_traces = _finalize_query_traces(
        query_traces,
        kept_ids,
        payload.get("rejected_source_ids", []),
        str(payload.get("reject_reason", "")).strip(),
    )
    
    existing_evidence = state.get("web_evidence", [])
    logger.info("[web_search_node] 节点完成 | 新增证据=%s | 累计证据=%s", len(evidence), len(existing_evidence) + len(evidence))
    return {
        "web_search": payload.get("summary", content),
        "web_evidence": existing_evidence + evidence,
        "web_retrieval_stats": web_retrieval_stats,
        "web_search_trace": query_traces,
        "messages": messages,
    }


def local_rag_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    """
    【5. 本地知识取证与精筛节点】 (LocalRAGScout)
    
    本地侧核心取证节点。
    与网页取证节点并列并行。调用 [tools.py](file:///d:/AI/deep_research/deep_research/app/mult_agents/tools.py) 的向量检索接口，
    高召回企业 Milvus 向量库切片数据，执行实体碰撞粗筛和模型 JSON 精筛，
    并将洗好的本地高可信局内证据追加至 `local_evidence`，同时归档 LQ-Trace。
    
    参数:
        state (ResearchState): 共享状态。
        agent (any): 本地知识过滤大语言模型。
        agent_name (str): 智能体中文代号。
        
    返回:
        ResearchState: 包含更新后的 local_evidence、本地检索统计及 LQ-Trace 的更新字典。
    """
    logger.info("%s 开始 | agent=%s", colorize("[local_rag]", "cyan"), colorize(agent_name, "magenta"))
    queries = _build_queries(state, "local")
    raw_records = []
    query_traces = state.get("local_rag_trace", [])
    
    iteration = state.get("iteration", 0)
    prefix = f"LOC{iteration+1}"
    
    for query_index, item in enumerate(queries, 1):
        records = search_knowledge_base_records(str(item.get("query", "")), limit=4)
        records = _assign_source_ids(records, f"{prefix}_{query_index}")
        for record in records:
            record["section_id"] = item.get("section_id")
            record["search_query"] = item.get("query")
        raw_records.extend(records)
        query_traces.append(
            {
                "iteration": iteration,
                "plan_step": query_index,
                "query": str(item.get("query", "")),
                "section_id": item.get("section_id"),
                "reason": item.get("reason", ""),
                "source_preference": item.get("source_preference", "local"),
                "raw_count": len(records),
                "raw_records": _summarize_records(records),
            }
        )
    raw_records = _dedupe_sources(raw_records, ["doc_id", "snippet"])
    raw_records = _minimal_record_filter(raw_records, ["snippet", "title", "doc_id"])
    
    local_retrieval_stats = state.get("local_retrieval_stats", {})
    local_retrieval_stats["query_count"] = local_retrieval_stats.get("query_count", 0) + len(queries)
    local_retrieval_stats["raw_count"] = local_retrieval_stats.get("raw_count", 0) + len(raw_records)
    
    log_inputs("local_rag", agent_name, {"query_count": str(len(queries)), "raw_count": str(len(raw_records))})
    if not raw_records:
        logger.info("%s 无可用本地证据，跳过本地上下文注入", colorize("[local_rag]", "yellow"))
        return {
            "local_rag": "未检索到可用本地知识库证据，已跳过本地上下文注入。",
            "local_evidence": state.get("local_evidence", []),
            "local_retrieval_stats": local_retrieval_stats,
            "local_rag_trace": query_traces,
        }
    fallback = _fallback_local_evidence(raw_records)
    payload, content, messages = _invoke_json_agent(
        state,
        "请基于以下知识库证据整理结构化 JSON。\n"
        f"原问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"原始知识库证据：\n{_format_raw_records(raw_records, 'local')}",
        agent,
        agent_name,
        "local_rag",
        fallback,
    )
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), list) else fallback["evidence"]
    allowed_source_ids = {str(item.get("source_id")) for item in raw_records if item.get("source_id")}
    evidence = _prune_evidence_to_allowed_sources(evidence, allowed_source_ids)
    
    local_retrieval_stats["kept_count"] = local_retrieval_stats.get("kept_count", 0) + len(evidence)
    local_retrieval_stats["dropped_count"] = local_retrieval_stats.get("dropped_count", 0) + max(len(raw_records) - len(evidence), 0)
    
    kept_ids = {str(item.get("source_id")) for item in evidence if item.get("source_id")}
    query_traces = _finalize_query_traces(
        query_traces,
        kept_ids,
        payload.get("rejected_source_ids", []),
        str(payload.get("reject_reason", "")).strip(),
    )
    
    existing_evidence = state.get("local_evidence", [])
    return {
        "local_rag": payload.get("summary", content),
        "local_evidence": existing_evidence + evidence,
        "local_retrieval_stats": local_retrieval_stats,
        "local_rag_trace": query_traces,
        "messages": messages,
    }


def deep_dive_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    """
    【6. 证据裁判与冲突审计节点】 (EvidenceJudge)
    
    证据链主裁判。
    汇融双源精筛证据，执行信源信用客观判分机制（RAG 享 0.92, 官方享 0.88 等），
    并在模型对矛盾信息进行辩论和识别后，生成 `audit_flags` 审计红牌与低置信度警告，
    构建全会话文献 Lookup 索引 `source_index` 及统一裁判证据池 `evidence_pool`。
    
    参数:
        state (ResearchState): 共享状态。
        agent (any): 证据裁判大语言模型。
        agent_name (str): 智能体中文代号。
        
    返回:
        ResearchState: 包含更新后的 evidence_pool、audit_flags 和 source_index 的更新字典。
    """
    logger.info("%s 开始 | agent=%s", colorize("[deep_dive]", "cyan"), colorize(agent_name, "magenta"))
    if not state.get("web_evidence") and not state.get("local_evidence"):
        logger.info("%s 等待检索结果", colorize("[deep_dive]", "yellow"))
        return {}
    fallback = _fallback_audit(state)
    payload, content, messages = _invoke_json_agent(
        state,
        "请对 web 与 local 证据进行评分、去重、冲突审计，并只输出 JSON。\n"
        f"问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"web_evidence：{json.dumps(state.get('web_evidence', []), ensure_ascii=False)}\n"
        f"local_evidence：{json.dumps(state.get('local_evidence', []), ensure_ascii=False)}",
        agent,
        agent_name,
        "deep_dive",
        fallback,
    )
    payload_pool = payload.get("evidence_pool") if isinstance(payload.get("evidence_pool"), list) else []
    raw_evidence = state.get("web_evidence", []) + state.get("local_evidence", [])
    allowed_source_ids = {str(item.get("source_id", "")).strip() for item in raw_evidence if item.get("source_id")}
    evidence_pool = []
    for item in payload_pool:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("source_id", "")).strip()
        if sid and sid in allowed_source_ids:
            evidence_pool.append(item)
    if not evidence_pool:
        evidence_pool = fallback["evidence_pool"]
    existing_ids = {str(item.get("source_id", "")).strip() for item in evidence_pool if isinstance(item, dict)}
    for record in raw_evidence:
        sid = str(record.get("source_id", "")).strip()
        if not sid or sid in existing_ids:
            continue
        score, reason = _score_evidence(record)
        evidence_pool.append(
            {
                "source_id": sid,
                "source_type": record.get("source_type", "source"),
                "title": record.get("title") or sid,
                "url": record.get("url", ""),
                "doc_id": record.get("doc_id", ""),
                "snippet": record.get("snippet", ""),
                "supports_questions": record.get("supports_questions", []),
                "reliability_score": score,
                "reliability_reason": reason,
                "source_label": record.get("title") or record.get("doc_id") or record.get("url") or sid,
            }
        )
        existing_ids.add(sid)
    audit_flags = payload.get("audit_flags") if isinstance(payload.get("audit_flags"), list) else fallback["audit_flags"]
    source_index = []
    for item in evidence_pool:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("source_id", "")).strip()
        if not sid:
            continue
        source_index.append(
            {
                "source_id": sid,
                "label": item.get("title") or item.get("source_label") or sid,
                "locator": item.get("url") or item.get("doc_id") or "",
                "source_type": item.get("source_type", "source"),
            }
        )
    source_index = _dedupe_sources(source_index, ["source_id"])
    return {
        "deep_dive": payload.get("summary", content),
        "audit": payload.get("summary", content),
        "evidence_pool": evidence_pool,
        "audit_flags": audit_flags,
        "source_index": source_index,
        "messages": messages,
    }


def analyze_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    """
    【7. 分析师结论归纳与缺口评估节点】 (Analyst)
    
    深度研究“思想家”。
    读取证据裁判后的池子，精炼归纳 Findings 断言，并与文献 ID 进行 Claim-Source 强力绑定。
    同时，极其严格地评估当前证据是否百分之百完备能结题。若存在明显缺口，
    向状态写入 `missing_gaps` 并将 `needs_more_research` 门阈开关设为 `True`，触发环路控制跳转。
    
    参数:
        state (ResearchState): 共享状态。
        agent (any): 分析师大语言模型。
        agent_name (str): 智能体中文代号。
        
    返回:
        ResearchState: 包含 findings, claim_map, needs_more_research 等关键指标的更新字典。
    """
    logger.info("%s 开始 | agent=%s", colorize("[analyze]", "cyan"), colorize(agent_name, "magenta"))
    fallback = _fallback_analysis_md(state)
    payload, content, messages = _invoke_json_agent(
        state,
        "请基于证据池输出结论映射 JSON，并评估证据完备性：\n"
        f"原问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"证据池：{json.dumps(state.get('evidence_pool', []), ensure_ascii=False)}\n"
        f"审计标记：{json.dumps(state.get('audit_flags', []), ensure_ascii=False)}",
        agent,
        agent_name,
        "analyze",
        fallback,
    )
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else fallback["findings"]
    claim_map = payload.get("claim_map") if isinstance(payload.get("claim_map"), list) else fallback["claim_map"]
    needs_more_research = payload.get("needs_more_research", False)
    missing_gaps = payload.get("missing_gaps", [])
    analysis_summary = payload.get("analysis_summary", content)
    return {
        "analysis": analysis_summary,
        "findings": findings,
        "claim_map": claim_map,
        "needs_more_research": needs_more_research,
        "missing_gaps": missing_gaps,
        "messages": messages,
    }


def reflect_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    """
    【8. 缺口反思与补搜改写节点】 (ResearchPlanner)
    
    补搜战术制定者。
    当状态触发补搜控制路由后运行。大模型对照历史所有的检索 Query 以及分析师反馈的
    `missing_gaps`（信息缺口说明），重新进行不重复的语义词汇替换与检索改写，
    输出全新的 `supplementary_queries` 补搜计划，将图迭代计数器递增，流转回取证节点。
    
    参数:
        state (ResearchState): 共享状态。
        agent (any): 改写大语言模型。
        agent_name (str): 智能体中文代号。
        
    返回:
        ResearchState: 包含累增后的 iteration 和新补搜计划的更新字典。
    """
    logger.info("%s 开始 | agent=%s", colorize("[reflect]", "cyan"), colorize(agent_name, "magenta"))
    missing_gaps = state.get("missing_gaps", [])
    log_inputs("reflect", agent_name, {"missing_gaps": str(missing_gaps)})
    fallback = {
        "reflection_summary": "默认补搜",
        "supplementary_queries": [{"section_id": "gap_1", "query": state["query"], "source_preference": "hybrid", "reason": "fallback"}]
    }
    prompt = (
        f"分析师指出当前证据不足以完全回答问题，存在以下信息缺口：\n{json.dumps(missing_gaps, ensure_ascii=False)}\n\n"
        f"原问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"已执行过的搜索计划：\n{json.dumps(state.get('search_plan', []), ensure_ascii=False)}\n"
        f"已执行过的补搜计划：\n{json.dumps(state.get('supplementary_queries', []), ensure_ascii=False)}\n\n"
        "请生成新的补搜计划以填补缺口。"
    )
    payload, content, messages = _invoke_json_agent(
        state,
        prompt,
        agent,
        agent_name,
        "reflect",
        fallback,
    )
    return {
        "iteration": state.get("iteration", 0) + 1,
        "supplementary_queries": payload.get("supplementary_queries", fallback["supplementary_queries"]),
        "messages": messages,
    }


def write_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    """
    【9. 研报长文撰写与引用校验节点】 (SeniorWriter)
    
    深度研报落笔与净化终审官。
    1. 【极致优化点】只保留本工序指令和 findings、文献库，彻底清空 messages 历史重连大模型，确保长篇大作（2000-3000字以上）火力全开，且杜绝输出杂乱字典或 JSON；
    2. 【正文引用强清洗】调用 `_validate_and_fix_citations`，利用正则抓取大模型自己打的所有引用标识，无情滤除大模型脑造/幻觉出的不存在引用；
    3. 【学术级文末拼接】自动调用 `_ensure_reference_section` 拼接路径级去重后的文献资料列表和检索明细附录，输出 `final` 终稿并完美收尾。
    
    参数:
        state (ResearchState): 共享状态。
        agent (any): 写作大语言模型。
        agent_name (str): 智能体中文代号。
        
    返回:
        ResearchState: 包含最终终稿报告 final、草稿 draft 及消息对的更新字典。
    """
    logger.info("%s 开始 | agent=%s", colorize("[write]", "cyan"), colorize(agent_name, "magenta"))
    valid_source_ids = [str(item.get("source_id", "")).strip() for item in state.get("source_index", []) if item.get("source_id")]
    valid_source_ids = [item for item in valid_source_ids if item][:80]
    valid_source_ids_set = set(valid_source_ids)
    prompt = (
        "请严格根据以下信息撰写最终的 Markdown 研报。请直接输出正文，绝对不要输出任何 JSON 结构，也不要复述你的指令。\n\n"
        f"核心问题：{state['query']}\n"
        f"子问题拆解：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n\n"
        "【分析结论 (Findings)】：\n"
        f"{json.dumps(state.get('findings', []), ensure_ascii=False)}\n\n"
        "【可用来源索引 (source_index)】：\n"
        f"{json.dumps(state.get('source_index', []), ensure_ascii=False)}\n\n"
        "【合法引用ID列表】：\n"
        f"{json.dumps(valid_source_ids, ensure_ascii=False)}\n\n"
        "【可能存在的风险/冲突 (Audit Flags)】：\n"
        f"{json.dumps(state.get('audit_flags', []), ensure_ascii=False)}\n\n"
        "要求：正文必须使用合法引用ID（例如 [WEB1_1-1]、[LOC1_1-3]）；禁止使用不存在的编号。"
        "结尾不需要你来列举引用列表，系统会自动拼接。"
    )
    human = HumanMessage(content=with_memory_context(state, prompt))
    result = agent.invoke({"messages": [human]})
    content = _last_content(result)
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"^```markdown\s*", "", content)
    content = re.sub(r"^```\s*", "", content)
    content = re.sub(r"```$", "", content.strip())
    content, used_citation_ids = _validate_and_fix_citations(content, valid_source_ids_set)
    final_content = _ensure_reference_section(content, state)
    emit("write", final_content)
    return {"draft": final_content, "final": final_content, "messages": [human, result["messages"][-1]]}


# ==============================================================================
# 模块六：辅助函数重载备份区
# ==============================================================================

def _fallback_analysis_md(state: ResearchState) -> dict:
    """
    分析师节点的高兼容度备用兜底字典生成器。
    
    主要为大模型发生极端格式断裂时，提供一份符合系统最高兼容规约的
    初始化默认字典，保障 needs_more_research 布尔开关和 Findings 的完美对接。
    
    参数:
        state (ResearchState): 共享状态。
        
    返回:
        dict: 默认退避使用的分析字典。
    """
    return {
        "analysis_summary": "默认分析结论",
        "needs_more_research": False,
        "missing_gaps": [],
        "findings": [],
        "claim_map": [],
        "next_actions": [],
    }


def _fallback_web_evidence(records: list[dict]) -> dict:
    """
    Tavily 网页证据的算法级容错兜底器。
    
    当 WebScout 节点大语言模型网络连接中断时由算法接管。
    直接遍历 Tavily 搜回的记录列表，强制生成合法的 web 格式证据 JSON 结构，
    保证引用链路绝对完整。
    
    参数:
        records (list): Tavily API 召回的数据。
        
    返回:
        dict: 结构化的安全网页证据字典。
    """
    evidence = []
    for record in records:
        evidence.append(
            {
                "source_id": record.get("source_id"),
                "title": record.get("title"),
                "url": record.get("url", ""),
                "snippet": record.get("snippet", ""),
                "domain": record.get("domain", ""),
                "source_type": "web",
                "reliability_hint": "official" if _is_official_domain(record.get("domain", "")) else "unknown",
                "supports": [],
                "notes": "",
            }
        )
    return {"summary": "完成网页证据采集。", "evidence": evidence, "gaps": []}


def _fallback_local_evidence(records: list[dict]) -> dict:
    """
    Local RAG 本地知识的算法级容错兜底器。
    
    当 LocalRAGScout 节点大语言模型发生调用坏账时由算法直接接管。
    遍历 Milvus 召回切片，强制转化为合法的 local 格式 RAG 证据 JSON 结构，
    保障正文 RAG 切片引用的安全回填。
    
    参数:
        records (list): Milvus 召回的数据片段。
        
    返回:
        dict: 结构化的本地知识证据字典。
    """
    evidence = []
    for record in records:
        evidence.append(
            {
                "source_id": record.get("source_id"),
                "doc_id": record.get("doc_id", ""),
                "title": record.get("title", "") or record.get("source_id", ""),
                "snippet": record.get("snippet", ""),
                "source_type": "local",
                "reliability_hint": "internal",
                "supports": [],
                "notes": "",
            }
        )
    return {"summary": "完成本地知识库证据采集。", "evidence": evidence, "gaps": []}
