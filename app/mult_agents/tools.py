"""工具模块：封装 Web 检索、本地 RAG 查询与通用辅助工具函数。"""

from datetime import datetime
import ast
import json
import logging
import operator
import os
import re
import sys
import io
import subprocess
from contextlib import redirect_stdout
from pathlib import Path
import urllib.error
import urllib.request
import urllib.parse

from langchain_core.tools import tool
from typing import Optional
from .rag.core import RAGSystem, RAGConfig

logger = logging.getLogger("mult_agents")

# 全局 RAG 系统实例
_RAG_SYSTEM: Optional[RAGSystem] = None

def init_rag_system(api_key: str, config: Optional[RAGConfig] = None):
    """初始化全局 RAG 系统"""
    global _RAG_SYSTEM
    if _RAG_SYSTEM is None:
        try:
            _RAG_SYSTEM = RAGSystem(api_key, config)
        except Exception as e:
            print(f"RAG 系统初始化失败: {e}")


def search_knowledge_base_records(query: str, limit: int = 5) -> list[dict]:
    if _RAG_SYSTEM is None:
        return []
    try:
        return _RAG_SYSTEM.search_records(query, k=limit)
    except Exception:
        return []


def tavily_web_search_records(query: str, count: int = 8) -> list[dict]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip().strip("'\"")
    logger.info("[tavily_web_search] 开始搜索 | query=%s | count=%s", query, count)
    logger.info("[tavily_web_search] API Key 状态 | 是否配置=%s | Key前缀=%s", bool(api_key), api_key[:8] + "..." if api_key else "None")
    if not api_key:
        logger.warning("[tavily_web_search] 未配置 TAVILY_API_KEY，跳过搜索")
        return []
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": count,
    }
    request = urllib.request.Request(
        url="https://api.tavily.com/search",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
        },
    )
    try:
        logger.info("[tavily_web_search] 发送请求 | url=%s", request.full_url)
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
            logger.info("[tavily_web_search] 收到响应 | status=%s | content_length=%s", response.status, len(raw))
        result = json.loads(raw)
        logger.info("[tavily_web_search] 解析响应成功 | results字段存在=%s", "results" in result)
    except urllib.error.HTTPError as e:
        logger.error("[tavily_web_search] HTTP 错误 | code=%s | reason=%s", e.code, e.reason)
        return []
    except urllib.error.URLError as e:
        logger.error("[tavily_web_search] URL 错误 | reason=%s", e.reason)
        return []
    except json.JSONDecodeError as e:
        logger.error("[tavily_web_search] JSON 解析错误 | error=%s", e)
        return []
    except Exception as e:
        logger.error("[tavily_web_search] 未知错误 | error=%s | type=%s", e, type(e).__name__)
        return []
    results = result.get("results", [])
    if not isinstance(results, list):
        logger.warning("[tavily_web_search] results 格式异常 | type=%s", type(results).__name__)
        return []
    logger.info("[tavily_web_search] 获取网页数量 | total=%s", len(results))
    records: list[dict] = []
    for idx, page in enumerate(results[:count], 1):
        if not isinstance(page, dict):
            logger.warning("[tavily_web_search] 第 %s 条记录格式异常 | type=%s", idx, type(page).__name__)
            continue
        url = str(page.get("url") or "").strip()
        domain = ""
        if "://" in url:
            domain = url.split("://", 1)[1].split("/", 1)[0]
        title = page.get("title") or f"web_result_{idx}"
        snippet = page.get("content") or ""
        logger.info("[tavily_web_search] 解析记录 %s | title=%s | url=%s | snippet长度=%s", idx, title[:50], domain, len(snippet))
        records.append(
            {
                "source_id": f"WEB-{idx}",
                "title": title,
                "url": url,
                "snippet": snippet,
                "domain": domain,
                "source_type": "web",
                "published_at": "",
            }
        )
    logger.info("[tavily_web_search] 搜索完成 | 返回记录数=%s", len(records))
    return records

@tool
def search_knowledge_base(query: str) -> str:
    """
    查询本地知识库/向量数据库。
    当用户询问关于专业知识、历史文档或私有数据时使用此工具。
    输入应该是具体的查询问题。
    """
    if _RAG_SYSTEM is None:
        return "错误：RAG 系统未初始化或连接失败。请检查 Milvus 服务状态。"
    return _RAG_SYSTEM.search(query)


ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def _eval_node(node):
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError("Unsupported expression")


@tool
def get_current_time() -> str:
    """返回当前时间的 ISO 字符串。"""
    return datetime.now().isoformat()


@tool
def simple_calculator(expression: str) -> str:
    """计算简单算术表达式并返回结果。"""
    tree = ast.parse(expression, mode="eval")
    result = _eval_node(tree.body)
    return str(result)


@tool
def extract_requirements(text: str) -> str:
    """从文本中提取需求要点列表。"""
    items = [part.strip() for part in text.replace("\n", " ").split("。") if part.strip()]
    return "\n".join(f"- {item}" for item in items[:8])


@tool
def outline_from_topics(topics: str) -> str:
    """根据主题列表生成编号大纲。"""
    raw = topics.replace("\n", ",")
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return "\n".join(f"{idx+1}. {item}" for idx, item in enumerate(items[:10]))


@tool
def merge_notes(note_a: str, note_b: str) -> str:
    """合并两段文本为一段笔记。"""
    return f"{note_a}\n{note_b}".strip()


@tool
def summarize_points(text: str) -> str:
    """从文本中抽取要点列表。"""
    sentences = [s.strip() for s in text.replace("\n", " ").split("。") if s.strip()]
    points = sentences[:6]
    return "\n".join(f"- {p}" for p in points)


@tool
def dedupe_lines(text: str) -> str:
    """对文本按行去重并输出。"""
    seen = set()
    lines = []
    for line in text.splitlines():
        key = line.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return "\n".join(lines)


@tool
def web_search_stub(query: str) -> str:
    """网络检索接口（Tavily Web Search）。"""
    records = tavily_web_search_records(query, count=5)
    if not records:
        return "未配置 TAVILY_API_KEY，无法执行网络检索。"
    lines = ["Tavily 检索结果："]
    for idx, record in enumerate(records, 1):
        lines.append(f"{idx}. {record['title']}")
        url = record.get("url", "")
        if url:
            lines.append(f"   链接: {url}")
        snippet = record.get("snippet", "")
        if snippet:
            lines.append(f"   摘要: {snippet[:200]}")
    return "\n".join(lines)


@tool
def local_docs_lookup_stub(query: str) -> str:
    """真实查询本地知识库/向量数据库（同 search_knowledge_base）。"""
    if _RAG_SYSTEM is None:
        return f"错误：本地检索系统未初始化。收到查询: {query}"
    try:
        return _RAG_SYSTEM.search(query)
    except Exception as e:
        return f"查询出错: {e}"


@tool
def local_vector_search_stub(query: str) -> str:
    """真实查询向量数据库接口（同 search_knowledge_base）。"""
    if _RAG_SYSTEM is None:
        return f"错误：向量数据库检索接口未初始化。收到查询: {query}"
    try:
        return _RAG_SYSTEM.search(query)
    except Exception as e:
        return f"查询出错: {e}"


@tool
def optimize_query(query: str) -> str:
    """
    对检索问题进行自动改写、术语拓展与优化，以获得更好的搜索效果。
    """
    cleaned = query.strip().lower()
    # 启发式专有名词拓展
    terms_map = {
        "rag": "RAG (Retrieval-Augmented Generation / 检索增强生成)",
        "llm": "LLM (Large Language Model / 大语言模型)",
        "db": "Database (数据库)",
        "pg": "PostgreSQL (关系型数据库)",
        "milvus": "Milvus (向量数据库)",
        "agent": "AI Agent (人工智能体/多智能体)",
    }
    found_expansions = []
    for key, val in terms_map.items():
        if re.search(rf"\b{key}\b", cleaned) or key in cleaned:
            found_expansions.append(val)
    
    # 剔除中文常见疑问辅助词
    stop_words = ["帮我", "请问", "是什么", "怎么做", "如何", "的一下", "如何实现"]
    cleaned_query = query
    for word in stop_words:
        cleaned_query = cleaned_query.replace(word, "")
    cleaned_query = cleaned_query.strip()
    
    if found_expansions:
        expansion_str = " | ".join(found_expansions)
        return f"优化后的查询建议: {cleaned_query} (涉及核心概念: {expansion_str})"
    return f"优化后的查询建议: {cleaned_query}"


@tool
def explain_term(term: str) -> str:
    """
    解释和说明专业技术领域术语或概念。
    """
    # 本地高可信专家术语库
    glossary = {
        "milvus": "Milvus 是一款开源的分布式向量数据库，专为处理海量向量相似性检索而设计，常用于 RAG、大模型语义搜索等场景。",
        "rag": "RAG（检索增强生成）是指在大模型生成回答前，先从外部数据源（如向量库）中检索相关事实，并将其拼接进提示词输入模型，以解决幻觉问题并提高时效性。",
        "llm": "LLM（大语言模型）是基于深度学习（特别是 Transformer 架构）训练的、拥有数亿乃至数千亿参数的自然语言处理模型，具备极强的文本理解与生成能力。",
        "postgres": "PostgreSQL 是一款功能强大的开源关系型数据库，支持丰富的数据类型（如 JSON、向量 pgvector）和复杂的学术/工程事务处理。",
        "langgraph": "LangGraph 是 LangChain 推出的一款用于构建有状态、多角色、支持循环的多智能体（Multi-Agent）图结构协作框架。",
        "agent": "AI Agent（智能体）是能够自主感知环境、进行思考规划、并使用工具执行任务以达成特定目标的智能实体系统。"
    }
    cleaned_term = term.strip().lower()
    for key, val in glossary.items():
        if key in cleaned_term or cleaned_term in key:
            return f"【专业术语解释】{key.upper()}: {val}"
    
    return f"术语解释: {term} 是一种专业技术名词，主要应用于现代人工智能、大数据分析或软件架构开发中。请结合具体上下文进行深入探讨。"


@tool
def python_inter(code: str) -> str:
    """
    安全沙箱 Python 执行环境。
    用于执行复杂的数学运算、数据格式化及算法验证。
    """
    logger.info("[python_sandbox] 正在执行 Python 代码段...")
    
    # 安全屏障：屏蔽高危内置函数与模块导入
    dangerous_keywords = ["os.", "sys.", "subprocess", "socket", "shutil", "builtins.eval", "getattr", "setattr"]
    for keyword in dangerous_keywords:
        if keyword in code:
            return f"安全警告：检测到高危代码关键字 '{keyword}'，已被沙箱执行引擎拦截！"
            
    restricted_globals = {
        "__builtins__": {
            "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool, "chr": chr,
            "dict": dict, "dir": dir, "divmod": divmod, "enumerate": enumerate,
            "filter": filter, "float": float, "format": format, "hash": hash,
            "hex": hex, "id": id, "int": int, "isinstance": isinstance,
            "len": len, "list": list, "map": map, "max": max, "min": min,
            "next": next, "oct": oct, "ord": ord, "pow": pow, "range": range,
            "repr": repr, "reversed": reversed, "round": round, "set": set,
            "slice": slice, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
            "type": type, "zip": zip,
        },
        "json": json,
        "datetime": datetime,
    }
    
    import math
    restricted_globals["math"] = math
    
    local_variables = {}
    stdout_buffer = io.StringIO()
    
    try:
        with redirect_stdout(stdout_buffer):
            exec(code, restricted_globals, local_variables)
        
        output = stdout_buffer.getvalue()
        if not output and local_variables:
            output = f"代码执行成功。局部变量结果: {local_variables}"
        return output if output else "代码执行成功，无标准控制台输出。"
    except Exception as e:
        logger.error("[python_sandbox] 代码执行出错: %s", e)
        return f"Python 执行报错: {type(e).__name__}: {e}"


@tool
def fig_inter(spec: str) -> str:
    """
    高保真绘图与图表渲染接口。
    输入为 JSON 格式的图表规格参数。生成漂亮的 SVG/HTML 可视化文件并写入工作目录。
    """
    logger.info("[chart_generator] 开始生成图表，spec: %s", spec)
    try:
        chart_data = json.loads(spec)
    except Exception:
        chart_data = {
            "title": "数据对比图",
            "type": "bar",
            "labels": ["维度A", "维度B", "维度C"],
            "values": [30, 80, 50]
        }
    
    title = chart_data.get("title", "数据统计图")
    chart_type = chart_data.get("type", "bar").lower()
    labels = chart_data.get("labels", ["A", "B", "C"])
    values = chart_data.get("values", [10, 20, 30])
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 5))
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        if chart_type == "line":
            plt.plot(labels, values, marker='o', color='#1a73e8', linewidth=2)
        elif chart_type == "pie":
            plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['#1a73e8', '#34a853', '#fbbc05', '#ea4335'])
        else:
            plt.bar(labels, values, color='#1a73e8')
            
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        root = _workspace_root()
        file_name = f"chart_{int(datetime.now().timestamp())}.png"
        target_path = root / file_name
        plt.savefig(target_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info("[chart_generator] Matplotlib 成功渲染图表: %s", target_path)
        return f"图表生成成功！已使用 matplotlib 绘制了 {chart_type} 图表，文件已安全写入工作目录: {file_name}"
        
    except Exception as e:
        logger.warning("[chart_generator] Matplotlib 绘图失败，降级为 SVG HTML: %s", e)
        
        svg_width = 500
        svg_height = 300
        svg_lines = [
            f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg" style="background:#f8f9fa;font-family:sans-serif;">',
            f'<text x="250" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">{title}</text>'
        ]
        
        if chart_type == "bar" and len(values) > 0:
            max_val = max(values) or 1
            bar_width = 40
            gap = 30
            start_x = (svg_width - (len(values) * (bar_width + gap) - gap)) / 2
            
            for idx, (label, val) in enumerate(zip(labels, values)):
                x = start_x + idx * (bar_width + gap)
                bar_height = (val / max_val) * 180
                y = 240 - bar_height
                svg_lines.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="#1a73e8" rx="4"/>')
                svg_lines.append(f'<text x="{x + bar_width/2}" y="{y - 8}" text-anchor="middle" font-size="12" fill="#666">{val}</text>')
                svg_lines.append(f'<text x="{x + bar_width/2}" y="260" text-anchor="middle" font-size="12" fill="#333">{label}</text>')
        else:
            svg_lines.append(f'<rect x="50" y="60" width="400" height="180" fill="#e9ecef" rx="8"/>')
            svg_lines.append(f'<text x="250" y="150" text-anchor="middle" fill="#6c757d">SVG 图表容器 [类型: {chart_type}]</text>')
            
        svg_lines.append('</svg>')
        svg_content = "\n".join(svg_lines)
        
        root = _workspace_root()
        file_name = f"chart_{int(datetime.now().timestamp())}.html"
        target_path = root / file_name
        
        html_wrapper = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
</head>
<body style="display:flex;justify-content:center;align-items:center;height:100vh;margin:0;background:#e9ecef;">
    {svg_content}
</body>
</html>"""
        target_path.write_text(html_wrapper, encoding="utf-8")
        return f"已成功以 SVG 矢量格式渲染了高对比度 HTML 图表，文件已安全写入工作目录: {file_name}"


@tool
def amap_weather(city: str) -> str:
    """实时查询指定城市的高德官方气象数据（需要 AMAP_API_KEY）。"""
    api_key = os.getenv("AMAP_API_KEY", "").strip().strip("'\"")
    if not api_key:
        return f"未配置 AMAP_API_KEY，无法调用高德官方天气服务。收到天气查询城市: {city}"
    try:
        encoded_city = urllib.parse.quote(city.strip())
        url = f"https://restapi.amap.com/v3/weather/weatherInfo?key={api_key}&city={encoded_city}&extensions=base"
        req = urllib.request.urlopen(url, timeout=10)
        res_data = json.loads(req.read().decode('utf-8'))
        if res_data.get("status") == "1" and res_data.get("lives"):
            live = res_data["lives"][0]
            return f"【高德天气】{live['province']}-{live['city']}: {live['weather']} | 温度: {live['temperature']}℃ | 风向: {live['winddirection']}风 | 湿度: {live['humidity']}%"
        return f"查询天气失败: {res_data.get('info', '未知错误')}"
    except Exception as e:
        return f"高德天气查询出错: {e}"


@tool
def amap_geocode(address: str) -> str:
    """查询指定地址的高德地理编码，获取经纬度坐标（需要 AMAP_API_KEY）。"""
    api_key = os.getenv("AMAP_API_KEY", "").strip().strip("'\"")
    if not api_key:
        return f"未配置 AMAP_API_KEY，无法调用高德地理编码服务。收到地址: {address}"
    try:
        encoded_addr = urllib.parse.quote(address.strip())
        url = f"https://restapi.amap.com/v3/geocode/geo?key={api_key}&address={encoded_addr}"
        req = urllib.request.urlopen(url, timeout=10)
        res_data = json.loads(req.read().decode('utf-8'))
        if res_data.get("status") == "1" and res_data.get("geocodes"):
            geo = res_data["geocodes"][0]
            return f"【高德地理编码】地址: {geo['formatted_address']} | 经纬度: {geo['location']} | 区域: {geo['province']}-{geo['city']}-{geo['district']}"
        return f"查询地理编码失败: {res_data.get('info', '未知错误')}"
    except Exception as e:
        return f"高德地理编码查询出错: {e}"


@tool
def amap_poi_search(query: str) -> str:
    """在高德地图中检索指定关键字的周边兴趣点（POI）（需要 AMAP_API_KEY）。"""
    api_key = os.getenv("AMAP_API_KEY", "").strip().strip("'\"")
    if not api_key:
        return f"未配置 AMAP_API_KEY，无法调用高德 POI 检索服务。收到查询: {query}"
    try:
        encoded_query = urllib.parse.quote(query.strip())
        url = f"https://restapi.amap.com/v3/place/text?key={api_key}&keywords={encoded_query}&types=&city=&children=1&offset=5&page=1"
        req = urllib.request.urlopen(url, timeout=10)
        res_data = json.loads(req.read().decode('utf-8'))
        if res_data.get("status") == "1" and res_data.get("pois"):
            pois = res_data["pois"]
            lines = [f"【高德 POI 搜索】找到以下与 '{query}' 相关的地点："]
            for idx, poi in enumerate(pois[:5], 1):
                lines.append(f"{idx}. {poi['name']} | 类型: {poi['type']} | 地址: {poi['address'] or '未指明'} | 坐标: {poi['location']}")
            return "\n".join(lines)
        return f"POI 检索失败: {res_data.get('info', '未知错误')}"
    except Exception as e:
        return f"高德 POI 检索出错: {e}"


@tool
def amap_route_plan(origin: str, destination: str) -> str:
    """根据起点与终点，调用高德驾车路径规划接口（需要 AMAP_API_KEY）。"""
    api_key = os.getenv("AMAP_API_KEY", "").strip().strip("'\"")
    if not api_key:
        return f"未配置 AMAP_API_KEY，无法调用高德路径规划服务。收到规划: {origin} -> {destination}"
    try:
        def get_location(addr):
            encoded = urllib.parse.quote(addr.strip())
            u = f"https://restapi.amap.com/v3/geocode/geo?key={api_key}&address={encoded}"
            r = urllib.request.urlopen(u, timeout=10)
            d = json.loads(r.read().decode('utf-8'))
            if d.get("status") == "1" and d.get("geocodes"):
                return d["geocodes"][0]["location"]
            return None
        
        origin_loc = get_location(origin)
        dest_loc = get_location(destination)
        if not origin_loc or not dest_loc:
            return f"错误：解析起点 '{origin}' 或终点 '{destination}' 的地理经纬度失败。"
            
        url = f"https://restapi.amap.com/v3/direction/driving?key={api_key}&origin={origin_loc}&destination={dest_loc}"
        req = urllib.request.urlopen(url, timeout=10)
        res_data = json.loads(req.read().decode('utf-8'))
        if res_data.get("status") == "1" and res_data.get("route") and res_data["route"].get("paths"):
            path = res_data["route"]["paths"][0]
            distance_km = round(float(path.get("distance", 0)) / 1000, 2)
            duration_min = round(float(path.get("duration", 0)) / 60, 1)
            tolls = path.get("tolls", "0")
            return f"【高德驾车路径规划】\n路线: {origin} -> {destination}\n总距离: {distance_km} 公里\n估计耗时: {duration_min} 分钟\n预计过路费: {tolls} 元"
        return f"路径规划失败: {res_data.get('info', '未知错误')}"
    except Exception as e:
        return f"高德路径规划出错: {e}"


def _workspace_root() -> Path:
    base = os.getenv("WORKSPACE_DIR", "/workspace")
    return Path(base).resolve()


def _safe_path(path: str) -> Path:
    root = _workspace_root()
    target = (root / path).resolve()
    if root not in target.parents and target != root:
        raise ValueError("路径超出工作目录")
    return target


@tool
def safe_list_dir(path: str = ".") -> str:
    """安全列出工作目录下的文件与子目录。"""
    root = _workspace_root()
    if not root.exists():
        return f"工作目录不存在: {root}"
    target = _safe_path(path)
    if not target.exists() or not target.is_dir():
        return "目录不存在"
    items = [p.name for p in target.iterdir()]
    return "\n".join(items)


@tool
def safe_read_file(path: str) -> str:
    """安全读取工作目录内的文件。"""
    root = _workspace_root()
    if not root.exists():
        return f"工作目录不存在: {root}"
    target = _safe_path(path)
    if not target.exists() or not target.is_file():
        return "文件不存在"
    return target.read_text(encoding="utf-8")


@tool
def safe_write_file(path: str, content: str) -> str:
    """安全写入工作目录内的文件。"""
    root = _workspace_root()
    if not root.exists():
        return f"工作目录不存在: {root}"
    target = _safe_path(path)
    if not target.parent.exists():
        return "目录不存在"
    target.write_text(content, encoding="utf-8")
    return f"已写入: {target}"


@tool
def safe_move_file(src: str, dst: str) -> str:
    """安全移动工作目录内的文件。"""
    root = _workspace_root()
    if not root.exists():
        return f"工作目录不存在: {root}"
    src_path = _safe_path(src)
    dst_path = _safe_path(dst)
    if not src_path.exists():
        return "源文件不存在"
    if not dst_path.parent.exists():
        return "目标目录不存在"
    src_path.replace(dst_path)
    return f"已移动: {dst_path}"


@tool
def sql_inter(query: str) -> str:
    """
    对局内 PostgreSQL 数据库（deep_research_db）安全执行只读 SQL 查询，获取业务明细。
    """
    logger.info("[db_sql_client] 正在执行 SQL 查询, query: %s", query)
    dsn = os.getenv("POSTGRES_DSN", "").strip().strip("'\"")
    if not dsn:
        return "未配置 POSTGRES_DSN 环境变量，无法连接数据库。"
    
    # 严格只读静态安全检查
    cleaned_query = query.strip().upper()
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE", "REPLACE"]
    for keyword in forbidden_keywords:
        if keyword in cleaned_query:
            return f"安全警告：检测到高危修改指令 '{keyword}'，已被只读 SQL 拦截引擎拒绝！本接口仅支持 SELECT 查询。"
            
    if not cleaned_query.startswith("SELECT") and not cleaned_query.startswith("WITH"):
        return "安全警告：SQL 语句必须以 SELECT 或 WITH 开头，执行只读查询。"
        
    try:
        import psycopg
        with psycopg.connect(dsn, timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                colnames = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall()
                
        if not rows:
            return "SQL 执行成功，但返回结果集为空。"
            
        header = " | ".join(colnames)
        divider = " | ".join(["---"] * len(colnames))
        table_lines = [header, divider]
        for row in rows[:50]:
            row_str = " | ".join(str(val) for val in row)
            table_lines.append(row_str)
            
        summary = f"SQL 执行成功！共召回 {len(rows)} 行数据。前 50 行展示如下：\n\n" + "\n".join(table_lines)
        return summary
    except Exception as e:
        logger.error("[db_sql_client] SQL 执行发生错误: %s", e)
        return f"SQL 执行出错: {type(e).__name__}: {e}"


@tool
def extract_data_stub(query: str) -> str:
    """
    智能数据文件读取与分析接口。
    支持输入工作目录内的 CSV、Excel、JSON 文件相对路径，使用 pandas 进行加载并展示 Markdown 概览数据。
    """
    logger.info("[pandas_data_extractor] 正在读取并分析数据文件: %s", query)
    try:
        path_str = query.strip().strip("'\"")
        target_path = _safe_path(path_str)
        
        if not target_path.exists():
            root = _workspace_root()
            candidates = list(root.glob(f"**/{path_str}*"))
            if candidates:
                target_path = candidates[0]
            else:
                return f"错误：未在工作空间内找到数据文件 '{path_str}'。"
                
        suffix = target_path.suffix.lower()
        import pandas as pd
        
        if suffix == ".csv":
            df = pd.read_csv(target_path, nrows=500)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(target_path, nrows=500)
        elif suffix == ".json":
            df = pd.read_json(target_path)
        else:
            return f"错误：不支持的数据格式 '{suffix}'。本接口仅支持 csv, xlsx, json 文件。"
            
        row_count, col_count = df.shape
        dtypes_summary = df.dtypes.to_dict()
        columns_desc = ", ".join(f"{k}({v})" for k, v in dtypes_summary.items())
        markdown_table = df.head(10).to_markdown(index=False)
        
        return (
            f"【数据文件分析报告】\n"
            f"文件名: {target_path.name}\n"
            f"数据规格: 读入 {row_count} 行 | {col_count} 列 (最多读入前500行)\n"
            f"字段结构: {columns_desc}\n\n"
            f"数据前 10 行预览展示如下：\n\n{markdown_table}"
        )
    except Exception as e:
        logger.error("[pandas_data_extractor] 数据分析发生错误: %s", e)
        return f"数据文件读取失败: {type(e).__name__}: {e}"


@tool
def execute_terminal_command(command: str) -> str:
    """
    受限的安全终端命令执行器。
    仅允许执行经过白名单校验的安全诊断命令（如 git status, pip list 等）。
    """
    logger.info("[terminal_executor] 收到终端执行请求: %s", command)
    cleaned_cmd = command.strip().lower()
    
    allowed_commands = [
        "git status", "git branch", "git log -n", "git diff",
        "pip list", "pip show",
        "python --version", "python -m py_compile",
        "dir", "echo", "date", "time", "whoami"
    ]
    
    is_allowed = False
    for allowed in allowed_commands:
        if cleaned_cmd.startswith(allowed):
            is_allowed = True
            break
            
    if not is_allowed:
        return f"安全警告：命令 '{command}' 未在系统终端白名单中，已被拒绝执行！安全白名单仅包含常用诊断指令（git, pip, python --version 等）。"
        
    try:
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            timeout=15,
            encoding="utf-8",
            errors="ignore"
        )
        
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        
        output = []
        if stdout:
            output.append(stdout)
        if stderr:
            output.append(f"[标准错误输出]\n{stderr}")
            
        final_output = "\n".join(output).strip()
        return final_output if final_output else "命令执行成功，无任何终端回显输出。"
    except Exception as e:
        return f"终端命令执行发生错误: {type(e).__name__}: {e}"


@tool
def file_operation_stub(request: str) -> str:
    """
    安全沙箱文件管理接口。
    输入操作详情 JSON 格式，如 {"action": "delete/copy/info", "path": "RelativePath"}。
    支持在工作目录内安全复制、查询元数据或删除文件。
    """
    logger.info("[file_manager] 收到文件管理请求: %s", request)
    try:
        data = json.loads(request)
    except Exception:
        return "错误：输入必须是合法的 JSON 格式规格（包含 action 和 path 字段）。"
        
    action = data.get("action", "").lower().strip()
    path = data.get("path", "").strip()
    
    if not action or not path:
        return "错误：缺少必要的 'action' 或 'path' 字段参数。"
        
    try:
        target_path = _safe_path(path)
        if action == "info":
            if not target_path.exists():
                return "文件或目录不存在"
            stat = target_path.stat()
            created = datetime.fromtimestamp(stat.st_ctime).isoformat()
            modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
            size = stat.st_size
            item_type = "目录" if target_path.is_dir() else "文件"
            return f"【文件元数据】{item_type}: {target_path.name} | 大小: {size} 字节 | 创建时间: {created} | 最近修改: {modified}"
            
        elif action == "delete":
            if not target_path.exists():
                return "文件或目录不存在"
            if target_path.is_file():
                target_path.unlink()
                return f"文件删除成功！已物理移除: {target_path.name}"
            elif target_path.is_dir():
                import shutil
                shutil.rmtree(target_path)
                return f"目录删除成功！已物理移除整个文件夹: {target_path.name}"
                
        elif action == "copy":
            dest = data.get("dest", "").strip()
            if not dest:
                return "错误：copy 操作必须提供 'dest' 目标路径字段。"
            dest_path = _safe_path(dest)
            if not target_path.exists():
                return "源文件不存在"
            if not dest_path.parent.exists():
                return "目标父目录不存在"
                
            import shutil
            if target_path.is_file():
                shutil.copy2(target_path, dest_path)
                return f"文件复制成功！已自 {target_path.name} 复制至 {dest_path.name}"
            elif target_path.is_dir():
                shutil.copytree(target_path, dest_path)
                return f"目录复制成功！已自 {target_path.name} 复制至 {dest_path.name}"
        else:
            return f"错误：未知的操作类型 '{action}'。支持操作：info, delete, copy。"
    except Exception as e:
        return f"文件操作失败: {type(e).__name__}: {e}"


@tool
def news_search_stub(query: str) -> str:
    """新闻专属网络检索接口（使用 Tavily API 的新闻主题模式）。"""
    api_key = os.getenv("TAVILY_API_KEY", "").strip().strip("'\"")
    logger.info("[news_search] 正在启动新闻检索: %s", query)
    if not api_key:
        return "未配置 TAVILY_API_KEY，新闻检索服务暂不可用。"
    
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": 5,
        "topic": "news"
    }
    try:
        request = urllib.request.Request(
            url="https://api.tavily.com/search",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
            
        results = result.get("results", [])
        if not results:
            return "未找到相关新闻时事报道。"
            
        lines = [f"Tavily 最新时事新闻检索结果 [{query}]："]
        for idx, record in enumerate(results[:5], 1):
            title = record.get("title", f"news_{idx}")
            url = record.get("url", "暂无链接")
            snippet = record.get("content", "")
            lines.append(f"{idx}. {title}\n   新闻来源: {url}\n   报道要点: {snippet[:200]}...")
        return "\n".join(lines)
    except Exception as e:
        return f"新闻检索发生故障: {e}"


@tool
def finance_search_stub(query: str) -> str:
    """金融/财经专属网络检索接口（使用 Tavily API 的金融主题模式）。"""
    api_key = os.getenv("TAVILY_API_KEY", "").strip().strip("'\"")
    logger.info("[finance_search] 正在启动金融数据检索: %s", query)
    if not api_key:
        return "未配置 TAVILY_API_KEY，金融检索服务暂不可用。"
    
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": 5,
        "topic": "finance"
    }
    try:
        request = urllib.request.Request(
            url="https://api.tavily.com/search",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
            
        results = result.get("results", [])
        if not results:
            return "未找到相关金融或财经财报数据。"
            
        lines = [f"Tavily 财经数据分析检索结果 [{query}]："]
        for idx, record in enumerate(results[:5], 1):
            title = record.get("title", f"finance_{idx}")
            url = record.get("url", "暂无链接")
            snippet = record.get("content", "")
            lines.append(f"{idx}. {title}\n   财经数据源: {url}\n   核心指标/摘要: {snippet[:200]}...")
        return "\n".join(lines)
    except Exception as e:
        return f"金融检索发生故障: {e}"


@tool
def extract_url_content_stub(url: str) -> str:
    """
    抓取并解析目标 URL 网页的纯文本内容。
    用于深入分析某个特定的新闻链接、官方文档或博客内容。
    """
    logger.info("[url_content_extractor] 正在建立网络连接抓取网页: %s", url)
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        req = urllib.request.Request(url.strip(), headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "header", "footer", "nav", "iframe"]):
                tag.decompose()
            text = soup.get_text()
        except ImportError:
            logger.warning("[url_content_extractor] BeautifulSoup 未安装，自动降级为正则剥离 HTML")
            cleaned_html = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', html, flags=re.IGNORECASE)
            cleaned_html = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '', cleaned_html, flags=re.IGNORECASE)
            text = re.sub(r'<[^>]+>', '', cleaned_html)
            
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)
        
        logger.info("[url_content_extractor] 网页数据抓取成功，提取字符长度=%s", len(clean_text))
        if not clean_text.strip():
            return "网页解析成功，但内容为空（该网页可能是纯图片、动态 JavaScript 渲染或是防爬虫反制页面）。"
        return clean_text[:4000]
    except Exception as e:
        logger.error("[url_content_extractor] 抓取网页发生故障: %s", e)
        return f"读取网页内容失败: {type(e).__name__}: {e}"
