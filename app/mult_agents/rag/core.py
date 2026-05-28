import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import connections, utility

logger = logging.getLogger(__name__)


# 使用 langchain-milvus 新包（类名是 Milvus，不是 MilvusVectorStore）
try:
    from langchain_milvus import Milvus as _MilvusVectorStore
    _MILVUS_BACKEND = "langchain_milvus"
except ImportError:
    from langchain_community.vectorstores import Milvus as _MilvusVectorStore
    _MILVUS_BACKEND = "langchain_community"


@dataclass(frozen=True)
class RAGConfig:
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    collection_name: str = "mult_agent_knowledge"
    embedding_model: str = "text-embedding-v1"
    chunk_size: int = 500
    chunk_overlap: int = 50


class RAGSystem:
    def __init__(self, api_key: str, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.api_key = api_key
        self.embeddings = DashScopeEmbeddings(
            model=self.config.embedding_model,
            dashscope_api_key=self.api_key,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        self._connect_to_milvus()
        self.vectorstore = _MilvusVectorStore(
            embedding_function=self.embeddings,
            collection_name=self.config.collection_name,
            connection_args={"uri": f"http://{self.config.milvus_host}:{self.config.milvus_port}"},
            auto_id=True,
        )
        logger.info("RAG backend=%s | collection=%s", _MILVUS_BACKEND, self.config.collection_name)

    def _connect_to_milvus(self) -> None:
        try:
            connections.connect(
                alias="default",
                host=self.config.milvus_host,
                port=self.config.milvus_port,
            )
        except Exception as exc:
            logger.error("连接 Milvus 失败: %s", exc)

    def search(self, query: str, k: int = 3) -> str:
        try:
            records = self.search_records(query, k=k)
            if not records:
                return "未找到相关信息。"
            lines: list[str] = ["检索到的相关信息："]
            for idx, record in enumerate(records, 1):
                lines.append(f"{idx}. {record['snippet']}")
                lines.append(f"   (来源: {record['doc_id']})")
            return "\n".join(lines)
        except Exception as exc:
            logger.error("检索失败: %s", exc)
            return f"检索过程中发生错误: {str(exc)}"

    def search_records(self, query: str, k: int = 5) -> list[dict]:
        if not utility.has_collection(self.config.collection_name):
            return []
        import json
        
        # 1. 向量相似度精准匹配，召回最相关的核心 Chunks
        docs = self.vectorstore.similarity_search(query, k=k)
        records: list[dict] = []
        
        # 提取动态主键的工具函数，自适应不同版本的 Milvus 字段定义
        pk_field = "pk"
        if hasattr(self.vectorstore, "schema") and self.vectorstore.schema and hasattr(self.vectorstore.schema, "primary_field"):
            if self.vectorstore.schema.primary_field:
                pk_field = self.vectorstore.schema.primary_field.name

        for idx, doc in enumerate(docs, 1):
            metadata = doc.metadata or {}
            source = str(metadata.get("source") or "").strip()
            title = Path(source).name if source else f"本地知识片段-{idx}"
            
            # 获取当前分片的物理主键（auto_id 自动生成的递增 ID）
            pk = metadata.get(pk_field) or getattr(doc, "id", None)
            
            # 初始化拼接后的正文（默认为当前命中的 500字 Chunks）
            sutured_content = doc.page_content
            
            # 2. 【核心进化：Window Buffer 上下文窗口拼接】
            # 如果主键是可进行数学加减的整数，并且能通过底层连接查询相邻分片
            if isinstance(pk, int) and hasattr(self.vectorstore, "col") and self.vectorstore.col is not None:
                try:
                    # 查询该分片物理上相邻的 前一个(pk - 1) 与 后一个(pk + 1) Chunks
                    expr = f"{pk_field} in [{pk - 1}, {pk + 1}]"
                    res = self.vectorstore.col.query(expr=expr, output_fields=[pk_field, "text", "metadata"])
                    
                    prev_text = ""
                    next_text = ""
                    
                    for r in res:
                        r_pk = r.get(pk_field)
                        r_text = r.get("text", "").strip()
                        r_meta = r.get("metadata", {})
                        
                        # 在某些 Milvus / LangChain 版本中，metadata 返回的是未经解析的 JSON 字符串
                        if isinstance(r_meta, str):
                            try:
                                r_meta = json.loads(r_meta)
                            except Exception:
                                r_meta = {}
                                
                        r_source = str(r_meta.get("source") or "").strip()
                        
                        # 【数据一致性防线】：只有当邻近分片的物理来源文件与当前分片完全一致时，才允许拼接
                        if r_source == source and r_text:
                            if r_pk == pk - 1:
                                prev_text = r_text
                            elif r_pk == pk + 1:
                                next_text = r_text
                                
                    # 用换行符和虚线分隔符拼接，展现完美前因后果，不污染语义
                    sutured_parts = []
                    if prev_text:
                        sutured_parts.append(prev_text)
                        sutured_parts.append("--- [以上为前置相邻上下文] ---")
                    
                    sutured_parts.append(doc.page_content)
                    
                    if next_text:
                        sutured_parts.append("--- [以下为后置相邻上下文] ---")
                        sutured_parts.append(next_text)
                        
                    sutured_content = "\n".join(sutured_parts).strip()
                    logger.debug("[rag] window buffer retrieval success | pk=%d | prev_len=%d next_len=%d", pk, len(prev_text), len(next_text))
                except Exception as query_exc:
                    logger.warning("[rag] failed to retrieve neighboring chunks: %s", query_exc)
            
            records.append(
                {
                    "source_id": f"LOC-{idx}",
                    "doc_id": source,
                    "title": title,
                    "snippet": sutured_content, # 将缝合后的上下文交付给智能体
                    "source_type": "local",
                    "metadata": metadata,
                }
            )
        return records

    def add_documents(self, documents: list[Document]) -> int:
        self.vectorstore.add_documents(documents)
        return len(documents)

    def ingest_text(self, text: str, source: str) -> int:
        docs = self.text_splitter.create_documents([text], metadatas=[{"source": source}])
        return self.add_documents(docs)

    def ingest_paths(self, paths: Iterable[Path]) -> int:
        total = 0
        for path in paths:
            text = path.read_text(encoding="utf-8")
            total += self.ingest_text(text, source=str(path))
        return total
