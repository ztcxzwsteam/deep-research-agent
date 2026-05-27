"""
短期记忆模块

基于 LangGraph Checkpoint 实现，管理当前对话线程的上下文
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from .base import BaseMemory, MemoryEntry, MemoryType

logger = logging.getLogger("mult_agents.memory")


class ConversationBuffer:
    """
    对话缓冲区
    
    管理单轮对话的消息历史，支持窗口裁剪和摘要生成
    """
    
    def __init__(
        self,
        max_messages: int = 20,
        max_tokens: int = 4000,
        summary_threshold: int = 10,
        summary_llm: Optional[Any] = None,
    ):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.summary_threshold = summary_threshold
        self.messages: List[BaseMessage] = []
        self.summary: Optional[str] = None
        self.token_count: int = 0
        self.summary_llm = summary_llm
    
    def add_message(self, message: BaseMessage) -> None:
        """添加消息到缓冲区"""
        if not getattr(message, "id", None):
            from uuid import uuid4
            message.id = str(uuid4())
        self.messages.append(message)
        self._update_token_count()
        
        # 如果超过阈值，触发裁剪或摘要
        if len(self.messages) > self.max_messages:
            self._compress_messages()
    
    def add_messages(self, messages: List[BaseMessage]) -> None:
        """批量添加消息"""
        for msg in messages:
            self.add_message(msg)
    
    def get_messages(
        self,
        include_summary: bool = True,
        last_n: Optional[int] = None
    ) -> List[BaseMessage]:
        """
        获取消息列表
        
        Args:
            include_summary: 是否包含历史摘要
            last_n: 只返回最近 N 条消息
            
        Returns:
            消息列表
        """
        result = []
        
        # 添加摘要作为系统消息
        if include_summary and self.summary:
            result.append(SystemMessage(content=f"历史对话摘要：{self.summary}"))
        
        # 添加实际消息
        messages_to_return = self.messages
        if last_n:
            messages_to_return = self.messages[-last_n:]
        
        result.extend(messages_to_return)
        return result
    
    def clear(self) -> None:
        """清空缓冲区"""
        self.messages = []
        self.summary = None
        self.token_count = 0
    
    def _update_token_count(self) -> None:
        """更新 token 计数（简化估算）"""
        # 简单估算：每个字符约 0.5 个 token
        total_chars = sum(len(str(msg.content)) for msg in self.messages)
        self.token_count = total_chars // 2
    
    def _compress_messages(self) -> None:
        """
        压缩消息历史
        
        策略：保留最近的消息，将旧消息生成摘要
        """
        if len(self.messages) <= self.summary_threshold:
            return
        
        # 保留最近的消息
        messages_to_summarize = self.messages[:-self.summary_threshold]
        self.messages = self.messages[-self.summary_threshold:]
        
        # 格式化待摘要的文本
        summary_parts = []
        for msg in messages_to_summarize:
            role = "用户" if isinstance(msg, HumanMessage) else "AI"
            summary_parts.append(f"{role}: {msg.content}")
        history_text = "\n".join(summary_parts)
        
        new_summary = ""
        is_llm_success = False
        # 如果提供了摘要模型，则调用大模型生成高质量递归摘要
        if self.summary_llm is not None:
            try:
                prompt = (
                    "你是对话压缩引擎。请在保留用户偏好、重要事实、任务进展和约束的前提下，进行递归摘要。\n"
                    f"已有摘要：{self.summary or '无'}\n"
                    "新增历史：\n"
                    f"{history_text}\n"
                    "请输出一段100-300字的紧凑中文摘要，不要包含多余的主观解释或格式包裹。"
                )
                response = self.summary_llm.invoke([HumanMessage(content=prompt)])
                new_summary = str(response.content).strip()
                is_llm_success = True
                logger.info("短期记忆：已使用大模型成功生成高质量历史摘要")
            except Exception as e:
                logger.error(f"短期记忆：调用大模型生成摘要出错: {e}，将降级到字符串拼接模式。")
        
        # 降级兜底方案：普通文本预览拼接
        if not new_summary:
            simple_parts = []
            for msg in messages_to_summarize:
                role = "用户" if isinstance(msg, HumanMessage) else "AI"
                content_preview = str(msg.content)[:100]
                simple_parts.append(f"{role}: {content_preview}...")
            new_summary = "\n".join(simple_parts)
            
        if self.summary and is_llm_success:
            # 如果是 LLM 成功生成的递归摘要，它已经整合了历史 self.summary，所以直接赋值
            self.summary = new_summary
        elif self.summary:
            # 如果是降级兜底方案（或者LLM调用失败），旧的摘要依然存在，需要以拼接的形式追加新的压缩内容，防止覆盖丢失
            self.summary = f"{self.summary}\n\n[更早的对话]\n{new_summary}"
        else:
            self.summary = new_summary
        
        logger.debug(f"消息历史已压缩，当前消息数: {len(self.messages)}")


class ShortTermMemory(BaseMemory):
    """
    短期记忆实现
    
    基于内存存储，管理当前线程的对话上下文和临时状态
    与 LangGraph 的 State 和 Checkpoint 集成
    """
    
    def __init__(
        self,
        ttl_seconds: int = 3600,  # 默认 1 小时过期
        max_threads: int = 100,
    ):
        super().__init__(MemoryType.SHORT_TERM)
        self.ttl_seconds = ttl_seconds
        self.max_threads = max_threads
        self.summary_llm = None
        
        # 存储结构: {thread_id: {"buffer": ConversationBuffer, "metadata": {}, "last_access": datetime}}
        self._storage: Dict[str, Dict[str, Any]] = {}
        self._checkpointer: Optional[BaseCheckpointSaver] = None
    
    def set_checkpointer(self, checkpointer: BaseCheckpointSaver) -> None:
        """设置 LangGraph Checkpoint 存储"""
        self._checkpointer = checkpointer
        
    def set_summary_llm(self, llm: Any) -> None:
        """设置对话历史摘要的大模型实例"""
        self.summary_llm = llm
    
    def get_or_create_buffer(self, thread_id: str) -> ConversationBuffer:
        """获取或创建对话缓冲区"""
        self._cleanup_expired()
        
        if thread_id not in self._storage:
            self._storage[thread_id] = {
                "buffer": ConversationBuffer(summary_llm=self.summary_llm),
                "metadata": {},
                "created_at": datetime.now(),
                "last_access": datetime.now(),
            }
            logger.debug(f"为新线程 {thread_id} 创建短期记忆缓冲区")
        else:
            self._storage[thread_id]["last_access"] = datetime.now()
            # 动态更新可能后设置的模型实例
            self._storage[thread_id]["buffer"].summary_llm = self.summary_llm
        
        return self._storage[thread_id]["buffer"]
    
    def add_message(
        self,
        thread_id: str,
        message: BaseMessage,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        添加消息到短期记忆
        
        Args:
            thread_id: 线程标识
            message: LangChain 消息对象
            metadata: 附加元数据
        """
        buffer = self.get_or_create_buffer(thread_id)
        buffer.add_message(message)
        
        if metadata:
            self._storage[thread_id]["metadata"].update(metadata)
        
        logger.debug(f"消息已添加到线程 {thread_id} 的短期记忆")
    
    def get_messages(
        self,
        thread_id: str,
        include_summary: bool = True,
        last_n: Optional[int] = None
    ) -> List[BaseMessage]:
        """
        获取指定线程的消息历史
        
        Args:
            thread_id: 线程标识
            include_summary: 是否包含历史摘要
            last_n: 只返回最近 N 条
            
        Returns:
            消息列表
        """
        if thread_id not in self._storage:
            return []
        
        self._storage[thread_id]["last_access"] = datetime.now()
        buffer = self._storage[thread_id]["buffer"]
        return buffer.get_messages(include_summary=include_summary, last_n=last_n)
    
    def get_thread_metadata(self, thread_id: str) -> Dict[str, Any]:
        """获取线程元数据"""
        if thread_id not in self._storage:
            return {}
        return self._storage[thread_id]["metadata"].copy()
    
    def update_thread_metadata(
        self,
        thread_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """更新线程元数据"""
        buffer = self.get_or_create_buffer(thread_id)
        self._storage[thread_id]["metadata"].update(metadata)
    
    def clear_thread(self, thread_id: str) -> bool:
        """清空指定线程的记忆"""
        if thread_id in self._storage:
            del self._storage[thread_id]
            logger.debug(f"线程 {thread_id} 的短期记忆已清空")
            return True
        return False
    
    def list_active_threads(self) -> List[str]:
        """列出所有活跃线程"""
        self._cleanup_expired()
        return list(self._storage.keys())
    
    # 实现 BaseMemory 抽象方法
    
    def save(self, entry: MemoryEntry) -> str:
        """保存记忆条目"""
        thread_id = entry.thread_id or "default"
        buffer = self.get_or_create_buffer(thread_id)
        
        # 将内容转换为消息
        if isinstance(entry.content, str):
            message = HumanMessage(content=entry.content)
        elif isinstance(entry.content, dict):
            content = entry.content.get("content", "")
            role = entry.content.get("role", "human")
            if role == "ai":
                message = AIMessage(content=content)
            else:
                message = HumanMessage(content=content)
        else:
            message = HumanMessage(content=str(entry.content))
        
        # 显式传递条目 ID
        message.id = entry.id
        
        buffer.add_message(message)
        
        # 更新元数据
        if entry.metadata:
            self._storage[thread_id]["metadata"].update(entry.metadata)
        
        return entry.id
    
    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """获取指定 ID 的记忆"""
        for thread_id, data in self._storage.items():
            buffer = data["buffer"]
            for msg in buffer.messages:
                if getattr(msg, "id", None) == memory_id:
                    role = "ai" if isinstance(msg, AIMessage) else "human"
                    return MemoryEntry(
                        id=msg.id,
                        content=msg.content,
                        memory_type=MemoryType.SHORT_TERM,
                        thread_id=thread_id,
                        metadata={"role": role},
                        created_at=data.get("created_at", datetime.now()),
                    )
        return None
    
    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        namespace: Optional[str] = None,
        limit: int = 5,
        **kwargs
    ) -> List[MemoryEntry]:
        """
        搜索短期记忆
        
        使用简单的相似度匹配对当前线程的消息进行相关度检索
        """
        thread_id = namespace or "default"
        messages = self.get_messages(thread_id, include_summary=False)
        
        from .utils import _simple_similarity
        from uuid import uuid4
        
        scored_entries = []
        for msg in messages:
            content_str = str(msg.content)
            sim = 1.0
            if query:
                sim = _simple_similarity(query, content_str)
                if sim == 0.0 and query.lower() in content_str.lower():
                    sim = 0.5
            
            if query and sim == 0.0:
                continue
                
            entry = MemoryEntry(
                id=getattr(msg, "id", None) or str(uuid4()),
                content=msg.content,
                memory_type=MemoryType.SHORT_TERM,
                thread_id=thread_id,
                user_id=user_id,
                metadata={"role": "ai" if isinstance(msg, AIMessage) else "human"},
            )
            scored_entries.append((entry, sim))
        
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scored_entries[:limit]]
    
    def delete(self, memory_id: str) -> bool:
        """删除指定记忆"""
        for thread_id, data in self._storage.items():
            buffer = data["buffer"]
            for i, msg in enumerate(buffer.messages):
                if getattr(msg, "id", None) == memory_id:
                    buffer.messages.pop(i)
                    buffer._update_token_count()
                    logger.debug(f"已从线程 {thread_id} 的短期记忆中删除了消息 {memory_id}")
                    return True
        return False
    
    def clear(
        self,
        user_id: Optional[str] = None,
        namespace: Optional[str] = None
    ) -> int:
        """清除短期记忆"""
        if namespace:
            # 清除指定线程
            if self.clear_thread(namespace):
                return 1
            return 0
        
        # 清除所有
        count = len(self._storage)
        self._storage.clear()
        logger.info(f"已清除所有短期记忆，共 {count} 个线程")
        return count
    
    def list_namespaces(self, user_id: Optional[str] = None) -> List[str]:
        """列出所有线程 ID 作为命名空间"""
        return self.list_active_threads()
    
    def _cleanup_expired(self) -> None:
        """清理过期的线程记忆"""
        now = datetime.now()
        expired_threads = []
        
        for thread_id, data in self._storage.items():
            last_access = data.get("last_access", data.get("created_at", now))
            if now - last_access > timedelta(seconds=self.ttl_seconds):
                expired_threads.append(thread_id)
        
        # 如果超过最大线程数，清理最旧的
        if len(self._storage) > self.max_threads:
            sorted_threads = sorted(
                self._storage.items(),
                key=lambda x: x[1].get("last_access", x[1].get("created_at", now))
            )
            threads_to_remove = len(self._storage) - self.max_threads
            expired_threads.extend([t[0] for t in sorted_threads[:threads_to_remove]])
        
        for thread_id in set(expired_threads):
            del self._storage[thread_id]
        
        if expired_threads:
            logger.debug(f"已清理 {len(set(expired_threads))} 个过期线程")
