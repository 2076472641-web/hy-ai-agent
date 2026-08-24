# -*- coding: utf-8 -*-
"""
对话记忆模块 - 提供会话缓存和历史管理
支持持久化到本地文件，避免进程重启丢失
"""
from __future__ import annotations

import os
import json
import time
import uuid
import threading
from typing import Any, Optional
from pathlib import Path

from config import get_config


class ConversationMemory:
    """
    对话缓存管理器
    - 支持多会话隔离
    - 支持持久化存储（JSON 文件）
    - 支持上下文压缩
    """

    def __init__(self, memory_dir: str = None):
        config = get_config()
        self.memory_dir = memory_dir or config.memory.memory_dir
        self.max_tokens = config.memory.max_context_tokens
        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()

        os.makedirs(self.memory_dir, exist_ok=True)
        self._load_sessions()

    def _session_file(self, session_id: str) -> str:
        return os.path.join(self.memory_dir, f"session_{session_id}.json")

    def _load_sessions(self) -> None:
        """加载所有会话文件"""
        if not os.path.exists(self.memory_dir):
            return
        for fname in os.listdir(self.memory_dir):
            if fname.startswith("session_") and fname.endswith(".json"):
                try:
                    with open(os.path.join(self.memory_dir, fname), "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._sessions[data["session_id"]] = data
                except (json.JSONDecodeError, KeyError):
                    pass

    def _save_session(self, session_id: str) -> None:
        """保存会话到文件"""
        session = self._sessions.get(session_id)
        if not session:
            return
        filepath = self._session_file(session_id)
        tmp_path = filepath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, filepath)

    def get_or_create_session(self, session_id: str = None) -> dict:
        """获取或创建会话"""
        if not session_id:
            session_id = str(uuid.uuid4())

        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    "session_id": session_id,
                    "created_at": time.time(),
                    "updated_at": time.time(),
                    "messages": [],
                    "metadata": {},
                    "summary": "",
                }
                self._save_session(session_id)

        return self._sessions[session_id]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict = None,
    ) -> dict:
        """添加消息到会话"""
        session = self.get_or_create_session(session_id)
        message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "created_at": time.time(),
        }

        with self._lock:
            session["messages"].append(message)
            session["updated_at"] = time.time()

        self._save_session(session_id)
        return message

    def get_messages(self, session_id: str, limit: int = None) -> list:
        """获取会话消息列表"""
        session = self._sessions.get(session_id, {})
        messages = session.get("messages", [])
        if limit:
            messages = messages[-limit:]
        return messages

    def get_chat_history(self, session_id: str, limit: int = 20) -> list:
        """获取 LangChain 格式的聊天历史"""
        messages = self.get_messages(session_id, limit=limit)
        history = []
        for msg in messages:
            if msg["role"] == "user":
                history.append(("human", msg["content"]))
            elif msg["role"] == "assistant":
                history.append(("ai", msg["content"]))
        return history

    def get_summary(self, session_id: str) -> str:
        """获取会话摘要"""
        session = self._sessions.get(session_id, {})
        return session.get("summary", "")

    def set_summary(self, session_id: str, summary: str) -> None:
        """设置会话摘要"""
        session = self.get_or_create_session(session_id)
        with self._lock:
            session["summary"] = summary
            session["updated_at"] = time.time()
        self._save_session(session_id)

    def count_tokens(self, session_id: str) -> int:
        """估算会话的 token 数量（简单估算）"""
        messages = self.get_messages(session_id)
        total = 0
        for msg in messages:
            total += len(msg["content"]) // 2  # 粗略估算：2字符≈1token
        return total

    def needs_compression(self, session_id: str) -> bool:
        """判断是否需要压缩上下文"""
        count = self.count_tokens(session_id)
        return count > self.max_tokens

    def compress(self, session_id: str, keep_recent: int = 6) -> dict:
        """
        压缩上下文：保留最近 N 条消息，其余压缩为摘要

        Returns:
            压缩结果统计
        """
        session = self.get_or_create_session(session_id)
        messages = session.get("messages", [])

        if len(messages) <= keep_recent:
            return {"compressed": False, "reason": "消息数量少"}

        # 分离保留的和要压缩的
        recent = messages[-keep_recent:]
        to_compress = messages[:-keep_recent]

        # 生成摘要（将历史消息内容拼接）
        summary_parts = []
        for msg in to_compress:
            if msg["role"] in ("user", "assistant"):
                prefix = "用户" if msg["role"] == "user" else "助手"
                summary_parts.append(f"{prefix}: {msg['content'][:200]}")

        old_summary = session.get("summary", "")
        new_summary = old_summary + "\n" + "\n".join(summary_parts)

        with self._lock:
            session["messages"] = recent
            session["summary"] = new_summary
            session["updated_at"] = time.time()

        self._save_session(session_id)

        return {
            "compressed": True,
            "removed_count": len(to_compress),
            "kept_count": len(recent),
            "summary_length": len(new_summary),
        }

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
        filepath = self._session_file(session_id)
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

    def list_sessions(self) -> list:
        """列出所有会话"""
        return [
            {
                "session_id": sid,
                "message_count": len(s.get("messages", [])),
                "updated_at": s.get("updated_at"),
            }
            for sid, s in self._sessions.items()
        ]

    def get_session_stats(self, session_id: str) -> dict:
        """获取会话统计信息"""
        session = self._sessions.get(session_id, {})
        messages = session.get("messages", [])
        return {
            "session_id": session_id,
            "message_count": len(messages),
            "user_messages": sum(1 for m in messages if m["role"] == "user"),
            "assistant_messages": sum(1 for m in messages if m["role"] == "assistant"),
            "estimated_tokens": self.count_tokens(session_id),
            "needs_compression": self.needs_compression(session_id),
            "has_summary": bool(session.get("summary")),
        }