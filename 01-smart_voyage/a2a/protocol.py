# -*- coding: utf-8 -*-
"""
A2A 协议 - 数据结构定义
定义 Agent 之间通信的消息、任务、状态等核心数据结构
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageType(str, Enum):
    TEXT = "text"
    STRUCTURED = "structured"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Part(BaseModel):
    """消息的一部分（支持多模态扩展）"""
    kind: str = "text"
    text: str = ""
    data: Optional[dict] = None
    metadata: Optional[dict] = None


class AgentMessage(BaseModel):
    """
    A2A 消息结构
    参考 Google A2A 协议的 Message 定义
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole = MessageRole.USER
    type: MessageType = MessageType.TEXT
    content: str = ""
    parts: list[Part] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

    def to_content(self) -> str:
        if self.content:
            return self.content
        return "\n".join(p.text for p in self.parts if p.text)


class AgentTask(BaseModel):
    """
    A2A 任务结构
    用于在 Agent 之间传递工作任务
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    messages: list[AgentMessage] = Field(default_factory=list)
    result: Optional[Any] = None
    error: Optional[str] = None
    assigned_to: Optional[str] = None
    parent_task_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def add_message(self, message: AgentMessage) -> None:
        self.messages.append(message)
        self.updated_at = time.time()

    def complete(self, result: Any = None) -> None:
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.updated_at = time.time()

    def fail(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.error = error
        self.updated_at = time.time()

    def progress(self) -> None:
        self.status = TaskStatus.IN_PROGRESS
        self.updated_at = time.time()