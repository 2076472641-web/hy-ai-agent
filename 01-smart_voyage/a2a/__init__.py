# -*- coding: utf-8 -*-
"""
A2A (Agent-to-Agent) 通信协议实现
参考 Google A2A 协议规范的精简实现
"""
from .protocol import (
    AgentMessage,
    AgentTask,
    TaskStatus,
    MessageRole,
    MessageType,
)
from .agent_card import AgentCard
from .transport import A2ATransport

__all__ = [
    "AgentMessage",
    "AgentTask",
    "TaskStatus",
    "MessageRole",
    "MessageType",
    "AgentCard",
    "A2ATransport",
]