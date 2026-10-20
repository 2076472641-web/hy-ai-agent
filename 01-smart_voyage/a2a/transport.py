# -*- coding: utf-8 -*-
"""
A2A 传输层 - 基于 HTTP 的 Agent 间通信
使用 httpx 实现异步的 Agent 间消息传递
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .protocol import AgentMessage, AgentTask, TaskStatus


class A2ATransport:
    """
    A2A 传输层 - 管理 Agent 之间的 HTTP 通信
    支持：
    - 向指定 Agent 发送任务
    - 接收其他 Agent 发来的任务
    - 任务状态回调
    """

    def __init__(self, agent_name: str, base_url: str, timeout: float = 30.0):
        self.agent_name = agent_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._handlers: dict[str, Callable] = {}
        self._task_cache: dict[str, AgentTask] = {}

    def register_handler(self, task_type: str, handler: Callable) -> None:
        """注册任务处理函数"""
        self._handlers[task_type] = handler

    def register_default_handler(self, handler: Callable) -> None:
        """注册默认任务处理函数"""
        self._handlers["default"] = handler

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def send_task(
        self,
        target_url: str,
        task: AgentTask,
        callback: Optional[Callable] = None,
    ) -> AgentTask:
        """
        向目标 Agent 发送任务
        """
        endpoint = f"{target_url.rstrip('/')}/a2a/tasks"
        payload = task.model_dump()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            result_data = response.json()

        result_task = AgentTask(**result_data)
        self._task_cache[result_task.id] = result_task

        if callback:
            try:
                callback(result_task)
            except Exception as e:
                pass

        return result_task

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def send_message(
        self,
        target_url: str,
        message: AgentMessage,
    ) -> AgentMessage:
        """
        向目标 Agent 发送简单消息并获取响应
        """
        endpoint = f"{target_url.rstrip('/')}/a2a/messages"
        payload = message.model_dump()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            result_data = response.json()

        return AgentMessage(**result_data)

    async def handle_task(self, task_data: dict) -> dict:
        """
        处理收到的任务（供 FastAPI 路由调用）
        """
        task = AgentTask(**task_data)
        task.progress()

        try:
            handler = self._handlers.get(
                task.metadata.get("task_type", "default"),
                self._handlers.get("default"),
            )
            if handler:
                result = await handler(task)
                if isinstance(result, AgentTask):
                    task = result
                else:
                    task.complete(result=result)
            else:
                task.fail(error=f"No handler registered for task_type: {task.metadata.get('task_type')}")
        except Exception as e:
            task.fail(error=str(e))

        return task.model_dump()

    async def handle_message(self, message_data: dict) -> dict:
        """
        处理收到的消息（供 FastAPI 路由调用）
        """
        message = AgentMessage(**message_data)

        handler = self._handlers.get(
            "message",
            self._handlers.get("default"),
        )
        if handler:
            task = AgentTask(
                title="message_processing",
                metadata={"task_type": "message", "message": message},
            )
            result = await handler(task)
            if isinstance(result, AgentMessage):
                return result.model_dump()
            elif isinstance(result, AgentTask) and result.messages:
                return result.messages[-1].model_dump()

        return AgentMessage(
            role="assistant",
            content="Service is busy, please try again later.",
        ).model_dump()

    def create_task(self, title: str, description: str, **metadata) -> AgentTask:
        """创建一个新任务"""
        return AgentTask(
            title=title,
            description=description,
            metadata=metadata,
        )

    def create_message(self, content: str, role: str = "user", **metadata) -> AgentMessage:
        """创建一条新消息"""
        return AgentMessage(
            content=content,
            role=role,
            metadata=metadata,
        )