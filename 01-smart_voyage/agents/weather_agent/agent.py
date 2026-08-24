# -*- coding: utf-8 -*-
"""
天气查询智能体 - 核心实现
基于 LangChain Agent + MCP 工具
"""
from __future__ import annotations

import json
import asyncio
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 把自己所在包目录也加进去，避免 from .xxx 相对导入失败（python weather_agent/main.py 顶级运行时无父包）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt import WEATHER_AGENT_PROMPT
from base_agent import create_llm, create_mcp_tools


class WeatherAgent:
    """天气查询智能体"""

    def __init__(self):
        self.llm = create_llm()
        self.tools = self._get_tools()
        self.agent = self._build_agent()

    def _get_tools(self):
        """获取可用工具"""
        try:
            return create_mcp_tools()
        except Exception:
            from base_agent import _get_builtin_tools
            return _get_builtin_tools()

    def _build_agent(self):
        """构建 LangChain Agent"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", WEATHER_AGENT_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
        )

    async def query(self, user_input: str, chat_history: list = None) -> dict:
        """
        查询天气 - 异步调用

        Args:
            user_input: 用户输入
            chat_history: 历史对话

        Returns:
            {"response": str, "tool_calls": list, "raw_output": str}
        """
        input_data = {"input": user_input}
        if chat_history:
            input_data["chat_history"] = chat_history

        result = await self.agent.ainvoke(input_data)

        return {
            "response": result.get("output", ""),
            "tool_calls": result.get("intermediate_steps", []),
            "raw_output": json.dumps(result, ensure_ascii=False, default=str),
        }

    async def handle_a2a_task(self, task_data: dict) -> dict:
        """
        处理来自其他智能体的 A2A 任务
        """
        metadata = task_data.get("metadata", {})
        query = metadata.get("query", "") or metadata.get("content", "")

        if not query and task_data.get("messages"):
            messages = task_data["messages"]
            if messages:
                query = messages[-1].get("content", "")

        if not query:
            return {"error": "No query provided in task"}

        result = await self.query(query)
        return {
            "agent_name": "weather_agent",
            "task_id": task_data.get("id"),
            "response": result["response"],
            "data": result,
        }