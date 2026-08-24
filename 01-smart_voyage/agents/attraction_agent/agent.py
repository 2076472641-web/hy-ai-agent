# -*- coding: utf-8 -*-
"""
景点推荐智能体 - 核心实现
"""
from __future__ import annotations

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt import ATTRACTION_AGENT_PROMPT
from base_agent import create_llm, create_mcp_tools


class AttractionAgent:
    """景点推荐智能体"""

    def __init__(self):
        self.llm = create_llm()
        self.tools = self._get_tools()
        self.agent = self._build_agent()

    def _get_tools(self):
        try:
            return create_mcp_tools()
        except Exception:
            from base_agent import _get_builtin_tools
            return _get_builtin_tools()

    def _build_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", ATTRACTION_AGENT_PROMPT),
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
            "agent_name": "attraction_agent",
            "task_id": task_data.get("id"),
            "response": result["response"],
            "data": result,
        }