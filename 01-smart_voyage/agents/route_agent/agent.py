# -*- coding: utf-8 -*-
"""
出行路线规划智能体 - 核心实现
"""
from __future__ import annotations

import json
import re
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

from prompt import ROUTE_AGENT_PROMPT
from base_agent import create_llm, create_mcp_tools


class RouteAgent:
    """出行路线规划智能体"""

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
            ("system", ROUTE_AGENT_PROMPT),
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
        # =========调试日志新增=========
        print("\n==========【RouteAgent 收到原始请求文本】==========")
        print(user_input)
        print("==================================================\n")
        # =========调试日志新增=========

        # 对输入进行结构化标注，防止 LLM 颠倒起点终点
        normalized_input = self._normalize_route_input(user_input)
        if normalized_input != user_input:
            print(f"==========【RouteAgent 结构化标注后】==========")
            print(normalized_input)
            print("==================================================\n")

        input_data = {"input": normalized_input}
        if chat_history:
            input_data["chat_history"] = chat_history

        result = await self.agent.ainvoke(input_data)

        return {
            "response": result.get("output", ""),
            "tool_calls": result.get("intermediate_steps", []),
            "raw_output": json.dumps(result, ensure_ascii=False, default=str),
        }

    def _normalize_route_input(self, user_input: str) -> str:
        """
        对路线查询输入进行结构化标注，明确标识出发城市和目的地城市，
        防止 LLM 在调用 plan_transport 工具时颠倒或替换 origin / destination。

        核心策略：以「从」字为锚点定位出发城市，以「到/去/前往」定位目的地城市，
        避免正则贪婪匹配把无关文本吞入城市名。

        仅处理主干城际路线的输入标注，目的地周边短途支线不受影响。
        """
        # 以「从」为锚点定位出发城市，以「到/去/前往」定位目的地城市。
        # 用负向前瞻排除路线关键词汉字，防止泄漏进城市名：
        #   "从成都出发去西安" → 若不排除"出发"，第一组会捕获"成都出发"
        #   "从湖南到上海的交通" → 若不排除"的"，第二组会捕获"上海的交通"
        # 关键：(?:...) 非捕获组把 lookahead+charclass 打包为一个单元，
        #       使 + 量词对每个字符都执行前瞻检查，而不是只检查首字符。
        _CJK = r"[一-龥A-Za-z0-9]"
        _ORIGIN_CHAR = r"(?:(?![出发去的])" + _CJK + r")"   # 每字符排除"出发去的"
        _DEST_CHAR = r"(?:(?![出发去的的])" + _CJK + r")"   # 每字符排除"出发去的的"
        pattern = re.compile(
            r"从(" + _ORIGIN_CHAR + r"+?)(?:出发)?(?:到|前往|去往|去)(" + _DEST_CHAR + r"+)"
        )
        match = pattern.search(user_input)
        if match:
            origin = match.group(1).strip()
            destination = match.group(2).strip()
            print(f"==========【RouteAgent 城市提取】==========")
            print(f"  出发城市(origin):      {origin}")
            print(f"  目的地城市(destination): {destination}")
            print("============================================\n")
            tagged = (
                f"=== 城市信息确认 ===\n"
                f"出发城市(origin) = {origin}\n"
                f"目的地城市(destination) = {destination}\n"
                f"========================\n"
                f"{user_input}"
            )
            return tagged
        return user_input

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
            "agent_name": "route_agent",
            "task_id": task_data.get("id"),
            "response": result["response"],
            "data": result,
        }