# -*- coding: utf-8 -*-
"""
上下文压缩器 - 智能压缩长对话历史
支持自动触发压缩、摘要生成、结构化提取
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from .conversation_memory import ConversationMemory


class ContextCompressor:
    """
    上下文压缩器
    - 当对话 token 数超过阈值时自动压缩
    - 使用 LLM 生成摘要（可选，回退到规则压缩）
    - 保留关键信息（决策、偏好、约束条件）
    """

    def __init__(self, memory: ConversationMemory, llm=None):
        self.memory = memory
        self.llm = llm

    def should_compress(self, session_id: str) -> bool:
        """判断是否需要压缩"""
        return self.memory.needs_compression(session_id)

    def auto_compress(self, session_id: str, keep_recent: int = 6) -> dict:
        """
        自动压缩会话
        - 检查 token 数量
        - 如果超过阈值则压缩
        - 使用 LLM 增强摘要（如果可用）
        """
        if not self.should_compress(session_id):
            return {"compressed": False}

        # 先做基础压缩
        result = self.memory.compress(session_id, keep_recent=keep_recent)

        # 如果有 LLM，尝试增强摘要
        if self.llm and result.get("compressed"):
            try:
                enhanced = self._enhance_summary_with_llm(session_id)
                if enhanced:
                    result["enhanced_summary"] = True
            except Exception:
                pass

        return result

    def _enhance_summary_with_llm(self, session_id: str) -> Optional[str]:
        """使用 LLM 优化摘要"""
        summary = self.memory.get_summary(session_id)
        if not summary or not self.llm:
            return None

        prompt = f"""请将以下对话摘要优化为简洁的中文摘要，保留关键信息、决策和偏好，不超过200字：

原摘要：
{summary[:2000]}

优化后的摘要："""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages)
            if hasattr(response, "content"):
                enhanced = response.content
            else:
                enhanced = str(response)

            if enhanced and len(enhanced) < len(summary):
                self.memory.set_summary(session_id, enhanced)
                return enhanced
        except Exception:
            pass

        return None

    def smart_compress(self, session_id: str, keep_recent: int = 6) -> dict:
        """
        智能压缩：保留关键节点，压缩普通对话
        """
        messages = self.memory.get_messages(session_id)
        if len(messages) <= keep_recent:
            return {"compressed": False}

        # 识别关键消息（包含决策、确认、偏好的消息）
        key_indices = set()
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                content = msg.get("content", "")
                if self._is_key_message(content):
                    key_indices.add(i)

        # 保留最近的 + 关键消息
        recent_start = len(messages) - keep_recent
        keep_indices = key_indices | set(range(recent_start, len(messages)))
        keep_indices = sorted(keep_indices)

        # 如果保留的数量仍太多，回退到基础压缩
        if len(keep_indices) > len(messages) * 0.6:
            return self.memory.compress(session_id, keep_recent=keep_recent)

        # 执行选择压缩
        kept_messages = [messages[i] for i in keep_indices]
        removed_messages = [messages[i] for i in range(len(messages)) if i not in keep_indices]

        # 生成被压缩消息的摘要
        summary_parts = []
        for msg in removed_messages:
            if msg["role"] in ("user", "assistant"):
                prefix = "用户" if msg["role"] == "user" else "助手"
                content = msg.get("content", "")
                if self._is_key_message(content):
                    summary_parts.append(f"[重要]{prefix}: {content[:200]}")
                else:
                    summary_parts.append(f"{prefix}: {content[:100]}")

        old_summary = self.memory.get_summary(session_id)
        new_summary = old_summary + "\n" + "\n".join(summary_parts)

        # 更新会话
        session = self.memory.get_or_create_session(session_id)
        import threading
        with threading.Lock():
            pass  # 简化处理
        # 直接更新
        session["messages"] = kept_messages
        session["summary"] = new_summary

        # 保存
        session_file = f"{self.memory.memory_dir}/session_{session_id}.json"
        with open(session_file, "w", encoding="utf-8") as f:
            import time
            session["updated_at"] = time.time()
            json.dump(session, f, ensure_ascii=False, indent=2)

        return {
            "compressed": True,
            "method": "smart",
            "kept_count": len(kept_messages),
            "compressed_count": len(removed_messages),
        }

    def _is_key_message(self, content: str) -> bool:
        """判断是否为关键消息"""
        keywords = [
            "决定", "就这么定", "确认", "必须", "重要",
            "偏好", "我喜欢", "我想要", "我倾向",
            "预算", "多少钱", "价格",
            "日期", "时间", "几点", "哪天",
            "不要", "避免", "排除",
            "总结", "规划完成", "行程",
        ]
        return any(kw in content for kw in keywords)

    def get_compression_stats(self, session_id: str) -> dict:
        """获取压缩相关统计"""
        stats = self.memory.get_session_stats(session_id)
        return {
            **stats,
            "should_compress": self.should_compress(session_id),
            "compression_ratio": (
                f"{stats['estimated_tokens']}/{self.memory.max_tokens}"
                if stats["estimated_tokens"] > 0 else "0/0"
            ),
        }