# -*- coding: utf-8 -*-
"""
设置【当前进程】所挂载控制台窗口的标题（仅 Windows 有效，其他系统自动跳过）。

用法（各脚本入口文件开头）：
    try:
        from _console_title_helper import set_my_title_from_env
        set_my_title_from_env()
    except Exception:
        pass
"""
from __future__ import annotations

import os
import sys


def set_my_title_from_env(env_key: str = "SVCS_TITLE", fallback: str = "") -> bool:
    """
    从环境变量读取标题并设置到当前进程的控制台窗口上。
    完全在本进程内部操作，不影响父进程/兄弟进程的控制台句柄。
    """
    if sys.platform != "win32":
        return False
    title = os.environ.get(env_key) or fallback or os.path.basename(sys.argv[0])
    if not title:
        return False
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        SetConsoleTitleW = kernel32.SetConsoleTitleW
        SetConsoleTitleW.argtypes = [wintypes.LPCWSTR]
        SetConsoleTitleW.restype = wintypes.BOOL
        return bool(SetConsoleTitleW(title))
    except Exception:
        return False
