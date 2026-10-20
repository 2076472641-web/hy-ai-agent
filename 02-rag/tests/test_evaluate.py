# -*- coding: utf-8 -*-
"""兼容层：让 from tests.test_evaluate 能指向 root 目录的 test_evaluate.py"""
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from test_evaluate import *  # noqa: F401,F403
from test_evaluate import (
    evaluate_qa_case,
    build_summary_report,
    format_report_console,
)  # 显式重导出，保证静态分析正确
