# -*- coding: utf-8 -*-
"""兼容层：让 from tests.test_rag_qa 能指向 root 目录的 test_rag_qa.py；也支持 python -m tests.test_rag_qa 调用"""
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from test_rag_qa import *  # noqa: F401,F403

if __name__ == "__main__":
    from test_rag_qa import main
    main()
