# -*- coding: utf-8 -*-
"""
docx2md 批量转换脚本（自包含）
用法：
  1. 把 .docx 文件放到本脚本同一目录
  2. 双击运行，或命令行执行：python convert_docx2md.py
  3. 自动在同目录生成同名 .md
"""
import sys
import os
import glob
import subprocess

def ensure_markitdown():
    """优先直接 import；失败则尝试多种方式 pip install。"""
    # 方式0：直接用
    try:
        from markitdown import MarkItDown
        return MarkItDown()
    except ImportError:
        pass

    print("[*] 首次运行，正在安装 markitdown[docx] ...")
    install_cmds = [
        [sys.executable, "-m", "pip", "install", "markitdown[docx]"],
        [sys.executable, "-m", "pip", "install", "--user", "markitdown[docx]"],
    ]
    for cmd in install_cmds:
        try:
            print(f"[*] 执行: {' '.join(cmd)}")
            subprocess.check_call(cmd)
            from markitdown import MarkItDown
            print("[*] markitdown 安装成功")
            return MarkItDown()
        except Exception as e:
            print(f"[!] 该方式失败: {e}")
    raise RuntimeError("markitdown 安装失败，请手动执行: pip install markitdown[docx]")

def main():
    target_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"[*] 目标目录: {target_dir}")

    docx_files = sorted(glob.glob(os.path.join(target_dir, "*.docx")))
    docx_files = [f for f in docx_files if not os.path.basename(f).startswith("~$")]
    print(f"[*] 找到 {len(docx_files)} 个 .docx 文件\n")

    if not docx_files:
        print("[!] 目录下没有 .docx 文件，请先放入 Word 文档。")
        return

    md = ensure_markitdown()
    success, failed = [], []

    for docx_path in docx_files:
        docx_name = os.path.basename(docx_path)
        md_name = os.path.splitext(docx_name)[0] + ".md"
        md_path = os.path.join(target_dir, md_name)

        print(f"[>] 转换: {docx_name}")
        try:
            result = md.convert(docx_path)
            text = result.text_content or ""
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(text)
            success.append((docx_name, md_name, text.strip()[:100]))
            print(f"    -> 成功: {md_name}")
        except Exception as e:
            failed.append((docx_name, f"{type(e).__name__}: {e}"))
            print(f"    -> 失败: {e}")

    print("\n" + "=" * 60)
    print(f"转换完成: 成功 {len(success)} 个, 失败 {len(failed)} 个")
    print("=" * 60)

    if success:
        print("\n【成功文件 + 前100字符预览】")
        for i, (src, out, preview) in enumerate(success, 1):
            print(f"\n{i}. {src}  ->  {out}")
            print(f"   预览: {preview}")

    if failed:
        print("\n【失败文件】")
        for src, err in failed:
            print(f"  - {src}: {err}")

if __name__ == "__main__":
    main()
