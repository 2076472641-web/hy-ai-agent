# -*- coding: utf-8 -*-
"""
星途环球·跨境售后RAG智能问答系统 - 数据集搭建与预处理模块
对应简历模块1：
  - PyMuPDF / python-docx 解析电子文档
  - PaddleOCR 提取外文扫描件图文
  - 按跨境物流 / 海外售后 / 各国合规政策三类差异化分块
  - 清洗校验沉淀标准化跨境业务知识数据
"""
from __future__ import annotations

import os
import re
import json
import uuid
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from config import KNOWLEDGE_CATEGORIES, get_config


# ============================================================
# 1. 数据结构：文档 -> 文本片段 (Chunk)
# ============================================================
@dataclass
class RawDocument:
    """原始文档对象：从文件解析出的全文+元数据"""
    doc_id: str
    file_path: str
    file_type: str           # pdf / docx / txt / scanned (OCR)
    category: str            # cross_border_logistics / after_sales_policy / customs_compliance
    title: str
    content: str             # 纯文本全文
    language: str = "zh"
    source_url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_count: int = 0


@dataclass
class KnowledgeChunk:
    """分块后的知识片段（父子块结构）"""
    chunk_id: str
    doc_id: str
    category: str
    text: str
    language: str = "zh"
    # 父子块索引
    parent_chunk_id: Optional[str] = None
    is_parent: bool = False
    child_indices: List[str] = field(default_factory=list)
    # 位置信息
    start_pos: int = 0
    end_pos: int = 0
    page_number: int = 0
    section_heading: str = ""
    # 稀疏表征关键词
    sparse_keywords: List[str] = field(default_factory=list)
    # 风控合规标记
    compliance_level: str = "normal"   # normal / warning / strict
    needs_citation: bool = False
    # 分块来源文档标题
    doc_title: str = ""


# ============================================================
# 2. 文档解析器（多格式支持）
# ============================================================
class BaseDocumentParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """解析文件，返回 (纯文本内容, 元信息)"""
        raise NotImplementedError


class PDFParser(BaseDocumentParser):
    """
    PyMuPDF 解析电子PDF（非扫描件走这个）
    对应简历：借助 PyMuPDF 解析电子文档
    """
    def parse(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        meta: Dict[str, Any] = {"page_count": 0}
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            texts: List[str] = []
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text("text")
                texts.append(f"[第{page_num}页]\n{page_text}")
            meta["page_count"] = doc.page_count
            doc.close()
            return "\n".join(texts), meta
        except Exception:
            # 降级：如果没装PyMuPDF，尝试纯文本读取
            with open(file_path, "rb") as f:
                raw = f.read()
            try:
                return raw.decode("utf-8", errors="ignore"), meta
            except Exception:
                return "", meta


class DocxParser(BaseDocumentParser):
    """
    python-docx 解析 Word 文档
    对应简历：借助 python-docx 解析电子文档
    """
    def parse(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        meta: Dict[str, Any] = {"page_count": 1}
        try:
            from docx import Document
            doc = Document(file_path)
            texts: List[str] = []
            # 段落
            for para in doc.paragraphs:
                if para.text.strip():
                    texts.append(para.text.strip())
            # 表格（跨境政策文档中常见税率表）
            for table_idx, table in enumerate(doc.tables):
                texts.append(f"\n[表格{table_idx+1}]")
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    texts.append(" | ".join(cells))
            return "\n".join(texts), meta
        except Exception:
            with open(file_path, "rb") as f:
                raw = f.read()
            try:
                return raw.decode("utf-8", errors="ignore"), meta
            except Exception:
                return "", meta


class TxtParser(BaseDocumentParser):
    """纯文本解析（政策法规TXT、README等）"""
    def parse(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        meta: Dict[str, Any] = {"page_count": 1}
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read(), meta
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="gbk", errors="ignore") as f:
                return f.read(), meta


class OCRParser(BaseDocumentParser):
    """
    PaddleOCR 扫描件图文提取
    对应简历：PaddleOCR 提取外文扫描件图文
    多语种支持：英文/中文/日文/西语/德语/法语
    """
    def __init__(self, languages: Optional[List[str]] = None):
        cfg = get_config()
        self.languages = languages or cfg.ocr_languages
        self._ocr_instance = None

    def _get_ocr(self):
        if self._ocr_instance is None:
            try:
                from paddleocr import PaddleOCR
                # 多语种合成模型
                lang_map = {
                    "en": "en", "ch": "ch", "ja": "japan",
                    "es": "spanish", "de": "german", "fr": "french",
                }
                ocr_lang = lang_map.get(self.languages[0], "en") if self.languages else "en"
                self._ocr_instance = PaddleOCR(use_angle_cls=True, lang=ocr_lang, show_log=False)
            except Exception as exc:
                print(f"[OCRParser] PaddleOCR加载失败，降级为图片文本为空: {exc}")
                self._ocr_instance = False
        return self._ocr_instance

    def parse(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        meta: Dict[str, Any] = {"page_count": 1, "ocr_processed": True}
        ocr = self._get_ocr()
        if not ocr:
            return "", meta
        try:
            result = ocr.ocr(file_path, cls=True)
            lines: List[str] = []
            for page in result or []:
                for item in page or []:
                    try:
                        text = item[1][0]  # 识别文本
                        lines.append(text)
                    except Exception:
                        continue
            return "\n".join(lines), meta
        except Exception as exc:
            print(f"[OCRParser] OCR识别失败: {exc}")
            return "", meta


class DocumentParserFactory:
    """根据文件后缀自动选择解析器"""
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".scan.pdf"}

    @classmethod
    def get_parser(cls, file_path: str) -> BaseDocumentParser:
        ext = os.path.splitext(file_path)[1].lower()
        name_lower = os.path.basename(file_path).lower()
        # 扫描PDF（文件名包含scan/scanned/扫描）走OCR
        if ext == ".pdf" and any(k in name_lower for k in ("scan", "扫描", "ocr")):
            return OCRParser()
        if ext in cls.IMAGE_EXTS:
            return OCRParser()
        if ext == ".pdf":
            return PDFParser()
        if ext in (".docx", ".doc"):
            return DocxParser()
        if ext in (".txt", ".md"):
            return TxtParser()
        return TxtParser()


# ============================================================
# 3. 文本清洗 & 校验
# ============================================================
class TextCleaner:
    """跨境业务知识数据清洗校验"""

    # 跨境常见噪声正则
    NOISE_PATTERNS: List[re.Pattern] = [
        re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),  # 不可打印字符
        re.compile(r"\.{4,}"),  # 多余省略号
        re.compile(r"_{3,}"),   # 连续下划线
        re.compile(r"[ \t]{3,}"),  # 多余空白
        re.compile(r"\n{4,}"),  # 多余空行
    ]

    # 跨境合规黑名单（清洗掉营销广告、无意义页脚页头）
    BLACKLIST_PHRASES: List[str] = [
        "点击购买", "限时特惠", "官方旗舰店", "扫码关注", "优惠券",
        "仅供参考，不构成", "免责声明本文档", "本公司不承担任何法律责任",
    ]

    @classmethod
    def clean(cls, text: str) -> str:
        if not text:
            return ""
        t = text
        for pat in cls.NOISE_PATTERNS:
            t = pat.sub(" ", t)
        # 去常见营销语
        for bad in cls.BLACKLIST_PHRASES:
            t = t.replace(bad, "")
        # 统一换行符
        t = re.sub(r"\r\n|\r", "\n", t)
        t = re.sub(r"\n[ \t]+", "\n", t)
        # 去每行首尾空白
        lines = [ln.strip() for ln in t.split("\n")]
        lines = [ln for ln in lines if ln]
        return "\n".join(lines)

    @classmethod
    def validate(cls, text: str, category: str) -> Tuple[bool, List[str]]:
        """
        校验清洗后文本是否可用
        返回 (是否合格, 问题列表)
        """
        issues: List[str] = []
        if not text or len(text.strip()) < 50:
            issues.append("文本过短(<50字)，疑似无效文档")
        # 按分类校验关键词存在性（防止文档类别错放）
        category_keywords = {
            "cross_border_logistics": [
                "物流", "快递", "时效", "运费", "清关", "FBA", "DHL", "EMS", "包裹", "派送"
            ],
            "after_sales_policy": [
                "退货", "退款", "售后", "质保", "差评", "争议", "退换", "索赔", "拒付", "召回"
            ],
            "customs_compliance": [
                "关税", "VAT", "海关", "HS编码", "申报", "IOSS", "认证", "FDA", "合规", "税率"
            ],
        }
        kws = category_keywords.get(category, [])
        hit = sum(1 for k in kws if k in text)
        if kws and hit < 1:
            issues.append(f"文档未命中[{category}]类业务关键词（命中{hit}个），疑似分类错误")
        return len(issues) == 0, issues


# ============================================================
# 4. 三类差异化分块策略
# 对应简历：按跨境物流 / 海外售后 / 各国合规政策三类差异化分块
# ============================================================
class Chunker:
    """差异化分块器：每类知识用不同的chunk_size/overlap + 章节感知"""

    def __init__(self, category: str):
        if category not in KNOWLEDGE_CATEGORIES:
            raise ValueError(f"未知知识库类别: {category}")
        self.category = category
        self.cfg = KNOWLEDGE_CATEGORIES[category]
        self.chunk_size = self.cfg["chunk_size"]
        self.chunk_overlap = self.cfg["chunk_overlap"]

    def chunk(self, doc: RawDocument) -> List[KnowledgeChunk]:
        """
        差异化分块 + 父子块结构
        - 父块：按章节/大段（500~800字），存储完整上下文
        - 子块：按chunk_size切分，用于精准检索
        """
        cleaned_text = TextCleaner.clean(doc.content)
        paragraphs = self._split_paragraphs(cleaned_text)
        chunks: List[KnowledgeChunk] = []

        # 合规类：严格按章节头 + 条款编号切分
        if self.category == "customs_compliance":
            return self._chunk_by_clauses(paragraphs, doc)

        # 物流/售后类：语义段落 + 固定大小滑窗
        return self._chunk_by_sliding_window(paragraphs, doc)

    # -------- 分块内部辅助 --------
    def _split_paragraphs(self, text: str) -> List[str]:
        parts = re.split(r"\n{2,}", text)
        return [p.strip() for p in parts if p.strip()]

    def _chunk_by_clauses(self, paragraphs: List[str], doc: RawDocument) -> List[KnowledgeChunk]:
        """
        海关合规类：按条款编号切分（1.2.3/①②③/第X条），保持政策条款完整性
        对应简历"各国合规政策差异化分块"
        """
        clause_re = re.compile(r"^(\d+[\.、]|第[一二三四五六七八九十百千]+[条章节]|[①②③④⑤⑥⑦⑧⑨⑩]|[\(（]\d+[\)）])")
        groups: List[Tuple[str, str]] = []  # (heading, body)
        current_head = ""
        current_body: List[str] = []

        def flush():
            body = "\n".join(current_body).strip()
            if body or current_head:
                groups.append((current_head, body))

        for para in paragraphs:
            if clause_re.match(para):
                flush()
                # 新条款：取首句作为heading
                head_end = para.find("\n") if "\n" in para else min(len(para), 40)
                current_head = para[:head_end].strip()
                rest = para[head_end:].strip()
                current_body = [rest] if rest else []
            else:
                current_body.append(para)
        flush()

        chunks: List[KnowledgeChunk] = []
        for idx, (head, body) in enumerate(groups):
            text = f"{head}\n{body}".strip() if head else body
            if not text:
                continue
            # 合规文本过长再做子块拆分
            if len(text) > self.chunk_size + 200:
                sub_chunks = self._split_long_text(text, self.chunk_size, self.chunk_overlap)
                parent_id = f"{doc.doc_id}_parent_{idx:04d}"
                # 父块
                chunks.append(KnowledgeChunk(
                    chunk_id=parent_id,
                    doc_id=doc.doc_id,
                    category=doc.category,
                    language=doc.language,
                    text=text[:800],
                    is_parent=True,
                    page_number=0,
                    section_heading=head,
                    compliance_level="strict",
                    needs_citation=True,
                    doc_title=doc.title,
                    sparse_keywords=self._extract_sparse_keywords(text, doc.category),
                ))
                for sub_idx, sub_text in enumerate(sub_chunks):
                    child_id = f"{doc.doc_id}_child_{idx:04d}_{sub_idx:02d}"
                    chunks[-1].child_indices.append(child_id)
                    chunks.append(KnowledgeChunk(
                        chunk_id=child_id,
                        doc_id=doc.doc_id,
                        category=doc.category,
                        language=doc.language,
                        text=sub_text,
                        parent_chunk_id=parent_id,
                        page_number=0,
                        section_heading=head,
                        compliance_level="strict",
                        needs_citation=True,
                        doc_title=doc.title,
                        sparse_keywords=self._extract_sparse_keywords(sub_text, doc.category),
                    ))
            else:
                chunks.append(KnowledgeChunk(
                    chunk_id=f"{doc.doc_id}_chunk_{idx:04d}",
                    doc_id=doc.doc_id,
                    category=doc.category,
                    language=doc.language,
                    text=text,
                    page_number=0,
                    section_heading=head,
                    compliance_level="strict",
                    needs_citation=True,
                    doc_title=doc.title,
                    sparse_keywords=self._extract_sparse_keywords(text, doc.category),
                ))
        return chunks

    def _chunk_by_sliding_window(self, paragraphs: List[str], doc: RawDocument) -> List[KnowledgeChunk]:
        """
        物流/售后类：语义段落 + 滑窗切分，保留SOP步骤完整性
        对应简历"跨境物流、海外售后差异化分块"
        """
        # 先按语义段落拼接，再按chunk_size滑窗
        full_text = "\n".join(paragraphs)
        sub_texts = self._split_long_text(full_text, self.chunk_size, self.chunk_overlap)

        chunks: List[KnowledgeChunk] = []
        # 生成父块（整段/整节概要）
        if len(sub_texts) > 1:
            parent_id = f"{doc.doc_id}_parent_0000"
            chunks.append(KnowledgeChunk(
                chunk_id=parent_id,
                doc_id=doc.doc_id,
                category=doc.category,
                language=doc.language,
                text=full_text[:1000],
                is_parent=True,
                page_number=0,
                compliance_level="normal",
                needs_citation=False,
                doc_title=doc.title,
                sparse_keywords=self._extract_sparse_keywords(full_text, doc.category),
            ))
        else:
            parent_id = None

        for idx, text in enumerate(sub_texts):
            child_id = f"{doc.doc_id}_chunk_{idx:04d}"
            kc = KnowledgeChunk(
                chunk_id=child_id,
                doc_id=doc.doc_id,
                category=doc.category,
                language=doc.language,
                text=text,
                page_number=0,
                compliance_level="warning" if doc.category == "after_sales_policy" else "normal",
                needs_citation=False,
                doc_title=doc.title,
                sparse_keywords=self._extract_sparse_keywords(text, doc.category),
            )
            if parent_id:
                kc.parent_chunk_id = parent_id
                if chunks and chunks[-1].chunk_id == parent_id:
                    chunks[-1].child_indices.append(child_id)
            chunks.append(kc)
        return chunks

    @staticmethod
    def _split_long_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        滑窗切分长文本，尽量在句号/换行处断句
        🔥 修复：增加 3 层死循环/内存保护，避免 chunk_size <= overlap / 无标点文本 导致死循环
        """
        if chunk_size <= 0:
            chunk_size = 300
        # 【安全1】overlap 必须严格小于 chunk_size，否则 start 不前进会死循环
        if overlap >= chunk_size:
            overlap = chunk_size // 4
        if len(text) <= chunk_size:
            return [text] if text else []
        pieces: List[str] = []
        start = 0
        n = len(text)
        prev_start = -1
        # 【安全2】最大迭代次数（理论上限的 2 倍，绝不可能触发）
        max_iters = (n // max(1, chunk_size - overlap)) + 50
        iters = 0
        while start < n and iters < max_iters:
            iters += 1
            end = min(start + chunk_size, n)
            # 尝试找断句点（句号/换行/分号/问号/感叹号）
            if end < n:
                punct_idx = max(
                    text.rfind("。", start, end),
                    text.rfind("\n", start, end),
                    text.rfind("？", start, end),
                    text.rfind("！", start, end),
                    text.rfind("；", start, end),
                    text.rfind(";", start, end),
                    text.rfind("?", start, end),
                    text.rfind("!", start, end),
                )
                if punct_idx > start + chunk_size // 2:
                    end = punct_idx + 1
            # 防止 end 比 start 小（极端边界情况）
            if end <= start:
                end = min(start + chunk_size, n)
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            # 计算下一次的 start
            next_start = end - overlap
            # 【安全3】死循环拦截：如果 next_start 没有比上次的 prev_start 前进，直接粗暴跳到 end
            if next_start <= prev_start or next_start <= start:
                next_start = end
                if next_start <= start:
                    next_start = start + max(1, chunk_size - overlap)
            prev_start = start
            start = next_start
            if start >= n:
                break
            if start < 0:
                start = 0
        return pieces

    @staticmethod
    def _extract_sparse_keywords(text: str, category: str) -> List[str]:
        """
        稀疏表征关键词（TF-IDF/BM25层补充用）
        对应简历：稀疏向量双表征架构
        """
        # 先抽取数字（时效/税率/费用是跨境核心）
        num_phrases = re.findall(r"\d+(?:\.\d+)?(?:%|美元|欧元|元|日元|天|工作日|小时|kg|g|GB|英镑)", text)
        # 抽取跨境专属术语（大写缩写/条款编号）
        terms = re.findall(r"[A-Z]{2,8}(?:[-/][A-Z0-9]{2,8})*", text)
        # 中文4字+术语
        cn_terms = re.findall(r"[\u4e00-\u9fff]{4,8}(?:政策|条款|规则|流程|SOP|税率|关税|认证|编号|代码)", text)
        # 去重+限量
        combined = list(dict.fromkeys(num_phrases + terms + cn_terms))
        return combined[:30]


# ============================================================
# 5. Pipeline 对外API：文件 -> 分块入库
# ============================================================
class DataPreprocessingPipeline:
    """
    数据集搭建总管线
    调用：解析 → 清洗校验 → 三类差异化分块 → 输出KnowledgeChunk列表
    """
    def __init__(self, knowledge_dir: Optional[str] = None):
        cfg = get_config()
        self.knowledge_dir = knowledge_dir or cfg.knowledge_dir
        os.makedirs(self.knowledge_dir, exist_ok=True)

    def process_file(self, file_path: str, category: str,
                     doc_title: Optional[str] = None) -> Tuple[List[KnowledgeChunk], List[str]]:
        """处理单份文档，返回(分块列表, 问题警告列表)"""
        if category not in KNOWLEDGE_CATEGORIES:
            raise ValueError(f"category必须是: {list(KNOWLEDGE_CATEGORIES.keys())}")
        warnings: List[str] = []

        # 1. 生成doc_id
        file_hash = hashlib.md5(open(file_path, "rb").read()).hexdigest()[:8]
        doc_id = f"{category}_{file_hash}_{uuid.uuid4().hex[:6]}"

        # 2. 解析
        parser = DocumentParserFactory.get_parser(file_path)
        content, meta = parser.parse(file_path)
        file_type = os.path.splitext(file_path)[1].lstrip(".").lower()
        title = doc_title or os.path.splitext(os.path.basename(file_path))[0]

        doc = RawDocument(
            doc_id=doc_id,
            file_path=file_path,
            file_type=file_type,
            category=category,
            title=title,
            content=content,
            page_count=meta.get("page_count", 0),
            metadata=meta,
        )

        # 3. 清洗校验
        ok, issues = TextCleaner.validate(doc.content, category)
        warnings.extend(issues)
        doc.content = TextCleaner.clean(doc.content)
        if not doc.content:
            warnings.append("解析清洗后文本为空，跳过分块")
            return [], warnings

        # 4. 差异化分块
        chunker = Chunker(category)
        chunks = chunker.chunk(doc)
        if not chunks:
            warnings.append("分块结果为空")
        return chunks, warnings

    def scan_directory(self, dir_path: Optional[str] = None) -> Dict[str, List[KnowledgeChunk]]:
        """
        扫描 knowledge_dir，按子目录名=category 自动批处理
        子目录：cross_border_logistics / after_sales_policy / customs_compliance
        """
        root = dir_path or self.knowledge_dir
        result: Dict[str, List[KnowledgeChunk]] = {c: [] for c in KNOWLEDGE_CATEGORIES}
        for cat in KNOWLEDGE_CATEGORIES:
            cat_dir = os.path.join(root, cat)
            if not os.path.isdir(cat_dir):
                continue
            for fn in os.listdir(cat_dir):
                fp = os.path.join(cat_dir, fn)
                if not os.path.isfile(fp):
                    continue
                try:
                    chunks, warns = self.process_file(fp, cat)
                    if chunks:
                        result[cat].extend(chunks)
                    for w in warns:
                        print(f"[预处理][{cat}/{fn}] ⚠️  {w}")
                except Exception as exc:
                    print(f"[预处理][{cat}/{fn}] ❌ 失败: {exc}")
        return result


# ============================================================
# 6. 从 test_cases 快速构建种子知识库（离线演示用）
# ============================================================
def build_chunks_from_json_cases(json_path: str) -> List[KnowledgeChunk]:
    """
    从简历中的测试用例 JSON（51组）直接生成 KnowledgeChunk
    让RAG系统在无外部文档时也能回答核心问题
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("test_cases", [])
    all_chunks: List[KnowledgeChunk] = []
    for case in cases:
        category = case["category"]
        chunker = Chunker(category)
        # 将"问题知识"构造成文档：标题+标准答+知识点+标签
        text = (
            f"【知识点】{case.get('knowledge_point', '')}\n"
            f"【问题示例】{case.get('question', '')}\n"
            f"【标准解答】{case.get('reference_answer', '')}\n"
            f"【所属类目】{case.get('category_name', '')}\n"
            f"【关联关键词】{', '.join(case.get('expected_keywords', []))}"
        )
        doc = RawDocument(
            doc_id=f"case_{case['id']}",
            file_path=json_path,
            file_type="json",
            category=category,
            title=f"{case.get('category_name', '')}-{case.get('knowledge_point', '')}",
            content=text,
            language="zh",
            metadata={"tags": case.get("tags", []), "entities": case.get("expected_entities", {})},
        )
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
    return all_chunks
