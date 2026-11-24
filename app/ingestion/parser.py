"""文档解析：按扩展名把文件字节解码为纯文本。"""
from __future__ import annotations

import io

SUPPORTED_EXT = {".txt", ".md", ".pdf", ".docx"}


def extract_text(filename: str, data: bytes) -> str:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in (".txt", ".md"):
        return data.decode("utf-8", errors="ignore").strip()
    if ext == ".pdf":
        from PyPDF2 import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if ext == ".docx":
        from docx import Document as DocxDocument

        doc = DocxDocument(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
    raise ValueError(f"不支持的文件类型: {ext or filename}（支持 {', '.join(sorted(SUPPORTED_EXT))}）")
