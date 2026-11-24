"""递归字符分块：按段落 -> 句子 -> 字符逐级切分，控制块长并保留重叠。"""
from __future__ import annotations

import re

# 句子边界：中文句号/问号/叹号/分号 + 英文标点
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；!?;])")


def _split_sentences(segment: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(segment) if p.strip()]
    return parts or [segment.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[str]:
    """把长文本切成不超过 chunk_size 的块，相邻块保留 chunk_overlap 字符重叠。

    先按段落聚合，段落超长再按句子聚合，句子仍超长时按字符硬切。
    """
    text = text.strip()
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            units.append(para)
            continue
        for sent in _split_sentences(para):
            if len(sent) <= chunk_size:
                units.append(sent)
            else:
                # 单句超长：按 chunk_size 硬切
                for i in range(0, len(sent), chunk_size):
                    units.append(sent[i : i + chunk_size])

    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
        elif len(current) + len(unit) + 1 <= chunk_size:
            current = f"{current} {unit}"
        else:
            chunks.append(current)
            # 携带上一块尾部作为重叠上下文
            tail = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            current = f"{tail} {unit}".strip()
    if current:
        chunks.append(current)
    return chunks
