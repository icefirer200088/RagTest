"""
chunk.py — 文本分块策略

把长文本切成适合检索的小块。
支持固定大小、按句子、按段落三种策略。
"""

import re


def chunk_by_size(text: str, chunk_size: int = 256, overlap: int = 32) -> list[str]:
    """
    固定字符数切块，带 overlap 保持上下文连贯。

    Args:
        text: 原始文本
        chunk_size: 每块字符数
        overlap: 相邻块重叠字符数

    Returns:
        list[str] 块列表
    """
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def chunk_by_sentence(text: str, max_chars: int = 512) -> list[str]:
    """
    按句子（。！？；\n）切块，单个句子超长时再按字符切。

    Args:
        text: 原始文本
        max_chars: 每块最大字符数

    Returns:
        list[str] 块列表
    """
    if not text:
        return []

    # 按标点/换行分句
    sentences = re.split(r'(?<=[。！？；\n])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""
    for s in sentences:
        if len(s) > max_chars:
            # 超长句子独立处理
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(chunk_by_size(s, chunk_size=max_chars, overlap=0))
        elif len(current) + len(s) <= max_chars:
            current += s
        else:
            if current:
                chunks.append(current)
            current = s

    if current:
        chunks.append(current)

    return chunks


def chunk_by_paragraph(text: str, max_chars: int = 1024) -> list[str]:
    """
    按段落（连续空行分隔）切块，长段落再切成固定大小。

    Args:
        text: 原始文本
        max_chars: 每块最大字符数

    Returns:
        list[str] 块列表
    """
    if not text:
        return []

    # 按两个以上换行分段落
    paragraphs = re.split(r'\n{2,}', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    for p in paragraphs:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            chunks.extend(chunk_by_size(p, chunk_size=max_chars, overlap=64))

    return chunks
