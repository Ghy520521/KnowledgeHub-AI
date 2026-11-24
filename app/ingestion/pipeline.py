"""文档摄入流水线：解析 -> 分块 -> 向量化 -> 落库。

同步路径供 API 与脚本调用；Celery 任务在 tasks.py 中复用本函数。
原始字节先写入本地暂存目录，便于同步流程与异步 worker 都能读取；
生产环境应替换为对象存储（S3/MinIO）。
"""
from __future__ import annotations

import logging
import os
import tempfile

from sqlalchemy.orm import Session

from app.config import settings
from app.ingestion import parser
from app.ingestion.chunker import chunk_text
from app.ingestion.embeddings import get_embedder
from app.models import Chunk, Document, DocumentStatus

logger = logging.getLogger(__name__)

_UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "knowledgehub_uploads")
os.makedirs(_UPLOAD_DIR, exist_ok=True)


def _pending_path(document_id: int) -> str:
    return os.path.join(_UPLOAD_DIR, f"{document_id}.bin")


def register_pending_bytes(document_id: int, data: bytes) -> None:
    with open(_pending_path(document_id), "wb") as f:
        f.write(data)


def process_document(db: Session, document_id: int) -> Document:
    doc = db.get(Document, document_id)
    if doc is None:
        raise ValueError(f"文档不存在: {document_id}")

    doc.status = DocumentStatus.PROCESSING
    db.commit()

    try:
        with open(_pending_path(document_id), "rb") as f:
            data = f.read()
        text = parser.extract_text(doc.source_filename, data)
        chunks = chunk_text(text, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        if not chunks:
            raise ValueError("解析结果为空，未生成任何分块")

        embedder = get_embedder()
        vectors = embedder.embed(chunks)

        db.query(Chunk).filter(Chunk.document_id == doc.id).delete()
        for idx, (content, vector) in enumerate(zip(chunks, vectors)):
            db.add(Chunk(document_id=doc.id, chunk_index=idx, content=content, embedding=vector))

        doc.chunk_count = len(chunks)
        doc.status = DocumentStatus.READY
        doc.error = None
        db.commit()
        logger.info("文档 %s 摄入完成，共 %s 块", doc.id, len(chunks))
    except Exception as exc:  # noqa: BLE001 - 兜底标记失败并保留原因
        db.rollback()
        doc = db.get(Document, document_id)
        doc.status = DocumentStatus.FAILED
        doc.error = str(exc)[:1000]
        db.commit()
        logger.exception("文档 %s 摄入失败", document_id)
        raise
    return doc
