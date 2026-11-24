"""Celery 应用与异步任务：文档解析/分块/向量化耗时，投递到 worker 执行。"""
from __future__ import annotations

from celery import Celery

from app.config import settings
from app.database import SessionLocal
from app.ingestion.pipeline import process_document

celery_app = Celery(
    "knowledgehub",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.task_ignore_result = True


@celery_app.task(name="ingest_document", bind=True)
def ingest_document_task(self, document_id: int) -> None:
    db = SessionLocal()
    try:
        process_document(db, document_id)
    finally:
        db.close()
