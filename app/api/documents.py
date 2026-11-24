"""文档路由：上传、列表、详情、删除。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.ingestion.pipeline import process_document, register_pending_bytes
from app.models import Document, DocumentStatus
from app.schemas import DocumentOut
from app.deps import get_current_user
from app.models import User

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED = {".txt", ".md", ".pdf", ".docx"}


@router.post("/upload", response_model=DocumentOut, status_code=201)
async def upload_document(
    file: UploadFile,
    title: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Document:
    ext = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED:
        raise HTTPException(400, detail=f"仅支持 {', '.join(sorted(ALLOWED))}")

    data = await file.read()
    doc = Document(
        title=title or (file.filename or "未命名文档"),
        source_filename=file.filename or "upload",
        file_type=ext.lstrip("."),
        status=DocumentStatus.PENDING,
        uploaded_by=user.id,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    register_pending_bytes(doc.id, data)

    if settings.task_queue_enabled:
        # 异步：投递到 Celery worker
        from app.tasks import ingest_document_task

        ingest_document_task.delay(doc.id)
    else:
        # 同步：请求内完成解析与向量化（本地演示/测试）
        process_document(db, doc.id)
        db.refresh(doc)
    return doc


@router.get("", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[Document]:
    return db.scalars(select(Document).order_by(Document.created_at.desc())).all()


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> Document:
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, detail="文档不存在")
    return doc


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> None:
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, detail="文档不存在")
    db.delete(doc)
    db.commit()
