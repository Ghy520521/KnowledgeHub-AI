"""数据库引擎、会话与声明式基类。"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


_connect_args = {}
if not settings.is_postgres:
    # SQLite（本地/测试回退）：关闭外键约束检查以外的线程限制
    _connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    """FastAPI 依赖：提供请求级数据库会话并保证关闭。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_pgvector_extension(db) -> None:
    """在 PostgreSQL 上启用 vector 扩展（SQLite 回退路径跳过）。"""
    if settings.is_postgres:
        db.execute(__import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector"))
        db.commit()
