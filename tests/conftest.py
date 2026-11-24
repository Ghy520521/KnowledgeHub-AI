"""pytest 公共夹具：测试前把环境切到 SQLite + 离线 fake 适配器。

必须在任何 app.* 模块导入之前设置环境变量，因此放在本文件顶部。
"""
from __future__ import annotations

import os
import tempfile

# 测试使用临时 SQLite 文件，避免依赖 PostgreSQL/Redis/外部模型服务
_TMP_DB = os.path.join(tempfile.gettempdir(), "kf_test.db")
for p in (_TMP_DB,):
    if os.path.exists(p):
        os.remove(p)

os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ["EMBEDDING_PROVIDER"] = "fake"
os.environ["RERANKER_PROVIDER"] = "fallback"
os.environ["LLM_PROVIDER"] = "fake"
os.environ["TASK_QUEUE_ENABLED"] = "false"

import pytest  # noqa: E402


@pytest.fixture()
def db_session():
    from app.database import Base, engine, SessionLocal

    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
