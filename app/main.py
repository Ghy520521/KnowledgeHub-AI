"""FastAPI 入口：装配路由、CORS 与启动初始化。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import auth, documents, retrieval
from app.database import Base, SessionLocal, engine, init_pgvector_extension


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：开发期直接建表；生产使用 alembic 迁移
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        init_pgvector_extension(db)
    finally:
        db.close()
    yield


app = FastAPI(
    title="KnowledgeHub AI",
    version="0.1.0",
    description="企业知识库与混合检索平台",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(retrieval.router)

# 静态前端页面：/ui 访问单页应用，根路径重定向到 /ui
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
