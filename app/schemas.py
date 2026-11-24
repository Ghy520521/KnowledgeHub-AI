"""Pydantic 请求/响应模型。"""
from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, EmailStr, Field


# ---------- Auth ----------
class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=128)


class UserOut(BaseModel):
    id: int
    email: EmailStr
    username: str
    role: str

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ---------- Documents ----------
class DocumentOut(BaseModel):
    id: int
    title: str
    source_filename: str
    file_type: str
    status: str
    error: str | None = None
    chunk_count: int
    created_at: dt.datetime

    model_config = {"from_attributes": True}


# ---------- Retrieval ----------
class Citation(BaseModel):
    chunk_id: int
    document_id: int
    document_title: str
    source_filename: str
    page: int | None = None
    snippet: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    strategy: str = Field(default="hybrid", pattern="^(vector|keyword|hybrid)$")
    top_k: int | None = Field(default=None, ge=1, le=50)


class ScoredChunk(BaseModel):
    chunk_id: int
    document_id: int
    document_title: str
    page: int | None = None
    score: float
    content: str


class SearchResponse(BaseModel):
    query: str
    strategy: str
    results: list[ScoredChunk]


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    history: list[dict] = Field(default_factory=list)


class AskResponse(BaseModel):
    query: str
    answer: str
    refused: bool
    citations: list[Citation]
    used_strategy: str = "hybrid"
