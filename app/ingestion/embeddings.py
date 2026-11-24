"""Embedding 适配器：OpenAI 兼容接口 + 离线确定性 Fake。"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx

from app.config import settings


class EmbeddingProvider(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbeddings:
    """调用 OpenAI / 兼容服务的 /embeddings 接口。"""

    def __init__(self) -> None:
        self.dim = settings.embedding_dim
        self.base_url = settings.embedding_base_url.rstrip("/")
        self.model = settings.embedding_model
        self.api_key = settings.embedding_api_key

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]


class FakeEmbeddings:
    """离线确定性向量：对词元哈希散列到各维并 L2 归一化，仅用于本地演示与测试。

    同一文本永远得到同一向量；语义相近性由词面重叠近似，不代表真实语义。
    """

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.embedding_dim

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in self._tokens(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


def get_embedder() -> EmbeddingProvider:
    if settings.embedding_provider == "openai_compatible":
        return OpenAICompatibleEmbeddings()
    return FakeEmbeddings()
