"""答案生成：基于检索到的证据组织回答，证据不足时拒答，回答必带引用。"""
from __future__ import annotations

import httpx

from app.config import settings
from app.retrieval.hybrid import Candidate

REFUSAL_TEXT = (
    "根据当前知识库检索到的资料，证据不足以可靠回答该问题，暂不给出结论，"
    "请补充更多文档或换一种问法。"
)

SYSTEM_PROMPT = (
    "你是企业知识库助手。只能依据给定的参考资料回答，禁止编造资料中不存在的事实；"
    "若资料不足以回答，直接说明证据不足。回答末尾用 [n] 标注引用编号，"
    "引用编号与参考资料顺序一致。"
)


def _build_context(candidates: list[Candidate]) -> str:
    blocks = []
    for i, cand in enumerate(candidates, start=1):
        blocks.append(f"[{i}] 《{cand.document_title}》(分块 {cand.chunk_id}, 第 {cand.page or '-'} 页)\n{cand.content}")
    return "\n\n".join(blocks)


class OpenAICompatibleChat:
    def chat(self, system: str, user: str) -> str:
        resp = httpx.post(
            f"{settings.llm_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            json={
                "model": settings.llm_model,
                "temperature": settings.llm_temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class FakeChat:
    """离线确定性拼装：取前几条证据的要点组成答案，便于无密钥演示与测试。"""

    def chat(self, system: str, user: str) -> str:
        lines = [line.strip() for line in user.splitlines() if line.strip()]
        evidence = [ln for ln in lines if ln.startswith("[")]
        snippet = "；".join(e.split("\n", 1)[-1][:120] for e in evidence[:3])
        return f"根据知识库资料：{snippet}。以上内容均来自引用文档，未补充资料外信息。"


def _chat_client():
    return OpenAICompatibleChat() if settings.llm_provider == "openai_compatible" else FakeChat()


def generate_answer(
    query: str,
    candidates: list[Candidate],
) -> tuple[str, bool, list[dict]]:
    """返回 (answer, refused, citations)。

    证据门控：重排后最高分低于 min_evidence_score 即拒答，避免模型幻觉。
    """
    if not candidates:
        return REFUSAL_TEXT, True, []

    best_score = candidates[0].rerank_score
    if best_score is None:
        best_score = candidates[0].rrf_score
    if best_score < settings.min_evidence_score:
        return REFUSAL_TEXT, True, []

    context = _build_context(candidates)
    user_msg = f"问题：{query}\n\n参考资料：\n{context}"
    answer = _chat_client().chat(SYSTEM_PROMPT, user_msg)

    citations = [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "document_title": c.document_title,
            "source_filename": None,
            "page": c.page,
            "snippet": c.content[:200],
        }
        for c in candidates[: settings.final_top_k]
    ]
    return answer, False, citations
