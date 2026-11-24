"""API 层回归：注册 -> 登录 -> 上传 -> 检索 -> 问答 -> 拒答。"""
from fastapi.testclient import TestClient

from app.database import Base, engine
from app.main import app

Base.metadata.create_all(bind=engine)
client = TestClient(app)


def test_full_flow():
    r = client.post(
        "/auth/register",
        json={"email": "demo@example.com", "username": "demouser", "password": "123456"},
    )
    assert r.status_code == 201

    r = client.post("/auth/login", data={"username": "demouser", "password": "123456"})
    assert r.status_code == 200
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    files = {"file": ("faq.txt", "耳机断连：放回充电盒10秒重新取出并重新配对。", "text/plain")}
    r = client.post("/documents/upload", files=files, headers=headers)
    assert r.status_code == 201
    assert r.json()["status"] == "ready"

    r = client.post(
        "/search",
        json={"query": "耳机断连怎么办", "strategy": "hybrid"},
        headers=headers,
    )
    assert r.status_code == 200
    assert len(r.json()["results"]) >= 1

    r = client.post("/ask", json={"query": "耳机断连怎么办"}, headers=headers)
    body = r.json()
    assert r.status_code == 200
    assert body["refused"] is False
    assert len(body["citations"]) >= 1

    r = client.post("/ask", json={"query": "如何在火星上种植水稻"}, headers=headers)
    assert r.json()["refused"] is True
