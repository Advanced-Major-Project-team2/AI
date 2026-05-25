# tests/test_health.py
"""헬스 체크 엔드포인트 검증.

- /api/ai/health (router 의 health.ping)
- /health (app 에 직접 등록된 health_check)
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_router_health():
    r = client.get("/api/ai/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_root_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
