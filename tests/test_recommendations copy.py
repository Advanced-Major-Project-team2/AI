# tests/test_recommendations.py
"""POST /api/ai/recommendations 엔드포인트 검증.

의존성(get_recommender_v1/v2)을 가짜로 오버라이드하여
v2 → v1 fallback 분기를 검증한다.

- v2 가 결과 반환 → 그대로 사용 (v1 미사용)
- v2 가 None → v1 fallback 사용
- v2 가 예외 → v1 fallback 사용
- v2 None + v1 예외 → 500 RECOMMENDATION_FAILED
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.deps import get_recommender_v1, get_recommender_v2

client = TestClient(app, raise_server_exceptions=False)

URL = "/api/ai/recommendations"


class _FakeRecommender:
    """rank() 호출 시 미리 지정한 값을 반환하거나 예외를 던지는 가짜 추천기."""
    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises
        self.called = False

    def rank(self, *args, **kwargs):
        self.called = True
        if self._raises is not None:
            raise self._raises
        return self._result


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def _override(v2, v1):
    app.dependency_overrides[get_recommender_v2] = lambda: v2
    app.dependency_overrides[get_recommender_v1] = lambda: v1


def test_v2_result_used_directly():
    v2 = _FakeRecommender(result=[1, 2, 3])
    v1 = _FakeRecommender(result=[9])
    _override(v2, v1)

    r = client.post(URL, json={"userId": 1, "age": 22})
    assert r.status_code == 200
    assert r.json() == {"gatheringsId": [1, 2, 3]}
    assert v1.called is False  # v2 성공 시 v1 은 호출되지 않아야 함


def test_v1_fallback_when_v2_returns_none():
    v2 = _FakeRecommender(result=None)
    v1 = _FakeRecommender(result=[9, 8])
    _override(v2, v1)

    r = client.post(URL, json={})
    assert r.status_code == 200
    assert r.json() == {"gatheringsId": [9, 8]}
    assert v1.called is True


def test_v1_fallback_when_v2_raises():
    v2 = _FakeRecommender(raises=RuntimeError("v2 폭발"))
    v1 = _FakeRecommender(result=[7])
    _override(v2, v1)

    r = client.post(URL, json={})
    assert r.status_code == 200
    assert r.json() == {"gatheringsId": [7]}


def test_500_when_both_fail():
    v2 = _FakeRecommender(result=None)
    v1 = _FakeRecommender(raises=RuntimeError("v1 폭발"))
    _override(v2, v1)

    r = client.post(URL, json={})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "RECOMMENDATION_FAILED"


def test_invalid_body_returns_422():
    # age 는 ge=20 제약 → 5 는 검증 실패 → RequestValidationError 핸들러(422)
    _override(_FakeRecommender(result=[1]), _FakeRecommender(result=[1]))
    r = client.post(URL, json={"age": 5})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"
