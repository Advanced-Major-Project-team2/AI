# tests/test_endpoints_clustering.py
"""POST /api/ai/refresh/clustering, /api/ai/refresh/popularity 엔드포인트 검증.

서비스(get_clustering_service_dep / get_popularity_service_dep)를 가짜로 오버라이드.

검증 포인트:
- 빈 입력 → 400 (EMPTY_USER_LIST / INVALID_REQUEST)
- 정상 → 200 + 요약 응답
- 서비스가 일반 예외 → 500 (CLUSTER_REFRESH_FAILED)
- 서비스가 AppException → 그대로 전파(except AppException: raise)
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.deps import get_clustering_service_dep, get_popularity_service_dep
from app.core.exception import AppException, ErrorCode
from app.models.schemas import ClusterRefreshResponse, PopularityRefreshResponse

client = TestClient(app, raise_server_exceptions=False)

CLUSTER_URL = "/api/ai/refresh/clustering"
POP_URL = "/api/ai/refresh/popularity"


class _FakeService:
    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises

    def refresh_clusters(self, req):
        if self._raises is not None:
            raise self._raises
        return self._result

    def refresh_popularity(self, req):
        if self._raises is not None:
            raise self._raises
        return self._result


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


# 유효한 요청 바디 (1명 / 로그 1건)
VALID_USER = {"userId": 1, "preferredCategories": ["음악"], "age": 22}
VALID_LOG = {"userId": 1, "gatheringId": 10, "status": "JOIN"}


# ----------------------------------------------------------------------
class TestRefreshClustering:
    def test_empty_users_returns_400(self):
        app.dependency_overrides[get_clustering_service_dep] = lambda: _FakeService()
        r = client.post(CLUSTER_URL, json={"users": []})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "EMPTY_USER_LIST"

    def test_success_returns_summary(self):
        resp_obj = ClusterRefreshResponse(
            n_users=1, n_clusters=5, inertia=1.23, cluster_sizes={0: 1}
        )
        app.dependency_overrides[get_clustering_service_dep] = lambda: _FakeService(result=resp_obj)
        r = client.post(CLUSTER_URL, json={"users": [VALID_USER]})
        assert r.status_code == 200
        body = r.json()
        assert body["n_users"] == 1
        assert body["n_clusters"] == 5

    def test_generic_exception_becomes_500(self):
        app.dependency_overrides[get_clustering_service_dep] = (
            lambda: _FakeService(raises=RuntimeError("학습 실패"))
        )
        r = client.post(CLUSTER_URL, json={"users": [VALID_USER]})
        assert r.status_code == 500
        assert r.json()["error"]["code"] == "CLUSTER_REFRESH_FAILED"

    def test_app_exception_propagates_unchanged(self):
        # 서비스가 NO_GATHERINGS(400) 를 던지면 그대로 400 으로 전파되어야 함
        app.dependency_overrides[get_clustering_service_dep] = (
            lambda: _FakeService(raises=AppException(ErrorCode.NO_GATHERINGS))
        )
        r = client.post(CLUSTER_URL, json={"users": [VALID_USER]})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "NO_GATHERINGS"


# ----------------------------------------------------------------------
class TestRefreshPopularity:
    def test_empty_loglist_returns_400(self):
        app.dependency_overrides[get_popularity_service_dep] = lambda: _FakeService()
        r = client.post(POP_URL, json={"logList": []})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "FAIL_GET_LOGLIST"  # INVALID_REQUEST 의 value

    def test_success_returns_summary(self):
        resp_obj = PopularityRefreshResponse(
            total_logs=1, n_clusters=1, top_n=1, cluster_popularity={0: [10]}
        )
        app.dependency_overrides[get_popularity_service_dep] = lambda: _FakeService(result=resp_obj)
        r = client.post(POP_URL, json={"logList": [VALID_LOG]})
        assert r.status_code == 200
        body = r.json()
        assert body["total_logs"] == 1
        assert body["cluster_popularity"] == {"0": [10]}

    def test_generic_exception_becomes_500(self):
        app.dependency_overrides[get_popularity_service_dep] = (
            lambda: _FakeService(raises=RuntimeError("집계 실패"))
        )
        r = client.post(POP_URL, json={"logList": [VALID_LOG]})
        assert r.status_code == 500
        # 주의: 현재 구현은 popularity 실패도 CLUSTER_REFRESH_FAILED 로 래핑함
        assert r.json()["error"]["code"] == "CLUSTER_REFRESH_FAILED"

    def test_app_exception_propagates_unchanged(self):
        app.dependency_overrides[get_popularity_service_dep] = (
            lambda: _FakeService(raises=AppException(ErrorCode.POPULARITY_REFRESH_FAILED))
        )
        r = client.post(POP_URL, json={"logList": [VALID_LOG]})
        assert r.status_code == 500
        assert r.json()["error"]["code"] == "POPULARITY_REFRESH_FAILED"
