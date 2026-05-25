# tests/test_deps.py
"""app/api/deps.py 의 의존성 주입 함수 검증.

각 함수는 request.app.state 에 올라간 서비스를 꺼내고,
없으면 적절한 ErrorCode 로 AppException 을 발생시켜야 한다.
"""

from types import SimpleNamespace

import pytest

from app.api.deps import (
    get_recommender_v1,
    get_recommender_v2,
    get_clustering_service_dep,
    get_popularity_service_dep,
)
from app.core.exception import AppException, ErrorCode


def _request_with_state(**state):
    """app.state 에 주어진 속성을 가진 가짜 Request 생성."""
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(**state)))


class TestGetRecommenderV1:
    def test_returns_when_present(self):
        sentinel = object()
        req = _request_with_state(recommender_v1=sentinel)
        assert get_recommender_v1(req) is sentinel

    def test_raises_when_missing(self):
        req = _request_with_state()  # 속성 없음 → None
        with pytest.raises(AppException) as ei:
            get_recommender_v1(req)
        assert ei.value.code == ErrorCode.RECOMMENDATION_FAILED

    def test_raises_when_none(self):
        req = _request_with_state(recommender_v1=None)
        with pytest.raises(AppException):
            get_recommender_v1(req)


class TestGetRecommenderV2:
    def test_returns_when_present(self):
        sentinel = object()
        req = _request_with_state(recommender_v2=sentinel)
        assert get_recommender_v2(req) is sentinel

    def test_raises_artifact_not_loaded_when_missing(self):
        req = _request_with_state()
        with pytest.raises(AppException) as ei:
            get_recommender_v2(req)
        assert ei.value.code == ErrorCode.ARTIFACT_NOT_LOADED


class TestGetClusteringServiceDep:
    def test_returns_when_present(self):
        sentinel = object()
        req = _request_with_state(clustering_service=sentinel)
        assert get_clustering_service_dep(req) is sentinel

    def test_raises_when_missing(self):
        req = _request_with_state()
        with pytest.raises(AppException) as ei:
            get_clustering_service_dep(req)
        assert ei.value.code == ErrorCode.CLUSTER_REFRESH_FAILED


class TestGetPopularityServiceDep:
    def test_returns_when_present(self):
        sentinel = object()
        req = _request_with_state(popularity_service=sentinel)
        assert get_popularity_service_dep(req) is sentinel

    def test_raises_when_missing(self):
        req = _request_with_state()
        with pytest.raises(AppException) as ei:
            get_popularity_service_dep(req)
        assert ei.value.code == ErrorCode.POPULARITY_REFRESH_FAILED
