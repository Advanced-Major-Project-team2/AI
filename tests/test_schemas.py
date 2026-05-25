# tests/test_schemas.py
"""app/models/schemas.py 의 Pydantic 모델 검증.

특히 GatheringIn.parse_created_at 의 모든 분기:
- datetime 그대로
- ISO 문자열
- LocalDateTime 배열 (6개 / 7개[+nano])
- 너무 짧은 배열 → ValueError
- 잘못된 타입 → ValueError
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from app.models.enums import Category, UserStatus
from app.models.schemas import (
    GatheringIn,
    RecommendByClusteringModelRequest,
    RecommendationResponse,
    ClusteringUserData,
    ClusterRefreshRequest,
    ClusterRefreshResponse,
    UserActionlog,
    PopularityRefreshRequest,
    PopularityRefreshResponse,
)


def _base_gathering(**overrides):
    data = dict(
        gatheringId=1,
        category=Category.MUSIC,
        hostAge=20,
        capacity=10,
        participantCount=5,
        createdAt=datetime(2024, 1, 1, 12, 0, 0),
    )
    data.update(overrides)
    return GatheringIn(**data)


class TestGatheringInBasics:
    def test_valid(self):
        g = _base_gathering()
        assert g.gatheringId == 1
        assert g.category == Category.MUSIC

    def test_gathering_id_must_be_positive(self):
        with pytest.raises(ValidationError):
            _base_gathering(gatheringId=0)

    def test_category_must_be_valid_enum(self):
        with pytest.raises(ValidationError):
            _base_gathering(category="존재하지않는카테고리")

    def test_host_age_upper_bound(self):
        with pytest.raises(ValidationError):
            _base_gathering(hostAge=3000)  # le=2100 초과


class TestParseCreatedAt:
    def test_datetime_passthrough(self):
        dt = datetime(2023, 5, 1, 9, 30, 0)
        g = _base_gathering(createdAt=dt)
        assert g.createdAt == dt

    def test_iso_string(self):
        g = _base_gathering(createdAt="2024-03-17T12:32:00")
        assert g.createdAt == datetime(2024, 3, 17, 12, 32, 0)

    def test_local_datetime_array_6_elements(self):
        g = _base_gathering(createdAt=[2024, 1, 2, 3, 4, 5])
        assert g.createdAt == datetime(2024, 1, 2, 3, 4, 5)

    def test_local_datetime_array_with_nano(self):
        # 7번째 원소는 나노초 → microsecond = nano // 1000
        g = _base_gathering(createdAt=[2024, 1, 2, 3, 4, 5, 123_000])
        assert g.createdAt == datetime(2024, 1, 2, 3, 4, 5, 123)

    def test_array_too_short_raises(self):
        with pytest.raises(ValidationError):
            _base_gathering(createdAt=[2024, 1, 2])

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError):
            _base_gathering(createdAt=12345)


class TestRecommendByClusteringModelRequest:
    def test_all_optional_defaults_to_none(self):
        req = RecommendByClusteringModelRequest()
        assert req.userId is None
        assert req.preferredCategories is None
        assert req.gatherings is None

    def test_user_id_must_be_positive(self):
        with pytest.raises(ValidationError):
            RecommendByClusteringModelRequest(userId=0)

    def test_preferred_categories_max_three(self):
        with pytest.raises(ValidationError):
            RecommendByClusteringModelRequest(
                preferredCategories=[Category.MUSIC, Category.GAME, Category.BOOK, Category.STUDY]
            )

    def test_age_range(self):
        with pytest.raises(ValidationError):
            RecommendByClusteringModelRequest(age=10)  # ge=20
        with pytest.raises(ValidationError):
            RecommendByClusteringModelRequest(age=200)  # le=100

    def test_valid_full(self):
        req = RecommendByClusteringModelRequest(
            userId=3, preferredCategories=[Category.MUSIC], age=25,
            enrollNumber=2021, userJoinCount=4,
        )
        assert req.userId == 3
        assert req.preferredCategories == [Category.MUSIC]


class TestSimpleResponseModels:
    def test_recommendation_response_default_empty(self):
        r = RecommendationResponse()
        assert r.gatheringsId == []

    def test_recommendation_response_with_ids(self):
        r = RecommendationResponse(gatheringsId=[5, 1, 9])
        assert r.gatheringsId == [5, 1, 9]

    def test_cluster_refresh_response(self):
        r = ClusterRefreshResponse(
            n_users=3, n_clusters=2, inertia=1.5, cluster_sizes={0: 2, 1: 1}
        )
        assert r.n_users == 3
        assert r.cluster_sizes == {0: 2, 1: 1}

    def test_popularity_refresh_response(self):
        r = PopularityRefreshResponse(
            total_logs=10, n_clusters=2, top_n=3,
            cluster_popularity={0: [1, 2, 3], 1: [4]},
        )
        assert r.total_logs == 10
        assert r.cluster_popularity[0] == [1, 2, 3]


class TestClusterAndPopularityRequests:
    def test_clustering_user_data_valid(self):
        u = ClusteringUserData(userId=1, preferredCategories=[Category.GAME], age=22)
        assert u.userId == 1

    def test_cluster_refresh_request(self):
        req = ClusterRefreshRequest(
            users=[ClusteringUserData(userId=1, preferredCategories=[Category.GAME], age=22)]
        )
        assert len(req.users) == 1

    def test_user_action_log_requires_gathering_id_positive(self):
        with pytest.raises(ValidationError):
            UserActionlog(userId=1, gatheringId=0, status=UserStatus.JOIN)

    def test_user_action_log_valid(self):
        log = UserActionlog(userId=1, gatheringId=10, status=UserStatus.CLICK)
        assert log.status == UserStatus.CLICK

    def test_popularity_refresh_request(self):
        req = PopularityRefreshRequest(
            logList=[UserActionlog(userId=1, gatheringId=10, status=UserStatus.JOIN)]
        )
        assert len(req.logList) == 1
