# tests/test_recommender_v1.py
"""app/services/v1/recommender.py 검증.

- CategoryIndex.__init__
- Recommender.__init__
- Recommender.rank : 콜드스타트 / 비콜드스타트 / limit / age None 예외
- Recommender._rank_coldstart : 인기도+최신도 결합 정렬
"""

from datetime import datetime, timezone

import pytest

from app.core.exception import AppException, ErrorCode
from app.models.enums import Category
from app.models.domain import RoomRecommandRoomMetaV1
from app.services.v1.recommender import Recommender, CategoryIndex

from tests.factories import make_reco_request, make_gathering

NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def recommender():
    return Recommender(cat_index=CategoryIndex())


class TestCategoryIndex:
    def test_index_and_dim(self):
        ci = CategoryIndex()
        # enum 정의 순서상 첫 항목은 SPORTS("스포츠")
        assert ci.index["스포츠"] == 0
        assert ci.dim == len(list(Category))
        # 모든 카테고리가 인덱싱됨
        assert set(ci.index.keys()) == {c.value for c in Category}


class TestRecommenderInit:
    def test_stores_cat_index(self):
        ci = CategoryIndex()
        r = Recommender(cat_index=ci)
        assert r.cat_index is ci


class TestRankNonColdstart:
    def test_scoring_and_order(self, recommender):
        # user: 음악 선호, 20세
        gatherings = [
            make_gathering(gathering_id=1, category=Category.MUSIC, host_age=20),  # 0.7+0.3=1.0
            make_gathering(gathering_id=2, category=Category.GAME, host_age=20),   # 0.0+0.3=0.3
            make_gathering(gathering_id=3, category=Category.MUSIC, host_age=40),  # 0.7+0.0=0.7
        ]
        req = make_reco_request(user_id=1, age=20,
                                preferred_categories=[Category.MUSIC],
                                gatherings=gatherings)
        result = recommender.rank(req, now=NOW)
        assert result == [1, 3, 2]

    def test_limit_applied(self, recommender):
        gatherings = [
            make_gathering(gathering_id=i, category=Category.MUSIC, host_age=20)
            for i in range(1, 6)
        ]
        req = make_reco_request(user_id=1, age=20,
                                preferred_categories=[Category.MUSIC],
                                gatherings=gatherings)
        result = recommender.rank(req, limit=2, now=NOW)
        assert len(result) == 2

    def test_age_none_raises_validation_error(self, recommender):
        req = make_reco_request(user_id=1, age=None,
                                preferred_categories=[Category.MUSIC],
                                gatherings=[make_gathering()])
        with pytest.raises(AppException) as ei:
            recommender.rank(req, now=NOW)
        assert ei.value.code == ErrorCode.VALIDATION_ERROR


class TestRankColdstart:
    def test_coldstart_when_no_user_id(self, recommender):
        # user_id 없음 → 콜드스타트(인기/최신) 경로
        gatherings = [
            make_gathering(gathering_id=1, participant_count=50, created_at=NOW),  # pop 1.0, recency 1.0
            make_gathering(gathering_id=2, participant_count=0, created_at=NOW),   # pop 0.0, recency 1.0
        ]
        req = make_reco_request(user_id=None, gatherings=gatherings)
        result = recommender.rank(req, now=NOW)
        assert result == [1, 2]

    def test_coldstart_empty_gatherings(self, recommender):
        req = make_reco_request(user_id=None, gatherings=[])
        assert recommender.rank(req, now=NOW) == []


class TestRankColdstartInternal:
    def test_rank_coldstart_direct(self, recommender):
        rooms = [
            RoomRecommandRoomMetaV1(room_id=10, category="음악", host_age=20,
                                    capacity_member=50, current_member=50, updated_at=NOW),
            RoomRecommandRoomMetaV1(room_id=20, category="게임", host_age=20,
                                    capacity_member=50, current_member=5, updated_at=NOW),
        ]
        result = recommender._rank_coldstart(rooms, now=NOW)
        # room10: 0.6*1.0 + 0.4*1.0 = 1.0 ; room20: 0.6*0.1 + 0.4*1.0 = 0.46
        assert result == [10, 20]

    def test_recency_decays_with_age(self, recommender):
        old = datetime(2024, 1, 1, tzinfo=timezone.utc)  # NOW 보다 한참 과거
        rooms = [
            RoomRecommandRoomMetaV1(room_id=1, category="음악", host_age=20,
                                    capacity_member=10, current_member=5, updated_at=NOW),
            RoomRecommandRoomMetaV1(room_id=2, category="음악", host_age=20,
                                    capacity_member=10, current_member=5, updated_at=old),
        ]
        result = recommender._rank_coldstart(rooms, now=NOW)
        # 인기도 동일, 최신도는 NOW(room1)가 더 높음 → room1 우선
        assert result[0] == 1

    def test_updated_at_none_gives_zero_recency(self, recommender):
        # updated_at 이 None 이면 recency 0 → 인기도만 반영
        rooms = [
            RoomRecommandRoomMetaV1(room_id=1, category="음악", host_age=20,
                                    capacity_member=10, current_member=50, updated_at=None),
            RoomRecommandRoomMetaV1(room_id=2, category="음악", host_age=20,
                                    capacity_member=10, current_member=0, updated_at=None),
        ]
        result = recommender._rank_coldstart(rooms, now=NOW)
        # room1: 0.6*1.0 + 0 = 0.6 > room2: 0
        assert result == [1, 2]
