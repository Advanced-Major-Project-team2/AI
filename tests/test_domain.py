# tests/test_domain.py
"""app/models/domain.py 의 내부 DTO(dataclass) 검증."""

from datetime import datetime, timezone

from app.models.domain import (
    RoomRecommandUserMetaV1,
    RoomRecommandUserMetaV2,
    RoomRecommandRoomMetaV1,
)
from app.models.enums import Category


class TestRoomRecommandUserMetaV1:
    def test_defaults(self):
        m = RoomRecommandUserMetaV1()
        assert m.user_id is None
        assert m.preferred_categories == tuple()
        assert m.user_age is None

    def test_explicit_values(self):
        m = RoomRecommandUserMetaV1(user_id=7, preferred_categories=("음악",), user_age=20)
        assert m.user_id == 7
        assert m.preferred_categories == ("음악",)
        assert m.user_age == 20

    def test_default_factory_is_independent(self):
        # field(default_factory=tuple) 이므로 인스턴스 간 공유되지 않아야 함
        a = RoomRecommandUserMetaV1()
        b = RoomRecommandUserMetaV1()
        assert a.preferred_categories is not b.preferred_categories or a.preferred_categories == ()


class TestRoomRecommandUserMetaV2:
    def test_defaults(self):
        m = RoomRecommandUserMetaV2()
        assert m.user_id is None
        assert m.preferred_categories == tuple()
        assert m.user_age is None
        assert m.user_enroll is None
        assert m.user_join_count is None
        assert m.cluster_id is None

    def test_explicit_values(self):
        m = RoomRecommandUserMetaV2(
            user_id=1,
            preferred_categories=("음악", "게임"),
            user_age=22,
            user_enroll=2021,
            user_join_count=5,
            cluster_id=3,
        )
        assert m.user_enroll == 2021
        assert m.user_join_count == 5
        assert m.cluster_id == 3


class TestRoomRecommandRoomMetaV1:
    def test_construction(self):
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        m = RoomRecommandRoomMetaV1(
            room_id=10,
            category=Category.MUSIC,
            host_age=20,
            capacity_member=8,
            current_member=3,
            updated_at=now,
        )
        assert m.room_id == 10
        assert m.category == Category.MUSIC
        assert m.host_age == 20
        assert m.capacity_member == 8
        assert m.current_member == 3
        assert m.updated_at == now

    def test_updated_at_optional(self):
        m = RoomRecommandRoomMetaV1(
            room_id=1, category=Category.GAME, host_age=20,
            capacity_member=5, current_member=1, updated_at=None,
        )
        assert m.updated_at is None
