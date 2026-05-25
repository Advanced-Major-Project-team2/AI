# tests/test_recommand_preprocessing.py
"""app/processors/recommand_preprocessing.py 의 DTO 변환 함수 검증."""

from datetime import datetime, timezone

from app.models.enums import Category
from app.models.domain import (
    RoomRecommandUserMetaV1,
    RoomRecommandUserMetaV2,
    RoomRecommandRoomMetaV1,
)
from app.processors.recommand_preprocessing import (
    to_user_meta,
    to_room_meta_list,
    clustering_request_usermeta,
)

from tests.factories import make_reco_request, make_gathering


class TestToUserMeta:
    def test_maps_external_to_v1_meta(self):
        req = make_reco_request(user_id=7, preferred_categories=[Category.MUSIC], age=20)
        meta = to_user_meta(req)
        assert isinstance(meta, RoomRecommandUserMetaV1)
        assert meta.user_id == 7
        assert meta.user_age == 20
        assert meta.preferred_categories == [Category.MUSIC]

    def test_handles_none_fields(self):
        req = make_reco_request()  # 모두 None
        meta = to_user_meta(req)
        assert meta.user_id is None
        assert meta.user_age is None


class TestToRoomMetaList:
    def test_basic_mapping(self):
        g = make_gathering(gathering_id=10, category=Category.GAME, host_age=25,
                           capacity=8, participant_count=4)
        rooms = to_room_meta_list([g])
        assert len(rooms) == 1
        rm = rooms[0]
        assert isinstance(rm, RoomRecommandRoomMetaV1)
        assert rm.room_id == 10
        assert rm.host_age == 25
        assert rm.capacity_member == 8       # capacity -> capacity_member
        assert rm.current_member == 4        # participantCount -> current_member

    def test_category_enum_converted_to_str(self):
        g = make_gathering(category=Category.MUSIC)
        rm = to_room_meta_list([g])[0]
        assert rm.category == "음악"
        assert isinstance(rm.category, str)

    def test_naive_datetime_gets_utc_tzinfo(self):
        naive = datetime(2024, 1, 1, 12, 0, 0)  # tzinfo 없음
        g = make_gathering(created_at=naive)
        rm = to_room_meta_list([g])[0]
        assert rm.updated_at.tzinfo == timezone.utc

    def test_aware_datetime_preserved(self):
        aware = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        g = make_gathering(created_at=aware)
        rm = to_room_meta_list([g])[0]
        assert rm.updated_at == aware

    def test_empty_list_returns_empty(self):
        assert to_room_meta_list([]) == []


class TestClusteringRequestUsermeta:
    def test_maps_to_v2_meta(self):
        req = make_reco_request(
            user_id=3, preferred_categories=[Category.MUSIC, Category.GAME],
            age=22, enroll_number=2021, user_join_count=5,
        )
        meta = clustering_request_usermeta(req)
        assert isinstance(meta, RoomRecommandUserMetaV2)
        assert meta.user_id == 3
        assert meta.user_age == 22
        assert meta.user_enroll == 2021
        assert meta.user_join_count == 5
        assert meta.preferred_categories == [Category.MUSIC, Category.GAME]

    def test_none_fields(self):
        meta = clustering_request_usermeta(make_reco_request())
        assert meta.user_id is None
        assert meta.user_enroll is None
        assert meta.user_join_count is None
