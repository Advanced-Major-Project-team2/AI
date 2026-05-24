# app/processors/recommend_preprocessing.py
# 역할:
# endpoint 걔충 -> schemas 사용
# service계층 -> domain 사용

# endpoint에서 외부 DTO를 서비스 계층에 넘겨줄 때 내부 DTo로 변환해주기 위함.

# app/processors/recommend_preprocessing.py

from datetime import timezone
from typing import Iterable, List
from app.models.schemas import GatheringIn, RecommendByClusteringModelRequest
from app.models.domain import RoomRecommandRoomMetaV1, RoomRecommandUserMetaV1, RoomRecommandUserMetaV2
from app.models.enums import Category


# =============================V1=============================
def to_user_meta(req: RecommendByClusteringModelRequest) -> RoomRecommandUserMetaV1:
    """
    v2 fallback 정책으로 사용되며 서비스 계층에서 호출해서 사용
    """
    return RoomRecommandUserMetaV1(
        user_id=req.userId,
        preferred_categories=req.preferredCategories,
        user_age=req.age
    )


def to_room_meta_list(gatherings: Iterable[GatheringIn]) -> List[RoomRecommandRoomMetaV1]:
    """
    서비스 계층에서 req.gatherings -> 서비스 내부 DTO로 변환
    """
    out: List[RoomRecommandRoomMetaV1] = []
    for g in gatherings:
        created_at = g.createdAt
        if created_at is not None and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        out.append(
            RoomRecommandRoomMetaV1(
                room_id=g.gatheringId,  # OK
                # Enum -> str 변환
                category=(g.category.value if isinstance(g.category, Category) else g.category),
                host_age=g.hostAge,
                # 도메인 필드명에 맞춰 매핑(★ 중요)
                capacity_member=g.capacity,
                current_member=g.participantCount,
                updated_at=created_at,
            )
        )
    return out


# =============================V2=============================
def clustering_request_usermeta(req: RecommendByClusteringModelRequest) -> RoomRecommandUserMetaV2:
    return RoomRecommandUserMetaV2(
        user_id=req.userId,
        preferred_categories=req.preferredCategories,
        user_age=req.age,
        user_enroll=req.enrollNumber,
        user_join_count=req.userJoinCount,
    )
