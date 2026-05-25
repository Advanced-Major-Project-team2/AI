# tests/factories.py
# 역할: 여러 테스트에서 공통으로 쓰는 입력 객체 생성 헬퍼.
#       conftest 가 아닌 일반 모듈이라 에디터(Pylance)에서 import 가 깔끔히 해석됩니다.

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from app.models.enums import Category, UserStatus
from app.models.schemas import (
    GatheringIn,
    RecommendByClusteringModelRequest,
    ClusteringUserData,
    UserActionlog,
)


def make_gathering(
    *,
    gathering_id: int = 1,
    category: Category = Category.MUSIC,
    host_age: int = 20,
    capacity: int = 10,
    participant_count: int = 5,
    created_at: Optional[datetime] = None,
) -> GatheringIn:
    """GatheringIn 한 건을 손쉽게 만드는 헬퍼."""
    if created_at is None:
        created_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return GatheringIn(
        gatheringId=gathering_id,
        category=category,
        hostAge=host_age,
        capacity=capacity,
        participantCount=participant_count,
        createdAt=created_at,
    )


def make_reco_request(
    *,
    user_id: Optional[int] = None,
    preferred_categories: Optional[List[Category]] = None,
    age: Optional[int] = None,
    enroll_number: Optional[int] = None,
    user_join_count: Optional[int] = None,
    gatherings: Optional[List[GatheringIn]] = None,
) -> RecommendByClusteringModelRequest:
    """RecommendByClusteringModelRequest 헬퍼."""
    return RecommendByClusteringModelRequest(
        userId=user_id,
        preferredCategories=preferred_categories,
        age=age,
        enrollNumber=enroll_number,
        userJoinCount=user_join_count,
        gatherings=gatherings,
    )


def make_clustering_user(
    *,
    user_id: Optional[int] = None,
    preferred_categories: Optional[List[Category]] = None,
    age: Optional[int] = None,
    enroll_number: Optional[int] = None,
    user_join_count: Optional[int] = None,
) -> ClusteringUserData:
    """ClusteringUserData 헬퍼."""
    return ClusteringUserData(
        userId=user_id,
        preferredCategories=preferred_categories,
        age=age,
        enrollNumber=enroll_number,
        userJoinCount=user_join_count,
    )


def make_action_log(
    *,
    user_id: Optional[int],
    gathering_id: int,
    status: UserStatus,
) -> UserActionlog:
    """UserActionlog 헬퍼."""
    return UserActionlog(userId=user_id, gatheringId=gathering_id, status=status)
