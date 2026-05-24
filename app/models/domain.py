# app/models/domain.py
# 역할: 내부 도메인/계산에 최적화된 경량 구조체(dataclass 등)를 정의,
#      검증 오버헤드가 적고, 서비스 계층에서 대량 계산에서 사용.
# 사용처: app/services/* (비즈니스 로직), 레포지토리 계층

from dataclasses import dataclass, field
from typing import Optional, Tuple
from datetime import datetime

from app.models.enums import Category


@dataclass
class RoomRecommandUserMetaV1:
    """
    추천 시 필요한 사용자 신호만 담는 내부용 DTO.
    - 외부 요청과 1:1이 아니며, 엔드포인트에서 가공해 만듭니다.
    """

    '''
    user_id -> 사용자들을 구별하기 위함.
    preferred_categories -> 사용자의 선호 카테고리를 확인 및 비교를 위함
    student_year -> 나이가 비슷한 호스트 있는 방을 추천하기 위함.
    '''
    user_id: Optional[int] = None
    preferred_categories: Tuple[str, ...] = field(default_factory=tuple)
    user_age: Optional[int] = None


@dataclass
class RoomRecommandUserMetaV2:
    """
    추천 시 필요한 사용자 신호만 담는 내부용 DTO.
    - 외부 요청과 1:1이 아니며, 엔드포인트에서 가공해 만듭니다.
    """

    '''
    user_id -> 사용자들을 구별하기 위함.
    preferred_categories -> 사용자의 선호 카테고리를 확인 및 비교를 위함
    student_year -> 나이가 비슷한 호스트 있는 방을 추천하기 위함.
    '''
    user_id: Optional[int] = None
    preferred_categories: Tuple[str, ...] = field(default_factory=tuple)
    user_age: Optional[int] = None
    user_enroll: Optional[int] = None
    user_join_count: Optional[int] = None
    # 해당 유저의 cluster 정보를 미리 받기 위함.(없음 말고)
    cluster_id: Optional[int] = None


@dataclass
class RoomRecommandRoomMetaV1:
    """
    추천 점수 계산을 위한 최소 필드만 담는 내부용 DTO.
    - DB/ORM 전체 모델과 1:1이 아닙니다(가벼움이 목적).
    """
    room_id: int
    category: Category
    host_age: int
    # 콜드스타트 경로에서만 사용
    capacity_member: int
    current_member: int
    updated_at: Optional[datetime]
