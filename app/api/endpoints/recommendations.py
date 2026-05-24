# app/api/v2/endpoints/recommendations.py
# 역할:
# - 추천 API HTTP 엔드포인트(컨트롤러).
# - 요청 스키마 검증 후, 서비스(app/services/recommender.py)의 Recommender.rank() 호출.
import logging
from fastapi import APIRouter, Depends
from datetime import datetime, timezone

from app.api.deps import get_recommender_v1, get_recommender_v2

# 외부 DTO
from app.core.exception import AppException, ErrorCode
from app.models.schemas import (
    RecommendByClusteringModelRequest,
    RecommendationResponse,
)

# Recommender 서비스 인스턴스
from app.services.v2.recommender import Recommender as v2_recommender

from app.services.v1.recommender import (
    Recommender as fallback_recommender
)

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="",
    tags=["recommendations"],
)


@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
    summary="사용자 선호 기반 방 추천")
def recommend(
    req: RecommendByClusteringModelRequest,
    recommender_v2: v2_recommender = Depends(get_recommender_v2),
    recommender_v1: fallback_recommender = Depends(get_recommender_v1),
) -> RecommendationResponse:
    """
    req로 들어온 RecommendByClusteringModelRequest (user_id + user cluster 지정해주기 위한 필드들이 존재.)
    1. user_id가 없는 경우 -> v1에서 했던 것처럼 coldstart진행
    2. user_id존재 -> gatherings_popularity에서 cluster당 인기 방 response
    """

    # 0. 요청 바디 로그
    logger.info(
        "POST /recommendations 요청 수신 user_id=%s preferred_categories=%s num_gatherings=%s",
        req.userId,
        req.preferredCategories,
        len(req.gatherings) if req.gatherings else 0,
    )

    # V2 try-exception 시작
    items = None
    try:
        items = recommender_v2.rank(
            req=req,
        )
        logger.info("V2 rank 결과: %s", items)

    except Exception as e:
        logger.warning("V2 rank 실패, V1 fallback 진행: %s", e)
    # V2 추론 try-exception 종료

    # fallback try-exception 시작
    if items is None:
        try:
            items = recommender_v1.rank(
                req=req,
                now=datetime.now(timezone.utc)
            )
            logger.info("V1 rank 결과: %s", items)

        except Exception as e:
            logger.error("V1 rank 실패: %s", e)  # 추가
            raise AppException(ErrorCode.RECOMMENDATION_FAILED, str(e))
    # fallback try-exception 종료

    logger.info("최종 추천 결과 items=%s", items)
    return RecommendationResponse(gatheringsId=items)
