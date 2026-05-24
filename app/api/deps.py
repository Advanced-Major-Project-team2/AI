from fastapi import Request

from app.core.exception import AppException, ErrorCode

from app.services.v1.recommender import Recommender as RecommenderV1
from app.services.v2.recommender import Recommender as RecommenderV2

from app.cluster.user_clustering import ClusteringTrainer
from app.cluster.gatherings_popularity import PopularityTrainer


def get_recommender_v1(request: Request) -> RecommenderV1:
    """
    서버 기동 시 app.state.recommender에 올려둔 서비스를 꺼내서 주입합니다.
    없으면 500으로 실패시켜 원인을 바로 알 수 있게 합니다.
    """
    recommender = getattr(request.app.state, "recommender_v1", None)
    if recommender is None:
        raise AppException(ErrorCode.RECOMMENDATION_FAILED, "recommender_v1이 초기화되지 않았습니다.")
    return recommender


def get_recommender_v2(request: Request) -> RecommenderV2:
    """
    서버 기동 시 app.state.recommender_v2에 올려둔
    RecommenderV2 인스턴스를 꺼내서 반환합니다.
    - 이 인스턴스는 내부에서 아티팩트(클러스터, 인기 방 테이블 등)를 캐싱하고,
    각 요청마다 같은 인스턴스를 재사용합니다.
    """
    recommender = getattr(request.app.state, "recommender_v2", None)
    if recommender is None:
        raise AppException(ErrorCode.ARTIFACT_NOT_LOADED, "recommender_v2가 초기화되지 않았습니다.")
    return recommender


# ------------------------------------------------------------------------------------------------
def get_clustering_service_dep(request: Request) -> ClusteringTrainer:
    """
    서버 기동 시 app.state.clustering_service에 올려둔
    ClusteringService 인스턴스를 꺼내서 반환합니다.
    초기화가 안 되어 있으면 500 에러를 발생시켜 문제를 바로 알 수 있게 합니다.
    """
    clustering_service = getattr(request.app.state, "clustering_service", None)
    if clustering_service is None:
        raise AppException(ErrorCode.CLUSTER_REFRESH_FAILED, "clustering_service가 초기화되지 않았습니다.")
    return clustering_service


def get_popularity_service_dep(request: Request) -> PopularityTrainer:
    """
    서버 기동 시 app.state.popularity_service에 올려둔
    PopularityService 인스턴스를 꺼내서 반환합니다.
    """
    popularity_service = getattr(request.app.state, "popularity_service", None)
    if popularity_service is None:
        raise AppException(ErrorCode.POPULARITY_REFRESH_FAILED, "popularity_service가 초기화되지 않았습니다.")
    return popularity_service
