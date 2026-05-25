# tests/conftest.py
# 역할:
#   - 프로젝트 루트를 sys.path 에 추가해 `app.*` / `tests.*` import 를 보장합니다.
#   - 프로젝트 루트의 .env 를 읽어 os.environ 에 주입합니다.
#   - 공통 픽스처를 제공합니다. (입력 생성 헬퍼는 tests/factories.py 참고)

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest
from dotenv import load_dotenv

# --- 프로젝트 루트(= app, tests 패키지의 부모)를 sys.path 에 추가 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- .env 로드 (RECOMMANDS_LIMIT 등) ---
load_dotenv(PROJECT_ROOT / ".env")

from app.models.schemas import ClusteringUserData  # noqa: E402
from tests.factories import make_clustering_user  # noqa: E402


# ----------------------------------------------------------------------
# 픽스처
# ----------------------------------------------------------------------
@pytest.fixture
def sample_clustering_users() -> List[ClusteringUserData]:
    """
    클러스터링 학습/전처리 테스트용 사용자 6명.
    - n_clusters(=5)보다 많고, 카테고리도 4종 이상 분포하도록 구성.
    """
    return [
        make_clustering_user(user_id=1, age=20, enroll_number=2022, user_join_count=3,
                             preferred_categories=["음악", "게임"]),
        make_clustering_user(user_id=2, age=22, enroll_number=2021, user_join_count=5,
                             preferred_categories=["게임"]),
        make_clustering_user(user_id=3, age=25, enroll_number=2020, user_join_count=1,
                             preferred_categories=["스터디"]),
        make_clustering_user(user_id=4, age=21, enroll_number=2023, user_join_count=8,
                             preferred_categories=["스포츠", "운동"]),
        make_clustering_user(user_id=5, age=28, enroll_number=2019, user_join_count=2,
                             preferred_categories=["독서"]),
        make_clustering_user(user_id=6, age=23, enroll_number=2021, user_join_count=4,
                             preferred_categories=["여행", "음악"]),
    ]


@pytest.fixture
def trained_clustering_dir(tmp_path, sample_clustering_users):
    """
    임시 디렉토리에 실제 클러스터링 아티팩트를 학습/저장하고 그 경로를 반환합니다.
    (scaler.pkl, svd.pkl, kmeans.pkl, category_vocab.json, user_clusters.json)

    v2 Recommender / PopularityTrainer 통합 테스트의 전제 조건으로 사용.
    """
    from app.cluster.user_clustering import ClusteringTrainer
    from app.models.schemas import ClusterRefreshRequest

    clu_dir = tmp_path / "user_clustering"
    trainer = ClusteringTrainer(artifacts_dir=clu_dir)
    trainer.refresh_clusters(ClusterRefreshRequest(users=sample_clustering_users))
    return clu_dir
