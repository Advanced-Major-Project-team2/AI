# tests/test_recommender_v2.py
"""app/services/v2/recommender.py 검증.

- __init__ : 아티팩트 로드 성공 / 실패(fallback 모드)
- rank :
    * fallback(1) 아티팩트 미로드 → None
    * fallback(2) 로그인 안 함(user_id None) → None
    * fallback(2) popularity 비어있음 → None
    * 정상 경로 → 인기 방 id 리스트
- predict_cluster : (cluster_id:int, X_scaled:ndarray) 반환

전제: trained_clustering_dir 픽스처(실제 학습된 아티팩트)가 필요.
"""

import json

import numpy as np
import pytest

from app.core import config as config_module
from app.models.enums import Category
from app.services.v2.recommender import Recommender as RecommenderV2
from app.processors.recommand_preprocessing import clustering_request_usermeta

from tests.factories import make_reco_request


@pytest.fixture
def popularity_dir(tmp_path, monkeypatch):
    """cluster_popularity.json 을 써 두고 settings.POPULARITY_ARTIFACT_DIR 로 지정."""
    pop_dir = tmp_path / "popularity"
    pop_dir.mkdir()
    # 클러스터 0~4 각각에 인기 방 id 부여
    payload = {"0": [100, 101], "1": [200], "2": [300], "3": [400], "4": [500]}
    (pop_dir / "cluster_popularity.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(config_module.settings, "POPULARITY_ARTIFACT_DIR", pop_dir)
    return pop_dir


@pytest.fixture
def v2(trained_clustering_dir, popularity_dir):
    """아티팩트가 모두 로드된 정상 v2 인스턴스."""
    return RecommenderV2(artifacts_dir=trained_clustering_dir)


class TestInit:
    def test_artifacts_loaded(self, v2):
        assert v2.artifacts_loaded is True
        assert len(v2.category_vocab) > 0
        assert v2.cluster_popularity  # 비어있지 않음

    def test_fallback_mode_when_missing(self, tmp_path, monkeypatch):
        # 빈 디렉토리 → scaler.pkl 없음 → FileNotFoundError → fallback 모드
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setattr(config_module.settings, "POPULARITY_ARTIFACT_DIR", tmp_path / "nope")
        r = RecommenderV2(artifacts_dir=empty)
        assert r.artifacts_loaded is False
        assert r.category_vocab == []
        assert r.cat_index == {}
        assert r.cluster_popularity == {}


class TestRankFallbacks:
    def test_fallback_1_artifacts_not_loaded(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setattr(config_module.settings, "POPULARITY_ARTIFACT_DIR", tmp_path / "nope")
        r = RecommenderV2(artifacts_dir=empty)
        req = make_reco_request(user_id=1, preferred_categories=[Category.MUSIC], age=20)
        assert r.rank(req) is None

    def test_fallback_2_anonymous_user(self, v2):
        # 아티팩트는 있지만 user_id 없음 → None
        req = make_reco_request(user_id=None, preferred_categories=[Category.MUSIC], age=20)
        assert v2.rank(req) is None

    def test_fallback_2_empty_popularity(self, trained_clustering_dir, tmp_path, monkeypatch):
        # popularity 파일이 없으면 cluster_popularity 가 {} → None
        monkeypatch.setattr(config_module.settings, "POPULARITY_ARTIFACT_DIR", tmp_path / "no_pop")
        r = RecommenderV2(artifacts_dir=trained_clustering_dir)
        assert r.cluster_popularity == {}
        req = make_reco_request(user_id=1, preferred_categories=[Category.MUSIC], age=20)
        assert r.rank(req) is None


class TestRankHappyPath:
    def test_returns_room_ids(self, v2):
        req = make_reco_request(user_id=1, preferred_categories=[Category.MUSIC],
                                age=22, enroll_number=2021, user_join_count=3)
        result = v2.rank(req)
        assert isinstance(result, list)
        assert len(result) > 0
        # 모든 반환값은 popularity 에 등록된 방 id 중 하나
        all_rooms = {r for rooms in v2.cluster_popularity.values() for r in rooms}
        assert set(result).issubset(all_rooms)

    def test_deduplicated_and_limited(self, v2):
        req = make_reco_request(user_id=1, preferred_categories=[Category.MUSIC],
                                age=22, enroll_number=2021, user_join_count=3)
        result = v2.rank(req, limit=2)
        assert len(result) <= 2
        assert len(result) == len(set(result))  # 중복 없음


class TestPredictCluster:
    def test_returns_cluster_id_and_scaled_vector(self, v2):
        req = make_reco_request(user_id=1, preferred_categories=[Category.MUSIC],
                                age=22, enroll_number=2021, user_join_count=3)
        user_meta = clustering_request_usermeta(req)
        cluster_id, x_scaled = v2.predict_cluster(user_meta)

        assert isinstance(cluster_id, int)
        assert 0 <= cluster_id < v2.kmeans.n_clusters
        assert isinstance(x_scaled, np.ndarray)
        # numeric(3) + svd 차원
        assert x_scaled.shape[0] == 1
        assert x_scaled.shape[1] == 3 + v2.svd.n_components
