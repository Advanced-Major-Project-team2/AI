# tests/test_gatherings_popularity.py
"""app/cluster/gatherings_popularity.py 의 PopularityTrainer 검증.

- __init__ : user_clusters.json 로드(있을 때/없을 때)
- refresh_popularity :
    * JOIN(1.0) / CLICK(0.1) 가중치
    * 익명 유저(userId None) 스킵
    * 군집 매핑 없는 유저 스킵
    * 점수순 정렬
    * cluster_popularity.json 저장 + 응답 요약
"""

import json

import pytest

from app.cluster.gatherings_popularity import PopularityTrainer
from app.core import config as config_module
from app.models.enums import UserStatus
from app.models.schemas import PopularityRefreshRequest

from tests.factories import make_action_log


@pytest.fixture
def patched_dirs(tmp_path, monkeypatch):
    """
    settings.CLUSTER_ARTIFACT_DIR / POPULARITY_ARTIFACT_DIR 를 임시 경로로 교체하고,
    user_clusters.json(매핑)을 미리 작성해 둔다.
    반환: (cluster_dir, popularity_dir)
    """
    cluster_dir = tmp_path / "user_clustering"
    pop_dir = tmp_path / "popularity"
    cluster_dir.mkdir(parents=True)
    pop_dir.mkdir(parents=True)

    # user 1,2 -> cluster 0 / user 3 -> cluster 1
    mapping = {"1": 0, "2": 0, "3": 1}
    (cluster_dir / "user_clusters.json").write_text(
        json.dumps(mapping), encoding="utf-8"
    )

    monkeypatch.setattr(config_module.settings, "CLUSTER_ARTIFACT_DIR", cluster_dir)
    monkeypatch.setattr(config_module.settings, "POPULARITY_ARTIFACT_DIR", pop_dir)
    return cluster_dir, pop_dir


class TestInit:
    def test_loads_user_clusters_mapping(self, patched_dirs):
        cluster_dir, pop_dir = patched_dirs
        trainer = PopularityTrainer(artifacts_dir=pop_dir)
        assert trainer.user_clusters == {"1": 0, "2": 0, "3": 1}

    def test_empty_mapping_when_file_missing(self, tmp_path, monkeypatch):
        cluster_dir = tmp_path / "no_mapping"
        pop_dir = tmp_path / "popularity"
        cluster_dir.mkdir()
        monkeypatch.setattr(config_module.settings, "CLUSTER_ARTIFACT_DIR", cluster_dir)
        trainer = PopularityTrainer(artifacts_dir=pop_dir)
        assert trainer.user_clusters == {}

    def test_creates_popularity_dir(self, tmp_path, monkeypatch):
        cluster_dir = tmp_path / "c"
        cluster_dir.mkdir()
        monkeypatch.setattr(config_module.settings, "CLUSTER_ARTIFACT_DIR", cluster_dir)
        pop_dir = tmp_path / "pop_new"
        PopularityTrainer(artifacts_dir=pop_dir)
        assert pop_dir.exists()


class TestRefreshPopularity:
    def test_join_outranks_click(self, patched_dirs):
        _, pop_dir = patched_dirs
        trainer = PopularityTrainer(artifacts_dir=pop_dir)

        # cluster 0 (user1): room 100 에 JOIN 2회(2.0), room 200 에 CLICK 1회(0.1)
        req = PopularityRefreshRequest(logList=[
            make_action_log(user_id=1, gathering_id=100, status=UserStatus.JOIN),
            make_action_log(user_id=1, gathering_id=100, status=UserStatus.JOIN),
            make_action_log(user_id=2, gathering_id=200, status=UserStatus.CLICK),
        ])
        resp = trainer.refresh_popularity(req)

        # 점수: room100(2.0) > room200(0.1) → 정렬 결과 [100, 200]
        assert resp.cluster_popularity[0] == [100, 200]

    def test_anonymous_and_unmapped_users_skipped(self, patched_dirs):
        _, pop_dir = patched_dirs
        trainer = PopularityTrainer(artifacts_dir=pop_dir)

        req = PopularityRefreshRequest(logList=[
            make_action_log(user_id=None, gathering_id=100, status=UserStatus.JOIN),   # 익명 → 스킵
            make_action_log(user_id=999, gathering_id=100, status=UserStatus.JOIN),    # 매핑 없음 → 스킵
            make_action_log(user_id=1, gathering_id=101, status=UserStatus.JOIN),      # 유효
        ])
        resp = trainer.refresh_popularity(req)

        # 유효 로그는 cluster 0 의 room 101 하나뿐
        assert resp.cluster_popularity == {0: [101]}
        assert resp.n_clusters == 1

    def test_response_summary_fields(self, patched_dirs):
        _, pop_dir = patched_dirs
        trainer = PopularityTrainer(artifacts_dir=pop_dir)

        req = PopularityRefreshRequest(logList=[
            make_action_log(user_id=1, gathering_id=100, status=UserStatus.JOIN),
            make_action_log(user_id=1, gathering_id=200, status=UserStatus.JOIN),
            make_action_log(user_id=3, gathering_id=300, status=UserStatus.JOIN),
        ])
        resp = trainer.refresh_popularity(req)

        assert resp.total_logs == 3            # 전체 로그 수(스킵 포함)
        assert resp.n_clusters == 2            # cluster 0, 1
        assert resp.top_n == 2                 # cluster0 이 방 2개 → 최댓값 2

    def test_writes_json_file(self, patched_dirs):
        _, pop_dir = patched_dirs
        trainer = PopularityTrainer(artifacts_dir=pop_dir)

        req = PopularityRefreshRequest(logList=[
            make_action_log(user_id=1, gathering_id=100, status=UserStatus.JOIN),
        ])
        trainer.refresh_popularity(req)

        path = pop_dir / "cluster_popularity.json"
        assert path.exists()
        saved = json.loads(path.read_text(encoding="utf-8"))
        # 저장 형식: {"0": [100]} (키는 문자열)
        assert saved == {"0": [100]}

    def test_no_valid_logs_returns_empty(self, patched_dirs):
        _, pop_dir = patched_dirs
        trainer = PopularityTrainer(artifacts_dir=pop_dir)

        req = PopularityRefreshRequest(logList=[
            make_action_log(user_id=None, gathering_id=1, status=UserStatus.JOIN),
        ])
        resp = trainer.refresh_popularity(req)
        assert resp.cluster_popularity == {}
        assert resp.n_clusters == 0
        assert resp.top_n == 0
