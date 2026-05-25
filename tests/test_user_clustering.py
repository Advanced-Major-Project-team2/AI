# tests/test_user_clustering.py
"""app/cluster/user_clustering.py 전체 메서드 단위 테스트.

대상:
- ClusterModelArtifacts.__init__
- ClusteringTrainer.__init__
- ClusteringTrainer.refresh_clusters (통합: 학습 + 저장 + 요약)
- _build_category_vocab
- _build_multi_hot_matrix
- _fit_svd (정상 / 카테고리 없음)
- _build_numeric_matrix
- _count_cluster_sizes
- _save_artifacts
"""

from typing import List

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from app.cluster.user_clustering import ClusteringTrainer, ClusterModelArtifacts
from app.core.config import settings
from app.models.schemas import ClusteringUserData, ClusterRefreshRequest

from tests.factories import make_clustering_user


@pytest.fixture
def trainer(tmp_path) -> ClusteringTrainer:
    # 아티팩트가 임시 디렉토리에 쓰이도록 하여 레포 오염 방지
    return ClusteringTrainer(artifacts_dir=tmp_path / "user_clustering")


@pytest.fixture
def three_users() -> List[ClusteringUserData]:
    return [
        make_clustering_user(user_id=1, age=20, enroll_number=2022, user_join_count=3,
                             preferred_categories=["음악", "게임"]),
        make_clustering_user(user_id=2, age=22, enroll_number=2021, user_join_count=5,
                             preferred_categories=["게임"]),
        make_clustering_user(user_id=3, age=None, enroll_number=None, user_join_count=None,
                             preferred_categories=["스터디"]),
    ]


# ----------------------------------------------------------------------
class TestClusterModelArtifacts:
    def test_stores_attributes(self):
        scaler, svd, kmeans, vocab = object(), object(), object(), ["음악"]
        art = ClusterModelArtifacts(scaler=scaler, svd=svd, kmeans=kmeans, category_vocab=vocab)
        assert art.scaler is scaler
        assert art.svd is svd
        assert art.kmeans is kmeans
        assert art.category_vocab == ["음악"]


class TestTrainerInit:
    def test_creates_artifacts_dir(self, tmp_path):
        target = tmp_path / "nested" / "user_clustering"
        ClusteringTrainer(artifacts_dir=target)
        assert target.exists() and target.is_dir()

    def test_default_dir_from_settings(self):
        t = ClusteringTrainer(artifacts_dir=None)
        assert str(t.artifacts_dir) == str(settings.CLUSTER_ARTIFACT_DIR)


# ----------------------------------------------------------------------
class TestBuildCategoryVocab:
    def test_unique_and_sorted(self, trainer, three_users):
        vocab = trainer._build_category_vocab(three_users)
        # 유니크
        assert len(vocab) == len(set(vocab))
        # 값 집합
        assert {str(v) for v in vocab} == {"음악", "게임", "스터디"}
        # 정렬되어 있음(deterministic)
        assert list(vocab) == sorted(vocab)

    def test_dedup_across_users(self, trainer):
        # 여러 사용자가 같은 카테고리를 가져도 vocab 에는 한 번만 등장해야 함
        users = [
            make_clustering_user(user_id=1, preferred_categories=["음악", "게임"]),
            make_clustering_user(user_id=2, preferred_categories=["게임"]),
            make_clustering_user(user_id=3, preferred_categories=["음악"]),
        ]
        vocab = trainer._build_category_vocab(users)
        assert {str(v) for v in vocab} == {"음악", "게임"}
        assert len(vocab) == 2


class TestBuildNumericMatrix:
    def test_shape_and_values_and_none_to_zero(self, trainer, three_users):
        mat = trainer._build_numeric_matrix(three_users)
        assert mat.shape == (3, 3)
        # user0
        assert (mat[0] == np.array([20, 2022, 3], dtype=np.float32)).all()
        # user1
        assert (mat[1] == np.array([22, 2021, 5], dtype=np.float32)).all()
        # user2: None → 0
        assert (mat[2] == np.array([0, 0, 0], dtype=np.float32)).all()


class TestBuildMultiHotMatrix:
    def test_multi_hot(self, trainer, three_users):
        vocab = ["스터디", "게임", "음악"]
        mat = trainer._build_multi_hot_matrix(three_users, vocab)
        assert mat.shape == (3, 3)
        np.testing.assert_array_equal(mat[0], np.array([0, 1, 1], dtype=np.float32))  # 음악,게임
        np.testing.assert_array_equal(mat[1], np.array([0, 1, 0], dtype=np.float32))  # 게임
        np.testing.assert_array_equal(mat[2], np.array([1, 0, 0], dtype=np.float32))  # 스터디

    def test_unknown_category_ignored(self, trainer):
        users = [make_clustering_user(user_id=1, preferred_categories=["음악"])]
        vocab = ["게임"]  # '음악'이 vocab에 없음
        mat = trainer._build_multi_hot_matrix(users, vocab)
        np.testing.assert_array_equal(mat[0], np.array([0], dtype=np.float32))


class TestFitSvd:
    def test_reduces_dimension(self, trainer, three_users):
        vocab = ["스터디", "게임", "음악"]
        multi_hot = trainer._build_multi_hot_matrix(three_users, vocab)
        svd, features = trainer._fit_svd(multi_hot, n_components=2)
        assert svd.n_components == 2
        assert features.shape == (3, 2)

    def test_n_components_capped(self, trainer, three_users):
        # 요청 10이지만 n_cats=3, n_users=3 → eff=3
        vocab = ["스터디", "게임", "음악"]
        multi_hot = trainer._build_multi_hot_matrix(three_users, vocab)
        svd, features = trainer._fit_svd(multi_hot, n_components=10)
        assert svd.n_components == 3
        assert features.shape == (3, 3)

    def test_no_categories_branch(self, trainer):
        multi_hot = np.zeros((3, 0), dtype=np.float32)
        svd, features = trainer._fit_svd(multi_hot, n_components=4)
        assert svd.n_components == 1
        assert features.shape == (3, 1)
        assert np.allclose(features, 0.0)


class TestCountClusterSizes:
    def test_counts_and_zero_fill(self, trainer):
        labels = np.array([0, 0, 2, 2, 2])
        sizes = trainer._count_cluster_sizes(labels, n_clusters=4)
        assert sizes == {0: 2, 1: 0, 2: 3, 3: 0}
        assert sum(sizes.values()) == len(labels)


class TestSaveArtifacts:
    def test_files_written(self, trainer):
        # 작은 실제 sklearn 객체로 아티팩트 구성
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        scaler = StandardScaler().fit(X)
        svd = TruncatedSVD(n_components=1, random_state=42).fit(X)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(X)
        art = ClusterModelArtifacts(scaler=scaler, svd=svd, kmeans=kmeans,
                                    category_vocab=["음악", "게임"])
        trainer._save_artifacts(art)

        d = trainer.artifacts_dir
        assert (d / "scaler.pkl").exists()
        assert (d / "svd.pkl").exists()
        assert (d / "kmeans.pkl").exists()
        assert (d / "category_vocab.json").exists()


# ----------------------------------------------------------------------
class TestRefreshClusters:
    def test_empty_users_raises(self, trainer):
        with pytest.raises(ValueError):
            trainer.refresh_clusters(ClusterRefreshRequest(users=[]))

    def test_full_train_summary_and_artifacts(self, trainer, sample_clustering_users):
        resp = trainer.refresh_clusters(ClusterRefreshRequest(users=sample_clustering_users))

        # 요약 응답 검증
        assert resp.n_users == len(sample_clustering_users)
        assert resp.n_clusters == settings.RECOMMENDER_N_CLUSTERS
        assert isinstance(resp.inertia, float)
        # cluster_sizes 합 == 전체 사용자 수
        assert sum(resp.cluster_sizes.values()) == len(sample_clustering_users)

        # 아티팩트 파일 저장 확인
        d = trainer.artifacts_dir
        for fname in ("scaler.pkl", "svd.pkl", "kmeans.pkl",
                      "category_vocab.json", "user_clusters.json"):
            assert (d / fname).exists(), f"{fname} 가 저장되지 않음"

    def test_user_clusters_json_keys_are_user_ids(self, trainer, sample_clustering_users):
        import json
        trainer.refresh_clusters(ClusterRefreshRequest(users=sample_clustering_users))
        data = json.loads((trainer.artifacts_dir / "user_clusters.json").read_text(encoding="utf-8"))
        # 모든 userId가 키로 들어가 있어야 함
        assert set(data.keys()) == {str(u.userId) for u in sample_clustering_users}
        # 값은 0..n_clusters-1 범위
        assert all(0 <= v < settings.RECOMMENDER_N_CLUSTERS for v in data.values())
