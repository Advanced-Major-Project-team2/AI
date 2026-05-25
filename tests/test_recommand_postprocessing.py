# tests/test_recommand_postprocessing.py
"""app/processors/recommand_postprocessing.py 의 KmeanEvaluation 검증."""

import numpy as np
import pytest

from app.processors.recommand_postprocessing import KmeanEvaluation


@pytest.fixture
def evaluator():
    return KmeanEvaluation()


# 두 개의 잘 분리된 군집 (각 3점)
@pytest.fixture
def two_clusters():
    embeddings = np.array(
        [
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],     # cluster 0
            [10.0, 10.0], [10.1, 10.0], [10.0, 10.1],  # cluster 1
        ]
    )
    labels = np.array([0, 0, 0, 1, 1, 1])
    return embeddings, labels


class TestCalculateSse:
    def test_sse_non_negative(self, evaluator, two_clusters):
        emb, labels = two_clusters
        sse = evaluator.calculate_sse(emb, labels)
        assert sse >= 0.0

    def test_sse_zero_for_identical_points(self, evaluator):
        # 각 군집의 모든 점이 동일 → centroid와 거리 0 → SSE 0
        emb = np.array([[1.0, 1.0], [1.0, 1.0], [5.0, 5.0], [5.0, 5.0]])
        labels = np.array([0, 0, 1, 1])
        assert evaluator.calculate_sse(emb, labels) == pytest.approx(0.0)

    def test_sse_known_value(self, evaluator):
        # 한 군집: (0,0),(2,0) → centroid (1,0) → 각 거리^2 =1 → 합 2
        emb = np.array([[0.0, 0.0], [2.0, 0.0]])
        labels = np.array([0, 0])
        assert evaluator.calculate_sse(emb, labels) == pytest.approx(2.0)


class TestCalculateDunnIndex:
    def test_positive_for_separated_clusters(self, evaluator, two_clusters):
        emb, labels = two_clusters
        dunn = evaluator.calculate_dunn_index(emb, labels)
        assert dunn > 0.0

    def test_returns_zero_when_no_intra_distance(self, evaluator):
        # 모든 군집이 단일 포인트 → intra-cluster 거리 없음 → 0 반환
        emb = np.array([[0.0, 0.0], [5.0, 5.0]])
        labels = np.array([0, 1])
        assert evaluator.calculate_dunn_index(emb, labels) == 0

    def test_known_geometry(self, evaluator):
        # cluster0: (0,0),(1,0) → intra max = 1
        # cluster1: (10,0),(11,0) → intra max = 1
        # inter min 거리 = 9 (1,0)-(10,0)
        # dunn = 9 / 1 = 9
        emb = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
        labels = np.array([0, 0, 1, 1])
        assert evaluator.calculate_dunn_index(emb, labels) == pytest.approx(9.0)


class TestEvaluate:
    def test_returns_three_metrics(self, evaluator, two_clusters):
        emb, labels = two_clusters
        silhouette, sse, dunn = evaluator.evaluate(emb, labels)
        # silhouette: 잘 분리된 군집이므로 1에 가깝고 양수
        assert -1.0 <= silhouette <= 1.0
        assert silhouette > 0.0
        assert sse >= 0.0
        assert dunn > 0.0
