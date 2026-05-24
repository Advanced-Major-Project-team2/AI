# app/cluster/user_clustering.py

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict

import json
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
from app.models.enums import Category
from app.core.config import settings
from app.models.schemas import (
    ClusteringUserData,
    ClusterRefreshRequest,
    ClusterRefreshResponse,
)


class ClusterModelArtifacts:
    """
    학습된 클러스터링 관련 아티팩트를 한 번에 들고 있는 단순 컨테이너입니다.
    - 실제 서빙(추천) 시에는 이 객체를 메모리에 올려두고 사용하게 됩니다.
    """

    def __init__(
        self,
        scaler: StandardScaler,
        svd: TruncatedSVD,
        kmeans: KMeans,
        category_vocab: List[str],
    ) -> None:
        self.scaler = scaler
        self.svd = svd
        self.kmeans = kmeans
        self.category_vocab = category_vocab


class ClusteringTrainer:
    """
    배치 클러스터링(하루 1회 등)을 담당하는 서비스 클래스입니다.
    - 책임:
        1) 배치 요청(모든 사용자 feature)을 받아서
        2) feature 전처리 (multi-hot → SVD, 스케일링)
        3) K-means 학습
        4) 결과 아티팩트를 디스크에 저장
        5) 간단한 요약 정보를 반환
    """
    def __init__(self, artifacts_dir: Path | None = None) -> None:
        # 아티팩트를 저장할 디렉토리 경로를 설정합니다.
        # 아티팩트 경로 : app/cluster/artifacts/user_clustering
        if artifacts_dir is None:
            artifacts_dir = Path(settings.CLUSTER_ARTIFACT_DIR)
        else:
            artifacts_dir = Path(artifacts_dir)

        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # public API
    # -------------------------

    def refresh_clusters(
        self,
        request: ClusterRefreshRequest,
    ) -> ClusterRefreshResponse:
        """
        배치 엔드포인트에서 호출할 메인 메서드.
        1) 요청으로 받은 사용자 리스트를 numpy 행렬로 변환하고
        2) SVD(카테고리) + StandardScaler(사용자 학번, 나이, 참가횟수) + KMeans를 학습
        3) 결과 디스크에 저장
        4) 요약 정보를 ClusterRefreshResult로 반환합니다.
        """

        users = request.users
        if not users:
            # 일말의 예외처리
            raise ValueError("사용자 리스트가 비어 있습니다. 최소 1명 이상 필요합니다.")

        # 1. 카테고리 vocab 생성
        category_vocab = self._build_category_vocab(users)

        # 2. multi-hot 행렬 생성 (n_users x n_categories)
        multi_hot = self._build_multi_hot_matrix(users, category_vocab)

        # 3. multi-hot → SVD dense vector로 압축
        svd, svd_features = self._fit_svd(multi_hot, n_components=settings.RECOMMENDER_SVD_DIM)

        # 4. 나이/학년/참가횟수 수치 feature 행렬 생성
        numeric_features = self._build_numeric_matrix(users)

        # ---------------- clustering 시작 ----------------
        # 최종 feature 행렬 구성: [numeric | svd_features]
        X = np.hstack([numeric_features, svd_features])

        # StandardScaler로 전체 feature 스케일링
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-means 학습
        kmeans = KMeans(
            n_clusters=settings.RECOMMENDER_N_CLUSTERS,
            random_state=42,
            n_init="auto",
        )
        kmeans.fit(X_scaled)
        # ---------------- clustering 끝 ----------------

        # 5. cluster 크기 집계
        labels = kmeans.labels_
        cluster_sizes = self._count_cluster_sizes(labels, settings.RECOMMENDER_N_CLUSTERS)

        # user_clustering안에 어떤 clustering안에 어떤 user가 있는지 저장 -> gatherings_popularity에서 용이하게 쓸 수 있음
        user_clusters: dict[str, int] = {}
        for u, cid in zip(users, labels):
            if u.userId is not None:
                user_clusters[str(u.userId)] = int(cid)

        user_clusters_path = self.artifacts_dir / "user_clusters.json"
        with user_clusters_path.open("w", encoding="utf-8") as f:
            json.dump(user_clusters, f, ensure_ascii=False, indent=2)
        # 아티팩트로 저장 (user_clustering 디렉토리 안에)

        # 6. 디스크에 아티팩트 저장
        artifacts = ClusterModelArtifacts(
            scaler=scaler,
            svd=svd,
            # multi-hot vec -> dense vec 변환위함
            kmeans=kmeans,
            # clustering 하기 위함.
            category_vocab=category_vocab,
        )
        self._save_artifacts(artifacts)

        # 7. 요약 결과 반환
        result = ClusterRefreshResponse(
            n_users=len(users),
            n_clusters=settings.RECOMMENDER_N_CLUSTERS,
            inertia=float(kmeans.inertia_),
            cluster_sizes=cluster_sizes,
        )
        return result

    def _build_category_vocab(self, users: List[ClusteringUserData]) -> List[str]:
        """
        1.
        모든 사용자에서 선호 카테고리를 모아서 '유니크 카테고리 리스트'를 만듭니다.
        - 이 순서가 multi-hot 벡터의 열 순서가 됩니다.
        - 나중에 서빙 시에도 같은 순서를 사용해야 하므로,
        vocab 자체를 파일로 저장해야 합니다.
        """
        vocab_set = set()
        for u in users:
            for cat in u.preferredCategories:
                if cat is not None:
                    vocab_set.add(cat)
        # 정렬해서 deterministic한 순서를 보장합니다.
        vocab = sorted(vocab_set)
        return vocab

    def _build_multi_hot_matrix(
        self,
        users: List[ClusteringUserData],
        vocab: List[str],
    ) -> np.ndarray:
        """
        2.
        사용자별 선호 카테고리를 multi-hot 벡터로 변환합니다.
        - 행: 사용자
        - 열: vocab에 정의된 카테고리
        - 값: 해당 사용자가 그 카테고리를 선호하면 1, 아니면 0
        """
        n_users = len(users)
        n_cats = len(vocab)
        cat_index: dict[str, int] = {cat: idx for idx, cat in enumerate(vocab)}

        multi_hot = np.zeros((n_users, n_cats), dtype=np.float32)

        for i, u in enumerate(users):
            for cat in u.preferredCategories:
                # Enum → 문자열로 정규화
                name = Category._cat_name(cat)  # "음악", "게임", "스터디" 같은 값
                idx = cat_index.get(name)
                if idx is None:
                    # vocab에 없는 카테고리는 조용히 무시
                    continue
                multi_hot[i, idx] = 1.0

        return multi_hot

    def _fit_svd(
        self,
        multi_hot: np.ndarray,
        n_components: int,
    ) -> Tuple[TruncatedSVD, np.ndarray]:
        """
        multi-hot 카테고리 행렬에 TruncatedSVD를 학습하고,
        저차원 dense 벡터로 변환합니다.
        - 카테고리가 거의 없거나(열 개수 <= n_components) multi-hot이 전부 0인 경우,
        굳이 차원을 줄이지 않고 그대로 쓰는 단순 처리도 가능합니다.
        """
        n_users, n_cats = multi_hot.shape

        if n_cats == 0:
            # 선호 카테고리가 전혀 없는 경우: 0 벡터를 반환합니다.
            svd = TruncatedSVD(n_components=1, random_state=42)
            # fit을 형식상 한 번은 호출해줘야 하므로,
            # 최소한의 더미 데이터를 사용합니다.
            dummy = np.zeros((2, 2), dtype=np.float32)
            svd.fit(dummy)
            features = np.zeros((n_users, 1), dtype=np.float32)
            return svd, features

        # 실제 사용할 차원 수는 카테고리 개수와 요청값 중 더 작은 쪽으로 선택합니다.
        n_components_eff = min(n_components, n_cats, n_users)
        if n_components_eff < 1:
            n_components_eff = 1

        svd = TruncatedSVD(n_components=n_components_eff, random_state=42)
        features = svd.fit_transform(multi_hot)

        return svd, features

    def _build_numeric_matrix(self, users: List[ClusteringUserData]) -> np.ndarray:
        """
            나이, 학번, 참가횟수 같은 수치형 feature를 하나의 행렬로 만듭니다.
            - None인 값은 0 또는 전체 평균/중앙값으로 대체할 수 있습니다.
            여기서는 단순하게 0으로 채워 넣고,
            이후 StandardScaler로 스케일링하여 분포를 맞춥니다.
        """
        n_users = len(users)
        # 열 순서: age, student_year, total_join_count
        mat = np.zeros((n_users, 3), dtype=np.float32)

        for i, u in enumerate(users):
            age = u.age if u.age is not None else 0
            enroll = u.enrollNumber if u.enrollNumber is not None else 0
            join_cnt = u.userJoinCount if u.userJoinCount is not None else 0

            mat[i, 0] = age
            mat[i, 1] = enroll
            mat[i, 2] = join_cnt

        return mat

    def _count_cluster_sizes(self, labels: np.ndarray, n_clusters: int) -> Dict[int, int]:
        """
        K-means가 할당한 cluster 라벨을 기반으로
        각 클러스터에 몇 명의 사용자가 속해 있는지 집계합니다.
        """
        sizes: Dict[int, int] = {k: 0 for k in range(n_clusters)}
        for label in labels:
            sizes[int(label)] += 1
        return sizes

    def _save_artifacts(self, artifacts: ClusterModelArtifacts) -> None:
        """
        학습된 아티팩트(scaler, svd, kmeans, category_vocab)를 디스크에 저장합니다.
        - 나중에 실시간 추천 엔드포인트에서 이 파일들을 로드해 사용합니다.
        - 간단히 joblib + json을 사용합니다.
        """
        scaler_path = self.artifacts_dir / "scaler.pkl"
        svd_path = self.artifacts_dir / "svd.pkl"
        kmeans_path = self.artifacts_dir / "kmeans.pkl"
        vocab_path = self.artifacts_dir / "category_vocab.json"

        # 각 객체를 joblib 또는 json으로 저장합니다.
        joblib.dump(artifacts.scaler, scaler_path)
        joblib.dump(artifacts.svd, svd_path)
        joblib.dump(artifacts.kmeans, kmeans_path)

        with vocab_path.open("w", encoding="utf-8") as f:
            json.dump(artifacts.category_vocab, f, ensure_ascii=False, indent=2)
