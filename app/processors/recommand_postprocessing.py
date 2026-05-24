import numpy as np

from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist


class KmeanEvaluation:
    def evaluate(self, embeddings, labels):
        silhouette_avg = silhouette_score(embeddings, labels)

        sse = self.calculate_sse(embeddings, labels)

        dunn_index = self.calculate_dunn_index(embeddings, labels)

        return silhouette_avg, sse, dunn_index

    def calculate_sse(self, embeddings, labels):
        sse = 0.0
        for label in np.unique(labels):
            cluster_points = embeddings[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            sse += np.sum((cluster_points - centroid) ** 2)
        return sse

    def calculate_dunn_index(self, embeddings, labels):
        unique_labels = np.unique(labels)
        inter_cluster_distances = []
        intra_cluster_distances = []

        # 클러스터 간 최소 거리 계산 (inter-cluster distance)
        for i, label_i in enumerate(unique_labels):
            cluster_i = embeddings[labels == label_i]
            for label_j in unique_labels[i + 1:]:
                cluster_j = embeddings[labels == label_j]
                distances = cdist(cluster_i, cluster_j)  # 클러스터 간 모든 포인트 쌍의 거리
                inter_cluster_distances.append(np.min(distances))  # 최소 거리 선택

        # 클러스터 내 최대 거리 계산 (intra-cluster distance)
        for label in unique_labels:
            cluster_points = embeddings[labels == label]
            if len(cluster_points) > 1:  # 클러스터 내 포인트가 2개 이상일 때만 거리 계산
                distances = cdist(cluster_points, cluster_points)
                intra_cluster_distances.append(np.max(distances))

        # Dunn Index = (최소 클러스터 간 거리) / (최대 클러스터 내 거리)
        if intra_cluster_distances:  # intra-cluster distance가 존재할 경우에만 계산
            dunn_index = np.min(inter_cluster_distances) / np.max(intra_cluster_distances)
            return dunn_index
        else:
            return 0  # 클러스터 내 거리가 없으면 0 반환
