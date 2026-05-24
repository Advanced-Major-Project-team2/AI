# app/cluster/gatherings_popularity.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import json

from app.core.config import settings

from app.models.enums import (
    UserStatus
)

from app.models.schemas import (
    PopularityRefreshRequest,
    PopularityRefreshResponse,
)


class PopularityTrainer:
    """
    군집별 인기 방 테이블을 만드는 배치 서비스입니다.

    책임:
    1) user_action_log (PopularityRefreshRequest)를 받아서
    2) 유저 → 클러스터 매핑(user_cluster_map)을 이용해
        cluster_id → room_id 별 카운트를 집계하고
    3) 각 cluster_id 마다 상위 N개 room_id를 뽑아서
        cluster_popularity.json 으로 저장합니다.
    """

    def __init__(self, artifacts_dir: Optional[Path] = None) -> None:
        """
        artifacts_dir 기본값:
        app/cluster/artifacts/popularity
        """
        if artifacts_dir is None:
            artifacts_dir = Path(settings.POPULARITY_ARTIFACT_DIR)
        else:
            artifacts_dir = Path(artifacts_dir)

        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # 1) 군집별 인기방 JSON 경로
        self.popularity_path = self.artifacts_dir / "cluster_popularity.json"

        # 2) user_clustering 쪽 user_id -> cluster_id 매핑 JSON 경로
        cluster_dir = Path(settings.CLUSTER_ARTIFACT_DIR)
        user_clusters_path = cluster_dir / "user_clusters.json"

        if user_clusters_path.exists():
            # user_clusters_path가 먼저 선행되어야함.
            with user_clusters_path.open("r", encoding="utf-8") as f:
                # {"123": 0, "456": 2, ...}
                self.user_clusters: Dict[str, int] = json.load(f)
        else:
            # 매핑이 없으면 빈 딕셔너리 → cluster 못 찾는 유저는 스킵
            self.user_clusters = {}

        # 실제 JSON 파일 경로
        self.popularity_path = self.artifacts_dir / "cluster_popularity.json"

    # -----------------------------
    # public API
    # -----------------------------
    def refresh_popularity(self, request: PopularityRefreshRequest) -> PopularityRefreshResponse:
        """
        PopularityRefreshRequest = log_list: List[UserActionlog]
        UserActionlog =
        {
            user_id: Optional[int]
            room_id: int
            status: UserStatus
        }

        log_list를 받아서 cluster_id별 인기 room_id 랭킹을 계산하고
        cluster_popularity.json에 저장합니다.

        저장 형식 예:
        {
            "0": [101, 105, 110],
            "1": [203, 201],
            ...
        }
        """
        # 1) cluster_id별 → room_id → count 집계용 딕셔너리
        cluster_room_scores: Dict[int, Dict[int, float]] = {}

        for log in request.logList:
            if log.userId is None:
                # 익명 유저거나 cluster 정보 없는 경우는 스킵하거나 별도 처리
                continue
            # log.status로 참여/탈퇴 구분해서 참여 -> 1.0 증가, 클릭 -> 0.1 증가

            # user_id -> cluster_id 매핑 가져오기
            cluster_id = self.user_clusters.get(str(log.userId))
            if cluster_id is None:
                # 아직 군집이 할당되지 않은 유저면 스킵
                continue

            room_id = log.gatheringId

            # 이벤트 타입별 가중치
            if log.status == UserStatus.JOIN:      # 실제 Enum 이름에 맞게 수정
                weight = 1.0
            elif log.status == UserStatus.CLICK:   # 실제 Enum 이름에 맞게 수정
                weight = 0.1
            else:
                # 나가기/취소 등은 일단 점수에 반영하지 않음 (원하면 감점도 가능)
                continue

            cluster_room_scores.setdefault(cluster_id, {})
            cluster_room_scores[cluster_id].setdefault(room_id, 0.0)
            cluster_room_scores[cluster_id][room_id] += weight

        # cluster별로 room을 점수 기준으로 정렬 cluster_room_scores 이거랑 다른거임.
        cluster_popularity: Dict[str, list[int]] = {}
        for cid, room_scores in cluster_room_scores.items():
            sorted_rooms = sorted(
                room_scores.items(),
                key=lambda x: x[1],  # score 기준 내림차순
                reverse=True,
            )
            room_ids_sorted = [room_id for room_id, _ in sorted_rooms]
            cluster_popularity[str(cid)] = room_ids_sorted

        # JSON 저장
        with self.popularity_path.open("w", encoding="utf-8") as f:
            json.dump(cluster_popularity, f, ensure_ascii=False, indent=2)

            return PopularityRefreshResponse(
                total_logs=len(request.logList),
                n_clusters=len(cluster_popularity),
                top_n=max((len(v) for v in cluster_popularity.values()), default=0),
                cluster_popularity={int(k): v for k, v in cluster_popularity.items()}
            )
