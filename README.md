# AI Server

모임(방) 추천을 담당하는 **FastAPI 기반 AI 서버**입니다. 사용자의 선호 카테고리·나이·학번·참여 횟수 같은 신호를 받아, 사용자에게 적합한 모임을 정렬해 반환합니다. 추천 엔진은 두 단계로 구성되어 있으며, 군집(clustering) 기반 추천(v2)을 우선 사용하고 문제가 있을 경우 카테고리 유사도 기반 추천(v1)으로 자동 폴백(fallback)합니다.

## 핵심 특징

- **2단계 추천 + 자동 폴백**: 군집 기반(v2) → 카테고리 유사도/콜드스타트(v1)
- **배치 학습 엔드포인트**: 하루 1회 등 주기적으로 사용자 군집과 군집별 인기 방 테이블을 재계산
- **계층 분리**: 외부 계약(`schemas`)과 내부 계산용 DTO(`domain`)를 분리해 검증 오버헤드를 최소화
- **중앙 집중식 예외 처리**: `ErrorCode` 기반의 일관된 JSON 에러 응답
- **JSON 구조화 로깅**: 요청/응답을 한 줄 JSON으로 기록하는 미들웨어

## 추천 로직

### 1. 실시간 추천 — `POST /api/ai/recommendations`

요청(`RecommendByClusteringModelRequest`)을 받아 다음 순서로 처리합니다.

1. **v2 (군집 기반)** 를 먼저 시도합니다.
   - 학습된 아티팩트가 없거나, 비로그인 사용자(`userId` 없음)이거나, 인기 방 테이블이 비어 있으면 `None`을 반환해 v1으로 넘깁니다.
   - 그 외에는 사용자 특징으로 군집을 예측하고, 가까운 군집부터 인기 방을 끌어와 중복 없이 정렬해 반환합니다.
2. v2가 `None`을 반환하거나 예외가 나면 **v1 (폴백)** 으로 처리합니다.
   - 비로그인 사용자 → **콜드스타트**: 인기도(0.6) + 최신도(0.4) 결합 점수로 정렬
   - 로그인 사용자 → **카테고리 유사도**: 선호 카테고리 일치 시 0.7점 + 호스트 나이 근접도(최대 0.3점)
3. 최종 결과를 `RecommendationResponse(gatheringsId=[...])` 형태로 응답합니다.

### 2. 사용자 군집 재학습 (배치) — `POST /api/ai/refresh/clustering`

전체 사용자 특징을 받아 군집 모델을 재학습하고 아티팩트를 저장합니다.

- 선호 카테고리 → multi-hot → **TruncatedSVD** 로 dense 벡터화
- 나이 / 학번 / 참여 횟수(수치형)와 결합 후 **StandardScaler** 로 스케일링
- **KMeans** 학습 → `scaler.pkl`, `svd.pkl`, `kmeans.pkl`, `category_vocab.json`, `user_clusters.json` 저장
- 응답: 사용자 수, 군집 수, inertia(SSE), 군집별 사용자 수 요약

### 3. 군집별 인기 방 재계산 (배치) — `POST /api/ai/refresh/popularity`

사용자 행동 로그를 받아 군집별 인기 방 순위를 계산합니다.

- `user_clusters.json` 으로 사용자 → 군집을 매핑 (매핑 없는 사용자/익명 사용자는 스킵)
- 행동 가중치: **JOIN = 1.0**, **CLICK = 0.1**
- 군집별로 방 점수를 합산·정렬해 `cluster_popularity.json` 저장
- 응답: 전체 로그 수, 군집 수, 군집별 상위 방 목록

> 참고: 추천 정확도를 위해 `clustering` 배치가 먼저 돌아 `user_clusters.json` 이 생성된 뒤 `popularity` 배치가 도는 것이 전제입니다.

## 프로젝트 구조

```
app/
├── main.py                 # FastAPI 엔트리포인트(create_app, startup 훅)
├── api/
│   ├── router.py           # /api/ai 하위 라우터 통합
│   ├── deps.py             # app.state 의 서비스 주입(의존성)
│   └── endpoints/
│       ├── health.py       # 헬스 체크
│       ├── recommendations.py  # 실시간 추천
│       └── clustering.py   # 군집/인기 방 재계산(배치)
├── services/
│   ├── v1/recommender.py   # 카테고리 유사도 + 콜드스타트
│   └── v2/recommender.py   # 군집 기반 추천(+ 아티팩트 로드)
├── cluster/
│   ├── user_clustering.py      # ClusteringTrainer(군집 학습/저장)
│   ├── gatherings_popularity.py # PopularityTrainer(인기 방 집계)
│   └── artifacts/          # 학습된 모델/매핑 산출물 저장 위치
├── processors/
│   ├── recommand_preprocessing.py  # 외부 DTO → 내부 DTO 변환
│   └── recommand_postprocessing.py # 군집 품질 지표(silhouette/SSE/Dunn)
├── models/
│   ├── enums.py            # Category, UserStatus
│   ├── schemas.py          # 외부 API 계약(Pydantic)
│   └── domain.py           # 내부 계산용 경량 DTO(dataclass)
└── core/
    ├── config.py           # 환경설정(pydantic-settings)
    ├── exception.py        # ErrorCode + 예외 핸들러
    └── logging.py          # JSON 로깅 + 요청/응답 미들웨어
tests/                      # 단위/통합 테스트 (pytest)
```

## API 요약

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| GET | `/health`, `/api/ai/health` | 헬스 체크 |
| POST | `/api/ai/recommendations` | 사용자 선호 기반 방 추천 |
| POST | `/api/ai/refresh/clustering` | 사용자 군집 재학습(배치) |
| POST | `/api/ai/refresh/popularity` | 군집별 인기 방 재계산(배치) |

에러는 모든 경우 다음 형식으로 통일됩니다.

```json
{ "error": { "code": "RECOMMENDATION_FAILED", "message": "추천 처리 중 오류가 발생했습니다." } }
```

### 지원 카테고리

스포츠, 친목, 독서, 여행, 음악, 스터디, 게임, 공연/축제, 봉사활동, 사진, 반려동물, 운동, 요리

## 설치 및 실행

Python 3.11 기준입니다.

```bash
# 1) 의존성 설치
pip install -r requirements.txt
# (개발 도구: black, flake8, pytest)
pip install -r requirements-dev.txt

# 2) 환경변수 파일 준비 (.env)
#    아래 "환경변수" 표 참고

# 3) 서버 실행
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

실행 후 Swagger 문서는 `http://127.0.0.1:8000/docs` 에서 확인할 수 있습니다.

## 환경변수 (`.env`)

| 키 | 예시 | 설명 |
| --- | --- | --- |
| `APP_NAME` | `gangku-ai-server` | 앱 이름 |
| `ENV` | `local` | 실행 환경 |
| `DEBUG` | `True` | 디버그 모드 |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | 바인딩 호스트/포트 |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | 허용 origin 목록(JSON 배열) |
| `LOG_LEVEL` | `INFO` | 로그 레벨 |
| `RECOMMANDS_LIMIT` | `50` | 추천 결과 최대 개수 |
| `RECOMMENDER_N_CLUSTERS` | `5` | KMeans 군집 수 |
| `RECOMMENDER_SVD_DIM` | `4` | TruncatedSVD 차원 수 |
| `CLUSTER_ARTIFACT_DIR` | `app/cluster/artifacts/user_clustering` | 군집 아티팩트 저장 경로 |
| `POPULARITY_ARTIFACT_DIR` | `app/cluster/artifacts/popularity` | 인기 방 아티팩트 저장 경로 |


## 테스트

```bash
# 전체 테스트
python -m pytest -q tests/

# 커버리지 포함
python -m pytest --cov=app --cov-report=term-missing tests/
```

`tests/` 는 모듈별로 단위 테스트가 분리되어 있으며, 엔드포인트는 의존성 오버라이드를 사용한 통합 테스트로 검증합니다. 공통 입력 생성 헬퍼는 `tests/factories.py`, 공유 픽스처와 경로/환경 설정은 `tests/conftest.py` 에 있습니다.

## 동작 전제 / 운영 메모

- v2 추천은 `clustering` → `popularity` 배치가 선행되어 아티팩트가 존재할 때만 활성화됩니다. 아티팩트가 없으면 자동으로 v1으로 폴백합니다.
- 군집 학습에는 사용자 수가 `RECOMMENDER_N_CLUSTERS` 이상이어야 합니다(KMeans 제약).
- 요청/응답 로깅 미들웨어는 민감정보 마스킹을 하지 않으므로, 운영 환경에서는 마스킹 정책 적용을 권장합니다.