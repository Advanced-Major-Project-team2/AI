# tests/test_config.py
"""app/core/config.py 의 Settings 검증."""

from pathlib import Path

from app.core.config import Settings, settings


class TestSettings:
    def test_singleton_instance_exists(self):
        assert isinstance(settings, Settings)

    def test_recommendation_numbers_are_int(self):
        # 기본값이 int 로 잡혀 있어야 KMeans(n_clusters=...) 등이 안전
        assert isinstance(settings.RECOMMANDS_LIMIT, int)
        assert isinstance(settings.RECOMMENDER_N_CLUSTERS, int)
        assert isinstance(settings.RECOMMENDER_SVD_DIM, int)

    def test_artifact_dirs_are_paths(self):
        assert isinstance(settings.CLUSTER_ARTIFACT_DIR, Path)
        assert isinstance(settings.POPULARITY_ARTIFACT_DIR, Path)

    def test_defaults_when_no_env(self):
        # .env 없이 직접 생성했을 때의 기본값(코드에 박힌 값) 확인
        s = Settings(_env_file=None)
        assert s.RECOMMANDS_LIMIT == 50
        assert s.RECOMMENDER_N_CLUSTERS == 5
        assert s.RECOMMENDER_SVD_DIM == 4
        assert s.APP_NAME == "gangku-ai-server"
        assert s.PORT == 8000

    def test_extra_env_ignored(self):
        # extra="ignore" 이므로 알 수 없는 키를 줘도 에러 없이 무시
        s = Settings(_env_file=None, SOME_UNKNOWN_KEY="x")
        assert not hasattr(s, "SOME_UNKNOWN_KEY")
