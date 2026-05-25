# tests/test_main.py
"""app/main.py 검증.

- create_app : FastAPI 앱 구성(라우트/타이틀)
- _init_logging / _init_cors / _init_routes : 미들웨어/라우트 등록
- _init_recommenders / _init_batch_services : app.state 에 서비스 적재
- startup_event / health_check : with TestClient 로 startup 훅 실행 후 동작
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import main as main_module
from app.core import config as config_module


@pytest.fixture(autouse=True)
def _isolated_artifact_dirs(tmp_path, monkeypatch):
    """서비스 초기화가 실제 레포 경로를 건드리지 않도록 임시 경로로 격리."""
    monkeypatch.setattr(config_module.settings, "CLUSTER_ARTIFACT_DIR", tmp_path / "clu")
    monkeypatch.setattr(config_module.settings, "POPULARITY_ARTIFACT_DIR", tmp_path / "pop")


class TestCreateApp:
    def test_returns_fastapi_with_title(self):
        app = main_module.create_app()
        assert isinstance(app, FastAPI)
        assert app.title == "AI Server"

    def test_routes_registered(self):
        app = main_module.create_app()
        paths = {r.path for r in app.routes}
        assert "/health" in paths
        assert "/api/ai/health" in paths
        assert "/api/ai/recommendations" in paths
        assert "/api/ai/refresh/clustering" in paths
        assert "/api/ai/refresh/popularity" in paths


class TestInitHelpers:
    def test_init_logging_adds_middleware(self):
        app = FastAPI()
        before = len(app.user_middleware)
        main_module._init_logging(app)
        assert len(app.user_middleware) > before

    def test_init_cors_adds_middleware(self):
        app = FastAPI()
        before = len(app.user_middleware)
        main_module._init_cors(app)
        assert len(app.user_middleware) > before

    def test_init_routes_includes_router(self):
        app = FastAPI()
        main_module._init_routes(app)
        paths = {r.path for r in app.routes}
        assert "/api/ai/recommendations" in paths

    def test_init_recommenders_sets_state(self):
        app = FastAPI()
        main_module._init_recommenders(app)
        # 정상 초기화되면 인스턴스, 실패해도 None 으로 세팅(속성 존재 보장)
        assert hasattr(app.state, "recommender_v1")
        assert hasattr(app.state, "recommender_v2")
        assert app.state.recommender_v1 is not None  # v1 은 항상 생성 가능

    def test_init_batch_services_sets_state(self):
        app = FastAPI()
        main_module._init_batch_services(app)
        assert hasattr(app.state, "clustering_service")
        assert hasattr(app.state, "popularity_service")
        assert app.state.clustering_service is not None
        assert app.state.popularity_service is not None


class TestStartupHook:
    def test_startup_populates_state_and_health(self):
        app = main_module.create_app()
        # with 컨텍스트 진입 시 startup_event 가 실행됨
        with TestClient(app) as client:
            assert app.state.recommender_v1 is not None
            assert app.state.clustering_service is not None
            r = client.get("/health")
            assert r.status_code == 200
            assert r.json() == {"status": "ok"}
