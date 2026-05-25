# tests/test_logging.py
"""app/core/logging.py 검증.

- JsonFormatter.format : 기본/extra dict/exc_info 직렬화
- setup_logging : 핸들러/레벨 구성
- get_logger : 로거 반환
- RequestResponseLoggerMiddleware.dispatch : 일반 요청 통과 + /stream 우회
"""

import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.logging import (
    JsonFormatter,
    RequestResponseLoggerMiddleware,
    setup_logging,
    get_logger,
)


class TestJsonFormatter:
    def _record(self, **kwargs):
        defaults = dict(
            name="test", level=logging.INFO, pathname=__file__, lineno=1,
            msg="hello", args=(), exc_info=None,
        )
        defaults.update(kwargs)
        return logging.LogRecord(**defaults)

    def test_basic_fields(self):
        out = json.loads(JsonFormatter().format(self._record()))
        assert out["level"] == "INFO"
        assert out["logger"] == "test"
        assert out["message"] == "hello"
        assert "ts" in out

    def test_extra_dict_merged(self):
        rec = self._record()
        rec.extra = {"event": "request", "request_id": "abc"}
        out = json.loads(JsonFormatter().format(rec))
        assert out["event"] == "request"
        assert out["request_id"] == "abc"

    def test_extra_non_dict_ignored(self):
        rec = self._record()
        rec.extra = "not-a-dict"
        out = json.loads(JsonFormatter().format(rec))
        assert "not-a-dict" not in out.values()

    def test_exc_info_included(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            rec = self._record(exc_info=sys.exc_info())
        out = json.loads(JsonFormatter().format(rec))
        assert "exc_info" in out
        assert "ValueError" in out["exc_info"]

    def test_message_with_args(self):
        rec = self._record(msg="value=%s", args=(42,))
        out = json.loads(JsonFormatter().format(rec))
        assert out["message"] == "value=42"


class TestSetupAndGetLogger:
    def test_get_logger_returns_logger(self):
        lg = get_logger("app.custom")
        assert isinstance(lg, logging.Logger)
        assert lg.name == "app.custom"

    def test_get_logger_default_name(self):
        assert get_logger().name == "app"

    def test_setup_logging_with_string_level(self):
        setup_logging("WARNING")
        assert logging.getLogger().level == logging.WARNING
        # 구성 대상 로거는 propagate=False
        assert logging.getLogger("app.request").propagate is False
        # 원복
        setup_logging("INFO")

    def test_setup_logging_with_int_level(self):
        setup_logging(logging.DEBUG)
        assert logging.getLogger().level == logging.DEBUG
        setup_logging("INFO")


@pytest.fixture
def middleware_client():
    app = FastAPI()
    app.add_middleware(RequestResponseLoggerMiddleware)

    @app.post("/echo")
    def echo(payload: dict):
        return {"received": payload}

    @app.get("/something/stream")
    def stream():
        return {"streamed": True}

    return TestClient(app)


class TestMiddleware:
    def test_post_body_reinjected_and_passes(self, middleware_client):
        # 미들웨어가 body를 읽고 재주입하므로, 엔드포인트가 정상적으로 body를 받아야 함
        r = middleware_client.post("/echo", json={"a": 1})
        assert r.status_code == 200
        assert r.json() == {"received": {"a": 1}}

    def test_stream_path_bypasses_logging(self, middleware_client):
        # /stream 으로 끝나는 경로는 미들웨어가 곧바로 통과시킴
        r = middleware_client.get("/something/stream")
        assert r.status_code == 200
        assert r.json() == {"streamed": True}

    def test_custom_sample_limit_stored(self):
        mw = RequestResponseLoggerMiddleware(app=FastAPI(), sample_limit=128)
        assert mw.sample_limit == 128
        assert mw.logger.name == "app.request"
