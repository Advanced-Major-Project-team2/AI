# tests/test_exception.py
"""app/core/exception.py 검증.

- AppException.__init__ : 기본 메시지/커스텀 메시지/상태코드 매핑
- AppException.to_response : JSONResponse 본문 형식
- _error_response : 공통 빌더
- register_exception_handlers : FastAPI 통합(핸들러가 올바른 형식/상태코드로 응답)
"""

import json

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.core.exception import (
    AppException,
    ErrorCode,
    _error_response,
    register_exception_handlers,
)


def _body(resp: JSONResponse) -> dict:
    return json.loads(bytes(resp.body))


class TestAppExceptionInit:
    def test_default_message_used_when_omitted(self):
        exc = AppException(ErrorCode.NO_GATHERINGS)
        assert exc.message == "gatherings 목록이 비어 있습니다."
        assert exc.code == ErrorCode.NO_GATHERINGS

    def test_custom_message_overrides_default(self):
        exc = AppException(ErrorCode.NO_GATHERINGS, "직접 지정한 메시지")
        assert exc.message == "직접 지정한 메시지"

    @pytest.mark.parametrize(
        "code,expected_status",
        [
            (ErrorCode.RECOMMENDATION_FAILED, 500),
            (ErrorCode.ARTIFACT_NOT_LOADED, 503),
            (ErrorCode.NO_GATHERINGS, 400),
            (ErrorCode.EMPTY_USER_LIST, 400),
            (ErrorCode.INVALID_REQUEST, 400),
            (ErrorCode.VALIDATION_ERROR, 422),
            (ErrorCode.INTERNAL_ERROR, 500),
            (ErrorCode.CLUSTER_REFRESH_FAILED, 500),
            (ErrorCode.POPULARITY_REFRESH_FAILED, 500),
        ],
    )
    def test_status_code_mapping(self, code, expected_status):
        assert AppException(code).status_code == expected_status

    def test_is_exception_subclass(self):
        assert isinstance(AppException(ErrorCode.INTERNAL_ERROR), Exception)


class TestToResponse:
    def test_response_shape_and_status(self):
        exc = AppException(ErrorCode.ARTIFACT_NOT_LOADED, "로드 안됨")
        resp = exc.to_response()
        assert resp.status_code == 503
        assert _body(resp) == {
            "error": {"code": "ARTIFACT_NOT_LOADED", "message": "로드 안됨"}
        }


class TestErrorResponseBuilder:
    def test_builder(self):
        resp = _error_response(418, "TEAPOT", "i am a teapot")
        assert resp.status_code == 418
        assert _body(resp) == {"error": {"code": "TEAPOT", "message": "i am a teapot"}}


@pytest.fixture
def client_with_handlers():
    """예외 핸들러가 등록된 임시 앱 + 테스트 라우트."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise-app")
    def _raise_app():
        raise AppException(ErrorCode.NO_GATHERINGS)

    @app.get("/raise-http")
    def _raise_http():
        raise HTTPException(status_code=404, detail="없음")

    @app.get("/raise-unhandled")
    def _raise_unhandled():
        raise RuntimeError("터짐")

    return TestClient(app, raise_server_exceptions=False)


class TestRegisterExceptionHandlers:
    def test_app_exception_handler(self, client_with_handlers):
        r = client_with_handlers.get("/raise-app")
        assert r.status_code == 400
        assert r.json() == {
            "error": {"code": "NO_GATHERINGS", "message": "gatherings 목록이 비어 있습니다."}
        }

    def test_http_exception_handler(self, client_with_handlers):
        r = client_with_handlers.get("/raise-http")
        assert r.status_code == 404
        body = r.json()
        assert body["error"]["code"] == "HTTP_404"
        assert body["error"]["message"] == "없음"

    def test_unhandled_exception_handler(self, client_with_handlers):
        r = client_with_handlers.get("/raise-unhandled")
        assert r.status_code == 500
        body = r.json()
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert "터짐" in body["error"]["message"]
