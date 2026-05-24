# app/core/logging.py
# 역할:
#   - 콘솔로 JSON 한 줄 로그를 출력합니다.
#   - 모든 요청/응답을 미들웨어에서 기록합니다(민감정보 마스킹 없음).
# 사용법:
#   - main.py에서 setup_logging() 호출 후, RequestResponseLoggerMiddleware를 add_middleware로 장착.

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message


class JsonFormatter(logging.Formatter):
    """로그 레코드를 JSON 문자열로 직렬화."""
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, object] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class RequestResponseLoggerMiddleware(BaseHTTPMiddleware):
    """
    요청/응답 로깅 미들웨어.
    - 요청: 메서드, 경로, 쿼리, 헤더, 바디(최대 4096바이트) 로깅
    - 응답: 상태코드, 소요시간(ms), 응답 바디(가능할 때만, 최대 4096바이트) 로깅
    - 민감정보 마스킹 없음(요청에 따라 간단하게 구성)
    """

    def __init__(self, app, *, sample_limit: int = 4096) -> None:
        super().__init__(app)
        self.logger = logging.getLogger("app.request")
        self.sample_limit = sample_limit

    async def dispatch(self, request: Request, call_next):
        if request.url.path.endswith("/stream"):
            return await call_next(request)

        started = time.perf_counter()
        req_id = request.headers.get("x-request-id") or str(uuid.uuid4())

        # 원본 바디 확보 및 재주입
        raw_body = await request.body()

        async def receive() -> Message:
            return {"type": "http.request", "body": raw_body, "more_body": False}

        request._receive = receive  # Starlette 관례적으로 허용되는 패턴

        # 요청 로그
        try:
            headers_logged = {k: v for k, v in request.headers.items()}
        except Exception:
            headers_logged = {}

        try:
            body_text = raw_body[: self.sample_limit].decode("utf-8", errors="replace")
        except Exception:
            body_text = ""

        self.logger.info(
            "request",
            extra={
                "extra": {
                    "event": "request",
                    "request_id": req_id,
                    "method": request.method,
                    "path": request.url.path,
                    "headers": headers_logged,
                    "body": body_text,
                }
            },
        )

        # 응답 생성
        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        # 응답 바디 캡처 (StreamingResponse면 건너뜀)
        resp_body_preview = None
        resp_len = None
        is_streaming = getattr(response, "body_iterator", None) is not None

        if not is_streaming:
            try:
                body_bytes = getattr(response, "body", b"")
                if isinstance(body_bytes, (bytes, bytearray)):
                    resp_len = len(body_bytes)
                    resp_body_preview = body_bytes[: self.sample_limit].decode("utf-8", errors="replace")
            except Exception:
                pass

        self.logger.info(
            "response",
            extra={
                "extra": {
                    "event": "response",
                    "request_id": req_id,
                    "path": request.url.path,
                    "status": response.status_code,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "response_size": resp_len,
                    "body": resp_body_preview,
                }
            },
        )

        return response


def setup_logging(level: str | int = "INFO") -> None:
    """
    루트/uvicorn 로거를 JSON 포맷으로 설정.
    """
    formatter = JsonFormatter()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level if isinstance(level, int) else level.upper())

    for name in ("app", "app.request", "uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.addHandler(handler)
        lg.setLevel(level if isinstance(level, int) else level.upper())
        lg.propagate = False  # 중복 출력 방지


def get_logger(name: str = "app") -> logging.Logger:
    """모듈에서 사용할 전용 로거를 얻습니다."""
    return logging.getLogger(name)
