# app/core/exception.py
# 역할:
#   - 서비스 전반에서 사용할 에러 코드 Enum과 커스텀 예외 클래스를 정의합니다.
#   - 모든 예외를 이 파일에서 일괄 처리합니다.
#
# 응답 형태 (모든 에러 공통):
#   {
#     "error": {
#       "code": "RECOMMENDATION_FAILED",
#       "message": "추천 처리 중 오류가 발생했습니다."
#     }
#   }
#
# 사용처:
#   1. main.py → create_app()에서 register_exception_handlers(app) 호출
#   2. 엔드포인트/서비스 → raise AppException(ErrorCode.XXX)
#                          raise AppException(ErrorCode.XXX, "상세 메시지")
#   3. 필터 서비스      → raise FilterNotAllowedError(field="title")
#   4. 자동소개문 서비스 → raise AIGenerationError(cause=e)

from __future__ import annotations

from enum import Enum

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


# --------------------------------------------------
# 에러 코드 정의
# --------------------------------------------------
class ErrorCode(str, Enum):
    # 추천
    RECOMMENDATION_FAILED = "RECOMMENDATION_FAILED"     # 추천 처리 중 알 수 없는 오류
    ARTIFACT_NOT_LOADED = "ARTIFACT_NOT_LOADED"       # 아티팩트 미로드 (v2 fallback)
    NO_GATHERINGS = "NO_GATHERINGS"             # gatherings 없음 (coldstart 불가)

    # 클러스터 배치
    CLUSTER_REFRESH_FAILED = "CLUSTER_REFRESH_FAILED"    # 클러스터 재학습 실패
    POPULARITY_REFRESH_FAILED = "POPULARITY_REFRESH_FAILED"  # 인기방 재계산 실패
    EMPTY_USER_LIST = "EMPTY_USER_LIST"           # users 리스트 비어 있음
    INVALID_REQUEST = "FAIL_GET_LOGLIST"

    # 공통
    VALIDATION_ERROR = "VALIDATION_ERROR"          # 입력값 검증 실패 (422)
    INTERNAL_ERROR = "INTERNAL_ERROR"            # 그 외 서버 내부 오류


# --------------------------------------------------
# 기본 메시지 (message 생략 시 사용)
# --------------------------------------------------
_DEFAULT_MESSAGE: dict[ErrorCode, str] = {
    # 추천
    ErrorCode.RECOMMENDATION_FAILED: "추천 처리 중 오류가 발생했습니다.",
    ErrorCode.ARTIFACT_NOT_LOADED: "모델 아티팩트가 로드되지 않았습니다.",
    ErrorCode.NO_GATHERINGS: "gatherings 목록이 비어 있습니다.",

    # 클러스터 배치
    ErrorCode.CLUSTER_REFRESH_FAILED: "클러스터 재학습 중 오류가 발생했습니다.",
    ErrorCode.POPULARITY_REFRESH_FAILED: "인기방 재계산 중 오류가 발생했습니다.",
    ErrorCode.EMPTY_USER_LIST: "사용자 리스트가 비어 있습니다.",
    ErrorCode.INVALID_REQUEST: "사용자 액션 로그 리스트가 비어 있습니다.",

    # 공통
    ErrorCode.VALIDATION_ERROR: "입력값이 올바르지 않습니다.",
    ErrorCode.INTERNAL_ERROR: "서버 내부 오류가 발생했습니다.",
}

# --------------------------------------------------
# HTTP 상태코드 매핑
# --------------------------------------------------
_STATUS_MAP: dict[ErrorCode, int] = {
    # 추천
    ErrorCode.RECOMMENDATION_FAILED: 500,
    ErrorCode.ARTIFACT_NOT_LOADED: 503,
    ErrorCode.NO_GATHERINGS: 400,

    # 클러스터 배치
    ErrorCode.CLUSTER_REFRESH_FAILED: 500,
    ErrorCode.POPULARITY_REFRESH_FAILED: 500,
    ErrorCode.EMPTY_USER_LIST: 400,
    ErrorCode.INVALID_REQUEST: 400,

    # 공통
    ErrorCode.VALIDATION_ERROR: 422,
    ErrorCode.INTERNAL_ERROR: 500,
}


# --------------------------------------------------
# 커스텀 예외
# --------------------------------------------------
class AppException(Exception):
    """
    서비스 계층 → 엔드포인트로 전파할 커스텀 예외.

    사용 예:
        raise AppException(ErrorCode.ARTIFACT_NOT_LOADED)
        raise AppException(ErrorCode.CLUSTER_REFRESH_FAILED, f"KMeans 학습 실패: {e}")
    """
    def __init__(self, code: ErrorCode, message: str = "") -> None:
        self.code = code
        self.message = message or _DEFAULT_MESSAGE.get(code, code.value)
        self.status_code = _STATUS_MAP.get(code, 500)
        super().__init__(self.message)

    def to_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code,
            content={
                "error": {
                    "code": self.code.value,
                    "message": self.message,
                }
            },
        )


# --------------------------------------------------
# 공통 에러 응답 빌더
# --------------------------------------------------
def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
            }
        },
    )


# --------------------------------------------------
# FastAPI 핸들러 등록
# --------------------------------------------------
def register_exception_handlers(app: FastAPI) -> None:
    """
    main.py의 create_app()에서 호출.

    처리 범위:
    1. AppException           → ErrorCode 기반 응답 (FilterNotAllowedError, AIGenerationError 포함)
    2. RequestValidationError → VALIDATION_ERROR 422 (Pydantic 검증 실패)
    3. HTTPException          → FastAPI 기본 HTTPException을 같은 형태로 래핑
    4. Exception (catch-all)  → INTERNAL_ERROR 500
    """

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        return exc.to_response()

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(
            status_code=422,
            code=ErrorCode.VALIDATION_ERROR.value,
            message=str("pylance 검증 오류"),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return _error_response(
            status_code=exc.status_code,
            code=f"HTTP_{exc.status_code}",
            message=exc.detail if isinstance(exc.detail, str) else str(exc.detail),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return _error_response(
            status_code=500,
            code=ErrorCode.INTERNAL_ERROR.value,
            message=str(exc),
        )
