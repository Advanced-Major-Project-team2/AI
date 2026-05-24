from fastapi import APIRouter

router = APIRouter()


@router.get("/health", summary="헬스 체크")
def ping():
    return {"status": "ok"}
