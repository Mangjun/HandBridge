from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.controllers import video
from .core.config import PROJECT_NAME, API_V1_STR, CORS_ORIGINS

app = FastAPI(
    title=PROJECT_NAME,
    description="실시간 수어 번역을 위한 키포인트 추출 API",
    version="1.0.0"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(
    video.router,
    prefix=f"{API_V1_STR}/video",
    tags=["video"]
) 