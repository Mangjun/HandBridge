from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from app.api.controllers.video import router as video_router
from app.api.controllers.emotion import router as emotion_router

# 업로드된 파일과 결과를 저장할 디렉토리 생성
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="HandBridge API",
    description="실시간 수어 번역을 위한 키포인트 추출 API"
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(video_router, prefix="/api/v1/video", tags=["video"])
app.include_router(emotion_router, prefix="/api/v1/emotion", tags=["emotion"])

@app.get("/")
def read_root():
    return {"message": "HandBridge API - 수어 번역을 위한 키포인트 추출 서비스"}