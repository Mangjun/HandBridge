from pathlib import Path

# 프로젝트 설정
PROJECT_NAME = "HandBridge API"
API_V1_STR = "/api/v1"

# CORS 설정
CORS_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",  # React Native 개발 서버
    "*"  # 테스트를 위해 모든 origin 허용
]

# 파일 업로드 설정
ALLOWED_VIDEO_TYPES = (".mp4", ".avi", ".mov")
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB

# 디렉토리 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"

# 디렉토리 생성
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Mediapipe 설정
MEDIAPIPE_CONFIG = {
    "hands": {
        "static_image_mode": False,
        "max_num_hands": 2,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5
    }
} 