from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Dict, Any
import json
import traceback
from pathlib import Path

from ...services.emotion_service import EmotionService
from ...core.config import UPLOAD_DIR, RESULTS_DIR

router = APIRouter(tags=["emotion"])
emotion_service = EmotionService()

@router.post("/analyze/{result_filename}")
async def analyze_emotion(result_filename: str) -> Dict[str, Any]:
    """
    저장된 비디오 파일에서 얼굴 감정을 분석합니다.
    """
    try:
        # 원본 비디오 파일 경로 찾기
        video_path = UPLOAD_DIR / f"{result_filename}.mp4"
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="비디오 파일을 찾을 수 없습니다.")
        
        # 감정 분석 수행
        frames_data, video_info = emotion_service.process_video(str(video_path))
        
        # 결과 저장
        result_path = RESULTS_DIR / f"{result_filename}_emotion.json"
        result = {
            "video_info": {
                **video_info,
                "filename": result_filename,
                "processed_at": datetime.now().isoformat()
            },
            "frames": frames_data
        }
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 프레임별 주요 감정 통계
        emotion_stats = {}
        total_faces = 0
        
        for frame in frames_data:
            for face in frame["emotions"]:
                emotion = face["dominant_emotion"]["emotion"]
                prob = face["dominant_emotion"]["probability"]
                if emotion not in emotion_stats:
                    emotion_stats[emotion] = {"count": 0, "total_prob": 0}
                emotion_stats[emotion]["count"] += 1
                emotion_stats[emotion]["total_prob"] += prob
                total_faces += 1
        
        # 평균 확률 계산
        if total_faces > 0:
            for emotion in emotion_stats:
                emotion_stats[emotion]["average_prob"] = emotion_stats[emotion]["total_prob"] / emotion_stats[emotion]["count"]
                emotion_stats[emotion]["percentage"] = (emotion_stats[emotion]["count"] / total_faces) * 100
        
        return {
            "success": True,
            "video_info": video_info,
            "emotion_statistics": emotion_stats,
            "total_frames_processed": len(frames_data),
            "total_faces_detected": total_faces,
            "result_file": str(result_path)
        }
        
    except Exception as e:
        error_msg = f"감정 분석 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_msg) 