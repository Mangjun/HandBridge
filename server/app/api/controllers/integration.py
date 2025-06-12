from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Dict, Any
import numpy as np
import cv2
import traceback
from pathlib import Path

from ...services.mediapipe_service import MediapipeService
from ...services.emotion_service import EmotionService
from ...services.sign_service import SignWordExtractor
from ...services.text_service import TextGenerator
from ...utils.file_utils import (
    validate_video_file,
    generate_unique_filename,
    save_upload_file,
    cleanup_old_files
)
from ...core.config import UPLOAD_DIR

router = APIRouter(tags=["integrated"])
mediapipe_service = MediapipeService()
emotion_service = EmotionService()
sign_word_extractor = SignWordExtractor(
    model_path="models/best_model.pth",
    label_map_path="models/label_map.json",
    window_size=30,
    stride=30,
    device="cpu"
)
text_service = TextGenerator()

@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    영상 파일을 받아 keypoints, 단어, 감정, 문장까지 한 번에 반환하는 통합 API
    """
    try:
        validate_video_file(file)
        unique_filename, _ = generate_unique_filename(file.filename)
        
        # 1. 파일 저장
        file_path = await save_upload_file(file, unique_filename)
        
        # 2. keypoints 추출
        keypoints = mediapipe_service.extract_keypoints_from_video(file_path)
        if len(keypoints) < 30:
            raise HTTPException(status_code=400, detail="프레임 수가 부족합니다. (최소 30프레임 필요)")
        
        # 3. 단어 리스트 추출 (window 단위)
        words = sign_word_extractor.predict_words(keypoints)
        
        # 4. 감정 분석 (예시: 마지막 프레임 기준)
        emotion_result, _ = emotion_service.process_video(str(file_path))
        emotion = emotion_result.get("emotion", "neutral")
        
        # 5. 문장 생성
        generated_sentence = text_service.generate_sentence(words, emotion)
        
        # 6. 백그라운드 파일 정리
        if background_tasks:
            background_tasks.add_task(cleanup_old_files, UPLOAD_DIR)
        
        # 7. 결과 반환
        return {
            "sign_words": words,
            "emotion": emotion,
            "generated_sentence": generated_sentence,
            "video_info": {
                "filename": unique_filename,
                "original_name": file.filename,
                "processed_at": datetime.now().isoformat(),
                "num_keypoint_frames": len(keypoints)
            }
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        error_msg = f"처리 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_msg)
