from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Dict, Any
import json
import cv2
import mediapipe as mp
import traceback
from pathlib import Path

from ...services.mediapipe_service import MediapipeService
from ...services.emotion_service import EmotionService
from ...utils.file_utils import (
    validate_video_file,
    generate_unique_filename,
    save_upload_file,
    save_results,
    cleanup_old_files
)
from ...core.config import UPLOAD_DIR, RESULTS_DIR
from ...models.schemas import ProcessingResult, ErrorResponse

router = APIRouter(tags=["video"])
mediapipe_service = MediapipeService()
emotion_service = EmotionService()

@router.post("/upload", 
            response_model=ProcessingResult,
            responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def upload_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    비디오 파일을 업로드하고 키포인트를 추출합니다.
    """
    try:
        validate_video_file(file)
        unique_filename, result_filename = generate_unique_filename(file.filename)
        
        # 파일 저장
        file_path = await save_upload_file(file, unique_filename)
        
        # 비디오 처리 - 키포인트 추출
        frames_data, video_info = mediapipe_service.process_video(file_path)
        
        # 감정 분석 - 마지막 프레임
        emotion_result, _ = emotion_service.process_video(str(file_path))
        
        # 비디오 정보 업데이트
        video_info.update({
            "filename": unique_filename,
            "original_name": file.filename,
            "processed_at": datetime.now().isoformat()
        })
        
        # 결과 저장
        json_path, npz_path = save_results(result_filename, frames_data, video_info)
        
        # 백그라운드에서 임시 파일 정리
        if background_tasks:
            background_tasks.add_task(cleanup_old_files, UPLOAD_DIR)
            background_tasks.add_task(cleanup_old_files, RESULTS_DIR)
        
        response_data = {
            "video_info": video_info,
            "keypoints": frames_data.tolist(),
            "emotion_analysis": emotion_result,
            "files": {
                "json": str(json_path),
                "npz": str(npz_path)
            }
        }
        
        return response_data
        
    except HTTPException as e:
        raise e
    except Exception as e:
        error_msg = f"처리 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/results/{filename}")
async def get_results(filename: str) -> Dict[str, Any]:
    """
    처리된 결과 파일을 조회합니다.
    """
    try:
        json_path = RESULTS_DIR / f"{filename}.json"
        if not json_path.exists():
            raise HTTPException(status_code=404, detail="결과 파일을 찾을 수 없습니다.")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
        
    except HTTPException as e:
        raise e
    except Exception as e:
        error_msg = f"결과 조회 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    서비스 상태를 확인합니다.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "mediapipe_version": mp.__version__,
        "opencv_version": cv2.__version__
    } 