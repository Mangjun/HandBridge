import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
from fastapi import UploadFile, HTTPException
from ..core.config import ALLOWED_VIDEO_TYPES, MAX_VIDEO_SIZE, UPLOAD_DIR, RESULTS_DIR

def validate_video_file(file: UploadFile) -> None:
    """업로드된 비디오 파일의 유효성을 검사합니다."""
    if not file.filename.lower().endswith(ALLOWED_VIDEO_TYPES):
        raise HTTPException(
            status_code=400,
            detail=f"지원되지 않는 파일 형식입니다. 지원 형식: {', '.join(ALLOWED_VIDEO_TYPES)}"
        )
    
    try:
        content = file.file.read()
        file.file.seek(0) 
        if len(content) > MAX_VIDEO_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"파일 크기가 너무 큽니다. 최대 허용 크기: {MAX_VIDEO_SIZE/1024/1024}MB"
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail="파일 크기를 확인하는 중 오류가 발생했습니다.")

def generate_unique_filename(original_filename: str) -> Tuple[str, str]:
    """고유한 파일명을 생성합니다."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(original_filename).stem
    suffix = Path(original_filename).suffix
    unique_filename = f"{stem}_{timestamp}{suffix}"
    result_filename = f"{stem}_{timestamp}"
    return unique_filename, result_filename

async def save_upload_file(file: UploadFile, filename: str) -> Path:
    """업로드된 파일을 저장합니다."""
    file_path = UPLOAD_DIR / filename
    try:
        UPLOAD_DIR.mkdir(exist_ok=True) 
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        return file_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 저장 중 오류 발생: {str(e)}")

def save_results(result_filename: str, keypoints_data: np.ndarray, video_info: Dict[str, Any]) -> Tuple[Path, Path]:
    """처리 결과를 JSON과 NPZ 형식으로 저장합니다."""
    try:
        RESULTS_DIR.mkdir(exist_ok=True)  
        
        json_path = RESULTS_DIR / f"{result_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "video_info": video_info,
                "keypoints_shape": keypoints_data.shape,
                "total_frames": len(keypoints_data)
            }, f, ensure_ascii=False, indent=2, default=str)

        npz_path = RESULTS_DIR / f"{result_filename}.npz"
        np.savez_compressed(
            npz_path,
            keypoints=keypoints_data,
            video_info=np.array(list(video_info.items()), dtype=object)
        )

        return json_path, npz_path

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"결과 저장 중 오류 발생: {str(e)}")

def cleanup_old_files(directory: Path, max_age_days: int = 7) -> None:
    """오래된 파일들을 정리합니다."""
    try:
        current_time = datetime.now()
        for file_path in directory.glob("*"):
            if file_path.is_file():
                file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age.days > max_age_days:
                    file_path.unlink()
    except Exception as e:
        print(f"파일 정리 중 오류 발생: {str(e)}") 