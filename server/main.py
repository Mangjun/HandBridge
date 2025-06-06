import os
import json
import numpy as np
import cv2
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
from datetime import datetime
from pathlib import Path

app = FastAPI(title="HandBridge API", description="실시간 수어 번역을 위한 키포인트 추출 API")

# CORS 미들웨어 추가 (React Native에서 접근 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 업로드된 파일과 결과를 저장할 디렉토리 생성
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints_from_video(video_path: str):
    """
    비디오에서 손과 포즈 키포인트를 추출하는 함수
    """
    cap = cv2.VideoCapture(video_path)
    
    # MediaPipe 모델 초기화
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands, mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        frames_data = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR을 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 손 키포인트 추출
            hands_results = hands.process(rgb_frame)
            
            # 포즈 키포인트 추출
            pose_results = pose.process(rgb_frame)
            
            frame_data = {
                "frame": frame_count,
                "timestamp": frame_count / cap.get(cv2.CAP_PROP_FPS),
                "hands": [],
                "pose": []
            }
            
            # 손 랜드마크 추출
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    hand_keypoints = []
                    for landmark in hand_landmarks.landmark:
                        hand_keypoints.append({
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z
                        })
                    frame_data["hands"].append(hand_keypoints)
            
            # 포즈 랜드마크 추출 (상체 위주)
            if pose_results.pose_landmarks:
                pose_keypoints = []
                # 상체 주요 랜드마크만 추출 (0-10: 얼굴/어깨, 11-16: 팔)
                important_indices = list(range(11, 17)) + list(range(0, 11))
                for i in important_indices:
                    if i < len(pose_results.pose_landmarks.landmark):
                        landmark = pose_results.pose_landmarks.landmark[i]
                        pose_keypoints.append({
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z,
                            "visibility": landmark.visibility
                        })
                frame_data["pose"] = pose_keypoints
            
            frames_data.append(frame_data)
            frame_count += 1
        
        cap.release()
        return frames_data

@app.get("/")
def read_root():
    return {"message": "HandBridge API - 수어 번역을 위한 키포인트 추출 서비스"}

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    비디오 파일을 업로드하고 키포인트를 추출하는 엔드포인트
    """
    try:
        # 파일 확장자 검증
        if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            raise HTTPException(status_code=400, detail="지원되지 않는 비디오 형식입니다. (.mp4, .mov, .avi, .mkv만 지원)")
        
        # 타임스탬프를 포함한 고유 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = Path(file.filename).stem
        extension = Path(file.filename).suffix
        unique_filename = f"{original_name}_{timestamp}{extension}"
        
        file_path = UPLOAD_DIR / unique_filename
        
        # 파일 저장
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # 키포인트 추출
        keypoints_data = extract_keypoints_from_video(str(file_path))
        
        # 결과 저장
        result_filename = f"{original_name}_{timestamp}"
        
        # JSON으로 저장
        json_path = RESULTS_DIR / f"{result_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump({
                "video_info": {
                    "filename": unique_filename,
                    "original_name": file.filename,
                    "processed_at": datetime.now().isoformat(),
                    "total_frames": len(keypoints_data)
                },
                "keypoints": keypoints_data
            }, json_file, ensure_ascii=False, indent=2)
        
        # NPZ로도 저장 (NumPy 배열 형태)
        npz_path = RESULTS_DIR / f"{result_filename}.npz"
        
        # 손과 포즈 데이터를 NumPy 배열로 변환
        hands_array = []
        pose_array = []
        
        for frame in keypoints_data:
            frame_hands = []
            if frame["hands"]:
                for hand in frame["hands"]:
                    hand_coords = [[kp["x"], kp["y"], kp["z"]] for kp in hand]
                    frame_hands.append(hand_coords)
            hands_array.append(frame_hands)
            
            if frame["pose"]:
                pose_coords = [[kp["x"], kp["y"], kp["z"]] for kp in frame["pose"]]
                pose_array.append(pose_coords)
            else:
                pose_array.append([])
        
        np.savez_compressed(
            npz_path,
            hands=np.array(hands_array, dtype=object),
            pose=np.array(pose_array, dtype=object),
            video_info=np.array({
                "filename": unique_filename,
                "total_frames": len(keypoints_data)
            })
        )
        
        # 처리 완료 후 업로드된 비디오 파일 삭제 (선택사항)
        # os.remove(file_path)
        
        return JSONResponse(content={
            "success": True,
            "message": "키포인트 추출 완료",
            "data": {
                "total_frames": len(keypoints_data),
                "json_file": str(json_path),
                "npz_file": str(npz_path),
                "sample_frame": keypoints_data[0] if keypoints_data else None
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")

@app.get("/results/{filename}")
async def get_result(filename: str):
    """
    처리된 결과 파일을 조회하는 엔드포인트
    """
    json_path = RESULTS_DIR / f"{filename}.json"
    
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="결과 파일을 찾을 수 없습니다.")
    
    with open(json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    return JSONResponse(content=data)

@app.get("/health")
def health_check():
    """
    서버 상태 확인 엔드포인트
    """
    return {
        "status": "healthy",
        "mediapipe_version": mp.__version__,
        "opencv_version": cv2.__version__
    }