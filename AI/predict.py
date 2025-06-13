import os
import glob
import json
import numpy as np

from mediapipe_service import MediapipeService
from sign_service import SignWordExtractor

# 경로 설정
VIDEO_DIR = './video_path/sign_data'
LABEL_DIR = './video_path/labels'
MODEL_PATH = './checkpoints/fine_tuned.pth'
LABEL_MAP_PATH = './numpy_train_data/label_map.json'

# 서비스 초기화
mp_service = MediapipeService(target_fps=30, window_size=30)
extractor = SignWordExtractor(
    model_path=MODEL_PATH,
    label_map_path=LABEL_MAP_PATH,
    device='cpu',
    window_size=30,
    stride=15  # 겹치는 윈도우 테스트 위해 stride 절반으로 설정
)

# 테스트 스크립트: 각 세그먼트별 예측 시 슬라이딩 윈도우 평균 확률 사용
for video_path in glob.glob(os.path.join(VIDEO_DIR, '*.mp4')):
    video_name = os.path.basename(video_path)
    prefix = os.path.splitext(video_name)[0]
    label_file = os.path.join(LABEL_DIR, f"{prefix}_morpheme.json")
    if not os.path.exists(label_file):
        print(f"[WARN] Label file not found for {video_name}")
        continue

    # Ground Truth 로드
    with open(label_file, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    segments = label_data.get('data', [])
    gt_words = [seg['attributes'][0]['name'] for seg in segments]

    # 전체 영상 키포인트 추출
    arr_full = mp_service.extract_keypoints_from_video(video_path)

    pred_words = []
    # 세그먼트별 예측
    for seg in segments:
        start_idx = int(seg['start'] * mp_service.target_fps)
        end_idx   = int(seg['end'] * mp_service.target_fps)
        seg_arr = arr_full[start_idx:end_idx]

        # 예측: SignWordExtractor 내부에서 슬라이딩 윈도우 평균 확률 적용
        pred = extractor.predict_word(seg_arr)
        pred_words.append(pred)

    # 결과 출력
    print(f"Video: {video_name}")
    print("GT:   ", gt_words)
    print("Pred: ", pred_words)
    print()
