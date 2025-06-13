import os
import glob
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
from multiprocessing import Pool

# 설정
VIDEO_DIR = './video_path/sign_data'
OUTPUT_DIR = './numpy_sentence_data'
FPS = 30
NUM_WORKERS = 8  # 멀티프로세싱 워커 수
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mediapipe Hands 초기화
mp_hands = mp.solutions.hands
# normalize_frame_kp는 MediapipeService에 정의된 정규화 함수
from mediapipe_service import MediapipeService
normalize = MediapipeService.normalize_frame_kp

# 개별 비디오에서 keypoints 추출 및 저장 함수
def process_and_save(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    keypoints_list = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Downsample
        if fps > FPS and frame_idx % int(fps // FPS) != 0:
            frame_idx += 1
            continue
        frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        results = hands.process(rgb)
        frame_kp = []
        if results.multi_hand_landmarks:
            lh, rh = None, None
            for hd, lm in zip(results.multi_handedness, results.multi_hand_landmarks):
                if hd.classification[0].label == 'Left':
                    lh = lm
                else:
                    rh = lm
            for hand in (lh, rh):
                if hand:
                    frame_kp.extend(normalize(hand))
                else:
                    frame_kp.extend([0.0] * 63)
        else:
            frame_kp = [0.0] * 126
        keypoints_list.append(frame_kp)
        hands.close()

    cap.release()
    # numpy 저장
    prefix = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(OUTPUT_DIR, prefix + '.npz')
    np.savez(out_path, keypoints=np.array(keypoints_list, dtype=np.float32))

if __name__ == '__main__':
    video_paths = glob.glob(os.path.join(VIDEO_DIR, '*.mp4'))
    # Pool로 멀티프로세싱
    with Pool(processes=NUM_WORKERS) as pool:
        list(tqdm(pool.imap_unordered(process_and_save, video_paths),
                  total=len(video_paths), desc='Sentence Videos'))
    print('✅ Sentences keypoints saved to', OUTPUT_DIR)
