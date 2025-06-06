import os
import glob
import numpy as np
import cv2
import mediapipe as mp
import json
from multiprocessing import Pool
from tqdm import tqdm

mp_hands = mp.solutions.hands

VIDEO_DIR = './processed_val_data/sign_data'
LABELS_DIR = './processed_val_data/labels'
OUTPUT_DIR = './numpy_val_data'
LABEL2IDX_PATH = './numpy_train_data/label2idx.json'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 라벨명 → 인덱스 맵
with open(LABEL2IDX_PATH, 'r', encoding='utf-8') as f:
    label2idx = json.load(f)

def get_label_from_json(json_path):
    # morpheme.json에서 label 추출
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    try:
        label = data["data"][0]["attributes"][0]["name"]
    except Exception:
        label = ""
    return label

def process_one(label_json_path):
    with open(label_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    meta_name = data.get("metaData", {}).get("name", "")
    if not meta_name:
        return

    prefix = meta_name.replace('.mp4', '')
    video_path = os.path.join(VIDEO_DIR, meta_name)
    if not os.path.exists(video_path):
        print(f"비디오 없음: {video_path}")
        return

    # 라벨명 추출
    try:
        label = data["data"][0]["attributes"][0]["name"]
    except Exception:
        label = ""

    if label not in label2idx:
        print(f"라벨맵에 없는 label: {label}")
        return

    label_idx = label2idx[label]

    npz_save_path = os.path.join(OUTPUT_DIR, prefix + ".npz")
    if os.path.exists(npz_save_path):
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # 두 손까지 탐지
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    keypoints_list = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 30
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if fps > target_fps and frame_count % int(fps // target_fps) != 0:
            frame_count += 1
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # 양손 keypoint (왼손-오른손 순)
        frame_keypoints = []
        if results.multi_hand_landmarks:
            left_hand = None
            right_hand = None
            for handedness, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                label_h = handedness.classification[0].label
                if label_h == 'Left':
                    left_hand = hand_landmarks
                elif label_h == 'Right':
                    right_hand = hand_landmarks
            for hand in [left_hand, right_hand]:
                if hand is not None:
                    kp = []
                    for lm in hand.landmark:
                        kp.extend([lm.x, lm.y, lm.z])
                    frame_keypoints.extend(kp)
                else:
                    frame_keypoints.extend([0]*63)
        else:
            frame_keypoints = [0]*126
        keypoints_list.append(frame_keypoints)
        frame_count += 1

    cap.release()
    hands.close()

    keypoints_array = np.array(keypoints_list)  # shape: (프레임, 126)
    np.savez(npz_save_path, keypoints=keypoints_array, label=label_idx)
    return

def main():
    label_files = glob.glob(os.path.join(LABELS_DIR, '*_morpheme.json'))

    with Pool(processes=8) as pool, tqdm(total=len(label_files)) as pbar:
        for _ in pool.imap_unordered(process_one, label_files):
            pbar.update()

if __name__ == "__main__":
    main()
