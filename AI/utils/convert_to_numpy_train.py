import os
import glob
import json
import numpy as np
import cv2
import mediapipe as mp
from multiprocessing import Pool
from tqdm import tqdm

mp_hands = mp.solutions.hands

LABELS_DIR = './processed_train_data/labels'
VIDEO_DIR = './processed_train_data/sign_data'
OUTPUT_DIR = './numpy_train_data'
LABELMAP_PATH = './numpy_train_data/label_map.json'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_one(label_json_path):
    with open(label_json_path, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    meta_name = label_data.get("metaData", {}).get("name", "")

    if not meta_name:
        return None

    prefix = meta_name.replace('.mp4', '')
    video_path = os.path.join(VIDEO_DIR, meta_name)
    if not os.path.exists(video_path):
        return None

    try:
        label = label_data["data"][0]["attributes"][0]["name"]
    except Exception:
        label = ""

    npz_save_path = os.path.join(OUTPUT_DIR, prefix + ".npz")
    # 이미 파일이 있으면 npz 생성 스킵
    if os.path.exists(npz_save_path):
        return (prefix, label)

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

        # 양손 프레임 keypoint 저장용 (왼손-오른손 순서)
        frame_keypoints = []

        if results.multi_hand_landmarks:
            # handedness에 따라 왼손/오른손 구분
            left_hand = None
            right_hand = None
            for handedness, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                label_h = handedness.classification[0].label
                if label_h == 'Left':
                    left_hand = hand_landmarks
                elif label_h == 'Right':
                    right_hand = hand_landmarks
            # 왼손, 오른손 순서로 저장
            for hand in [left_hand, right_hand]:
                if hand is not None:
                    kp = []
                    for lm in hand.landmark:
                        kp.extend([lm.x, lm.y, lm.z])
                    frame_keypoints.extend(kp)
                else:
                    frame_keypoints.extend([0]*63)
        else:
            # 둘 다 인식 안 됨
            frame_keypoints = [0]*126

        keypoints_list.append(frame_keypoints)
        frame_count += 1

    cap.release()
    hands.close()

    keypoints_array = np.array(keypoints_list)  # shape: (프레임, 126)
    np.savez(npz_save_path, keypoints=keypoints_array)
    return (prefix, label)

def main():
    label_files = glob.glob(os.path.join(LABELS_DIR, '*.json'))
    label_map = dict()

    with Pool(processes=8) as pool, tqdm(total=len(label_files)) as pbar:
        for result in pool.imap_unordered(process_one, label_files):
            pbar.update()
            if result:
                prefix, label = result
                label_map[prefix] = label

    with open(LABELMAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"라벨맵 저장됨: {LABELMAP_PATH}")

if __name__ == "__main__":
    main()
