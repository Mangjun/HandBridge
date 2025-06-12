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

def normalize_hand_keypoints(hand_landmarks):
    x_list = [lm.x for lm in hand_landmarks.landmark]
    y_list = [lm.y for lm in hand_landmarks.landmark]
    z_list = [lm.z for lm in hand_landmarks.landmark]

    x_center = np.mean(x_list)
    y_center = np.mean(y_list)
    z_center = np.mean(z_list)

    width = max(x_list) - min(x_list)
    height = max(y_list) - min(y_list)
    depth = max(z_list) - min(z_list)
    scale = max(width, height, depth) + 1e-6

    normalized = []
    for x, y, z in zip(x_list, y_list, z_list):
        norm_x = (x - x_center) / scale
        norm_y = (y - y_center) / scale
        norm_z = (z - z_center) / scale
        normalized.extend([norm_x, norm_y, norm_z])
    return normalized

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
    if os.path.exists(npz_save_path):
        return (prefix, label)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
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
                    norm_kp = normalize_hand_keypoints(hand)
                    frame_keypoints.extend(norm_kp)
                else:
                    frame_keypoints.extend([0]*63)
        else:
            frame_keypoints = [0]*126

        keypoints_list.append(frame_keypoints)
        frame_count += 1

    cap.release()
    hands.close()

    keypoints_array = np.array(keypoints_list)
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

    os.system("shutdown /s /t 1")

if __name__ == "__main__":
    main()
