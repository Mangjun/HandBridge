import torch
import numpy as np
import cv2
import mediapipe as mp
import json
import glob
import os
import random
from models.sign_model import SignModel

# === 경로 설정 ===
VIDEO_DIR = './processed_train_data/sign_data'
LABEL_DIR = './processed_train_data/labels'
MODEL_PATH = './checkpoints/best_model.pth'
LABEL_MAP_PATH = './numpy_train_data/label_map.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Mediapipe 세팅 ===
mp_hands = mp.solutions.hands

def extract_keypoints_from_video(video_path):
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❗영상 파일을 열 수 없습니다: {video_path}")
        return np.zeros((0, 126), dtype=np.float32)
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 30
    frame_count = 0

    keypoints_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if fps > target_fps and frame_count % int(fps // target_fps) != 0:
            frame_count += 1
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
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
    return np.array(keypoints_list)  # [T, 126]

# === 모델/라벨맵 로딩 ===
with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    label_map = json.load(f)
label_names = sorted(set(label_map.values()))
idx2label = {i: v for i, v in enumerate(label_names)}
num_classes = len(idx2label)

model = SignModel(input_size=126, num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === 폴더 내 모든 mp4 파일 반복 ===
video_paths = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
random.shuffle(video_paths)

for video_path in video_paths:
    video_name = os.path.basename(video_path)
    prefix = os.path.splitext(video_name)[0]
    label_json_path = os.path.join(LABEL_DIR, prefix + "_morpheme.json")

    print(f"\n=== 영상: {video_name} ===")

    # === keypoint 추출 ===
    keypoints = extract_keypoints_from_video(video_path)
    if keypoints.shape[0] == 0:
        print("❗프레임에서 keypoint를 전혀 추출하지 못했습니다.")
        continue

    length = keypoints.shape[0]
    input_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([length], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor, lengths)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_label = idx2label[pred_idx]
        top5_idx = torch.topk(probs, 5, dim=1).indices[0].cpu().numpy()
        top5_labels = [idx2label[i] for i in top5_idx]

    print(f"  ✅ 예측: {pred_label} (class #{pred_idx})")
    print(f"  Top-5 예측: {top5_labels}")

    # === 실제 라벨 추출 ===
    if os.path.exists(label_json_path):
        with open(label_json_path, "r", encoding="utf-8") as f:
            label_json = json.load(f)
        try:
            real_label = label_json["data"][0]["attributes"][0]["name"]
        except Exception:
            real_label = "(라벨 없음)"
        print(f"  실제 라벨: {real_label}")
    else:
        print("  ❗라벨 json 파일을 찾을 수 없습니다.")

