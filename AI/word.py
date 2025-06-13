import os
import glob
import json
import numpy as np
import torch
import cv2
import mediapipe as mp
from models.sign_model import SignModel

# ==== 설정 ==== 
VIDEO_DIR      = './processed_val_data/sign_data'
LABELS_DIR     = './processed_val_data/labels'
MODEL_PATH     = './checkpoints/transformer_finetuned.pth'
LABEL_MAP_PATH = './numpy_train_data/label_map.json'
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE    = 30
STRIDE         = 30

# Mediapipe Hands
mp_hands = mp.solutions.hands

# === 프레임 단위 정규화 함수 ===
def normalize_hand_keypoints(hand_landmarks):
    x_list = [lm.x for lm in hand_landmarks.landmark]
    y_list = [lm.y for lm in hand_landmarks.landmark]
    z_list = [lm.z for lm in hand_landmarks.landmark]
    x_center = np.mean(x_list); y_center = np.mean(y_list); z_center = np.mean(z_list)
    width = max(x_list) - min(x_list)
    height = max(y_list) - min(y_list)
    depth = max(z_list) - min(z_list)
    scale = max(width, height, depth) + 1e-6
    norm = []
    for x, y, z in zip(x_list, y_list, z_list):
        norm.append((x - x_center) / scale)
        norm.append((y - y_center) / scale)
        norm.append((z - z_center) / scale)
    return norm

# === 영상에서 키포인트 추출 및 전처리 ===
def extract_keypoints(video_path: str) -> np.ndarray:
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    keypoints_list = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # FPS downsampling
        if fps > WINDOW_SIZE and frame_count % int(fps // WINDOW_SIZE) != 0:
            frame_count += 1
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        frame_kp = []
        if results.multi_hand_landmarks:
            left, right = None, None
            for hd, lm in zip(results.multi_handedness, results.multi_hand_landmarks):
                if hd.classification[0].label == 'Left': left = lm
                else:                                right = lm
            for hand in (left, right):
                if hand:
                    frame_kp.extend(normalize_hand_keypoints(hand))
                else:
                    frame_kp.extend([0.0] * 63)
        else:
            frame_kp = [0.0] * 126

        keypoints_list.append(frame_kp)
        frame_count += 1

    cap.release()
    hands.close()

    arr = np.array(keypoints_list, dtype=np.float32)  # [T, 126]
    # 시퀀스별 표준화 (훈련과 일치)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-5
    arr = (arr - mean) / std

    # 짧은 시퀀스 패딩
    if arr.shape[0] < WINDOW_SIZE:
        pad = np.zeros((WINDOW_SIZE - arr.shape[0], arr.shape[1]), dtype=np.float32)
        arr = np.vstack([arr, pad])

    return arr

# === 슬라이딩 윈도우 예측 ===
def sliding_window_predict(keypoints: np.ndarray,
                           model: SignModel,
                           idx2label: dict) -> str:
    length = keypoints.shape[0]
    # 윈도우 시작 인덱스 생성 (끝부분 포함)
    if length >= WINDOW_SIZE:
        starts = list(range(0, length - WINDOW_SIZE + 1, STRIDE))
        if (length - WINDOW_SIZE) % STRIDE != 0:
            starts.append(length - WINDOW_SIZE)
    else:
        starts = [0]

    probs_accum = []
    for start in starts:
        window = keypoints[start:start + WINDOW_SIZE]
        tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([WINDOW_SIZE], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor, lengths)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs_accum.append(probs)

    avg_probs = np.mean(probs_accum, axis=0)
    pred_idx = int(np.argmax(avg_probs))
    return idx2label[pred_idx]

# ==== 라벨맵 및 모델 로딩 ====
with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    label_map = json.load(f)
labels = sorted(set(label_map.values()))
idx2label = {i: v for i, v in enumerate(labels)}
model = SignModel(input_size=126, num_classes=len(labels))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

# ==== 추론 수행 ====
total, correct = 0, 0
for vp in glob.glob(os.path.join(VIDEO_DIR, '*.mp4')):
    prefix = os.path.splitext(os.path.basename(vp))[0]
    # GT 라벨 추출
    gt = None
    json_path = os.path.join(LABELS_DIR, f"{prefix}_morpheme.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as jf:
            data = json.load(jf).get('data', [])
            if data and data[0].get('attributes'):
                gt = data[0]['attributes'][0].get('name')

    kp = extract_keypoints(vp)
    pred = sliding_window_predict(kp, model, idx2label)
    flag = '⭕' if pred == gt else '❌'
    print(f"{prefix:40s} | GT: {gt:10s} | Pred: {pred:10s} | {flag}")
    total += 1
    if pred == gt:
        correct += 1

print(f"\n전체 정확도: {correct}/{total} = {correct/total:.3f}")
