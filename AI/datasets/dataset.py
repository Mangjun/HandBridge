import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json

class SignLanguageDataset(Dataset):
    """NumPy로 변환된 keypoints와 라벨맵을 로드하는 데이터셋"""
    def __init__(self, npz_dir):
        self.npz_dir = Path(npz_dir)
        # .npz 파일 목록 로드
        self.files = sorted(self.npz_dir.glob("*.npz"))
        # 라벨맵 로드
        label_map_path = self.npz_dir / "label_map.json"
        with open(label_map_path, encoding="utf-8") as f:
            self.label_map = json.load(f)
        # 역인덱스 생성 (예측 후 문자열 라벨 반환에 사용)
        self.idx_to_label = {int(v): k for k, v in self.label_map.items()}

        print(f"[INFO] Loaded {len(self.files)} samples from {npz_dir}")
        print(f"[INFO] Label count: {len(self.label_map)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # .npz 로드
        npz_path = self.files[idx]
        data = np.load(npz_path)
        keypoints = torch.tensor(data["keypoints"], dtype=torch.float32)
        label = torch.tensor(int(data["label"]), dtype=torch.long)
        # 시퀀스 길이 반환 (for LSTM/GRU)
        length = torch.tensor(keypoints.shape[0], dtype=torch.long)
        return keypoints, label, length

    def get_label_map(self):
        return self.label_map

    def get_label_str(self, idx):
        return self.idx_to_label.get(idx, "Unknown")
