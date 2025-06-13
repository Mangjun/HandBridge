import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from mediapipe_service import MediapipeService

def augment_segment(kp: np.ndarray) -> np.ndarray:
    # 1) Gaussian noise
    kp = kp + np.random.normal(0, 0.01, kp.shape)
    # 2) Time warping
    T = kp.shape[0]
    scale = np.random.uniform(0.9, 1.1)
    idxs = np.clip((np.arange(T) * scale).astype(int), 0, T-1)
    kp = kp[idxs]
    # 3) Frame dropout
    drop = np.random.choice(T, size=int(0.1 * T), replace=False)
    kp[drop] = 0
    # 4) Horizontal flip
    if np.random.rand() < 0.5:
        kp[:, 0::3] *= -1
    return kp

class SignLanguageDataset(Dataset):
    def __init__(
        self,
        npz_dir: str,
        label_map_path: str,
        label_json_dir: str = None,
        train: bool = False,
        transform=None,
    ):
        self.npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
        self.npz_dir = npz_dir
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        # 라벨명 → 인덱스
        labels = sorted(set(label_map.values()))
        self.label2idx = {v: i for i, v in enumerate(labels)}
        self.idx2label = {i: v for v, i in self.label2idx.items()}
        self.label_map = label_map
        self.label_json_dir = label_json_dir
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.npz_files)

    def get_label_map(self):
        return self.label2idx

    def __getitem__(self, idx):
        npz_file = self.npz_files[idx]
        prefix = os.path.splitext(npz_file)[0]
        data = np.load(os.path.join(self.npz_dir, npz_file))['keypoints']  # [T,126]

        # 1) 시퀀스 정규화
        if data.shape[0] > 0:
            mean = data.mean(axis=0)
            std  = data.std(axis=0) + 1e-5
            data = (data - mean) / std
            if self.train and self.transform:
                data = self.transform(data)

        # 2) 레이블 추출
        if self.label_json_dir is None:
            label_name = self.label_map.get(prefix)
        else:
            json_path = os.path.join(self.label_json_dir, prefix + '_morpheme.json')
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    j = json.load(f)
                    label_name = j['data'][0]['attributes'][0]['name']
            except:
                label_name = None

        if label_name not in self.label2idx:
            return None

        label = self.label2idx[label_name]
        length = data.shape[0]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long), torch.tensor(length, dtype=torch.long)
