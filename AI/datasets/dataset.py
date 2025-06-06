import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json

class SignLanguageDataset(torch.utils.data.Dataset):
    def __init__(self, npz_dir, label_map_path, label_json_dir=None):
        self.npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
        self.npz_dir = npz_dir
        with open(label_map_path, "r", encoding="utf-8") as f:
            self.label_map = json.load(f)
        self.label2idx = {v: i for i, v in enumerate(sorted(set(self.label_map.values())))}
        self.idx2label = {i: v for v, i in self.label2idx.items()}

        # 학습셋: npz prefix로 label_map 바로 사용
        # 검증셋: npz prefix로 label_json_dir에서 json 불러와 라벨 추출 → label_map에서 인덱스
        self.label_json_dir = label_json_dir

    def __len__(self):
        return len(self.npz_files)
    
    def get_label_map(self):
        return self.label2idx

    def __getitem__(self, idx):
        npz_file = self.npz_files[idx]
        prefix = os.path.splitext(npz_file)[0]
        npz_path = os.path.join(self.npz_dir, npz_file)
        keypoints = np.load(npz_path)['keypoints']

        # 라벨명 추출
        if self.label_json_dir is None:
            # 학습셋
            label_name = self.label_map[prefix]
        else:
            # 검증셋: json에서 라벨명 추출
            json_path = os.path.join(self.label_json_dir, prefix + "_morpheme.json")
            with open(json_path, "r", encoding="utf-8") as f:
                label_json = json.load(f)
                label_name = label_json["data"][0]["attributes"][0]["name"]

        # label2idx에 없으면 None 반환
        if label_name not in self.label2idx:
            return None

        label_idx = self.label2idx[label_name]
        length = keypoints.shape[0]
        return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(label_idx), torch.tensor(length)
