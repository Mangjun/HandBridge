import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from mediapipe_service import MediapipeService
from tqdm import tqdm
from multiprocessing import Pool

class SentenceSegmentDataset(Dataset):
    def __init__(self, npz_dir: str, label_dir: str, label_map: dict,
                 window_size=30, transform=None, target_fps=30):
        self.window_size = window_size
        self.transform = transform
        self.target_fps = target_fps

        # 1) npz 파일 리스트 & segment info
        all_npz = sorted(glob.glob(os.path.join(npz_dir, '*.npz')))
        self.items = []
        for npz_path in all_npz:
            prefix = os.path.splitext(os.path.basename(npz_path))[0]
            jp = os.path.join(label_dir, prefix + '_morpheme.json')
            if not os.path.exists(jp): continue
            with open(jp, 'r', encoding='utf-8') as f:
                data = json.load(f).get('data', [])
            for seg in data:
                w = seg['attributes'][0]['name']
                if w not in label_map: continue
                self.items.append({
                    'npz': npz_path,
                    'start': seg['start'],
                    'end': seg['end'],
                    'label': label_map[w]
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        arr = np.load(it['npz'])['keypoints']  # [T,126]
        s = int(it['start'] * self.target_fps)
        e = int(it['end']   * self.target_fps)
        seg = arr[s:e]
        T, C = seg.shape
        if T >= self.window_size:
            idxs = np.linspace(0, T-1, num=self.window_size, dtype=int)
            seg = seg[idxs]
        else:
            pad = np.zeros((self.window_size-T, C), dtype=np.float32)
            seg = np.vstack([seg, pad])
        if self.transform:
            seg = self.transform(seg)
        return torch.tensor(seg, dtype=torch.float32), torch.tensor(it['label'], dtype=torch.long)

