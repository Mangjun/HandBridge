import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json

def augment_keypoints(keypoints):
    # Gaussian noise
    keypoints += np.random.normal(0, 0.01, keypoints.shape)
    # (필요하다면 scaling, shift, flip 등 추가 가능)
    return keypoints

class SignLanguageDataset(Dataset):
    def __init__(
        self,
        npz_dir,
        label_map_path,
        label_json_dir=None,
        prefix_list=None,         # 사용하려는 prefix만 사용 (cross-validation 지원)
        train=False               # train이면 augmentation 적용
    ):
        # prefix_list로 split 지정 가능 (없으면 전체 사용)
        all_npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
        if prefix_list is not None:
            self.npz_files = [f for f in all_npz_files if os.path.splitext(f)[0] in prefix_list]
        else:
            self.npz_files = sorted(all_npz_files)
        self.npz_dir = npz_dir
        with open(label_map_path, "r", encoding="utf-8") as f:
            self.label_map = json.load(f)
        self.label2idx = {v: i for i, v in enumerate(sorted(set(self.label_map.values())))}
        self.idx2label = {i: v for v, i in self.label2idx.items()}
        self.label_json_dir = label_json_dir
        self.train = train

    def __len__(self):
        return len(self.npz_files)
    
    def get_label_map(self):
        return self.label2idx

    def __getitem__(self, idx):
        npz_file = self.npz_files[idx]
        prefix = os.path.splitext(npz_file)[0]
        npz_path = os.path.join(self.npz_dir, npz_file)
        keypoints = np.load(npz_path)['keypoints']

        # === keypoint normalization은 npz 생성 시 이미 적용됨 ===
        # if keypoints.shape[0] > 0:
        #     keypoints = (keypoints - keypoints.mean(axis=0)) / (keypoints.std(axis=0) + 1e-5)
        #     # 학습 셋에만 augmentation
        #     if self.train:
        #         keypoints = augment_keypoints(keypoints)

        if keypoints.shape[0] > 0 and self.train:
            keypoints = augment_keypoints(keypoints)

        # === label 추출 ===
        if self.label_json_dir is None:
            # train: prefix로 바로 매칭
            label_name = self.label_map[prefix]
        else:
            # val: json에서 라벨명 추출
            json_path = os.path.join(self.label_json_dir, prefix + "_morpheme.json")
            with open(json_path, "r", encoding="utf-8") as f:
                label_json = json.load(f)
                label_name = label_json["data"][0]["attributes"][0]["name"]

        # label2idx에 없으면 None
        if label_name not in self.label2idx:
            return None

        label_idx = self.label2idx[label_name]
        length = keypoints.shape[0]
        return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(label_idx), torch.tensor(length)
