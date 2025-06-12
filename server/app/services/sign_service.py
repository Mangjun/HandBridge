import torch
import numpy as np
import json

from app.models.sign_model import SignModel

class SignWordExtractor:
    def __init__(self, model_path, label_map_path, input_size=126, device="cpu", window_size=30, stride=30):
        # 라벨맵 로드
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        self.label2idx = {v: i for i, v in enumerate(sorted(set(label_map.values())))}
        self.idx2label = {i: v for v, i in self.label2idx.items()}
        num_classes = len(self.label2idx)
        
        # 모델 로드
        self.model = SignModel(input_size=input_size, num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.device = device

        self.window_size = window_size
        self.stride = stride

    def predict_words(self, all_keypoints):
        pred_words = []
        all_keypoints = np.array(all_keypoints)
        total_frames = len(all_keypoints)
        for start in range(0, total_frames - self.window_size + 1, self.stride):
            window_kp = all_keypoints[start:start + self.window_size]
            keypoints_tensor = torch.tensor(window_kp, dtype=torch.float32).unsqueeze(0)
            lengths_tensor = torch.tensor([self.window_size])
            with torch.no_grad():
                outputs = self.model(keypoints_tensor, lengths_tensor)
                probs = torch.softmax(outputs, dim=1)
                top1_idx = probs.argmax(dim=1).item()
                top1_label = self.idx2label[top1_idx]
                pred_words.append(top1_label)
        return pred_words
