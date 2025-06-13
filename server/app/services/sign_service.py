import torch
import numpy as np
import json

from app.models.sign_model import SignModel

class SignWordExtractor:
    def __init__(
        self,
        model_path: str,
        label_map_path: str,
        input_size: int = 126,
        device: str = "cpu",
        window_size: int = 30,
        stride: int = 30,
    ):
        # 장치 설정
        self.device = torch.device(device)

        # 라벨맵 로드
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        labels = sorted(set(label_map.values()))
        self.label2idx = {v: i for i, v in enumerate(labels)}
        self.idx2label = {i: v for i, v in enumerate(labels)}

        # 모델 로드
        num_classes = len(labels)
        self.model = SignModel(input_size=input_size, num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device).eval()

        self.window_size = window_size
        self.stride = stride

    def predict_word(self, kp: np.ndarray) -> str:
        """
        전체 세그먼트 키포인트에서 한 단어 추출.

        개선: 슬라이딩 윈도우 대신 균일 샘플링으로 keyframes 선택
        """
        # numpy -> 변환
        arr = np.array(kp, dtype=np.float32)
        T, C = arr.shape

        # 시퀀스 표준화
        mean = arr.mean(axis=0)
        std = arr.std(axis=0) + 1e-5
        arr = (arr - mean) / std

        # 동적 샘플링: segment 길이 상관없이 window_size만큼 샘플링
        if T >= self.window_size:
            # 균일 간격으로 인덱스 선택
            idxs = np.linspace(0, T-1, num=self.window_size, dtype=int)
            sampled = arr[idxs]
        else:
            # 짧으면 패딩
            pad = np.zeros((self.window_size - T, C), dtype=np.float32)
            sampled = np.vstack([arr, pad])

        # 모델 예측
        x = torch.tensor(sampled, dtype=torch.float32).unsqueeze(0).to(self.device)
        lengths = torch.tensor([self.window_size], dtype=torch.long).to(self.device)
        with torch.no_grad():
            logits = self.model(x, lengths)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        return self.idx2label[pred_idx]
