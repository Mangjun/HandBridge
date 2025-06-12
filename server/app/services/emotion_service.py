import cv2
import numpy as np
from typing import Dict, List, Any, Tuple
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class EmotionService:
    def __init__(self):
        """
        감정 인식 서비스 초기화
        """
        self.model_name = "dima806/facial_emotions_image_detection"
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 감정 라벨 매핑
        self.labels = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        단일 프레임에서 감정을 분석합니다.
        
        Args:
            frame: BGR 형식의 이미지 프레임
            
        Returns:
            감정 분석 결과
        """
        # BGR에서 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        
        # 이미지 전처리
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 감정 예측
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            predicted_idx = torch.argmax(probs).item()
            
        # 결과 포맷팅
        emotions = {label: prob.item() for label, prob in zip(self.labels, probs)}
        
        result = {
            'emotions': emotions,
            'dominant_emotion': {
                'emotion': self.labels[predicted_idx],
                'probability': probs[predicted_idx].item()
            }
        }
            
        return result
    
    def process_video(self, video_path: str) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        비디오에서 감정을 분석합니다.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("비디오 파일을 열 수 없습니다.")

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                raise ValueError("비디오에 프레임이 없습니다.")

            # 중간 프레임을 사용
            middle_frame_idx = frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                # 중간 프레임을 읽지 못하면 첫 번째 프레임 시도
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    raise ValueError("프레임을 읽을 수 없습니다.")

            # 프레임에서 감정 분석 수행
            emotion_result = self.process_frame(frame)
            dominant_emotion = emotion_result['dominant_emotion']

            cap.release()
            
            return {
                "emotion": dominant_emotion['emotion'],
                "confidence": dominant_emotion['probability'],
                "frame_analyzed": middle_frame_idx,
                "emotions": emotion_result['emotions']  # 전체 감정 분포도 포함
            }, frame

        except Exception as e:
            logger.error(f"감정 분석 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본값 반환
            return {
                "emotion": "neutral",
                "confidence": 1.0,
                "frame_analyzed": 0,
                "error": str(e),
                "emotions": {emotion: 0.0 for emotion in self.labels}
            }, np.zeros((480, 640, 3), dtype=np.uint8)  # 빈 프레임 반환 