import cv2
import numpy as np
from typing import Dict, List, Any, Tuple
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

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
    
    def process_video(self, video_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        비디오 파일의 마지막 프레임에서 감정을 분석합니다.
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            (감정 분석 결과, 비디오 정보)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
            
        # 비디오 정보 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 비디오 정보
        video_info = {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height
        }
        
        # 마지막 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, last_frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError("마지막 프레임을 읽을 수 없습니다.")
        
        # 마지막 프레임 감정 분석
        emotion_result = self.process_frame(last_frame)
        
        # 결과 출력
        print("\n=== 감정 분석 결과 ===")
        print(f"주요 감정: {emotion_result['dominant_emotion']['emotion']}")
        print(f"확률: {emotion_result['dominant_emotion']['probability']:.2%}")
        print("\n전체 감정 분포:")
        for emotion, prob in emotion_result['emotions'].items():
            print(f"- {emotion}: {prob:.2%}")
        print("==================\n")
        
        return emotion_result, video_info 