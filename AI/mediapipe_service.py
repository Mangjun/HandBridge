import cv2
import numpy as np
import mediapipe as mp

class MediapipeService:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        target_fps: int = 30,
        window_size: int = 30,
    ):
        """
        Mediapipe 기반 영상 전처리 서비스

        target_fps: 전처리 시 목표 fps
        window_size: 최소 프레임 수 (짧은 영상 패딩용)
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.target_fps = target_fps
        self.window_size = window_size

    @staticmethod
    def normalize_frame_kp(hand_landmarks) -> list:
        """프레임별 손 랜드마크 중심/스케일 정규화"""
        x_list = [lm.x for lm in hand_landmarks.landmark]
        y_list = [lm.y for lm in hand_landmarks.landmark]
        z_list = [lm.z for lm in hand_landmarks.landmark]

        x_center, y_center, z_center = np.mean(x_list), np.mean(y_list), np.mean(z_list)
        width, height, depth = max(x_list) - min(x_list), max(y_list) - min(y_list), max(z_list) - min(z_list)
        scale = max(width, height, depth) + 1e-6

        normalized = []
        for x, y, z in zip(x_list, y_list, z_list):
            normalized.extend([
                (x - x_center) / scale,
                (y - y_center) / scale,
                (z - z_center) / scale,
            ])
        return normalized

    def extract_keypoints_from_video(self, video_path: str) -> np.ndarray:
        """
        1) FPS downsampling → target_fps
        2) 손 키포인트 중심/스케일 정규화
        3) 시퀀스별 평균·표준편차 정규화
        4) 짧은 영상 패딩 to window_size

        Returns:
            np.ndarray: shape (T', 126)
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        frame_count = 0
        keypoints_list = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1) downsample
            if fps > self.target_fps and frame_count % int(fps // self.target_fps) != 0:
                frame_count += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            # 2) per-frame normalization
            frame_kp = []
            if results.multi_hand_landmarks:
                left, right = None, None
                for hd, lm in zip(results.multi_handedness, results.multi_hand_landmarks):
                    if hd.classification[0].label == 'Left':
                        left = lm
                    else:
                        right = lm
                for hand in (left, right):
                    if hand is not None:
                        frame_kp.extend(self.normalize_frame_kp(hand))
                    else:
                        frame_kp.extend([0.0] * 63)
            else:
                frame_kp = [0.0] * 126

            keypoints_list.append(frame_kp)
            frame_count += 1

        cap.release()

        # 배열 변환
        arr = np.array(keypoints_list, dtype=np.float32)  # [T,126]
        if arr.size == 0:
            # 프레임 추출 실패 시 최소 크기 반환
            return np.zeros((self.window_size, 126), dtype=np.float32)

        # 3) 시퀀스별 표준화
        mean = arr.mean(axis=0)
        std = arr.std(axis=0) + 1e-5
        arr = (arr - mean) / std

        # 4) 짧은 영상 패딩
        if arr.shape[0] < self.window_size:
            pad = np.zeros((self.window_size - arr.shape[0], arr.shape[1]), dtype=np.float32)
            arr = np.vstack([arr, pad])

        return arr

    def close(self):
        """MediaPipe 리소스 해제"""
        self.hands.close()
