import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from ..core.config import MEDIAPIPE_CONFIG
from scipy.interpolate import interp1d

class MediapipeService:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(**MEDIAPIPE_CONFIG["hands"])
        self.target_fps = 30 

    def process_video(self, video_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """비디오를 처리하고 키포인트를 추출합니다."""
        cap = cv2.VideoCapture(str(video_path))
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / original_fps if original_fps > 0 else 0

        target_frames = int(duration * self.target_fps)
        
        original_timestamps = np.linspace(0, duration, total_frames)
        target_timestamps = np.linspace(0, duration, target_frames)

        video_info = {
            "original_fps": original_fps,
            "target_fps": self.target_fps,
            "total_frames": total_frames,
            "target_frames": target_frames,
            "duration": duration,
            "resolution": (width, height)
        }

        all_keypoints = []
        frame_idx = 0
        last_valid_keypoints = None  

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                results = self.hands.process(rgb_frame)
                
                frame_keypoints = []
                hands_detected = False
                
                if results.multi_hand_landmarks:
                    left_hand = None
                    right_hand = None
                    for handedness, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                        label = handedness.classification[0].label
                        if label == 'Left':
                            left_hand = hand_landmarks
                        elif label == 'Right':
                            right_hand = hand_landmarks
                    
                    for hand in [left_hand, right_hand]:
                        if hand is not None:
                            kp = []
                            for lm in hand.landmark:
                                kp.extend([lm.x, lm.y, lm.z])
                            frame_keypoints.extend(kp)
                            hands_detected = True
                        else:
                            if last_valid_keypoints is not None:
                                frame_keypoints.extend(last_valid_keypoints[len(frame_keypoints):len(frame_keypoints)+63])
                            else:
                                frame_keypoints.extend([0]*63)
                    
                    if hands_detected:
                        last_valid_keypoints = frame_keypoints.copy()
                else:
                    if last_valid_keypoints is not None:
                        frame_keypoints = last_valid_keypoints.copy()
                    else:
                        frame_keypoints = [0]*126
                
                all_keypoints.append(frame_keypoints)
                frame_idx += 1

        finally:
            cap.release()
            self.hands.close()

        all_keypoints = np.array(all_keypoints)
        
        for i in range(len(all_keypoints)):
            if np.all(all_keypoints[i] == 0) and i > 0:
                all_keypoints[i] = all_keypoints[i-1].copy()

        interpolated = np.zeros((target_frames, 126))
        for j in range(126):
            if not np.all(all_keypoints[:, j] == 0):  
                interpolator = interp1d(original_timestamps, all_keypoints[:, j], kind='linear', bounds_error=False, fill_value="extrapolate")
                interpolated[:, j] = interpolator(target_timestamps)

        video_info["processed_frames"] = len(interpolated)

        return interpolated, video_info 