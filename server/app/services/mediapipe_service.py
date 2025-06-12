import cv2
import numpy as np
import mediapipe as mp

class MediapipeService:
    def __init__(self, static_image_mode=False, max_num_hands=2,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    @staticmethod
    def normalize_hand_keypoints(hand_landmarks):
        x_list = [lm.x for lm in hand_landmarks.landmark]
        y_list = [lm.y for lm in hand_landmarks.landmark]
        z_list = [lm.z for lm in hand_landmarks.landmark]

        x_center = np.mean(x_list)
        y_center = np.mean(y_list)
        z_center = np.mean(z_list)

        width = max(x_list) - min(x_list)
        height = max(y_list) - min(y_list)
        depth = max(z_list) - min(z_list)
        scale = max(width, height, depth) + 1e-6

        normalized = []
        for x, y, z in zip(x_list, y_list, z_list):
            norm_x = (x - x_center) / scale
            norm_y = (y - y_center) / scale
            norm_z = (z - z_center) / scale
            normalized.extend([norm_x, norm_y, norm_z])
        return normalized

    def extract_keypoints_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        all_keypoints = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            frame_keypoints = []
            if results.multi_hand_landmarks:
                left_hand, right_hand = None, None
                for handedness, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                    label_h = handedness.classification[0].label
                    if label_h == 'Left':
                        left_hand = hand_landmarks
                    elif label_h == 'Right':
                        right_hand = hand_landmarks
                for hand in [left_hand, right_hand]:
                    if hand is not None:
                        norm_kp = self.normalize_hand_keypoints(hand)
                        frame_keypoints.extend(norm_kp)
                    else:
                        frame_keypoints.extend([0]*63)
            else:
                frame_keypoints = [0]*126
            all_keypoints.append(frame_keypoints)

        cap.release()
        return np.array(all_keypoints)  # shape: (num_frames, 126)