# tracking/hand_tracker.py
import cv2
import math
import os
from collections import Counter, deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Định nghĩa các khớp nối để vẽ skeleton bàn tay
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


class GestureStabilizer:
    """
    Lớp ổn định cử chỉ bàn tay bằng majority voting trong cửa sổ N frames liên tiếp.
    Giúp chống nhiễu / jitter do MediaPipe nhận sai 1-2 frame lẻ.
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def update(self, raw_gesture: str) -> str:
        self.history.append(raw_gesture)
        if not self.history:
            return "IDLE"
        counts = Counter(self.history)
        most_common, count = counts.most_common(1)[0]
        # Cần đa số trong cửa sổ để chốt gesture
        if count >= (len(self.history) // 2 + 1):
            return most_common
        return self.history[-1]

    def reset(self):
        self.history.clear()


def classify_hand_gesture(hand_landmarks) -> str:
    """
    Nhận diện các cử chỉ điều khiển Air-Writing từ 21 landmarks:
    1. WRITING: Ngón trỏ duỗi (OPEN), ngón giữa/áp út/út co (CLOSED).
    2. SPACE: Ngón trỏ & ngón út duỗi (OPEN), ngón giữa & áp út co (CLOSED).
    3. DELETE: Tất cả 4 ngón trỏ/giữa/áp út/út co (CLOSED - Nắm tay).
    4. MODE: Ngón trỏ & ngón giữa duỗi (OPEN), ngón áp út & út co (CLOSED - V-Sign).
    """
    # 8: Index tip, 6: Index PIP
    # 12: Middle tip, 10: Middle PIP
    # 16: Ring tip, 14: Ring PIP
    # 20: Pinky tip, 18: Pinky PIP
    index_open = hand_landmarks[8].y < hand_landmarks[6].y
    middle_open = hand_landmarks[12].y < hand_landmarks[10].y
    ring_open = hand_landmarks[16].y < hand_landmarks[14].y
    pinky_open = hand_landmarks[20].y < hand_landmarks[18].y

    if index_open and not middle_open and not ring_open and not pinky_open:
        return "WRITING"
    elif index_open and pinky_open and not middle_open and not ring_open:
        return "SPACE"
    elif not index_open and not middle_open and not ring_open and not pinky_open:
        return "DELETE"
    elif index_open and middle_open and not ring_open and not pinky_open:
        return "MODE"
    else:
        return "IDLE"


class HandTracker:
    def __init__(self, min_detection_con=0.7, min_tracking_con=0.5, pinch_threshold=40, stabilization_frames=5):
        # Tự động tải file model nếu chưa có
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            print("Đang tải model hand_landmarker.task...")
            import urllib.request
            urllib.request.urlretrieve(
                'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
                model_path
            )
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_detection_con,
            min_hand_presence_confidence=min_tracking_con
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.pinch_threshold = pinch_threshold # Giữ tham số tương thích cấu hình cũ
        self.stabilizer = GestureStabilizer(window_size=stabilization_frames)

    def process_frame(self, frame):
        """
        Xử lý frame bằng MediaPipe Tasks API, nhận diện gesture ổn định và trả về tọa độ ngón trỏ.
        Trả về: (frame, stable_gesture, fingertip_pos)
        stable_gesture: "WRITING", "SPACE", "DELETE", "MODE", "IDLE", "NO_HAND"
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        results = self.detector.detect(mp_image)
        
        raw_gesture = "NO_HAND"
        fingertip_pos = None
        
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                h, w, _ = frame.shape
                
                # Vẽ các đường nối skeleton
                for connection in HAND_CONNECTIONS:
                    pt1 = hand_landmarks[connection[0]]
                    pt2 = hand_landmarks[connection[1]]
                    x1, y1 = int(pt1.x * w), int(pt1.y * h)
                    x2, y2 = int(pt2.x * w), int(pt2.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                
                # Vẽ các điểm landmark
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                
                # Điểm 8 là đầu ngón trỏ
                lm8 = hand_landmarks[8]
                px8, py8 = int(lm8.x * w), int(lm8.y * h)
                fingertip_pos = (px8, py8)
                
                # Phân loại gesture thô từ landmarks
                raw_gesture = classify_hand_gesture(hand_landmarks)

        # Ổn định gesture bằng Majority Vote qua các frames
        stable_gesture = self.stabilizer.update(raw_gesture)
                    
        return frame, stable_gesture, fingertip_pos

