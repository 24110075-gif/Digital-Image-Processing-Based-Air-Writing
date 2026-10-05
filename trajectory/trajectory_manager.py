# trajectory/trajectory_manager.py
import math
from .smoothing import apply_simple_moving_average, filter_coordinate_jumps

class TrajectoryManager:
    def __init__(self, smoothing_window=5, max_jump_distance=50.0, min_stroke_points=3, max_consecutive_rejects=3):
        self.raw_strokes = []
        self.current_stroke = []
        self._cached_smoothed_raw_strokes = []  # Cache cho các nét đã hoàn thành
        self.smoothing_window = smoothing_window
        self.max_jump_distance = max_jump_distance
        self.min_stroke_points = min_stroke_points
        self.max_consecutive_rejects = max_consecutive_rejects
        
        # Thống kê Debug theo dõi hiệu năng và mất nét
        self.points_received = 0
        self.points_rejected = 0
        self.large_jumps = 0
        self.reanchor_count = 0
        self.consecutive_rejects = 0

    def add_point(self, point):
        """Thêm một điểm vào stroke hiện tại (kèm lọc nhảy vị trí & cơ chế RE-ANCHOR khi viết nhanh)."""
        if point is not None:
            self.points_received += 1
            if self.current_stroke:
                last_pt = self.current_stroke[-1]
                dist = math.hypot(point[0] - last_pt[0], point[1] - last_pt[1])
                if dist > self.max_jump_distance:
                    self.large_jumps += 1
                    self.consecutive_rejects += 1
                    if self.consecutive_rejects < self.max_consecutive_rejects:
                        # Nhiễu nhảy ngẫu nhiên (Impulse Noise Spike): Bỏ qua
                        self.points_rejected += 1
                        return
                    else:
                        # Người dùng viết nhanh liên tục: RE-ANCHOR tiếp tục tracking, không bị freeze!
                        self.reanchor_count += 1
                        self.consecutive_rejects = 0
                else:
                    self.consecutive_rejects = 0
            
            self.current_stroke.append(point)

    def finish_stroke(self):
        """Kết thúc một nét chữ (khi dừng viết) và lưu lại nếu không quá ngắn."""
        if len(self.current_stroke) >= self.min_stroke_points:
            self.raw_strokes.append(self.current_stroke)
            # Tối ưu: Mượt nét vừa xong 1 lần và lưu vào cache (không lọc lại lần 2 để tránh mất nét)
            smoothed = apply_simple_moving_average(self.current_stroke, self.smoothing_window)
            self._cached_smoothed_raw_strokes.append(smoothed)
        self.current_stroke = []
        self.consecutive_rejects = 0

    def get_smoothed_strokes(self):
        """Trả về toàn bộ các stroke (sử dụng cache nét cũ + mượt nét hiện tại)."""
        smoothed_strokes = list(self._cached_smoothed_raw_strokes)
        if len(self.current_stroke) >= self.min_stroke_points:
            smoothed = apply_simple_moving_average(self.current_stroke, self.smoothing_window)
            smoothed_strokes.append(smoothed)

        return smoothed_strokes

    def clear(self):
        """Xóa toàn bộ các nét đã viết và reset thống kê."""
        self.raw_strokes = []
        self.current_stroke = []
        self._cached_smoothed_raw_strokes = []
        self.consecutive_rejects = 0


