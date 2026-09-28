# trajectory/trajectory_manager.py
import math
from .smoothing import apply_simple_moving_average, filter_coordinate_jumps

class TrajectoryManager:
    def __init__(self, smoothing_window=5, max_jump_distance=50.0, min_stroke_points=3):
        self.raw_strokes = []
        self.current_stroke = []
        self.smoothing_window = smoothing_window
        self.max_jump_distance = max_jump_distance
        self.min_stroke_points = min_stroke_points

    def add_point(self, point):
        """Thêm một điểm vào stroke hiện tại (kèm lọc nhảy vị trí)."""
        if point is not None:
            if self.current_stroke:
                last_pt = self.current_stroke[-1]
                dist = math.hypot(point[0] - last_pt[0], point[1] - last_pt[1])
                if dist > self.max_jump_distance:
                    # Bỏ qua điểm bị giật / nhảy vọt do nhiễu tracking
                    return
            self.current_stroke.append(point)

    def finish_stroke(self):
        """Kết thúc một nét chữ (khi dừng viết) và lưu lại nếu không quá ngắn."""
        if len(self.current_stroke) >= self.min_stroke_points:
            self.raw_strokes.append(self.current_stroke)
        self.current_stroke = []

    def get_smoothed_strokes(self):
        """Trả về toàn bộ các stroke đã lọc cú nhảy và làm mượt bằng SMA."""
        smoothed_strokes = []
        all_strokes = list(self.raw_strokes)
        if len(self.current_stroke) >= self.min_stroke_points:
            all_strokes.append(self.current_stroke)

        for stroke in all_strokes:
            filtered = filter_coordinate_jumps(stroke, self.max_jump_distance)
            if len(filtered) >= self.min_stroke_points:
                smoothed = apply_simple_moving_average(filtered, self.smoothing_window)
                smoothed_strokes.append(smoothed)

        return smoothed_strokes

    def clear(self):
        """Xóa toàn bộ các nét đã viết."""
        self.raw_strokes = []
        self.current_stroke = []
