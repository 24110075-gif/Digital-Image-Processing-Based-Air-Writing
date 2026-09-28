# processing/pipeline_processor.py
import cv2
import numpy as np

def keep_main_group(image_mask: np.ndarray, kernel_size=(15, 15)) -> np.ndarray:
    """
    keep_main_group():
    1. Phép nở Morphological Dilation với kernel lớn (15x15) để nối các nét gần nhau của 1 ký tự.
    2. Phân tích thành phần liên thông (Connected Components) giữ lại cụm chữ chính lớn nhất.
    3. Loại bỏ hoàn toàn các điểm chấm / nét rác nằm tách biệt ở xa.
    """
    if image_mask is None or cv2.countNonZero(image_mask) == 0:
        return image_mask

    # 1. Dilation với kernel hình ellipse lớn
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    dilated = cv2.dilate(image_mask, kernel, iterations=1)

    # 2. Phân tích thành phần liên thông Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    if num_labels <= 1:
        return image_mask

    # Tìm nhãn có diện tích lớn nhất (bỏ qua background nhãn 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Tạo mask vùng chính từ nhãn lớn nhất
    main_group_mask = np.uint8(labels == largest_label) * 255

    # Lấy giao của ảnh gốc với mask nhóm chính đã lọc
    cleaned_mask = cv2.bitwise_and(image_mask, image_mask, mask=main_group_mask)
    return cleaned_mask


def render_trajectory_to_64x64(strokes, target_size=(64, 64), padding=6, line_thickness=3) -> np.ndarray:
    """
    render_trajectory():
    1. Lấy tất cả các điểm tọa độ từ danh sách strokes.
    2. Chuẩn hóa: Dịch chuyển về tâm/bbox, scale giữ đúng tỉ lệ aspect ratio.
    3. Vẽ trực tiếp các nét mượt lên mảng ảnh 64x64 bằng cv2.LINE_AA (khử răng cưa).
    4. Áp dụng keep_main_group() lọc các nét lạc.
    5. Áp dụng Dilation 3x3 để nét chữ đậm đều tương thích CNN.
    """
    all_points = [pt for stroke in strokes for pt in stroke]
    if not all_points:
        return np.zeros(target_size, dtype=np.uint8)

    # 1. Lấy Bounding Box của toàn bộ tập điểm
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    w = max(1, max_x - min_x)
    h = max(1, max_y - min_y)

    target_w, target_h = target_size
    drawable_w = target_w - 2 * padding
    drawable_h = target_h - 2 * padding

    # 2. Tính scale giữ tỷ lệ aspect ratio
    scale = min(drawable_w / w, drawable_h / h)

    # Căn giữa trong vùng vẽ
    scaled_w = w * scale
    scaled_h = h * scale
    offset_x = padding + (drawable_w - scaled_w) / 2.0
    offset_y = padding + (drawable_h - scaled_h) / 2.0

    # Chuyển đổi các điểm sang tọa độ không gian 64x64
    normalized_strokes = []
    for stroke in strokes:
        norm_stroke = []
        for (x, y) in stroke:
            nx = int(round((x - min_x) * scale + offset_x))
            ny = int(round((y - min_y) * scale + offset_y))
            norm_stroke.append((nx, ny))
        normalized_strokes.append(norm_stroke)

    # 3. Vẽ lên canvas 64x64 với cv2.LINE_AA (khử răng cưa)
    img_64 = np.zeros(target_size, dtype=np.uint8)
    for stroke in normalized_strokes:
        if len(stroke) == 1:
            cv2.circle(img_64, stroke[0], line_thickness, 255, -1)
        else:
            for i in range(1, len(stroke)):
                cv2.line(img_64, stroke[i - 1], stroke[i], 255, line_thickness, cv2.LINE_AA)

    # 4. keep_main_group(): Loại bỏ các điểm rác / nét vẽ lạc ở xa
    img_64 = keep_main_group(img_64, kernel_size=(15, 15))

    # 5. Dilation 3x3 làm đậm đường nét vừa vặn đầu vào CNN
    kernel_3x3 = np.ones((3, 3), np.uint8)
    img_64 = cv2.dilate(img_64, kernel_3x3, iterations=1)

    return img_64
