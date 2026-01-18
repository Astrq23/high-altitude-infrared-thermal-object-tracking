import numpy as np
import math

class DistanceEstimator:
    def __init__(self, focal_length=800, image_height=1080):
        """
        focal_length: Tiêu cự tính theo pixel.
        image_height: Chiều cao khung hình (để xác định tâm ảnh - Optical Center).
        """
        self.focal_length = focal_length
        self.cy = image_height / 2  # Tọa độ Y của tâm ảnh

    def estimate(self, bbox, drone_altitude, gimbal_pitch_deg):
        """
        Tính khoảng cách dựa trên góc nghiêng camera và vị trí điểm ảnh.
        
        Args:
            bbox: [x1, y1, x2, y2] - Tọa độ bounding box.
            drone_altitude: Độ cao bay (mét).
            gimbal_pitch_deg: Góc nghiêng camera (độ). 0=ngang, 90=thẳng đứng xuống.
            
        Returns:
            ground_dist: Khoảng cách mặt đất (mét).
            slant_dist: Khoảng cách đường chim bay (mét).
        """
        # Lấy chân của bounding box (y_max)
        _, _, _, y_bottom = bbox
        
        # 1. Tính góc lệch của pixel so với tâm ảnh (Alpha)
        # y_bottom > cy (nửa dưới ảnh) -> góc nhìn thấp hơn -> cộng thêm góc
        y_offset = y_bottom - self.cy
        alpha_rad = math.atan(y_offset / self.focal_length)
        
        # 2. Tổng góc nghiêng (Pitch + Alpha)
        # Lưu ý: Giả định pitch dương (0-90). Nếu SDK trả về âm cần abs() trước khi truyền vào.
        pitch_rad = math.radians(abs(gimbal_pitch_deg))
        total_angle = pitch_rad + alpha_rad

        # 3. Tính khoảng cách
        # Tránh chia cho 0 hoặc góc quá nhỏ
        if total_angle <= 0.01: return float('inf'), float('inf')
        
        # Nếu nhìn gần như thẳng đứng (gần 90 độ)
        if total_angle >= (math.pi / 2) - 0.01:
            return 0.0, drone_altitude

        ground_dist = drone_altitude / math.tan(total_angle)
        slant_dist = drone_altitude / math.sin(total_angle)
        
        return round(ground_dist, 2), round(slant_dist, 2)