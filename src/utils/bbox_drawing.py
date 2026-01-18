"""Bounding box drawing utilities for object tracking visualization"""

import cv2


def draw_simple_box(img, box, track_id, distance, is_target=False):
    """Vẽ bounding box đơn giản với hình chữ nhật"""
    x1, y1, x2, y2 = map(int, box)
    color = (0, 0, 255) if is_target else (0, 255, 0)
    thickness = 1
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Vẽ điểm chân
    cv2.circle(img, (int((x1+x2)/2), y2), 4, (0, 0, 255), -1)
    
    # Hiển thị thông tin
    label = f"ID:{track_id} | DST:{distance:.1f}m"
    if is_target:
        label = f"TARGET {track_id} | DST:{distance:.1f}m"
    
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_corners_only(img, box, track_id, distance, is_target=False):
    """Vẽ chỉ các góc của bounding box"""
    x1, y1, x2, y2 = map(int, box)
    color = (0, 0, 255) if is_target else (0, 255, 0)
    thickness = 1
    line_len = min(int((x2-x1) * 0.25), int((y2-y1) * 0.25), 25)
    
    # Vẽ 4 góc
    cv2.line(img, (x1, y1), (x1 + line_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + line_len), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_len), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_len), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_len), color, thickness)
    
    # Hiển thị thông tin
    label = f"ID:{track_id} | DST:{distance:.1f}m"
    if is_target:
        label = f"TARGET {track_id} | DST:{distance:.1f}m"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_circle_box(img, box, track_id, distance, is_target=False):
    """Vẽ vòng tròn xung quanh đối tượng"""
    x1, y1, x2, y2 = map(int, box)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    radius = int(max(x2 - x1, y2 - y1) / 2) + 5
    color = (0, 0, 255) if is_target else (0, 255, 0)
    thickness = 1
    
    cv2.circle(img, (cx, cy), radius, color, thickness)
    
    # Vẽ dấu tại tâm
    cv2.circle(img, (cx, cy), 1, color, -1)
    
    # Hiển thị thông tin
    label = f"ID:{track_id} | DST:{distance:.1f}m"
    if is_target:
        label = f"TARGET {track_id} | DST:{distance:.1f}m"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (cx - w//2, cy - radius - 25), (cx + w//2, cy - radius - 5), color, -1)
    cv2.putText(img, label, (cx - w//2, cy - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_rounded_box(img, box, track_id, distance, is_target=False):
    """Vẽ bounding box với các góc bo tròn"""
    x1, y1, x2, y2 = map(int, box)
    color = (0, 0, 255) if is_target else (0, 255, 0)
    thickness = 1
    radius = 10
    
    # Vẽ các cạnh
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.line(img, (x2 - radius, y2), (x1 + radius, y2), color, thickness)
    cv2.line(img, (x1, y2 - radius), (x1, y1 + radius), color, thickness)
    
    # Vẽ các góc bo tròn
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 270, 0, 90, color, thickness)
    
    # Hiển thị thông tin
    label = f"ID:{track_id} | DST:{distance:.1f}m"
    if is_target:
        label = f"TARGET {track_id} | DST:{distance:.1f}m"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_tech_callout(img, box, track_id, distance, is_target=False):
    """Vẽ bounding box với style technical callout (góc + đường dẫn)"""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    
    primary_color = (0, 0, 255) if is_target else (0, 255, 0)
    
    line_len = min(int(w * 0.3), int(h * 0.3), 30)
    thickness = 1

    cv2.line(img, (x1, y1), (x1 + line_len, y1), primary_color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + line_len), primary_color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_len, y1), primary_color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_len), primary_color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_len, y2), primary_color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_len), primary_color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_len, y2), primary_color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_len), primary_color, thickness)

    img_h, img_w = img.shape[:2]
    
    start_pt = (x2, y1)
    elbow_pt = (x2 + 30, y1 - 30)
    end_pt = (x2 + 140, y1 - 30)
    
    text_align_left = True

    if x2 + 150 > img_w:
        start_pt = (x1, y1)
        elbow_pt = (x1 - 30, y1 - 30)
        end_pt = (x1 - 140, y1 - 30)
        text_align_left = False
    
    if y1 - 50 < 0:
        start_pt = (start_pt[0], y2)
        elbow_pt = (elbow_pt[0], y2 + 30)
        end_pt = (end_pt[0], y2 + 30)

    cv2.line(img, start_pt, elbow_pt, primary_color, 2)
    cv2.line(img, elbow_pt, end_pt, primary_color, 2)
    cv2.circle(img, elbow_pt, 3, primary_color, -1)

    label_id = f"ID: {track_id}"
    label_dist = f"DST: {distance:.1f}m"
    if is_target:
        label_id = f"TARGET-{track_id}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thick = 2
    
    (w_id, h_id), _ = cv2.getTextSize(label_id, font, font_scale, font_thick)
    (w_dst, h_dst), _ = cv2.getTextSize(label_dist, font, font_scale, 1)
    
    max_w = max(w_id, w_dst)
    
    if text_align_left:
        txt_x = elbow_pt[0] + 10
    else:
        txt_x = elbow_pt[0] - max_w - 10

    cv2.putText(img, label_id, (txt_x, elbow_pt[1] - 5), font, font_scale, primary_color, font_thick)
    cv2.putText(img, label_dist, (txt_x, elbow_pt[1] + 18), font, font_scale, (200, 200, 200), 1)
