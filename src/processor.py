import cv2
import numpy as np
import os
import subprocess
import motmetrics as mm
from ultralytics import YOLO
import gradio as gr

# Import class mới đã sửa
from src.utils.distance import DistanceEstimator
from src.tracker.gmc import GMC
from src.tracker.kalman import KalmanBoxTracker
from src.tracker.association import associate_detections_to_trackers
from src.utils.gt_loader import load_mot_gt
import torch

if torch.cuda.is_available():
    DEVICE = 0
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

print(f"Đang sử dụng thiết bị: {DEVICE}")

MODEL_OPTIONS = {
    "EfficientNetB0": "models/best.pt",
    "EfficientNetB3": "yolov8n.pt",
    "MobileNet": "yolov8s.pt",
    "ConvNext-T": "yolov8n.pt",
    "ConvNext-S": "yolov8s.pt"
}

def calculate_iou_single(box1, box2):
    xx1 = max(box1[0], box2[0]); yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2]); yy2 = min(box1[3], box2[3])
    w = max(0, xx2 - xx1); h = max(0, yy2 - yy1)
    inter = w * h
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter/union if union > 0 else 0

# CẬP NHẬT: Thêm tham số drone_altitude và gimbal_pitch vào hàm
def process_video(video_path, gt_path, model_selection, conf_threshold, iou_threshold, target_boxes_list, 
                  drone_altitude=130.0, gimbal_pitch=35.0, progress=gr.Progress()):
    """
    drone_altitude: Độ cao mặc định (mét).
    gimbal_pitch: Góc camera mặc định (độ).
    """
    if video_path is None: return None, "Vui lòng upload video."

    model_path = MODEL_OPTIONS.get(model_selection, "yolov8n.pt")
    
    target_track_ids = set()
    is_selective_mode = target_boxes_list is not None and len(target_boxes_list) > 0

    try:
        model = YOLO(model_path)
    except:
        model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1: fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = "temp_output.mp4"
    final_output_path = "result_video.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    gt_data = load_mot_gt(gt_path)
    has_gt = len(gt_data) > 0
    acc = mm.MOTAccumulator(auto_id=True)
    
    trackers = []
    gmc = GMC(downscale=2)
    
    # CẬP NHẬT: Khởi tạo DistanceEstimator với image_height thay vì object_real_height
    # focal_length=1470 cần được calib lại theo camera drone thực tế
    dist_estimator = DistanceEstimator(focal_length=1470, image_height=height)
    
    KalmanBoxTracker.count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        if frame_idx % 10 == 0:
            progress(frame_idx / total_frames, desc=f"Processing {frame_idx}/{total_frames}")

        # YOLO Detection
        results = model(frame, verbose=False, iou=0.45, conf=0.1, device=DEVICE)[0]
        dets = []
        if results.boxes:
             for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = float(box.conf[0].cpu().numpy())
                dets.append([x1, y1, x2, y2, score])
        dets = np.array(dets) if len(dets) > 0 else np.empty((0, 5))

        # GMC & Prediction
        gmc.apply(frame, trackers)
        trks = np.zeros((len(trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)): to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del): trackers.pop(t)

        # Matching
        if len(dets) > 0:
            inds_high = dets[:, 4] >= conf_threshold
            inds_low = (dets[:, 4] > 0.1) & (dets[:, 4] < conf_threshold)
            dets_high = dets[inds_high]
            dets_low = dets[inds_low]
        else:
            dets_high = np.empty((0, 5)); dets_low = np.empty((0, 5))

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_high, trks, iou_threshold)

        # IOU Match for low score dets
        trks_remain = trks[unmatched_trks]
        dets_remain = dets_low
        if len(trks_remain) > 0 and len(dets_remain) > 0:
            matched_l, _, _ = associate_detections_to_trackers(dets_remain, trks_remain, 0.1)
            for m in matched_l:
                trackers[unmatched_trks[m[1]]].update(dets_remain[m[0]][:4], dets_remain[m[0]][4])
        
        for m in matched:
            trackers[m[1]].update(dets_high[m[0]][:4], dets_high[m[0]][4])
        
        for i in unmatched_dets:
            trackers.append(KalmanBoxTracker(dets_high[i][:4]))

        # Track management & Output collection
        i = len(trackers)
        ret_trackers = []
        for trk in reversed(trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= 3 or frame_idx <= 3):
                ret_trackers.append(np.concatenate((d,[trk.id])).reshape(1,-1))
            i -= 1
            if(trk.time_since_update > 30): trackers.pop(i)

        # Selective ID Logic (Frame 1)
        if frame_idx == 1 and is_selective_mode and len(ret_trackers) > 0:
            for target_box in target_boxes_list:
                best_iou = 0
                best_id = -1
                for trk_data in ret_trackers:
                    d = trk_data[0]
                    trk_box = [d[0], d[1], d[2], d[3]]
                    iou = calculate_iou_single(target_box, trk_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_id = int(d[4])
                if best_iou > 0.5:
                    target_track_ids.add(best_id)
            if len(target_track_ids) == 0:
                is_selective_mode = False

        # Metrics update
        if has_gt:
            t_ids = []; t_boxes = []
            for trk_data in ret_trackers:
                d = trk_data[0]
                t_ids.append(int(d[4]))
                t_boxes.append([d[0], d[1], d[2]-d[0], d[3]-d[1]])
            g_ids = []; g_boxes = []
            if frame_idx in gt_data:
                for item in gt_data[frame_idx]:
                    g_ids.append(int(item[4]))
                    g_boxes.append([item[0], item[1], item[2]-item[0], item[3]-item[1]])
            
            dist = mm.distances.iou_matrix(g_boxes, t_boxes, max_iou=0.5) if (len(g_boxes)>0 and len(t_boxes)>0) else []
            acc.update(g_ids, t_ids, dist)

        # Drawing
        for d in ret_trackers:
            d = d[0]
            x1, y1, x2, y2, tid = int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4])
            
            # CẬP NHẬT: Logic tính khoảng cách mới
            # Truyền bounding box và tham số bay
            g_dist, s_dist = dist_estimator.estimate([x1, y1, x2, y2], drone_altitude, gimbal_pitch)
            
            should_draw = True
            color = (0, 255, 0)
            thickness = 2
            
            if is_selective_mode:
                if tid in target_track_ids:
                    color = (0, 0, 255); thickness = 3
                else:
                    should_draw = False
            
            if should_draw:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.circle(frame, (int((x1+x2)/2), y2), 4, (0, 0, 255), -1) # Vẽ điểm chân

                # CẬP NHẬT: Hiển thị khoảng cách mặt đất (G) và đường chéo (S)
                label = f"ID:{tid} | G:{g_dist}m"
                if is_selective_mode and tid in target_track_ids:
                    label = f"TARGET {tid} | G:{g_dist}m"
                
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)

    cap.release(); out.release()

    # Finalize Metrics
    metrics_str = "Metrics:"
    if has_gt:
        mh = mm.metrics.create()
        try:
            summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'idf1', 'mostly_tracked', 'mostly_lost', 'num_switches'], name='acc')
            metrics_str = mm.io.render_summary(summary, formatters=mh.formatters, namemap={'num_frames': 'Frames', 'mota': 'MOTA', 'motp': 'MOTP', 'idf1': 'IDF1', 'mostly_tracked': 'MT', 'mostly_lost': 'ML', 'num_switches': 'ID Sw'})
        except: metrics_str = "Error calculating metrics"

    if os.path.exists(final_output_path): os.remove(final_output_path)
    try:
        subprocess.call(args=f"ffmpeg -y -i {output_path} -c:v libx264 {final_output_path} -loglevel quiet", shell=True)
    except:
        final_output_path = output_path
        
    return final_output_path, metrics_str