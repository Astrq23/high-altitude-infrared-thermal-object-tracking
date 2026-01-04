from pyexpat import model
import gradio as gr
import cv2
from ultralytics import YOLO
from src.processor import process_video, MODEL_OPTIONS
import torch

if torch.cuda.is_available():
    DEVICE = 0
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
def detect_objects_frame_1(video_path, model_name):
    if video_path is None: return [], [], []
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret: return [], [], []

    model_path = MODEL_OPTIONS.get(model_name, "yolov8n.pt")
    try: model = YOLO(model_path)
    except: model = YOLO("yolov8n.pt")
    results = model(frame, verbose=False, iou=0.45, conf=0.1, device=DEVICE)[0]
    gallery_images = []
    detected_boxes = [] 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if results.boxes:
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            crop = frame_rgb[y1:y2, x1:x2]
            gallery_images.append((crop, f"Object {i}"))
            detected_boxes.append([x1, y1, x2, y2])
            
    return gallery_images, detected_boxes, []

def on_select_object(evt: gr.SelectData, detected_boxes, current_selection_indices):
    if current_selection_indices is None: current_selection_indices = []
    index = evt.index
    if index in current_selection_indices:
        current_selection_indices.remove(index)
    else:
        current_selection_indices.append(index)
    current_selection_indices.sort()
    selected_boxes = [detected_boxes[i] for i in current_selection_indices if i < len(detected_boxes)]
    feedback_str = f"Đang chọn các Object: {current_selection_indices}" if current_selection_indices else "Chưa chọn đối tượng nào (Sẽ track tất cả)"
    return feedback_str, current_selection_indices, selected_boxes

def clear_selection():
    return "Đã xóa chọn. Tracking tất cả.", [], []

def create_ui():
    with gr.Blocks(title="UAV Tracking System") as demo:
        gr.Markdown("# UAV Multi-Target Tracking System")
        gr.Markdown("Upload video -> Quét -> Click chọn nhiều đối tượng (Click lần nữa để bỏ chọn) -> Start.")

        detected_boxes_state = gr.State([])
        selected_indices_state = gr.State([])
        selected_boxes_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                input_video = gr.Video(label="1. Upload Video")
                model_dd = gr.Dropdown(choices=list(MODEL_OPTIONS.keys()), value="EfficientNetB0", label="Model")
                
                with gr.Row():
                    btn_scan = gr.Button("Quét đối tượng", variant="secondary")
                    btn_clear = gr.Button("Xóa chọn", variant="stop")

                gr.Markdown("### 3. Gallery (Click để Chọn/Bỏ chọn):")
                gallery = gr.Gallery(
                    label="Danh sách đối tượng", show_label=True, elem_id="gallery",
                    columns=4, rows=2, height="auto", object_fit="contain", allow_preview=False
                )
                selection_info = gr.Textbox(label="Trạng thái chọn", value="Chưa chọn (Track All)", interactive=False)
                
                with gr.Row():
                    conf_slide = gr.Slider(0.1, 0.9, 0.5, label="Conf Threshold")
                    iou_slide = gr.Slider(0.1, 0.9, 0.2, label="IoU Threshold")
                
                input_gt = gr.File(label="Upload GT (.txt)")
                btn_run = gr.Button("START TRACKING", variant="primary")

            with gr.Column(scale=1):
                output_video = gr.Video(label="Kết quả Tracking")
                output_metrics = gr.Textbox(label="Metrics Report", lines=10)

        btn_scan.click(detect_objects_frame_1, inputs=[input_video, model_dd], outputs=[gallery, detected_boxes_state, selected_indices_state])
        gallery.select(on_select_object, inputs=[detected_boxes_state, selected_indices_state], outputs=[selection_info, selected_indices_state, selected_boxes_state])
        btn_clear.click(clear_selection, outputs=[selection_info, selected_indices_state, selected_boxes_state])
        btn_run.click(process_video, inputs=[input_video, input_gt, model_dd, conf_slide, iou_slide, selected_boxes_state], outputs=[output_video, output_metrics])
    
    return demo