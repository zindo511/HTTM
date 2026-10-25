"""
LUỒNG TRÍCH XUẤT DỮ LIỆU OFFLINE (OFFLINE SKELETON EXTRACTION)
-------------------------------------------------------------
Kịch bản chạy trên Google Colab hoặc máy cá nhân:
1. Quét qua các video trong thư mục dữ liệu mẫu (ví dụ: normal/, running/, falling/).
2. Dùng YOLOv8n-Pose trích xuất 17 khớp xương người theo từng khung hình.
3. Áp dụng chuẩn hóa Root-Relative theo trung điểm khớp hông.
4. Cắt thành các cửa sổ trượt T = 30 frames và lưu thành X_dataset.npy, y_dataset.npy.
"""

import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from src.utils.geometry import normalize_skeleton_root_relative


def extract_features_from_video(video_path: str, model: YOLO, window_size: int = 30, step_size: int = 10):
    cap = cv2.VideoCapture(video_path)
    frames_skeletons = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, verbose=False)[0]
        # Nếu có phát hiện ít nhất 1 người
        if results.boxes is not None and len(results.boxes) > 0 and results.keypoints is not None:
            # Lấy đối tượng có diện tích lớn nhất (chủ thể chính)
            boxes = results.boxes.xyxy.cpu().numpy()
            kpts = results.keypoints.xy.cpu().numpy()
            
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            max_idx = np.argmax(areas)
            
            best_bbox = boxes[max_idx]
            best_kpts = kpts[max_idx]
            
            norm_vec = normalize_skeleton_root_relative(best_kpts, best_bbox)
            frames_skeletons.append(norm_vec)
        else:
            # Nếu frame không thấy người: giữ vector 0
            frames_skeletons.append(np.zeros(34, dtype=np.float32))
            
    cap.release()
    
    # Cắt thành các cửa sổ trượt (Sliding Windows) kích thước 30x34
    samples = []
    total_frames = len(frames_skeletons)
    for i in range(0, total_frames - window_size + 1, step_size):
        window = frames_skeletons[i : i + window_size]
        samples.append(np.array(window, dtype=np.float32))
        
    return samples


def main():
    print("[INFO] Bắt đầu quá trình trích xuất đặc trưng khung xương từ video...")
    model = YOLO("yolov8n-pose.pt")
    
    # Cấu trúc thư mục dữ liệu mẫu ví dụ:
    # datasets/
    #   ├── 0_normal/
    #   ├── 1_running/
    #   └── 2_falling/
    data_dir = "datasets"
    classes = {"0_normal": 0, "1_running": 1, "2_falling": 2}
    
    X_data = []
    y_data = []
    
    for cls_name, cls_idx in classes.items():
        cls_folder = os.path.join(data_dir, cls_name)
        if not os.path.exists(cls_folder):
            print(f"[NOTE] Thư mục {cls_folder} chưa có dữ liệu. Vui lòng đặt video vào đây.")
            continue
            
        video_files = glob.glob(os.path.join(cls_folder, "*.mp4")) + glob.glob(os.path.join(cls_folder, "*.avi"))
        print(f"[INFO] Đang xử lý lớp {cls_name} ({len(video_files)} video)...")
        
        for v_path in video_files:
            samples = extract_features_from_video(v_path, model)
            for s in samples:
                X_data.append(s)
                y_data.append(cls_idx)
                
    if len(X_data) > 0:
        X_arr = np.array(X_data, dtype=np.float32)
        y_arr = np.array(y_data, dtype=np.int64)
        
        os.makedirs("data", exist_ok=True)
        np.save("data/X_dataset.npy", X_arr)
        np.save("data/y_dataset.npy", y_arr)
        print(f"[SUCCESS] Trích xuất hoàn tất! Tổng cộng {len(X_arr)} mẫu (Kích thước: {X_arr.shape}). Đã lưu vào data/")
    else:
        print("[WARNING] Chưa có dữ liệu video mẫu để trích xuất.")


if __name__ == "__main__":
    main()
