# HỆ THỐNG TRINH SÁT AN NINH THỜI GIAN THỰC (AI SURVEILLANCE SYSTEM)

Hệ thống trinh sát và giám sát an ninh thông minh ứng dụng thị giác máy tính và học sâu (Deep Learning) phát hiện tự động 3 nhóm hành vi bất thường: **Xâm nhập (Intrusion)**, **Lảng vảng (Loitering)**, và **Chạy / Té ngã (Running / Falling)** theo thời gian thực.

---

## 🌟 Tính Năng Nổi Bật

1. **Kiến trúc 4 Tầng & Pipeline Đa Luồng (Multi-threading):**
   - Bộ đệm **Drop-Oldest Circular Buffer** chủ động loại bỏ khung hình cũ khi đầy, duy trì độ trễ camera tiệm cận $0\text{ms}$ (Zero-Latency Streaming), loại bỏ hoàn toàn hiện tượng tràn trễ buffer RTSP.
2. **Theo Dõi Bền Vững Đa Khung Hình (ByteTrack + Grace Period):**
   - Cơ chế vùng đệm gia hạn (Exit Grace Period = 2.0s) chống rung lắc biên Bounding Box và duy trì bộ đếm thời gian lảng vảng ngay cả khi bị che khuất tạm thời (Lost Track).
3. **Phát Hiện Hành Vi Chuẩn Hóa Phối Cảnh & Kiến Trúc 2 Tầng (Two-Tier Hybrid):**
   - **Xâm nhập (Intrusion):** Thuật toán Ray Casting kiểm tra điểm tiếp xúc mặt đất của bàn chân (Foot Point) bên trong vùng cấm đa giác ảo.
   - **Lảng vảng (Loitering):** Tích hợp thời gian lưu trú ($Dwell \ge 10\text{s}$) và bán kính giam cầm không gian ($R \le 60\text{px}$).
   - **Chạy bứt tốc (Running):** Chuẩn hóa vận tốc theo chiều cao cơ thể ($V_{norm} \ge 1.5 \, BH/s$) kết hợp phân tích góc nghiêng thân người $\theta$.
   - **Té ngã (Falling):** Phối hợp **Tier 1 Heuristic** ($AR \le 0.8$, $\theta < 40^\circ$) với **Tier 2 Bi-LSTM** phân loại chuỗi 17 khớp xương $30 \times 34$, giảm $85.7\%$ số lần báo động giả khi cúi/ngồi.
4. **Trung Tâm Chỉ Huy Thông Minh (Command Dashboard):**
   - Giao diện trực quan tích hợp công cụ vẽ vùng cấm đa giác ROI trực tiếp bằng chuột.
   - Cảnh báo đẩy thời gian thực tức thời qua WebSocket, phát âm thanh chuông báo và tự động cắt ảnh snapshot hiện trường lưu trữ SQLite.

---

## 📂 Cấu Trúc Thư Mục Dự Án

```text
btl-httm/
├── docs/                       # Tài liệu học thuật 4 chương hoàn chỉnh
│   ├── 01_tong_quan_bai_toan.md
│   ├── 02_co_so_ly_thuyet.md
│   ├── 03_thiet_ke_he_thong.md
│   └── 04_thuc_nghiem_va_danh_gia.md
├── configs/
│   └── config.yaml             # Cấu hình camera, ngưỡng tham số, cooldown, vùng cấm
├── weights/                    # Thư mục lưu trọng số model (yolov8n-pose.pt, fall_bilstm.pt)
├── data/
│   ├── test_videos/            # Video mẫu dùng để test
│   ├── snapshots/              # Ảnh chụp bằng chứng vi phạm
│   └── surveillance.db         # Database SQLite lưu vết sự kiện
├── scripts/                    # Luồng huấn luyện Offline trên Colab
│   ├── extract_pose_dataset.py # Trích xuất tọa độ 17 khớp xương chuẩn hóa
│   └── train_bilstm.py         # Huấn luyện mạng Bi-LSTM 50 Epochs & vẽ Confusion Matrix
├── src/
│   ├── core/
│   │   ├── capture.py          # CaptureThread với Drop-Oldest Queue
│   │   ├── alert_manager.py    # Điều phối Cooldown & Snapshot không nghẽn FPS
│   │   └── pipeline.py         # Bộ điều phối xử lý chính
│   ├── tracking/
│   │   └── track_manager.py    # Quản lý TrackHistory, Grace Period 2s, dọn dẹp TTL
│   ├── behaviors/
│   │   ├── base.py
│   │   ├── intrusion.py        # Module 1: Xâm nhập vùng cấm
│   │   ├── loitering.py        # Module 2: Lảng vảng
│   │   └── fall_run.py         # Module 3: Chạy & Té ngã
│   ├── models/
│   │   └── bilstm_verifier.py  # Định nghĩa mạng Bi-LSTM PyTorch Tier 2
│   ├── database/
│   │   └── db_service.py       # Quản lý SQLite (bảng ZONES, TRACKS, EVENTS)
│   └── utils/
│       ├── geometry.py         # Công thức hình thái học & chuẩn hóa Root-Relative
│       └── visualizer.py       # Vẽ Bounding Box, 17 khớp xương, đa giác ROI, HUD
├── web/
│   ├── app.py                  # Server FastAPI & WebSocket
│   ├── static/                 # CSS, JS, Audio cảnh báo
│   └── templates/
│       └── index.html          # Web UI Dashboard trung tâm chỉ huy
├── main.py                     # Entry point khởi chạy toàn bộ hệ thống
└── requirements.txt            # Thư viện phụ thuộc
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Hệ Thống

### 1. Yêu cầu môi trường
- Python 3.10 trở lên.
- Đã cài đặt [Git](https://git-scm.com/).

### 2. Cài đặt các thư viện
Mở Terminal / PowerShell tại thư mục dự án và chạy:
```bash
pip install -r requirements.txt
```

### 3. Cấu hình nguồn video
Mở file `configs/config.yaml`:
- Để dùng **Webcam máy tính:** giữ nguyên `source: 0`.
- Để test bằng **file video:** đặt video vào `data/test_videos/video_test.mp4` và đổi cấu hình `source: "data/test_videos/video_test.mp4"`.
- Để dùng **Camera IP / Điện thoại qua DroidCam:** đổi thành link RTSP/HTTP, ví dụ: `source: "http://192.168.1.100:4747/video"`.

### 4. Khởi chạy hệ thống
```bash
python main.py
```
Sau khi khởi chạy, mở trình duyệt web truy cập:
👉 **`http://localhost:8000`**

---

## 🛠️ Luồng Huấn Luyện Offline (Nghiên cứu & Mở rộng)

Nếu muốn tự huấn luyện thêm mô hình mạng nơ-ron Bi-LSTM trên Google Colab:
1. Đặt các video mẫu vào thư mục `datasets/` phân theo 3 nhãn: `0_normal/`, `1_running/`, `2_falling/`.
2. Trích xuất đặc trưng chuỗi khung xương $30 \times 34$:
   ```bash
   python scripts/extract_pose_dataset.py
   ```
3. Huấn luyện mạng Bi-LSTM 50 Epochs:
   ```bash
   python scripts/train_bilstm.py
   ```
   Trọng số tối ưu sẽ tự động được lưu vào `weights/fall_bilstm.pt` và biểu đồ Ma trận nhầm lẫn sẽ được xuất ra `docs/confusion_matrix.png`.
