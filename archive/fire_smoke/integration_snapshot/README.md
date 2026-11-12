# HỆ THỐNG GIÁM SÁT HÀNH VI BẤT THƯỜNG THỜI GIAN THỰC

Hệ thống giám sát camera dùng thị giác máy tính để tự động phát hiện các hành vi và sự kiện bất thường theo thời gian thực: **Chạy nhanh (Running)**, **Té ngã (Falling)**, **Đánh nhau (Fighting)** và **Khói & Lửa / Châm lửa (Smoke & Fire Detection)**.

---

## Trạng thái hiện tại

| Thành phần | Trạng thái |
| :--- | :--- |
| Thu hình đa luồng, YOLOv8n-Pose + ByteTrack, dashboard web, SQLite | Hoạt động |
| Té ngã — FSM hình học theo thời gian (`docs/fall_fsm.md`) | Có cảnh báo trên clip demo, xem kết quả theo clip trong báo cáo đánh giá |
| Chạy nhanh — vận tốc chuẩn hoá theo chiều cao cơ thể | Có cảnh báo trên clip CASIA góc chéo trong menu demo |
| Đánh nhau — luật theo cặp người + mô hình clip (`weights/fight_clip.pt`) | Hoạt động trên clip demo; mất dấu cặp người khi vật lộn gộp thành một khung |
| Khói & Lửa sớm / Châm lửa (`weights/fire_smoke_yolov8n.pt` + kiểm định vật lý) | Hoạt động (`mAP@50 = 86,8%`), lọc bóng đèn lóa trắng và liên kết khớp cổ tay |
| Bi-LSTM xác minh té ngã (Tier 2) | **Tắt** — `weights/fall_bilstm.pt` chưa được train bằng dữ liệu thật |
| Số liệu trong `docs/04_thuc_nghiem_va_danh_gia.md` | **Chưa được kiểm chứng**, sẽ thay bằng số đo thật |

Xem [kết quả đánh giá trên 98 clip](docs/public_evaluation_review.md) và [kịch bản demo trên web](docs/demo.md).

---

## Cách hoạt động

1. **Thu hình (`src/core/capture.py`)**: webcam và RTSP dùng hàng đợi giữ khung hình mới nhất. File demo phát một lần theo nhịp lấy mẫu và chờ pipeline nhận đủ khung đã chọn.
2. **Phát hiện & bám vết (`src/core/pipeline.py`)**: YOLOv8n-Pose cho khung người và 17 khớp xương, ByteTrack gán ID; `src/tracking/track_manager.py` lưu lịch sử từng người.
3. **Phân tích hành vi (`src/behaviors/`)**:
   - **Chạy nhanh** (`fall_run.py`): vận tốc tâm khung trên cửa sổ 0,5 giây, chia cho chiều cao cơ thể (BH/s), kèm độ nghiêng thân. Điều kiện chạy phải duy trì 0,15 giây.
   - **Té ngã** (`fall_fsm.py`): `NORMAL → IMPACT_PENDING → LYING_VERIFY → FALL_LATCHED` — hạ người nhanh, sau đó giữ tư thế thấp/nằm ngang đủ 0,5 giây; giữ cảnh báo đến khi đứng dậy.
   - **Đánh nhau** (`fighting.py`): với mỗi cặp người đứng gần nhau, xét ra đòn tay/chân nhanh vào vùng đối phương, giằng co sát người, hoặc lao vào nhau; kết hợp mô hình phân loại clip `fight_clip.pt`.
   - **Khói & Lửa / Châm lửa** (`fire_smoke.py`): phát hiện vùng lửa/khói bằng `fire_smoke_yolov8n.pt`, lọc bóng đèn pha lóa trắng bằng độ ấm quầng sáng (`Red - Blue`) và độ biến thiên mức xám, đồng thời liên kết với khớp cổ tay/mặt để bắt hành vi châm lửa trên tay người.
4. **Cảnh báo (`src/core/alert_manager.py`)**: lưu ảnh bằng chứng, ghi SQLite, đẩy qua WebSocket lên dashboard (`web/`).

---

## Cấu trúc thư mục

```text
btl-httm/
├── configs/config.yaml         # Nguồn video, ngưỡng từng hành vi
├── weights/                    # yolov8n-pose.pt, fall_bilstm.pt (không commit)
├── data/
│   ├── test_videos/            # Video thử (không commit)
│   ├── snapshots/              # Ảnh bằng chứng sinh ra khi chạy (không commit)
│   └── surveillance.db         # SQLite sinh ra khi chạy (không commit)
├── src/
│   ├── core/                   # capture, pipeline, alert_manager, shared_stream
│   ├── tracking/               # TrackHistory, TrackManager
│   ├── behaviors/              # fall_run, fall_fsm, fighting
│   ├── models/                 # Bi-LSTM (Tier 2, đang tắt)
│   ├── database/               # SQLite
│   └── utils/                  # hình học, vẽ khung
├── web/                        # FastAPI + WebSocket + dashboard
├── scripts/                    # Trích xuất khớp xương, train Bi-LSTM, đánh giá video
├── tests/                      # Kiểm thử tự động
├── docs/                       # Báo cáo và ghi chú kỹ thuật
└── main.py                     # Điểm khởi chạy
```

---

## Cài đặt và chạy

Yêu cầu Python 3.10 trở lên (đã thử với 3.12).

```bash
pip install -r requirements.txt
```

Phiên bản thư viện đã thử nằm trong `requirements-tested.txt`.

Chọn nguồn video trong `configs/config.yaml`, mục `camera.source`:
- Webcam: `source: 0`
- File: `source: "data/test_videos/ten_video.mp4"`
- Camera IP / DroidCam: `source: "http://192.168.1.100:4747/video"`

`system.device` để `"cuda:0"` nếu có GPU; hệ thống tự chuyển sang CPU khi không có CUDA.

```bash
python main.py
```

Sau đó mở `http://localhost:8000`. Có thể đổi nguồn hoặc tải video lên ngay trên dashboard.

Để trình bày, bấm **Đi bộ**, **Chạy nhanh**, **Té ngã** và **Đánh nhau** trong hàng **Video mẫu** dưới khung hình. Bấm lại cùng nút để xem từ đầu. Với `camera.file_sample_fps: 15`, mỗi file phát một lần và giữ đủ các khung hình đã chọn, theo cùng nhịp lấy mẫu với bài đánh giá. Xem [kịch bản demo](docs/demo.md) để biết mốc cảnh báo và video dự phòng.

---

## Kiểm thử

Kiểm thử tự động (logic FSM ngã, luật đánh nhau, luồng stream dùng chung):

```bash
python -m unittest discover -s tests -v
```

Chạy lại các video mẫu qua toàn bộ pipeline và so với kết quả mong đợi:

```bash
python -m scripts.evaluate_behaviors_local --check
```

Lưu ý: script này hiện tắt hành vi chạy nhanh và chỉ kiểm tra "có ít nhất một cảnh báo đúng loại" trên 5 clip. Đây chưa phải phép đo precision/recall.

---

## Huấn luyện Bi-LSTM (chưa hoàn thành)

1. Đặt video vào `datasets/0_normal/`, `datasets/1_running/`, `datasets/2_falling/`.
2. `python scripts/extract_pose_dataset.py` — tạo `data/X_dataset.npy`, `data/y_dataset.npy`.
3. `python scripts/train_bilstm.py` — lưu `weights/fall_bilstm.pt`, `docs/training_curves.png`, `docs/confusion_matrix.png`.

Nếu thiếu file dữ liệu, script train tự sinh dữ liệu ngẫu nhiên để thử luồng code. Các biểu đồ hiện có trong `docs/` là kết quả của lần chạy như vậy, không phản ánh mô hình thật. Chỉ bật `enable_bilstm_tier2` sau khi train bằng dữ liệu thật.
