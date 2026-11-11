# HỆ THỐNG GIÁM SÁT HÀNH VI BẤT THƯỜNG THỜI GIAN THỰC

Hệ thống giám sát camera dùng thị giác máy tính để tự động phát hiện 3 hành vi bất thường theo thời gian thực: **Chạy nhanh (Running)**, **Té ngã (Falling)** và **Đánh nhau (Fighting)**.

---

## Trạng thái hiện tại

| Thành phần | Trạng thái |
| :--- | :--- |
| Thu hình đa luồng, YOLOv8n-Pose + ByteTrack, dashboard web, SQLite | Hoạt động |
| Té ngã — FSM hình học theo thời gian (`docs/fall_fsm.md`) | Hoạt động, mới thử trên 1 clip ngã + 3 clip bình thường |
| Chạy nhanh — vận tốc chuẩn hoá theo chiều cao cơ thể | Hoạt động, **chưa có video chạy để đánh giá** |
| Đánh nhau — luật theo cặp người (khoảng cách, vận tốc tay/chân) | **Yếu**: mất dấu khi hai người vật lộn bị gộp thành một khung |
| Bi-LSTM xác minh té ngã (Tier 2) | **Tắt** — `weights/fall_bilstm.pt` chưa được train bằng dữ liệu thật |
| Số liệu trong `docs/04_thuc_nghiem_va_danh_gia.md` | **Chưa được kiểm chứng**, sẽ thay bằng số đo thật |

Kết quả tự kiểm tra gần nhất: `docs/behavior_self_test.md`.

---

## Cách hoạt động

1. **Thu hình (`src/core/capture.py`)**: luồng riêng đọc webcam / file / RTSP, đẩy vào hàng đợi chỉ giữ khung hình mới nhất để không tích luỹ độ trễ.
2. **Phát hiện & bám vết (`src/core/pipeline.py`)**: YOLOv8n-Pose cho khung người và 17 khớp xương, ByteTrack gán ID; `src/tracking/track_manager.py` lưu lịch sử từng người.
3. **Phân tích hành vi (`src/behaviors/`)**:
   - **Chạy nhanh** (`fall_run.py`): vận tốc tâm khung trên cửa sổ 0,5 giây, chia cho chiều cao cơ thể (BH/s), kèm độ nghiêng thân; cần vượt ngưỡng nhiều lần liên tiếp.
   - **Té ngã** (`fall_fsm.py`): `NORMAL → IMPACT_PENDING → LYING_VERIFY → FALL_LATCHED` — hạ người nhanh, sau đó giữ tư thế thấp/nằm ngang đủ 0,5 giây; giữ cảnh báo đến khi đứng dậy.
   - **Đánh nhau** (`fighting.py`): với mỗi cặp người đứng gần nhau, xét ra đòn tay/chân nhanh vào vùng đối phương, giằng co sát người, hoặc lao vào nhau.
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
