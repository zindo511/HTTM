# HỆ THỐNG GIÁM SÁT HÀNH VI BẤT THƯỜNG THỜI GIAN THỰC

Hệ thống giám sát camera dùng thị giác máy tính để tự động phát hiện các hành vi bất thường của con người theo thời gian thực: **Chạy nhanh (Running)**, **Té ngã (Falling)** và **Đánh nhau (Fighting)**.

---

## Trạng thái hiện tại

| Thành phần | Trạng thái |
| :--- | :--- |
| Thu hình đa luồng, YOLOv8n-Pose + ByteTrack, dashboard web, SQLite | Hoạt động |
| Té ngã — FSM hình học theo thời gian (`docs/fall_fsm.md`) | Có cảnh báo trên clip demo, xem kết quả theo clip trong báo cáo đánh giá |
| Chạy bất thường — tốc độ + dáng chạy + ngữ cảnh, máy trạng thái 4 bước | Có kiểm thử tự động (cả chạy ngang và chạy thẳng về phía camera); chưa đo lại trên bộ 98 clip |
| Đánh nhau — luật theo cặp người + mô hình clip (`weights/fight_clip.pt`) | Hoạt động trên clip demo; mất dấu cặp người khi vật lộn gộp thành một khung |
| Bi-LSTM xác minh té ngã (Tier 2) | **Tắt** — `weights/fall_bilstm.pt` chưa được train bằng dữ liệu thật |
| Số liệu trong `docs/04_thuc_nghiem_va_danh_gia.md` | **Chưa được kiểm chứng**, sẽ thay bằng số đo thật |

Xem [kết quả đánh giá trên 98 clip](docs/public_evaluation_review.md) và [kịch bản demo trên web](docs/demo.md).

Phần phát hiện Khói & Lửa đã được tách khỏi hệ thống vì không phải hành vi của người; mã nguồn, notebook train và trọng số được lưu trong `archive/fire_smoke/`.

---

## Cách hoạt động

1. **Thu hình (`src/core/capture.py`)**: webcam và RTSP dùng hàng đợi giữ khung hình mới nhất. File demo phát một lần theo nhịp lấy mẫu và chờ pipeline nhận đủ khung đã chọn.
2. **Phát hiện & bám vết (`src/core/pipeline.py`)**: YOLOv8n-Pose cho khung người và 17 khớp xương, ByteTrack gán ID; `src/tracking/track_manager.py` lưu lịch sử từng người.
3. **Phân tích hành vi (`src/behaviors/`)**:
   - **Chạy bất thường** (`running.py`, gọi từ `fall_run.py`): xem [mục bên dưới](#phát-hiện-chạy-bất-thường).
   - **Té ngã** (`fall_fsm.py`): `NORMAL → IMPACT_PENDING → LYING_VERIFY → FALL_LATCHED` — hạ người nhanh, sau đó giữ tư thế thấp/nằm ngang đủ 0,5 giây; giữ cảnh báo đến khi đứng dậy.
   - **Đánh nhau** (`fighting.py`): với mỗi cặp người đứng gần nhau, xét ra đòn tay/chân nhanh vào vùng đối phương, giằng co sát người, hoặc lao vào nhau; kết hợp mô hình phân loại clip `fight_clip.pt`.
4. **Cảnh báo (`src/core/alert_manager.py`)**: lưu ảnh bằng chứng, ghi SQLite, đẩy qua WebSocket lên dashboard (`web/`).

Cả 3 hành vi được xét song song trên mỗi khung hình, không cần chọn trước loại hành vi. Té ngã được ưu tiên: khi một người đang có dấu hiệu ngã thì phần chạy của người đó tạm dừng.

### Phát hiện chạy bất thường

Kết hợp 3 tiêu chí (tham số trong `configs/config.yaml`, mục `behaviors.running`):

| Tiêu chí | Cách đo | Bắt buộc |
| :--- | :--- | :--- |
| Tốc độ | Quãng đường điểm chân / (chiều cao người × Δt), đơn vị BH/s, trên cửa sổ 0,5 giây. Ngưỡng `v_norm_threshold: 1.3` (≈ 2,3 m/s; đi bộ trên các video mẫu đo được ≤ 1,21) | Có |
| Dáng chạy | Thân nghiêng, vung chân, vung tay. Nhìn chính diện: gối nâng hạ luân phiên, khuỷu tay gập. Thiếu pose thì cần tốc độ ≥ 1,5 × ngưỡng | Nên có |
| Ngữ cảnh | Điểm chân nằm trong `zones` và thời điểm nằm trong `active_hours`. Để trống = toàn khung hình, mọi lúc | Có |

**Chạy thẳng về phía camera:** tâm khung gần như đứng yên nhưng khung to dần. Khoảng cách tới camera tính theo chiều cao người là `D = f / h` (f: tiêu cự pixel, h: chiều cao khung), tốc độ tiến/lùi là `|dD/dt|`, ước lượng bằng hồi quy trên cửa sổ 1 giây (dài hơn một nhịp bước chân, vì khung co lại khi nhấc chân). Khi tốc độ chủ yếu đến từ chiều sâu thì **bắt buộc** có dáng chạy, vì đi bộ về phía camera cũng làm khung to dần. Nên đặt `focal_length_px` sau khi calibration; để `null` thì tạm lấy `focal_length_ratio` (0,6) × chiều rộng khung, ứng với camera giám sát góc rộng ~80°; ống kính hẹp thì tăng tỷ lệ này. Khung chạm mép ảnh (người bị cắt) thì bỏ qua tốc độ chiều sâu. Nếu đã có ma trận `homography` (ảnh → mặt sàn, mét) thì hệ thống đo bằng m/s, ngưỡng `speed_mps_threshold`.

**Máy trạng thái** theo từng người:

- `NORMAL → SUSPECTED`: tốc độ vượt ngưỡng trong vùng/khung giờ giám sát.
- `SUSPECTED → CONFIRMED`: dấu hiệu chạy duy trì `confirm_seconds`; phát **một** cảnh báo. Mất tín hiệu ngắn hơn `drop_grace_seconds` (bị che khuất) vẫn giữ trạng thái.
- `CONFIRMED → COOLDOWN`: người dừng chạy. Chạy liên tục lâu vẫn chỉ tính là một sự kiện.
- `COOLDOWN → NORMAL`: sau `cooldown_seconds` kể từ lần cảnh báo.

Rung camera hoặc khung giật qua lại (dịch chuyển thực nhỏ so với quãng đường tích lũy, `min_path_efficiency`) không được tính là chạy.

Ví dụ chỉ cảnh báo ở nửa dưới khung hình, ban đêm:

```yaml
zones: [{name: "hanh lang", normalized: true, points: [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]}]
active_hours: ["22:00-06:00"]
```

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
│   ├── behaviors/              # running, fall_run, fall_fsm, fighting
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
- Camera IP / DroidCam: `source: "http://192.168.1.100:4747/video"`. Nếu hình bị nằm ngang (điện thoại dựng đứng), đặt `camera.rotate: 90` hoặc `270`. Hình xoay sai làm người đứng trông như đang nằm, YOLO dễ vẽ 2 khung cho 1 người.

`system.device` để `"cuda:0"` nếu có GPU; hệ thống tự chuyển sang CPU khi không có CUDA.

```bash
python main.py
```

Sau đó mở `http://localhost:8000`. Có thể đổi nguồn hoặc tải video lên ngay trên dashboard.

Để trình bày, bấm **Đi bộ**, **Chạy nhanh**, **Té ngã** và **Đánh nhau** trong hàng **Video mẫu** dưới khung hình. Bấm lại cùng nút để xem từ đầu. Với `camera.file_sample_fps: 15`, mỗi file phát một lần và giữ đủ các khung hình đã chọn, theo cùng nhịp lấy mẫu với bài đánh giá. Xem [kịch bản demo](docs/demo.md) để biết mốc cảnh báo và video dự phòng.

---

## Kiểm thử

Kiểm thử tự động (logic FSM ngã, chạy bất thường, luật đánh nhau, luồng stream dùng chung):

```bash
python -m unittest discover -s tests -v
```

Chỉ kiểm thử phần chạy:

```bash
python -m unittest tests.test_running tests.test_running_fsm -v
```

### Thử bằng camera

Đặt `camera.source` (webcam `0` hoặc DroidCam), chạy `python main.py`, mở `http://localhost:8000`. Với `debug_overlay: true`, cạnh mỗi người hiện `V=…BH/s` (thêm chữ `sau` khi tốc độ chủ yếu do tiến/lùi so với camera), `Pose=x/3 cues`, `Ctx=…` và `Run:<trạng thái>`.

Đặt camera cố định cao 1,5–2 m, đứng cách 6–10 m và để thấy toàn thân. Các tình huống nên thử:

| Tình huống | Kết quả mong đợi |
| :--- | :--- |
| Đi bộ ngang / đi nhanh | Không cảnh báo |
| Chạy ngang camera | 1 cảnh báo cho mỗi lượt chạy |
| Đi bộ về phía camera | Không cảnh báo |
| Chạy về phía camera | 1 cảnh báo (cần thấy rõ tay chân) |
| Chạy ngoài vùng / ngoài giờ cấu hình | Không cảnh báo |
| Ngã, đánh nhau | Đúng loại, không báo thành chạy |

Chạy thật mà không báo thì hạ `v_norm_threshold` theo số `V` trên màn hình; đi bộ bị báo nhầm thì tăng ngưỡng này.

Để xem số liệu từng frame của một video (ID, độ tin cậy, tốc độ, pose, trạng thái) mà không cần mở web:

```bash
python scripts/debug_running.py data/test_videos/run_ngang.mp4 --cache run_ngang_yolo.json
```

`--cache` lưu kết quả YOLO để lần chạy sau gần như tức thì; `--focal` thử tiêu cự khác.

Chạy lại các video mẫu qua toàn bộ pipeline và so với kết quả mong đợi:

```bash
python -m scripts.evaluate_behaviors_local --check
```

Lưu ý: script này mặc định tắt hành vi chạy (thêm `--with-running` để bật) và chỉ kiểm tra "có ít nhất một cảnh báo đúng loại" trên 5 clip. Đây chưa phải phép đo precision/recall.

---

## Huấn luyện Bi-LSTM (chưa hoàn thành)

1. Đặt video vào `datasets/0_normal/`, `datasets/1_running/`, `datasets/2_falling/`.
2. `python scripts/extract_pose_dataset.py` — tạo `data/X_dataset.npy`, `data/y_dataset.npy`.
3. `python scripts/train_bilstm.py` — lưu `weights/fall_bilstm.pt`, `docs/training_curves.png`, `docs/confusion_matrix.png`.

Nếu thiếu file dữ liệu, script train tự sinh dữ liệu ngẫu nhiên để thử luồng code. Các biểu đồ hiện có trong `docs/` là kết quả của lần chạy như vậy, không phản ánh mô hình thật. Chỉ bật `enable_bilstm_tier2` sau khi train bằng dữ liệu thật.
