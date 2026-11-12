# HIỆN TRẠNG, KẾ HOẠCH CÔNG VIỆC VÀ QUY TẮC LÀM VIỆC NHÓM

Cập nhật lần cuối: 11/11/2026. Phạm vi dự án: phát hiện ba hành vi **Chạy nhanh**, **Té ngã**, **Đánh nhau**.

---

## 1. Hiện trạng

### 1.1. Đã hoàn thành

- Khung hệ thống: thu hình đa luồng, YOLOv8n-Pose + ByteTrack, quản lý lịch sử từng người, cảnh báo bất đồng bộ, SQLite, dashboard web (video trực tiếp, cột cảnh báo, đổi nguồn, tải video lên).
- Té ngã: máy trạng thái theo thời gian (`src/behaviors/fall_fsm.py`), có kiểm thử tự động.
- Chạy nhanh: vận tốc chuẩn hoá theo chiều cao cơ thể, tính theo thời gian thực (`src/behaviors/fall_run.py`).
- Đánh nhau: luật theo cặp người (`src/behaviors/fighting.py`), có kiểm thử tự động.
- 9 vấn đề kỹ thuật của đợt phân công trước (kẹt trạng thái, `dt` gán cứng, thiếu điều kiện đổ người và vận tốc rơi, double softmax, biểu đồ huấn luyện, lệch toạ độ, lịch sử cảnh báo khi tải lại trang, bộ đếm cảnh báo) đã được xử lý.
- Đợt dọn dẹp 11/2026: bỏ hành vi xâm nhập và lảng vảng cùng công cụ vẽ vùng; sửa nạp trọng số Bi-LSTM, chọn thiết bị, tải video lên, lưu mô tả cảnh báo; gỡ dữ liệu sinh ra khi chạy khỏi git.

### 1.2. Chưa hoàn thành

| Hạng mục | Tình trạng |
| :--- | :--- |
| Bộ video có nhãn | Chưa có. Mỗi hành vi ngã/đánh nhau có 1 clip, chạy nhanh chưa có clip nào |
| Công cụ đánh giá theo sự kiện | Chưa có. Script hiện tại chỉ kiểm tra "có cảnh báo đúng loại" và tắt hành vi chạy |
| Bi-LSTM xác minh té ngã | Chưa huấn luyện bằng dữ liệu thật, đang tắt |
| Đánh nhau | Yếu khi hai người bị gộp thành một khung; chưa có tập âm tính |
| Chạy nhanh | Chưa kiểm chứng trên video chạy thật; chưa có kiểm thử tự động |
| Số liệu báo cáo | Chương 4 mới có kết quả sơ bộ, chưa có Precision/Recall |

Chi tiết: `docs/04_thuc_nghiem_va_danh_gia.md`.

---

## 2. Kế hoạch công việc

Các giai đoạn theo thứ tự ưu tiên. Giai đoạn 2 là điều kiện cho các giai đoạn sau: không có dữ liệu có nhãn thì không đo được cải tiến nào.

### Giai đoạn 2 — Dữ liệu và công cụ đánh giá

| # | Việc | Sản phẩm | Hoàn thành khi |
| :---: | :--- | :--- | :--- |
| 2.1 | Thu thập video: mỗi hành vi 15–20 clip dương tính + tập âm tính khó (bảng ở mục 4.4.2 Chương 4) | Video trong `data/test_videos/`, kèm ghi chú nguồn gốc | Đủ số lượng cho cả 3 hành vi |
| 2.2 | Gán nhãn thời điểm bắt đầu/kết thúc từng sự kiện | `data/labels.csv` | Mọi clip dương tính đều có nhãn |
| 2.3 | Viết script đánh giá theo sự kiện, bật cả 3 hành vi | `scripts/evaluate_events.py`, xuất bảng TP/FP/FN, Precision/Recall/F1, độ trễ | Chạy một lệnh ra bảng kết quả cho toàn bộ tập |
| 2.4 | Đo kết quả gốc của hệ thống hiện tại | Bảng số liệu trong Chương 4 | Có số liệu cho cả 3 hành vi |

### Giai đoạn 3 — Cải thiện phát hiện

| # | Việc | Ghi chú |
| :---: | :--- | :--- |
| 3.1 | Chỉnh ngưỡng chạy nhanh theo số liệu; thêm kiểm thử tự động cho chạy | Cân nhắc thêm tín hiệu tần suất bước chân |
| 3.2 | Huấn luyện Bi-LSTM xác minh té ngã bằng dữ liệu thật | Lấy mẫu theo thời gian thay vì theo khung hình; chia tập theo video; xử lý trường hợp thiếu 30 bộ khớp (mục 3.3.3 Chương 3) |
| 3.3 | Nâng cấp đánh nhau | Giữ luật hiện tại để khoanh vùng nghi vấn; thêm mô hình phân loại clip ngắn trên vùng ảnh gộp, huấn luyện trên RWF-2000 / Hockey Fight |
| 3.4 | Đo lại sau mỗi thay đổi bằng script của Giai đoạn 2 | So với kết quả gốc ở 2.4 |

### Giai đoạn 4 — Hoàn thiện báo cáo

- Điền số liệu thật vào Chương 4: bảng theo sự kiện, so sánh có/không Tier 2, so sánh phương pháp đánh nhau, tốc độ xử lý.
- Thay `docs/training_curves.png` và `docs/confusion_matrix.png` bằng kết quả huấn luyện thật.
- Rà soát Chương 1–3 cho khớp với mã nguồn cuối cùng.

### Giai đoạn 5 — Hoàn thiện demo

- Nút "Đã xử lý" cho từng cảnh báo (cột `resolution_status` đã có sẵn trong CSDL).
- Xoá trạng thái bám vết khi file video phát lại từ đầu.
- Kịch bản demo cho từng hành vi và video dự phòng.

Phân công từng việc cho thành viên: nhóm tự thống nhất và ghi vào cột dưới đây.

| Việc | Phụ trách | Hạn |
| :--- | :--- | :--- |
| 2.1 – 2.2 Thu thập và gán nhãn video | | |
| 2.3 – 2.4 Script đánh giá và đo kết quả gốc | | |
| 3.1 Chạy nhanh | | |
| 3.2 Bi-LSTM té ngã | | |
| 3.3 Đánh nhau | | |
| 4 Báo cáo | | |
| 5 Demo | | |

---

## 3. Quy tắc làm việc nhóm

### 3.1. Phân vùng mã nguồn

Để hạn chế xung đột khi ghép mã, mỗi việc nên chỉ chạm vào một nhóm file:

| Mảng | Thư mục / file |
| :--- | :--- |
| Thuật toán hành vi, bám vết, mô hình | `src/behaviors/`, `src/tracking/`, `src/models/`, `scripts/`, `tests/` |
| Web, cơ sở dữ liệu | `web/`, `src/database/` |
| Điều phối, cấu hình, tài liệu | `src/core/`, `configs/config.yaml`, `main.py`, `docs/` |

Thay đổi ngưỡng trong `configs/config.yaml` cần báo cho cả nhóm, vì nó ảnh hưởng tới kết quả đánh giá của mọi người.

### 3.2. Giao ước dữ liệu dùng chung

1. **Mã loại sự kiện (`event_type`):** `"RUNNING"`, `"FALLING"`, `"FIGHTING"`.
2. **Gói tin WebSocket `/ws/alerts`** và phần tử của `/api/events`:
   ```json
   {
     "event_type": "RUNNING | FALLING | FIGHTING",
     "track_id": 3,
     "zone_id": "N/A",
     "confidence": 90.0,
     "timestamp": "2026-11-11 10:30:00",
     "snapshot_url": "/snapshots/FALLING_3_1793400000.jpg",
     "extra_info": "Mô tả sự kiện"
   }
   ```
3. **Mô hình Bi-LSTM:** `predict(sequence)` nhận mảng NumPy `(1, 30, 34)` hoặc `(30, 34)`, trả về bộ ba xác suất `(bình thường, chạy, ngã)` có tổng bằng 1.
4. **Bộ phát hiện hành vi:** phát cảnh báo qua `alert_manager.trigger_alert(event_type, track_id, frame, confidence, extra_info)`, và tự kiểm tra cooldown bằng `track.can_trigger_alert(...)`.

### 3.3. Quy trình Git

1. Lấy mã mới nhất và tạo nhánh riêng cho từng việc:
   ```powershell
   git checkout main
   git pull origin main
   git checkout -b feat/ten-viec
   ```
2. Đặt tên commit theo dạng `feat: ...`, `fix: ...`, `docs: ...`, `refactor: ...`.
3. Đẩy nhánh lên và tạo Pull Request; không đẩy trực tiếp vào `main`.
4. Không đưa lên git: trọng số mô hình, video, ảnh bằng chứng, file CSDL (đã khai báo trong `.gitignore`).

### 3.4. Tiêu chí hoàn thành một việc

1. Kiểm thử tự động đạt:
   ```powershell
   python -m unittest discover -s tests -v
   ```
2. Các video mẫu vẫn cho kết quả mong đợi:
   ```powershell
   python -m scripts.evaluate_behaviors_local --check
   ```
3. `python main.py` chạy không lỗi trên terminal và trên console của trình duyệt.
4. Với thay đổi thuật toán: có số liệu trước/sau trên tập có nhãn (khi Giai đoạn 2 đã xong).
5. Tài liệu trong `docs/` được cập nhật nếu hành vi của hệ thống thay đổi.
