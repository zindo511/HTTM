# Lưu trữ: phát hiện Khói & Lửa

Phần này đã được tách khỏi hệ thống ngày 12/11/2026. Đề tài giám sát **hành vi bất thường của người**: chạy nhanh, té ngã, đánh nhau. Khói và lửa là vật thể, không phải hành vi, nên không thuộc phạm vi đề tài.

## Lý do tách

- Ngoài phạm vi đề tài. Các tài liệu báo cáo `docs/01`–`05` không nhắc tới khói/lửa.
- Mô hình `fire_smoke_yolov8n.pt` báo nhầm ở 15/22 video không có lửa khi kiểm tra ngày 12/11/2026. Nó nhận ba lô, quần jean, áo, giày đỏ và đèn thả trần ánh vàng thành khói/lửa, với độ tin cậy 45–71%. Nâng ngưỡng không sửa được; cần train lại với thêm ảnh âm tính.
- Mỗi khung hình tốn thêm khoảng 85 ms trên CPU, làm video demo chạy chậm còn khoảng 0,5x.

## Nội dung

| Đường dẫn | Vị trí cũ |
|---|---|
| `src/behaviors/fire_smoke.py` | `src/behaviors/fire_smoke.py` — bộ phát hiện `FireSmokeDetector` |
| `tests/test_fire_smoke.py` | `tests/test_fire_smoke.py` — kèm test nút bật/tắt trên pipeline |
| `train_fire_smoke_colab.ipynb` | Thư mục gốc — notebook train trên Colab |
| `weights/fire_smoke_yolov8n.pt` | `weights/` — trọng số YOLOv8n 2 lớp (`0: fire`, `1: smoke`) |
| `data/test_videos/khoi.mp4`, `fire_smoke_sample.mp4` | `data/test_videos/` — video kiểm tra |
| `integration_snapshot/` | Bản đầy đủ của các file còn lại trong hệ thống tại thời điểm tách, khi còn tích hợp khói/lửa |

## Gắn lại

So sánh từng file trong `integration_snapshot/` với file hiện tại để lấy lại phần tích hợp:

- `src/core/pipeline.py`: khởi tạo `FireSmokeDetector`, `set_fire_smoke_enabled()`, gọi `process_all()` sau phần đánh nhau và phát cảnh báo `FIRE_SMOKE`.
- `src/utils/visualizer.py`: `draw_fire_smoke_overlay()` và nhãn `FIRE_ALERT`.
- `src/core/alert_manager.py`: thời gian chờ `FIRE_SMOKE`.
- `web/app.py`, `web/templates/index.html`, `web/static/js/dashboard.js`, `web/static/css/style.css`: nút bật/tắt, video mẫu, bộ đếm và giọng đọc cảnh báo.
- `configs/config.yaml`: mục `behaviors.fire_smoke`.

Sau đó chép các file trong thư mục này về vị trí cũ.
