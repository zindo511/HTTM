# Tự kiểm tra đánh nhau và té ngã — 29/10/2026

Môi trường đã cài: `.venv-test` (Python 3.12, CPU). `.venv` cũ trỏ tới một
Python không còn dùng được. Phiên bản thư viện của môi trường mới được lưu trong
`requirements-tested.txt`; `pip check` không phát hiện xung đột.

## Chạy ứng dụng và kiểm tra lại

```powershell
.\.venv-test\Scripts\python.exe main.py
.\.venv-test\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-test\Scripts\python.exe -m scripts.evaluate_behaviors_local --check
```

Bài kiểm tra video dùng YOLO-Pose + ByteTrack và toàn bộ pipeline xử lý/render.
Chỉ tắt chạy/lảng vảng và bỏ ROI để kiểm tra riêng hai hành vi. Cảnh báo thử được
thu vào báo cáo riêng, không ghi vào SQLite hoặc snapshot vận hành.
Timestamp lấy theo vị trí trong video; tốc độ lấy mẫu khoảng 12–16 FPS.

| Video | Frame xử lý | Cảnh báo đánh nhau | Cảnh báo ngã |
|---|---:|---|---|
| fight1.mp4 | 234 | 1 lần, 0,48 giây | Không |
| fall_sample.mp4 | 136 | Không | 1 lần, 5,088 giây |
| 01_video.mp4 | 720 | Không | Không |
| 02_video.mp4 | 606 | Không | Không |
| people_walking.mp4 | 596 | Không | Không |

Tổng cộng 2.292 frame. Kết quả, thời điểm chuyển trạng thái và video đã vẽ khung
nằm trong `data/behavior_evaluation/`. Hai clip đánh nhau/ngã còn được chạy lại
YOLO từ đầu bằng môi trường mới, với cùng kết quả, tại
`data/behavior_evaluation_fresh/`. Đã xem ảnh kết quả: khung ngã duy trì khi nằm,
banner đánh nhau vừa chiều rộng video dọc.

## Các lỗi đã sửa trong lần kiểm tra

- Xác nhận đánh nhau theo 0,15 giây thay vì phụ thuộc hoàn toàn vào số frame.
- Giữ vùng cảnh báo 1,5 giây kể cả khi bám vết tạm mất một/hai người.
- Lọc hai người đứng có chiều cao ảnh chênh quá lớn để giảm nhầm tiền cảnh/hậu
  cảnh; không áp điều kiện này khi một người nằm/vật lộn.
- Giữ ưu tiên trạng thái ngã khi đồng thời có dấu hiệu đánh nhau.
- Chạy xử lý đánh nhau cả ở frame không có người để cập nhật thời gian giữ và
  dọn trạng thái; sửa banner bị cắt ở video dọc.

23 test tự động đã qua, gồm: ngã ở 10/15/30/60 FPS và nhiều tỉ lệ ảnh; cúi chậm;
vấp không nằm xuống; phục hồi sau ngã; mất pose; BiLSTM chấp nhận/từ chối;
ra đòn từ pose thật giả lập; chuyển động đồng đều; đứng gần; ở xa; chênh tiền
cảnh/hậu cảnh; cooldown; mất dấu; xác nhận theo thời gian; hai người xem dùng
cùng kết quả AI; JPEG thực và worker tiếp tục sau một frame lỗi.

Đây là kết quả trên bộ video mẫu đã xem, chưa phải phép đo độ chính xác trên
dataset được gán nhãn. Che khuất lâu hoặc YOLO gộp hai người thành một vẫn có
thể làm bỏ sót đoạn đánh nhau.

Pose được cache để chạy lại cùng dữ liệu và so sánh thay đổi detector. Muốn
chạy YOLO từ đầu, chọn thư mục đầu ra mới:

```powershell
.\.venv-test\Scripts\python.exe -m scripts.evaluate_behaviors_local fight1.mp4 fall_sample.mp4 --output data/behavior_evaluation_new --check
```
