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

## Bổ sung 11/11/2026 — chạy nhanh & đánh nhau

Kiểm tra lại bằng pose đã cache (không cần video/YOLO/torch), **bật cả detector chạy**:

```bash
python -m scripts.replay_pose_cache --check      # 5/5 video khớp kỳ vọng
python -m unittest discover -s tests -v          # 38 test
```

Lỗi đã sửa:

- **Chạy không bao giờ báo ở >= 60 FPS:** lịch sử bbox chỉ giữ 30 mẫu (< 0,5 s
  cửa sổ đo). Tăng `bbox_history`/`timestamp_history` lên 90 mẫu.
- **Báo chạy nhầm khi ByteTrack đổi ID / mất dấu:** bbox nhảy xa trong 1 frame bị
  tính là vận tốc lớn. Bỏ phép đo khi tốc độ từng bước > `max_step_speed_bh` hoặc
  có khoảng mất dấu > `max_gap_seconds`.
- **Báo ẩu đả nhầm khi 2 người chạy bộ cạnh nhau hoặc chạy ngược chiều lướt qua
  nhau** (tay vung khi chạy ~4 BH/s vượt ngưỡng ra đòn). Bỏ qua cặp khi cả hai
  cùng di chuyển >= `pass_by_speed_bh`. Lao vào đánh người đứng yên vẫn báo.

Test mới: `tests/test_running.py` (đứng yên có rung bbox, đi bộ, chạy có/không
nghiêng người, 10–60 FPS, nhiều kích thước người, đổi ID, che khuất, tắt
detector) và `tests/test_fight_running.py` (chạy cạnh nhau, chạy lướt qua nhau,
đấm tại chỗ, lao vào rồi đánh, ưu tiên trạng thái ẩu đả hơn chạy).

Giới hạn còn lại: cần >= 0,15 s tín hiệu ra đòn **liên tục** mới xác nhận, nên
cú đấm lẻ có khoảng nghỉ dài có thể bị bỏ sót. Nới điều kiện này làm `01_video`
(đông người) báo nhầm, nên giữ nguyên. Chưa có video thật có người chạy để đo
độ nhạy của detector chạy — nên quay thêm 1 clip chạy cho `data/test_videos/`.
