# Sửa tốc độ chạy và kiểm tra YOLO/ByteTrack — 11/11/2026

## Thay đổi

- Lịch sử tính tốc độ chạy lưu theo thời gian, mặc định 2 giây, tách khỏi bộ đệm pose 30 mẫu. Pipeline tự mở rộng lịch sử nếu cấu hình cửa sổ chạy dài hơn.
- Xác nhận chạy bằng `confirm_seconds: 0.15`, thay cho 3 frame. Khoảng mất dấu trên 0,3 giây phải gom lại cửa sổ tốc độ và xác nhận mới. Timestamp lùi bắt đầu lịch sử chạy mới.
- Giữ nguyên ngưỡng tốc độ 1,5 BH/s và nghiêng thân 15 độ; chưa hiệu chỉnh tiêu chí phân loại hành vi.
- YOLO nhận detection từ confidence 0,1 thay vì 0,4, để ByteTrack có detection yếu cho bước nối track. Ngưỡng ghép bước đầu và tạo ID mới là 0,4; ngưỡng nối detection yếu là 0,1. Như vậy bằng chứng yếu có thể giữ ID đã có nhưng không tự khởi tạo ID mới.
- Khai báo rõ `system.inference_imgsz: 640` và `tracking.tracker_path: configs/bytetrack.yaml`. Giữ `track_buffer: 30`, `match_thresh: 0.8`, `fuse_score: true`.
- Script đánh giá video mẫu đọc đúng các tham số này, cập nhật khóa cache và hỗ trợ `--with-running`.

## Kiểm tra logic

36 test tự động đã qua. 6 test chạy mới kiểm tra: tốc độ và thời gian xác nhận ở 10/15/30/60/120 FPS; đi bộ; khoảng mất dấu; đợt chạy nhanh quá ngắn; timestamp lùi; giới hạn lịch sử theo thời gian.

Trước sửa, lịch sử 30 frame không đủ cửa sổ 0,5 giây ở 60 FPS nên luôn thiếu tốc độ. Sau sửa, chuyển động mô phỏng 3 BH/s đo đúng ở cả 5 mức FPS. Test ngã, đánh nhau, mô hình clip và luồng chia sẻ vẫn qua.

## So sánh đầu vào trên 6 clip

Script: `scripts/compare_motion_tracking.py`. Kết quả: `data/motion_tracking_review/comparison.json`.

Chạy YOLO thực trên video, lấy mẫu mục tiêu 15 FPS (25 FPS gốc lấy mỗi 2 frame thành 12,5 FPS). Cache detection thô ở confidence 0,1 cho từng kích thước ảnh; phát lại cùng dữ liệu qua ByteTrack và detector ngã/chạy cho 6 cấu hình chính. Baseline ở đây là cấu hình YOLO/tracking cũ, dùng cùng mã detector chạy đã sửa để chỉ so sánh ảnh hưởng đầu vào.

Số frame có ít nhất một ID đang được quan sát:

| Clip | Tổng frame | Cấu hình cũ | Cấu hình chọn |
|---|---:|---:|---:|
| Chạy CASIA ngang p01 | 52 | 0 | 0 |
| Chạy CASIA chéo p01 | 61 | 23 | 31 |
| Ngã CASIA ngang p08 | 133 | 1 | 14 |
| UR fall-01 | 80 | 58 | 62 |
| Đi bộ CASIA ngang p02 | 161 | 0 | 0 |
| UR adl-01 | 75 | 69 | 69 |

Clip chạy chéo chuyển từ 3 ID sang 1 ID và từ không cảnh báo sang cảnh báo RUNNING tại 1,76 giây. Đã thử ngưỡng tạo ID 0,15/0,25, tắt fuse_score, và ảnh 960: không cải thiện nhất quán. Cấu hình YOLO 0,1 + ByteTrack 0,25 ban đầu tăng số frame bám được nhiều hơn nhưng gây thêm cảnh báo ngã trên `fight1.mp4`; bị loại sau kiểm tra hồi quy. Cấu hình được chọn khắc phục hồi quy này trong phép phát lại detection. Các phép thử bổ sung lưu tại `data/motion_tracking_review/conservative_start.json`, `no_fusion.json` và `data/motion_tracking_regression/comparison.json`.

Ảnh 960 có chi phí YOLO khoảng 1,8–2,3 lần ảnh 640 trên 5/6 clip; clip đầu có chi phí khởi tạo và chạy đồng thời test nên không dùng để kết luận tốc độ. Đây là thời gian đo CPU tham khảo, gồm lời gọi predict, không phải benchmark FPS vận hành được kiểm soát. Chọn 640 để giữ tốc độ và vì 960 chưa cải thiện ổn định mọi clip.

**Giới hạn:** frame có ID không đồng nghĩa bám đúng người; chưa gán nhãn bbox/ID để đo detection recall hay ID switch. 3/4 clip dương tính khó trong bảng vẫn chưa phát cảnh báo ngã/chạy sau thay đổi. Hai clip bình thường không có cảnh báo; riêng clip đi bộ ở 640 còn không có ID nên kết quả âm tính này không đủ chứng minh khả năng phân biệt đi bộ/chạy. Người nhỏ/mờ vẫn là hạn chế rõ ràng của YOLOv8n-Pose hiện tại. Đây là nhóm clip chẩn đoán có dùng để chọn cấu hình, không phải tập kiểm tra độc lập để công bố độ chính xác.

## Tái chạy

Kiểm tra cuối bằng YOLO + ByteTrack + toàn bộ pipeline, bật cả 3 hành vi: **4/4 kiểm tra qua, 1.027 frame**.

| Video | Kết quả cuối |
|---|---|
| fall_sample.mp4 | Ngã tại 5,088 giây, không cảnh báo chạy/đánh nhau |
| people_walking.mp4 | Không cảnh báo trong 596 frame |
| fight1.mp4 | Đánh nhau tại 0,48 và 1,6 giây; không cảnh báo ngã/chạy |
| running__casia_angleview_p01_run_a1.mp4 | Chạy tại 1,76 giây; không cảnh báo ngã/đánh nhau |

Kiểm tra theo loại sự kiện xuất hiện trong clip, chưa chấm độ trễ theo mốc hành vi được gán nhãn. Clip đánh nhau phát hai cảnh báo với ID khác nhau; cooldown hiện theo ID nên chưa gộp thành một sự kiện cấp clip.

```powershell
.\.venv-test\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-test\Scripts\python.exe -m scripts.compare_motion_tracking
.\.venv-test\Scripts\python.exe -m scripts.evaluate_behaviors_local fall_sample.mp4 people_walking.mp4 fight1.mp4 public/sphar/running__casia_angleview_p01_run_a1.mp4 --output data/behavior_evaluation_motion_final --with-running --check
```

So sánh tracker và kiểm tra pipeline dùng nơi nhận cảnh báo thử, không ghi sự kiện vào SQLite vận hành. Các video đã vẽ khung của lần kiểm tra pipeline cuối nằm trong `data/behavior_evaluation_motion_final/`. Thư mục `data/behavior_evaluation_motion_update/` giữ kết quả cấu hình thử đã bị loại, không phải kết quả cuối.
