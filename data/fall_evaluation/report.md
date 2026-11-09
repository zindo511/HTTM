# Báo cáo thử video – 29/10/2026

Chạy YOLOv8n-Pose + ByteTrack + FSM hiện tại, CPU, confidence 0.4, timestamp theo FPS gốc, mọi frame. Không chạy Bi-LSTM; không ghi vào cơ sở dữ liệu cảnh báo. Chỉ đánh giá detector ngã, chưa kiểm tra giao diện live.

| Video | Thời lượng | Frame | Cảnh báo |
|---|---:|---:|---|
| fall_sample.mp4 | 11.34 s | 272 | Track 1 tại 5.046 s |
| 01_video.mp4 | 44.97 s | 1439 | 0 |
| 02_video.mp4 | 37.84 s | 1211 | 0 |
| people_walking.mp4 | 49.67 s | 596 | 0 |

Tổng 3518 frame, khoảng 143.82 giây. Ảnh tiếp xúc lấy tại 10/30/50/70/90% mỗi clip và ảnh thời điểm báo động nằm cùng thư mục results.json. Quan sát ảnh mẫu của fall_sample xác nhận có người ngã và nằm dưới sàn; các ảnh mẫu 01/02 là người đi lại. Đây chỉ là rà soát ảnh lấy mẫu, chưa gán nhãn toàn bộ video theo từng sự kiện; không công bố precision/recall hoặc độ trễ từ chạm đất.

Đã tìm nguồn công khai https://github.com/koesan/Mediapipe-Fall-Detector nhưng tải qua GitHub API gặp lỗi TLS Authentication failed. Do đó các kết quả trên dùng video sẵn có trong dự án; chưa xác minh nguồn gốc của các video này và chưa kiểm thử video mới tải trên mạng.

Chưa kiểm chứng: ngã dọc trục ở nhiều góc, ngồi xổm nhanh, chủ động nằm nhanh, mất keypoint thật, khoảng cách 2m/6m và so sánh với bản cũ. Kết quả bước đầu tích cực nhưng mẫu ngã quá ít để kết luận độ tin cậy thực tế.

Tái chạy trong môi trường Python có thư viện dự án: python -m scripts.evaluate_falls_local. Script hiện chứa đường dẫn thư viện runtime riêng của máy này.
