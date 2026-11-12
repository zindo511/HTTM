# Đánh giá lại 98 clip — 11/11/2026

Lần chạy hiện tại dùng YOLOv8n-Pose + ByteTrack trên CPU với nhịp lấy mẫu mục tiêu 15 FPS. Hai lần đánh giá dùng cùng 98 tên video. Bộ đánh nhau hiện dùng luật theo cặp người. Kết quả gốc nằm trong [báo cáo cũ](../data/public_evaluation/results.json) và [báo cáo hiện tại](../data/public_evaluation_current/results.json).

| Hành vi | Phát hiện cũ | Phát hiện hiện tại | Báo khác nhãn cũ | Báo khác nhãn hiện tại |
|---|---:|---:|---:|---:|
| Chạy nhanh | 0/14 | 1/14 | 1/84 | 2/84 |
| Té ngã | 6/19 | 7/19 | 0/79 | 1/79 |
| Đánh nhau | 2/30 | 2/30 | 0/68 | 0/68 |

Mỗi clip có nhãn chính lấy từ tên file. Clip được tính là phát hiện nếu có ít nhất một cảnh báo cùng loại, còn báo khác nhãn được đếm theo số clip có cảnh báo khác loại. Số frame có ID tính bất kỳ ID đang được quan sát nào, nên cần xem lại video và đường bám để xác minh người thực hiện hành vi.

## Kết quả chẩn đoán

- **Chạy nhanh.** 9/13 clip bỏ sót không có ID, gồm cả 5 clip ngang và 4 clip nhìn từ trên xuống. Thêm 2 clip có ID nhưng thiếu mẫu tốc độ hợp lệ. Hai clip còn lại cần xem ngưỡng tốc độ và độ nghiêng. Vì vậy giữ ngưỡng hiện tại và ưu tiên khôi phục đầu vào bám người trên hai góc quay đang thiếu.
- **Ngã.** 5/12 clip bỏ sót đã tới pha xác minh tư thế thấp, 1 clip chỉ tới pha nghi ngã. 4 clip có chiều cao tham chiếu nhưng chưa vào pha nghi ngã, 1 clip thiếu chiều cao tham chiếu và 1 clip không có ID. Nên xem lại 5 clip đã vào pha xác minh để phân biệt đứt ID với tư thế thấp bị gián đoạn, rồi dùng clip ngồi xuống và đi bộ để kiểm tra báo nhầm nếu chỉnh điều kiện.

### Chạy nhanh

| Clip bỏ sót | Frame có ID / frame xử lý | Khâu quan sát được |
|---|---:|---|
| `sphar/running__casia_angleview_p02_run_a2.mp4` | 21/61 | Không có mẫu tốc độ hợp lệ |
| `sphar/running__casia_angleview_p04_run_a2.mp4` | 7/50 | Không có mẫu tốc độ hợp lệ |
| `sphar/running__casia_angleview_p06_run_a1.mp4` | 48/86 | Tốc độ dưới ngưỡng |
| `sphar/running__casia_angleview_p08_run_a1.mp4` | 52/71 | Vượt ngưỡng tốc độ thấp nhưng thiếu độ nghiêng và chưa đạt ngưỡng tốc độ cao |
| `sphar/running__casia_horizontalview_p01_run_a2.mp4` | 0/52 | Không có ID |
| `sphar/running__casia_horizontalview_p03_run_a2.mp4` | 0/68 | Không có ID |
| `sphar/running__casia_horizontalview_p05_run_a1.mp4` | 0/72 | Không có ID |
| `sphar/running__casia_horizontalview_p07_run_a1.mp4` | 0/59 | Không có ID |
| `sphar/running__casia_horizontalview_p08_run_a2.mp4` | 0/66 | Không có ID |
| `sphar/running__casia_topdownview_p02_run_a2.mp4` | 0/71 | Không có ID |
| `sphar/running__casia_topdownview_p04_run_a1.mp4` | 0/119 | Không có ID |
| `sphar/running__casia_topdownview_p06_run_a1.mp4` | 0/146 | Không có ID |
| `sphar/running__casia_topdownview_p07_run_a2.mp4` | 0/107 | Không có ID |

### Té ngã

| Clip bỏ sót | Frame có ID / frame xử lý | Khâu quan sát được |
|---|---:|---|
| `sphar/falling__casia_angleview_p01_faint_a1.mp4` | 98/171 | Có chiều cao tham chiếu, chưa vào pha nghi ngã |
| `sphar/falling__casia_angleview_p18_faint_a1.mp4` | 121/343 | Có chiều cao tham chiếu, chưa vào pha nghi ngã |
| `sphar/falling__casia_horizontalview_p08_faint_a1.mp4` | 14/133 | Có chiều cao tham chiếu, chưa vào pha nghi ngã |
| `sphar/falling__casia_horizontalview_p22_faint_a1.mp4` | 0/234 | Không có ID |
| `sphar/falling__caviar_Rest_FallOnFloor.mp4` | 136/508 | Chưa có chiều cao đứng tham chiếu |
| `sphar/falling__caviar_Rest_SlumpOnFloor.mp4` | 221/460 | Đã vào pha xác minh, chưa chốt cảnh báo |
| `sphar/falling__caviar_Rest_WiggleOnFloor.mp4` | 144/651 | Đã nghi ngã, chưa vào pha xác minh tư thế thấp |
| `ur_fall/fall-01.mp4` | 62/80 | Đã vào pha xác minh, chưa chốt cảnh báo |
| `ur_fall/fall-03.mp4` | 105/108 | Có chiều cao tham chiếu, chưa vào pha nghi ngã |
| `ur_fall/fall-05.mp4` | 71/76 | Đã vào pha xác minh, chưa chốt cảnh báo |
| `ur_fall/fall-07.mp4` | 71/78 | Đã vào pha xác minh, chưa chốt cảnh báo |
| `ur_fall/fall-09.mp4` | 75/93 | Đã vào pha xác minh, chưa chốt cảnh báo |

## Cảnh báo khác nhãn

| Clip | Loại cảnh báo | Mốc trong video |
|---|---|---:|
| `sphar/falling__caviar_Rest_SlumpOnFloor.mp4` | Chạy nhanh | 36,56 giây |
| `sphar/falling__caviar_Rest_WiggleOnFloor.mp4` | Chạy nhanh | 4,40 giây |
| `sphar/walking__caviar_Browse1.mp4` | Té ngã | 29,44 giây |

## Kiểm tra demo

Bốn nguồn demo được thử qua API đổi nguồn và pipeline thật, gồm 1.027 frame đã lấy mẫu. Clip chạy gửi một cảnh báo WebSocket, clip ngã gửi một cảnh báo, clip ẩu đả gửi hai cảnh báo và clip đi bộ không gửi cảnh báo. Đã kiểm tra JPEG, ảnh sự kiện và lịch sử trong cơ sở dữ liệu thử. Cả 37 test tự động đều qua. Xem [kịch bản demo](demo.md) và [kết quả kiểm tra web](../data/demo_smoke/results.json).

```powershell
.\.venv-test\Scripts\python.exe -m scripts.evaluate_public_clips --output data/public_evaluation_current
```

