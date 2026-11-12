# Demo phát hiện hành vi

Chạy từ thư mục dự án bằng PowerShell.

```powershell
.\.venv-test\Scripts\python.exe main.py
```

Mở [dashboard](http://localhost:8000). Bấm các nút trong hàng **Video mẫu** ngay dưới khung video. Bấm lại cùng nút để bắt đầu clip từ đầu.

Sau khi sửa `configs/config.yaml` hoặc code, tắt `main.py` bằng `Ctrl+C` rồi chạy lại. Tiến trình đang chạy vẫn giữ cấu hình cũ.

## Chuẩn bị trước buổi demo

- **Tốc độ phát:** trên CPU không có GPU, mỗi khung hình mất khoảng 100 ms cho YOLO-Pose và khoảng 20 ms cho mô hình đánh nhau. Hệ thống phát file ở nhịp 10 khung/giây (`file_sample_fps: 10`) và không bỏ khung, nên video chạy khoảng 0,8–0,9x tốc độ thật.
- **Không hạ nhịp dưới 10.** Ở 8 khung/giây, người chạy nhanh dịch chuyển quá xa giữa hai khung nên ByteTrack mất dấu và clip **Chạy nhanh** không báo. Các ngưỡng té ngã trong `configs/config.yaml` đã được chỉnh và kiểm tra ở nhịp 8–10.
- **Video nặng:** dashboard tự thu nhỏ khung về cạnh dài 1280 px (`max_frame_side`). Video 4K vẫn tốn thời gian giải mã, nên dùng bản 720p khi có, ví dụ `fight2_720p.mp4` thay cho `fight2.mp4`. Tạo bản nhẹ bằng lệnh dưới đây.

```powershell
ffmpeg -i data/test_videos/fight3.mp4 -vf "scale=-2:1280,fps=30" -c:v libx264 -crf 23 -an data/test_videos/fight3_720p.mp4
```

- Đóng các chương trình nặng khác. Nếu máy có GPU NVIDIA, cài PyTorch bản CUDA; hệ thống dùng `cuda:0` khi có và không còn bị chậm.
- Bộ đếm và cột cảnh báo lấy cả lịch sử cũ. Để bắt đầu sạch, dừng `main.py` và đổi tên `data/surveillance.db` trước buổi demo.

| Thứ tự | Nguồn | Nội dung trình bày |
|---:|---|---|
| 1 | Đi bộ | Chiếu 5–10 giây để thấy hệ thống vẽ khung người và bộ đếm giữ nguyên. |
| 2 | Chạy nhanh | Hộp người đổi màu và thẻ **Chạy nhanh** xuất hiện khoảng 1,76 giây theo thời gian video. |
| 3 | Té ngã | Hệ thống báo **Té ngã** khoảng 5,0 giây. Khung cảnh báo giữ khi người còn nằm. |
| 4 | Đánh nhau | Cảnh báo đầu tiên khoảng 0,48 giây đến từ luật theo cặp người. Chiếu đến khoảng 12,16 giây để thấy thẻ **Mô hình nhận diện xô xát**. |

Các mốc trên đo ở nhịp 10 khung/giây với cấu hình hiện tại. Video `fight2_720p.mp4`, `fight3.mp4` và `danh_nhau_tiktok.mp4` cũng báo **Đánh nhau** (khoảng 4,4 giây, 7,9 giây và 6,2 giây). Trong cảnh xô xát, người bị quật ngã hoặc gập người nằm ngang sát camera có thể kèm thẻ **Té ngã**.

### Giới hạn đã biết

- **Té ngã:** trên 98 clip công khai ở nhịp 10 khung/giây, hệ thống bắt 9/19 clip ngã và không báo nhầm clip nào. Các clip sót chủ yếu là CCTV xa, người nhỏ hoặc ngã chậm kéo dài nhiều giây.
- **Đánh nhau:** bắt 8/30 clip công khai, không báo nhầm. Clip CCTV xa, người nhỏ và bị che vẫn khó.

Dashboard xử lý video bằng YOLO-Pose và ByteTrack ngay khi phát. Khi máy xử lý chậm, clip có thể chạy chậm hơn thời lượng gốc để giữ đủ khung hình đã lấy mẫu. Kết thúc clip, màn hình giữ khung cuối để có thời gian giải thích. Cột cảnh báo lấy cả lịch sử đã lưu, nên quan sát thẻ mới ở đầu danh sách hoặc mức tăng của bộ đếm khi đổi clip.

## Diễn té ngã trực tiếp

Ngã ra trước hay ra sau đều được. Điều quan trọng là hướng ngã so với camera.

- **Tốt nhất là ngã ngang khung hình.** Camera nhìn từ bên hông, người đi ngang qua rồi ngã. Khung người đổi rõ từ đứng sang nằm.
- **Ngã thẳng về phía camera hoặc ra xa camera** vẫn bắt được nhờ nhánh ngã chậm, nhưng góc thân đo kém ổn định hơn. Chỉ dùng làm cảnh phụ.

Các bước:

1. **Đặt camera cố định.** Thấy trọn người từ đầu tới chân trong suốt cú ngã, kể cả lúc nằm. Người cao ít nhất khoảng 1/3 khung hình. Không đứng sau bàn ghế, đủ sáng, chỉ một người trong khung.
2. **Đứng hoặc đi lại ít nhất 2 giây trước khi ngã.** Hệ thống cần khoảng 1,5 giây đứng thẳng để đo chiều cao tham chiếu. Trên overlay, chờ `Href=warming` chuyển thành `H/Href=1.00`.
3. **Ngã dứt khoát,** xuống sàn trong khoảng dưới 1 giây. Không ngồi xuống rồi mới nằm: ngồi rồi nằm (giường, sofa) được thiết kế để không báo.
4. **Nằm yên ít nhất 2 giây.** Hệ thống cần thấy tư thế nằm liên tục khoảng 0,5 giây trong 1,2 giây sau cú rơi. Bật dậy ngay sẽ không có cảnh báo.
5. Overlay đổi `Fall=IMPACT_PENDING` → `LYING_VERIFY` → `FALL_LATCHED`, khung người chuyển đỏ và thẻ **Té ngã** xuất hiện.
6. **Ngã lần hai:** cùng một ID phải chờ 15 giây (`cooldown_seconds`). Đi ra khỏi khung rồi quay lại để có ID mới.

Trải nệm hoặc thảm dày chỗ ngã. Nệm không ảnh hưởng tới nhận diện.

Video dự phòng trong `data/test_videos/hf_examples/` (bấm **Tải video lên** hoặc chọn trong **Nguồn video**):

| File | Nội dung | Kết quả mong đợi |
|---|---|---|
| `hf1_fall_forward.mp4` | Đi tới rồi ngã sấp | Báo **Té ngã** |
| `hf4_fall_sideways.mp4` | Đi tới rồi ngã nghiêng | Báo **Té ngã** |
| `hf2_fall_backward.mp4` (giống `fall_s.mp4`) | Ngã ngửa chậm | Báo **Té ngã** |
| `hf1_nearfall_forward_recover.mp4` | Suýt ngã rồi đứng dậy | **Không** báo |
| `hf3_nearfall_recover_sit_sofa.mp4` | Suýt ngã rồi ngồi sofa | **Không** báo |

## Video dự phòng đã vẽ khung

Mở [clip chạy](../data/behavior_evaluation_motion_final/running__casia_angleview_p01_run_a1-annotated.mp4), [clip ngã](../data/behavior_evaluation_motion_final/fall_sample-annotated.mp4) hoặc [clip ẩu đả](../data/behavior_evaluation_motion_final/fight1-annotated.mp4) nếu cần video dự phòng đã vẽ khung.

Cấu hình demo dùng mô hình `weights/fight_clip.pt` với ngưỡng 0,45. Mô hình chỉ phát cảnh báo mới khi có ít nhất hai người được bám trong suốt thời gian xác nhận, hoặc khi một khung người trùm lên đúng chỗ một cặp người vừa đứng sát nhau trong 3 giây trước (`merge_memory_seconds`): hai người ôm vật nhau thường bị YOLO gộp thành một khung. [Báo cáo 98 clip](public_evaluation_review.md) ghi kết quả của cấu hình trước khi bật mô hình này.

Để dùng điện thoại, cho máy tính và điện thoại vào cùng Wi-Fi. Mở ứng dụng camera IP trên điện thoại, bật phát video và sao chép địa chỉ luồng MJPEG hoặc RTSP mà ứng dụng cung cấp. Trên dashboard máy tính, mở **Camera điện thoại**, dán địa chỉ luồng và bấm **Kết nối**. Dùng URL luồng video, ví dụ `http://192.168.1.50:8080/video`, theo đúng địa chỉ và cổng trên điện thoại.
