# Demo phát hiện hành vi

Chạy từ thư mục dự án bằng PowerShell.

```powershell
.\.venv-test\Scripts\python.exe main.py
```

Mở [dashboard](http://localhost:8000). Bấm các nút trong hàng **Video mẫu** ngay dưới khung video. Bấm lại cùng nút để bắt đầu clip từ đầu.

Sau khi sửa `configs/config.yaml` hoặc code, tắt `main.py` bằng `Ctrl+C` rồi chạy lại. Tiến trình đang chạy vẫn giữ cấu hình cũ.

## Chuẩn bị trước buổi demo

- **Khói & Lửa mặc định tắt.** Hàng **Mô hình** dưới các nút video mẫu có nút **🔥 Khói & Lửa: TẮT/BẬT**. Bấm để bật ngay khi cần, không phải khởi động lại. Lần bật đầu mất khoảng 1–2 giây để nạp mô hình. Nút đỏ nghĩa là đang bật.
- **Thứ tự nên dùng:** demo đi bộ, chạy, té ngã, đánh nhau khi mô hình Khói & Lửa còn tắt. Đến phần khói lửa mới bật, xong thì tắt lại.
- **Vì sao phải tắt:** trên CPU không có GPU, mỗi khung hình mất khoảng 100 ms cho YOLO-Pose, khoảng 85 ms cho Khói & Lửa và khoảng 20 ms cho mô hình đánh nhau. Hệ thống phát file ở nhịp 10 khung/giây (`file_sample_fps: 10`) và không bỏ khung. Bật đủ ba mô hình, video chạy khoảng 0,5x tốc độ thật. Tắt Khói & Lửa, video chạy khoảng 0,8–0,9x.
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
| 5 | Khói & Lửa | **Bật nút Khói & Lửa trước.** Giai đoạn đầu không báo bóng đèn pha góc trái; khoảng 28,7 giây báo **Đám khói (`SMOKE`)** bốc lên từ thùng rác; khoảng 31,7 giây chuyển sang **Ngọn lửa (`FIRE`)** khi thùng rác bùng cháy. |

Các mốc trên đo ở nhịp 10 khung/giây với cấu hình hiện tại. Video `fight2_720p.mp4`, `fight3.mp4` và `danh_nhau_tiktok.mp4` cũng báo **Đánh nhau** (khoảng 4,4 giây, 7,9 giây và 6,2 giây). Trong cảnh xô xát, người bị quật ngã hoặc gập người nằm ngang sát camera có thể kèm thẻ **Té ngã**.

### Giới hạn đã biết

- **Khói & Lửa:** mô hình hiện nhận nhầm ba lô, quần jean, áo, giày đỏ và đèn vàng thành khói/lửa (độ tin cậy 45–71%) ở phần lớn video không có lửa. Chỉ bật khi chiếu `khoi.mp4` hoặc diễn bật lửa, rồi tắt ngay. Cần train lại với thêm ảnh âm tính (người mặc quần áo nhiều màu, đèn vàng, phòng trong nhà).
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

Cấu hình demo dùng mô hình `weights/fight_clip.pt` với ngưỡng 0,45. Mô hình chỉ phát cảnh báo mới khi có ít nhất hai người được bám trong suốt thời gian xác nhận. Khi hai người bị gộp thành một ID trước khi xác nhận, cảnh báo có thể đến muộn hoặc bị bỏ sót. [Báo cáo 98 clip](public_evaluation_review.md) ghi kết quả của cấu hình trước khi bật mô hình này.

Để dùng điện thoại, cho máy tính và điện thoại vào cùng Wi-Fi. Mở ứng dụng camera IP trên điện thoại, bật phát video và sao chép địa chỉ luồng MJPEG hoặc RTSP mà ứng dụng cung cấp. Trên dashboard máy tính, mở **Camera điện thoại**, dán địa chỉ luồng và bấm **Kết nối**. Dùng URL luồng video, ví dụ `http://192.168.1.50:8080/video`, theo đúng địa chỉ và cổng trên điện thoại.

---

## Chi tiết kịch bản Demo Khói & Lửa (`FIRE_SMOKE`)

Hệ thống sử dụng mô hình `weights/fire_smoke_yolov8n.pt` (2 lớp `0: fire`, `1: smoke`) kết hợp tầng kiểm định vật lý màu sắc quầng sáng (`Red - Blue`) và liên kết khớp xương bàn tay/mặt từ `YOLOv8n-Pose`.

Trước phần này, bấm **🔥 Khói & Lửa: TẮT** trong hàng **Mô hình** để chuyển sang **BẬT**. Xong phần khói lửa thì bấm lại để tắt.

Mô hình hiện tại có thể nhận nhầm **đèn thả trần ánh vàng/cam** là lửa ở độ tin cậy khoảng 36%, sát ngưỡng `conf_threshold: 0.35`. Tránh để đèn vàng trong khung hình khi diễn trực tiếp.

### 1. Trình bày trên Video mẫu CCTV (`data/test_videos/khoi.mp4`)

Bấm nút **Khói & Lửa** trong hàng **Video mẫu** và thuyết minh theo 3 mốc thời gian:

| Mốc video | Hiện tượng trên màn hình | Nội dung thuyết trình |
| :---: | :--- | :--- |
| **0s – 25s** | Người đàn ông ngồi cạnh thùng rác rồi rời đi. Bóng đèn pha ở góc trái chiếu sáng rất gắt nhưng **không bị khoanh cảnh báo**. | Tầng kiểm định vật lý (`_verify_and_classify_region` trong `src/behaviors/fire_smoke.py`) kiểm tra độ ấm quầng sáng (`Red - Blue`) và tỷ lệ tâm lóa trắng nên loại bỏ hoàn toàn nguồn sáng trắng cố định của bóng đèn pha. |
| **26s – 31s** | Khói bắt đầu bốc lên từ thùng rác $\rightarrow$ Xuất hiện khung cam **`SMOKE`** và thẻ cảnh báo **Đám khói**. | Ngay khi khói xám mới bốc lên từ thùng rác mà chưa có ngọn lửa, hệ thống đã phát hiện sớm vùng **`SMOKE`** và gửi cảnh báo tức thời lên Dashboard. |
| **Từ 31,4s** | Thùng rác bùng lửa $\rightarrow$ Chuyển sang khung đỏ **`FIRE (75%–94%)`** và phát cảnh báo **Ngọn lửa**. | Khi đám cháy bùng phát, hệ thống tự động nâng cấp cảnh báo từ Khói sang **Ngọn lửa**, lưu ảnh chụp bằng chứng và phát âm thanh báo động khẩn cấp. |

### 2. Trình bày trực tiếp qua Webcam / Camera điện thoại (Hành vi Châm lửa)

Chuyển nguồn sang **Webcam máy tính** (hoặc **Camera điện thoại**) và thực hiện 2 bước:

1. **Bước 1 — Kiểm tra chống báo giả với đèn LED (5 giây):**
   - Bật đèn Flash trắng của điện thoại chiếu về phía camera.
   - Hệ thống **không báo động**, chứng minh trực tiếp khả năng phân biệt ánh đèn trắng với ngọn lửa thật.
2. **Bước 2 — Diễn hành vi "Châm lửa trái phép" (10 giây):**
   - Cầm chiếc **bật lửa (quẹt gas)** cách camera khoảng `0,5m – 1,2m` (ưu tiên đưa qua trước nền áo tối màu để camera dễ lấy nét màu cam của lửa) và bật lửa giữ khoảng `0,5 – 1 giây`.
   - Hệ thống khoanh đỏ ngọn lửa (`FIRE`), vẽ **đường nối màu vàng từ khớp cổ tay (`wrist`) tới ngọn lửa**, đổi trạng thái người sang **`ID: 01 [CHAM LUA / KHOI!]`** và phát cảnh báo **`Phát hiện hành vi châm lửa`** kèm ảnh chụp bằng chứng.
