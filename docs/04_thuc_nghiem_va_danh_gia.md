# CHƯƠNG 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ

> **Nguyên tắc của chương này:** chỉ ghi những con số đã đo được trên hệ thống thật, kèm cách tái lập. Những phần chưa đo được ghi rõ là **chưa có**. Cập nhật lần cuối: 11/11/2026.

---

## 4.1. Môi trường thử nghiệm

| Thành phần | Giá trị |
| :--- | :--- |
| Máy thử | Laptop, CPU Intel Core i7-9750H (6 nhân / 12 luồng), RAM 16 GB |
| GPU | Không dùng (PyTorch bản CPU) |
| Hệ điều hành | Windows 11 |
| Python | 3.12 |
| Thư viện chính | `ultralytics` 8.4, `torch` 2.15 (CPU), `opencv-python` 4.11, `fastapi` 0.162 — danh sách đầy đủ: `requirements-tested.txt` |
| Mô hình | `yolov8n-pose.pt` (trọng số huấn luyện sẵn của Ultralytics), ByteTrack mặc định |

Chưa thử trên máy có GPU, chưa thử với camera IP/RTSP thật.

---

## 4.2. Dữ liệu thử nghiệm hiện có

Thư mục `data/test_videos/` (không đưa lên git):

| Video | Thời lượng | Nội dung | Đã dùng để đánh giá |
| :--- | ---: | :--- | :---: |
| `fall_sample.mp4` | 11,3 s | Một người ngã và nằm trên sàn | Có |
| `fight1.mp4` | 18,7 s | Hai người vật lộn dưới sàn | Có |
| `01_video.mp4` | 45,0 s | Người đi lại | Có (âm tính) |
| `02_video.mp4` | 37,8 s | Người đi lại | Có (âm tính) |
| `people_walking.mp4` | 49,7 s | Nhiều người đi bộ | Có (âm tính) |
| `03_video.mp4`, `05_video.mp4`, `walking_corridor.mp4` | — | Chưa xem xét | Chưa |

**Nhận xét về dữ liệu:**
- Mỗi hành vi ngã và đánh nhau chỉ có **một** clip dương tính. **Chưa có clip chạy nhanh nào.**
- Chưa có clip âm tính khó (cúi nhặt đồ, ngồi xổm, nằm nghỉ; đi nhanh; ôm, bắt tay, nhảy, thể thao).
- Chưa có file nhãn ghi thời điểm bắt đầu/kết thúc của từng sự kiện; nguồn gốc các video chưa được ghi lại.

Với lượng dữ liệu này **chưa thể tính Precision/Recall có ý nghĩa**.

---

## 4.3. Kết quả đã đo

### 4.3.1. Kiểm thử tự động

```bash
python -m unittest discover -s tests -v
```

23/23 bài kiểm thử đạt. Các bài này kiểm tra **logic** trên dữ liệu mô phỏng:
- Té ngã (`tests/test_fall_fsm.py`): ngã ở 10/15/30/60 FPS và nhiều kích thước người; cúi chậm; vấp nhưng không nằm; đứng dậy sau ngã; mất khớp xương; Tier 2 chấp nhận/từ chối (dùng mô hình giả).
- Đánh nhau (`tests/test_fighting.py`): ra đòn; chuyển động đồng đều; đứng gần nhau; ở xa; chênh lệch tiền cảnh/hậu cảnh; cooldown; mất dấu; xác nhận theo thời gian.
- Luồng video dùng chung (`tests/test_shared_stream.py`): nhiều người xem dùng chung một kết quả AI; tiếp tục chạy sau một khung hình lỗi.

Chưa có bài kiểm thử tự động cho hành vi chạy nhanh. Các bài trên không đo độ chính xác trên video thật.

### 4.3.2. Chạy lại video mẫu qua toàn bộ pipeline

```bash
python -m scripts.evaluate_behaviors_local --check
```

Script chạy YOLOv8n-Pose + ByteTrack + toàn bộ pipeline trên 5 video, lấy mẫu khoảng 12–16 khung hình mỗi giây, mốc thời gian theo vị trí trong video. **Script này tắt hành vi chạy nhanh**, và tiêu chí đạt chỉ là "tập loại cảnh báo phát ra đúng với mong đợi".

| Video | Mong đợi | Cảnh báo phát ra | Đạt tiêu chí |
| :--- | :--- | :--- | :---: |
| `fall_sample.mp4` | Ngã | 1 cảnh báo ngã tại 5,09 s | Có |
| `fight1.mp4` | Đánh nhau | 1 cảnh báo đánh nhau tại 0,48 s | Có |
| `01_video.mp4` | Không | Không | Có |
| `02_video.mp4` | Không | Không | Có |
| `people_walking.mp4` | Không | Không | Có |

**Đọc kết quả này một cách thận trọng:**
- **Đánh nhau:** trên `fight1.mp4` trạng thái cảnh báo chỉ được giữ khoảng 0,16 giây (từ 0,48 s đến 0,64 s) trong clip dài 18,7 giây. Phần còn lại của clip hai người bị YOLO gộp thành một khung nên không còn cặp người để xét. Tiêu chí "có ít nhất một cảnh báo" đạt, nhưng khả năng phát hiện thực tế trên clip này là yếu.
- **Ngã:** một clip là quá ít để kết luận.

### 4.3.3. Hành vi chạy nhanh trên các video không có người chạy

Phát lại dữ liệu khớp xương đã lưu của 5 video trên với hành vi chạy **bật**: không có cảnh báo chạy nào. Vận tốc chuẩn hoá lớn nhất ghi nhận trên các video người đi bộ nằm trong khoảng 1,1–1,3 BH/s; riêng `01_video.mp4` có một người đạt đỉnh 2,55 BH/s nhưng không đủ số lần liên tiếp để báo.

Kết quả này chỉ cho thấy **không có báo động giả trên 5 clip này**. Vì chưa có clip chạy thật, **chưa biết hệ thống có bắt được người chạy hay không**.

### 4.3.4. Tốc độ xử lý

Đo thời gian của `process_frame` (YOLO + ByteTrack + phân tích hành vi + vẽ khung, **không** gồm mã hoá JPEG và truyền mạng) trên CPU, video 640×360, 120 khung hình sau khi bỏ 10 khung đầu:

| Video | Trung vị | Phân vị 90 | Tương đương |
| :--- | ---: | ---: | ---: |
| `fall_sample.mp4` | 48 ms | 55 ms | ≈ 21 khung hình/giây |
| `01_video.mp4` | 51 ms | 56 ms | ≈ 20 khung hình/giây |

Chưa đo: tốc độ trên GPU, tốc độ với video độ phân giải cao hoặc đông người, độ trễ từ camera đến màn hình, và so sánh độ trễ giữa kiến trúc đa luồng với vòng lặp đơn luồng.

### 4.3.5. Mô hình Bi-LSTM

**Chưa có kết quả hợp lệ.** Hai hình `docs/training_curves.png` và `docs/confusion_matrix.png` sinh ra từ một lần chạy script huấn luyện khi không có file dữ liệu; trong trường hợp đó script tự sinh 300 mẫu ngẫu nhiên để thử luồng mã. Dấu hiệu: tập kiểm tra có đúng 45 mẫu (15% của 300), độ chính xác trên tập kiểm định dao động quanh 35% (mức đoán ngẫu nhiên với 3 lớp) và hàm mất mát trên tập kiểm định tăng liên tục. Hai hình này **không được dùng làm kết quả** và sẽ được thay khi huấn luyện bằng dữ liệu thật.

---

## 4.4. Phương pháp đánh giá sẽ áp dụng

### 4.4.1. Đánh giá theo sự kiện

Tính độ chính xác theo từng khung hình không phản ánh đúng mục đích của hệ thống cảnh báo: một cú ngã kéo dài nhiều khung hình nhưng chỉ cần **một** cảnh báo đúng lúc. Vì vậy đánh giá theo **sự kiện**:

- **TP (bắt đúng):** một sự kiện có nhãn được ít nhất một cảnh báo đúng loại trong khoảng thời gian của nó (cho phép lệch một khoảng định trước ở hai đầu).
- **FN (bỏ sót):** sự kiện có nhãn không có cảnh báo nào khớp.
- **FP (báo giả):** cảnh báo không khớp với sự kiện có nhãn nào. Nhiều cảnh báo lặp lại trong cùng một sự kiện chỉ tính một lần.

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Báo cáo thêm:
- **Số báo động giả mỗi giờ** video âm tính.
- **Độ trễ phát hiện:** từ lúc sự kiện bắt đầu (theo nhãn) đến cảnh báo đầu tiên.
- Riêng đánh nhau: **tỷ lệ thời gian được phủ** — phần thời lượng của sự kiện mà hệ thống giữ trạng thái cảnh báo.

### 4.4.2. Dữ liệu cần có

Mỗi hành vi tối thiểu 15–20 clip dương tính và một tập âm tính khó:

| Hành vi | Dương tính | Âm tính khó |
| :--- | :--- | :--- |
| Chạy nhanh | Chạy ngang, chạy chéo, chạy về phía/ra xa camera; gần và xa | Đi nhanh, đi bộ bình thường, người đi sát camera |
| Té ngã | Ngã ngang, ngã trước/sau, ngã chéo; gần và xa | Cúi nhặt đồ, ngồi xổm nhanh/chậm, ngồi ghế, chủ động nằm xuống |
| Đánh nhau | Đấm, đá, xô đẩy, vật lộn đứng, vật lộn dưới sàn | Ôm, bắt tay, đập tay, nhảy, chơi thể thao, đi sát nhau |

Nguồn công khai có thể dùng: UR Fall Detection, Le2i Fall (ngã); RWF-2000, Hockey Fight (đánh nhau). Mỗi clip cần ghi nguồn gốc.

Định dạng nhãn dự kiến (một dòng cho mỗi sự kiện; clip âm tính không có dòng nào):

```csv
video,event_type,start_seconds,end_seconds
ten_video.mp4,FALLING,4.2,9.0
```

### 4.4.3. Các thí nghiệm dự kiến

1. **Bảng Precision / Recall / F1 theo sự kiện** cho từng hành vi trên tập có nhãn.
2. **So sánh có/không Tier 2** cho té ngã, sau khi Bi-LSTM được huấn luyện bằng dữ liệu thật và chia tập theo video.
3. **So sánh phương pháp đánh nhau:** luật theo cặp người hiện tại so với phương án bổ sung mô hình phân loại clip.
4. **Độ nhạy theo ngưỡng** của các tham số chính (ngưỡng vận tốc chạy, thời gian xác nhận nằm, ngưỡng vận tốc tứ chi).
5. **Tốc độ xử lý** trên CPU và GPU, theo số người trong khung hình.

---

## 4.5. Hạn chế hiện tại

1. **Đánh nhau** phụ thuộc vào việc tách được hai khung người; mất tác dụng khi hai người bị gộp thành một (vật lộn sát nhau, nằm trên sàn).
2. **Chạy nhanh** đo dịch chuyển trong mặt phẳng ảnh nên yếu với chuyển động dọc trục camera; chưa được kiểm chứng trên video chạy thật.
3. **Té ngã** không phát hiện người đã nằm sẵn khi vào khung hình hoặc ngã rất chậm; ngồi xổm rất nhanh hoặc chủ động nằm nhanh có thể gây báo giả.
4. **Bám vết:** khi ByteTrack đổi ID của một người, trạng thái ngã đang khoá và cooldown của người đó bị mất, có thể gây cảnh báo lặp.
5. **Điểm tin cậy** của cả ba hành vi là điểm quy ước, không phải xác suất.
6. **Điều kiện hình ảnh:** thiếu sáng, che khuất kéo dài, hoặc góc nhìn thẳng từ trần xuống làm giảm chất lượng khớp xương và ảnh hưởng tới cả ba hành vi.

## 4.6. Hướng phát triển

1. Xây dựng bộ video có nhãn và công cụ đánh giá theo sự kiện (mục 4.4) — điều kiện tiên quyết cho mọi cải tiến khác.
2. Huấn luyện tầng xác minh té ngã bằng dữ liệu thật.
3. Bổ sung mô hình phân loại clip cho đánh nhau.
4. Thêm tín hiệu tần suất bước chân cho chạy nhanh.
5. Thử nghiệm trên GPU và camera IP thật; xuất mô hình sang ONNX/TensorRT nếu cần tốc độ cao hơn.
