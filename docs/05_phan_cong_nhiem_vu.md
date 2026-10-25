# BẢNG KIỂM TOÁN KỸ THUẬT, PHÂN CÔNG NHIỆM VỤ & THỎA THUẬN LÀM VIỆC NHÓM

> [!IMPORTANT]
> Tài liệu này quy định rõ các hạng mục kỹ thuật cần hoàn thiện của dự án **Hệ thống Trinh sát Phát hiện Hành vi Bất thường (AI Surveillance)**, phân định ranh giới mã nguồn (File Ownership) và quy tắc phối hợp Git để các thành viên triển khai độc lập, hiệu quả, **tuyệt đối không xảy ra xung đột mã nguồn (Git Merge Conflict)**.

---

## 1. HIỆN TRẠNG DỰ ÁN (PHẦN NỀN TẢNG ĐÃ HOÀN THÀNH)

**Nhóm trưởng (Kiến trúc sư hệ thống)** đã thiết lập và hoàn thiện phần khung lõi (Core Foundation) của toàn dự án:
- Hoàn thành bộ tài liệu thiết kế học thuật từ Chương 1 đến Chương 4 trong thư mục `docs/`.
- Xây dựng kiến trúc đa luồng Tầng 1 (`src/core/capture.py` với `DropOldestQueue`) chống trễ luồng RTSP/Video.
- Tích hợp mô hình `YOLOv8n-Pose` và thuật toán bám vết `ByteTrack` trong `src/core/pipeline.py`.
- Xây dựng bộ điều phối cảnh báo bất đồng bộ `src/core/alert_manager.py`, khung cơ sở dữ liệu `src/database/db_service.py` và khung giao diện Web FastAPI + WebSocket (`web/`).

Để hệ thống đạt độ hoàn thiện **100% phục vụ báo cáo và chấm điểm thực nghiệm**, phần công việc hoàn thiện **9 điểm kỹ thuật còn lại** được chia đều cho **2 thành viên** theo 2 mảng độc lập dưới đây.

---

## 2. DANH SÁCH 9 VẤN ĐỀ KỸ THUẬT CẦN HOÀN THIỆN

### Nhóm vấn đề 1: Thuật toán Hành vi, Tracking & Huấn luyện Mô hình AI (`src/behaviors/`, `src/models/`, `scripts/`)

1. **Lỗi "Kẹt trạng thái" (Stuck State Bug) sau khi hết vi phạm:**
   - *Hiện trạng:* Trong [`src/behaviors/intrusion.py`](file:///e:/btl-httm/src/behaviors/intrusion.py#L55-L56) và [`src/behaviors/fall_run.py`](file:///e:/btl-httm/src/behaviors/fall_run.py#L64-L75), khi đối tượng đã bước ra ngoài vùng cấm hoặc ngừng chạy/đứng dậy sau khi ngã, biến `track.current_state` vẫn bị giữ nguyên ở `"INTRUSION_ALERT"`, `"RUNNING_ALERT"` hoặc `"FALLING_ALERT"` mà không trả về `"NORMAL"`.
   - *Yêu cầu:* Cập nhật logic reset `track.current_state = "NORMAL"` ngay khi điều kiện vi phạm không còn thỏa mãn.
2. **Giả định cứng bước thời gian `dt = 5.0 / 30.0` khi tính vận tốc chạy:**
   - *Hiện trạng:* Tại dòng 60 file [`src/behaviors/fall_run.py`](file:///e:/btl-httm/src/behaviors/fall_run.py#L60), `dt` đang gán cố định cho video 30 FPS. Khi chạy video có FPS khác (ví dụ 12 FPS hoặc 20 FPS), vận tốc chuẩn hóa `v_norm` bị tính sai lệch.
   - *Yêu cầu:* Lưu thêm hàng đợi `timestamps: deque(maxlen=30)` trong `TrackHistory` ([`src/tracking/track_manager.py`](file:///e:/btl-httm/src/tracking/track_manager.py)) và tính `dt = track.timestamps[-1] - track.timestamps[-5]` theo thời gian thực.
3. **Thiếu công thức kiểm tra độ nghiêng thân người (`body_lean_min`) và vận tốc rơi thẳng đứng (`vy_drop_min`):**
   - *Hiện trạng:* Trong [`src/behaviors/fall_run.py`](file:///e:/btl-httm/src/behaviors/fall_run.py#L25-L31), hai tham số `self.body_lean_min` (góc đổ người khi chạy) và `self.vy_drop_min` (vận tốc rơi của khớp hông theo trục $Oy$ khi ngã) đã được khai báo ở `__init__` nhưng chưa được đưa vào biểu thức điều kiện trong hàm `process()`.
   - *Yêu cầu:* Bổ sung phép tính độ nghiêng trục lưng khi chạy và vận tốc rơi thẳng đứng $V_y = \frac{y_{hip}^{(t)} - y_{hip}^{(t-5)}}{\Delta t \cdot H}$ khi phát hiện té ngã ở Tier 1.
4. **Lỗi "Double Softmax" làm giảm khả năng hội tụ khi train mạng Bi-LSTM:**
   - *Hiện trạng:* Trong [`src/models/bilstm_verifier.py`](file:///e:/btl-httm/src/models/bilstm_verifier.py#L37), hàm `forward()` trả về `self.softmax(logits)`. Tuy nhiên trong [`scripts/train_bilstm.py`](file:///e:/btl-httm/scripts/train_bilstm.py#L79) lại dùng `nn.CrossEntropyLoss()` (hàm này trong PyTorch đã tự động áp dụng `LogSoftmax` bên trong).
   - *Yêu cầu:* Sửa `BiLSTMActionClassifier.forward()` chỉ trả về `logits` thô để phục vụ `CrossEntropyLoss` khi train, và chỉ gọi `torch.softmax(logits, dim=1)` bên trong hàm `BiLSTMInferenceWrapper.predict()` khi suy luận.
5. **Thiếu biểu đồ đường cong huấn luyện Loss/Accuracy (`docs/training_curve.png`) và file trọng số thật `weights/fall_bilstm.pt`:**
   - *Hiện trạng:* File [`scripts/train_bilstm.py`](file:///e:/btl-httm/scripts/train_bilstm.py#L160-L169) mới chỉ xuất ra `docs/confusion_matrix.png`, chưa vẽ biểu đồ đường cong `train_loss`, `val_loss`, `train_acc`, `val_acc` qua 50 Epochs; đồng thời chưa có file trọng số `weights/fall_bilstm.pt` được train từ dữ liệu thật nên `enable_bilstm_tier2` trong `config.yaml` đang phải đặt là `false`.
   - *Yêu cầu:* Bổ sung đoạn code `matplotlib` vẽ đồ thị Loss/Accuracy lưu ra `docs/training_curve.png`, chạy trích xuất dữ liệu khớp xương và train trên Google Colab để lấy file `weights/fall_bilstm.pt`, sau đó bật `enable_bilstm_tier2: true`.

---

### Nhóm vấn đề 2: Giao diện Web Dashboard, Backend API, SQLite & Kiểm thử (`web/`, `src/database/`)

6. **Lỗi lệch tọa độ vùng cấm ROI khi vẽ bằng chuột trên trình duyệt:**
   - *Hiện trạng:* Tại dòng 105–106 và 116–117 của [`web/static/js/dashboard.js`](file:///e:/btl-httm/web/static/js/dashboard.js#L105-L117), tỷ lệ quy đổi tọa độ đang gán cứng là `1280` và `720`. Khi phát video có độ phân giải khác (như `people_walking.mp4` kích thước `768x432`), đa giác vẽ trên trình duyệt bị lệch hoàn toàn so với đa giác OpenCV vẽ dưới backend.
   - *Yêu cầu:* Thay hằng số `1280` và `720` bằng kích thước khung hình thực tế của thẻ ảnh stream: `videoStream.naturalWidth || 1280` và `videoStream.naturalHeight || 720`.
7. **Chưa cho phép chọn loại vùng "Giám sát Lảng vảng (`LOITERING_ZONE`)" trên giao diện Web:**
   - *Hiện trạng:* Trong hàm `saveROI()` của [`web/static/js/dashboard.js`](file:///e:/btl-httm/web/static/js/dashboard.js#L156-L161), thuộc tính `type` đang bị gán cứng là `"RESTRICTED_INTRUSION"`, khiến người dùng không thể vẽ vùng phát hiện Lảng vảng từ giao diện.
   - *Yêu cầu:* Thêm một thẻ `<select id="zoneTypeSelect">` trên thanh công cụ ở [`web/templates/index.html`](file:///e:/btl-httm/web/templates/index.html) cho phép chọn giữa 2 chế độ: `Vùng cấm Xâm nhập (RESTRICTED_INTRUSION)` và `Vùng giám sát Lảng vảng (LOITERING_ZONE)` trước khi bấm Lưu vùng.
8. **Mất lịch sử cảnh báo và không đồng bộ vùng ROI với SQLite khi tải lại trang (F5):**
   - *Hiện trạng:*
     - Khi nhấn F5 trình duyệt, cột cảnh báo bên phải bị trống trơn vì `dashboard.js` chưa gọi API `GET /api/events` để nạp lại lịch sử cảnh báo đã lưu trong SQLite.
     - Khi khởi động server hoặc bấm "Xóa sạch vùng" (`POST /api/zones/clear` trong [`web/app.py`](file:///e:/btl-httm/web/app.py#L121-L126)), hệ thống mới chỉ xóa mảng `zones` trên RAM chứ chưa xóa trong bảng `zones` của SQLite (`src/database/db_service.py`).
   - *Yêu cầu:* Bổ sung hàm `clear_all_zones()` trong `DatabaseService`, tự động nạp danh sách vùng từ SQLite khi khởi tạo, và gọi `fetch("/api/events")` khi trang web vừa tải xong (`window.onload`).
9. **Hiển thị sai tổng số cảnh báo trên HUD video và thiếu bộ đếm phân loại dưới Footer:**
   - *Hiện trạng:* Tại dòng 154 của [`src/core/pipeline.py`](file:///e:/btl-httm/src/core/pipeline.py#L154), hàm đang gọi `self.db_service.get_recent_events(limit=1)` rồi lấy `len(events)` nên góc màn hình HUD luôn chỉ hiển thị tối đa `01` cảnh báo. Ngoài ra, thanh `<footer>` trên web mới chỉ hiện chữ tĩnh chứ chưa đếm số lượng từng loại cảnh báo (`Xâm nhập: 0 | Lảng vảng: 0 | Chạy: 0 | Té ngã: 0`).
   - *Yêu cầu:* Thêm hàm `count_total_events()` trong `db_service.py` (hoặc dùng biến đếm trong `AlertManager`) để hiển thị đúng tổng số cảnh báo trên HUD, đồng thời cập nhật số đếm từng loại hành vi dưới thanh `<footer>` mỗi khi có cảnh báo mới.

---

## 3. BẢNG PHÂN CÔNG NHIỆM VỤ CHI TIẾT CHO 2 THÀNH VIÊN

| Thành viên | Vai trò chuyên môn | Thư mục / File độc quyền chỉnh sửa | Danh sách đầu việc cụ thể | Sản phẩm bàn giao (Deliverables) |
| :--- | :--- | :--- | :--- | :--- |
| **Thành viên A** | **Kỹ sư AI & Thuật toán Hành vi**<br>*(Behavior Algorithms & Deep Learning)* | 📁 `src/behaviors/`<br>📁 `src/tracking/`<br>📁 `src/models/`<br>📁 `scripts/`<br>📁 `weights/` | 1. Giải quyết **Vấn đề #1**: Tự động reset `track.current_state = "NORMAL"` trong `intrusion.py` và `fall_run.py` khi đối tượng hết vi phạm.<br>2. Giải quyết **Vấn đề #2**: Thêm `timestamps` vào `TrackHistory` (`track_manager.py`) để tính đúng `dt` thực tế.<br>3. Giải quyết **Vấn đề #3**: Đưa công thức góc nghiêng lưng `body_lean_min` và vận tốc rơi thẳng đứng `vy_drop_min` vào hàm `process()` của `fall_run.py`.<br>4. Giải quyết **Vấn đề #4 & #5**: Sửa lỗi Double Softmax trong `bilstm_verifier.py`, thêm code vẽ `docs/training_curve.png` vào `train_bilstm.py`, chạy train trên Colab để xuất ra `weights/fall_bilstm.pt`. | • Nhánh Git: `feat/ai-and-behaviors`<br>• Các thuật toán Xâm nhập, Lảng vảng, Chạy, Ngã hoạt động chuẩn xác, tự trả về khung xanh khi hết vi phạm.<br>• File trọng số `weights/fall_bilstm.pt` + 2 biểu đồ `confusion_matrix.png`, `training_curve.png`. |
| **Thành viên B** | **Kỹ sư Fullstack Web, Database & Kiểm thử**<br>*(Web Dashboard, SQLite & QA)* | 📁 `web/`<br>📁 `src/database/`<br>📄 `src/core/pipeline.py` *(chỉ sửa dòng 154)*<br>📁 `data/test_videos/` | 1. Giải quyết **Vấn đề #6**: Sửa công thức tính `scaleX`, `scaleY` theo `videoStream.naturalWidth / naturalHeight` trong `web/static/js/dashboard.js`.<br>2. Giải quyết **Vấn đề #7**: Thêm dropdown chọn loại vùng (`RESTRICTED_INTRUSION` / `LOITERING_ZONE`) trên giao diện `index.html` và gửi đúng `type` khi lưu vùng.<br>3. Giải quyết **Vấn đề #8**: Thêm hàm xóa/đồng bộ bảng `zones` trong `db_service.py` và viết hàm JS tự tải lại lịch sử cảnh báo từ `/api/events` khi F5 trang web.<br>4. Giải quyết **Vấn đề #9**: Sửa số đếm tổng cảnh báo trên HUD và thêm bộ đếm theo từng loại hành vi dưới thanh `<footer>` của trang web.<br>5. Quay/sưu tầm đủ bộ video kiểm thử thực tế cho 4 kịch bản đưa vào `data/test_videos/`. | • Nhánh Git: `feat/web-and-database`<br>• Giao diện Dashboard vẽ chuẩn tọa độ trên mọi độ phân giải video, chọn được cả vùng Xâm nhập & Lảng vảng, F5 vẫn giữ nguyên lịch sử cảnh báo.<br>• Bộ video kiểm thử thực tế chạy mượt trên hệ thống. |
| **Nhóm trưởng (Bạn)** | **Kiến trúc sư Hệ thống & Quản lý Dự án**<br>*(System Architect & Lead)* | 📄 `configs/config.yaml`<br>📄 `main.py`<br>📁 `docs/` | 1. Đã xây dựng toàn bộ kiến trúc nền tảng 4 tầng, đa luồng Capture/Pipeline và tài liệu 4 chương.<br>2. Hỗ trợ kỹ thuật, giải đáp thắc mắc cho 2 thành viên.<br>3. Kiểm duyệt mã nguồn (Review Pull Request), tích hợp 2 nhánh của Thành viên A & B vào nhánh `main` và kiểm thử tổng thể lần cuối. | • Nhánh chính: `main`<br>• Hệ thống hợp nhất hoàn chỉnh và bộ báo cáo/slide bảo vệ cuối kỳ. |

---

## 4. BIÊN BẢN THỎA THUẬN & QUY TẮC PHỐI HỢP (TEAM AGREEMENT)

Để quá trình ghép code diễn ra trơn tru 100%, tất cả thành viên thống nhất tuân thủ nghiêm ngặt **4 quy tắc** sau:

### 4.1. Quy tắc Phân vùng File (Strict File Ownership)
- **Thành viên A** tuyệt đối không sửa các file trong `web/` và `src/database/`.
- **Thành viên B** tuyệt đối không sửa các file trong `src/behaviors/`, `src/tracking/`, `src/models/` và `scripts/`.
- Nếu cần thay đổi thông số trong `configs/config.yaml`, thành viên đề xuất vào nhóm chat để Nhóm trưởng cập nhật.

### 4.2. Giao ước Chuẩn Dữ liệu (Data Contracts - Không tự ý đổi tên khóa/hàm)
1. **Chuẩn giao tiếp Mô hình Bi-LSTM (Thành viên A đảm bảo):**
   - Hàm `bilstm_model.predict(sequence_30x34)` nhận đầu vào là mảng NumPy kích thước `(1, 30, 34)` và bắt buộc trả về bộ 3 số thực `(prob_normal, prob_running, prob_fall)` có tổng bằng `1.0`.
2. **Chuẩn mã định danh Sự kiện & Vùng giám sát (Dùng chung giữa Backend và Frontend):**
   - Mã 4 loại hành vi (`event_type`): `"INTRUSION"`, `"LOITERING"`, `"RUNNING"`, `"FALLING"`.
   - Mã 2 loại vùng (`zone.type`): `"RESTRICTED_INTRUSION"` (vùng cấm xâm nhập) và `"LOITERING_ZONE"` (vùng giám sát lảng vảng).
3. **Chuẩn gói tin JSON đẩy qua WebSocket `/ws/alerts`:**
   ```json
   {
     "event_type": "INTRUSION | LOITERING | RUNNING | FALLING",
     "track_id": 3,
     "zone_id": "ZONE_01",
     "confidence": 94.5,
     "timestamp": "2026-10-25 17:30:00",
     "snapshot_url": "/snapshots/INTRUSION_3_1761388200.jpg",
     "extra_info": "Mô tả chi tiết sự kiện"
   }
   ```

### 4.3. Quy trình Làm việc với Git (Git Workflow)
1. **Lấy code mới nhất về máy và kích hoạt môi trường Python 3.12:**
   ```powershell
   git checkout main
   git pull origin main
   .\.venv\Scripts\Activate.ps1
   ```
2. **Tạo nhánh làm việc riêng (Mỗi thành viên 1 nhánh):**
   - Thành viên A: `git checkout -b feat/ai-and-behaviors`
   - Thành viên B: `git checkout -b feat/web-and-database`
3. **Quy tắc đặt tên Commit (Conventional Commits):**
   - Thêm tính năng: `git commit -m "feat: them tuy chon ve vung lang vang tren web"`
   - Sửa lỗi: `git commit -m "fix: sua loi ket trang thai trong intrusion va fall_run"`
4. **Đẩy nhánh lên GitHub và tạo Pull Request:**
   ```powershell
   git push -u origin <ten-nhanh-cua-ban>
   ```
   *(Tuyệt đối không push trực tiếp vào nhánh `main`. Nhóm trưởng sẽ kiểm tra và Merge).*

### 4.4. Tiêu chí Nghiệm thu Công việc (Definition of Done - DoD)
Một đầu việc chỉ được tính là **Hoàn thành** khi thỏa mãn đủ 3 điều kiện:
1. Chạy lệnh `.\.venv\Scripts\python.exe main.py` mượt mà, không phát sinh bất kỳ lỗi Exception nào trên Terminal hay Console trình duyệt.
2. Không làm giảm tốc độ xử lý FPS của luồng video.
3. Có ảnh chụp màn hình hoặc video ngắn (10–15 giây) gửi vào nhóm chứng minh tính năng/lỗi đó đã được xử lý triệt để.
