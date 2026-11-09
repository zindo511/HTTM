# Phát hiện té ngã theo thời gian

Luồng: `NORMAL → IMPACT_PENDING → LYING_VERIFY → FALL_LATCHED`.
Ba pha phát hiện nằm sau trạng thái bình thường. Mỗi track có trạng thái riêng.

- Học chiều cao đứng tham chiếu từ trung vị các mẫu trong 1,5 giây gần nhất;
  cần tối thiểu 0,3 giây quan sát đứng để khởi tạo. Đóng băng giá trị này trong
  quá trình nghi ngờ ngã và khi cảnh báo đang khóa.
- Vận tốc dùng chiều cao đứng làm đơn vị (BH/s), trên cửa sổ 0,2 giây.
  Dùng hông nếu cả hai đầu cửa sổ có điểm hợp lệ; nếu không dùng tâm bbox.
- Mở cửa sổ 1,2 giây khi hạ người nhanh, hoặc khi sụt chiều cao kèm tốc độ
  hạ đỉnh bbox đủ lớn. Sụt chiều cao đơn thuần không kích hoạt.
- Xác nhận tư thế thấp/ngang liên tục ít nhất 0,5 giây trong cửa sổ trên.
  Nếu tư thế trở lại bình thường, bộ đếm xác nhận bắt đầu lại.
- Khóa `FALLING_ALERT` đến khi chiều cao và tỷ lệ khung phục hồi liên tục
  0,5 giây. Ngồi dậy nhưng chưa đủ ngưỡng phục hồi vẫn giữ cảnh báo.
- Khoảng trống quan sát trên 0,3 giây hủy ứng viên chưa xác nhận, nhưng không
  tự xóa cảnh báo đã khóa. Track bị xóa theo TTL hoặc đổi ID không giữ được latch.
- Vai/hông dùng confidence tối thiểu 0,3, cho phép chỉ một bên nhìn thấy.
  Khi thiếu cả hai, bỏ tiêu chí góc và dùng bbox.
- Forward-fill tối đa 5 frame và tối đa 0,25 giây kể từ pose hợp lệ gần nhất.
  Các bản sao được đánh dấu trong `keypoint_observed_history`, ghép với bbox
  gốc của pose. FSM không dùng pose sao chép để tính góc/vận tốc hông.
  Đây là bảo toàn cấu trúc chuỗi, không khôi phục chuyển động đã mất.
- FSM được xét trước phát hiện chạy. Pipeline giữ ưu tiên hiển thị ngã đã khóa
  khi bộ phát hiện đánh nhau cùng cập nhật trạng thái.
- Bi-LSTM vẫn tắt theo cấu hình hiện tại. Nếu bật và nạp model, phải đủ
  30 mẫu và vượt ngưỡng xác minh. Điểm 0,9 của Tier 1 chỉ là điểm heuristic,
  không phải xác suất chính xác được hiệu chuẩn.

Các ngưỡng nằm trong `configs/config.yaml`, mục `behaviors.falling` và `tracking`.
Không tự thu thập hay gán nhãn dữ liệu huấn luyện trong bản nâng cấp này.

## Kiểm thử tự động

Từ thư mục dự án, với Python đã có numpy và OpenCV:

```powershell
python -m unittest discover -s tests -v
```

Các kiểm thử mô phỏng kiểm tra logic FSM, 10/15/30/60 FPS, kích thước người
0,5/1/2 lần, ngã dọc trục, phục hồi, cú hạ người chậm, vấp rồi hồi phục,
confidence thấp, mất pose và xác minh Bi-LSTM bằng model giả.
Chúng không đo độ chính xác của YOLO hay recall trên video thật.

## Đánh giá webcam/video

Dùng cùng bộ video có nhãn để so sánh bản cũ và mới. Ghi lại khoảng cách,
góc quay, FPS thực tế và điều kiện che khuất. Mỗi clip nên có đoạn đứng ban đầu
để học chiều cao tham chiếu. Các nhóm cần thử:

1. Ngã mô phỏng trong điều kiện an toàn: ngang, trước/sau, chéo; gần và xa.
2. Cúi nhặt đồ lâu, ngồi xổm nhanh/chậm, ngồi ghế, chủ động nằm xuống.
3. Vấp rồi đứng lại, đi/chạy bình thường, người đi ra/vào khung hình.
4. Nằm yên sau ngã, đứng dậy thoáng qua rồi hạ xuống, hồi phục ổn định.
5. Che khuất ngắn/dài, người sát mép ảnh, nhiều người gây đổi track ID.

Đếm theo sự kiện: TP là ngã được cảnh báo, FN là ngã bị bỏ sót, FP là cảnh báo
trên hành vi bình thường. Báo cáo recall = TP/(TP+FN), precision = TP/(TP+FP),
FP/giờ và độ trễ từ lúc bắt đầu tư thế thấp/nằm đến cảnh báo. Quy định trước
cửa sổ ghép một cảnh báo với một lần ngã và không đếm lặp cùng sự kiện.

0,5 giây là thời gian xác nhận bằng timestamp, chưa bao gồm độ trễ camera,
suy luận và truyền hình ảnh. Không bảo đảm 100%, 360 độ hay mọi khoảng cách.
Bbox thấp không chứng minh nằm trên sàn; ngồi xổm nhanh hoặc chủ động nằm
nhanh vẫn có thể báo giả. Người đã nằm khi mới xuất hiện, ngã rất chậm,
che khuất kéo dài hoặc đổi ID có thể bị bỏ sót. Cần dữ liệu thật để điều chỉnh
ngưỡng hoặc huấn luyện bộ xác minh, thay vì chỉ nới ngưỡng.
