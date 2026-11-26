# Quy tắc bắt buộc cho giao_dien.py (demo) — bám đúng Báo cáo 1 và Báo cáo 2

Nhóm 19 · Hệ CSDL đa phương tiện · cập nhật 26/11/2026, sau khi đã nộp Báo cáo 2

Nguồn đối chiếu là hai bản **đã nộp, không sửa nữa**:
- Báo cáo 1: `bao_cao/[D23-Lớp 3] Nhóm 19-báo cáo 1.pdf` (nộp 23/10/2026).
- Báo cáo 2: `bao_cao/[D23-Lớp 3] Nhóm 19-báo cáo 2.pdf`, 28 trang, nộp 26/11/2026. Bản Word cùng tên chỉ dùng để gộp báo cáo cuối kỳ.

Demo chiếm 60% điểm và thầy chấm bằng cách đối chiếu demo với hai báo cáo. Báo cáo đã nộp nên **demo phải theo báo cáo**, không sửa báo cáo cho theo demo. Demo nói khác báo cáo ở chỗ nào thì chỗ đó bị coi là sai.

Trong Báo cáo 2, phần nhận xét viết bằng lời, không ghi số; **mọi số liệu nằm trong bảng và hình**. Khi thầy hỏi một con số, chỉ vào đúng bảng hoặc hình ở mục 4 dưới đây, hoặc chạy demo cho thầy xem.

## 1. Những gì KHÔNG được đổi (đã ghi trong báo cáo)

Các tệp `tien_xu_ly.py`, `xu_ly_anh.py`, `trich_rut.py`, `tim_kiem.py`, `csdl.py` đã **đóng băng**. Giao diện chỉ được gọi tới chúng, không được viết lại phép tính của chúng.

| Khâu | Giá trị đã cam kết | Nằm ở |
|---|---|---|
| Ảnh trong CSDL | 1005 ảnh JPG 256 × 256, tên man_0001 … man_1005 | BC1 Phần 1, mục 1.1 |
| Tiền xử lý 6 bước | xám Y = 0,299R + 0,587G + 0,114B; độ nét Laplace ≥ 20; dlib 1 mặt + 68 điểm mốc, kéo giãn 1,25, độ lệch < 0,08; căn chỉnh hai mắt cách nhau 64 px, trung điểm tại (128; 100); lọc nền đa giác viền hàm + lông mày dịch 0,6 lần khoảng cách hai mắt, nền gán trắng | BC1 Phần 1, mục 2 |
| Vector | 39 đặc trưng X0 – X38: hình học X0 – X4, kết cấu X5, X6, Color Histogram X7 – X38 (H 8 bin 0° – 40°, S 4 bin) | BC1 Phần 2; BC2 Phần 3 mục 1 |
| Vai trò | X0, X1, X7 – X38 tìm sự tương đồng; X2 – X6 tìm sự khác biệt | BC1 Bảng 2.1 |
| CSDL | SQLite, 2 bảng `anh` và `dac_trung` (khóa chính, khóa ngoại `ma_anh`); min, max ở `min_max.json` | BC2 Phần 3 mục 2 |
| Chuẩn hóa | min-max bằng min, max của CSDL; ảnh truy vấn ngoài [min; max] thì kẹp về [0; 1] | BC1 mục 2.4 Phần 2; BC2 Phần 4 mục 1.4 |
| Khoảng cách | D_b = khoảng cách Euclid trong bộ b chia √n_b; D = 1/3·D_hh + 1/3·D_kc + 1/3·D_ms; S = 1/(1 + D) | BC2 Phần 4 mục 1.3 |
| Trọng số | 1/3 cho mỗi bộ, **hằng số trong chương trình** | BC2 Phần 3 mục 2.2, Phần 4 mục 1.3 |

## 2. Luồng bắt buộc khi người dùng chọn một ảnh

1. Tệp không phải JPG thì từ chối và báo lý do.
2. Ảnh chưa đúng 256 × 256 thì **làm vuông** (`tien_xu_ly.lam_vuong`): cắt hình vuông quanh khuôn mặt, cạnh bằng 2 lần khung mặt, rồi thu về 256 × 256. Báo cáo 2 ghi việc này ở mục 1.2 Phần 4.
3. Chạy đủ 6 bước tiền xử lý, kể cả các bước kiểm tra. Ảnh không đạt thì dừng và báo đúng lý do: ảnh mờ (độ nét < 20), không đúng 1 khuôn mặt, hoặc điểm mốc không ổn định (độ lệch ≥ 0,08).
4. Trích 39 đặc trưng, chuẩn hóa, so với cả 1005 vector, sắp S giảm dần; S bằng nhau thì giữ thứ tự mã ảnh (BC2 Phần 4 mục 1.4, bước 5 – 6).
5. Ảnh lấy từ thư mục ảnh của CSDL thì **bỏ chính ảnh đó** khỏi kết quả.
6. Lấy 5 ảnh đầu.

## 3. Những gì giao diện PHẢI hiển thị (báo cáo đã hứa)

- 5 ảnh kết quả xếp theo S giảm dần, mỗi ảnh có **hạng, S, D, D_hh, D_kc, D_ms** (báo cáo ghi 4 chữ số thập phân, ví dụ 0,9548).
- Phần giải thích (bước 7, mục 1.4 Phần 4):
  - tỉ lệ ảnh xa hơn theo từng bộ;
  - X0, X1 và bin màu da lớn nhất (đặc trưng tìm sự tương đồng);
  - đặc trưng gần nhất và lệch nhất trong X2 – X6 (đặc trưng tìm sự khác biệt).
- Thông báo lý do khi ảnh bị loại (mục 2 ở trên).
- Ghi chú khi ảnh đã được làm vuông, và khi ảnh có trong CSDL nên bị bỏ khỏi kết quả.

## 4. Chạy thử demo phải ra đúng số trong báo cáo

Hai ảnh ở `data/anh_moi/` là hai trường hợp đề bài nêu (BC2 Phần 4 mục 2).

| Ảnh đầu vào | Top-5 (S) | Ghi chú |
|---|---|---|
| `Bruno-Fernandes-commons.jpg`, người đã có trong CSDL (Bảng 4.2) | man_0636 (0,9548), man_0890 (0,9452), man_0540 (0,9356), man_0795 (0,9329), man_0843 (0,9314) | ảnh cùng người man_0108 đứng hạng 26, S = 0,9161 |
| `chu-ngoc-nguyen-luc.jpg`, người chưa có trong CSDL (Bảng 4.3) | man_0636 (0,9482), man_0564 (0,9339), man_0496 (0,9299), man_0500 (0,9248), man_0625 (0,9243) | — |

Số để trả lời miệng khi thầy hỏi về đánh giá (BC2 Phần 4 mục 3.2): trên 8040 ảnh biến đổi, ảnh gốc đứng hạng 1 ở 96,4% số lần và nằm trong top-5 ở 99,0% số lần. Hai số này ghi trên cột "Tất cả 8 kiểu" của Hình 4.7; lời văn trong báo cáo không ghi số.

Bản đồ tìm số nhanh trong Báo cáo 2:

| Thầy hỏi về | Mở |
|---|---|
| Giá trị X0 – X6 của ảnh mẫu; nhỏ nhất, lớn nhất, trung bình, độ lệch chuẩn | Bảng 3.1 |
| Tham số và Color Histogram của ảnh mẫu | Bảng 3.2, Bảng 3.3 |
| Vector 39 giá trị của ảnh mẫu, vị trí trong bảng dac_trung | Hình 3.4 |
| Kết quả kiểm tra toàn vẹn CSDL | Bảng 3.4 |
| Phân bố đặc trưng, khoảng cách giữa các cặp ảnh | Hình 3.6 – 3.8 |
| Chức năng, dữ liệu vào, ra của từng khối | Bảng 4.1 |
| Đặc trưng, D từng bộ, D, S, hạng của trường hợp 1 / 2 | Bảng 4.2 / Bảng 4.3 |
| Tổng hợp đánh giá hai trường hợp | Bảng 4.4 |
| Tám kiểu biến đổi; tỉ lệ tìm lại ảnh gốc | Hình 4.6; Hình 4.7 |

## 5. Chỗ hiện tại còn lệch báo cáo, PHẢI sửa khi chỉnh giao diện

| # | Chỗ lệch | Báo cáo ghi | Cách sửa |
|---|---|---|---|
| 1 | 3 thanh kéo trọng số và nút "Về 1/3" (`giao_dien.py` dòng 232 – 239) | trọng số 1/3 là hằng số | Bỏ 3 thanh kéo; chỉ ghi công thức D = 1/3·D_hh + 1/3·D_kc + 1/3·D_ms |
| 2 | Chữ "lược đồ màu" hiện trên màn hình (dòng 685 – 686) và trong chú thích (dòng 19, 113, 623, 682, 725) | "Color Histogram" (BC1 mục 2.3.2 Phần 2) | Đổi thành "Color Histogram" |
| 3 | Chú thích trong code trỏ số mục, số bảng cũ: `tim_kiem.py` dòng 24, 78, 125; `danh_gia.py` dòng 149; `csdl.py` dòng 214 | công thức ở Phần 4 mục 1.3; kiểm tra toàn vẹn là Bảng 3.4; phép thử biến đổi là Hình 4.7 | Chỉ sửa chú thích, không đụng phép tính |

## 6. Được phép đổi tự do

Bố cục, màu, cỡ chữ, thứ tự các khung, cách vẽ biểu đồ, bớt các con số phụ không có trong báo cáo, đo và hiện thời gian chạy (báo cáo không ghi thời gian nên hiện hay không đều được).

## 7. Kiểm tra sau mỗi lần sửa

1. Chạy `venv310\Scripts\python.exe -X utf8 bao_cao\so_lieu_bc2\doi_chieu_so_lieu.py`, phải ra **113/113 KHỚP**. Lệch một số là demo hoặc báo cáo sai.
2. Mở giao diện, chạy hai ảnh ở mục 4, đối chiếu top-5 và S với Bảng 4.2, 4.3.
3. Không thêm học sâu (CNN, embedding), không đổi định nghĩa 39 đặc trưng, không đổi tham số tiền xử lý.
