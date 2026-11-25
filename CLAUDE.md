# Bối cảnh dự án

> **Đọc `HUONG_DAN_CHUNG.md` trước.** Tệp đó là bản mới nhất (pipeline, quy tắc viết báo cáo); khi khác với phần dưới thì theo `HUONG_DAN_CHUNG.md`.

BTL môn Hệ CSDL đa phương tiện (PTIT, GV Nguyễn Đình Hóa), nhóm 19, 3 người, định hướng BA, không mạnh code.

**Đề:** xây hệ CSDL lưu trữ và tìm kiếm ảnh chân dung đàn ông. Có ≥ 1000 ảnh cùng kích thước, mỗi ảnh 1 mặt nằm giữa, chụp thẳng. Nhập 1 ảnh mới → trả về 5 ảnh giống nhất, xếp giảm dần theo độ tương đồng.

**Mốc:**
- Báo cáo 1 (yêu cầu 1–2: dữ liệu, bộ thuộc tính) nộp 22/10/2026.
- Báo cáo 2 (yêu cầu 3–4: trích rút, CSDL, tìm kiếm) nộp 26/11/2026.
- Demo vào ngày thi, chấm cá nhân.

## Dữ liệu
- 1005 ảnh thu thập thủ công, mỗi người đúng 1 ảnh, đã resize 256×256 .jpg, đặt trong `data/anh256/`, tên `man_0001.jpg` … `man_1005.jpg`.
- Không có nhiều ảnh cho cùng một người → không có "đáp án đúng" theo người.

## Pipeline hiện tại
- `xu_ly_anh.py`: các phép xử lý tự code bằng numpy (ảnh xám, Laplace, xoay + nội suy song tuyến tính, RGB→HSV, LBP).
- `tien_xu_ly.py` (6 bước): B1 đọc/kiểm tra → B2 xám → B3 độ nét → B4 phát hiện mặt + 68 điểm mốc (dlib) → B5 căn chỉnh cho 2 mắt nằm ngang → B6 lọc nền bằng mặt nạ landmark; sau đó lưu vào `data/tien_xu_ly/`.
- `trich_rut.py`: vector 39 chiều
  - **Hình học (5):** tỉ lệ khung mặt d(0,16)/dY(27,8); gò má/hàm d(0,16)/d(4,12); khoảng cách mắt/rộng mặt; rộng cánh mũi d(31,35)/khoảng cách mắt; cao mũi dY(27,33)/dY(27,8).
  - **Kết cấu (2):** phương sai LBP vùng cằm–miệng (điểm 33, 54, 12, 8, 4, 48); mật độ lông mày (Otsu + LBP khác 0 và 255).
  - **Màu (32):** lược đồ H×S trên mặt nạ da (2 má + trán); H chia 8 ô trong 0–40°, S chia 4 ô.
- `tim_kiem.py`: chuẩn hóa min-max (lưu `min_max.json`); khoảng cách Euclid tính riêng từng bộ, chia √(số chiều), trọng số 1/3; S = 1/(1+D), lấy top-5. Vector, min/max, trọng số đọc từ CSDL.
- `csdl.py`: CSDL SQLite `data/khuon_mat.db`, 2 bảng `anh` 1–1 `dac_trung` (khóa ngoại `ma_anh`); min/max ở `min_max.json`, trọng số 1/3 là hằng số `tim_kiem.TRONG_SO` (chốt 24/11/2026); kiểm tra toàn vẹn; thêm ảnh mới.
- `giao_dien.py`: giao diện Tkinter; `danh_gia.py`: phép thử ảnh biến đổi.
- `chay_toan_bo.ipynb`: chạy toàn bộ.

## Giảng viên (quan trọng khi viết code và giải thích)
- Ưu tiên code tay hơn gọi thư viện. Nếu dùng thư viện, phải giải thích được bản chất: tự code thì gồm những bước gì.
- Xuất thân toán, hay hỏi ánh xạ công thức ↔ dòng code. Hỏi "tại sao" cho mọi con số, kể cả min/max và miền giá trị.
- Bài toán là **tìm kiếm tương đồng, không phải phân loại**. Phải giải thích được top-5 giống nhau ở điểm nào.
- Thích đơn giản hóa: "Đừng phức tạp hóa bài toán lên".
- Sẽ hỏi sâu về mô hình CSDL: chọn mô hình gì, vì sao, lệnh tạo bảng ra sao.

## Quy tắc khi làm việc trong repo này
- Code đơn giản, ít file, chú thích tiếng Việt, ghi rõ công thức ngay trong docstring.
- Không đổi định nghĩa 39 đặc trưng khi chưa hỏi. Tên cột trong `trich_rut.TEN_COT` được dùng chung cho báo cáo và CSDL.
- Phép xử lý mới ưu tiên tự code bằng numpy. Chỉ dùng OpenCV/dlib khi tự code quá phức tạp, và khi đó ghi bản chất vào docstring.
- Không thêm học sâu (CNN, embedding) vào pipeline chính.
- Báo cáo không đề cập râu, kính, phụ kiện.
- Trả lời bằng tiếng Việt, ngắn gọn.
- Khi giải thích hoặc viết nội dung báo cáo: chỉ dựa trên tài liệu nhóm cung cấp và ghi nguồn. Chỗ không có nguồn thì đánh dấu, không bịa.
