# Hướng dẫn chung – BTL Hệ CSDL đa phương tiện, nhóm 19

Dành cho cả nhóm và cho mọi phiên làm việc với Claude. Khi nội dung khác `CLAUDE.md`, **ưu tiên tệp này** (mới hơn).

## 1. Bài toán và mốc
- Đề: hệ CSDL lưu trữ và tìm kiếm ảnh chân dung đàn ông. ≥ 1000 ảnh cùng kích thước, mỗi ảnh 1 mặt nằm giữa, chụp thẳng. Nhập 1 ảnh mới → trả 5 ảnh giống nhất, xếp giảm dần theo độ tương đồng.
- Đây là **tìm kiếm tương đồng, không phải phân loại**: mỗi người chỉ 1 ảnh, không có "đáp án đúng"; phải giải thích được top-5 giống nhau ở điểm nào.
- Báo cáo 1 (yêu cầu 1–2: dữ liệu, bộ thuộc tính) nộp **22/10/2026**. Báo cáo 2 (yêu cầu 3–4: trích rút, CSDL, tìm kiếm) nộp **26/11/2026**. Demo vào ngày thi, chấm cá nhân.
- Nhóm: Bùi Quang Vinh (B23DCCN926), Phạm Ngọc Minh (B23DCCN563), Trần Hoàng Nam (B23DCCN591). Mục tiêu của nhóm: qua môn.

## 2. Quy tắc viết báo cáo (bắt buộc)
1. **Chỉ viết kết quả cuối.** Không kể quá trình thử sai, lần chạy trước, số ảnh từng bị loại hay đã thay. Mọi con số là số của lần chạy cuối.
2. **Không ghi phân công chi tiết** từng người từng việc. Bảng đầu báo cáo chỉ có họ tên và mã sinh viên.
3. **Được dùng bảng** nếu thầy đọc dễ (user cho phép 21/10/2026). **Không có bảng đối chiếu và không có phụ lục** (không đối chiếu yêu cầu đề, không đối chiếu câu hỏi vấn đáp, không ánh xạ công thức – code).
4. **Trình bày theo bản mẫu** `báo cáo dpt nhận diện ảnh 1-file mẫu 8.5đ.pdf`: trang bìa, mục lục, lời nói đầu, yêu cầu, các “PHẦN”, mục con đánh số “1.”, “1.1,”, gạch đầu dòng “-” và “+”, có “Ưu điểm / Nhược điểm”, “Ví dụ”, câu hỏi “Tại sao…?” và kết luận “=>”. Đề mục theo dàn ý nhóm đã chọn (Phần 1: bộ dữ liệu và tiền xử lý; Phần 2: bộ thuộc tính).
5. **Nội dung theo nguyên tắc 80–20:** mỗi mục trả lời đúng những câu thầy hay hỏi: là gì, vì sao chọn con số này (kể cả min/max, miền giá trị), công thức, ví dụ tính tay, giá trị thông tin.
6. **Không đề cập râu, kính, khẩu trang, phụ kiện** trong báo cáo. Nhóm đã quyết định bộ dữ liệu có ảnh để râu là bình thường, không coi là lỗi dữ liệu.
7. **Chỉ dựa tài liệu nhóm cung cấp, không bịa.** Trong thân báo cáo **không chèn ký hiệu nguồn** (`[TKn tr.x]`, `[cần nguồn]`); nguồn chỉ liệt kê ở mục Tài liệu tham khảo cuối bài. Chỗ chưa có thông tin thì để `[điền]`. Phần mô tả bộ dữ liệu viết chung, không nêu tên thư mục hay tên tệp ảnh.
10. **Mục mô tả bộ dữ liệu:** hình minh họa là **ảnh gốc đầy đủ** (chưa cắt, chưa lọc nền); ảnh đã xử lý chỉ để show khi demo. Không viết lý thuyết ảnh số (ma trận, số bit, dung lượng chưa nén) vì đó là nội dung môn Xử lý ảnh. Không đưa số liệu thống kê khó kiểm chứng (phần trăm điểm da, tỉ lệ khung mặt trong ảnh…); phần giống/khác viết chung, ngắn.
8. Giữ **vector 39 chiều** (5 hình học + 2 kết cấu + 32 màu). Tên cột trong `trich_rut.TEN_COT` dùng chung cho báo cáo và CSDL; không đổi khi chưa hỏi cả nhóm. Không dùng học sâu (CNN, embedding).
9. Trả lời bằng tiếng Việt, ngắn gọn.

## 3. Pipeline hiện tại
`data/anh256/` (1005 ảnh 256×256 .jpg) → `tien_xu_ly.py` → `trich_rut.py` → `csdl.py` (SQLite) → `tim_kiem.py` → `giao_dien.py`. Chạy tất cả bằng `chay_toan_bo.ipynb`; đánh giá bằng `danh_gia.py`.

| Bước | Việc làm | Cách làm |
|---|---|---|
| B1 | Đọc, kiểm tra tệp đúng là JPG (2 byte đầu FF D8), 256×256×3 | tự code phần kiểm tra; OpenCV chỉ để giải nén |
| B2 | Ảnh xám Y = 0,299R + 0,587G + 0,114B | tự code |
| B3 | Độ nét = phương sai Laplace, ngưỡng 20 | tự code |
| B4 | Phát hiện khuôn mặt + 68 điểm mốc; kiểm tra ổn định: dò lại trên ảnh kéo giãn dọc 1,25 lần, độ lệch < 0,08 | dlib; tự code phần kiểm tra |
| B5 | Căn chỉnh: xoay + co giãn + tịnh tiến, hai mắt nằm ngang, cách nhau 64 px, trung điểm ở (128; 100) | tự code, nội suy song tuyến tính |
| B6 | Lọc nền bằng mặt nạ đa giác (viền hàm + lông mày dịch lên 0,6 × khoảng cách mắt) | tự code |

Tiền xử lý **6 bước** (chốt 14/10/2026, gộp phát hiện mặt với 68 điểm mốc; "lưu" không tính là bước). Sau B6, `tien_xu_ly.chay()` lưu `data/tien_xu_ly/` (ảnh màu đã xử lý `.jpg` chỉ để xem, `diem_moc.csv`, `nhat_ky.csv`).

**Chỉ dùng một định dạng JPG** (chốt 12/10/2026, tránh bị hỏi "sao vừa JPG vừa PNG"). Vì JPG nén có tổn thất, `trich_rut.chay()` không đọc lại ảnh trong `data/tien_xu_ly/` mà gọi `tien_xu_ly.xu_ly_mot_anh` trên ảnh gốc rồi trích đặc trưng ngay, đúng đường đi của ảnh truy vấn.

- **Hình học (5):** tỉ lệ khung mặt d(0,16)/dY(27,8); gò má/hàm d(0,16)/d(4,12); khoảng cách mắt/rộng mặt; rộng cánh mũi d(31,35)/khoảng cách mắt; cao mũi dY(27,33)/dY(27,8).
- **Kết cấu (2):** phương sai LBP vùng cằm – miệng (điểm 33, 54, 12, 8, 4, 48); mật độ lông mày (Otsu + LBP khác 0 và 255).
- **Màu (32):** lược đồ H × S trên mặt nạ da (2 má + trán); H chia 8 ô trong 0–40°, S chia 4 ô.
- **Tìm kiếm (phương án A, chốt 21/10/2026):** chuẩn hóa min-max (min, max đọc từ `data/min_max.json`); khoảng cách Euclid tính riêng từng bộ, chia √(số chiều), trọng số 1/3 (hằng số `tim_kiem.TRONG_SO`); S = 1/(1+D); lấy top-5 (`tim_kiem.xep_hang`). Phép thử 8040 ảnh biến đổi: hạng 1 96,4%, top-5 99,0%.
- **CSDL (chốt 24/11/2026: rút từ 4 về 2 bảng):** SQLite `data/khuon_mat.db`, 2 bảng `anh` (PK ma_anh, UNIQUE duong_dan; đặc trưng tổng quát) 1–1 `dac_trung` (ma_anh vừa PK vừa FK, 39 cột REAL giá trị gốc; đặc trưng nội dung). Lý do: thầy (tuần 5) "không có bảng nào đứng riêng" — 2 bảng phụ cũ `thuoc_tinh`/`bo_thuoc_tinh` mô tả cột, không nối khóa ngoại/JOIN được với `dac_trung`. min, max ở `min_max.json` (`kiem_tra` đối chiếu với SELECT MIN(), MAX()); tên, vai trò X0–X38 là hằng số `trich_rut.TEN_DAC_TRUNG`, `trich_rut.VAI_TRO`. `PRAGMA foreign_keys = ON` mỗi lần kết nối; `csdl.kiem_tra()` chạy các câu lệnh kiểm tra toàn vẹn; `them_anh()` nạp ảnh mới trong một giao dịch, min/max giữ cố định.
- **App:** `giao_dien.py` (Tkinter): Upload ảnh (ảnh chưa 256×256 được `tien_xu_ly.lam_vuong` cắt vuông quanh mặt, cạnh = 2 × khung mặt, không kéo giãn) / Chọn ảnh trong CSDL (bỏ chính nó). Màn hình demo (18/11, theo gợi ý của thầy "cần thấy kết quả trung gian"): (1) dải ảnh B1→B6 kèm số liệu kiểm tra; (2) bảng truy vấn – top-5: 7 đặc trưng (x, x' sau min-max, |q'−x'|, ô càng xanh càng gần), lược đồ màu 32 ô, D từng bộ, D, S, hạng; bấm ảnh kết quả → cửa sổ thế số từng bước (chuẩn hóa → D bộ → D → S → xếp hạng → nhận xét đặc trưng tương đồng / khác biệt theo `trich_rut.VAI_TRO`); (3) biểu đồ đóng góp 3 bộ vào D, các lệnh SQL đã chạy. Ô tùy chọn vẽ đường đo X0–X4 và vùng X5, X6, mặt nạ da. Thanh trượt trọng số chỉ để thử (mặc định 1/3 = `tim_kiem.TRONG_SO`).

## 4. Tệp và thư mục
Code và dữ liệu nằm ở thư mục gốc (đường dẫn trong code không đổi). Ba thư mục `bao_cao/`, `tai_lieu/`, `tai_lieu_thay/` không đưa lên git.

| Đường dẫn | Nội dung |
|---|---|
| `xu_ly_anh.py` | các phép xử lý tự code bằng numpy |
| `tien_xu_ly.py`, `trich_rut.py`, `tim_kiem.py` | tiền xử lý 6 bước, trích đặc trưng, tìm kiếm |
| `csdl.py`, `giao_dien.py`, `danh_gia.py` | CSDL SQLite 2 bảng, giao diện Tkinter, phép thử ảnh biến đổi |
| `data/khuon_mat.db`, `data/danh_gia.csv`, `data/truy_van_mau/` | CSDL, kết quả phép thử, ảnh dùng thử khi demo |
| `chay_toan_bo.ipynb` | chạy toàn bộ, tạo hình vào `figs/` (kể cả 2 sơ đồ khối Hình 1.3, 2.1) |
| `data/anh256/` | 1005 ảnh chuẩn man_0001 … man_1005 (21/10/2026 xóa ảnh trùng man_0827 cũ, đánh số lại) |
| `data/ten_goc.csv` | mã ảnh ↔ tên tệp gốc |
| `data/dac_trung.csv`, `data/min_max.json` | vector đặc trưng gốc và tham số chuẩn hóa |
| `data/tien_xu_ly/` | kết quả tiền xử lý (sinh lại được) |
| `models/` | file dlib `shape_predictor_68_face_landmarks.dat` (95 MB, tải về từ dlib.net) |
| `bao_cao/Bao-cao-1-Nhom19-theo-mau-v6.docx` | **báo cáo 1 bản nộp** (nhóm sửa tay, không dựng lại từ script) |
| `bao_cao/Bao-cao-2-Nhom19.docx`, `bao_cao/build_bao_cao_2.js` | báo cáo 2 và script dựng (`node build_bao_cao_2.js`) |
| `bao_cao/ban_giao_bc2/` | bàn giao BC2, script thử 3 phương án tìm kiếm (`thu_bien_the.py`, `chon_phuong_an.py`), chụp app (`chup_app.py`), xóa ảnh trùng (`xoa_anh_trung.py`) |
| `bao_cao/build_bao_cao_1.js` | script dựng báo cáo 1 (toàn bộ nội dung nằm trong script). Chạy: `cd bao_cao`, `npm install docx` (một lần), `node build_bao_cao_1.js`. Sửa nội dung thì sửa script rồi dựng lại; nếu ai đã sửa tay tệp .docx thì so text trước khi ghi đè |
| `bao_cao/nhap/` | bản nháp cũ của Minh/Nam, `anh_da_thay.csv` (danh sách ảnh đã thay, không đưa vào báo cáo) |
| `tai_lieu/slide/XLA/` | 8 slide môn Xử lý ảnh số: Laplacian, gradient, Otsu, phân vùng (ch.5); lược đồ xám (ch.3); hệ màu HSV, JPG/PNG (ch.4); JPEG/DCT (ch.6); hàm phân lớp tuyến tính (ch.8) |
| `tai_lieu/slide/DPT Tiếng Việt/` | 12 slide Hệ CSDL ĐPT của thầy Hóa; quan trọng nhất bài 11 (truy vấn ảnh), bài 8 (L1/L2-norm, phân cụm), bài 1 (precision/recall), bài 2, 3, 5–7 (mô hình dữ liệu, cây k-d/R) |
| `tai_lieu/sach/dpt 2-2011.pdf` | giáo trình Đỗ Trung Tuấn (tr.127–132: hàm khoảng cách, CSDL ảnh quan hệ) |
| `tai_lieu/sach/Book-Multimedia_..._(Artech House).pdf` | Guojun Lu (1999). Tiền xử lý trước khi trích đặc trưng (tr.7, 64), ảnh xám/màu là mảng 2 chiều (tr.28–29), nền lấn át tiền cảnh, tách tiền cảnh – nền (tr.137), bất biến tịnh tiến/xoay/tỉ lệ (tr.142), chuẩn hóa xoay và tỉ lệ (tr.148–149, 152), RGB không hợp cho tìm kiếm (tr.138–139), kết hợp nhiều đặc trưng (tr.131), lược đồ màu + L1/Euclid (tr.133–134). Không có Laplace, phát hiện mặt, điểm mốc, nội suy, chẵn–lẻ. Chương 9 (cấu trúc chỉ mục) chưa đọc – dùng cho báo cáo 2 |
| `tai_lieu/bao_cao_mau/` | báo cáo ảnh hoa khóa trước: bản 8,5đ (bản mẫu trình bày) và bản 7đ |
| `tai_lieu_thay/` | tóm tắt bài giảng tuần 2–7, bản đồ 28 câu hỏi mẫu, file review câu hỏi D22, `PHAN_TICH_THAY_HOA.md`, danh sách lớp. Nhãn [Thầy], [Slide] dẫn được; nhãn [Claude] là suy luận |

Ảnh nhỏ hơn 256×256 hoặc chưa vuông: cắt vuông quanh khuôn mặt, **không kéo giãn**, rồi đưa về 256×256.

## 5. Quy tắc viết code
- Code đơn giản, ít tệp, chú thích tiếng Việt, ghi công thức ngay trong docstring.
- Phép xử lý mới ưu tiên tự code bằng numpy. Chỉ dùng OpenCV/dlib khi tự code quá phức tạp, và khi đó ghi bản chất vào docstring.
- Giảng viên thích code tay, hay hỏi ánh xạ công thức ↔ dòng code, hỏi "tại sao" cho mọi con số, và sẽ hỏi sâu mô hình CSDL (chọn gì, vì sao, lệnh tạo bảng).

## 6. Vấn đáp (tóm tắt)
Thầy hỏi theo ba mạch: dữ liệu, CSDL, đặc trưng và cách so sánh hai ảnh. Ngập ngừng quá khoảng 15 giây là bị dừng. Show ảnh **đã xử lý**, không show ảnh thô. Mỗi người phải biết toàn bộ project.

## 7. Việc còn lại
- **Báo cáo 1 (nộp 22/10):**
  - điền nguồn ảnh (`[điền nguồn cụ thể]` ở mục 1 Phần 1);
  - bản v5 (16/10) cần cả nhóm đọc duyệt, mở bằng Word và bấm cập nhật mục lục;
  - (đã xử lý 16/10) Hình 1.2 vẽ lại, bỏ ảnh có cạnh trắng (`vien_trang` trong notebook); câu mô tả đổi thành "làm vuông mà không kéo giãn". Hệ số 0,6, 1,25, S = 4 ô đã có lý do trong báo cáo và docstring. Ảnh truy vấn qua đủ các bước kiểm tra như ảnh trong CSDL (bỏ tham số `kiem_tra_net`); 1006 ảnh vẫn đạt, `dac_trung.csv` không đổi. Hai bước kiểm tra không loại ảnh nào: báo cáo giải thích là chặn ảnh nạp thêm và ảnh truy vấn;
  - nhiều ảnh có râu (nhóm chấp nhận, báo cáo không nhắc) → chuẩn bị câu trả lời miệng.
- **Báo cáo 2:** đã có bản nháp đầy đủ `bao_cao/Bao-cao-2-Nhom19.docx` (dựng bằng `bao_cao/build_bao_cao_2.js`, 21/10/2026): PHẦN 3 (trích rút, CSDL), PHẦN 4 (sơ đồ khối, quy trình, kết quả trung gian, đánh giá, demo). Cần cả nhóm đọc duyệt, mở Word cập nhật mục lục.
- **Báo cáo 1:** v6 vẫn ghi 1006 ảnh, phải sửa thành 1005 (việc của chat sau).

## 8. Bàn giao (chat ngày 12–16/10/2026)
**Cách làm việc đang dùng:** duyệt báo cáo 1 từng mục. Với mỗi mục: Claude giải thích nội dung, tự lấy câu hỏi thầy từng hỏi (D22, bản đồ câu hỏi, tóm tắt bài giảng) và tự trả lời, chỉ ra chỗ yếu; user chốt rồi Claude sửa script và dựng lại báo cáo. Bản hiện tại là **v5** (`bao_cao/Bao-cao-1-Nhom19-theo-mau-v5.docx`).

**Đã chốt trong chat này:**
- Không ký hiệu nguồn trong thân bài; nguồn chỉ ở danh mục cuối. Đã bỏ 3 bài báo nhóm không có tệp (Ojala, Kazemi–Sullivan, Dalal–Triggs), thêm sách Guojun Lu.
- Mục 1 Phần 1 viết gộp, chung; Hình 1.1, 1.2 dùng **ảnh gốc đầy đủ** (ảnh đã xử lý chỉ show khi demo); không lý thuyết môn Xử lý ảnh (RGB 8 bit, dung lượng); không số liệu khó kiểm chứng.
- **Chỉ dùng JPG.** `trich_rut.chay()` trích đặc trưng thẳng từ `tien_xu_ly.xu_ly_mot_anh` trên ảnh gốc (như ảnh truy vấn), không đọc lại ảnh đã lưu. Đã chạy lại: `data/dac_trung.csv` giống hệt từng byte bản trước.
- **Tiền xử lý 6 bước** (gộp phát hiện mặt + 68 điểm mốc; "lưu" không là bước). Code, notebook, Hình 1.3, 1.4, 2.1 đã đổi theo.
- ~~Mỗi bước tiền xử lý viết theo khuôn Khái niệm – Nguyên lý (≤ 30 từ) – Công thức~~ → **thay ngày 16/10/2026**: user thấy bản v3 "khá AI, cụt ngủn". Bản v4 viết câu đầy đủ theo văn phong các báo cáo cũ của Vinh (`tai_lieu/bao_cao_mau/lối hành văn cho bớt AI của tác giả/`) và bản mẫu 7đ: mỗi bước có Mục đích – Cách làm – Công thức – câu hỏi "Tại sao…?" cho các con số; kỹ thuật có Ý tưởng – Cách làm – Ưu điểm – Nhược điểm. Vẫn không ví dụ tính tay ở tiền xử lý.
- Văn phong: dùng skill `humanizer-vi` (và `translationese-cleaner-vi`, `style-guide-vi`, `grammar-checker-vi`) cài ở `.claude/skills/`, lấy từ github.com/longhang2004/vietnamese-humanizer. Tránh chuỗi dấu ";", "đồng thời/qua đó/từ đó", "đóng vai trò… trong việc", "được… bởi", nhiều câu liền nhau cùng mở đầu bằng "Nhóm".
- Bản v5 (16/10/2026, script `bao_cao/build_bao_cao_1.js`; v4, v3 giữ ở `build_bao_cao_1_v4.js`, `_v3.js`):
  + bìa theo mẫu báo cáo CSDL phân tán của Vinh (viền trang thin–thick chỉ ở trang đầu, logo PTIT `bao_cao/logo_ptit.jpg`, danh sách thành viên), lời nói đầu trang trọng; lề trái 3 cm, phải 1,5 cm, trên/dưới 2 cm;
  + đánh số thống nhất "1." → "1.1," → "1.1.1,", mọi mục là đề mục thật, mục lục 4 cấp; Phần 2 bỏ "I, II";
  + công thức là phương trình Word (OMML), chữ tiếng Việt để ngoài phương trình (bản v4 lỗi phông vì gõ công thức bằng chữ Cambria Math lẫn tiếng Việt và ký tự chỉ số Unicode); mỗi bước tiền xử lý tối đa 1 công thức, bỏ các câu giải thích phụ ("ba trọng số cộng lại bằng 1"…) vì thầy càng thấy chi tiết càng hỏi;
  + mỗi đặc trưng theo khuôn Công thức → Miền giá trị → Giá trị thông tin (tương đồng hay khác biệt, ứng với nhận xét nào ở mục 1.3 Phần 1) → Ví dụ; X0, X1 và X7–X38 tìm tương đồng, X2–X6 tìm khác biệt;
  + hình 68 điểm mốc chuyển về Phần 1 (Hình 1.5); thêm Hình 1.6 (30 ảnh sau tiền xử lý, `figs/fig_prep30.png`); Phần 2 còn Hình 2.1, 2.2;
  + không đưa ý của bản nháp Minh/Nam vào mục "kỹ thuật không dùng" (user 16/10).
- Nguồn công thức: Laplace ở slide Xử lý ảnh bài 5 tr.28–29; Otsu bài 5 tr.65–67; HSV tách độ sáng bài 4 tr.36–40. Chưa có nguồn trong tài liệu nhóm: trọng số ảnh xám 0,299/0,587/0,114, phương sai Laplace làm độ nét, nội suy song tuyến tính, quy tắc chẵn – lẻ.
- Khi trả lời thầy: tránh chữ "mô hình nhận diện" (bài toán là tìm kiếm tương đồng); lọc nền gán **trắng 255**, không phải 0; đặc trưng dùng ảnh màu đã lọc nền + ảnh xám + 68 điểm mốc.

**Các thay đổi code chưa commit:** `xu_ly_anh.py`, `tien_xu_ly.py` (thêm `la_jpg`, bỏ `kiem_tra_net`), `tim_kiem.py`, `trich_rut.py`, `chay_toan_bo.ipynb` (ô vẽ Hình 1.2 mới và Hình 1.6), `figs/*`, `.gitignore`, `README.md`, `CLAUDE.md`, `HUONG_DAN_CHUNG.md` (chưa có trên git). Notebook chưa được chạy lại trong Jupyter nên ô kết quả đang trống.
