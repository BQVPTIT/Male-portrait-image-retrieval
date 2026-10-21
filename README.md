# Male-portrait-image-retrieval

Hệ CSDL lưu trữ và tìm kiếm ảnh chân dung đàn ông (top-5 ảnh tương đồng) – BTL Hệ CSDL Đa phương tiện, Nhóm 19, D23 PTIT.

## Cấu trúc

```
common.py        các phép xử lý ảnh tự code bằng numpy
preprocess.py    tiền xử lý 6 bước: đọc → xám → độ nét → mặt + 68 điểm mốc → căn chỉnh → lọc nền;
                 làm vuông ảnh truy vấn chưa đúng 256×256
features.py      vector đặc trưng 39 chiều (5 hình học + 2 kết cấu + 32 màu)
database.py      CSDL SQLite 4 bảng (anh, dac_trung, bo_thuoc_tinh, thuoc_tinh): tạo, nạp, kiểm tra, thêm ảnh
search.py        chuẩn hóa min-max, khoảng cách từng bộ, S = 1/(1+D), top-5 (đọc dữ liệu từ CSDL)
danh_gia.py      phép thử ảnh biến đổi (8 kiểu × 1005 ảnh) → data/danh_gia.csv
app.py           giao diện demo Tkinter
run_all.ipynb    chạy toàn bộ và vẽ hình vào figs/

data/raw256/     1005 ảnh 256×256 .jpg (man_0001 … man_1005)
data/            features.csv, norm_params.json, name_map.csv, khuon_mat.db (CSDL), danh_gia.csv;
                 prep/ sinh lại được khi chạy; truy_van_mau/ ảnh dùng thử khi demo
figs/            hình dùng trong báo cáo
models/          shape_predictor_68_face_landmarks.dat (tải từ dlib.net, không đưa lên git)
```

## Chạy

```
pip install -r requirements.txt
python database.py                    # tạo lại CSDL từ features.csv + norm_params.json và kiểm tra toàn vẹn
python app.py                         # giao diện: Upload ảnh / Chọn ảnh trong CSDL
python search.py duong_dan_anh.jpg    # tìm 5 ảnh giống một ảnh mới (in ra màn hình)
python danh_gia.py                    # phép thử ảnh biến đổi (khoảng 8 phút)
jupyter notebook run_all.ipynb        # chạy lần lượt từ trên xuống
```

Trên máy nhóm dùng Python của môi trường ảo: `venv310\Scripts\python.exe app.py`.
