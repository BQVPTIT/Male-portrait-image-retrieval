# Male-portrait-image-retrieval

Hệ CSDL lưu trữ và tìm kiếm ảnh chân dung đàn ông (top-5 ảnh tương đồng) – BTL Hệ CSDL Đa phương tiện, Nhóm 19, D23 PTIT.

## Cấu trúc

```
common.py        các phép xử lý ảnh tự code bằng numpy
preprocess.py    tiền xử lý 6 bước: đọc → xám → độ nét → mặt + 68 điểm mốc → căn chỉnh → lọc nền
features.py      vector đặc trưng 39 chiều (5 hình học + 2 kết cấu + 32 màu)
search.py        chuẩn hóa min-max, khoảng cách, top-5
run_all.ipynb    chạy toàn bộ và vẽ hình vào figs/

data/raw256/     1006 ảnh 256×256 .jpg (man_0001 … man_1006)
data/            features.csv, norm_params.json, name_map.csv; prep/ sinh lại được khi chạy
figs/            hình dùng trong báo cáo
models/          shape_predictor_68_face_landmarks.dat (tải từ dlib.net, không đưa lên git)
```

## Chạy

```
pip install -r requirements.txt
jupyter notebook run_all.ipynb        # chạy lần lượt từ trên xuống
python search.py duong_dan_anh.jpg    # tìm 5 ảnh giống một ảnh mới
```
