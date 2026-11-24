"""
search.py - Chuẩn hóa đặc trưng và tìm 5 ảnh giống ảnh truy vấn nhất (Báo cáo 2, Phần 4).

- Quy trình tìm kiếm:
  + 1. Chuẩn hóa min-max từng chiều: x' = (x - min) / (max - min)
       (min, max tính một lần trên toàn CSDL, lưu ở data/norm_params.json)
  + 2. Khoảng cách từng bộ: D_bộ = căn(tổng (q'_i - x'_i)^2) / căn(n_bộ), n = 5, 2, 32
  + 3. Khoảng cách chung: D = (D_hình_học + D_kết_cấu + D_màu_sắc) / 3   (TRONG_SO = 1/3)
  + 4. Độ tương đồng: S = 1 / (1 + D)   (D = 0 thì S = 1; D càng lớn S càng nhỏ)
  + 5. Sắp xếp S giảm dần, lấy 5 ảnh đầu.
- Vector đặc trưng đọc từ bảng dac_trung của CSDL SQLite (database.py).
- Chạy:  python search.py duong_dan_anh_truy_van.jpg
"""
import sys
import json
import numpy as np

import common as C
import preprocess
import features as F
import database as DB

TOP_K = 5
TRONG_SO = {"hinh_hoc": 1 / 3, "ket_cau": 1 / 3, "mau_sac": 1 / 3}    # Báo cáo 2, Phần 4 mục 2.4


def load_features():
    """
    - File: search.py. Dùng để: đọc vector đặc trưng của mọi ảnh từ CSDL.
    - Dựa trên: lệnh SELECT trên bảng dac_trung (database.doc_dac_trung)
      + mở kết nối, đọc xong thì đóng kết nối (khối try – finally bảo đảm luôn đóng)
      + trả về (danh sách mã ảnh, ma trận N x 39 giá trị gốc)
    - Ứng dụng: dữ liệu để so khớp với ảnh truy vấn, tính min, max.
    - Cách khác: đọc thẳng features.csv (không qua CSDL, không có ràng buộc toàn vẹn).
    """
    con = DB.ket_noi()
    try:
        return DB.doc_dac_trung(con)
    finally:
        con.close()


def fit_norm(X):
    """
    - File: search.py. Dùng để: tính min, max của từng đặc trưng và lưu lại (chạy một lần).
    - Dựa trên: tham số của chuẩn hóa min-max
      + min, max lấy theo từng cột (axis=0) trên cả 1005 ảnh
      + lưu vào data/norm_params.json kèm tên 39 cột
    - Ứng dụng: ảnh truy vấn được chuẩn hóa theo đúng tham số đã dùng cho CSDL (BC1 mục 2.4).
    - Cách khác: chuẩn hóa Z-score, cần lưu trung bình và độ lệch chuẩn.
    """
    mn, mx = X.min(axis=0), X.max(axis=0)
    with open(C.NORM_JSON, "w", encoding="utf-8") as f:
        json.dump({"ten": F.FEATURE_NAMES, "min": mn.tolist(), "max": mx.tolist()}, f, indent=1)
    return mn, mx


def load_norm():
    """
    - File: search.py. Dùng để: đọc lại min, max đã lưu của 39 đặc trưng.
    - Dựa trên: tệp data/norm_params.json (database.doc_min_max)
      + min, max giữ cố định, không tính lại theo từng truy vấn
    - Ứng dụng: chuẩn hóa vector truy vấn và vector trong CSDL theo cùng một thang,
      dùng ở xep_hang, danh_gia.py và notebook.
    - Cách khác: tính lại SELECT MIN(), MAX() mỗi lần, nhưng thêm ảnh mới sẽ làm đổi thang
      của mọi ảnh.
    """
    return DB.doc_min_max()


def load_weights():
    """
    - File: search.py. Dùng để: lấy trọng số của 3 bộ thuộc tính.
    - Dựa trên: hằng số TRONG_SO, mỗi bộ 1/3
      + ba bộ đã cùng thang [0; 1], không có căn cứ để coi bộ nào quan trọng hơn
      + trả về bản sao để nơi gọi có sửa cũng không đổi hằng số gốc
    - Ứng dụng: gộp D_hình_học, D_kết_cấu, D_màu_sắc thành D.
    - Cách khác: đã thử màu 0,5 và tìm 2 tầng, không tốt hơn rõ rệt (BC2 mục 4.1).
    """
    return dict(TRONG_SO)


def normalize(X, mn, mx):
    """
    - File: search.py. Dùng để: đưa mọi đặc trưng về cùng thang [0; 1].
    - Dựa trên: chuẩn hóa min-max
      + x' = (x - min) / (max - min); chiều có max = min thì cho bằng 0
      + kẹp về [0; 1]: ảnh truy vấn nhỏ hơn min thành 0, lớn hơn max thành 1
    - Ứng dụng: X5 cỡ hàng nghìn không lấn át các đặc trưng cỡ 1; một giá trị lệch thường
      không đẩy khoảng cách vượt [0; 1].
    - Cách khác: Z-score (không có miền cố định).
    """
    rong = mx - mn                                   # độ rộng khoảng giá trị của từng chiều
    chia = np.where(rong == 0, 1.0, rong)            # tránh chia cho 0
    x_moi = (X - mn) / chia * (rong != 0)            # chiều có max = min thì nhân 0
    return np.clip(x_moi, 0.0, 1.0)                  # kẹp về [0; 1]


def distances(q, Xn, trong_so):
    """
    - File: search.py. Dùng để: tính khoảng cách từ ảnh truy vấn tới từng ảnh trong CSDL.
    - Dựa trên: khoảng cách Euclid (L2, slide bài 8 tr.17) tính riêng từng bộ
      + D_bộ = căn(tổng (q'_i - x'_i)^2) / căn(n_bộ), đưa cả ba bộ về [0; 1]
      + D = tổng trọng_số_bộ * D_bộ
    - Ứng dụng: bộ màu 32 chiều không lấn át bộ kết cấu 2 chiều; D từng bộ cho biết giống ở đâu.
    - Cách khác: cosine (bỏ qua độ lớn), Manhattan L1.
    """
    d_bo = {}
    for ten, vt in F.BO.items():                     # vt: vị trí các cột của bộ trong vector 39 chiều
        hieu = Xn[:, vt] - q[vt]                     # hiệu từng đặc trưng, mọi ảnh cùng lúc
        so_chieu = hieu.shape[1]
        d_bo[ten] = np.sqrt((hieu ** 2).sum(axis=1)) / np.sqrt(so_chieu)
    D = sum(trong_so[ten] * d_bo[ten] for ten in F.BO)
    return D, d_bo


def xep_hang(v, bo_qua=None, trong_so=None):
    """
    - File: search.py. Dùng để: so vector truy vấn với mọi ảnh, giữ kết quả trung gian.
    - Dựa trên: chuẩn hóa -> khoảng cách -> S = 1/(1 + D) -> sắp xếp
      + argsort(-S) sắp S giảm dần, "stable" giữ thứ tự cũ khi S bằng nhau
      + bo_qua: mã ảnh không xét (khi lấy chính ảnh trong CSDL làm truy vấn)
      + trong_so: None dùng 1/3, khác None chỉ để thử trên giao diện
    - Ứng dụng: giao diện cần cả X, Xn, D từng bộ để thế số.
    - Cách khác: cây k-d, phân cụm (BC2 mục 2.5).
    """
    ids, X = load_features()
    mn, mx = load_norm()
    if trong_so is None:
        trong_so = load_weights()
    q, Xn = normalize(v, mn, mx), normalize(X, mn, mx)
    D, d_bo = distances(q, Xn, trong_so)
    S = 1.0 / (1.0 + D)
    con_lai = np.array([m != bo_qua for m in ids])                 # True = ảnh được xét
    thu_tu = np.argsort(-S, kind="stable")                         # chỉ số ảnh, S giảm dần
    top = [i for i in thu_tu if con_lai[i]][:TOP_K]
    return dict(ids=ids, X=X, mn=mn, mx=mx, Xn=Xn, q=q, trong_so=trong_so, d_bo=d_bo, D=D, S=S,
                top=top, con_lai=con_lai)


def search_vector(v, bo_qua=None):
    """
    - File: search.py. Dùng để: lấy 5 ảnh giống nhất với một vector đặc trưng gốc.
    - Dựa trên: gọi xep_hang rồi rút gọn kết quả
      + mỗi ảnh: ma_anh, S, D và D từng bộ (D_hinh_hoc, D_ket_cau, D_mau_sac)
      + danh sách xếp theo S giảm dần
    - Ứng dụng: dùng cho search_image, phép thử đánh giá, notebook in bảng kết quả.
    - Cách khác: trả cả 1005 ảnh rồi cắt sau (tốn bộ nhớ không cần thiết).
    """
    xh = xep_hang(v, bo_qua)
    ket_qua = []
    for i in xh["top"]:
        ket_qua.append({"ma_anh": xh["ids"][i], "S": float(xh["S"][i]), "D": float(xh["D"][i]),
                        "D_hinh_hoc": float(xh["d_bo"]["hinh_hoc"][i]),
                        "D_ket_cau": float(xh["d_bo"]["ket_cau"][i]),
                        "D_mau_sac": float(xh["d_bo"]["mau_sac"][i])})
    return ket_qua


def search_image(path):
    """
    - File: search.py. Dùng để: tìm kiếm với một ảnh mới từ đầu đến cuối.
    - Dựa trên: đúng đường đi của ảnh trong CSDL
      + tiền xử lý 6 bước (kể cả các bước kiểm tra) -> trích 39 đặc trưng -> so với CSDL
      + ảnh không đạt tiền xử lý thì vector và top-5 là None, lý do ở kq["ly_do"]
    - Ứng dụng: lệnh python search.py anh.jpg, notebook.
    - Cách khác: bỏ bước kiểm tra cho ảnh truy vấn (dễ cho kết quả sai mà không báo).
    """
    kq = preprocess.process_one(path)
    if not kq["dat"]:
        return kq, None, None
    v = F.extract(kq["rgb"], kq["gray"], kq["pts"])
    return kq, v, search_vector(v)


if __name__ == "__main__":
    kq, v, top = search_image(sys.argv[1])
    if top is None:
        print("Anh truy van khong dat tien xu ly:", kq["ly_do"])
    else:
        for i, r in enumerate(top, start=1):
            print(f"{i}. {r['ma_anh']}  S={r['S']:.4f}  D={r['D']:.4f}  "
                  f"(hinh hoc {r['D_hinh_hoc']:.4f}, ket cau {r['D_ket_cau']:.4f}, mau {r['D_mau_sac']:.4f})")
