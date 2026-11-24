"""
search.py - Chuẩn hóa đặc trưng và tìm 5 ảnh giống ảnh truy vấn nhất.

    1. Chuẩn hóa min-max từng chiều:  x' = (x - min) / (max - min)
       (min, max tính một lần trên toàn bộ CSDL, lưu ở data/norm_params.json để dùng lại cho ảnh truy vấn)
    2. Khoảng cách riêng cho từng bộ thuộc tính, chia căn số chiều để 3 bộ cùng thang:
           D_bộ = sqrt( tổng_i (q_i - x_i)^2 ) / sqrt(n_bộ)       n = 5, 2, 32
    3. Khoảng cách chung:  D = (D_hình_học + D_kết_cấu + D_màu_sắc) / 3     (trọng số TRONG_SO = 1/3)
    4. Độ tương đồng:      S = 1 / (1 + D)   (D = 0 -> S = 1; D càng lớn S càng gần 0)
    5. Sắp xếp S giảm dần, lấy 5 ảnh đầu.

Vector đặc trưng đọc từ bảng dac_trung của CSDL SQLite (database.py).

Chạy:  python search.py duong_dan_anh_truy_van.jpg
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
    """Đọc bảng dac_trung -> (danh sách mã ảnh, ma trận N x 39 giá trị gốc)."""
    con = DB.ket_noi()
    try:
        return DB.doc_dac_trung(con)
    finally:
        con.close()


def fit_norm(X):
    """Tính min, max từng chiều trên toàn bộ CSDL và lưu vào norm_params.json (chạy một lần)."""
    mn, mx = X.min(axis=0), X.max(axis=0)
    with open(C.NORM_JSON, "w", encoding="utf-8") as f:
        json.dump({"ten": F.FEATURE_NAMES, "min": mn.tolist(), "max": mx.tolist()}, f, indent=1)
    return mn, mx


def load_norm():
    """Đọc min, max của 39 chiều từ data/norm_params.json."""
    return DB.doc_min_max()


def load_weights():
    """Trọng số của 3 bộ thuộc tính (1/3 mỗi bộ)."""
    return dict(TRONG_SO)


def normalize(X, mn, mx):
    """Chuẩn hóa min-max: x' = (x - min) / (max - min), rồi kẹp vào [0, 1]. Chiều có max = min thì cho bằng 0.

    Ảnh trong CSDL luôn cho x' trong [0, 1]. Ảnh truy vấn có giá trị nhỏ hơn min hoặc lớn hơn max
    của CSDL (ví dụ ảnh bị chỉnh màu quá đậm) thì x' được kẹp về 0 hoặc 1:
        x' = min(1, max(0, x'))
    Nhờ vậy một đặc trưng lệch thường không kéo khoảng cách của cả bộ vượt khỏi [0, 1].
    """
    rong = mx - mn
    return np.clip((X - mn) / np.where(rong == 0, 1.0, rong) * (rong != 0), 0.0, 1.0)


def distances(q, Xn, trong_so):
    """Khoảng cách từ vector truy vấn q (đã chuẩn hóa) tới từng hàng của Xn (đã chuẩn hóa).

    Trả về (D chung, dict D từng bộ). D_bộ = Euclid / sqrt(số chiều của bộ);
    D = tổng trọng_số_bộ * D_bộ.
    """
    d_bo = {}
    for ten, vt in F.BO.items():
        hieu = Xn[:, vt] - q[vt]
        d_bo[ten] = np.sqrt((hieu ** 2).sum(axis=1)) / np.sqrt(hieu.shape[1])
    D = sum(trong_so[ten] * d_bo[ten] for ten in F.BO)
    return D, d_bo


def xep_hang(v, bo_qua=None, trong_so=None):
    """So vector gốc v (39 chiều) với MỌI ảnh trong CSDL, giữ lại các kết quả trung gian.

    trong_so: None -> TRONG_SO (1/3 mỗi bộ). Truyền dict khác chỉ để THỬ trên giao diện (app.py).
    Trả về dict: ids, X (vector gốc của CSDL), mn, mx, Xn (vector CSDL đã chuẩn hóa), q (v đã chuẩn
    hóa), trong_so, d_bo (D từng bộ), D, S (mảng cho mọi ảnh), top (chỉ số TOP_K ảnh có S lớn nhất,
    giảm dần), con_lai (mặt nạ các ảnh được xét, tức trừ ảnh bo_qua).
    """
    ids, X = load_features()
    mn, mx = load_norm()
    if trong_so is None:
        trong_so = load_weights()
    q, Xn = normalize(v, mn, mx), normalize(X, mn, mx)
    D, d_bo = distances(q, Xn, trong_so)
    S = 1.0 / (1.0 + D)
    con_lai = np.array([m != bo_qua for m in ids])
    top = [i for i in np.argsort(-S, kind="stable") if con_lai[i]][:TOP_K]
    return dict(ids=ids, X=X, mn=mn, mx=mx, Xn=Xn, q=q, trong_so=trong_so, d_bo=d_bo, D=D, S=S,
                top=top, con_lai=con_lai)


def search_vector(v, bo_qua=None):
    """Tìm TOP_K ảnh giống nhất với vector đặc trưng gốc v (39 chiều).

    bo_qua: mã ảnh không đưa vào kết quả (dùng khi lấy chính một ảnh trong CSDL làm truy vấn).
    Trả về danh sách dict: ma_anh, S, D, D_hinh_hoc, D_ket_cau, D_mau_sac (S giảm dần).
    """
    xh = xep_hang(v, bo_qua)
    return [{"ma_anh": xh["ids"][i], "S": float(xh["S"][i]), "D": float(xh["D"][i]),
             "D_hinh_hoc": float(xh["d_bo"]["hinh_hoc"][i]), "D_ket_cau": float(xh["d_bo"]["ket_cau"][i]),
             "D_mau_sac": float(xh["d_bo"]["mau_sac"][i])} for i in xh["top"]]


def search_image(path):
    """Tìm kiếm với một ảnh mới: tiền xử lý -> trích 39 đặc trưng -> so với CSDL.

    Trả về (kết quả tiền xử lý, vector đặc trưng gốc, danh sách top-5). Ảnh truy vấn qua đủ các
    bước kiểm tra như ảnh trong CSDL; không đạt (không phải JPG, sai kích thước, ảnh mờ, không thấy
    đúng 1 khuôn mặt, điểm mốc không ổn định) thì vector và top-5 là None, lý do ở kq["ly_do"].
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
