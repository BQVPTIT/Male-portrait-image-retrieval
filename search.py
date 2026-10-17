"""
search.py - Chuẩn hóa đặc trưng và tìm 5 ảnh giống ảnh truy vấn nhất.

    1. Chuẩn hóa min-max từng chiều:  x' = (x - min) / (max - min)
       (min, max tính trên toàn bộ CSDL, lưu ở data/norm_params.json để dùng lại cho ảnh truy vấn)
    2. Khoảng cách riêng cho từng bộ thuộc tính, chia căn số chiều để 3 bộ cùng thang:
           D_bộ = sqrt( tổng_i (q_i - x_i)^2 ) / sqrt(n_bộ)       n = 5, 2, 32
    3. Khoảng cách chung:  D = (1/3)*D_hình_học + (1/3)*D_kết_cấu + (1/3)*D_màu_sắc
    4. Độ tương đồng:      S = 1 / (1 + D)   (D = 0 -> S = 1; D càng lớn S càng gần 0)
    5. Sắp xếp S giảm dần, lấy 5 ảnh đầu.

Chạy:  python search.py duong_dan_anh_truy_van.jpg
"""
import sys
import csv
import json
import numpy as np

import common as C
import preprocess
import features as F

TRONG_SO = {"hinh_hoc": 1 / 3, "ket_cau": 1 / 3, "mau_sac": 1 / 3}
TOP_K = 5


def load_features():
    """Đọc data/features.csv -> (danh sách mã ảnh, ma trận N x 39 giá trị gốc)."""
    with open(C.FEATURE_CSV, newline="") as f:
        rows = list(csv.reader(f))[1:]
    return [r[0] for r in rows], np.array([r[1:] for r in rows], dtype=np.float64)


def fit_norm(X):
    """Tính min, max từng chiều trên toàn bộ CSDL và lưu vào norm_params.json."""
    mn, mx = X.min(axis=0), X.max(axis=0)
    with open(C.NORM_JSON, "w", encoding="utf-8") as f:
        json.dump({"ten": F.FEATURE_NAMES, "min": mn.tolist(), "max": mx.tolist()}, f, indent=1)
    return mn, mx


def load_norm():
    """Đọc lại min, max đã lưu."""
    with open(C.NORM_JSON, encoding="utf-8") as f:
        p = json.load(f)
    return np.array(p["min"]), np.array(p["max"])


def normalize(X, mn, mx):
    """Chuẩn hóa min-max: x' = (x - min) / (max - min). Chiều có max = min thì cho bằng 0.

    Ảnh trong CSDL luôn cho x' trong [0, 1]; ảnh truy vấn mới có thể hơi ra ngoài khoảng này
    nếu giá trị của nó nhỏ hơn min hoặc lớn hơn max của CSDL.
    """
    rong = mx - mn
    return (X - mn) / np.where(rong == 0, 1.0, rong) * (rong != 0)


def distances(q, Xn):
    """Khoảng cách từ vector truy vấn q (đã chuẩn hóa) tới từng hàng của Xn (đã chuẩn hóa).

    Trả về (D chung, dict D từng bộ). D_bộ = Euclid / sqrt(số chiều của bộ).
    """
    d_bo = {}
    for ten, vt in F.BO.items():
        hieu = Xn[:, vt] - q[vt]
        d_bo[ten] = np.sqrt((hieu ** 2).sum(axis=1)) / np.sqrt(hieu.shape[1])
    D = sum(TRONG_SO[ten] * d_bo[ten] for ten in F.BO)
    return D, d_bo


def search_vector(v, bo_qua=None):
    """Tìm TOP_K ảnh giống nhất với vector đặc trưng gốc v (39 chiều).

    bo_qua: mã ảnh không đưa vào kết quả (dùng khi lấy chính một ảnh trong CSDL làm truy vấn).
    Trả về danh sách dict: ma_anh, S, D, D_hinh_hoc, D_ket_cau, D_mau_sac (S giảm dần).
    """
    ids, X = load_features()
    mn, mx = load_norm()
    D, d_bo = distances(normalize(v, mn, mx), normalize(X, mn, mx))
    S = 1.0 / (1.0 + D)
    thu_tu = [i for i in np.argsort(-S) if ids[i] != bo_qua][:TOP_K]
    return [{"ma_anh": ids[i], "S": float(S[i]), "D": float(D[i]),
             "D_hinh_hoc": float(d_bo["hinh_hoc"][i]), "D_ket_cau": float(d_bo["ket_cau"][i]),
             "D_mau_sac": float(d_bo["mau_sac"][i])} for i in thu_tu]


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
