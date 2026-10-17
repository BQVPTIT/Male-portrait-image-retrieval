"""
features.py - Trích rút vector đặc trưng 39 chiều cho một ảnh đã tiền xử lý.

    Hình học (5)  : 5 tỉ lệ khoảng cách giữa các điểm mốc
    Kết cấu  (2)  : phương sai LBP vùng cằm - miệng, mật độ lông mày
    Màu sắc  (32) : lược đồ H x S (8 x 4 ô) trên mặt nạ da (2 má + trán)

Đầu vào: ảnh màu đã căn chỉnh + lọc nền (rgb), ảnh xám đã căn chỉnh (gray), 68 điểm mốc (pts).
Giá trị lưu trong data/features.csv là giá trị GỐC (chưa chuẩn hóa); chuẩn hóa làm ở search.py.

Chạy:  python features.py
"""
import os
import csv
import numpy as np
import cv2

import common as C
import preprocess

SO_O_H, SO_O_S = 8, 4      # số ô của kênh H và kênh S
H_MAX = 40.0               # dải sắc độ da được chia ô: 0..40 độ (5 độ / ô)

TEN_HINH_HOC = ["geo_khung_mat", "geo_goma_ham", "geo_kc_mat", "geo_rong_mui", "geo_cao_mui"]
TEN_KET_CAU = ["tex_lbp_cam", "tex_long_may"]
TEN_MAU = [f"col_h{i}_s{j}" for i in range(SO_O_H) for j in range(SO_O_S)]
FEATURE_NAMES = TEN_HINH_HOC + TEN_KET_CAU + TEN_MAU       # 5 + 2 + 32 = 39 tên cột

# vị trí của từng bộ trong vector 39 chiều (dùng lại ở search.py)
BO = {"hinh_hoc": slice(0, 5), "ket_cau": slice(5, 7), "mau_sac": slice(7, 39)}


# ---------------------------------------------------------------- hình học
def geometric_features(pts):
    """5 đặc trưng hình học. d = khoảng cách Euclid, dY = |y_P - y_Q| (theo trục đứng).

        a. geo_khung_mat = d(P0, P16) / dY(P27, P8)              rộng mặt / cao mặt
        b. geo_goma_ham  = d(P0, P16) / d(P4, P12)               rộng gò má / rộng hàm
        c. geo_kc_mat    = d(mắt trái, mắt phải) / d(P0, P16)    khoảng cách mắt / rộng mặt
        d. geo_rong_mui  = d(P31, P35) / d(mắt trái, mắt phải)   rộng cánh mũi / khoảng cách mắt
        e. geo_cao_mui   = dY(P27, P33) / dY(P27, P8)            cao mũi / cao mặt
    Tâm mắt = trung bình 6 điểm quanh mắt (36-41 và 42-47).
    """
    trai, phai = C.eye_centers(pts)
    rong_mat = C.dist(pts[0], pts[16])
    cao_mat = abs(pts[27][1] - pts[8][1])
    rong_ham = C.dist(pts[4], pts[12])
    kc_mat = C.dist(trai, phai)
    rong_mui = C.dist(pts[31], pts[35])
    cao_mui = abs(pts[27][1] - pts[33][1])
    return [rong_mat / cao_mat, rong_mat / rong_ham, kc_mat / rong_mat,
            rong_mui / kc_mat, cao_mui / cao_mat]


# ---------------------------------------------------------------- các vùng trên mặt
def chin_region(pts):
    """Vùng cằm - miệng: đa giác qua các điểm 33 (đáy mũi), 54, 12, 8, 4, 48."""
    return C.polygon_mask(pts[[33, 54, 12, 8, 4, 48]])


def brow_regions(pts):
    """Hai vùng lông mày: hộp bao các điểm 17-21 (trái ảnh) và 22-26 (phải ảnh),
    nới lên và xuống 0,15 x khoảng cách hai mắt để lấy trọn bề dày lông mày."""
    trai, phai = C.eye_centers(pts)
    noi = 0.15 * C.dist(trai, phai)
    ys, xs = np.mgrid[0:C.SIZE, 0:C.SIZE]
    vung = []
    for nhom in (pts[17:22], pts[22:27]):
        x1, x2 = nhom[:, 0].min(), nhom[:, 0].max()
        y1, y2 = nhom[:, 1].min() - noi, nhom[:, 1].max() + noi
        vung.append((xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2))
    return vung


def skin_mask(pts):
    """Mặt nạ da = má trái + má phải + trán (không chứa mắt, lông mày, mũi, miệng).

        má trái : đa giác qua các điểm 1, 2, 3, 31, 40, 41
        má phải : đa giác qua các điểm 15, 14, 13, 35, 47, 46
        trán    : hình chữ nhật, ngang từ x(P19) đến x(P24) (đỉnh hai lông mày),
                  dọc từ (đỉnh lông mày - 0,45*d) đến (đỉnh lông mày - 0,10*d), d = khoảng cách hai mắt.
                  0,10*d: chừa mép trên lông mày; 0,45*d: nằm trong mặt nạ khuôn mặt (0,6*d).
    """
    trai, phai = C.eye_centers(pts)
    d = C.dist(trai, phai)
    ma = C.polygon_mask(pts[[1, 2, 3, 31, 40, 41]]) | C.polygon_mask(pts[[15, 14, 13, 35, 47, 46]])
    ys, xs = np.mgrid[0:C.SIZE, 0:C.SIZE]
    dinh = pts[17:27, 1].min()
    tran = (xs >= pts[19, 0]) & (xs <= pts[24, 0]) & (ys >= dinh - 0.45 * d) & (ys <= dinh - 0.10 * d)
    return ma | tran


# ---------------------------------------------------------------- kết cấu
def texture_features(gray, pts, co_du_lieu):
    """2 đặc trưng kết cấu, tính trên ảnh xám qua mã LBP (common.lbp).

    a. tex_lbp_cam: phương sai các mã LBP trong vùng cằm - miệng
           sigma^2 = (1/N) * tổng (c_i - trung bình)^2
       Miền giá trị [0 ; 127,5^2 = 16256,25] (lớn nhất khi nửa số mã = 0, nửa = 255).
    b. tex_long_may: mật độ lông mày = số điểm "sợi lông" / tổng số điểm của vùng, trung bình
       hai bên. Một điểm là "sợi lông" khi (1) tối hơn hoặc bằng ngưỡng Otsu tính riêng trong
       vùng đó VÀ (2) mã LBP khác 0 và 255 (không phải vùng phẳng). Miền giá trị [0 ; 1].
    co_du_lieu: mặt nạ các điểm ảnh có dữ liệu thật (không phải phần trắng thêm vào khi căn chỉnh).
    """
    ma = C.lbp(gray)

    cam = chin_region(pts) & co_du_lieu
    lbp_cam = float(np.var(ma[cam]))

    mat_do = []
    for vung in brow_regions(pts):
        vung = vung & co_du_lieu
        if vung.sum() == 0:
            continue
        xam = gray[vung]
        nguong = C.otsu_threshold(xam)
        soi = (xam <= nguong) & (ma[vung] != 0) & (ma[vung] != 255)
        mat_do.append(soi.sum() / float(vung.sum()))
    return [lbp_cam, float(np.mean(mat_do)) if mat_do else 0.0]


# ---------------------------------------------------------------- màu sắc
def hs_bins(h, s):
    """Xếp từng điểm da vào ô H (0..7) và ô S (0..3).

    H: 8 ô đều trong 0..40 độ (5 độ/ô). H >= 330 độ (đỏ sát 360) dồn vào ô đầu;
       40 < H < 330 dồn vào ô cuối.
    S: 4 ô đều trong [0, 1] (0,25/ô); S = 1 thuộc ô cuối.
    """
    o_h = np.floor(h / (H_MAX / SO_O_H)).astype(int)
    o_h = np.where(h >= 330.0, 0, np.clip(o_h, 0, SO_O_H - 1))
    o_s = np.clip(np.floor(s * SO_O_S).astype(int), 0, SO_O_S - 1)
    return o_h, o_s


def color_features(rgb, pts, co_du_lieu):
    """32 đặc trưng màu: lược đồ H x S chuẩn hóa trên mặt nạ da.

        h_k = n_k / N     n_k: số điểm da rơi vào ô k, N: tổng số điểm da  ->  tổng h_k = 1
    Thứ tự ô: k = (ô H) * 4 + (ô S), tức col_h0_s0, col_h0_s1, ..., col_h7_s3.
    Không dùng kênh V (độ sáng) để giảm ảnh hưởng của ánh sáng lúc chụp.
    """
    da = skin_mask(pts) & co_du_lieu
    h, s, _ = C.rgb_to_hsv(rgb[da])
    o_h, o_s = hs_bins(h, s)
    dem = np.bincount(o_h * SO_O_S + o_s, minlength=SO_O_H * SO_O_S).astype(np.float64)
    return list(dem / max(dem.sum(), 1.0))


# ---------------------------------------------------------------- ghép vector
def extract(rgb, gray, pts):
    """Vector 39 chiều (đúng thứ tự FEATURE_NAMES) của một ảnh đã tiền xử lý.

    Quy ước từ bước lọc nền: điểm trắng tuyệt đối (255, 255, 255) = nền / không có dữ liệu,
    nên không đưa vào thống kê kết cấu và màu.
    """
    pts = np.asarray(pts, dtype=np.float64)
    co_du_lieu = ~(rgb == 255).all(axis=2)
    v = geometric_features(pts) + texture_features(gray, pts, co_du_lieu) + color_features(rgb, pts, co_du_lieu)
    return np.array(v, dtype=np.float64)


def read_prep(ma):
    """Đọc ảnh đã tiền xử lý của một mã ảnh để XEM / vẽ hình (OpenCV trả B, G, R nên đảo kênh).

    Trả về (ảnh màu, ảnh xám tính từ ảnh màu). Không dùng để tính đặc trưng, xem run().
    """
    rgb = cv2.imread(os.path.join(C.ALIGNED_DIR, ma + ".jpg"), cv2.IMREAD_COLOR)[:, :, ::-1]
    return rgb, np.round(C.to_gray(rgb)).astype(np.uint8)


def read_landmarks():
    """Đọc landmarks.csv -> dict {mã ảnh: ma trận 68 x 2}."""
    with open(C.LANDMARK_CSV, newline="") as f:
        rows = list(csv.reader(f))[1:]
    return {r[0]: np.array(r[1:], dtype=np.float64).reshape(68, 2) for r in rows}


def run():
    """Trích đặc trưng cho mọi ảnh đạt tiền xử lý, lưu data/features.csv (mã ảnh + 39 cột).

    Mỗi ảnh gốc được đưa qua preprocess.process_one rồi trích đặc trưng ngay trên kết quả,
    đúng đường đi của ảnh truy vấn trong search.search_image -> ảnh trong CSDL và ảnh truy vấn
    được tính theo cùng một cách.
    """
    lm = read_landmarks()
    with open(C.FEATURE_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ma_anh"] + FEATURE_NAMES)
        for ma in sorted(lm):
            kq = preprocess.process_one(os.path.join(C.RAW_DIR, ma + ".jpg"))
            w.writerow([ma] + [f"{x:.6f}" for x in extract(kq["rgb"], kq["gray"], kq["pts"])])
    print(f"Trich dac trung xong: {len(lm)} anh x {len(FEATURE_NAMES)} chieu.")
    return len(lm)


if __name__ == "__main__":
    run()
