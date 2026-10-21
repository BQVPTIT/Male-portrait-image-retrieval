"""
danh_gia.py - Đánh giá hệ tìm kiếm bằng phép thử ảnh biến đổi (yêu cầu 4c).

Mỗi người trong CSDL chỉ có một ảnh nên không có sẵn "đáp án đúng". Để có ảnh mới của cùng một
người, mỗi ảnh trong CSDL được biến đổi theo 8 kiểu (KIEU) giống các khác biệt khi chụp lại:
độ sáng, mặt hơi nghiêng, đứng lệch hoặc gần máy hơn, ảnh nén lại. Biến thể đi đúng đường của ảnh
truy vấn: preprocess.process_one (6 bước, kể cả kiểm tra) -> features.extract -> so với CSDL
(search.normalize, search.distances). Ảnh gốc vẫn nằm trong CSDL; hệ thống tốt thì ảnh gốc phải
đứng đầu.

    hạng của ảnh gốc = 1 + số ảnh trong CSDL có D nhỏ hơn D của ảnh gốc
    Chỉ số: % biến thể có ảnh gốc ở hạng 1, % lọt top-5, % lọt top-10, hạng trung bình.

Kết quả: data/danh_gia.csv (mỗi biến thể một dòng: ma_anh, kieu, dat, ly_do, hang).
Chạy:  python danh_gia.py      (khoảng 8 phút, chạy song song 8 tiến trình)
"""
import os
import csv
import shutil
import tempfile
import numpy as np
import cv2
from multiprocessing import Pool

import common as C
import preprocess
import features as F
import search as S

KQ_CSV = os.path.join(C.ROOT, "data", "danh_gia.csv")
KIEU = ["sang", "toi", "xoay_trai", "xoay_phai", "dich", "phong_to", "nen_jpg", "tong_hop"]
MO_TA = {"sang": "tăng độ sáng 15%", "toi": "giảm độ sáng 15%", "xoay_trai": "xoay 4°",
         "xoay_phai": "xoay -4°", "dich": "dịch khung 6 px ngang, 4 px dọc", "phong_to": "phóng to 10%",
         "nen_jpg": "nén lại JPG chất lượng 40",
         "tong_hop": "sáng 10%, xoay 3°, dịch (-4; 3) px, thu nhỏ 5%, nén JPG 60"}


def affine(rgb, goc=0.0, ti_le=1.0, dx=0.0, dy=0.0):
    """Xoay góc `goc` (độ) quanh tâm ảnh, co giãn `ti_le` lần, dịch (dx, dy) px.

    Phần ảnh bị thiếu ở mép lấy lặp lại điểm mép gần nhất (không để viền trắng như ảnh chụp thật).
    Dùng OpenCV vì đây là công cụ tạo dữ liệu thử, không thuộc hệ thống tìm kiếm.
    """
    M = cv2.getRotationMatrix2D((C.CENTER, C.CENTER), goc, ti_le)
    M[:, 2] += (dx, dy)
    return cv2.warpAffine(rgb, M, (C.SIZE, C.SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def bien_doi(rgb, kieu):
    """Ảnh biến đổi theo `kieu` và chất lượng JPG khi lưu (95 = gần như không mất thêm chi tiết)."""
    f = rgb.astype(np.float64)
    if kieu == "sang":
        return np.clip(f * 1.15, 0, 255).astype(np.uint8), 95
    if kieu == "toi":
        return np.clip(f * 0.85, 0, 255).astype(np.uint8), 95
    if kieu == "xoay_trai":
        return affine(rgb, goc=4), 95
    if kieu == "xoay_phai":
        return affine(rgb, goc=-4), 95
    if kieu == "dich":
        return affine(rgb, dx=6, dy=4), 95
    if kieu == "phong_to":
        return affine(rgb, ti_le=1.1), 95
    if kieu == "nen_jpg":
        return rgb.copy(), 40
    if kieu == "tong_hop":
        anh = np.clip(f * 1.10, 0, 255).astype(np.uint8)
        return affine(anh, goc=3, ti_le=0.95, dx=-4, dy=3), 60
    raise ValueError(kieu)


def vector_bien_the(args):
    """Tạo 8 biến thể của một ảnh, cho qua tiền xử lý + trích rút. Trả về [(ma, kieu, dat, ly_do, v)]."""
    ma, thu_muc = args
    rgb = preprocess.read_rgb(os.path.join(C.RAW_DIR, ma + ".jpg"))
    kq_ds = []
    for kieu in KIEU:
        anh, chat_luong = bien_doi(rgb, kieu)
        path = os.path.join(thu_muc, f"{ma}_{kieu}.jpg")
        cv2.imwrite(path, anh[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, chat_luong])
        kq = preprocess.process_one(path)
        v = F.extract(kq["rgb"], kq["gray"], kq["pts"]) if kq["dat"] else None
        kq_ds.append((ma, kieu, kq["dat"], kq["ly_do"], v))
    return kq_ds


def run():
    ids, X = S.load_features()
    mn, mx = S.load_norm()
    trong_so = S.load_weights()
    Xn = S.normalize(X, mn, mx)
    vi_tri = {m: i for i, m in enumerate(ids)}
    thu_muc = tempfile.mkdtemp(prefix="danh_gia_")
    try:
        with Pool(8) as pool:
            ket_qua = [r for nhom in pool.imap(vector_bien_the, [(m, thu_muc) for m in ids], chunksize=4)
                       for r in nhom]
    finally:
        shutil.rmtree(thu_muc, ignore_errors=True)

    with open(KQ_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ma_anh", "kieu", "dat", "ly_do", "hang"])
        for ma, kieu, dat, ly_do, v in ket_qua:
            hang = ""
            if dat:
                D, _ = S.distances(S.normalize(v, mn, mx), Xn, trong_so)
                hang = 1 + int((D < D[vi_tri[ma]]).sum())
            w.writerow([ma, kieu, int(dat), ly_do, hang])
    tong_ket()


def tong_ket():
    """In bảng kết quả theo từng kiểu biến đổi (đọc data/danh_gia.csv)."""
    with open(KQ_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"{'kieu':10s} {'so':>5s} {'dat':>5s} {'hang1%':>7s} {'top5%':>6s} {'top10%':>7s} {'hangTB':>7s}")
    for kieu in KIEU + ["tong"]:
        nhom = [r for r in rows if kieu == "tong" or r["kieu"] == kieu]
        h = np.array([int(r["hang"]) for r in nhom if r["dat"] == "1"])
        print(f"{kieu:10s} {len(nhom):5d} {len(h):5d} {100 * (h == 1).mean():7.1f} {100 * (h <= 5).mean():6.1f}"
              f" {100 * (h <= 10).mean():7.1f} {h.mean():7.2f}")


if __name__ == "__main__":
    run()
