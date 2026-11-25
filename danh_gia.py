"""
danh_gia.py - Đánh giá hệ tìm kiếm bằng phép thử ảnh biến đổi (yêu cầu 4c).

Mỗi người trong CSDL chỉ có một ảnh nên không có sẵn "đáp án đúng". Để có ảnh mới của cùng một
người, mỗi ảnh trong CSDL được biến đổi theo 8 kiểu (KIEU) giống các khác biệt khi chụp lại:
độ sáng, mặt hơi nghiêng, đứng lệch hoặc gần máy hơn, ảnh nén lại. Biến thể đi đúng đường của ảnh
truy vấn: tien_xu_ly.xu_ly_mot_anh (6 bước, kể cả kiểm tra) -> trich_rut.trich_vector -> so với CSDL
(tim_kiem.chuan_hoa, tim_kiem.khoang_cach). Ảnh gốc vẫn nằm trong CSDL; hệ thống tốt thì ảnh gốc phải
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

import xu_ly_anh as XL
import tien_xu_ly as TXL
import trich_rut as TR
import tim_kiem as TK
TEP_KET_QUA = os.path.join(XL.THU_MUC_GOC, "data", "danh_gia.csv")
KIEU = ["sang", "toi", "xoay_trai", "xoay_phai", "dich", "phong_to", "nen_jpg", "tong_hop"]
MO_TA = {"sang": "tăng độ sáng 15%", "toi": "giảm độ sáng 15%", "xoay_trai": "xoay 4°",
         "xoay_phai": "xoay -4°", "dich": "dịch khung 6 px ngang, 4 px dọc", "phong_to": "phóng to 10%",
         "nen_jpg": "nén lại JPG chất lượng 40",
         "tong_hop": "sáng 10%, xoay 3°, dịch (-4; 3) px, thu nhỏ 5%, nén JPG 60"}


def bien_doi_affine(rgb, goc=0.0, ti_le=1.0, dx=0.0, dy=0.0):
    """
    - File: danh_gia.py. Dùng để: xoay, co giãn, dịch ảnh để tạo ảnh thử.
    - Dựa trên: phép biến đổi affine của OpenCV
      + ma trận xoay quanh tâm ảnh (getRotationMatrix2D), cộng thêm độ dịch (dx, dy)
      + phần thiếu ở mép lấy lặp lại điểm mép gần nhất, không để viền trắng
    - Ứng dụng: giả lập ảnh chụp lại cùng người (hơi nghiêng, đứng lệch, gần máy hơn).
    - Cách khác: dùng xu_ly_anh.can_chinh_anh (tự code). Dùng OpenCV vì đây là công cụ tạo dữ liệu thử,
      không thuộc hệ thống tìm kiếm.
    """
    M = cv2.getRotationMatrix2D((XL.TAM_ANH, XL.TAM_ANH), goc, ti_le)
    M[:, 2] += (dx, dy)
    return cv2.warpAffine(rgb, M, (XL.CANH_ANH, XL.CANH_ANH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def bien_doi(rgb, kieu):
    """
    - File: danh_gia.py. Dùng để: tạo một biến thể của ảnh theo kiểu cho trước.
    - Dựa trên: 8 kiểu thay đổi khi chụp lại (KIEU)
      + sáng, tối: nhân độ sáng 1,15 hoặc 0,85
      + xoay ±4 độ, dịch (6; 4) px, phóng to 10%, nén JPG chất lượng 40
      + tổng hợp: sáng 10%, xoay 3 độ, dịch, thu nhỏ 5%, nén JPG 60
    - Ứng dụng: trả về (ảnh, chất lượng JPG khi lưu) để làm ảnh truy vấn thử.
    - Cách khác: chụp thật nhiều ảnh mỗi người.
    """
    f = rgb.astype(np.float64)
    if kieu == "sang":
        return np.clip(f * 1.15, 0, 255).astype(np.uint8), 95
    if kieu == "toi":
        return np.clip(f * 0.85, 0, 255).astype(np.uint8), 95
    if kieu == "xoay_trai":
        return bien_doi_affine(rgb, goc=4), 95
    if kieu == "xoay_phai":
        return bien_doi_affine(rgb, goc=-4), 95
    if kieu == "dich":
        return bien_doi_affine(rgb, dx=6, dy=4), 95
    if kieu == "phong_to":
        return bien_doi_affine(rgb, ti_le=1.1), 95
    if kieu == "nen_jpg":
        return rgb.copy(), 40
    if kieu == "tong_hop":
        anh = np.clip(f * 1.10, 0, 255).astype(np.uint8)
        return bien_doi_affine(anh, goc=3, ti_le=0.95, dx=-4, dy=3), 60
    raise ValueError(kieu)


def vector_bien_the(tham_so):
    """
    - File: danh_gia.py. Dùng để: tạo 8 biến thể của một ảnh và tính vector của từng biến thể.
    - Dựa trên: đúng đường đi của ảnh truy vấn
      + lưu biến thể ra tệp JPG tạm
      + tien_xu_ly.xu_ly_mot_anh (6 bước, kể cả kiểm tra) rồi trich_rut.trich_vector
      + trả về danh sách (mã ảnh, kiểu, đạt hay không, lý do, vector)
    - Ứng dụng: mỗi tiến trình con chạy hàm này cho một ảnh, 8 tiến trình chạy song song.
    - Cách khác: biến đổi vector trực tiếp (không kiểm tra được tiền xử lý).
    """
    ma, thu_muc = tham_so
    rgb = TXL.doc_rgb(os.path.join(XL.THU_MUC_ANH_GOC, ma + ".jpg"))
    kq_ds = []
    for kieu in KIEU:
        anh, chat_luong = bien_doi(rgb, kieu)
        tep = os.path.join(thu_muc, f"{ma}_{kieu}.jpg")
        cv2.imwrite(tep, anh[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, chat_luong])
        kq = TXL.xu_ly_mot_anh(tep)
        v = TR.trich_vector(kq["rgb"], kq["xam"], kq["moc"]) if kq["dat"] else None
        kq_ds.append((ma, kieu, kq["dat"], kq["ly_do"], v))
    return kq_ds


def chay():
    """
    - File: danh_gia.py. Dùng để: chạy phép thử ảnh biến đổi cho cả 1005 ảnh (8040 biến thể).
    - Dựa trên: ảnh gốc vẫn nằm trong CSDL, hệ thống tốt thì ảnh gốc phải đứng đầu
      + hạng của ảnh gốc = 1 + số ảnh có D nhỏ hơn D của ảnh gốc
      + chạy song song bằng multiprocessing.Pool(8)
      + ghi data/danh_gia.csv rồi in bảng tổng kết
    - Ứng dụng: đánh giá kết quả (yêu cầu 4c): hạng 1 96,4%, top-5 99,0%.
    - Cách khác: precision, recall (cần nhãn đúng, bộ dữ liệu không có).
    """
    ds_ma, X = TK.doc_vector_csdl()
    mn, mx = TK.doc_min_max()
    trong_so = TK.doc_trong_so()
    Xn = TK.chuan_hoa(X, mn, mx)
    vi_tri = {m: i for i, m in enumerate(ds_ma)}
    thu_muc = tempfile.mkdtemp(prefix="danh_gia_")
    try:
        with Pool(8) as cac_tien_trinh:
            ket_qua = [r for nhom in cac_tien_trinh.imap(vector_bien_the, [(m, thu_muc) for m in ds_ma], chunksize=4)
                       for r in nhom]
    finally:
        shutil.rmtree(thu_muc, ignore_errors=True)

    with open(TEP_KET_QUA, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ma_anh", "kieu", "dat", "ly_do", "hang"])
        for ma, kieu, dat, ly_do, v in ket_qua:
            hang = ""
            if dat:
                D, _ = TK.khoang_cach(TK.chuan_hoa(v, mn, mx), Xn, trong_so)
                hang = 1 + int((D < D[vi_tri[ma]]).sum())
            w.writerow([ma, kieu, int(dat), ly_do, hang])
    tong_ket()


def tong_ket():
    """
    - File: danh_gia.py. Dùng để: in bảng kết quả theo từng kiểu biến đổi.
    - Dựa trên: đọc data/danh_gia.csv
      + số biến thể, số đạt tiền xử lý
      + % ảnh gốc ở hạng 1, % lọt top-5, % lọt top-10, hạng trung bình
      + dòng "tong" gộp cả 8 kiểu
    - Ứng dụng: số liệu cho Bảng 4.4 của Báo cáo 2.
    - Cách khác: vẽ biểu đồ cột theo kiểu biến đổi.
    """
    with open(TEP_KET_QUA, newline="") as f:
        ds_dong = list(csv.DictReader(f))
    print(f"{'kieu':10s} {'so':>5s} {'dat':>5s} {'hang1%':>7s} {'top5%':>6s} {'top10%':>7s} {'hangTB':>7s}")
    for kieu in KIEU + ["tong"]:
        nhom = [r for r in ds_dong if kieu == "tong" or r["kieu"] == kieu]
        h = np.array([int(r["hang"]) for r in nhom if r["dat"] == "1"])
        print(f"{kieu:10s} {len(nhom):5d} {len(h):5d} {100 * (h == 1).mean():7.1f} {100 * (h <= 5).mean():6.1f}"
              f" {100 * (h <= 10).mean():7.1f} {h.mean():7.2f}")


if __name__ == "__main__":
    chay()
