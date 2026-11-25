"""
tien_xu_ly.py - Tiền xử lý ảnh chân dung (6 bước), sau đó lưu kết quả.

- 6 bước (Báo cáo 1, Phần 1, mục 2):
  + B1 đọc, kiểm tra quy cách -> B2 ảnh xám -> B3 độ nét
  + B4 phát hiện mặt + 68 điểm mốc (dlib) -> B5 căn chỉnh theo 2 mắt (xoay + co giãn + tịnh tiến)
  + B6 lọc nền
- Tự code (xu_ly_anh.py): ảnh xám, Laplace, căn chỉnh + nội suy song tuyến tính, tô đa giác.
- Dùng thư viện: OpenCV chỉ để đọc/ghi tệp ảnh; dlib cho B4 (bản chất ghi ở docstring).
- Chạy:  python tien_xu_ly.py
"""
import os
import csv
import shutil
import numpy as np
import cv2
import dlib

import xu_ly_anh as XL

# Ngưỡng độ nét (phương sai Laplace). Chọn sau khi xem các ảnh có độ nét thấp nhất
# trong bộ dữ liệu: ảnh kém nét nhất có độ nét 23,7 mà vẫn rõ mắt, mũi, miệng, lông mày.
NGUONG_NET = 20.0

# Điểm SVM của bộ dò mặt: >= 0 là "có mặt". Khi không cửa sổ nào đạt 0, lấy cửa sổ điểm cao
# nhất trong khoảng (-1, 0) - tức còn nằm trong lề của SVM (|w.x + b| < 1), vì quy cách dữ
# liệu bảo đảm mỗi ảnh có đúng 1 khuôn mặt. Các ảnh này được đánh dấu trong log để soát bằng mắt.
NGUONG_MAT_YEU = -1.0

# Kiểm tra điểm mốc có đáng tin không: dò 68 điểm 2 lần, trên ảnh gốc và trên ảnh kéo giãn
# dọc HE_SO_KEO lần, rồi đo độ lệch trung bình giữa hai kết quả (tính theo khoảng cách hai mắt).
# Điểm mốc bám đúng mắt, mũi, miệng thì hai lần cho cùng vị trí (lệch khoảng 0,02);
# điểm mốc đặt sai thì hai lần cho hai kết quả khác hẳn nhau -> loại ảnh.
HE_SO_KEO = 1.25           # mặt dài thêm 1/4: đủ để điểm mốc đặt sai lộ rõ, bộ dò vẫn nhận ra mặt ở cả 1005 ảnh
NGUONG_LECH_MOC = 0.08     # phân bố cột lech_moc: 94% ảnh < 0,04 (trung vị 0,02); từ 0,08 trở lên điểm mốc sai thấy rõ bằng mắt

COT_NHAT_KY = ["dat", "ly_do", "do_net", "so_mat", "diem_mat", "lech_moc", "ti_le_khung", "lech_tam",
           "goc_nghieng", "he_so", "ti_le_mat", "thieu"]

_bo_do_mat = None           # bộ dò mặt dlib, nạp một lần khi cần (xem tim_diem_moc)
_bo_do_diem_moc = None          # bộ dò 68 điểm mốc dlib


def danh_so_anh(thu_muc_nguon):
    """
    - File: tien_xu_ly.py. Dùng để: chép ảnh thu thập vào data/anh256 và đặt tên thống nhất.
    - Dựa trên: đánh số theo thứ tự tên tệp gốc (không phân biệt hoa thường)
      + tên mới: man_0001.jpg, man_0002.jpg, ...
      + bảng đối chiếu mã ảnh với tên gốc lưu ở data/ten_goc.csv
    - Ứng dụng: mã ảnh man_xxxx là khóa chính ma_anh của CSDL; ảnh gốc giữ nguyên.
    - Cách khác: đặt tên theo mã băm nội dung (khó đọc khi báo cáo).
    """
    os.makedirs(XL.THU_MUC_ANH_GOC, exist_ok=True)
    ds_tep = sorted((f for f in os.listdir(thu_muc_nguon) if f.lower().endswith(".jpg")), key=str.lower)
    with open(os.path.join(XL.THU_MUC_GOC, "data", "ten_goc.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ma_anh", "ten_goc"])
        for i, ten_tep in enumerate(ds_tep, start=1):
            ma = f"man_{i:04d}"
            shutil.copyfile(os.path.join(thu_muc_nguon, ten_tep), os.path.join(XL.THU_MUC_ANH_GOC, ma + ".jpg"))
            w.writerow([ma, ten_tep])
    return len(ds_tep)


def thu_nho_khoi(anh, k):
    """
    - File: tien_xu_ly.py. Dùng để: thu nhỏ ảnh k lần (k nguyên).
    - Dựa trên: lấy trung bình khối k x k điểm
      + mỗi điểm mới = (1/k^2) * tổng k x k điểm cũ
      + cắt bớt phần lẻ để chiều cao, chiều rộng chia hết cho k
    - Ứng dụng: thu nhỏ ảnh lớn trước khi dò mặt, và trước khi đưa về 256 x 256
      (lam_vuong, chuan_hoa_anh.py); trung bình khối làm mượt nên ít răng cưa.
    - Cách khác: bỏ bớt điểm (lấy 1 trên k), nhanh nhưng răng cưa.
    """
    if k <= 1:
        return anh.astype(np.float64)
    h, w = anh.shape[0] // k * k, anh.shape[1] // k * k
    # reshape thành (h/k, k, w/k, k, kênh) rồi lấy trung bình theo 2 trục k -> mỗi khối k x k thành 1 điểm
    return anh[:h, :w].astype(np.float64).reshape(h // k, k, w // k, k, -1).mean(axis=(1, 3))


def lam_vuong(tep, tep_ra):
    """
    - File: tien_xu_ly.py. Dùng để: đưa ảnh truy vấn JPG chưa đúng 256 x 256 về 256 x 256 (trước 6 bước).
    - Dựa trên: cắt vuông quanh khuôn mặt rồi thu nhỏ
      + dò mặt bằng dlib, phải có đúng 1 mặt
      + cạnh hình vuông = 2 x cạnh khung mặt (mặt chiếm khoảng 1/4 ảnh như CSDL), không vượt cạnh ngắn
      + thu nhỏ khối k x k, phần lẻ dùng nội suy song tuyến tính, lưu JPG chất lượng 95
    - Ứng dụng: nút "Upload ảnh" của giao diện.
    - Cách khác: cắt ở giữa ảnh (chuan_hoa_anh.py).
    """
    if not la_jpg(tep):
        return False, "khong phai JPG"
    rgb = doc_rgb(tep)
    if rgb is None or rgb.ndim != 3:
        return False, "khong doc duoc"
    h, w = rgb.shape[:2]
    # 1. dò mặt trên bản thu nhỏ (cạnh dài <= 640) cho nhanh
    k_do = max(1, int(np.ceil(max(h, w) / 640)))
    nho = thu_nho_khoi(rgb, k_do)
    so_mat, khung, _, _ = tim_diem_moc(np.round(XL.chuyen_xam(nho)).astype(np.uint8))
    if so_mat != 1:
        return False, "so khuon mat khac 1"
    # 2. hình vuông tâm là tâm khung mặt, cạnh a = 2 x cạnh khung mặt; chạm mép thì dịch vào trong
    l, t, r, b = [x * k_do for x in khung]
    a = int(min(2 * max(r - l, b - t), h, w))
    x0 = int(np.clip(round((l + r) / 2 - a / 2), 0, w - a))
    y0 = int(np.clip(round((t + b) / 2 - a / 2), 0, h - a))
    # 3. thu nhỏ khối k x k, phần lẻ còn lại dùng nội suy song tuyến tính
    vuong = thu_nho_khoi(rgb[y0:y0 + a, x0:x0 + a], max(1, a // XL.CANH_ANH))
    n = vuong.shape[0]
    toa_do = np.minimum((np.arange(XL.CANH_ANH) + 0.5) * n / XL.CANH_ANH - 0.5, n - 1.001).clip(0)
    ys, xs = np.meshgrid(toa_do, toa_do, indexing="ij")
    ra = np.clip(np.round(XL.noi_suy_song_tuyen(vuong, xs, ys)), 0, 255).astype(np.uint8)
    # 4. ghi JPG (OpenCV cần thứ tự kênh B, G, R)
    ok, bo_dem = cv2.imencode(".jpg", ra[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    bo_dem.tofile(tep_ra)
    return True, ""


# ---------------------------------------------------------------- B1
def la_jpg(tep):
    """
    - File: tien_xu_ly.py. Dùng để: kiểm tra tệp có thật là ảnh JPG không (bước 1).
    - Dựa trên: "chữ ký" đầu tệp của chuẩn JPEG
      + tệp JPEG luôn mở đầu bằng mã SOI FF D8, ngay sau là FF của đoạn kế tiếp
      + đọc 3 byte đầu và so với FF D8 FF
    - Ứng dụng: hệ thống chỉ dùng JPG; tệp PNG bị đổi đuôi thành .jpg vẫn bị phát hiện.
    - Cách khác: chỉ xem đuôi tên tệp (dễ bị lừa).
    """
    try:
        with open(tep, "rb") as f:
            return f.read(3) == b"\xff\xd8\xff"
    except OSError:
        return False


def doc_rgb(tep):
    """
    - File: tien_xu_ly.py. Dùng để: đọc tệp ảnh thành ma trận H x W x 3 (R, G, B), bước 1.
    - Dựa trên: giải nén JPEG bằng OpenCV
      + bên trong: giải mã Huffman, lượng tử hóa ngược, DCT ngược khối 8 x 8, đổi YCbCr sang RGB
      + OpenCV trả B, G, R nên đảo kênh; đọc qua np.fromfile để đường dẫn tiếng Việt vẫn chạy
    - Ứng dụng: mọi bước sau làm việc trên ma trận này. Lỗi thì trả về None.
    - Cách khác: thư viện Pillow.
    """
    try:
        du_lieu = np.fromfile(tep, dtype=np.uint8)
        bgr = cv2.imdecode(du_lieu, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if bgr is None:
        return None
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        return bgr                      # trả về để bước kiểm tra báo "sai số kênh"
    return bgr[:, :, ::-1].copy()       # B, G, R -> R, G, B


# ---------------------------------------------------------------- B4
def tim_diem_moc(xam_u8):
    """
    - File: tien_xu_ly.py. Dùng để: tìm khuôn mặt và 68 điểm mốc (bước 4), bằng thư viện dlib.
    - Dựa trên: học máy cổ điển đã huấn luyện sẵn
      + dò mặt: cửa sổ trượt nhiều tỉ lệ, đặc trưng HOG (lược đồ hướng gradient), SVM tuyến tính w.x + b
      + 68 điểm: đặt hình mặt trung bình vào khung, nhiều tầng cây hồi quy (ERT) dịch dần các điểm
    - Ứng dụng: trả về (số mặt, khung mặt, 68 x 2 điểm, điểm SVM); mọi đặc trưng dựa vào điểm mốc.
    - Cách khác: bộ dò CNN, MediaPipe (nặng, khó giải thích).
    """
    global _bo_do_mat, _bo_do_diem_moc
    if _bo_do_mat is None:               # nạp mô hình một lần, các lần sau dùng lại
        _bo_do_mat = dlib.get_frontal_face_detector()
        _bo_do_diem_moc = dlib.shape_predictor(XL.TEP_MO_HINH)
    # 1 = phóng ảnh lên 2 lần trước khi dò; trả về mọi cửa sổ có điểm >= NGUONG_MAT_YEU
    khung, diem, _ = _bo_do_mat.run(xam_u8, 1, NGUONG_MAT_YEU)
    manh = [i for i, d in enumerate(diem) if d >= 0]          # cửa sổ chắc chắn có mặt (điểm >= 0)
    if len(manh) >= 2 or len(khung) == 0:
        return len(manh), None, None, None
    i = manh[0] if manh else int(np.argmax(diem))             # không có cửa sổ mạnh: lấy cửa sổ điểm cao nhất
    r = khung[i]
    hinh_moc = _bo_do_diem_moc(xam_u8, r)
    moc = np.array([[hinh_moc.part(k).x, hinh_moc.part(k).y] for k in range(68)], dtype=np.float64)
    return 1, (r.left(), r.top(), r.right(), r.bottom()), moc, float(diem[i])


def lech_diem_moc(xam, moc):
    """
    - File: tien_xu_ly.py. Dùng để: đo độ ổn định của 68 điểm mốc (kiểm tra ở bước 4).
    - Dựa trên: dò điểm mốc 2 lần rồi so
      + lần 2 trên một bản tạm kéo giãn dọc 1,25 lần (nội suy song tuyến tính), chia tung độ về ảnh gốc
      + độ lệch = trung bình d(P_i, Q_i) / khoảng cách hai mắt, i = 0..67
      + không thấy mặt ở lần 2 thì trả về 1 (coi như không ổn định)
    - Ứng dụng: loại ảnh có độ lệch từ 0,08.
    - Cách khác: so điểm SVM.
    """
    cao = int(round(XL.CANH_ANH * HE_SO_KEO))
    ys, xs = np.mgrid[0:cao, 0:XL.CANH_ANH].astype(np.float64)
    keo = XL.noi_suy_song_tuyen(xam, xs, np.minimum(ys / HE_SO_KEO, XL.CANH_ANH - 1.001))    # bản tạm kéo giãn dọc
    so_mat, _, q, _ = tim_diem_moc(np.round(keo).astype(np.uint8))
    if so_mat != 1:
        return 1.0
    q[:, 1] /= HE_SO_KEO                                       # đưa tung độ về ảnh gốc
    d_mat = XL.khoang_cach_diem(*XL.tam_hai_mat(moc))
    return float(np.mean(np.sqrt(((moc - q) ** 2).sum(axis=1))) / d_mat)


# ---------------------------------------------------------------- B6
def mat_na_khuon_mat(moc):
    """
    - File: tien_xu_ly.py. Dùng để: tạo mặt nạ khuôn mặt để lọc nền (bước 6).
    - Dựa trên: đa giác từ điểm mốc đã căn chỉnh
      + viền hàm (điểm 0 -> 16) nối với 10 điểm lông mày (26 -> 17)
      + điểm lông mày dịch lên 0,6 x khoảng cách hai mắt (khoảng 38 px) để giữ vùng trán
      + tô bằng quy tắc chẵn – lẻ (xu_ly_anh.mat_na_da_giac)
    - Ứng dụng: điểm ngoài mặt nạ tô trắng, nền không làm sai thống kê màu.
    - Cách khác: tách nền bằng GrabCut, học sâu.
    """
    trai, phai = XL.tam_hai_mat(moc)
    d_mat = XL.khoang_cach_diem(trai, phai)
    tran = moc[26:16:-1].copy()         # điểm 26, 25, ..., 17
    tran[:, 1] -= 0.6 * d_mat           # dịch lên trên (y giảm)
    return XL.mat_na_da_giac(np.vstack([moc[0:17], tran]))


# ---------------------------------------------------------------- xử lý 1 ảnh
def xu_ly_mot_anh(tep, giu_trung_gian=False):
    """
    - File: tien_xu_ly.py. Dùng để: chạy đủ 6 bước B1 -> B6 cho một ảnh.
    - Dựa trên: đầu ra bước trước là đầu vào bước sau; không đạt bước 1, 3, 4 thì dừng
      + trả về dict: dat, ly_do, số liệu kiểm tra (do_net, lech_moc, goc_nghieng, he_so...)
      + đạt thì có thêm rgb, gray, pts đã căn chỉnh và lọc nền
      + giu_trung_gian=True: giữ ảnh từng bước để giao diện hiển thị, không đổi kết quả
    - Ứng dụng: dùng chung cho ảnh nạp CSDL, ảnh truy vấn và ảnh biến đổi khi đánh giá.
    - Cách khác: không có.
    """
    kq = {"dat": False, "ly_do": ""}
    kq.update({c: "" for c in COT_NHAT_KY[2:]})

    # B1 - đọc và kiểm tra quy cách (JPG, 256 x 256, 3 kênh)
    if not la_jpg(tep):
        kq["ly_do"] = "khong phai JPG"
        return kq
    rgb = doc_rgb(tep)
    if rgb is None:
        kq["ly_do"] = "khong doc duoc"
        return kq
    if rgb.shape != (XL.CANH_ANH, XL.CANH_ANH, 3):
        kq["ly_do"] = "sai kich thuoc/so kenh"
        return kq

    # B2 - ảnh xám
    xam = XL.chuyen_xam(rgb)

    # B3 - độ nét
    kq["do_net"] = XL.phuong_sai_laplace(xam)
    if kq["do_net"] < NGUONG_NET:
        kq["ly_do"] = "anh mo"
        return kq

    # B4 - khuôn mặt và 68 điểm mốc
    so_mat, khung, moc, diem = tim_diem_moc(np.round(xam).astype(np.uint8))
    kq["so_mat"] = so_mat
    if so_mat != 1:
        kq["ly_do"] = "so khuon mat khac 1"
        return kq
    l, t, r, b = khung
    kq["diem_mat"] = diem
    kq["lech_moc"] = lech_diem_moc(xam, moc)
    if kq["lech_moc"] >= NGUONG_LECH_MOC:
        kq["ly_do"] = "diem moc khong on dinh"
        return kq
    kq["ti_le_khung"] = (r - l) * (b - t) / float(XL.CANH_ANH * XL.CANH_ANH)     # diện tích khung mặt / diện tích ảnh
    kq["lech_tam"] = XL.khoang_cach_diem(((l + r) / 2.0, (t + b) / 2.0), (XL.TAM_ANH, XL.TAM_ANH))

    # B5 - căn chỉnh theo hai mắt
    #   theta = atan2(y_phải - y_trái, x_phải - x_trái);  s = D_MAT / d(mắt trái, mắt phải)
    rgb_tho, xam_tho = rgb, xam                              # giữ ảnh B1, B2 (chỉ để hiển thị)
    trai, phai = XL.tam_hai_mat(moc)
    theta = np.arctan2(phai[1] - trai[1], phai[0] - trai[0])  # góc nghiêng đường nối hai mắt
    s = XL.D_MAT / XL.khoang_cach_diem(trai, phai)                           # hệ số co giãn
    m = (trai + phai) / 2.0                                    # trung điểm hai mắt
    kq["goc_nghieng"] = float(np.degrees(theta))
    kq["he_so"] = s
    co_du_lieu = XL.can_chinh_anh(np.ones((XL.CANH_ANH, XL.CANH_ANH)), theta, s, m, nen=0.0) > 0.5   # vùng có dữ liệu thật
    rgb = np.clip(np.round(XL.can_chinh_anh(rgb, theta, s, m)), 0, 255).astype(np.uint8)
    moc_tho = moc
    moc = np.round(XL.can_chinh_diem(moc, theta, s, m), 3)        # điểm mốc đi theo cùng công thức
    xam = np.round(XL.chuyen_xam(rgb)).astype(np.uint8)           # ảnh xám của ảnh đã căn chỉnh

    # B6 - lọc nền: ngoài mặt nạ (hoặc chỗ ảnh thô không có dữ liệu) gán trắng
    mat = mat_na_khuon_mat(moc)
    kq["ti_le_mat"] = float(mat.mean())
    kq["thieu"] = float((mat & ~co_du_lieu).sum() / max(mat.sum(), 1))
    if giu_trung_gian:
        kq["trung_gian"] = dict(rgb_tho=rgb_tho, xam_tho=xam_tho, khung=khung,
                                moc_tho=moc_tho, rgb_can_chinh=rgb.copy(), mat_na=mat & co_du_lieu)
    rgb[~(mat & co_du_lieu)] = 255

    kq.update(dat=True, rgb=rgb, xam=xam, moc=moc)
    return kq


# ---------------------------------------------------------------- chạy cả bộ
def chay():
    """
    - File: tien_xu_ly.py. Dùng để: tiền xử lý cả bộ dữ liệu data/anh256.
    - Dựa trên: gọi xu_ly_mot_anh cho từng ảnh, lưu vào data/tien_xu_ly
      + da_can_chinh/man_xxxx.jpg: ảnh đã căn chỉnh, lọc nền (chỉ để xem)
      + diem_moc.csv: 68 điểm mốc sau căn chỉnh
      + nhat_ky.csv: kết quả kiểm tra của từng ảnh
    - Ứng dụng: chạy một lần khi xây dựng bộ dữ liệu; kết quả 1005/1005 ảnh đạt.
    - Cách khác: đặc trưng không tính từ ảnh đã lưu JPG mà tính thẳng trong trich_rut.chay.
    """
    shutil.rmtree(XL.THU_MUC_TIEN_XU_LY, ignore_errors=True)     # xóa kết quả lần chạy trước để không sót ảnh đã bị loại
    os.makedirs(XL.THU_MUC_DA_CAN_CHINH)
    ds_tep = sorted(f for f in os.listdir(XL.THU_MUC_ANH_GOC) if f.lower().endswith(".jpg"))
    so_dat = 0
    with open(XL.TEP_DIEM_MOC, "w", newline="") as f_moc, open(XL.TEP_NHAT_KY, "w", newline="") as f_nk:
        ghi_moc, ghi_nk = csv.writer(f_moc), csv.writer(f_nk)
        ghi_moc.writerow(["ma_anh"] + [f"{c}{i}" for i in range(68) for c in "xy"])
        ghi_nk.writerow(["ma_anh"] + COT_NHAT_KY)
        for ten_tep in ds_tep:
            ma = os.path.splitext(ten_tep)[0]
            kq = xu_ly_mot_anh(os.path.join(XL.THU_MUC_ANH_GOC, ten_tep))
            ghi_nk.writerow([ma] + [int(kq[c]) if c == "dat" else kq[c] for c in COT_NHAT_KY])
            if not kq["dat"]:
                continue
            so_dat += 1
            cv2.imwrite(os.path.join(XL.THU_MUC_DA_CAN_CHINH, ma + ".jpg"), kq["rgb"][:, :, ::-1])
            ghi_moc.writerow([ma] + [f"{v:.3f}" for v in kq["moc"].ravel()])
    print(f"Tien xu ly xong: {so_dat}/{len(ds_tep)} anh dat.")
    return so_dat


if __name__ == "__main__":
    chay()
