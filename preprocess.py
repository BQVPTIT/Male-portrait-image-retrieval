"""
preprocess.py - Tiền xử lý ảnh chân dung (6 bước), sau đó lưu kết quả.

    B1 đọc, kiểm tra quy cách -> B2 ảnh xám -> B3 độ nét
    -> B4 phát hiện mặt + 68 điểm mốc (dlib) -> B5 căn chỉnh theo 2 mắt (xoay + co giãn + tịnh tiến)
    -> B6 lọc nền

Tự code (common.py): ảnh xám, Laplace, căn chỉnh + nội suy song tuyến tính, tô đa giác.
Dùng thư viện: OpenCV chỉ để đọc/ghi tệp ảnh; dlib cho B4 (bản chất ghi ở docstring).

Chạy:  python preprocess.py
"""
import os
import csv
import shutil
import numpy as np
import cv2
import dlib

import common as C

# Ngưỡng độ nét (phương sai Laplace). Chọn sau khi xem các ảnh có độ nét thấp nhất
# trong bộ dữ liệu: ảnh dưới ngưỡng này nhìn bằng mắt đã nhòe, không rõ lông mày.
NGUONG_NET = 20.0

# Điểm SVM của bộ dò mặt: >= 0 là "có mặt". Khi không cửa sổ nào đạt 0, lấy cửa sổ điểm cao
# nhất trong khoảng (-1, 0) - tức còn nằm trong lề của SVM (|w.x + b| < 1), vì quy cách dữ
# liệu bảo đảm mỗi ảnh có đúng 1 khuôn mặt. Các ảnh này được đánh dấu trong log để soát bằng mắt.
NGUONG_MAT_YEU = -1.0

# Kiểm tra điểm mốc có đáng tin không: dò 68 điểm 2 lần, trên ảnh gốc và trên ảnh kéo giãn
# dọc HE_SO_KEO lần, rồi đo độ lệch trung bình giữa hai kết quả (tính theo khoảng cách hai mắt).
# Điểm mốc bám đúng mắt, mũi, miệng thì hai lần cho cùng vị trí (lệch khoảng 0,02);
# điểm mốc đặt sai thì hai lần cho hai kết quả khác hẳn nhau -> loại ảnh.
HE_SO_KEO = 1.25           # mặt dài thêm 1/4: đủ để điểm mốc đặt sai lộ rõ, bộ dò vẫn nhận ra mặt ở cả 1006 ảnh
NGUONG_LECH_MOC = 0.08     # phân bố cột lech_moc: 86% ảnh < 0,04 (trung vị 0,02); từ 0,08 trở lên điểm mốc sai thấy rõ bằng mắt

COT_LOG = ["dat", "ly_do", "do_net", "so_mat", "diem_mat", "lech_moc", "ti_le_khung", "lech_tam",
           "goc_nghieng", "he_so", "ti_le_mat", "thieu"]

_detector = None
_predictor = None


def danh_so_anh(src_dir):
    """Chép ảnh thu thập được vào data/raw256, đặt lại tên man_0001.jpg, man_0002.jpg...

    Thứ tự đánh số = thứ tự tên tệp gốc (không phân biệt hoa thường). Ảnh gốc giữ nguyên.
    Bảng đối chiếu mã ảnh <-> tên gốc lưu ở data/name_map.csv.
    """
    os.makedirs(C.RAW_DIR, exist_ok=True)
    files = sorted((f for f in os.listdir(src_dir) if f.lower().endswith(".jpg")), key=str.lower)
    with open(os.path.join(C.ROOT, "data", "name_map.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ma_anh", "ten_goc"])
        for i, name in enumerate(files, start=1):
            ma = f"man_{i:04d}"
            shutil.copyfile(os.path.join(src_dir, name), os.path.join(C.RAW_DIR, ma + ".jpg"))
            w.writerow([ma, name])
    return len(files)


# ---------------------------------------------------------------- B1
def la_jpg(path):
    """Kiểm tra tệp có đúng là ảnh JPG không (hệ thống chỉ dùng một định dạng JPG).

    Tệp JPEG luôn mở đầu bằng mã SOI (Start Of Image) = FF D8, ngay sau là FF của đoạn kế tiếp.
    Kiểm tra theo nội dung tệp chứ không theo đuôi, nên tệp PNG đổi đuôi thành .jpg vẫn bị phát hiện.
    """
    try:
        with open(path, "rb") as f:
            return f.read(3) == b"\xff\xd8\xff"
    except OSError:
        return False


def read_rgb(path):
    """B1 - Đọc tệp ảnh thành ma trận H x W x 3 (R, G, B). Không đọc được thì trả về None.

    OpenCV giải nén JPEG (giải mã Huffman -> lượng tử hóa ngược -> DCT ngược trên từng
    khối 8x8 -> đổi YCbCr về màu) và trả về thứ tự kênh B, G, R nên phải đảo lại thành R, G, B.
    Đọc qua np.fromfile để đường dẫn có dấu tiếng Việt vẫn đọc được.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if bgr is None:
        return None
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        return bgr                      # trả về để bước kiểm tra báo "sai số kênh"
    return bgr[:, :, ::-1].copy()


# ---------------------------------------------------------------- B4
def detect_landmarks(gray_u8):
    """B4 - Phát hiện khuôn mặt và 68 điểm mốc bằng dlib.

    Trả về (số mặt, khung mặt (trái, trên, phải, dưới), ma trận 68 x 2 điểm mốc, điểm SVM).

    Bản chất phát hiện mặt (dlib.get_frontal_face_detector, HOG + SVM tuyến tính):
      1. Trượt một cửa sổ trên ảnh xám ở nhiều tỉ lệ thu nhỏ (kim tự tháp ảnh).
      2. Trong mỗi cửa sổ tính đặc trưng HOG: gradient Gx, Gy tại từng điểm,
         độ lớn G = sqrt(Gx^2 + Gy^2), hướng = arctan(Gy / Gx); gom thành lược đồ hướng
         theo từng ô 8x8, chuẩn hóa theo khối.
      3. Bộ phân loại SVM tuyến tính (đã huấn luyện sẵn) tính điểm w.x + b cho vector HOG;
         điểm dương -> cửa sổ có mặt.
      4. Gộp các cửa sổ chồng lấn, giữ cửa sổ điểm cao nhất.
    Bản chất 68 điểm mốc (dlib.shape_predictor, cây hồi quy ERT):
      1. Đặt hình dạng mặt trung bình (68 điểm) vào khung mặt.
      2. Qua nhiều tầng cây hồi quy; mỗi cây so sánh hiệu độ sáng của các cặp điểm ảnh
         quanh vị trí hiện tại để chọn lá, lá cho lượng dịch chuyển của 68 điểm.
      3. Cộng dồn các lượng dịch chuyển -> 68 điểm khớp dần với khuôn mặt thật.
    """
    global _detector, _predictor
    if _detector is None:
        _detector = dlib.get_frontal_face_detector()
        _predictor = dlib.shape_predictor(C.MODEL_PATH)
    # 1 = phóng ảnh lên 2 lần trước khi dò; trả về mọi cửa sổ có điểm >= NGUONG_MAT_YEU
    khung, diem, _ = _detector.run(gray_u8, 1, NGUONG_MAT_YEU)
    manh = [i for i, d in enumerate(diem) if d >= 0]
    if len(manh) >= 2 or len(khung) == 0:
        return len(manh), None, None, None
    i = manh[0] if manh else int(np.argmax(diem))
    r = khung[i]
    shape = _predictor(gray_u8, r)
    pts = np.array([[shape.part(k).x, shape.part(k).y] for k in range(68)], dtype=np.float64)
    return 1, (r.left(), r.top(), r.right(), r.bottom()), pts, float(diem[i])


def landmark_shift(gray, pts):
    """Độ lệch điểm mốc giữa 2 lần dò (ảnh gốc và ảnh kéo giãn dọc HE_SO_KEO lần).

        lệch = trung bình_{i=0..67} d(P_i, Q_i) / d(mắt trái, mắt phải)
    P_i: điểm mốc trên ảnh gốc; Q_i: điểm mốc dò trên ảnh kéo giãn rồi chia tung độ cho
    HE_SO_KEO để đưa về ảnh gốc. Ảnh kéo giãn tạo bằng nội suy song tuyến tính (tự code).
    Không dò được mặt trên ảnh kéo giãn thì trả về 1 (coi như không ổn định).
    """
    cao = int(round(C.SIZE * HE_SO_KEO))
    ys, xs = np.mgrid[0:cao, 0:C.SIZE].astype(np.float64)
    keo = C.bilinear_sample(gray, xs, np.minimum(ys / HE_SO_KEO, C.SIZE - 1.001))
    so_mat, _, q, _ = detect_landmarks(np.round(keo).astype(np.uint8))
    if so_mat != 1:
        return 1.0
    q[:, 1] /= HE_SO_KEO
    d_mat = C.dist(*C.eye_centers(pts))
    return float(np.mean(np.sqrt(((pts - q) ** 2).sum(axis=1))) / d_mat)


# ---------------------------------------------------------------- B6
def face_mask(pts):
    """B6 - Mặt nạ khuôn mặt (True = thuộc mặt) dựng từ điểm mốc đã căn chỉnh.

    Đa giác gồm: viền hàm (điểm 0 -> 16) nối với 10 điểm lông mày (26 -> 17) đã dịch LÊN
    0,6 x khoảng cách hai mắt để giữ vùng trán (bộ 68 điểm không có điểm nào trên trán).
    0,6 (khoảng 38 px) lấy được phần trán ngay trên lông mày; dịch cao hơn thì hai góc trên
    của mặt nạ dễ lấn sang tóc ở hai bên thái dương.
    Tô đa giác bằng common.polygon_mask (quy tắc chẵn - lẻ, tự code).
    """
    trai, phai = C.eye_centers(pts)
    d_mat = C.dist(trai, phai)
    tran = pts[26:16:-1].copy()         # điểm 26, 25, ..., 17
    tran[:, 1] -= 0.6 * d_mat
    return C.polygon_mask(np.vstack([pts[0:17], tran]))


# ---------------------------------------------------------------- xử lý 1 ảnh
def process_one(path):
    """Chạy B1 -> B6 cho một ảnh. Trả về dict:

        dat (True/False), ly_do, so_mat, diem_mat (điểm SVM của khung mặt), do_net,
        lech_moc (độ lệch điểm mốc giữa 2 lần dò, xem landmark_shift),
        ti_le_khung, lech_tam, goc_nghieng (độ), he_so (hệ số co giãn s): đo trên ảnh THÔ;
        ti_le_mat (diện tích mặt nạ mặt / diện tích ảnh), thieu (phần mặt nạ rơi ra ngoài
        ảnh thô): đo SAU căn chỉnh;
        và nếu đạt: rgb (ảnh màu đã căn chỉnh, lọc nền), gray (ảnh xám đã căn chỉnh),
        pts (68 điểm mốc sau căn chỉnh).
    Dùng chung cho ảnh trong CSDL và ảnh truy vấn: cả hai qua đủ các bước kiểm tra.
    """
    kq = {"dat": False, "ly_do": ""}
    kq.update({c: "" for c in COT_LOG[2:]})

    # B1 - đọc và kiểm tra quy cách (JPG, 256 x 256, 3 kênh)
    if not la_jpg(path):
        kq["ly_do"] = "khong phai JPG"
        return kq
    rgb = read_rgb(path)
    if rgb is None:
        kq["ly_do"] = "khong doc duoc"
        return kq
    if rgb.shape != (C.SIZE, C.SIZE, 3):
        kq["ly_do"] = "sai kich thuoc/so kenh"
        return kq

    # B2 - ảnh xám
    gray = C.to_gray(rgb)

    # B3 - độ nét
    kq["do_net"] = C.laplacian_var(gray)
    if kq["do_net"] < NGUONG_NET:
        kq["ly_do"] = "anh mo"
        return kq

    # B4 - khuôn mặt và 68 điểm mốc
    so_mat, khung, pts, diem = detect_landmarks(np.round(gray).astype(np.uint8))
    kq["so_mat"] = so_mat
    if so_mat != 1:
        kq["ly_do"] = "so khuon mat khac 1"
        return kq
    l, t, r, b = khung
    kq["diem_mat"] = diem
    kq["lech_moc"] = landmark_shift(gray, pts)
    if kq["lech_moc"] > NGUONG_LECH_MOC:
        kq["ly_do"] = "diem moc khong on dinh"
        return kq
    kq["ti_le_khung"] = (r - l) * (b - t) / float(C.SIZE * C.SIZE)     # diện tích khung mặt / diện tích ảnh
    kq["lech_tam"] = C.dist(((l + r) / 2.0, (t + b) / 2.0), (C.CENTER, C.CENTER))

    # B5 - căn chỉnh theo hai mắt
    #   theta = atan2(y_phải - y_trái, x_phải - x_trái);  s = D_MAT / d(mắt trái, mắt phải)
    trai, phai = C.eye_centers(pts)
    theta = np.arctan2(phai[1] - trai[1], phai[0] - trai[0])
    s = C.D_MAT / C.dist(trai, phai)
    m = (trai + phai) / 2.0                                    # trung điểm hai mắt
    kq["goc_nghieng"] = float(np.degrees(theta))
    kq["he_so"] = s
    co_du_lieu = C.align_image(np.ones((C.SIZE, C.SIZE)), theta, s, m, fill=0.0) > 0.5
    rgb = np.clip(np.round(C.align_image(rgb, theta, s, m)), 0, 255).astype(np.uint8)
    pts = np.round(C.align_points(pts, theta, s, m), 3)        # điểm mốc đi theo cùng công thức
    gray = np.round(C.to_gray(rgb)).astype(np.uint8)           # ảnh xám của ảnh đã căn chỉnh

    # B6 - lọc nền: ngoài mặt nạ (hoặc chỗ ảnh thô không có dữ liệu) gán trắng
    mat = face_mask(pts)
    kq["ti_le_mat"] = float(mat.mean())
    kq["thieu"] = float((mat & ~co_du_lieu).sum() / max(mat.sum(), 1))
    rgb[~(mat & co_du_lieu)] = 255

    kq.update(dat=True, rgb=rgb, gray=gray, pts=pts)
    return kq


# ---------------------------------------------------------------- chạy cả bộ
def run():
    """Chạy tiền xử lý (B1 -> B6) cho mọi ảnh trong data/raw256 và lưu kết quả vào data/prep.

    aligned/man_xxxx.jpg : ảnh màu đã căn chỉnh, đã lọc nền (cùng định dạng JPG với ảnh gốc)
    landmarks.csv        : mã ảnh + 136 tọa độ (x0, y0, ..., x67, y67)
    preprocess_log.csv   : mã ảnh + các cột trong COT_LOG
    Ảnh lưu ở đây chỉ để XEM và vẽ hình. Đặc trưng không tính từ tệp này mà tính thẳng từ
    kết quả process_one (xem features.run), vì nén JPG làm đổi nhẹ giá trị điểm ảnh.
    """
    shutil.rmtree(C.PREP_DIR, ignore_errors=True)     # xóa kết quả lần chạy trước để không sót ảnh đã bị loại
    os.makedirs(C.ALIGNED_DIR)
    files = sorted(f for f in os.listdir(C.RAW_DIR) if f.lower().endswith(".jpg"))
    so_dat = 0
    with open(C.LANDMARK_CSV, "w", newline="") as f_lm, open(C.LOG_CSV, "w", newline="") as f_log:
        w_lm, w_log = csv.writer(f_lm), csv.writer(f_log)
        w_lm.writerow(["ma_anh"] + [f"{c}{i}" for i in range(68) for c in "xy"])
        w_log.writerow(["ma_anh"] + COT_LOG)
        for name in files:
            ma = os.path.splitext(name)[0]
            kq = process_one(os.path.join(C.RAW_DIR, name))
            w_log.writerow([ma] + [int(kq[c]) if c == "dat" else kq[c] for c in COT_LOG])
            if not kq["dat"]:
                continue
            so_dat += 1
            cv2.imwrite(os.path.join(C.ALIGNED_DIR, ma + ".jpg"), kq["rgb"][:, :, ::-1])
            w_lm.writerow([ma] + [f"{v:.3f}" for v in kq["pts"].ravel()])
    print(f"Tien xu ly xong: {so_dat}/{len(files)} anh dat.")
    return so_dat


if __name__ == "__main__":
    run()
