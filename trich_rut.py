"""
trich_rut.py - Trích rút vector đặc trưng 39 chiều cho một ảnh đã tiền xử lý.

- Ba bộ thuộc tính (Báo cáo 1, Phần 2):
  + Hình học (5)  X0 – X4  : tỉ số khoảng cách giữa các điểm mốc (dáng mặt, tỉ lệ ngũ quan)
  + Kết cấu  (2)  X5, X6   : phương sai LBP vùng cằm – miệng, mật độ lông mày
  + Màu sắc  (32) X7 – X38 : Color Histogram H x S (8 x 4 ô) trên vùng da (2 má + trán)
- Đầu vào: ảnh màu đã căn chỉnh + lọc nền (rgb), ảnh xám đã căn chỉnh (gray), 68 điểm mốc (pts).
- Đầu ra: data/dac_trung.csv lưu giá trị GỐC (chưa chuẩn hóa); chuẩn hóa làm ở tim_kiem.py.
- Chạy:  python trich_rut.py
"""
import os
import csv
import numpy as np
import cv2

import xu_ly_anh as XL
import tien_xu_ly as TXL

SO_O_H, SO_O_S = 8, 4      # số ô của kênh H và kênh S
H_MAX = 40.0               # dải sắc độ da được chia ô: 0..40 độ (5 độ / ô)

TEN_HINH_HOC = ["hh_khung_mat", "hh_goma_ham", "hh_kc_mat", "hh_rong_mui", "hh_cao_mui"]
TEN_KET_CAU = ["kc_lbp_cam", "kc_long_may"]
TEN_MAU = [f"ms_h{i}_s{j}" for i in range(SO_O_H) for j in range(SO_O_S)]
TEN_COT = TEN_HINH_HOC + TEN_KET_CAU + TEN_MAU       # 5 + 2 + 32 = 39 tên cột

# vị trí của từng bộ trong vector 39 chiều (dùng lại ở tim_kiem.py)
BO = {"hinh_hoc": slice(0, 5), "ket_cau": slice(5, 7), "mau_sac": slice(7, 39)}

# tên tiếng Việt của X0 - X38 (dùng cho giao diện, đúng Bảng 2.1 Báo cáo 1)
TEN_DAC_TRUNG = (["Tỉ lệ khung mặt", "Tỉ lệ gò má trên quai hàm", "Tỉ lệ khoảng cách hai mắt",
                  "Tỉ lệ độ rộng cánh mũi", "Tỉ lệ chiều cao mũi",
                  "Phương sai LBP vùng cằm - miệng", "Mật độ lông mày"]
                 + [f"Color Histogram vùng da, ô H{i} - ô S{j}" for i in range(SO_O_H) for j in range(SO_O_S)])

# vai trò theo Bảng 2.1 Báo cáo 1: X0, X1 (dáng mặt) và X7 - X38 (tông da) tìm sự tương đồng;
# X2 - X4 (tỉ lệ ngũ quan) và X5, X6 (bề mặt) tìm sự khác biệt.
VAI_TRO = ["tuong_dong"] * 2 + ["khac_biet"] * 5 + ["tuong_dong"] * 32


# ---------------------------------------------------------------- hình học
def dac_trung_hinh_hoc(moc):
    """
    - File: trich_rut.py. Dùng để: tính 5 đặc trưng hình học X0 – X4.
    - Dựa trên: tỉ số hai khoảng cách giữa điểm mốc (d: Euclid, dY: theo trục đứng)
      + X0 = d(P0,P16)/dY(P27,P8), X1 = d(P0,P16)/d(P4,P12)
      + X2 = d(hai tâm mắt)/d(P0,P16), X3 = d(P31,P35)/d(hai tâm mắt)
      + X4 = dY(P27,P33)/dY(P27,P8)
    - Ứng dụng: tỉ số không đổi khi mặt to, nhỏ; X0, X1 tìm tương đồng, X2 – X4 tìm khác biệt.
    - Cách khác: đo khoảng cách tuyệt đối (phụ thuộc cỡ mặt).
    """
    trai, phai = XL.tam_hai_mat(moc)
    rong_mat = XL.khoang_cach_diem(moc[0], moc[16])           # bề rộng mặt ngang thái dương
    cao_mat = abs(moc[27][1] - moc[8][1])        # chiều cao từ gốc mũi đến đỉnh cằm
    rong_ham = XL.khoang_cach_diem(moc[4], moc[12])           # bề rộng hai góc quai hàm
    kc_mat = XL.khoang_cach_diem(trai, phai)                  # khoảng cách hai tâm mắt
    rong_mui = XL.khoang_cach_diem(moc[31], moc[35])          # bề ngang hai cánh mũi
    cao_mui = abs(moc[27][1] - moc[33][1])       # chiều dài mũi
    x0 = rong_mat / cao_mat
    x1 = rong_mat / rong_ham
    x2 = kc_mat / rong_mat
    x3 = rong_mui / kc_mat
    x4 = cao_mui / cao_mat
    return [x0, x1, x2, x3, x4]


# ---------------------------------------------------------------- các vùng trên mặt
def vung_cam(moc):
    """
    - File: trich_rut.py. Dùng để: tạo mặt nạ vùng cằm – miệng.
    - Dựa trên: đa giác nối 6 điểm mốc
      + P33 (đáy mũi), P54 và P48 (hai khóe miệng), P12 và P4 (quai hàm), P8 (đỉnh cằm)
      + tô đa giác bằng quy tắc chẵn – lẻ (xu_ly_anh.mat_na_da_giac)
    - Ứng dụng: vùng tính X5 (phương sai LBP), phân biệt cằm – miệng nhẵn hay nhiều chi tiết.
    - Cách khác: hình chữ nhật bao quanh miệng, đơn giản hơn nhưng lẫn má và nền.
    """
    return XL.mat_na_da_giac(moc[[33, 54, 12, 8, 4, 48]])


def vung_long_may(moc):
    """
    - File: trich_rut.py. Dùng để: tạo hai mặt nạ vùng lông mày (trái, phải ảnh).
    - Dựa trên: hộp chữ nhật bao các điểm mốc lông mày
      + trái ảnh: điểm 17 – 21; phải ảnh: điểm 22 – 26
      + nới lên và xuống 0,15 x khoảng cách hai mắt (9,6 px) để lấy trọn bề dày lông mày
    - Ứng dụng: vùng tính X6 (mật độ lông mày).
    - Cách khác: đa giác ôm sát lông mày, chính xác hơn nhưng phức tạp và dễ hụt.
    """
    trai, phai = XL.tam_hai_mat(moc)
    noi = 0.15 * XL.khoang_cach_diem(trai, phai)
    ys, xs = np.mgrid[0:XL.CANH_ANH, 0:XL.CANH_ANH]
    vung = []
    for nhom in (moc[17:22], moc[22:27]):        # lông mày trái ảnh, rồi phải ảnh
        x1, x2 = nhom[:, 0].min(), nhom[:, 0].max()
        y1, y2 = nhom[:, 1].min() - noi, nhom[:, 1].max() + noi
        vung.append((xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2))
    return vung


def vung_da(moc):
    """
    - File: trich_rut.py. Dùng để: tạo mặt nạ vùng da để tính màu.
    - Dựa trên: ghép 3 vùng từ điểm mốc
      + má trái: đa giác điểm 1, 2, 3, 31, 40, 41; má phải: 15, 14, 13, 35, 47, 46
      + trán: chữ nhật từ x(P19) đến x(P24), cách đỉnh lông mày 0,10 – 0,45 lần khoảng cách hai mắt
    - Ứng dụng: Color Histogram chỉ đo màu da, không lẫn mắt, lông mày, miệng.
    - Cách khác: lọc da theo ngưỡng màu, dễ lẫn tóc và nền.
    """
    trai, phai = XL.tam_hai_mat(moc)
    d = XL.khoang_cach_diem(trai, phai)
    ma = XL.mat_na_da_giac(moc[[1, 2, 3, 31, 40, 41]]) | XL.mat_na_da_giac(moc[[15, 14, 13, 35, 47, 46]])
    ys, xs = np.mgrid[0:XL.CANH_ANH, 0:XL.CANH_ANH]
    dinh = moc[17:27, 1].min()                   # đỉnh cao nhất của hai lông mày
    tran = (xs >= moc[19, 0]) & (xs <= moc[24, 0]) & (ys >= dinh - 0.45 * d) & (ys <= dinh - 0.10 * d)
    return ma | tran


# ---------------------------------------------------------------- kết cấu
def dac_trung_ket_cau(xam, moc, co_du_lieu):
    """
    - File: trich_rut.py. Dùng để: tính 2 đặc trưng kết cấu X5, X6.
    - Dựa trên: mã LBP và ngưỡng Otsu (xu_ly_anh.py)
      + X5 = phương sai mã LBP vùng cằm – miệng = (1/N) tổng (c_i - mu)^2, miền [0; 16256,25]
      + X6 = số điểm "sợi lông" / số điểm vùng lông mày, trung bình hai bên, miền [0; 1]
      + "sợi lông": tối hơn hoặc bằng ngưỡng Otsu và mã LBP khác 0, 255
    - Ứng dụng: tìm khác biệt về bề mặt. co_du_lieu loại phần trắng thêm vào khi căn chỉnh.
    - Cách khác: GLCM, Gabor.
    """
    ma = XL.ma_lbp(xam)

    # X5: phương sai LBP trong vùng cằm - miệng
    cam = vung_cam(moc) & co_du_lieu
    lbp_cam = float(np.var(ma[cam]))

    # X6: mật độ lông mày, tính riêng từng bên rồi lấy trung bình
    mat_do = []
    for vung in vung_long_may(moc):
        vung = vung & co_du_lieu
        if vung.sum() == 0:                      # vùng rơi hết ra ngoài ảnh: bỏ qua bên này
            continue
        xam_vung = xam[vung]
        nguong = XL.nguong_otsu(xam_vung)
        soi = (xam_vung <= nguong) & (ma[vung] != 0) & (ma[vung] != 255)
        mat_do.append(soi.sum() / float(vung.sum()))
    long_may = float(np.mean(mat_do)) if mat_do else 0.0
    return [lbp_cam, long_may]


# ---------------------------------------------------------------- màu sắc
def o_hs(h, s):
    """
    - File: trich_rut.py. Dùng để: xếp mỗi điểm da vào một ô H và một ô S.
    - Dựa trên: lượng tử hóa đều trong dải màu da
      + H: 8 ô đều trong 0 – 40 độ, mỗi ô 5 độ; H >= 330 (đỏ sát 360) vào ô đầu, 40 – 330 vào ô cuối
      + S: 4 ô đều trong [0; 1], mỗi ô 0,25; S = 1 thuộc ô cuối
    - Ứng dụng: tạo 32 ô của Color Histogram (X7 – X38).
    - Cách khác: chia đều 0 – 360 độ, nhưng da chỉ nằm ở vài ô đầu.
    """
    o_h = np.floor(h / (H_MAX / SO_O_H)).astype(int)            # H / 5 độ -> chỉ số ô
    o_h = np.where(h >= 330.0, 0, np.clip(o_h, 0, SO_O_H - 1))
    o_s = np.clip(np.floor(s * SO_O_S).astype(int), 0, SO_O_S - 1)
    return o_h, o_s


def dac_trung_mau(rgb, moc, co_du_lieu):
    """
    - File: trich_rut.py. Dùng để: tính 32 đặc trưng màu X7 – X38.
    - Dựa trên: Color Histogram chuẩn hóa (slide bài 11 tr.9; Guojun Lu tr.133–134)
      + đổi các điểm da sang HSV, xếp vào 8 x 4 ô (H, S), bỏ kênh V
      + h_k = n_k / N: tỉ lệ điểm da rơi vào ô k, tổng 32 giá trị bằng 1
      + thứ tự ô k = (ô H) * 4 + (ô S)
    - Ứng dụng: tìm tương đồng về tông da, ít phụ thuộc ánh sáng.
    - Cách khác: color moments (trung bình, độ lệch chuẩn).
    """
    da = vung_da(moc) & co_du_lieu
    h, s, _ = XL.rgb_sang_hsv(rgb[da])
    o_h, o_s = o_hs(h, s)
    dem = np.bincount(o_h * SO_O_S + o_s, minlength=SO_O_H * SO_O_S).astype(np.float64)   # n_k
    return list(dem / max(dem.sum(), 1.0))                                                 # n_k / N


# ---------------------------------------------------------------- ghép vector
def trich_vector(rgb, xam, moc):
    """
    - File: trich_rut.py. Dùng để: tạo vector 39 chiều của một ảnh đã tiền xử lý.
    - Dựa trên: ghép 3 bộ thuộc tính theo đúng thứ tự TEN_COT
      + 5 hình học + 2 kết cấu + 32 màu sắc = 39 giá trị gốc
      + điểm trắng tuyệt đối (255, 255, 255) là nền, không đưa vào thống kê
    - Ứng dụng: dùng chung cho ảnh nạp vào CSDL và ảnh truy vấn, nên hai bên được tính
      theo cùng một cách.
    - Cách khác: một vector duy nhất từ mạng học sâu (nhóm không dùng).
    """
    moc = np.asarray(moc, dtype=np.float64)
    co_du_lieu = ~(rgb == 255).all(axis=2)       # True = điểm có dữ liệu thật (không phải nền trắng)
    v = dac_trung_hinh_hoc(moc) + dac_trung_ket_cau(xam, moc, co_du_lieu) + dac_trung_mau(rgb, moc, co_du_lieu)
    return np.array(v, dtype=np.float64)


def doc_anh_da_xu_ly(ma):
    """
    - File: trich_rut.py. Dùng để: đọc ảnh đã tiền xử lý của một mã ảnh để XEM, vẽ hình.
    - Dựa trên: đọc tệp JPG trong data/tien_xu_ly/da_can_chinh bằng OpenCV
      + OpenCV trả thứ tự kênh B, G, R nên đảo lại thành R, G, B
      + trả về (ảnh màu, ảnh xám tính bằng xu_ly_anh.chuyen_xam)
    - Ứng dụng: giao diện, notebook vẽ hình.
    - Cách khác: không dùng để tính đặc trưng, vì nén JPG làm đổi nhẹ giá trị điểm ảnh.
    """
    rgb = cv2.imread(os.path.join(XL.THU_MUC_DA_CAN_CHINH, ma + ".jpg"), cv2.IMREAD_COLOR)[:, :, ::-1]
    return rgb, np.round(XL.chuyen_xam(rgb)).astype(np.uint8)


def doc_diem_moc():
    """
    - File: trich_rut.py. Dùng để: đọc tọa độ 68 điểm mốc đã lưu của mọi ảnh.
    - Dựa trên: tệp data/tien_xu_ly/diem_moc.csv
      + mỗi dòng: mã ảnh + 136 số (x0, y0, ..., x67, y67)
      + đổi thành ma trận 68 x 2 cho từng ảnh
    - Ứng dụng: lấy danh sách ảnh đạt tiền xử lý, vẽ điểm mốc trên giao diện.
    - Cách khác: chạy lại dlib cho từng ảnh, chậm hơn nhiều (khoảng 3,5 phút cả bộ).
    """
    with open(XL.TEP_DIEM_MOC, newline="") as f:
        ds_dong = list(csv.reader(f))[1:]           # bỏ dòng tiêu đề
    return {r[0]: np.array(r[1:], dtype=np.float64).reshape(68, 2) for r in ds_dong}


def chay():
    """
    - File: trich_rut.py. Dùng để: trích đặc trưng cho cả bộ dữ liệu, lưu data/dac_trung.csv.
    - Dựa trên: với mỗi ảnh đạt tiền xử lý
      + chạy lại tien_xu_ly.xu_ly_mot_anh rồi trich_vector ngay trên kết quả
      + ghi mã ảnh + 39 giá trị, làm tròn 6 chữ số thập phân
    - Ứng dụng: ảnh trong CSDL và ảnh truy vấn đi đúng cùng một đường, nên so sánh được
      với nhau. Tệp này là đầu vào của csdl.tao_csdl.
    - Cách khác: đọc ảnh đã lưu JPG (sai lệch nhẹ do nén).
    """
    ds_moc = doc_diem_moc()
    with open(XL.TEP_DAC_TRUNG, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ma_anh"] + TEN_COT)
        for ma in sorted(ds_moc):
            kq = TXL.xu_ly_mot_anh(os.path.join(XL.THU_MUC_ANH_GOC, ma + ".jpg"))
            w.writerow([ma] + [f"{x:.6f}" for x in trich_vector(kq["rgb"], kq["xam"], kq["moc"])])
    print(f"Trich dac trung xong: {len(ds_moc)} anh x {len(TEN_COT)} chieu.")
    return len(ds_moc)


if __name__ == "__main__":
    chay()
