"""
xu_ly_anh.py - Các phép xử lý ảnh TỰ CODE bằng numpy, dùng chung cho cả pipeline.

- Vai trò: "hộp công cụ" toán học cho tiền xử lý (tien_xu_ly.py) và trích rút (trich_rut.py).
- Quy ước dữ liệu:
  + Ảnh màu: ma trận H x W x 3, thứ tự kênh R, G, B, giá trị 0..255.
  + Ảnh xám: ma trận H x W, giá trị 0..255.
  + Tọa độ điểm: (x, y), x là cột (trái -> phải), y là hàng (trên -> dưới).
- Vì sao tự code: thầy yêu cầu hiểu bản chất; mỗi hàm dưới đây chỉ dùng phép cộng, nhân,
  so sánh trên ma trận numpy, không gọi hàm xử lý ảnh có sẵn của OpenCV.
"""
import os
import numpy as np

# ---------------------------------------------------------------- đường dẫn
THU_MUC_GOC = os.path.dirname(os.path.abspath(__file__))
THU_MUC_ANH_GOC = os.path.join(THU_MUC_GOC, "data", "anh256")        # ảnh thô 256x256
THU_MUC_TIEN_XU_LY = os.path.join(THU_MUC_GOC, "data", "tien_xu_ly")         # kết quả tiền xử lý
THU_MUC_DA_CAN_CHINH = os.path.join(THU_MUC_TIEN_XU_LY, "da_can_chinh")       # ảnh màu đã căn chỉnh, lọc nền (.jpg, để xem)
TEP_DIEM_MOC = os.path.join(THU_MUC_TIEN_XU_LY, "diem_moc.csv")
TEP_NHAT_KY = os.path.join(THU_MUC_TIEN_XU_LY, "nhat_ky.csv")
TEP_DAC_TRUNG = os.path.join(THU_MUC_GOC, "data", "dac_trung.csv")
TEP_MIN_MAX = os.path.join(THU_MUC_GOC, "data", "min_max.json")
TEP_CSDL = os.path.join(THU_MUC_GOC, "data", "khuon_mat.db")          # CSDL SQLite (csdl.py)
TEP_MO_HINH = os.path.join(THU_MUC_GOC, "models", "shape_predictor_68_face_landmarks.dat")

CANH_ANH = 256            # cạnh ảnh (pixel)
TAM_ANH = CANH_ANH / 2.0   # tâm ảnh (128, 128)

# Vị trí chuẩn của hai mắt sau căn chỉnh (dùng cho MỌI ảnh -> mặt nằm giữa, cùng tỉ lệ):
D_MAT = 64.0          # khoảng cách hai tâm mắt = 64 px, xấp xỉ khoảng cách hai mắt ở ảnh gốc (trung vị 63,5 px)
MAT_X = 128.0         # trung điểm hai mắt nằm trên trục dọc giữa ảnh
MAT_Y = 100.0         # ... và ở hàng 100, để trán (phía trên) và cằm (phía dưới) cùng lọt trong khung


# ---------------------------------------------------------------- ảnh xám
def chuyen_xam(rgb):
    """
    - File: xu_ly_anh.py. Dùng để: chuyển ảnh màu thành ảnh xám (bước 2 tiền xử lý).
    - Dựa trên: công thức độ sáng theo cảm nhận của mắt người
      + Y = 0,299*R + 0,587*G + 0,114*B, tổng ba trọng số bằng 1 nên Y vẫn trong [0; 255]
      + mắt nhạy nhất với màu lục nên G có trọng số lớn nhất
    - Ứng dụng: độ nét, dò mặt, LBP chỉ cần độ sáng, không cần màu.
    - Cách khác: lấy trung bình (R + G + B) / 3, đơn giản hơn nhưng sai cảm nhận.
    """
    rgb = rgb.astype(np.float64)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


# ---------------------------------------------------------------- độ nét
def loc_laplace(xam):
    """
    - File: xu_ly_anh.py. Dùng để: lọc ảnh xám bằng toán tử Laplace (đạo hàm bậc 2).
    - Dựa trên: mặt nạ [[0, 1, 0], [1, -4, 1], [0, 1, 0]] (slide XLA5 tr.28–29)
      + L(x, y) = I(x, y-1) + I(x-1, y) + I(x+1, y) + I(x, y+1) - 4*I(x, y)
      + vùng phẳng cho L gần 0, biên cạnh cho |L| lớn
    - Ứng dụng: đầu vào của hàm độ nét phuong_sai_laplace.
    - Cách khác: toán tử Sobel (đạo hàm bậc 1). Viền 1 pixel bị bỏ vì thiếu lân cận.
    """
    g = xam.astype(np.float64)
    tam = g[1:-1, 1:-1]                      # các điểm bên trong (bỏ viền 1 pixel)
    tren, duoi = g[:-2, 1:-1], g[2:, 1:-1]   # lân cận phía trên, phía dưới
    trai, phai = g[1:-1, :-2], g[1:-1, 2:]   # lân cận bên trái, bên phải
    return tren + trai + phai + duoi - 4.0 * tam


def phuong_sai_laplace(xam):
    """
    - File: xu_ly_anh.py. Dùng để: đo độ nét của ảnh (bước 3 tiền xử lý).
    - Dựa trên: phương sai của ảnh đã lọc Laplace
      + Độ nét = (1/N) * tổng (L_i - trung bình L)^2
      + ảnh nét có nhiều biên sắc nên L dao động mạnh, phương sai lớn
    - Ứng dụng: loại ảnh mờ, độ nét nhỏ hơn 20 thì không nạp vào CSDL.
    - Cách khác: đếm số điểm biên sau Sobel, hoặc năng lượng tần số cao (Fourier).
    """
    L = loc_laplace(xam)
    return float(np.mean((L - L.mean()) ** 2))


# ---------------------------------------------------------------- nội suy, xoay
def noi_suy_song_tuyen(anh, xs, ys, nen=255.0):
    """
    - File: xu_ly_anh.py. Dùng để: lấy giá trị ảnh tại tọa độ lẻ (không nguyên).
    - Dựa trên: nội suy song tuyến tính từ 4 điểm nguyên bao quanh
      + điểm (x + a, y + b), 0 <= a, b < 1:
        f = (1-a)(1-b)f(x,y) + a(1-b)f(x+1,y) + (1-a)b f(x,y+1) + ab f(x+1,y+1)
      + điểm rơi ra ngoài ảnh gán giá trị fill (mặc định trắng 255)
    - Ứng dụng: xoay, co giãn ảnh khi căn chỉnh, kéo giãn ảnh, thu nhỏ ảnh.
    - Cách khác: láng giềng gần nhất, nhanh hơn nhưng ảnh bị răng cưa.
    """
    anh = anh.astype(np.float64)
    h, w = anh.shape[:2]
    x0 = np.floor(xs).astype(int)            # góc trên trái của ô 2x2 bao quanh điểm
    y0 = np.floor(ys).astype(int)
    a = xs - x0                              # phần lẻ theo chiều ngang
    b = ys - y0                              # phần lẻ theo chiều dọc
    trong = (x0 >= 0) & (y0 >= 0) & (x0 + 1 <= w - 1) & (y0 + 1 <= h - 1)
    x0 = np.clip(x0, 0, w - 2)               # kẹp chỉ số để không truy cập ngoài mảng
    y0 = np.clip(y0, 0, h - 2)
    if anh.ndim == 3:                        # ảnh màu: nhân trọng số cho cả 3 kênh
        a = a[..., None]
        b = b[..., None]
        trong = trong[..., None]
    f = ((1 - a) * (1 - b) * anh[y0, x0] + a * (1 - b) * anh[y0, x0 + 1]
         + (1 - a) * b * anh[y0 + 1, x0] + a * b * anh[y0 + 1, x0 + 1])
    return np.where(trong, f, nen)


def can_chinh_diem(moc, theta, s, m):
    """
    - File: xu_ly_anh.py. Dùng để: đưa 68 điểm mốc theo phép căn chỉnh (bước 5).
    - Dựa trên: phép đồng dạng = xoay + co giãn + tịnh tiến
      + x' = s * ( cos(theta)*(x - mx) + sin(theta)*(y - my)) + MAT_X
      + y' = s * (-sin(theta)*(x - mx) + cos(theta)*(y - my)) + MAT_Y
      + theta: góc nghiêng hai mắt, s = 64 / khoảng cách hai mắt, m: trung điểm hai mắt
    - Ứng dụng: điểm mốc đi cùng ảnh, hai mắt nằm ngang tại (128; 100).
    - Cách khác: phép biến đổi affine 3 điểm (cv2.getAffineTransform).
    """
    moc = np.asarray(moc, dtype=np.float64)
    c, si = np.cos(theta), np.sin(theta)
    dx = moc[:, 0] - m[0]                    # tọa độ so với trung điểm hai mắt
    dy = moc[:, 1] - m[1]
    x_moi = s * (c * dx + si * dy) + MAT_X
    y_moi = s * (-si * dx + c * dy) + MAT_Y
    return np.stack([x_moi, y_moi], axis=1)


def can_chinh_anh(anh, theta, s, m, nen=255.0):
    """
    - File: xu_ly_anh.py. Dùng để: xoay, co giãn, dịch cả bức ảnh (bước 5 căn chỉnh).
    - Dựa trên: ánh xạ ngược của can_chinh_diem
      + với mỗi điểm (x', y') ảnh mới, tìm điểm (x, y) trên ảnh gốc:
        x = (cos*(x' - MAT_X) - sin*(y' - MAT_Y)) / s + mx
      + lấy giá trị bằng nội suy song tuyến tính nên ảnh mới không bị lỗ trống
    - Ứng dụng: mọi khuôn mặt cùng hướng, cùng cỡ, cùng vị trí.
    - Cách khác: ánh xạ xuôi (đẩy từng điểm sang ảnh mới) để lại lỗ trống.
    """
    ys, xs = np.mgrid[0:CANH_ANH, 0:CANH_ANH].astype(np.float64)   # lưới tọa độ của ảnh mới
    c, si = np.cos(theta), np.sin(theta)
    x_goc = (c * (xs - MAT_X) - si * (ys - MAT_Y)) / s + m[0]
    y_goc = (si * (xs - MAT_X) + c * (ys - MAT_Y)) / s + m[1]
    return noi_suy_song_tuyen(anh, x_goc, y_goc, nen)


# ---------------------------------------------------------------- tô đa giác
def mat_na_da_giac(moc, kich_thuoc=(CANH_ANH, CANH_ANH)):
    """
    - File: xu_ly_anh.py. Dùng để: xác định điểm ảnh nằm trong một đa giác (True/False).
    - Dựa trên: quy tắc chẵn – lẻ (ray casting)
      + từ mỗi điểm kẻ tia ngang sang phải, đếm số cạnh đa giác tia cắt qua
      + số lần cắt lẻ là trong, chẵn là ngoài
      + hoành độ giao điểm: x1 + (y - y1)(x2 - x1)/(y2 - y1)
    - Ứng dụng: mặt nạ lọc nền, vùng cằm – miệng, vùng da hai má.
    - Cách khác: cv2.fillPoly (thư viện), hoặc quy tắc số vòng quấn.
    """
    moc = np.asarray(moc, dtype=np.float64)
    ys, xs = np.mgrid[0:kich_thuoc[0], 0:kich_thuoc[1]].astype(np.float64)
    trong = np.zeros(kich_thuoc, dtype=bool)
    n = len(moc)
    for i in range(n):                       # duyệt từng cạnh (đỉnh i -> đỉnh i+1)
        x1, y1 = moc[i]
        x2, y2 = moc[(i + 1) % n]
        if y1 == y2:                         # cạnh nằm ngang không cắt tia ngang
            continue
        giua = (y1 > ys) != (y2 > ys)        # hàng y nằm giữa hai đầu cạnh
        x_giao = x1 + (ys - y1) * (x2 - x1) / (y2 - y1)
        trong ^= giua & (xs < x_giao)        # mỗi lần cắt thì đảo trạng thái trong/ngoài
    return trong


# ---------------------------------------------------------------- RGB -> HSV
def rgb_sang_hsv(rgb):
    """
    - File: xu_ly_anh.py. Dùng để: đổi màu RGB sang hệ HSV.
    - Dựa trên: hệ màu HSV tách độ sáng khỏi thông tin màu (slide XLA4 tr.36–40)
      + V = max, S = (max - min)/max
      + H = 60*((G-B)/(max-min) mod 6) nếu max là R, cộng 2 nếu max là G, cộng 4 nếu max là B
      + màu xám (max = min) có H = 0
    - Ứng dụng: Color Histogram dùng H, S, bỏ V để bớt ảnh hưởng ánh sáng.
    - Cách khác: hệ Lab, YCbCr, hoặc cv2.cvtColor.
    """
    x = rgb.astype(np.float64) / 255.0       # đưa R, G, B về [0; 1]
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = x.max(axis=-1)
    mn = x.min(axis=-1)
    delta = mx - mn
    d = np.where(delta == 0, 1.0, delta)     # tránh chia 0, kết quả được gán lại bên dưới
    h = np.where(mx == r, ((g - b) / d) % 6.0,
                 np.where(mx == g, (b - r) / d + 2.0, (r - g) / d + 4.0))
    h = np.where(delta == 0, 0.0, 60.0 * h)
    s = np.where(mx == 0, 0.0, delta / np.where(mx == 0, 1.0, mx))
    return h, s, mx


# ---------------------------------------------------------------- LBP
def ma_lbp(xam):
    """
    - File: xu_ly_anh.py. Dùng để: tính mã LBP (mẫu nhị phân cục bộ) cho từng điểm ảnh.
    - Dựa trên: so sánh điểm tâm với 8 lân cận trong khối 3 x 3
      + LBP = tổng_{p=0..7} s(g_p - g_c) * 2^p, s(x) = 1 khi x >= 0
      + p đánh số theo chiều kim đồng hồ từ góc trên trái, mã từ 0 đến 255
    - Ứng dụng: X5 (phương sai LBP cằm – miệng), X6 (mật độ lông mày).
    - Cách khác: GLCM, bộ lọc Gabor, LBP bán kính lớn hơn.
    """
    g = xam.astype(np.float64)
    tam = g[1:-1, 1:-1]
    h, w = g.shape
    #        p0 p1 p2
    #        p7  c p3        (dy, dx) của 8 lân cận theo thứ tự p = 0..7
    #        p6 p5 p4
    lech = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    ma = np.zeros(tam.shape, dtype=np.int32)
    for p, (dy, dx) in enumerate(lech):
        lan_can = g[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        ma += (lan_can >= tam).astype(np.int32) << p      # bit p = 1 nếu lân cận >= tâm; << p là nhân 2^p
    ket_qua = np.zeros((h, w), dtype=np.int32)                # viền ngoài 1 pixel (thiếu lân cận) gán 0
    ket_qua[1:-1, 1:-1] = ma
    return ket_qua


# ---------------------------------------------------------------- ngưỡng Otsu
def nguong_otsu(gia_tri):
    """
    - File: xu_ly_anh.py. Dùng để: tìm ngưỡng tự động chia điểm ảnh thành tối và sáng.
    - Dựa trên: phương pháp Otsu (slide XLA5 tr.65–67)
      + thử mọi ngưỡng t = 0..255, chia thành lớp <= t và lớp > t
      + chọn t có phương sai giữa hai lớp lớn nhất: w0*w1*(mu0 - mu1)^2
      + w0, w1 là tỉ lệ số điểm, mu0, mu1 là độ xám trung bình của mỗi lớp
    - Ứng dụng: tách sợi lông mày (tối) khỏi da (sáng) cho X6.
    - Cách khác: ngưỡng cố định, cv2.threshold với cờ OTSU.
    """
    v = np.clip(np.round(gia_tri), 0, 255).astype(int).ravel()
    luoc_do = np.bincount(v, minlength=256).astype(np.float64)   # số điểm ở từng mức xám
    p = luoc_do / luoc_do.sum()                                     # tỉ lệ điểm ở từng mức xám
    muc = np.arange(256)
    w0 = np.cumsum(p)                 # tỉ lệ lớp tối (<= t)
    w1 = 1.0 - w0                     # tỉ lệ lớp sáng (> t)
    tong0 = np.cumsum(p * muc)        # tổng (mức xám * tỉ lệ) của lớp tối
    ok = (w0 > 0) & (w1 > 0)          # chỉ xét ngưỡng mà cả hai lớp đều có điểm
    mu0 = np.where(ok, tong0 / np.where(ok, w0, 1), 0)
    mu1 = np.where(ok, (tong0[-1] - tong0) / np.where(ok, w1, 1), 0)
    sigma_b = np.where(ok, w0 * w1 * (mu0 - mu1) ** 2, -1.0)
    return int(np.argmax(sigma_b))


# ---------------------------------------------------------------- khoảng cách
def khoang_cach_diem(p, q):
    """
    - File: xu_ly_anh.py. Dùng để: đo khoảng cách giữa hai điểm trên ảnh.
    - Dựa trên: khoảng cách Euclid (chuẩn L2, slide bài 8 tr.17)
      + d = căn((xP - xQ)^2 + (yP - yQ)^2)
    - Ứng dụng: đo các đoạn cho 5 tỉ số hình học X0 – X4, khoảng cách hai mắt
      để căn chỉnh và để chia độ lệch điểm mốc.
    - Cách khác: khoảng cách Manhattan |xP - xQ| + |yP - yQ|, tính nhanh hơn nhưng phụ
      thuộc hướng của đoạn thẳng.
    """
    return float(np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2))


def tam_hai_mat(moc):
    """
    - File: xu_ly_anh.py. Dùng để: lấy tâm hai mắt từ 68 điểm mốc.
    - Dựa trên: trung bình tọa độ 6 điểm viền mỗi mắt
      + tâm mắt trái ảnh = trung bình điểm 36 – 41
      + tâm mắt phải ảnh = trung bình điểm 42 – 47
    - Ứng dụng: góc nghiêng và hệ số co giãn khi căn chỉnh, X2, X3, mặt nạ trán,
      độ lệch điểm mốc.
    - Cách khác: dùng khóe mắt (điểm 36, 45), nhưng dễ lệch khi mắt nheo.
    """
    moc = np.asarray(moc, dtype=np.float64)
    return moc[36:42].mean(axis=0), moc[42:48].mean(axis=0)
