"""
common.py - Các phép xử lý ảnh TỰ CODE bằng numpy, dùng chung cho cả pipeline.

- Vai trò: "hộp công cụ" toán học cho tiền xử lý (preprocess.py) và trích rút (features.py).
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
ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, "data", "raw256")        # ảnh thô 256x256
PREP_DIR = os.path.join(ROOT, "data", "prep")         # kết quả tiền xử lý
ALIGNED_DIR = os.path.join(PREP_DIR, "aligned")       # ảnh màu đã căn chỉnh, lọc nền (.jpg, để xem)
LANDMARK_CSV = os.path.join(PREP_DIR, "landmarks.csv")
LOG_CSV = os.path.join(PREP_DIR, "preprocess_log.csv")
FEATURE_CSV = os.path.join(ROOT, "data", "features.csv")
NORM_JSON = os.path.join(ROOT, "data", "norm_params.json")
DB_PATH = os.path.join(ROOT, "data", "khuon_mat.db")          # CSDL SQLite (database.py)
MODEL_PATH = os.path.join(ROOT, "models", "shape_predictor_68_face_landmarks.dat")

SIZE = 256            # cạnh ảnh (pixel)
CENTER = SIZE / 2.0   # tâm ảnh (128, 128)

# Vị trí chuẩn của hai mắt sau căn chỉnh (dùng cho MỌI ảnh -> mặt nằm giữa, cùng tỉ lệ):
D_MAT = 64.0          # khoảng cách hai tâm mắt = 64 px, xấp xỉ khoảng cách hai mắt ở ảnh gốc (trung vị 63,5 px)
MAT_X = 128.0         # trung điểm hai mắt nằm trên trục dọc giữa ảnh
MAT_Y = 100.0         # ... và ở hàng 100, để trán (phía trên) và cằm (phía dưới) cùng lọt trong khung


# ---------------------------------------------------------------- ảnh xám
def to_gray(rgb):
    """
    - File: common.py. Dùng để: chuyển ảnh màu thành ảnh xám (bước 2 tiền xử lý).
    - Dựa trên: công thức độ sáng theo cảm nhận của mắt người
      + Y = 0,299*R + 0,587*G + 0,114*B, tổng ba trọng số bằng 1 nên Y vẫn trong [0; 255]
      + mắt nhạy nhất với màu lục nên G có trọng số lớn nhất
    - Ứng dụng: độ nét, dò mặt, LBP chỉ cần độ sáng, không cần màu.
    - Cách khác: lấy trung bình (R + G + B) / 3, đơn giản hơn nhưng sai cảm nhận.
    """
    rgb = rgb.astype(np.float64)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


# ---------------------------------------------------------------- độ nét
def laplacian(gray):
    """
    - File: common.py. Dùng để: lọc ảnh xám bằng toán tử Laplace (đạo hàm bậc 2).
    - Dựa trên: mặt nạ [[0, 1, 0], [1, -4, 1], [0, 1, 0]] (slide XLA5 tr.28–29)
      + L(x, y) = I(x, y-1) + I(x-1, y) + I(x+1, y) + I(x, y+1) - 4*I(x, y)
      + vùng phẳng cho L gần 0, biên cạnh cho |L| lớn
    - Ứng dụng: đầu vào của hàm độ nét laplacian_var.
    - Cách khác: toán tử Sobel (đạo hàm bậc 1). Viền 1 pixel bị bỏ vì thiếu lân cận.
    """
    g = gray.astype(np.float64)
    tam = g[1:-1, 1:-1]                      # các điểm bên trong (bỏ viền 1 pixel)
    tren, duoi = g[:-2, 1:-1], g[2:, 1:-1]   # lân cận phía trên, phía dưới
    trai, phai = g[1:-1, :-2], g[1:-1, 2:]   # lân cận bên trái, bên phải
    return tren + trai + phai + duoi - 4.0 * tam


def laplacian_var(gray):
    """
    - File: common.py. Dùng để: đo độ nét của ảnh (bước 3 tiền xử lý).
    - Dựa trên: phương sai của ảnh đã lọc Laplace
      + Độ nét = (1/N) * tổng (L_i - trung bình L)^2
      + ảnh nét có nhiều biên sắc nên L dao động mạnh, phương sai lớn
    - Ứng dụng: loại ảnh mờ, độ nét nhỏ hơn 20 thì không nạp vào CSDL.
    - Cách khác: đếm số điểm biên sau Sobel, hoặc năng lượng tần số cao (Fourier).
    """
    L = laplacian(gray)
    return float(np.mean((L - L.mean()) ** 2))


# ---------------------------------------------------------------- nội suy, xoay
def bilinear_sample(img, xs, ys, fill=255.0):
    """
    - File: common.py. Dùng để: lấy giá trị ảnh tại tọa độ lẻ (không nguyên).
    - Dựa trên: nội suy song tuyến tính từ 4 điểm nguyên bao quanh
      + điểm (x + a, y + b), 0 <= a, b < 1:
        f = (1-a)(1-b)f(x,y) + a(1-b)f(x+1,y) + (1-a)b f(x,y+1) + ab f(x+1,y+1)
      + điểm rơi ra ngoài ảnh gán giá trị fill (mặc định trắng 255)
    - Ứng dụng: xoay, co giãn ảnh khi căn chỉnh, kéo giãn ảnh, thu nhỏ ảnh.
    - Cách khác: láng giềng gần nhất, nhanh hơn nhưng ảnh bị răng cưa.
    """
    img = img.astype(np.float64)
    h, w = img.shape[:2]
    x0 = np.floor(xs).astype(int)            # góc trên trái của ô 2x2 bao quanh điểm
    y0 = np.floor(ys).astype(int)
    a = xs - x0                              # phần lẻ theo chiều ngang
    b = ys - y0                              # phần lẻ theo chiều dọc
    trong = (x0 >= 0) & (y0 >= 0) & (x0 + 1 <= w - 1) & (y0 + 1 <= h - 1)
    x0 = np.clip(x0, 0, w - 2)               # kẹp chỉ số để không truy cập ngoài mảng
    y0 = np.clip(y0, 0, h - 2)
    if img.ndim == 3:                        # ảnh màu: nhân trọng số cho cả 3 kênh
        a = a[..., None]
        b = b[..., None]
        trong = trong[..., None]
    f = ((1 - a) * (1 - b) * img[y0, x0] + a * (1 - b) * img[y0, x0 + 1]
         + (1 - a) * b * img[y0 + 1, x0] + a * b * img[y0 + 1, x0 + 1])
    return np.where(trong, f, fill)


def align_points(pts, theta, s, m):
    """
    - File: common.py. Dùng để: đưa 68 điểm mốc theo phép căn chỉnh (bước 5).
    - Dựa trên: phép đồng dạng = xoay + co giãn + tịnh tiến
      + x' = s * ( cos(theta)*(x - mx) + sin(theta)*(y - my)) + MAT_X
      + y' = s * (-sin(theta)*(x - mx) + cos(theta)*(y - my)) + MAT_Y
      + theta: góc nghiêng hai mắt, s = 64 / khoảng cách hai mắt, m: trung điểm hai mắt
    - Ứng dụng: điểm mốc đi cùng ảnh, hai mắt nằm ngang tại (128; 100).
    - Cách khác: phép biến đổi affine 3 điểm (cv2.getAffineTransform).
    """
    pts = np.asarray(pts, dtype=np.float64)
    c, si = np.cos(theta), np.sin(theta)
    dx = pts[:, 0] - m[0]                    # tọa độ so với trung điểm hai mắt
    dy = pts[:, 1] - m[1]
    x_moi = s * (c * dx + si * dy) + MAT_X
    y_moi = s * (-si * dx + c * dy) + MAT_Y
    return np.stack([x_moi, y_moi], axis=1)


def align_image(img, theta, s, m, fill=255.0):
    """
    - File: common.py. Dùng để: xoay, co giãn, dịch cả bức ảnh (bước 5 căn chỉnh).
    - Dựa trên: ánh xạ ngược của align_points
      + với mỗi điểm (x', y') ảnh mới, tìm điểm (x, y) trên ảnh gốc:
        x = (cos*(x' - MAT_X) - sin*(y' - MAT_Y)) / s + mx
      + lấy giá trị bằng nội suy song tuyến tính nên ảnh mới không bị lỗ trống
    - Ứng dụng: mọi khuôn mặt cùng hướng, cùng cỡ, cùng vị trí.
    - Cách khác: ánh xạ xuôi (đẩy từng điểm sang ảnh mới) để lại lỗ trống.
    """
    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)   # lưới tọa độ của ảnh mới
    c, si = np.cos(theta), np.sin(theta)
    src_x = (c * (xs - MAT_X) - si * (ys - MAT_Y)) / s + m[0]
    src_y = (si * (xs - MAT_X) + c * (ys - MAT_Y)) / s + m[1]
    return bilinear_sample(img, src_x, src_y, fill)


# ---------------------------------------------------------------- tô đa giác
def polygon_mask(pts, shape=(SIZE, SIZE)):
    """
    - File: common.py. Dùng để: xác định điểm ảnh nằm trong một đa giác (True/False).
    - Dựa trên: quy tắc chẵn – lẻ (ray casting)
      + từ mỗi điểm kẻ tia ngang sang phải, đếm số cạnh đa giác tia cắt qua
      + số lần cắt lẻ là trong, chẵn là ngoài
      + hoành độ giao điểm: x1 + (y - y1)(x2 - x1)/(y2 - y1)
    - Ứng dụng: mặt nạ lọc nền, vùng cằm – miệng, vùng da hai má.
    - Cách khác: cv2.fillPoly (thư viện), hoặc quy tắc số vòng quấn.
    """
    pts = np.asarray(pts, dtype=np.float64)
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    trong = np.zeros(shape, dtype=bool)
    n = len(pts)
    for i in range(n):                       # duyệt từng cạnh (đỉnh i -> đỉnh i+1)
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if y1 == y2:                         # cạnh nằm ngang không cắt tia ngang
            continue
        giua = (y1 > ys) != (y2 > ys)        # hàng y nằm giữa hai đầu cạnh
        x_giao = x1 + (ys - y1) * (x2 - x1) / (y2 - y1)
        trong ^= giua & (xs < x_giao)        # mỗi lần cắt thì đảo trạng thái trong/ngoài
    return trong


# ---------------------------------------------------------------- RGB -> HSV
def rgb_to_hsv(rgb):
    """
    - File: common.py. Dùng để: đổi màu RGB sang hệ HSV.
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
def lbp(gray):
    """
    - File: common.py. Dùng để: tính mã LBP (mẫu nhị phân cục bộ) cho từng điểm ảnh.
    - Dựa trên: so sánh điểm tâm với 8 lân cận trong khối 3 x 3
      + LBP = tổng_{p=0..7} s(g_p - g_c) * 2^p, s(x) = 1 khi x >= 0
      + p đánh số theo chiều kim đồng hồ từ góc trên trái, mã từ 0 đến 255
    - Ứng dụng: X5 (phương sai LBP cằm – miệng), X6 (mật độ lông mày).
    - Cách khác: GLCM, bộ lọc Gabor, LBP bán kính lớn hơn.
    """
    g = gray.astype(np.float64)
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
    out = np.zeros((h, w), dtype=np.int32)                # viền ngoài 1 pixel (thiếu lân cận) gán 0
    out[1:-1, 1:-1] = ma
    return out


# ---------------------------------------------------------------- ngưỡng Otsu
def otsu_threshold(values):
    """
    - File: common.py. Dùng để: tìm ngưỡng tự động chia điểm ảnh thành tối và sáng.
    - Dựa trên: phương pháp Otsu (slide XLA5 tr.65–67)
      + thử mọi ngưỡng t = 0..255, chia thành lớp <= t và lớp > t
      + chọn t có phương sai giữa hai lớp lớn nhất: w0*w1*(mu0 - mu1)^2
      + w0, w1 là tỉ lệ số điểm, mu0, mu1 là độ xám trung bình của mỗi lớp
    - Ứng dụng: tách sợi lông mày (tối) khỏi da (sáng) cho X6.
    - Cách khác: ngưỡng cố định, cv2.threshold với cờ OTSU.
    """
    v = np.clip(np.round(values), 0, 255).astype(int).ravel()
    hist = np.bincount(v, minlength=256).astype(np.float64)   # số điểm ở từng mức xám
    p = hist / hist.sum()                                     # tỉ lệ điểm ở từng mức xám
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
def dist(p, q):
    """
    - File: common.py. Dùng để: đo khoảng cách giữa hai điểm trên ảnh.
    - Dựa trên: khoảng cách Euclid (chuẩn L2, slide bài 8 tr.17)
      + d = căn((xP - xQ)^2 + (yP - yQ)^2)
    - Ứng dụng: đo các đoạn cho 5 tỉ số hình học X0 – X4, khoảng cách hai mắt
      để căn chỉnh và để chia độ lệch điểm mốc.
    - Cách khác: khoảng cách Manhattan |xP - xQ| + |yP - yQ|, tính nhanh hơn nhưng phụ
      thuộc hướng của đoạn thẳng.
    """
    return float(np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2))


def eye_centers(pts):
    """
    - File: common.py. Dùng để: lấy tâm hai mắt từ 68 điểm mốc.
    - Dựa trên: trung bình tọa độ 6 điểm viền mỗi mắt
      + tâm mắt trái ảnh = trung bình điểm 36 – 41
      + tâm mắt phải ảnh = trung bình điểm 42 – 47
    - Ứng dụng: góc nghiêng và hệ số co giãn khi căn chỉnh, X2, X3, mặt nạ trán,
      độ lệch điểm mốc.
    - Cách khác: dùng khóe mắt (điểm 36, 45), nhưng dễ lệch khi mắt nheo.
    """
    pts = np.asarray(pts, dtype=np.float64)
    return pts[36:42].mean(axis=0), pts[42:48].mean(axis=0)
