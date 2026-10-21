"""
common.py - Các phép xử lý ảnh TỰ CODE bằng numpy, dùng chung cho cả pipeline.

Quy ước:
- Ảnh màu: ma trận H x W x 3, thứ tự kênh R, G, B, giá trị 0..255.
- Ảnh xám: ma trận H x W, giá trị 0..255.
- Tọa độ điểm: (x, y) với x là cột (trái -> phải), y là hàng (trên -> dưới).
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
    """Chuyển ảnh màu sang ảnh xám.

    Công thức: Y = 0,299*R + 0,587*G + 0,114*B
    Tổng 3 trọng số = 1 nên Y vẫn nằm trong [0, 255].
    Trả về ma trận số thực H x W.
    """
    rgb = rgb.astype(np.float64)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


# ---------------------------------------------------------------- độ nét
def laplacian_var(gray):
    """Độ nét = phương sai của ảnh sau khi lọc Laplace.

    Laplace (mặt nạ [[0,1,0],[1,-4,1],[0,1,0]]):
        L(x,y) = I(x,y-1) + I(x-1,y) + I(x+1,y) + I(x,y+1) - 4*I(x,y)
    Độ nét = (1/N) * tổng (L_i - trung bình L)^2
    Chỉ tính cho các điểm bên trong (bỏ viền 1 pixel vì thiếu lân cận).
    """
    g = gray.astype(np.float64)
    tam = g[1:-1, 1:-1]
    L = g[:-2, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] + g[2:, 1:-1] - 4.0 * tam
    return float(np.mean((L - L.mean()) ** 2))


# ---------------------------------------------------------------- nội suy, xoay
def bilinear_sample(img, xs, ys, fill=255.0):
    """Lấy giá trị ảnh tại các tọa độ thực (xs, ys) bằng nội suy song tuyến tính.

    Điểm cần tính ở (x + a, y + b) với x, y nguyên và 0 <= a, b < 1:
        f = (1-a)(1-b)*f(x,y) + a(1-b)*f(x+1,y) + (1-a)b*f(x,y+1) + ab*f(x+1,y+1)
    Điểm rơi ra ngoài ảnh được gán giá trị `fill` (mặc định trắng 255).
    """
    img = img.astype(np.float64)
    h, w = img.shape[:2]
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    a = xs - x0
    b = ys - y0
    trong = (x0 >= 0) & (y0 >= 0) & (x0 + 1 <= w - 1) & (y0 + 1 <= h - 1)
    x0 = np.clip(x0, 0, w - 2)
    y0 = np.clip(y0, 0, h - 2)
    if img.ndim == 3:            # ảnh màu: nhân trọng số cho cả 3 kênh
        a = a[..., None]
        b = b[..., None]
        trong = trong[..., None]
    f = ((1 - a) * (1 - b) * img[y0, x0] + a * (1 - b) * img[y0, x0 + 1]
         + (1 - a) * b * img[y0 + 1, x0] + a * b * img[y0 + 1, x0 + 1])
    return np.where(trong, f, fill)


def align_points(pts, theta, s, m):
    """Biến đổi các điểm mốc theo phép căn chỉnh (xoay + co giãn + tịnh tiến).

    m = (mx, my): trung điểm hai mắt trên ảnh gốc; theta: góc nghiêng đường nối hai mắt;
    s = D_MAT / (khoảng cách hai mắt trên ảnh gốc): hệ số co giãn.
        x' = s * ( cos(theta)*(x - mx) + sin(theta)*(y - my)) + MAT_X
        y' = s * (-sin(theta)*(x - mx) + cos(theta)*(y - my)) + MAT_Y
    Sau biến đổi: hai mắt nằm ngang, cách nhau đúng D_MAT, trung điểm ở (MAT_X, MAT_Y).
    """
    pts = np.asarray(pts, dtype=np.float64)
    c, si = np.cos(theta), np.sin(theta)
    dx = pts[:, 0] - m[0]
    dy = pts[:, 1] - m[1]
    return np.stack([s * (c * dx + si * dy) + MAT_X, s * (-si * dx + c * dy) + MAT_Y], axis=1)


def align_image(img, theta, s, m, fill=255.0):
    """Căn chỉnh ảnh (xoay + co giãn + tịnh tiến) bằng ÁNH XẠ NGƯỢC.

    Với mỗi điểm (x', y') của ảnh mới, tìm vị trí (x, y) trên ảnh gốc bằng phép biến đổi
    ngược của align_points (ma trận xoay nghịch đảo = chuyển vị, chia cho s):
        x = ( cos(theta)*(x' - MAT_X) - sin(theta)*(y' - MAT_Y)) / s + mx
        y = ( sin(theta)*(x' - MAT_X) + cos(theta)*(y' - MAT_Y)) / s + my
    rồi lấy giá trị bằng nội suy song tuyến tính -> ảnh mới không có lỗ trống.
    Điểm ứng với vị trí ngoài ảnh gốc được gán trắng (255).
    """
    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    c, si = np.cos(theta), np.sin(theta)
    src_x = (c * (xs - MAT_X) - si * (ys - MAT_Y)) / s + m[0]
    src_y = (si * (xs - MAT_X) + c * (ys - MAT_Y)) / s + m[1]
    return bilinear_sample(img, src_x, src_y, fill)


# ---------------------------------------------------------------- tô đa giác
def polygon_mask(pts, shape=(SIZE, SIZE)):
    """Mặt nạ (True/False) các điểm ảnh nằm trong đa giác có đỉnh `pts`.

    Quy tắc chẵn - lẻ: từ điểm ảnh (x, y) kẻ tia ngang sang phải; đếm số cạnh đa giác
    mà tia cắt qua. Số lần cắt lẻ -> điểm nằm trong, chẵn -> nằm ngoài.
    Cạnh nối (x1,y1)-(x2,y2) bị tia cắt khi y nằm giữa y1 và y2 và
        x < x1 + (y - y1) * (x2 - x1) / (y2 - y1)   (hoành độ giao điểm)
    """
    pts = np.asarray(pts, dtype=np.float64)
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    trong = np.zeros(shape, dtype=bool)
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if y1 == y2:             # cạnh nằm ngang không cắt tia ngang
            continue
        giua = (y1 > ys) != (y2 > ys)
        x_giao = x1 + (ys - y1) * (x2 - x1) / (y2 - y1)
        trong ^= giua & (xs < x_giao)
    return trong


# ---------------------------------------------------------------- RGB -> HSV
def rgb_to_hsv(rgb):
    """Đổi RGB (0..255) sang HSV. Trả về H (độ, 0..360), S (0..1), V (0..1).

    Đưa R, G, B về [0, 1]; max = lớn nhất, min = nhỏ nhất của 3 kênh, delta = max - min.
        V = max
        S = delta / max                      (max = 0 thì S = 0)
        H = 60 * ((G - B) / delta mod 6)     nếu max là R
        H = 60 * ((B - R) / delta + 2)       nếu max là G
        H = 60 * ((R - G) / delta + 4)       nếu max là B
        (delta = 0, tức màu xám, thì H = 0)
    """
    x = rgb.astype(np.float64) / 255.0
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
    """Mã LBP 8 lân cận (bán kính 1) cho mọi điểm ảnh bên trong.

        LBP(tâm) = tổng_{p=0..7} s(g_p - g_tâm) * 2^p,   s(x) = 1 nếu x >= 0, ngược lại 0
    Thứ tự p đi theo chiều kim đồng hồ, bắt đầu từ góc trên trái:
        p0 p1 p2
        p7  c p3
        p6 p5 p4
    Kết quả là số nguyên 0..255. Viền ngoài 1 pixel (thiếu lân cận) gán 0.
    """
    g = gray.astype(np.float64)
    tam = g[1:-1, 1:-1]
    h, w = g.shape
    lech = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]  # (dy, dx)
    ma = np.zeros(tam.shape, dtype=np.int32)
    for p, (dy, dx) in enumerate(lech):
        lan_can = g[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        ma += (lan_can >= tam).astype(np.int32) << p
    out = np.zeros((h, w), dtype=np.int32)
    out[1:-1, 1:-1] = ma
    return out


# ---------------------------------------------------------------- ngưỡng Otsu
def otsu_threshold(values):
    """Ngưỡng Otsu cho một tập giá trị xám 0..255.

    Thử lần lượt từng ngưỡng t = 0..255, chia tập điểm thành 2 lớp (<= t và > t),
    chọn t làm phương sai GIỮA hai lớp lớn nhất:
        sigma_B^2(t) = w0(t) * w1(t) * (mu0(t) - mu1(t))^2
    w0, w1: tỉ lệ số điểm của mỗi lớp; mu0, mu1: độ xám trung bình của mỗi lớp.
    """
    v = np.clip(np.round(values), 0, 255).astype(int).ravel()
    hist = np.bincount(v, minlength=256).astype(np.float64)
    p = hist / hist.sum()
    muc = np.arange(256)
    w0 = np.cumsum(p)                 # tỉ lệ lớp tối (<= t)
    w1 = 1.0 - w0
    tong0 = np.cumsum(p * muc)        # tổng (mức xám * tỉ lệ) của lớp tối
    ok = (w0 > 0) & (w1 > 0)
    mu0 = np.where(ok, tong0 / np.where(ok, w0, 1), 0)
    mu1 = np.where(ok, (tong0[-1] - tong0) / np.where(ok, w1, 1), 0)
    sigma_b = np.where(ok, w0 * w1 * (mu0 - mu1) ** 2, -1.0)
    return int(np.argmax(sigma_b))


# ---------------------------------------------------------------- khoảng cách
def dist(p, q):
    """Khoảng cách Euclid giữa 2 điểm: d = sqrt((xP - xQ)^2 + (yP - yQ)^2)."""
    return float(np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2))


def eye_centers(pts):
    """Tâm mắt trái ảnh = trung bình điểm 36-41; tâm mắt phải ảnh = trung bình điểm 42-47."""
    pts = np.asarray(pts, dtype=np.float64)
    return pts[36:42].mean(axis=0), pts[42:48].mean(axis=0)
