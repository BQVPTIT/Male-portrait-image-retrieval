"""
chuan_hoa_anh.py - Đưa ảnh bất kỳ về đúng quy cách của bộ dữ liệu: JPG, 256 x 256, 3 kênh màu.

Làm TRƯỚC 6 bước tiền xử lý (preprocess.py). Bước 1 của tiền xử lý chỉ KIỂM TRA quy cách,
còn tệp này ĐƯA ảnh về quy cách đó, cho ảnh thu thập thêm hoặc ảnh truy vấn.

Các bước cho mỗi ảnh:
  1. Đọc ảnh mọi định dạng phổ biến (JPG, PNG, BMP, WEBP, TIFF...).
       - Ảnh xám (1 kênh)        -> nhân thành 3 kênh R = G = B.
       - Ảnh có kênh trong suốt A -> trộn lên nền trắng:  C = A*C_anh + (1 - A)*255,  A trong [0, 1].
  2. Cắt hình vuông ở giữa ảnh, cạnh a = cạnh ngắn của ảnh (mặt nằm giữa ảnh nên không bị cắt).
     Cắt chứ không co giãn hai chiều khác nhau, nên khuôn mặt KHÔNG bị kéo méo.
  3. Thu nhỏ hình vuông a x a về 256 x 256 (tự code bằng numpy):
       - Trung bình khối k x k, k = floor(a / 256):  điểm mới = (1/k^2) * tổng k x k điểm cũ.
       - Phần lẻ còn lại: nội suy song tuyến tính
             f = (1-u)(1-v) f(x,y) + u(1-v) f(x+1,y) + (1-u) v f(x,y+1) + u v f(x+1,y+1).
     Ảnh nhỏ hơn 256 thì chỉ dùng nội suy (phóng to).
  4. Ghi ra tệp .jpg (chất lượng 95), tên giữ nguyên, đổi đuôi thành .jpg.

Chạy:
    python chuan_hoa_anh.py <ảnh hoặc thư mục ảnh> <thư mục ra>
Ví dụ:
    python chuan_hoa_anh.py anh_moi/ data/anh_256/
"""
import os
import sys
import numpy as np
import cv2

import common as C
from preprocess import thu_nho_khoi

DUOI_DOC = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif")


def doc_anh(path):
    """
    - File: chuan_hoa_anh.py. Dùng để: đọc ảnh mọi định dạng thành ma trận RGB 3 kênh.
    - Dựa trên: giải nén bằng OpenCV qua bộ đệm byte (chạy được với đường dẫn tiếng Việt)
      + ảnh 16 bit: chia 257 về 8 bit
      + ảnh xám 1 kênh: nhân thành 3 kênh R = G = B
      + ảnh có kênh trong suốt A: trộn lên nền trắng, C = A*C_ảnh + (1 - A)*255
    - Ứng dụng: nhận cả PNG, BMP, WEBP, TIFF, không chỉ JPG.
    - Cách khác: thư viện Pillow.
    """
    buf = np.fromfile(path, dtype=np.uint8)
    anh = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if anh is None:
        return None
    if anh.dtype != np.uint8:                      # ảnh 16 bit -> 8 bit
        anh = (anh / 257.0).round().astype(np.uint8)
    if anh.ndim == 2:                              # ảnh xám -> 3 kênh
        return np.stack([anh] * 3, axis=2)
    if anh.shape[2] == 4:                          # BGRA: trộn lên nền trắng
        a = anh[:, :, 3:4] / 255.0
        anh = np.round(a * anh[:, :, :3] + (1 - a) * 255.0).astype(np.uint8)
    return anh[:, :, 2::-1].copy()                 # BGR -> RGB


def ve_256(rgb):
    """
    - File: chuan_hoa_anh.py. Dùng để: đưa ảnh bất kỳ về đúng 256 x 256.
    - Dựa trên: cắt vuông rồi thu nhỏ
      + cắt hình vuông ở giữa ảnh, cạnh = cạnh ngắn (mặt nằm giữa ảnh nên không bị cắt)
      + thu nhỏ khối k x k (k = cạnh / 256), phần lẻ dùng nội suy song tuyến tính
      + cắt chứ không co giãn hai chiều khác nhau, nên khuôn mặt không bị méo
    - Ứng dụng: quy cách cỡ ảnh mà bước 1 tiền xử lý kiểm tra.
    - Cách khác: cắt quanh khuôn mặt (preprocess.lam_vuong).
    """
    h, w = rgb.shape[:2]
    a = min(h, w)
    y0, x0 = (h - a) // 2, (w - a) // 2
    vuong = thu_nho_khoi(rgb[y0:y0 + a, x0:x0 + a], max(1, a // C.SIZE))
    n = vuong.shape[0]
    # tọa độ tâm điểm ảnh mới trên ảnh vuông n x n
    toa_do = np.minimum((np.arange(C.SIZE) + 0.5) * n / C.SIZE - 0.5, n - 1.001).clip(0)
    ys, xs = np.meshgrid(toa_do, toa_do, indexing="ij")
    return np.clip(np.round(C.bilinear_sample(vuong, xs, ys)), 0, 255).astype(np.uint8)


def chuan_hoa_mot(path_vao, thu_muc_ra):
    """
    - File: chuan_hoa_anh.py. Dùng để: chuẩn hóa một tệp ảnh và ghi ra JPG.
    - Dựa trên: doc_anh -> ve_256 -> ghi JPG chất lượng 95
      + tên giữ nguyên, đổi đuôi thành .jpg
      + OpenCV ghi theo thứ tự kênh B, G, R nên đảo kênh trước khi ghi
      + trả về (True, đường dẫn ra) hoặc (False, lý do)
    - Ứng dụng: đưa ảnh thu thập thêm hoặc ảnh truy vấn về đúng quy cách JPG 256 x 256
      trước 6 bước tiền xử lý.
    - Cách khác: chuyển đổi thủ công bằng phần mềm chỉnh ảnh.
    """
    rgb = doc_anh(path_vao)
    if rgb is None:
        return False, "khong doc duoc anh"
    ra = ve_256(rgb)
    ten = os.path.splitext(os.path.basename(path_vao))[0] + ".jpg"
    path_ra = os.path.join(thu_muc_ra, ten)
    ok, buf = cv2.imencode(".jpg", ra[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return False, "khong ghi duoc JPG"
    buf.tofile(path_ra)
    return True, path_ra


def run(vao, thu_muc_ra):
    """
    - File: chuan_hoa_anh.py. Dùng để: chuẩn hóa một ảnh hoặc cả một thư mục ảnh.
    - Dựa trên: duyệt mọi tệp có đuôi ảnh phổ biến (DUOI_DOC)
      + gọi chuan_hoa_mot cho từng tệp
      + in OK hoặc LOI kèm lý do, cuối cùng in tổng số ảnh đạt
    - Ứng dụng: chạy bằng lệnh python chuan_hoa_anh.py <vào> <thư mục ra>; tệp hỏng chỉ báo lỗi,
      không làm dừng cả lượt.
    - Cách khác: chuẩn hóa từng ảnh bằng tay.
    """
    os.makedirs(thu_muc_ra, exist_ok=True)
    if os.path.isdir(vao):
        ds = [os.path.join(vao, f) for f in sorted(os.listdir(vao)) if f.lower().endswith(DUOI_DOC)]
    else:
        ds = [vao]
    dat = 0
    for p in ds:
        ok, kq = chuan_hoa_mot(p, thu_muc_ra)
        dat += ok
        print(("OK   " if ok else "LOI  ") + os.path.basename(p) + ("" if ok else "  -> " + kq))
    print(f"Xong: {dat}/{len(ds)} anh -> {thu_muc_ra}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
