"""
app.py - Giao diện demo (Tkinter): chọn một ảnh chân dung -> 5 ảnh giống nhất trong CSDL,
kèm các kết quả trung gian để giải thích từng bước.

    Upload ảnh / Chọn ảnh trong CSDL : chọn ảnh truy vấn. Ảnh chưa đúng 256 x 256 được làm vuông quanh
                                       khuôn mặt (preprocess.lam_vuong). Ảnh lấy từ CSDL thì không cho
                                       chính nó vào kết quả.
    Đường đo hình học                : vẽ 6 đoạn dùng cho X0 - X4: d(P0,P16), dY(P27,P8), d(P4,P12),
                                       d(mắt trái, mắt phải), d(P31,P35), dY(P27,P33).
    Vùng kết cấu, mặt nạ da          : tô vùng cằm - miệng (X5), vùng lông mày (X6), vùng da 2 má + trán
                                       (X7 - X38), đúng các mặt nạ mà features.py dùng.
    Trọng số 3 bộ                    : mặc định 1/3 mỗi bộ (search.TRONG_SO, Báo cáo 1). Kéo thanh trượt rồi
                                       bấm "Tìm lại" để THỬ trọng số khác (chuẩn hóa cho tổng = 1); trọng số
                                       thử không ghi vào CSDL.

Màn hình kết quả, từ trên xuống:
    1. Tiền xử lý ảnh truy vấn: ảnh sau từng bước B1 -> B6 và số liệu kiểm tra của từng bước.
    2. Bảng so sánh: cột ảnh truy vấn + 5 cột kết quả. Mỗi hàng là một đặc trưng: giá trị gốc x, giá trị
       chuẩn hóa x', chênh lệch |q' - x'| (ô càng xanh càng gần truy vấn); lược đồ màu 32 ô; D từng bộ, D, S.
       Bấm vào ảnh kết quả -> cửa sổ thế số từng bước: chuẩn hóa -> D từng bộ -> D -> S -> xếp hạng.
    3. Biểu đồ đóng góp của 3 bộ vào D, và các lệnh SQL đã chạy để lấy dữ liệu.

Chạy:  python app.py
"""
import os
import time
import colorsys
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

import common as C
import preprocess
import features as F
import search as S
import database as DB

TEN_BO = {"hinh_hoc": "Hình học", "ket_cau": "Kết cấu", "mau_sac": "Màu sắc"}
KY_HIEU_BO = {"hinh_hoc": "D_hh", "ket_cau": "D_kc", "mau_sac": "D_ms"}
MAU_BO = {"hinh_hoc": "#4C78A8", "ket_cau": "#E0802A", "mau_sac": "#4E9A47"}       # màu của 3 bộ (biểu đồ)
NEN_BO = {"hinh_hoc": "#e3ecf6", "ket_cau": "#fbeadb", "mau_sac": "#e4f1e2"}
LY_DO = {"khong phai JPG": "Tệp không phải ảnh JPG (hệ thống chỉ dùng định dạng JPG).",
         "khong doc duoc": "Không đọc được tệp ảnh.",
         "sai kich thuoc/so kenh": "Ảnh không đúng 256 × 256 × 3 kênh màu.",
         "anh mo": "Ảnh mờ: độ nét {do_net} nhỏ hơn ngưỡng 20.",
         "so khuon mat khac 1": "Không tìm thấy đúng một khuôn mặt trong ảnh.",
         "diem moc khong on dinh": "Điểm mốc không ổn định: độ lệch {lech_moc} từ 0,08 trở lên."}
CONG_THUC = ["d(P0,P16) / dY(P27,P8)", "d(P0,P16) / d(P4,P12)", "d(mắt T, mắt P) / d(P0,P16)",
             "d(P31,P35) / d(mắt T, mắt P)", "dY(P27,P33) / dY(P27,P8)",
             "phương sai mã LBP vùng cằm - miệng", "tỉ lệ điểm sợi lông (Otsu, LBP ≠ 0, 255)"]

# 6 đoạn đo dùng cho X0 - X4: (tên, màu)
DOAN_DO = [("d(P0,P16)", "#d62728"), ("dY(P27,P8)", "#9467bd"), ("d(P4,P12)", "#8c564b"),
           ("d(mắt T, mắt P)", "#1f77b4"), ("d(P31,P35)", "#e377c2"), ("dY(P27,P33)", "#17a2b8")]
# 3 vùng dùng cho X5, X6, X7 - X38: (tên, màu)
VUNG = [("X5 vùng cằm - miệng", "#1e5aff"), ("X6 vùng lông mày", "#ff8c00"), ("X7–X38 da 2 má + trán", "#00aa00")]

CO_ANH = 150        # cỡ ảnh trong bảng so sánh
CO_BUOC = 140       # cỡ ảnh các bước tiền xử lý
CAT_MAT = (40, 28, 216, 204)   # khung 176 x 176 quanh mặt đã căn chỉnh (2 mắt ở hàng 100), chỉ để phóng to khi xem
FONT = ("Segoe UI", 9)
FONT_DAM = ("Segoe UI", 9, "bold")


def so(x, n=3):
    """Số kiểu Việt Nam: dấu phẩy thập phân."""
    return f"{x:.{n}f}".replace(".", ",")


def so_x(k, x):
    """X5 (phương sai LBP, cỡ hàng nghìn) lấy 1 chữ số thập phân, còn lại 3 chữ số."""
    return so(x, 1) if k == 5 else so(x, 3)


def o_mau(k):
    """Mô tả ô màu thứ k của lược đồ H x S (k = ô H * 4 + ô S)."""
    i, j = divmod(k, F.SO_O_S)
    h = f"H {i * 5}–{i * 5 + 5}°" if 0 < i < F.SO_O_H - 1 else ("H 0–5° (cả ≥ 330°)" if i == 0 else "H 35–330°")
    s = [f"{x:g}".replace(".", ",") for x in (j * 0.25, j * 0.25 + 0.25)]
    return f"{h}, S {s[0]}–{s[1]}"


def mau_o(k):
    """Màu thật ở giữa ô k (H giữa ô, S giữa ô, V = 0,85) để tô cột lược đồ."""
    i, j = divmod(k, F.SO_O_S)
    h = 37.5 if i == F.SO_O_H - 1 else i * 5 + 2.5
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, j * 0.25 + 0.125, 0.85)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def rgb_hex(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def phong(co):
    try:
        return ImageFont.truetype("arial.ttf", co)
    except OSError:
        return ImageFont.load_default()


def ve_len_anh(anh, pts, duong_do=False, vung=False):
    """Vẽ lên ảnh đã căn chỉnh (chỉ để xem):
        vung     : tô 3 vùng lấy đặc trưng kết cấu / màu (dùng đúng hàm của features.py);
        duong_do : 6 đoạn đo của 5 đặc trưng hình học và các điểm mốc liên quan.
    """
    if pts is None or not (duong_do or vung):
        return Image.fromarray(anh)
    img = anh.astype(np.float64)
    if vung:
        co_du_lieu = ~(anh == 255).all(axis=2)
        long_may = F.brow_regions(pts)
        for m, (_, mau) in zip([F.chin_region(pts), long_may[0] | long_may[1], F.skin_mask(pts)], VUNG):
            m = m & co_du_lieu
            img[m] = 0.5 * img[m] + 0.5 * np.array(rgb_hex(mau))
    img = Image.fromarray(np.clip(np.round(img), 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(img)
    if vung:
        for x, y in pts:
            dr.ellipse([x - 1.2, y - 1.2, x + 1.2, y + 1.2], fill=(200, 0, 0))
    if duong_do:
        trai, phai = C.eye_centers(pts)
        x27 = pts[27, 0]
        doan = [(pts[0], pts[16]), ((x27, pts[27, 1]), (x27, pts[8, 1])), (pts[4], pts[12]), (trai, phai),
                (pts[31], pts[35]), ((x27 + 9, pts[27, 1]), (x27 + 9, pts[33, 1]))]
        for (a, b), (_, mau) in zip(doan, DOAN_DO):
            dr.line([tuple(a), tuple(b)], fill=rgb_hex(mau), width=3)
        f = phong(11)
        for k in (0, 4, 8, 12, 16, 27, 31, 33, 35):
            x, y = pts[k]
            dr.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill=(255, 255, 0), outline=(0, 0, 0))
            dr.text((x + 3, y - 12), str(k), fill=(0, 0, 0), font=f)
        for p in (trai, phai):
            dr.ellipse([p[0] - 2.5, p[1] - 2.5, p[0] + 2.5, p[1] + 2.5], fill=(255, 255, 0), outline=(0, 0, 0))
    return img


class App:
    def __init__(self, root):
        self.root = root
        root.title("Tìm kiếm ảnh chân dung đàn ông - Nhóm 19")
        root.geometry(f"{min(1460, root.winfo_screenwidth() - 40)}x{min(960, root.winfo_screenheight() - 80)}+10+10")
        self.ids, _ = S.load_features()
        self.vi_tri = {m: i for i, m in enumerate(self.ids)}
        self.lm = F.read_landmarks()
        self.tt = DB.doc_thuoc_tinh()             # mô tả đặc trưng: ký hiệu, tên, vai trò (Bảng 2.1)
        self.ts_db = S.load_weights()             # trọng số 3 bộ (1/3, Báo cáo 1)
        self.tuong_dong = [k for k in range(7) if self.tt[k]["vai_tro"] == "tuong_dong"]    # X0, X1
        self.khac_biet = [k for k in range(7) if self.tt[k]["vai_tro"] == "khac_biet"]      # X2 - X6
        preprocess.detect_landmarks(np.zeros((64, 64), dtype=np.uint8))     # nạp sẵn mô hình dlib
        self.ket_qua = None
        self.anh_tk = []

        # hàng 1: chọn ảnh, tùy chọn vẽ
        thanh = tk.Frame(root, pady=4, padx=8)
        thanh.pack(fill="x")
        tk.Button(thanh, text="Upload ảnh", width=14, command=self.upload, font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Button(thanh, text="Chọn ảnh trong CSDL", width=18, command=self.chon_trong_csdl).pack(side="left", padx=6)
        self.ve_do = tk.BooleanVar(value=False)
        self.ve_vung = tk.BooleanVar(value=False)
        tk.Checkbutton(thanh, text="Đường đo hình học (X0–X4)", variable=self.ve_do,
                       command=lambda: self.hien_thi(giu_vi_tri=True)).pack(side="left", padx=(12, 4))
        tk.Checkbutton(thanh, text="Vùng kết cấu, mặt nạ da (X5–X38)", variable=self.ve_vung,
                       command=lambda: self.hien_thi(giu_vi_tri=True)).pack(side="left", padx=4)
        self.trang_thai = tk.Label(thanh, anchor="w", fg="#444",
                                   text=f"CSDL: {len(self.ids)} ảnh, {len(F.FEATURE_NAMES)} đặc trưng ({os.path.basename(C.DB_PATH)})")
        self.trang_thai.pack(side="left", padx=12)

        # hàng 2: trọng số 3 bộ
        thanh2 = tk.Frame(root, padx=8)
        thanh2.pack(fill="x")
        tk.Label(thanh2, text="Trọng số khi gộp  D = w_hh·D_hh + w_kc·D_kc + w_ms·D_ms :", font=FONT_DAM).pack(side="left")
        self.w = {}
        for b in F.BO:
            self.w[b] = tk.DoubleVar(value=self.ts_db[b])
            tk.Scale(thanh2, from_=0, to=1, resolution=0.01, orient="horizontal", length=130, variable=self.w[b],
                     label=f"w {TEN_BO[b].lower()}", font=FONT, fg=MAU_BO[b]).pack(side="left", padx=4)
        tk.Button(thanh2, text="Tìm lại", width=9, command=self.tim_lai).pack(side="left", padx=4)
        tk.Button(thanh2, text="Về 1/3 (Báo cáo 1)", command=self.ve_mac_dinh).pack(side="left", padx=4)
        self.nhan_ts = tk.Label(thanh2, text="", font=FONT, justify="left")
        self.nhan_ts.pack(side="left", padx=8)

        khung = tk.Frame(root)
        khung.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(khung, highlightthickness=0)
        cuon = tk.Scrollbar(khung, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=cuon.set)
        cuon.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.noi_dung = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.noi_dung, anchor="nw")
        self.noi_dung.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        root.bind_all("<MouseWheel>", self.cuon_chuot)
        tk.Label(self.noi_dung, text="Bấm \"Upload ảnh\" để chọn một ảnh chân dung (JPG).",
                 font=("Segoe UI", 12), pady=40, padx=20).pack()

    def cuon_chuot(self, e):
        """Lăn chuột chỉ cuộn màn hình chính khi con trỏ ở cửa sổ chính (cửa sổ chi tiết tự cuộn)."""
        try:
            if str(e.widget.winfo_toplevel()) != str(self.root):
                return
        except Exception:
            return
        self.canvas.yview_scroll(int(-e.delta / 120), "units")

    # ------------------------------------------------------------ chọn ảnh
    def upload(self):
        path = filedialog.askopenfilename(title="Chọn ảnh chân dung",
                                          filetypes=[("Ảnh JPG", "*.jpg *.jpeg *.JPG *.JPEG"), ("Tất cả tệp", "*.*")])
        if path:
            self.tim(path)

    def chon_trong_csdl(self):
        path = filedialog.askopenfilename(title="Chọn ảnh trong CSDL", initialdir=C.RAW_DIR,
                                          filetypes=[("Ảnh JPG", "*.jpg")])
        if path:
            ma = os.path.splitext(os.path.basename(path))[0]
            self.tim(path, bo_qua=ma if ma in self.vi_tri else None)

    # ------------------------------------------------------------ tìm kiếm
    def tim(self, path, bo_qua=None):
        self.trang_thai.config(text="Đang tiền xử lý và tìm kiếm...")
        self.root.config(cursor="watch")
        self.root.update()
        t0 = time.time()
        try:
            ghi_chu = ""
            rgb = preprocess.read_rgb(path) if preprocess.la_jpg(path) else None
            if rgb is not None and rgb.ndim == 3 and rgb.shape[:2] != (C.SIZE, C.SIZE):
                tam = os.path.join(tempfile.gettempdir(), "nhom19_truy_van.jpg")
                ok, ly_do = preprocess.lam_vuong(path, tam)
                if not ok:
                    self.bao_loi(ly_do, {})
                    return
                ghi_chu = f"Ảnh gốc {rgb.shape[1]} × {rgb.shape[0]}: đã cắt vuông quanh khuôn mặt, đưa về 256 × 256."
                path = tam
            kq = preprocess.process_one(path, giu_trung_gian=True)
            if not kq["dat"]:
                self.bao_loi(kq["ly_do"], kq)
                return
            v = F.extract(kq["rgb"], kq["gray"], kq["pts"])
            self.ket_qua = dict(path=path, kq=kq, v=v, bo_qua=bo_qua, ghi_chu=ghi_chu)
            self.xep_hang()
            self.ket_qua["thoi_gian"] = time.time() - t0
            self.hien_thi()
        finally:
            self.root.config(cursor="")

    def trong_so_thu(self):
        """Trọng số trên thanh trượt. Ba giá trị bằng nhau -> None (dùng trọng số trong CSDL);
        ngược lại chuẩn hóa w_b / (w_hh + w_kc + w_ms) để tổng bằng 1."""
        w = {b: self.w[b].get() for b in F.BO}
        if max(w.values()) - min(w.values()) < 1e-9:
            return None
        tong = sum(w.values())
        return {b: w[b] / tong for b in F.BO}

    def xep_hang(self):
        """So vector truy vấn với CSDL (search.xep_hang) rồi đọc thông tin 5 ảnh kết quả từ bảng anh."""
        r = self.ket_qua
        ts = self.trong_so_thu()
        xh = S.xep_hang(r["v"], bo_qua=r["bo_qua"], trong_so=ts)
        con = DB.ket_noi()
        r["anh_top"] = [DB.doc_anh(con, xh["ids"][i]) for i in xh["top"]]
        con.close()
        r["xh"], r["ts_thu"] = xh, ts
        if ts is None:
            self.nhan_ts.config(text="Trọng số trong CSDL:\n1/3 mỗi bộ (đúng Báo cáo 1)", fg="#2a6a2a")
        else:
            self.nhan_ts.config(text="Đang THỬ trọng số " + " / ".join(so(ts[b], 2) for b in F.BO)
                                     + "\n(chuẩn hóa tổng = 1), không ghi vào CSDL", fg="#b00020")

    def tim_lai(self):
        if self.ket_qua is None:
            return
        if sum(self.w[b].get() for b in F.BO) == 0:
            messagebox.showwarning("Trọng số", "Ba trọng số không thể cùng bằng 0.")
            return
        self.xep_hang()
        self.hien_thi(giu_vi_tri=True)

    def ve_mac_dinh(self):
        for b in F.BO:
            self.w[b].set(self.ts_db[b])
        self.tim_lai()

    def bao_loi(self, ly_do, kq):
        thong_bao = LY_DO.get(ly_do, ly_do).format(do_net=so(kq.get("do_net") or 0, 1),
                                                   lech_moc=so(kq.get("lech_moc") or 0, 3))
        self.trang_thai.config(text="Ảnh không đạt tiền xử lý: " + thong_bao)
        messagebox.showwarning("Ảnh không đạt tiền xử lý", thong_bao + "\nẢnh không được đưa vào tìm kiếm.")

    # ------------------------------------------------------------ tiện ích hiển thị
    def anh(self, img, co=CO_ANH, giu=None):
        """PIL Image / mảng numpy / đường dẫn -> ảnh Tk cỡ co x co (giữ tham chiếu để Tk không xóa)."""
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        anh_tk = ImageTk.PhotoImage(img.resize((co, co)))
        (self.anh_tk if giu is None else giu).append(anh_tk)
        return anh_tk

    def anh_ve(self, anh, pts, co=CO_ANH, giu=None):
        """Ảnh đã căn chỉnh + hình vẽ theo 2 ô tùy chọn, cắt khung CAT_MAT cho mặt to hơn."""
        return self.anh(ve_len_anh(anh, pts, self.ve_do.get(), self.ve_vung.get()).crop(CAT_MAT), co, giu)

    def anh_csdl(self, ma):
        """Ảnh đã căn chỉnh, lọc nền của một ảnh trong CSDL (data/prep/aligned) + 68 điểm mốc."""
        return np.array(Image.open(os.path.join(C.ALIGNED_DIR, ma + ".jpg")).convert("RGB")), self.lm.get(ma)

    def ten(self, k):
        """Ký hiệu + tên đặc trưng thứ k (ví dụ "X0 tỉ lệ khung mặt")."""
        return f"{self.tt[k]['ky_hieu']} {self.tt[k]['ten_dac_trung'][0].lower()}{self.tt[k]['ten_dac_trung'][1:]}"

    def tieu_de(self, cha, chu, nen):
        tk.Label(cha, text=chu, bg=nen, fg="#1a3d7c", font=("Segoe UI", 11, "bold"), anchor="w").pack(fill="x", padx=6, pady=(4, 2))

    def ts_chu(self, b):
        w = self.ket_qua["xh"]["trong_so"][b]
        return "1/3" if abs(w - 1 / 3) < 1e-9 else so(w, 2)

    def hang_cua(self, i):
        """Hạng của ảnh i (1 = giống nhất) trong các ảnh được xét."""
        xh = self.ket_qua["xh"]
        return int((xh["S"][xh["con_lai"]] > xh["S"][i]).sum()) + 1

    # ------------------------------------------------------------ màn hình kết quả
    def hien_thi(self, giu_vi_tri=False):
        if self.ket_qua is None:
            return
        vi_tri = self.canvas.yview()[0]
        for w in self.noi_dung.winfo_children():
            w.destroy()
        self.anh_tk = []
        self.khung_tien_xu_ly()
        self.khung_so_sanh()
        self.khung_bieu_do_sql()
        r = self.ket_qua
        self.trang_thai.config(text=f"Xong: top-5 trong {int(r['xh']['con_lai'].sum())} ảnh của CSDL "
                                    f"({so(r['thoi_gian'], 2)} s).")
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(vi_tri if giu_vi_tri else 0)

    def khung_tien_xu_ly(self):
        """Phần 1: ảnh truy vấn sau từng bước B1 -> B6, kèm số liệu kiểm tra."""
        r = self.ket_qua
        kq = r["kq"]
        tg = kq["trung_gian"]
        nen = "#eef3fb"
        f = tk.Frame(self.noi_dung, bg=nen, bd=1, relief="solid")
        f.pack(fill="x", padx=8, pady=(6, 4))
        self.tieu_de(f, "BƯỚC 1. TIỀN XỬ LÝ ẢNH TRUY VẤN (B1 → B6)", nen)

        L = np.abs(C.laplacian(tg["gray_tho"]))
        b3 = np.clip(L / max(np.percentile(L, 99), 1e-9) * 255, 0, 255).astype(np.uint8)
        b4 = Image.fromarray(tg["rgb_tho"])
        dr = ImageDraw.Draw(b4)
        dr.rectangle(tg["khung"], outline=(0, 200, 0), width=2)
        for x, y in tg["pts_tho"]:
            dr.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(255, 0, 0))
        b5 = Image.fromarray(tg["rgb_can_chinh"])
        dr = ImageDraw.Draw(b5)
        dr.line([(0, C.MAT_Y), (C.SIZE, C.MAT_Y)], fill=(0, 120, 255), width=1)
        dr.line([(C.MAT_X, 0), (C.MAT_X, C.SIZE)], fill=(0, 120, 255), width=1)
        for p in C.eye_centers(kq["pts"]):
            dr.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], outline=(255, 0, 0), width=2)

        buoc = [(tg["rgb_tho"], "B1 Đọc, kiểm tra", "JPG, 256 × 256 × 3: đạt"),
                (np.round(tg["gray_tho"]).astype(np.uint8), "B2 Ảnh xám", "Y = 0,299R + 0,587G + 0,114B"),
                (b3, "B3 Độ nét (lọc Laplace |L|)", f"độ nét = var(L) = {so(kq['do_net'], 1)} ≥ 20: đạt"),
                (b4, "B4 Khuôn mặt + 68 điểm mốc", f"1 khuôn mặt; độ lệch mốc {so(kq['lech_moc'])} < 0,08"),
                (b5, "B5 Căn chỉnh theo 2 mắt", f"góc 2 mắt θ = {so(kq['goc_nghieng'], 1)}° → xoay về 0°\n"
                                                f"co giãn s = 64 / d(mắt) = {so(kq['he_so'])}"),
                (kq["rgb"], "B6 Lọc nền (mặt nạ điểm mốc)", f"vùng mặt chiếm {so(100 * kq['ti_le_mat'], 0)}% ảnh")]
        hang = tk.Frame(f, bg=nen)
        hang.pack(anchor="w", padx=6, pady=(0, 6))
        for j, (img, t1, t2) in enumerate(buoc):
            if j:
                tk.Label(hang, text="→", bg=nen, font=("Segoe UI", 16)).grid(row=0, column=2 * j - 1)
            o = tk.Frame(hang, bg=nen)
            o.grid(row=0, column=2 * j, sticky="n")
            tk.Label(o, image=self.anh(img, CO_BUOC), bg=nen).pack()
            tk.Label(o, text=t1, bg=nen, font=FONT_DAM).pack()
            tk.Label(o, text=t2, bg=nen, font=FONT, fg="#333", justify="center").pack()
        if r["ghi_chu"]:
            tk.Label(f, text=r["ghi_chu"], bg=nen, font=FONT, fg="#333").pack(anchor="w", padx=8, pady=(0, 4))
        if r["bo_qua"]:
            tk.Label(f, text=f"Ảnh truy vấn có trong CSDL ({r['bo_qua']}) nên không đưa chính nó vào kết quả.",
                     bg=nen, font=FONT, fg="#333").pack(anchor="w", padx=8, pady=(0, 4))

    def o(self, g, hang, cot, chu="", nen="white", font=FONT, **kw):
        lb = tk.Label(g, text=chu, bg=nen, font=font, justify=kw.pop("justify", "center"), padx=4, pady=2, **kw)
        lb.grid(row=hang, column=cot, sticky="nsew", padx=1, pady=1)
        return lb

    def dai_bo(self, g, hang, b):
        """Hàng tiêu đề của một bộ thuộc tính, trải hết bề ngang bảng."""
        n = F.BO[b].stop - F.BO[b].start
        lb = tk.Label(g, text=f"BỘ {TEN_BO[b].upper()} ({n} chiều)", bg=NEN_BO[b], fg=MAU_BO[b],
                      font=("Segoe UI", 9, "bold"), anchor="w", padx=6)
        lb.grid(row=hang, column=0, columnspan=7, sticky="nsew", padx=1, pady=1)

    def khung_so_sanh(self):
        """Phần 2: bảng truy vấn - top-5. Cột 0 là tên hàng, cột 1 truy vấn, cột 2..6 kết quả #1..#5."""
        r = self.ket_qua
        xh, v, kq = r["xh"], r["v"], r["kq"]
        top = xh["top"]
        n_xet = int(xh["con_lai"].sum())
        f = tk.Frame(self.noi_dung, bg="white", bd=1, relief="solid")
        f.pack(fill="x", padx=8, pady=4)
        self.tieu_de(f, "BƯỚC 2–3. ĐẶC TRƯNG VÀ ĐỘ TƯƠNG ĐỒNG   (bấm vào ảnh kết quả để xem cách tính từng bước)", "white")
        g = tk.Frame(f, bg="#c9d2de")
        g.pack(anchor="w", padx=6, pady=(0, 6))

        # đầu bảng và ảnh
        self.o(g, 0, 0, "", "#f4f6f9")
        self.o(g, 0, 1, "ẢNH TRUY VẤN", "#dfe8f6", FONT_DAM)
        for c, i in enumerate(top, start=2):
            self.o(g, 0, c, f"#{c - 1}   {xh['ids'][i]}", "#f4f6f9", FONT_DAM)
        self.o(g, 1, 0, "Ảnh gốc\n(đường dẫn lấy từ bảng anh)", "#f4f6f9", justify="left", anchor="w")
        self.o(g, 1, 1, nen="#dfe8f6", image=self.anh(r["path"]))
        for c, a in enumerate(r["anh_top"], start=2):
            self.o(g, 1, c, image=self.anh(DB.duong_dan_day_du(a["duong_dan"])))

        o_chu = tk.Frame(g, bg="#f4f6f9")
        o_chu.grid(row=2, column=0, sticky="nsew", padx=1, pady=1)
        tk.Label(o_chu, text="Đã căn chỉnh, lọc nền", bg="#f4f6f9", font=FONT).pack(anchor="w", padx=4)
        if self.ve_do.get():
            for ten, mau in DOAN_DO:
                tk.Label(o_chu, text="━━ " + ten, bg="#f4f6f9", fg=mau, font=FONT_DAM).pack(anchor="w", padx=4)
        if self.ve_vung.get():
            for ten, mau in VUNG:
                tk.Label(o_chu, text="■ " + ten, bg="#f4f6f9", fg=mau, font=FONT_DAM).pack(anchor="w", padx=4)
        self.o(g, 2, 1, nen="#dfe8f6", image=self.anh_ve(kq["rgb"], kq["pts"]))
        for c, i in enumerate(top, start=2):
            lb = self.o(g, 2, c, image=self.anh_ve(*self.anh_csdl(xh["ids"][i])), cursor="hand2")
            lb.bind("<Button-1>", lambda e, i=i: self.chi_tiet(i))

        hang = 3
        # 7 đặc trưng hình học + kết cấu
        for k in range(7):
            if k in (0, 5):
                self.dai_bo(g, hang, "hinh_hoc" if k == 0 else "ket_cau")
                hang += 1
            t = self.tt[k]
            vai_tro = "tìm tương đồng" if t["vai_tro"] == "tuong_dong" else "tìm khác biệt"
            mau_chu = VUNG[k - 5][1] if k >= 5 else "#000"
            self.o(g, hang, 0, f"{self.ten(k)}\n= {CONG_THUC[k]}\nmin {so_x(k, t['gia_tri_min'])} – max "
                               f"{so_x(k, t['gia_tri_max'])}  ({vai_tro})", "#f4f6f9", justify="left",
                   anchor="w", fg=mau_chu)
            self.o(g, hang, 1, f"x = {so_x(k, v[k])}\nx' = {so(xh['q'][k])}", "#dfe8f6", FONT_DAM)
            for c, i in enumerate(top, start=2):
                lech = abs(xh["Xn"][i, k] - xh["q"][k])
                t_ = min(lech / 0.25, 1.0)                  # lệch 0 -> xanh đậm nhất, lệch >= 0,25 -> trắng
                nen = "#%02x%02x%02x" % tuple(int(a + (255 - a) * t_) for a in (150, 215, 150))
                self.o(g, hang, c, f"x = {so_x(k, xh['X'][i, k])}\nx' = {so(xh['Xn'][i, k])}\n|q' − x'| = {so(lech)}", nen)
            hang += 1

        # lược đồ màu
        self.dai_bo(g, hang, "mau_sac")
        hang += 1
        self.o(g, hang, 0, "X7–X38 lược đồ màu da H × S\n8 ô H (0–40°, 5°/ô) × 4 ô S\n"
                           "h_k = n_k / N (tổng 32 ô = 1)\nvạch đen: lược đồ ảnh truy vấn",
               "#f4f6f9", justify="left", anchor="w", fg=VUNG[2][1])
        self.luoc_do(g, hang, 1, v[7:], None, "#dfe8f6")
        for c, i in enumerate(top, start=2):
            self.luoc_do(g, hang, c, xh["X"][i, 7:], v[7:], "white")
        hang += 1

        # khoảng cách từng bộ, D, S, hạng
        for b in F.BO:
            n = F.BO[b].stop - F.BO[b].start
            self.o(g, hang, 0, f"{KY_HIEU_BO[b]} = √Σ(q' − x')² / √{n}", NEN_BO[b], FONT_DAM, justify="left",
                   anchor="w", fg=MAU_BO[b])
            self.o(g, hang, 1, "—", "#dfe8f6")
            for c, i in enumerate(top, start=2):
                d = xh["d_bo"][b][i]
                gan_hon = 100.0 * (xh["d_bo"][b][xh["con_lai"]] > d).mean()
                self.o(g, hang, c, f"{so(d, 4)}\n(nhỏ hơn {so(gan_hon, 1)}% ảnh)", NEN_BO[b])
            hang += 1
        self.o(g, hang, 0, "D = " + " + ".join(f"{self.ts_chu(b)}·{KY_HIEU_BO[b]}" for b in F.BO),
               "#f4f6f9", FONT_DAM, justify="left", anchor="w")
        self.o(g, hang, 1, "—", "#dfe8f6")
        for c, i in enumerate(top, start=2):
            self.o(g, hang, c, so(xh["D"][i], 4), font=FONT_DAM)
        hang += 1
        self.o(g, hang, 0, "S = 1 / (1 + D)", "#fff4c2", ("Segoe UI", 11, "bold"), justify="left", anchor="w")
        self.o(g, hang, 1, "—", "#dfe8f6")
        for c, i in enumerate(top, start=2):
            self.o(g, hang, c, so(xh["S"][i], 4), "#fff4c2", ("Segoe UI", 12, "bold"))
        hang += 1
        self.o(g, hang, 0, "Hạng (S giảm dần)", "#f4f6f9", justify="left", anchor="w")
        self.o(g, hang, 1, "—", "#dfe8f6")
        for c, i in enumerate(top, start=2):
            self.o(g, hang, c, f"{self.hang_cua(i)} / {n_xet}")

    def luoc_do(self, g, hang, cot, h, h_q, nen):
        """Vẽ lược đồ 32 ô vào một ô của bảng; cột tô đúng màu của ô (H, S); vạch đen = truy vấn."""
        o = tk.Frame(g, bg=nen)
        o.grid(row=hang, column=cot, sticky="nsew", padx=1, pady=1)
        W, H, day = CO_ANH, 86, 72
        cv = tk.Canvas(o, width=W, height=H, bg="white", highlightthickness=0)
        cv.pack(padx=2, pady=(2, 0))
        tran = max(h.max(), 0 if h_q is None else h_q.max(), 1e-9)
        rong = (W - 8 - 7 * 2) / 32.0
        for k in range(32):
            x = 4 + k * rong + (k // 4) * 2
            cv.create_rectangle(x, day - h[k] / tran * (day - 4), x + rong - 0.6, day, fill=mau_o(k), outline="")
            if h_q is not None and h_q[k] > 0:
                y = day - h_q[k] / tran * (day - 4)
                cv.create_line(x - 0.5, y, x + rong, y, fill="black", width=1.5)
        cv.create_line(2, day, W - 2, day, fill="#888")
        for i in range(0, 8, 2):
            cv.create_text(4 + i * 4 * rong + i * 2, day + 7, text=f"{i * 5}°", anchor="w", font=("Segoe UI", 7), fill="#555")
        k = int(np.argmax(h))
        tk.Label(o, text=f"nhiều nhất ({so(100 * h[k], 0)}% điểm da):\n{o_mau(k)}", bg=nen, font=("Segoe UI", 8)).pack()

    def khung_bieu_do_sql(self):
        """Phần 3: biểu đồ đóng góp của 3 bộ vào D (trái) và các lệnh SQL đã chạy (phải)."""
        r = self.ket_qua
        xh = r["xh"]
        top = xh["top"]
        f = tk.Frame(self.noi_dung, bg="white", bd=1, relief="solid")
        f.pack(fill="x", padx=8, pady=(4, 10))
        trai = tk.Frame(f, bg="white")
        trai.pack(side="left", anchor="n", padx=6, pady=4)
        self.tieu_de(trai, "ĐÓNG GÓP CỦA 3 BỘ VÀO KHOẢNG CÁCH D (thanh càng ngắn càng giống)", "white")
        W, H = 700, 64 + 42 * len(top)
        cv = tk.Canvas(trai, width=W, height=H, bg="white", highlightthickness=0)
        cv.pack()
        x = 10
        for b in F.BO:
            cv.create_rectangle(x, 8, x + 14, 22, fill=MAU_BO[b], outline="")
            chu = f"{self.ts_chu(b)} × {KY_HIEU_BO[b]} ({TEN_BO[b].lower()})"
            cv.create_text(x + 20, 15, text=chu, anchor="w", font=FONT)
            x += 30 + 7 * len(chu)
        x0, x_cuoi = 120, W - 190
        d_max = max(xh["D"][i] for i in top)
        ti_le = (x_cuoi - x0) / max(d_max, 1e-9)
        for hang, i in enumerate(top):
            y = 40 + hang * 42
            cv.create_text(10, y + 14, text=f"#{hang + 1}  {xh['ids'][i]}", anchor="w", font=FONT_DAM)
            x = x0
            for b in F.BO:
                phan = xh["trong_so"][b] * xh["d_bo"][b][i]
                dx = phan * ti_le
                cv.create_rectangle(x, y, x + dx, y + 28, fill=MAU_BO[b], outline="white")
                if dx > 44:
                    cv.create_text(x + dx / 2, y + 14, text=so(phan, 3), fill="white", font=("Segoe UI", 8, "bold"))
                x += dx
            cv.create_text(x + 8, y + 14, text=f"D = {so(xh['D'][i], 4)}  →  S = {so(xh['S'][i], 4)}",
                           anchor="w", font=FONT)

        phai = tk.Frame(f, bg="white")
        phai.pack(side="left", anchor="n", fill="both", expand=True, padx=6, pady=4)
        self.tieu_de(phai, "LỆNH SQL ĐÃ CHẠY (SQLite, " + os.path.basename(C.DB_PATH) + ")", "white")
        txt = tk.Text(phai, height=21, width=72, font=("Consolas", 9), wrap="word", bg="#f7f7f7", relief="flat")
        txt.pack(fill="both", expand=True)
        ten = F.FEATURE_NAMES
        sql1 = DB.SQL_DAC_TRUNG.replace(", ".join(ten), f"{ten[0]}, {ten[1]}, …, {ten[-1]}")
        dong = ["-- 1) Vector đặc trưng (giá trị gốc) của mọi ảnh:", sql1 + ";",
                f"--    → {len(xh['ids'])} hàng × {len(ten)} cột REAL", "",
                f"-- 2) Thông tin 5 ảnh kết quả, tra bảng anh theo ma_anh (chạy {len(top)} lần):"]
        dong += [DB.SQL_ANH + ";"]
        dong += [f"--    ? = '{a['ma_anh']}' → {a['duong_dan']} ({a['rong']}×{a['cao']} {a['dinh_dang']})"
                 for a in r["anh_top"]]
        dong += ["", "-- Không đọc từ CSDL (tham số tìm kiếm, không phải dữ liệu của ảnh):",
                 f"--   min, max {len(ten)} cột: data/norm_params.json (= MIN(), MAX() của dac_trung)",
                 "--   trọng số 3 bộ: " + (", ".join(f"{b} = {so(self.ts_db[b])}" for b in F.BO)
                                         if r["ts_thu"] is None else "đang THỬ trên thanh trượt")]
        dong += ["", "-- Ảnh truy vấn chỉ tính vector trong bộ nhớ, không ghi vào CSDL.",
                 f"-- So 1 vector truy vấn với {len(xh['ids'])} vector bằng numpy (search.py)."]
        txt.insert("1.0", "\n".join(dong))
        txt.config(state="disabled")

    # ------------------------------------------------------------ cửa sổ thế số
    def chi_tiet(self, i):
        """Cửa sổ thế số đầy đủ cho cặp (ảnh truy vấn, ảnh kết quả i)."""
        r = self.ket_qua
        xh, v, kq = r["xh"], r["v"], r["kq"]
        ma = xh["ids"][i]
        hang = self.hang_cua(i)
        w = tk.Toplevel(self.root)
        w.title(f"Cách tính độ tương đồng: ảnh truy vấn ~ {ma} (#{hang})")
        w.geometry(f"1180x{min(900, self.root.winfo_screenheight() - 80)}+40+30")
        w.anh_tk = []
        tren = tk.Frame(w)
        tren.pack(fill="x", padx=8, pady=6)
        a = DB.duong_dan_day_du(r["anh_top"][xh["top"].index(i)]["duong_dan"])
        for j, (img, nhan) in enumerate([(self.anh(r["path"], 130, w.anh_tk), "truy vấn (gốc)"),
                                         (self.anh_ve(kq["rgb"], kq["pts"], 130, w.anh_tk), "truy vấn (đã xử lý)"),
                                         (self.anh(a, 130, w.anh_tk), f"{ma} (gốc)"),
                                         (self.anh_ve(*self.anh_csdl(ma), 130, w.anh_tk), f"{ma} (đã xử lý)")]):
            tk.Label(tren, image=img).grid(row=0, column=j, padx=4)
            tk.Label(tren, text=nhan, font=FONT).grid(row=1, column=j)
        tk.Label(tren, text=f"Hạng #{hang} / {int(xh['con_lai'].sum())}\n\nS = {so(xh['S'][i], 4)}\n"
                            f"D = {so(xh['D'][i], 4)}", font=("Segoe UI", 14, "bold"), justify="left",
                 fg="#1a3d7c").grid(row=0, column=4, padx=20, sticky="w")

        khung = tk.Frame(w)
        khung.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        txt = tk.Text(khung, font=("Consolas", 10), wrap="none")
        cy = tk.Scrollbar(khung, orient="vertical", command=txt.yview)
        cx = tk.Scrollbar(khung, orient="horizontal", command=txt.xview)
        txt.configure(yscrollcommand=cy.set, xscrollcommand=cx.set)
        cy.pack(side="right", fill="y")
        cx.pack(side="bottom", fill="x")
        txt.pack(side="left", fill="both", expand=True)
        txt.tag_configure("muc", font=("Consolas", 11, "bold"), foreground="#1a3d7c", spacing1=8)
        txt.tag_configure("dam", font=("Consolas", 10, "bold"))
        txt.tag_configure("kq", font=("Consolas", 10, "bold"), foreground="#b00020")
        for b in F.BO:
            txt.tag_configure(b, font=("Consolas", 10, "bold"), foreground=MAU_BO[b])

        def ghi(chu="", tag=None):
            txt.insert("end", chu + "\n", tag)

        q, x = xh["q"], xh["Xn"][i]
        mn, mx = xh["mn"], xh["mx"]
        so4 = lambda k, t: so(t, 1) if k == 5 else so(t, 4)

        ghi("BƯỚC 1. CHUẨN HÓA MIN-MAX TỪNG CHIỀU", "muc")
        ghi("   x' = (x − min) / (max − min), kẹp vào [0, 1].  min, max của từng cột đọc từ norm_params.json,")
        ghi(f"   tính trên toàn bộ {len(xh['ids'])} ảnh của CSDL. q' là giá trị chuẩn hóa của ảnh truy vấn, x' của ảnh {ma}.")
        tieu = (f"   {'Ký hiệu':<7} {'Đặc trưng':<34} {'min':>9} {'max':>9} | {'x truy vấn':>10} {'q_':>7} |"
                f" {'x ' + ma:>14} {'x_':>7} | {'(q_ − x_)²':>11}").replace("q_", "q'").replace("x_", "x'")
        d_rieng = {}
        for b, vt in F.BO.items():
            n = vt.stop - vt.start
            ghi()
            ghi(f"── Bộ {TEN_BO[b].lower()} ({n} chiều) " + "─" * 70, b)
            ghi(tieu, "dam")
            binh_phuong = []
            for k in range(vt.start, vt.stop):
                ten = self.tt[k]["ten_dac_trung"] if k < 7 else o_mau(k - 7)
                bp = (q[k] - x[k]) ** 2
                binh_phuong.append(bp)
                ghi(f"   {self.tt[k]['ky_hieu']:<7} {ten[:34]:<34} {so4(k, mn[k]):>9} {so4(k, mx[k]):>9} |"
                    f" {so4(k, v[k]):>10} {so(q[k], 4):>7} | {so4(k, xh['X'][i, k]):>14} {so(x[k], 4):>7} |"
                    f" {so(bp, 5):>11}")
                if k < 7:       # thế số cụ thể cho 7 đặc trưng hình học, kết cấu
                    for nhan, gt, gn in (("q'", v[k], q[k]), ("x'", xh["X"][i, k], x[k])):
                        tho = (gt - mn[k]) / (mx[k] - mn[k]) if mx[k] > mn[k] else 0.0
                        kep = f" → kẹp = {so(gn, 4)}" if abs(tho - gn) > 1e-12 else ""
                        ghi(f"{'':<12}{nhan} = ({so4(k, gt)} − {so4(k, mn[k])}) / ({so4(k, mx[k])} − {so4(k, mn[k])})"
                            f" = {so(tho, 4)}{kep}")
            tong = float(sum(binh_phuong))
            d = np.sqrt(tong) / np.sqrt(n)
            d_rieng[b] = d
            ghi()
            if n <= 7:
                ghi(f"   Σ(q' − x')² = " + " + ".join(so(t, 5) for t in binh_phuong) + f" = {so(tong, 5)}")
            else:
                ghi(f"   Σ(q' − x')² = tổng {n} số ở cột cuối = {so(tong, 5)}")
            ghi(f"   {KY_HIEU_BO[b]} = √Σ / √{n} = √{so(tong, 5)} / √{n} = {so(np.sqrt(tong), 4)} / {so(np.sqrt(n), 4)}"
                f" = {so(d, 4)}", b)
            gan_hon = 100.0 * (xh["d_bo"][b][xh["con_lai"]] > xh["d_bo"][b][i]).mean()
            ghi(f"   ({KY_HIEU_BO[b]} của {ma} nhỏ hơn {so(gan_hon, 1)}% số ảnh trong CSDL)")

        ghi()
        ghi("BƯỚC 2. KHOẢNG CÁCH CHUNG (trọng số 1/3 mỗi bộ, Báo cáo 1)" if r["ts_thu"] is None
            else "BƯỚC 2. KHOẢNG CÁCH CHUNG (đang THỬ trọng số trên thanh trượt)", "muc")
        ghi("   D = " + " + ".join(f"{self.ts_chu(b)} × {KY_HIEU_BO[b]}" for b in F.BO))
        ghi("     = " + " + ".join(f"{self.ts_chu(b)} × {so(d_rieng[b], 4)}" for b in F.BO))
        ghi("     = " + " + ".join(so(xh["trong_so"][b] * d_rieng[b], 4) for b in F.BO) + f" = {so(xh['D'][i], 4)}", "kq")
        phan = {b: xh["trong_so"][b] * d_rieng[b] for b in F.BO}
        ghi("   Phần đóng góp vào D: " + ", ".join(f"{TEN_BO[b].lower()} {so(100 * phan[b] / max(xh['D'][i], 1e-12), 0)}%"
                                                  for b in F.BO))

        ghi()
        ghi("BƯỚC 3. ĐỘ TƯƠNG ĐỒNG", "muc")
        ghi(f"   S = 1 / (1 + D) = 1 / (1 + {so(xh['D'][i], 4)}) = {so(xh['S'][i], 4)}", "kq")
        ghi("   (D = 0 → S = 1: hai vector trùng nhau; D càng lớn → S càng gần 0)")

        ghi()
        ghi("BƯỚC 4. XẾP HẠNG", "muc")
        xet = np.where(xh["con_lai"])[0]
        thu_tu = xet[np.argsort(-xh["S"][xet], kind="stable")]
        ghi(f"   Tính S cho cả {len(xet)} ảnh rồi sắp giảm dần; {ma} đứng thứ {hang}.")
        for h_, j in enumerate(thu_tu[:7], start=1):
            dau = "  ← ảnh này" if j == i else ("" if h_ <= 5 else "  (ngoài top-5)")
            ghi(f"   #{h_:<2} {xh['ids'][j]}   S = {so(xh['S'][j], 4)}   D = {so(xh['D'][j], 4)}{dau}",
                "dam" if j == i else None)

        ghi()
        ghi("NHẬN XÉT (máy tính số liệu, người đọc và giải thích)", "muc")
        ghi("   Đặc trưng tìm tương đồng (diện mạo chung), truy vấn ~ ảnh này:")
        for k in self.tuong_dong:
            ghi(f"      {self.ten(k)}: {so_x(k, v[k])} ~ {so_x(k, xh['X'][i, k])}   (|q' − x'| = {so(abs(q[k] - x[k]))})")
        kq_, kx = int(np.argmax(v[7:])), int(np.argmax(xh["X"][i, 7:]))
        ghi(f"      ô màu da nhiều nhất: {o_mau(kq_)} ({so(100 * v[7 + kq_], 0)}%) ~ {o_mau(kx)} ({so(100 * xh['X'][i, 7 + kx], 0)}%)")
        lech = {k: abs(x[k] - q[k]) for k in self.khac_biet}
        gan, xa = min(lech, key=lech.get), max(lech, key=lech.get)
        ghi("   Đặc trưng tìm khác biệt (từng người):")
        ghi(f"      gần nhất  {self.ten(gan)}: {so_x(gan, v[gan])} ~ {so_x(gan, xh['X'][i, gan])}   (|q' − x'| = {so(lech[gan])})")
        ghi(f"      lệch nhất {self.ten(xa)}: {so_x(xa, v[xa])} so với {so_x(xa, xh['X'][i, xa])}   (|q' − x'| = {so(lech[xa])})")
        txt.config(state="disabled")


if __name__ == "__main__":
    try:                                    # Windows: vẽ theo đúng độ phân giải màn hình (chữ, ảnh không bị nhòe)
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()
