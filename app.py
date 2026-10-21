"""
app.py - Giao diện demo (Tkinter): chọn một ảnh chân dung -> 5 ảnh giống nhất trong CSDL.

    Upload ảnh            : ảnh bất kỳ (JPG). Ảnh chưa đúng 256 x 256 được làm vuông quanh khuôn
                            mặt rồi đưa về 256 x 256 (preprocess.lam_vuong), sau đó qua đủ 6 bước.
    Chọn ảnh trong CSDL   : lấy một ảnh có sẵn làm truy vấn, không cho chính nó vào kết quả.
    Hiện điểm mốc, vùng   : vẽ 68 điểm mốc, vùng cằm - miệng, vùng lông mày, vùng da lên ảnh đã xử lý.

Mỗi hàng kết quả: ảnh gốc và ảnh đã căn chỉnh, lọc nền (cùng cỡ 256 x 256), độ tương đồng S,
khoảng cách từng bộ thuộc tính và các đặc trưng giống nhất / khác nhất với ảnh truy vấn.
Toàn bộ dữ liệu (vector, min, max, trọng số) đọc từ CSDL qua search.py.

Chạy:  python app.py
"""
import os
import time
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageTk

import common as C
import preprocess
import features as F
import search as S
import database as DB

TEN_BO = {"hinh_hoc": "Hình học (dáng mặt, mắt, mũi)", "ket_cau": "Kết cấu (lông mày, cằm - miệng)",
          "mau_sac": "Màu sắc (tông da)"}
TEN_NGAN = {"hinh_hoc": "hình học", "ket_cau": "kết cấu", "mau_sac": "màu sắc"}
LY_DO = {"khong phai JPG": "Tệp không phải ảnh JPG (hệ thống chỉ dùng định dạng JPG).",
         "khong doc duoc": "Không đọc được tệp ảnh.",
         "sai kich thuoc/so kenh": "Ảnh không đúng 256 × 256 × 3 kênh màu.",
         "anh mo": "Ảnh mờ: độ nét {do_net} nhỏ hơn ngưỡng 20.",
         "so khuon mat khac 1": "Không tìm thấy đúng một khuôn mặt trong ảnh.",
         "diem moc khong on dinh": "Điểm mốc không ổn định: độ lệch {lech_moc} từ 0,08 trở lên."}
CO_ANH = 256


def so(x, n=3):
    """Số kiểu Việt Nam: dấu phẩy thập phân."""
    return f"{x:.{n}f}".replace(".", ",")


def so_x(k, x):
    return so(x, 1) if k == 5 else so(x, 3)


def o_mau(k):
    """Mô tả ô màu thứ k của lược đồ H x S (k = ô H * 4 + ô S)."""
    i, j = divmod(k, F.SO_O_S)
    h = f"H {i * 5}–{i * 5 + 5}°" if 0 < i < F.SO_O_H - 1 else ("H 0–5° (cả ≥ 330°)" if i == 0 else "H 35–330°")
    s = [f"{x:g}".replace(".", ",") for x in (j * 0.25, j * 0.25 + 0.25)]
    return f"{h}, S {s[0]}–{s[1]}"


def ve_vung(anh, pts):
    """Vẽ 68 điểm mốc và các vùng lấy đặc trưng lên ảnh đã xử lý (chỉ để xem)."""
    img = Image.fromarray(anh).convert("RGB")
    dr = ImageDraw.Draw(img)
    for idx in ([1, 2, 3, 31, 40, 41], [15, 14, 13, 35, 47, 46]):                 # hai má (vùng da)
        dr.polygon([tuple(p) for p in pts[idx]], outline=(0, 170, 0))
    trai, phai = C.eye_centers(pts)
    d = C.dist(trai, phai)
    dinh = pts[17:27, 1].min()
    dr.rectangle([pts[19, 0], dinh - 0.45 * d, pts[24, 0], dinh - 0.10 * d], outline=(0, 170, 0))  # trán
    dr.polygon([tuple(p) for p in pts[[33, 54, 12, 8, 4, 48]]], outline=(30, 90, 255))           # cằm - miệng
    for nhom in (pts[17:22], pts[22:27]):                                                       # lông mày
        dr.rectangle([nhom[:, 0].min(), nhom[:, 1].min() - 0.15 * d, nhom[:, 0].max(), nhom[:, 1].max() + 0.15 * d],
                     outline=(255, 140, 0))
    for x, y in pts:
        dr.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(230, 0, 0))
    return img


class App:
    def __init__(self, root):
        self.root = root
        root.title("Tìm kiếm ảnh chân dung đàn ông - Nhóm 19")
        root.geometry(f"{min(1420, root.winfo_screenwidth() - 40)}x{min(940, root.winfo_screenheight() - 80)}+10+10")
        self.ids, self.X = S.load_features()
        self.vi_tri = {m: i for i, m in enumerate(self.ids)}
        self.lm = F.read_landmarks()
        con = DB.ket_noi()
        self.tt = DB.doc_thuoc_tinh(con)          # từ điển đặc trưng: ký hiệu, tên, vai trò (Bảng 2.1)
        con.close()
        self.tuong_dong = [k for k in range(7) if self.tt[k]["vai_tro"] == "tuong_dong"]    # X0, X1
        self.khac_biet = [k for k in range(7) if self.tt[k]["vai_tro"] == "khac_biet"]      # X2 - X6
        preprocess.detect_landmarks(np.zeros((64, 64), dtype=np.uint8))     # nạp sẵn mô hình dlib
        self.ket_qua = None
        self.anh_tk = []

        thanh = tk.Frame(root, pady=6, padx=8)
        thanh.pack(fill="x")
        tk.Button(thanh, text="Upload ảnh", width=16, command=self.upload, font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Button(thanh, text="Chọn ảnh trong CSDL", width=20, command=self.chon_trong_csdl).pack(side="left", padx=6)
        self.ve = tk.BooleanVar(value=False)
        tk.Checkbutton(thanh, text="Hiện điểm mốc và vùng đặc trưng", variable=self.ve,
                       command=self.hien_thi).pack(side="left", padx=10)
        self.trang_thai = tk.Label(thanh, anchor="w", fg="#444",
                                   text=f"CSDL: {len(self.ids)} ảnh, {len(F.FEATURE_NAMES)} đặc trưng ({os.path.basename(C.DB_PATH)})")
        self.trang_thai.pack(side="left", padx=10)

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
        root.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-e.delta / 120), "units"))
        tk.Label(self.noi_dung, text="Bấm \"Upload ảnh\" để chọn một ảnh chân dung (JPG).",
                 font=("Segoe UI", 12), pady=40, padx=20).pack()

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
            kq = preprocess.process_one(path)
            if not kq["dat"]:
                self.bao_loi(kq["ly_do"], kq)
                return
            v = F.extract(kq["rgb"], kq["gray"], kq["pts"])
            xh = S.xep_hang(v, bo_qua=bo_qua)
            self.ket_qua = dict(path=path, kq=kq, v=v, xh=xh, bo_qua=bo_qua, ghi_chu=ghi_chu,
                                thoi_gian=time.time() - t0)
            self.hien_thi()
        finally:
            self.root.config(cursor="")

    def bao_loi(self, ly_do, kq):
        thong_bao = LY_DO.get(ly_do, ly_do).format(do_net=so(kq.get("do_net") or 0, 1),
                                                   lech_moc=so(kq.get("lech_moc") or 0, 3))
        self.trang_thai.config(text="Ảnh không đạt tiền xử lý: " + thong_bao)
        messagebox.showwarning("Ảnh không đạt tiền xử lý", thong_bao + "\nẢnh không được đưa vào tìm kiếm.")

    # ------------------------------------------------------------ hiển thị
    def anh(self, mang_hoac_duong_dan, pts=None):
        if isinstance(mang_hoac_duong_dan, str):
            img = Image.open(mang_hoac_duong_dan).convert("RGB")
        elif pts is not None and self.ve.get():
            img = ve_vung(mang_hoac_duong_dan, pts)
        else:
            img = Image.fromarray(mang_hoac_duong_dan)
        anh_tk = ImageTk.PhotoImage(img.resize((CO_ANH, CO_ANH)))
        self.anh_tk.append(anh_tk)
        return anh_tk

    def hang(self, tieu_de, anh_goc, anh_xl, chu, nen):
        f = tk.Frame(self.noi_dung, bg=nen, bd=1, relief="solid", padx=8, pady=6)
        f.pack(fill="x", padx=8, pady=4)
        tk.Label(f, text=tieu_de, bg=nen, font=("Segoe UI", 11, "bold"), width=12, anchor="n",
                 justify="center").grid(row=0, column=0, sticky="n")
        for cot, (a, nhan) in enumerate([(anh_goc, "ảnh gốc"), (anh_xl, "đã căn chỉnh, lọc nền")], start=1):
            tk.Label(f, image=a, bg=nen).grid(row=0, column=cot, padx=4)
            tk.Label(f, text=nhan, bg=nen, fg="#555").grid(row=1, column=cot)
        tk.Label(f, text=chu, bg=nen, justify="left", anchor="nw", font=("Segoe UI", 10),
                 wraplength=680).grid(row=0, column=3, rowspan=2, sticky="nw", padx=10)

    def hien_thi(self):
        if self.ket_qua is None:
            return
        for w in self.noi_dung.winfo_children():
            w.destroy()
        self.anh_tk = []
        r = self.ket_qua
        kq, v, xh = r["kq"], r["v"], r["xh"]

        # hàng ảnh truy vấn
        chu = [r["ghi_chu"]] if r["ghi_chu"] else []
        chu.append(f"Tiền xử lý: đạt  |  độ nét {so(kq['do_net'], 1)}  |  độ lệch điểm mốc {so(kq['lech_moc'])}"
                   f"  |  góc nghiêng {so(kq['goc_nghieng'], 1)}°")
        chu.append("Đặc trưng (giá trị gốc):")
        for k in range(7):
            chu.append(f"    {self.ten(k)} = {so_x(k, v[k])}")
        k = int(np.argmax(v[7:]))
        chu.append(f"    X7 – X38 lược đồ màu da: ô nhiều nhất {o_mau(k)} ({so(100 * v[7 + k], 0)}% điểm da)")
        if r["bo_qua"]:
            chu.append(f"Ảnh này có trong CSDL ({r['bo_qua']}) nên không đưa chính nó vào kết quả.")
        chu.append(f"Thời gian tiền xử lý + tìm kiếm: {so(r['thoi_gian'], 2)} s")
        self.hang("ẢNH\nTRUY VẤN", self.anh(r["path"]), self.anh(kq["rgb"], kq["pts"]), "\n".join(chu), "#eef3fb")

        # 5 hàng kết quả
        for hang, i in enumerate(xh["top"], start=1):
            ma = self.ids[i]
            chu = self.giai_thich(i)
            goc = os.path.join(C.RAW_DIR, ma + ".jpg")
            xl = np.array(Image.open(os.path.join(C.ALIGNED_DIR, ma + ".jpg")).convert("RGB"))
            self.hang(f"#{hang}\n{ma}", self.anh(goc), self.anh(xl, self.lm.get(ma)), chu, "white")
        self.trang_thai.config(text=f"Xong: top-5 trong {len(self.ids)} ảnh của CSDL "
                                    f"({so(r['thoi_gian'], 2)} s).")
        self.canvas.yview_moveto(0)

    def giai_thich(self, i):
        """Phần chữ của một hàng kết quả: S, D, D từng bộ và các đặc trưng giống / khác nhất."""
        xh, v = self.ket_qua["xh"], self.ket_qua["v"]
        ts = {b: "1/3" if abs(w - 1 / 3) < 1e-9 else so(w, 3) for b, w in xh["trong_so"].items()}
        chu = [f"Độ tương đồng S = 1 / (1 + D) = {so(xh['S'][i], 4)}",
               "D = " + " + ".join(f"{ts[b]} × D_{TEN_NGAN[b]}" for b in F.BO),
               "    = " + " + ".join(f"{ts[b]} × {so(xh['d_bo'][b][i], 3)}" for b in F.BO) + f" = {so(xh['D'][i], 4)}"]
        con_lai = xh["con_lai"]
        for b, ten in TEN_BO.items():
            gan_hon = 100.0 * (xh["d_bo"][b][con_lai] > xh["d_bo"][b][i]).mean()
            muc = "gần nhất trong CSDL" if gan_hon >= 99.95 else f"gần hơn {so(gan_hon, 1)}% số ảnh"
            chu.append(f"   • {ten}: D = {so(xh['d_bo'][b][i], 3)}, {muc}")
        # đặc trưng tương đồng: dáng mặt và tông da có gần nhau không
        chu.append("Đặc trưng tương đồng (diện mạo chung), truy vấn ~ ảnh này:")
        for k in self.tuong_dong:
            chu.append(f"    {self.ten(k)}: {so_x(k, v[k])} ~ {so_x(k, self.X[i, k])}")
        kq, kx = int(np.argmax(v[7:])), int(np.argmax(self.X[i, 7:]))
        chu.append(f"    ô màu da nhiều nhất: {o_mau(kq)} ({so(100 * v[7 + kq], 0)}%) ~ "
                   f"{o_mau(kx)} ({so(100 * self.X[i, 7 + kx], 0)}%)")
        # đặc trưng khác biệt: trong 5 đặc trưng X2 - X6, cái gần nhất và cái lệch nhất (theo giá trị đã chuẩn hóa)
        lech = {k: abs(xh["Xn"][i, k] - xh["q"][k]) for k in self.khac_biet}
        gan, xa = min(lech, key=lech.get), max(lech, key=lech.get)
        chu.append("Đặc trưng khác biệt (từng người):")
        chu.append(f"    gần nhất {self.ten(gan)}: {so_x(gan, v[gan])} ~ {so_x(gan, self.X[i, gan])}")
        chu.append(f"    lệch nhất {self.ten(xa)}: {so_x(xa, v[xa])} so với {so_x(xa, self.X[i, xa])}")
        return "\n".join(chu)

    def ten(self, k):
        """Ký hiệu + tên đặc trưng thứ k, lấy từ bảng thuoc_tinh (ví dụ "X0 tỉ lệ khung mặt")."""
        return f"{self.tt[k]['ky_hieu']} {self.tt[k]['ten_dac_trung'][0].lower()}{self.tt[k]['ten_dac_trung'][1:]}"


if __name__ == "__main__":
    try:                                    # Windows: vẽ theo đúng độ phân giải màn hình (chữ, ảnh không bị nhòe)
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()
