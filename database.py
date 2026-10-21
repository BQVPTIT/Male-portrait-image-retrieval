"""
database.py - CSDL quan hệ (SQLite) quản trị đặc trưng của ảnh.

Bốn bảng:
    anh           ma_anh (PK), duong_dan (UNIQUE), rong, cao, dinh_dang, ngay_nap
                  -> đặc trưng tổng quát của ảnh; ảnh gốc lưu ngoài CSDL, bảng chỉ giữ đường dẫn
    dac_trung     ma_anh (PK, đồng thời FK -> anh) + 39 cột REAL đặt tên theo features.FEATURE_NAMES
                  -> đặc trưng nội dung, mỗi hàng là một vector 39 chiều, lưu GIÁ TRỊ GỐC
    bo_thuoc_tinh ma_bo (PK), ten_bo, so_chieu, trong_so
                  -> 3 bộ thuộc tính: hình học (5), kết cấu (2), màu sắc (32)
    thuoc_tinh    ten_cot (PK), ky_hieu (UNIQUE), ten_dac_trung, ma_bo (FK -> bo_thuoc_tinh),
                  vai_tro, gia_tri_min, gia_tri_max
                  -> từ điển dữ liệu: mỗi hàng mô tả một cột của dac_trung, kèm min, max để chuẩn hóa

Quan hệ: anh 1 - 1 dac_trung (qua ma_anh); bo_thuoc_tinh 1 - n thuoc_tinh (qua ma_bo).

SQLite: cả CSDL là một tệp data/khuon_mat.db, đọc ghi bằng thư viện chuẩn sqlite3 của Python,
không cần máy chủ. Câu lệnh SQL (CREATE TABLE, INSERT, SELECT, khóa chính, khóa ngoại) giống MySQL.

Chạy:  python database.py      (tạo lại CSDL từ data/features.csv + data/norm_params.json rồi kiểm tra)
"""
import os
import csv
import json
import shutil
import sqlite3
import datetime
import numpy as np
import cv2

import common as C
import preprocess
import features as F

# ---------------------------------------------------------------- dữ liệu mô tả
# 3 bộ thuộc tính của Báo cáo 1. Trọng số là trọng số của khoảng cách từng bộ khi gộp (search.py).
BO_THUOC_TINH = [("hinh_hoc", "Hình học", 5, 1 / 3),
                 ("ket_cau", "Kết cấu", 2, 1 / 3),
                 ("mau_sac", "Màu sắc", 32, 1 / 3)]

TEN_DAC_TRUNG = (["Tỉ lệ khung mặt", "Tỉ lệ gò má trên quai hàm", "Tỉ lệ khoảng cách hai mắt",
                  "Tỉ lệ độ rộng cánh mũi", "Tỉ lệ chiều cao mũi",
                  "Phương sai LBP vùng cằm - miệng", "Mật độ lông mày"]
                 + [f"Lược đồ màu da, ô H{i} - ô S{j}" for i in range(F.SO_O_H) for j in range(F.SO_O_S)])

# Vai trò theo Bảng 2.1 Báo cáo 1: X0, X1 (dáng mặt) và X7 - X38 (tông da) tìm sự tương đồng;
# X2 - X4 (tỉ lệ ngũ quan) và X5, X6 (bề mặt) tìm sự khác biệt.
VAI_TRO = ["tuong_dong"] * 2 + ["khac_biet"] * 5 + ["tuong_dong"] * 32

# ---------------------------------------------------------------- lệnh tạo bảng
COT_DAC_TRUNG = ",\n".join(f"    {ten:<13} REAL NOT NULL" for ten in F.FEATURE_NAMES)

DDL = f"""
CREATE TABLE anh (
    ma_anh     TEXT PRIMARY KEY,                   -- man_0001
    duong_dan  TEXT NOT NULL UNIQUE,               -- data/raw256/man_0001.jpg
    rong       INTEGER NOT NULL CHECK (rong > 0),  -- 256
    cao        INTEGER NOT NULL CHECK (cao > 0),   -- 256
    dinh_dang  TEXT NOT NULL CHECK (dinh_dang = 'JPG'),
    ngay_nap   TEXT NOT NULL                       -- ngày nạp vào CSDL, dạng YYYY-MM-DD
);

CREATE TABLE bo_thuoc_tinh (
    ma_bo      TEXT PRIMARY KEY,                   -- hinh_hoc / ket_cau / mau_sac
    ten_bo     TEXT NOT NULL,
    so_chieu   INTEGER NOT NULL CHECK (so_chieu > 0),
    trong_so   REAL NOT NULL CHECK (trong_so >= 0)
);

CREATE TABLE thuoc_tinh (
    ten_cot       TEXT PRIMARY KEY,                -- tên cột trong bảng dac_trung
    ky_hieu       TEXT NOT NULL UNIQUE,            -- X0 ... X38
    ten_dac_trung TEXT NOT NULL,
    ma_bo         TEXT NOT NULL REFERENCES bo_thuoc_tinh(ma_bo),
    vai_tro       TEXT NOT NULL CHECK (vai_tro IN ('tuong_dong', 'khac_biet')),
    gia_tri_min   REAL NOT NULL,                   -- min, max của cột trên toàn CSDL,
    gia_tri_max   REAL NOT NULL,                   -- dùng để chuẩn hóa min-max
    CHECK (gia_tri_min <= gia_tri_max)
);

CREATE TABLE dac_trung (
    ma_anh        TEXT PRIMARY KEY REFERENCES anh(ma_anh),
{COT_DAC_TRUNG}
);
"""


# ---------------------------------------------------------------- kết nối
def ket_noi(db_path=C.DB_PATH):
    """Mở kết nối tới tệp CSDL và bật kiểm tra khóa ngoại.

    SQLite mặc định KHÔNG kiểm tra khóa ngoại; lệnh PRAGMA phải chạy lại mỗi lần mở kết nối.
    """
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def duong_dan_day_du(duong_dan):
    """Đường dẫn trong bảng anh là đường dẫn tương đối so với thư mục dự án."""
    return os.path.join(C.ROOT, *duong_dan.split("/"))


# ---------------------------------------------------------------- tạo và nạp CSDL
def doc_features_csv():
    """Đọc data/features.csv (đầu ra của bước trích rút) -> (mã ảnh, ma trận N x 39 giá trị gốc)."""
    with open(C.FEATURE_CSV, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["ma_anh"] + F.FEATURE_NAMES, "features.csv khong dung thu tu FEATURE_NAMES"
    return [r[0] for r in rows[1:]], np.array([r[1:] for r in rows[1:]], dtype=np.float64)


def tao_csdl(db_path=C.DB_PATH):
    """Tạo lại CSDL và nạp dữ liệu từ kết quả có sẵn (KHÔNG tính lại đặc trưng):

        data/features.csv     -> bảng dac_trung (giá trị gốc của 39 đặc trưng)
        data/norm_params.json -> gia_tri_min, gia_tri_max của bảng thuoc_tinh
        data/raw256/*.jpg     -> bảng anh (đường dẫn, kích thước đọc từ tệp ảnh)
    Toàn bộ lệnh INSERT nằm trong MỘT giao dịch: lỗi ở bất kỳ dòng nào thì không dòng nào được ghi.
    """
    if os.path.exists(db_path):
        os.remove(db_path)
    ids, X = doc_features_csv()
    with open(C.NORM_JSON, encoding="utf-8") as f:
        p = json.load(f)
    assert p["ten"] == F.FEATURE_NAMES, "norm_params.json khong dung thu tu FEATURE_NAMES"

    con = ket_noi(db_path)
    con.executescript(DDL)
    ngay = datetime.date.today().isoformat()
    cot = ", ".join(["ma_anh"] + F.FEATURE_NAMES)
    dau_hoi = ", ".join(["?"] * (1 + len(F.FEATURE_NAMES)))
    with con:
        con.executemany("INSERT INTO bo_thuoc_tinh VALUES (?, ?, ?, ?)", BO_THUOC_TINH)
        for j, ten in enumerate(F.FEATURE_NAMES):
            ma_bo = next(bo for bo, vt in F.BO.items() if vt.start <= j < vt.stop)
            con.execute("INSERT INTO thuoc_tinh VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (ten, f"X{j}", TEN_DAC_TRUNG[j], ma_bo, VAI_TRO[j], p["min"][j], p["max"][j]))
        for ma, v in zip(ids, X):
            duong_dan = f"data/raw256/{ma}.jpg"
            rgb = preprocess.read_rgb(duong_dan_day_du(duong_dan))
            con.execute("INSERT INTO anh VALUES (?, ?, ?, ?, ?, ?)",
                        (ma, duong_dan, rgb.shape[1], rgb.shape[0], "JPG", ngay))
            con.execute(f"INSERT INTO dac_trung ({cot}) VALUES ({dau_hoi})", [ma] + v.tolist())
    con.close()
    print(f"Tao CSDL xong: {len(ids)} anh -> {db_path}")


# ---------------------------------------------------------------- đọc dữ liệu
def doc_dac_trung(con):
    """SELECT toàn bộ bảng dac_trung -> (danh sách mã ảnh, ma trận N x 39 giá trị gốc)."""
    cot = ", ".join(F.FEATURE_NAMES)
    rows = con.execute(f"SELECT ma_anh, {cot} FROM dac_trung ORDER BY ma_anh").fetchall()
    return [r[0] for r in rows], np.array([r[1:] for r in rows], dtype=np.float64)


def doc_min_max(con):
    """min, max của 39 cột (bảng thuoc_tinh), xếp đúng thứ tự FEATURE_NAMES."""
    bang = {r[0]: r[1:] for r in con.execute("SELECT ten_cot, gia_tri_min, gia_tri_max FROM thuoc_tinh")}
    mm = np.array([bang[ten] for ten in F.FEATURE_NAMES], dtype=np.float64)
    return mm[:, 0], mm[:, 1]


def doc_thuoc_tinh(con):
    """Từ điển đặc trưng (bảng thuoc_tinh), xếp đúng thứ tự FEATURE_NAMES -> danh sách dict."""
    cot = ["ten_cot", "ky_hieu", "ten_dac_trung", "ma_bo", "vai_tro", "gia_tri_min", "gia_tri_max"]
    bang = {r[0]: dict(zip(cot, r)) for r in con.execute(f"SELECT {', '.join(cot)} FROM thuoc_tinh")}
    return [bang[ten] for ten in F.FEATURE_NAMES]


def doc_trong_so(con):
    """Trọng số của từng bộ thuộc tính (bảng bo_thuoc_tinh) -> {ma_bo: trọng số}."""
    return dict(con.execute("SELECT ma_bo, trong_so FROM bo_thuoc_tinh").fetchall())


def doc_anh(con, ma_anh):
    """Thông tin chung của một ảnh (bảng anh) -> dict, hoặc None nếu không có."""
    r = con.execute("SELECT ma_anh, duong_dan, rong, cao, dinh_dang, ngay_nap FROM anh WHERE ma_anh = ?",
                    (ma_anh,)).fetchone()
    return None if r is None else dict(zip(["ma_anh", "duong_dan", "rong", "cao", "dinh_dang", "ngay_nap"], r))


# ---------------------------------------------------------------- kiểm tra toàn vẹn
CAU_KIEM_TRA = [
    ("So anh", "SELECT COUNT(*) FROM anh"),
    ("So vector dac trung", "SELECT COUNT(*) FROM dac_trung"),
    ("Anh chua co vector", "SELECT COUNT(*) FROM anh a LEFT JOIN dac_trung d ON a.ma_anh = d.ma_anh "
                           "WHERE d.ma_anh IS NULL"),
    ("Vector khong thuoc anh nao", "SELECT COUNT(*) FROM dac_trung d LEFT JOIN anh a ON d.ma_anh = a.ma_anh "
                                   "WHERE a.ma_anh IS NULL"),
    ("So thuoc tinh", "SELECT COUNT(*) FROM thuoc_tinh"),
    ("So thuoc tinh theo bo", "SELECT ma_bo, COUNT(*) FROM thuoc_tinh GROUP BY ma_bo ORDER BY ma_bo"),
    ("Bo co so_chieu sai", "SELECT COUNT(*) FROM bo_thuoc_tinh b WHERE so_chieu <> "
                           "(SELECT COUNT(*) FROM thuoc_tinh t WHERE t.ma_bo = b.ma_bo)"),
    ("Vai tro", "SELECT vai_tro, COUNT(*) FROM thuoc_tinh GROUP BY vai_tro ORDER BY vai_tro"),
    ("Khoa ngoai cua dac_trung", "PRAGMA foreign_key_list(dac_trung)"),
    ("Khoa ngoai cua thuoc_tinh", "PRAGMA foreign_key_list(thuoc_tinh)"),
    ("Vi pham khoa ngoai", "PRAGMA foreign_key_check"),
]


def kiem_tra(db_path=C.DB_PATH):
    """Chạy các câu lệnh kiểm tra toàn vẹn và in kết quả. Trả về True nếu mọi kiểm tra đều đạt.

    Ngoài các câu lệnh trong CAU_KIEM_TRA còn kiểm tra:
      - cột của dac_trung trùng với ten_cot của thuoc_tinh (đúng thứ tự X0 ... X38);
      - min, max lưu trong thuoc_tinh bằng MIN(), MAX() của từng cột dac_trung (đúng ngay sau
        tao_csdl; sau khi them_anh một ảnh có giá trị ngoài [min, max] thì sẽ lệch, vì min, max giữ cố định);
      - khóa ngoại hoạt động: chèn vector của một ảnh không có thật phải bị từ chối.
    """
    con = ket_noi(db_path)
    kq = {}
    for ten, sql in CAU_KIEM_TRA:
        kq[ten] = con.execute(sql).fetchall()
        print(f"{ten:28s}: {kq[ten]}")

    cot = [r[1] for r in con.execute("PRAGMA table_info(dac_trung)")][1:]
    ten_cot = [r[0] for r in con.execute("SELECT ten_cot FROM thuoc_tinh "
                                         "ORDER BY CAST(SUBSTR(ky_hieu, 2) AS INTEGER)")]
    print(f"{'Cot dac_trung = thuoc_tinh':28s}: {cot == ten_cot == F.FEATURE_NAMES}")

    mn, mx = doc_min_max(con)
    sql = ", ".join(f"MIN({c}), MAX({c})" for c in F.FEATURE_NAMES)
    that = np.array(con.execute(f"SELECT {sql} FROM dac_trung").fetchone()).reshape(-1, 2)
    min_max_dung = bool((that[:, 0] == mn).all() and (that[:, 1] == mx).all())
    print(f"{'min, max = MIN(), MAX()':28s}: {min_max_dung}")

    try:
        with con:
            con.execute("INSERT INTO dac_trung (ma_anh, " + ", ".join(F.FEATURE_NAMES) + ") VALUES ("
                        + ", ".join(["'khong_co'"] + ["0"] * len(F.FEATURE_NAMES)) + ")")
        tu_choi = False
    except sqlite3.IntegrityError as e:
        tu_choi = True
        print(f"{'Chen vector anh khong co':28s}: bi tu choi ({e})")
    con.close()

    dat = (kq["So anh"] == kq["So vector dac trung"] and kq["Anh chua co vector"] == [(0,)]
           and kq["Vector khong thuoc anh nao"] == [(0,)] and kq["So thuoc tinh"] == [(39,)]
           and kq["Bo co so_chieu sai"] == [(0,)] and kq["Vi pham khoa ngoai"] == []
           and cot == ten_cot and min_max_dung and tu_choi)
    print("=> Toan ven:", "DAT" if dat else "KHONG DAT")
    return dat


# ---------------------------------------------------------------- nạp thêm ảnh
def them_anh(path, db_path=C.DB_PATH):
    """Nạp thêm một ảnh mới vào CSDL. Trả về (mã ảnh mới hoặc None, lý do).

    1. Ảnh qua đủ 6 bước tiền xử lý (preprocess.process_one), kể cả các bước kiểm tra.
       Không đạt thì từ chối, không ghi gì vào CSDL (đúng Báo cáo 1: ảnh không đạt không được nạp).
    2. Trích 39 đặc trưng (features.extract).
    3. Mã mới = man_xxxx với số thứ tự lớn nhất hiện có + 1. Trong MỘT giao dịch: INSERT bảng anh,
       INSERT bảng dac_trung, chép ảnh gốc vào data/raw256, lưu ảnh đã xử lý và 68 điểm mốc vào
       data/prep. Lỗi ở bất kỳ bước nào thì giao dịch bị hủy (ROLLBACK).
    min, max trong thuoc_tinh giữ nguyên, để vector cũ và vector mới dùng cùng tham số chuẩn hóa;
    giá trị của ảnh mới vượt ngoài [min, max] thì x' được kẹp về [0, 1] (xem search.normalize).
    Muốn cập nhật min, max thì phải tính lại trên toàn CSDL (thao tác bảo trì, không làm tự động).
    """
    kq = preprocess.process_one(path)
    if not kq["dat"]:
        return None, kq["ly_do"]
    v = F.extract(kq["rgb"], kq["gray"], kq["pts"])

    con = ket_noi(db_path)
    so_cuoi = con.execute("SELECT MAX(CAST(SUBSTR(ma_anh, 5) AS INTEGER)) FROM anh").fetchone()[0] or 0
    ma = f"man_{so_cuoi + 1:04d}"
    duong_dan = f"data/raw256/{ma}.jpg"
    cot = ", ".join(["ma_anh"] + F.FEATURE_NAMES)
    dau_hoi = ", ".join(["?"] * (1 + len(F.FEATURE_NAMES)))
    try:
        with con:
            con.execute("INSERT INTO anh VALUES (?, ?, ?, ?, ?, ?)",
                        (ma, duong_dan, C.SIZE, C.SIZE, "JPG", datetime.date.today().isoformat()))
            con.execute(f"INSERT INTO dac_trung ({cot}) VALUES ({dau_hoi})",
                        [ma] + [float(f"{x:.6f}") for x in v])      # cùng độ chính xác với features.csv
            shutil.copyfile(path, os.path.join(C.RAW_DIR, ma + ".jpg"))
            cv2.imwrite(os.path.join(C.ALIGNED_DIR, ma + ".jpg"), kq["rgb"][:, :, ::-1])
            with open(C.LANDMARK_CSV, "a", newline="") as f:
                csv.writer(f).writerow([ma] + [f"{x:.3f}" for x in kq["pts"].ravel()])
    finally:
        con.close()
    return ma, "da nap"


if __name__ == "__main__":
    tao_csdl()
    kiem_tra()
