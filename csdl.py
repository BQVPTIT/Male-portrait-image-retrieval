"""
csdl.py - CSDL quan hệ (SQLite) quản trị đặc trưng của ảnh.

Hai bảng:
    anh        ma_anh (PK), duong_dan (UNIQUE), rong, cao, dinh_dang, ngay_nap
               -> đặc trưng tổng quát của ảnh; ảnh gốc lưu ngoài CSDL, bảng chỉ giữ đường dẫn
    dac_trung  ma_anh (PK, đồng thời FK -> anh) + 39 cột REAL đặt tên theo trich_rut.TEN_COT
               -> đặc trưng nội dung, mỗi hàng là một vector 39 chiều, lưu GIÁ TRỊ GỐC

Quan hệ: anh 1 - 1 dac_trung qua ma_anh (khóa chính của dac_trung đồng thời là khóa ngoại).

Tham số dùng khi tìm kiếm, không phải dữ liệu của từng ảnh nên không lưu thành bảng:
    min, max của 39 cột  -> data/min_max.json (tính một lần trên toàn CSDL, giữ cố định để
                            chuẩn hóa ảnh truy vấn; kiem_tra đối chiếu với SELECT MIN(), MAX())
    trọng số 3 bộ (1/3)  -> hằng số tim_kiem.TRONG_SO
    tên, vai trò X0-X38  -> hằng số trich_rut.TEN_DAC_TRUNG, trich_rut.VAI_TRO

SQLite: cả CSDL là một tệp data/khuon_mat.db, đọc ghi bằng thư viện chuẩn sqlite3 của Python,
không cần máy chủ. Câu lệnh SQL (CREATE TABLE, INSERT, SELECT, khóa chính, khóa ngoại) giống MySQL.

Chạy:  python csdl.py      (tạo lại CSDL từ data/dac_trung.csv + data/min_max.json rồi kiểm tra)
"""
import os
import csv
import json
import shutil
import sqlite3
import datetime
import numpy as np
import cv2

import xu_ly_anh as XL
import tien_xu_ly as TXL
import trich_rut as TR
# ---------------------------------------------------------------- lệnh tạo bảng
COT_DAC_TRUNG = ",\n".join(f"    {ten:<13} REAL NOT NULL" for ten in TR.TEN_COT)

DDL = f"""
CREATE TABLE anh (
    ma_anh     TEXT PRIMARY KEY,                   -- man_0001
    duong_dan  TEXT NOT NULL UNIQUE,               -- data/anh256/man_0001.jpg
    rong       INTEGER NOT NULL CHECK (rong > 0),  -- 256
    cao        INTEGER NOT NULL CHECK (cao > 0),   -- 256
    dinh_dang  TEXT NOT NULL CHECK (dinh_dang = 'JPG'),
    ngay_nap   TEXT NOT NULL                       -- ngày nạp vào CSDL, dạng YYYY-MM-DD
);

CREATE TABLE dac_trung (
    ma_anh        TEXT PRIMARY KEY REFERENCES anh(ma_anh),
{COT_DAC_TRUNG}
);
"""


# ---------------------------------------------------------------- kết nối
def ket_noi(tep_csdl=XL.TEP_CSDL):
    """
    - File: csdl.py. Dùng để: mở kết nối tới tệp CSDL SQLite.
    - Dựa trên: thư viện chuẩn sqlite3 của Python
      + cả CSDL là một tệp data/khuon_mat.db, không cần máy chủ
      + SQLite mặc định KHÔNG kiểm tra khóa ngoại, nên mỗi lần mở phải chạy PRAGMA foreign_keys = ON
    - Ứng dụng: mọi thao tác đọc, ghi CSDL đều đi qua hàm này, bảo đảm ràng buộc khóa ngoại luôn bật.
    - Cách khác: MySQL, PostgreSQL (cần cài và chạy máy chủ).
    """
    con = sqlite3.connect(tep_csdl)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def duong_dan_day_du(duong_dan):
    """
    - File: csdl.py. Dùng để: đổi đường dẫn lưu trong bảng anh thành đường dẫn đầy đủ.
    - Dựa trên: bảng anh lưu đường dẫn TƯƠNG ĐỐI, ví dụ data/anh256/man_0001.jpg
      + ghép với thư mục dự án (xu_ly_anh.THU_MUC_GOC)
      + tách theo "/" để chạy đúng trên cả Windows lẫn Linux
    - Ứng dụng: giao diện mở đúng tệp ảnh dù dự án được chép sang máy khác, ổ khác.
    - Cách khác: lưu đường dẫn tuyệt đối (hỏng khi chuyển máy).
    """
    return os.path.join(XL.THU_MUC_GOC, *duong_dan.split("/"))


# ---------------------------------------------------------------- tạo và nạp CSDL
def doc_tep_dac_trung():
    """
    - File: csdl.py. Dùng để: đọc kết quả trích rút từ data/dac_trung.csv.
    - Dựa trên: tệp CSV (đầu ra của trich_rut.chay)
      + dòng đầu là tiêu đề, phải đúng thứ tự TEN_COT, sai thì dừng (assert)
      + trả về (danh sách mã ảnh, ma trận N x 39 giá trị gốc)
    - Ứng dụng: nguồn dữ liệu để nạp bảng dac_trung khi tạo CSDL.
    - Cách khác: tính lại đặc trưng trực tiếp từ ảnh (mất khoảng 3,5 phút).
    """
    with open(XL.TEP_DAC_TRUNG, newline="") as f:
        ds_dong = list(csv.reader(f))
    assert ds_dong[0] == ["ma_anh"] + TR.TEN_COT, "dac_trung.csv khong dung thu tu TEN_COT"
    return [r[0] for r in ds_dong[1:]], np.array([r[1:] for r in ds_dong[1:]], dtype=np.float64)


def tao_csdl(tep_csdl=XL.TEP_CSDL):
    """
    - File: csdl.py. Dùng để: tạo lại CSDL 2 bảng và nạp dữ liệu.
    - Dựa trên: mô hình quan hệ, lệnh CREATE TABLE trong DDL
      + mỗi ảnh: INSERT bảng anh trước, rồi bảng dac_trung (vì dac_trung tham chiếu anh)
      + toàn bộ INSERT trong MỘT giao dịch: lỗi ở bất kỳ dòng nào thì không dòng nào được ghi
      + kiểm tra min_max.json đúng thứ tự cột, không nạp vào CSDL
    - Ứng dụng: luồng nạp CSDL, chạy một lần.
    - Cách khác: nạp từng ảnh riêng lẻ (dễ dở dang).
    """
    if os.path.exists(tep_csdl):
        os.remove(tep_csdl)
    ds_ma, X = doc_tep_dac_trung()
    with open(XL.TEP_MIN_MAX, encoding="utf-8") as f:
        p = json.load(f)
    assert p["ten"] == TR.TEN_COT, "min_max.json khong dung thu tu TEN_COT"

    con = ket_noi(tep_csdl)
    con.executescript(DDL)
    ngay = datetime.date.today().isoformat()
    cot = ", ".join(["ma_anh"] + TR.TEN_COT)
    dau_hoi = ", ".join(["?"] * (1 + len(TR.TEN_COT)))
    with con:
        for ma, v in zip(ds_ma, X):
            duong_dan = f"data/anh256/{ma}.jpg"
            rgb = TXL.doc_rgb(duong_dan_day_du(duong_dan))
            con.execute("INSERT INTO anh VALUES (?, ?, ?, ?, ?, ?)",
                        (ma, duong_dan, rgb.shape[1], rgb.shape[0], "JPG", ngay))
            con.execute(f"INSERT INTO dac_trung ({cot}) VALUES ({dau_hoi})", [ma] + v.tolist())
    con.close()
    print(f"Tao CSDL xong: {len(ds_ma)} anh -> {tep_csdl}")


# ---------------------------------------------------------------- đọc dữ liệu
# Các lệnh SELECT mà bước tìm kiếm dùng (giao_dien.py hiện lại đúng các lệnh này)
SQL_DAC_TRUNG = "SELECT ma_anh, " + ", ".join(TR.TEN_COT) + " FROM dac_trung ORDER BY ma_anh"
# min, max thật của từng cột ngay trong CSDL (dùng để đối chiếu với min_max.json)
SQL_MIN_MAX = "SELECT " + ", ".join(f"MIN({c}), MAX({c})" for c in TR.TEN_COT) + " FROM dac_trung"
SQL_ANH = "SELECT ma_anh, duong_dan, rong, cao, dinh_dang, ngay_nap FROM anh WHERE ma_anh = ?"


def doc_dac_trung(con):
    """
    - File: csdl.py. Dùng để: đọc vector đặc trưng của mọi ảnh.
    - Dựa trên: lệnh SQL_DAC_TRUNG
      + SELECT ma_anh và 39 cột FROM dac_trung ORDER BY ma_anh
      + kết quả đổi thành ma trận numpy để tính khoảng cách cho cả 1005 ảnh cùng lúc
    - Ứng dụng: bước so khớp của tìm kiếm (tim_kiem.doc_vector_csdl), phép thử đánh giá.
    - Cách khác: đọc từng ảnh bằng WHERE ma_anh = ? (1005 lần truy vấn, chậm hơn nhiều).
    """
    ds_dong = con.execute(SQL_DAC_TRUNG).fetchall()
    return [r[0] for r in ds_dong], np.array([r[1:] for r in ds_dong], dtype=np.float64)


def doc_min_max():
    """
    - File: csdl.py. Dùng để: đọc min, max đã lưu của 39 đặc trưng.
    - Dựa trên: tệp data/min_max.json (tim_kiem.tinh_min_max ghi ra)
      + kiểm tra tên cột đúng thứ tự TEN_COT
      + trả về hai mảng min, max
    - Ứng dụng: tham số chuẩn hóa min-max cho ảnh truy vấn và ảnh trong CSDL. Không lưu thành
      bảng vì đây là tham số của cả hệ thống, không phải dữ liệu của từng ảnh.
    - Cách khác: SELECT MIN(), MAX() mỗi lần (thang đổi khi thêm ảnh).
    """
    with open(XL.TEP_MIN_MAX, encoding="utf-8") as f:
        p = json.load(f)
    assert p["ten"] == TR.TEN_COT, "min_max.json khong dung thu tu TEN_COT"
    return np.array(p["min"], dtype=np.float64), np.array(p["max"], dtype=np.float64)


def doc_thuoc_tinh():
    """
    - File: csdl.py. Dùng để: lấy mô tả 39 đặc trưng cho giao diện.
    - Dựa trên: ghép hằng số và tệp min, max
      + ký hiệu X0 – X38, tên tiếng Việt (trich_rut.TEN_DAC_TRUNG), bộ thuộc tính (trich_rut.BO)
      + vai trò tìm tương đồng hay khác biệt (trich_rut.VAI_TRO, Bảng 2.1)
      + min, max từ min_max.json
    - Ứng dụng: giao diện ghi tên đặc trưng và chia phần giải thích "giống ở đâu".
    - Cách khác: bảng từ điển dữ liệu riêng (đã bỏ khi rút về 2 bảng).
    """
    mn, mx = doc_min_max()
    return [dict(ten_cot=ten, ky_hieu=f"X{j}", ten_dac_trung=TR.TEN_DAC_TRUNG[j],
                 ma_bo=next(b for b, vt in TR.BO.items() if vt.start <= j < vt.stop),
                 vai_tro=TR.VAI_TRO[j], gia_tri_min=mn[j], gia_tri_max=mx[j])
            for j, ten in enumerate(TR.TEN_COT)]


def doc_anh(con, ma_anh):
    """
    - File: csdl.py. Dùng để: đọc thông tin chung của một ảnh theo mã ảnh.
    - Dựa trên: lệnh SQL_ANH
      + SELECT ... FROM anh WHERE ma_anh = ? (dấu ? là tham số, chống lỗi chèn chuỗi SQL)
      + trả về dict gồm mã, đường dẫn, rộng, cao, định dạng, ngày nạp; không có thì None
    - Ứng dụng: giao diện lấy đường dẫn ảnh của 5 kết quả để hiển thị.
    - Cách khác: JOIN anh với dac_trung một lần cho cả 5 ảnh.
    """
    r = con.execute(SQL_ANH, (ma_anh,)).fetchone()
    return None if r is None else dict(zip(["ma_anh", "duong_dan", "rong", "cao", "dinh_dang", "ngay_nap"], r))


# ---------------------------------------------------------------- kiểm tra toàn vẹn
CAU_KIEM_TRA = [
    ("So anh", "SELECT COUNT(*) FROM anh"),
    ("So vector dac trung", "SELECT COUNT(*) FROM dac_trung"),
    ("Anh chua co vector", "SELECT COUNT(*) FROM anh a LEFT JOIN dac_trung d ON a.ma_anh = d.ma_anh "
                           "WHERE d.ma_anh IS NULL"),
    ("Vector khong thuoc anh nao", "SELECT COUNT(*) FROM dac_trung d LEFT JOIN anh a ON d.ma_anh = a.ma_anh "
                                   "WHERE a.ma_anh IS NULL"),
    ("Khoa ngoai cua dac_trung", "PRAGMA foreign_key_list(dac_trung)"),
    ("Vi pham khoa ngoai", "PRAGMA foreign_key_check"),
]


def kiem_tra(tep_csdl=XL.TEP_CSDL):
    """
    - File: csdl.py. Dùng để: kiểm tra toàn vẹn CSDL sau khi nạp (Bảng 3.6, Báo cáo 2).
    - Dựa trên: các câu lệnh SQL trong CAU_KIEM_TRA và 3 phép thử thêm
      + số ảnh bằng số vector, không có ảnh thiếu vector hay vector mồ côi (LEFT JOIN)
      + 39 cột đúng tên; min, max đã lưu bằng SELECT MIN(), MAX()
      + chèn vector của ảnh không có thật phải bị khóa ngoại từ chối
    - Ứng dụng: chứng minh dữ liệu đúng, in "DAT" hoặc "KHONG DAT".
    - Cách khác: kiểm tra bằng mắt.
    """
    con = ket_noi(tep_csdl)
    kq = {}
    for ten, sql in CAU_KIEM_TRA:
        kq[ten] = con.execute(sql).fetchall()
        print(f"{ten:28s}: {kq[ten]}")

    cot = [r[1] for r in con.execute("PRAGMA table_info(dac_trung)")][1:]
    print(f"{'39 cot dung TEN_COT':28s}: {cot == TR.TEN_COT}")

    mn, mx = doc_min_max()
    that = np.array(con.execute(SQL_MIN_MAX).fetchone()).reshape(-1, 2)
    min_max_dung = bool((that[:, 0] == mn).all() and (that[:, 1] == mx).all())
    print(f"{'min, max = MIN(), MAX()':28s}: {min_max_dung}")

    try:
        with con:
            con.execute("INSERT INTO dac_trung (ma_anh, " + ", ".join(TR.TEN_COT) + ") VALUES ("
                        + ", ".join(["'khong_co'"] + ["0"] * len(TR.TEN_COT)) + ")")
        tu_choi = False
    except sqlite3.IntegrityError as e:
        tu_choi = True
        print(f"{'Chen vector anh khong co':28s}: bi tu choi ({e})")
    con.close()

    dat = (kq["So anh"] == kq["So vector dac trung"] and kq["Anh chua co vector"] == [(0,)]
           and kq["Vector khong thuoc anh nao"] == [(0,)] and kq["Vi pham khoa ngoai"] == []
           and cot == TR.TEN_COT and min_max_dung and tu_choi)
    print("=> Toan ven:", "DAT" if dat else "KHONG DAT")
    return dat


# ---------------------------------------------------------------- nạp thêm ảnh
def them_anh(tep, tep_csdl=XL.TEP_CSDL):
    """
    - File: csdl.py. Dùng để: nạp thêm một ảnh mới mà không tạo lại CSDL.
    - Dựa trên: đúng đường đi của ảnh trong CSDL, trong một giao dịch
      + qua đủ 6 bước tiền xử lý, không đạt thì từ chối, không ghi gì
      + trích 39 đặc trưng, mã mới = số lớn nhất hiện có + 1
      + INSERT bảng anh, bảng dac_trung, lưu tệp; lỗi thì hủy toàn bộ (ROLLBACK)
    - Ứng dụng: mở rộng CSDL. min, max giữ cố định, giá trị vượt ngoài được kẹp về [0; 1].
    - Cách khác: tạo lại toàn bộ CSDL.
    """
    kq = TXL.xu_ly_mot_anh(tep)
    if not kq["dat"]:
        return None, kq["ly_do"]
    v = TR.trich_vector(kq["rgb"], kq["xam"], kq["moc"])

    con = ket_noi(tep_csdl)
    so_cuoi = con.execute("SELECT MAX(CAST(SUBSTR(ma_anh, 5) AS INTEGER)) FROM anh").fetchone()[0] or 0
    ma = f"man_{so_cuoi + 1:04d}"
    duong_dan = f"data/anh256/{ma}.jpg"
    cot = ", ".join(["ma_anh"] + TR.TEN_COT)
    dau_hoi = ", ".join(["?"] * (1 + len(TR.TEN_COT)))
    try:
        with con:
            con.execute("INSERT INTO anh VALUES (?, ?, ?, ?, ?, ?)",
                        (ma, duong_dan, XL.CANH_ANH, XL.CANH_ANH, "JPG", datetime.date.today().isoformat()))
            con.execute(f"INSERT INTO dac_trung ({cot}) VALUES ({dau_hoi})",
                        [ma] + [float(f"{x:.6f}") for x in v])      # cùng độ chính xác với dac_trung.csv
            shutil.copyfile(tep, os.path.join(XL.THU_MUC_ANH_GOC, ma + ".jpg"))
            cv2.imwrite(os.path.join(XL.THU_MUC_DA_CAN_CHINH, ma + ".jpg"), kq["rgb"][:, :, ::-1])
            with open(XL.TEP_DIEM_MOC, "a", newline="") as f:
                csv.writer(f).writerow([ma] + [f"{x:.3f}" for x in kq["moc"].ravel()])
    finally:
        con.close()
    return ma, "da nap"


if __name__ == "__main__":
    tao_csdl()
    kiem_tra()
