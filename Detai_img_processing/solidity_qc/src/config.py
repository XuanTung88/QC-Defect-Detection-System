"""Cau hinh chung.
Cau truc du lieu mong doi:  XuLyAnh/data/<ten_bo>/ok/*.png|jpg   va   .../defect/*.png|jpg"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]        # .../solidity_qc
DATA_ROOT = Path(__file__).resolve().parents[3] / 'data'  # .../XuLyAnh/data
OUT_DIR = PROJECT_ROOT / 'outputs'

# Moi bo du lieu co cach tach vat rieng (vi nen/vat khac nhau)
#   channel       : 'gray' (anh xam) hoac 'sat' (kenh Saturation cua HSV)
#   object_bright : True neu vat SANG hon nen o kenh da chon
#   crop          : so pixel cat bo moi canh (neu anh co vien)
#   interior      : 'mask'   -> vung than vat = mask Otsu (giu lo that cua vat, vd banh co lo o giua)
#                   'filled' -> vung than vat = contour da to dac (bo qua cac cham toi nho ben trong)
EXTS = ('.png', '.jpg', '.jpeg')
DATASETS = {
    'biscuits': dict(exts=EXTS, channel='gray', object_bright=True, crop=0, interior='mask',
                     role='Chinh: loi hinh dang (me, sut)'),
    'hazelnut': dict(exts=EXTS, channel='sat',  object_bright=True, crop=0, interior='filled',
                     role='Chinh: loi be mat (nut, lo, cat) + hinh dang'),
    # pills: nen toi, vien thuoc trang (loai chu "FF"). Neu thu muc pills van la vien thuoc XANH
    # (nen be, co vien xanh trong anh) thi doi thanh channel='sat', crop=6.
    'pills':    dict(exts=EXTS, channel='gray', object_bright=True, crop=0, interior='filled',
                     role='Doi chung: loi tinh te, ky thuat co dien kho bat'),
}

OK_NAMES = ('ok', 'good')
DEFECT_NAMES = ('defect', 'def')
BLUR_KSIZE = 5          # Median blur (Ch2)
MIN_AREA_RATIO = 0.005  # bo contour nho hon 0.5% dien tich anh (nhieu)
ERODE_RATIO = 0.08      # thu nho vung than vat de tranh vien: k = 0.08 * sqrt(dien tich)
