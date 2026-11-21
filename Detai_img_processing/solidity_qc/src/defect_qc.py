"""
defect_qc.py — Module RIÊNG, ĐỘC LẬP phát hiện khuyết tật (sứt mẻ, vỡ)
trên BẤT KỲ vật thể có hình dạng khép kín (tròn, oval, đa giác, v.v.).

MỤC TIÊU: Chỉ phát hiện khuyết tật STRUCTURAL (mẻ, vỡ, nứt, thiếu mảnh, lỗ hổng bất thường).
KHÔNG kiểm soát hình dạng chuẩn, kích thước, màu sắc.

HOẠT ĐỘNG ĐỘC LẬP hoàn toàn:
  - KHÔNG import gì từ pipeline.py, patches.py, oneclass.py gốc
  - KHÔNG ảnh hưởng đến 3 bộ dataset chuẩn (biscuits, hazelnut, pills)
  - Có thể xử lý: vòng đệm, đai ốc, nắp chai, bánh răng, ống sứ, v.v.

Thuật toán (3 bước):
  B1. Phân vùng nền/vật (Otsu) → tách hoàn toàn vật thể
  B2. Trích đặc trưng khuyết tật (5 chỉ số chuyên biệt cho sứt mẻ)
  B3. Vẽ trực quan hóa (overlay contour ngoài + contour trong nếu có lỗ)
"""

import cv2
import numpy as np

# ─────────────────────────────────────────────────
# CÁC THAM SỐ MẶC ĐỊNH
# ─────────────────────────────────────────────────
_BLUR_K = 5           # median blur trước Otsu
_MIN_AREA_RATIO = 0.01  # contour nhỏ hơn 1% diện tích ảnh → bỏ qua (nhiễu)
_MORPH_CLOSE_K = 7    # kernel morphological closing để liền vết nứt nhỏ

# Tên các đặc trưng & hướng bất thường (1 = cao là xấu, -1 = thấp là xấu)
DEFECT_FEATS = [
    'solidity',          # Bao lồi: thấp → vật bị khuyết / mẻ
    'convexity',         # Tỉ lệ chu vi contour/hull: thấp → viền không đều, có vết lõm
    'extent',            # Tỉ lệ diện tích vật / bounding rect: thấp → bị thiếu góc
    'hole_ratio',        # Tỉ lệ diện tích lỗ bên trong / diện tích vật: cao → vỡ/thủng bất thường
    'boundary_roughness' # Độ gồ ghề biên: cao → viền bị vỡ, nứt
]
DEFECT_DIRECTION = {
    'solidity': -1,
    'convexity': -1,
    'extent': -1,
    'hole_ratio': 1,
    'boundary_roughness': 1
}


# ─────────────────────────────────────────────────
# B1: PHÂN VÙNG (tách vật khỏi nền)
# ─────────────────────────────────────────────────

def segment(bgr: np.ndarray, channel: str = 'gray', object_bright: bool = True) -> np.ndarray:
    """
    Trả về mask nhị phân (255 = vật thể, 0 = nền).
    channel: 'gray' | 'sat' | 'val'
    """
    if channel == 'sat':
        ch = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1]
    elif channel == 'val':
        ch = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2]
    else:
        ch = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(ch, _BLUR_K)
    flag = cv2.THRESH_BINARY if object_bright else cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(blurred, 0, 255, flag | cv2.THRESH_OTSU)

    # Morphological closing để bít các lỗ nhỏ do bóng đèn/ánh sáng
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_MORPH_CLOSE_K, _MORPH_CLOSE_K))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask_closed, mask  # (mask đã close, mask Otsu thuần)


def get_outer_contour(mask: np.ndarray):
    """Lấy contour NGOÀI lớn nhất (thân chính của vật thể)."""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < _MIN_AREA_RATIO * mask.size:
        return None
    return c


def get_all_contours_hierarchical(mask: np.ndarray):
    """
    Lấy TẤT CẢ contour kèm phân cấp cha-con.
    Trả về: (contours, hierarchy) — cần để phân biệt lỗ trong.
    """
    cs, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    return cs, hier


# ─────────────────────────────────────────────────
# B2: TRÍCH ĐẶC TRƯNG KHUYẾT TẬT
# ─────────────────────────────────────────────────

def extract(bgr: np.ndarray,
            channel: str = 'gray',
            object_bright: bool = True) -> dict | None:
    """
    Hàm chính: trích xuất 5 đặc trưng phát hiện sứt mẻ / vỡ / nứt.
    Trả về dict đặc trưng hoặc None nếu không phân vùng được.

    Các đặc trưng:
    ┌──────────────────────┬─────────────────────────────────────────────────────┐
    │ solidity             │ Area / HullArea — thấp → mẻ / khuyết lớn           │
    │ convexity            │ Perimeter(hull) / Perimeter(contour) — thấp → lõm  │
    │ extent               │ Area / BBoxArea — thấp → góc bị cắt / thiếu mảnh  │
    │ hole_ratio           │ Σ(hole_areas) / area_outer — cao → thủng bất thường│
    │ boundary_roughness   │ std(r_i - mean_r) / mean_r — cao → biên gồ ghề vỡ │
    └──────────────────────┴─────────────────────────────────────────────────────┘
    """
    h_img, w_img = bgr.shape[:2]

    # --- Phân vùng ---
    mask_closed, mask_raw = segment(bgr, channel, object_bright)
    c_outer = get_outer_contour(mask_closed)
    if c_outer is None:
        return None

    area = cv2.contourArea(c_outer)
    if area < _MIN_AREA_RATIO * (h_img * w_img):
        return None

    # --- Convex Hull ---
    hull = cv2.convexHull(c_outer)
    hull_area = cv2.contourArea(hull)
    solidity = float(area / hull_area) if hull_area > 0 else 0.0

    # --- Chu vi ---
    perim_contour = cv2.arcLength(c_outer, True)
    perim_hull = cv2.arcLength(hull, True)
    convexity = float(perim_hull / perim_contour) if perim_contour > 0 else 0.0
    convexity = min(convexity, 1.0)   # không bao giờ > 1

    # --- Extent (tỉ lệ diện tích / bbox) ---
    x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(c_outer)
    bbox_area = float(w_bb * h_bb)
    extent = float(area / bbox_area) if bbox_area > 0 else 0.0

    # --- Hole ratio: lỗ bên trong ---
    cs_all, hier_all = get_all_contours_hierarchical(mask_closed)
    hole_area_total = 0.0
    hole_contours = []
    if hier_all is not None and len(cs_all) > 0:
        for i, (cx_i, cy_i, cw_i, ch_i) in enumerate(hier_all[0]):
            # Contour có cha (parent != -1) → đây là lỗ bên trong
            parent = hier_all[0][i][3]
            if parent != -1:
                h_a = cv2.contourArea(cs_all[i])
                if h_a > 50:   # bỏ qua lỗ siêu nhỏ (nhiễu)
                    hole_area_total += h_a
                    hole_contours.append(cs_all[i])
    hole_ratio = float(hole_area_total / area) if area > 0 else 0.0

    # --- Boundary roughness: độ gồ ghề biên ---
    pts = c_outer.reshape(-1, 2).astype(np.float64)
    # Tìm tâm (hình học)
    cx_geo = float(pts[:, 0].mean())
    cy_geo = float(pts[:, 1].mean())
    radii = np.sqrt((pts[:, 0] - cx_geo) ** 2 + (pts[:, 1] - cy_geo) ** 2)
    r_mean = float(radii.mean())
    r_std = float(radii.std())
    boundary_roughness = float(r_std / (r_mean + 1e-6))

    return dict(
        # --- đặc trưng ---
        solidity=solidity,
        convexity=convexity,
        extent=extent,
        hole_ratio=hole_ratio,
        boundary_roughness=boundary_roughness,
        # --- hình học để vẽ ---
        contour=c_outer,
        hull=hull,
        hole_contours=hole_contours,
        # --- meta ---
        area=float(area),
        hull_area=float(hull_area),
        bbox=(x_bb, y_bb, w_bb, h_bb),
        center=(cx_geo, cy_geo),
        radius_mean=r_mean,
        otsu_mask=mask_raw,
        fill_ratio=float(area / (h_img * w_img))
    )


# ─────────────────────────────────────────────────
# B3: TRỰC QUAN HÓA
# ─────────────────────────────────────────────────

def draw(bgr: np.ndarray, m: dict, text: str = '', color=(0, 200, 0)) -> np.ndarray:
    """
    Vẽ overlay lên ảnh gốc:
      - Convex Hull  → vàng (vùng bao lồi lý tưởng)
      - Contour ngoài → xanh lá (OK) / đỏ (DEFECT)
      - Contour lỗ bên trong → xanh cyan
      - Vùng khuyết giữa hull và contour → tô màu hồng nhạt (highlight defect zone)
    """
    out = bgr.copy()
    if m is None:
        return out

    # Vùng khuyết: vẽ hull fill màu hồng mờ, sau đó fill contour màu trắng → chỉ thấy vùng thiếu
    overlay = out.copy()
    cv2.drawContours(overlay, [m['hull']], -1, (180, 100, 255), -1)
    cv2.drawContours(overlay, [m['contour']], -1, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)

    # Convex Hull: vàng
    cv2.drawContours(out, [m['hull']], -1, (0, 220, 255), 2)

    # Contour ngoài: xanh lá / đỏ tuỳ nhãn
    cv2.drawContours(out, [m['contour']], -1, color, 2)

    # Lỗ bên trong: cyan
    if m.get('hole_contours'):
        cv2.drawContours(out, m['hole_contours'], -1, (255, 230, 0), 2)

    # Text label
    x, y, w, h = m['bbox']
    cv2.putText(out, text, (max(x, 2), max(y - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────
# ĐÁNH GIÁ MỘT LỚP (One-class Z-score)
# Hệ hiệu chuẩn từ ảnh OK, chấm điểm ảnh mới
# ─────────────────────────────────────────────────

def robust_z(values: np.ndarray, cal_idx: np.ndarray, direction: float) -> np.ndarray:
    """Robust z-score (giống oneclass.py gốc nhưng self-contained, không import nó)."""
    ref = values[cal_idx]
    med = float(np.median(ref))
    mad = float(np.median(np.abs(ref - med)))
    scale = max(1.4826 * mad, 0.5 * ref.std(), 1e-3 * abs(med), 1e-9)
    return direction * (values - med) / scale


def maxz_scores_defect(feats_dict: dict, cal_idx: np.ndarray) -> np.ndarray:
    """
    feats_dict: { feat_name: np.array(n,) }
    cal_idx: chỉ số các ảnh OK dùng để hiệu chuẩn
    Trả về: mảng (n,) điểm bất thường (max Z qua các đặc trưng)
    """
    Z_list = []
    for f in DEFECT_FEATS:
        v = feats_dict[f]
        d = DEFECT_DIRECTION[f]
        Z_list.append(robust_z(v, cal_idx, d))
    return np.max(Z_list, axis=0)


def calibrate(ok_feats: list[dict]) -> dict:
    """
    Hiệu chuẩn từ danh sách dict đặc trưng của ảnh OK.
    Trả về dict calibration có thể dùng để score ảnh mới.
    """
    cal = {}
    for f in DEFECT_FEATS:
        vals = np.array([r[f] for r in ok_feats], dtype=np.float64)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        scale = max(1.4826 * mad, 0.5 * vals.std(), 1e-3 * abs(med), 1e-9)
        cal[f] = dict(median=med, scale=scale, direction=DEFECT_DIRECTION[f])
    return cal


def score_one(feats: dict, cal: dict) -> float:
    """Chấm điểm 1 ảnh từ calibration đã tính sẵn. Trả về điểm max-Z."""
    zs = []
    for f in DEFECT_FEATS:
        c = cal[f]
        z = c['direction'] * (feats[f] - c['median']) / c['scale']
        zs.append(z)
    return float(max(zs))
