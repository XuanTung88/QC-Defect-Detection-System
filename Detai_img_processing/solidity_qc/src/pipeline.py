"""Tien xu ly -> phan vung -> contour/convex hull -> dac trung hinh dang + be mat."""
import cv2
import numpy as np
from .config import BLUR_KSIZE, MIN_AREA_RATIO, ERODE_RATIO

# Dac trung; DIRECTION = +1 nghia la GIA TRI CAO la bat thuong, -1 nghia la GIA TRI THAP la bat thuong
SHAPE_FEATS = ['solidity', 'circularity']
SURFACE_FEATS = ['edge_density', 'sobel_mean', 'gray_std', 'hist_dist']
BASE_FEATS = ['solidity', 'circularity', 'edge_density', 'sobel_mean', 'gray_std']  # hist_dist tinh sau (can mau chuan)
ALL_FEATS = SHAPE_FEATS + SURFACE_FEATS
DIRECTION = dict(solidity=-1, circularity=-1, edge_density=1, sobel_mean=1, gray_std=1, hist_dist=1)


def load(path, crop=0):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)   # PNG 4 kenh -> tu bo alpha
    if img is None:
        raise FileNotFoundError(path)
    if crop:
        img = img[crop:-crop, crop:-crop]
    return img


def to_channel(bgr, channel='gray'):                # Ch1: he mau
    if channel == 'sat':
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1]
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def segment(ch, object_bright=True, ksize=BLUR_KSIZE):   # Ch2 (loc) + Ch4 (Otsu)
    ch = cv2.medianBlur(ch, ksize)
    flag = cv2.THRESH_BINARY if object_bright else cv2.THRESH_BINARY_INV
    t, mask = cv2.threshold(ch, 0, 255, flag | cv2.THRESH_OTSU)
    return mask, t


def measure(mask, min_area_ratio=MIN_AREA_RATIO):   # Ch5: contour + convex hull
    """Lay contour NGOAI LON NHAT (moi anh 1 vat). Tra ve dict hoac None."""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area_ratio * mask.size:
        return None
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    perim = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    return dict(contour=c, hull=hull, area=area, hull_area=hull_area, perimeter=perim,
                bbox=(x, y, w, h),
                solidity=area / hull_area if hull_area > 0 else 0.0,
                circularity=4 * np.pi * area / perim ** 2 if perim > 0 else 0.0,
                fill_ratio=area / mask.size)        # ~1.0 => phan vung hong (ca anh la vat)


def extract(bgr, cfg):
    """Tra ve dict dac trung (kem contour/hull/interior de ve), hoac None neu phan vung that bai."""
    mask, otsu = segment(to_channel(bgr, cfg['channel']), cfg['object_bright'])
    m = measure(mask)
    if m is None or m['fill_ratio'] > 0.9:
        return None
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, [m['contour']], -1, 255, -1)
    body = filled if cfg['interior'] == 'filled' else cv2.bitwise_and(mask, filled)
    k = max(3, int(ERODE_RATIO * np.sqrt(m['area'])))
    interior = cv2.erode(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    sel = interior > 0
    if sel.sum() < 100:
        return None

    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)   # Ch2: loc nhieu
    edges = cv2.Canny(gray, 50, 120)                                           # Ch4: Canny
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)                            # Ch4: Sobel
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], interior, [18, 8], [0, 180, 0, 256])    # Ch3: histogram
    hist = cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1).flatten().astype(np.float32)

    m.update(otsu=otsu, interior=interior, hist=hist,
             edge_density=float((edges[sel] > 0).mean()),
             sobel_mean=float(mag[sel].mean()),
             gray_std=float(gray[sel].astype(np.float64).std()))
    return m


def draw(bgr, m, text='', color=(0, 200, 0)):
    out = bgr.copy()
    if m is None:
        return out
    cv2.drawContours(out, [m['hull']], -1, (255, 200, 0), 2)       # hull: xanh duong nhat
    cv2.drawContours(out, [m['contour']], -1, color, 2)            # contour: do/xanh la
    ic, _ = cv2.findContours(m['interior'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, ic, -1, (0, 255, 255), 1)                # vung do dac trung be mat: vang
    x, y, w, h = m['bbox']
    cv2.putText(out, text, (max(x, 2), max(y - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out
