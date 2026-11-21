"""Dac trung CUC BO theo o (patch) + ngan hang o OK (y tuong PatchCore, nhung dung dac trung thu cong).
Moi o ~ 24x24 px (sau khi chuan hoa kich thuoc vat) duoc mo ta bang 12 so BAT BIEN KHI XOAY
(mau Lab, do manh bien, do co dap huong...) nen hop voi vat nam o goc bat ky (vd qua phi).
Diem bat thuong cua mot o = khoang cach den cac o OK gan nhat trong ngan hang."""
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

NORM_SIZE = 200.0   # chuan hoa: sqrt(dien tich vat) = 200 px de o co kich thuoc tuong doi on dinh
PATCH = 24
STRIDE = 12
MIN_INSIDE = 0.95   # o phai nam >= 95% trong vung than vat

DESC_NAMES = ['L_mean', 'a_mean', 'b_mean', 'L_std', 'a_std', 'b_std',
              'grad_mean', 'lap_mean', 'edge_density', 'coherence', 'min_rel', 'max_rel']
PATCH_FEATS = ['p_max', 'p_top5', 'p_p90', 'p_mean']


def _box(x, p=PATCH):
    return cv2.boxFilter(x, -1, (p, p), normalize=True, borderType=cv2.BORDER_REFLECT)


def patch_descriptors(bgr, r, patch=PATCH):
    """r = ket qua pipeline.extract(). Tra ve dict(desc, xy, scale) hoac None. patch = kich thuoc o (px)."""
    stride = max(patch // 2, 1)
    s = NORM_SIZE / np.sqrt(r['area'])
    img = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    inter = cv2.resize(r['interior'], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    if (inter > 0).sum() < patch * patch:
        return None

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    g8 = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    g = g8.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
    edges = (cv2.Canny(g8, 50, 120) > 0).astype(np.float32)
    med = float(np.median(g[inter > 0]))

    _b = lambda x: _box(x, patch)
    chans = []
    for c in range(3):                                   # mau Lab: trung binh + do lech chuan
        ch = lab[..., c]
        chans.append(_b(ch))
    for c in range(3):
        ch = lab[..., c]
        chans.append(np.sqrt(np.maximum(_b(ch * ch) - _b(ch) ** 2, 0)))
    chans.append(np.log1p(_b(mag)))                    # do manh bien
    chans.append(np.log1p(_b(lap)))                    # Laplace
    chans.append(_b(edges))                            # mat do Canny
    jxx, jyy, jxy = _b(gx * gx), _b(gy * gy), _b(gx * gy)   # tensor cau truc -> do co dap huong
    disc = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    chans.append(disc / (jxx + jyy + 1e-3))              # coherence, khong doi khi xoay
    k = np.ones((patch, patch), np.uint8)
    chans.append(cv2.erode(g, k) - med)                  # vung toi nhat trong o so voi vat
    chans.append(cv2.dilate(g, k) - med)                 # vung sang nhat trong o so voi vat
    M = np.stack(chans, -1)

    inside = _b((inter > 0).astype(np.float32))
    h, w = inter.shape
    ys, xs = np.meshgrid(np.arange(patch // 2, h - patch // 2, stride),
                         np.arange(patch // 2, w - patch // 2, stride), indexing='ij')
    ys, xs = ys.ravel(), xs.ravel()
    keep = inside[ys, xs] >= MIN_INSIDE
    ys, xs = ys[keep], xs[keep]
    if len(ys) < 5:
        return None
    return dict(desc=M[ys, xs].astype(np.float32), xy=np.stack([xs, ys], 1), scale=s)


class PatchBank:
    """Ngan hang cac o cua anh OK. score(desc) -> diem bat thuong cho tung o."""
    def __init__(self, k=3, max_bank=40000, seed=0):
        self.k, self.max_bank, self.seed = k, max_bank, seed

    def fit(self, descs):
        X = np.vstack(descs)
        if len(X) > self.max_bank:
            X = X[np.random.default_rng(self.seed).choice(len(X), self.max_bank, replace=False)]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(X))).fit((X - self.mu) / self.sd)
        return self

    def score(self, desc):
        d, _ = self.nn.kneighbors((desc - self.mu) / self.sd)
        return d.mean(1)


def summarize(sc):
    s = np.sort(sc)[::-1]
    return [s[0], s[:5].mean(), np.percentile(sc, 90), sc.mean()]   # theo thu tu PATCH_FEATS
