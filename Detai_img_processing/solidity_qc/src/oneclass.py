"""Phuong phap co so (baseline): diem max-z cua cac dac trung toan cuc, hieu chuan tu anh OK."""
import cv2
import numpy as np
from . import pipeline as P


def _z(v, cal, direction):
    ref = v[cal]
    med = np.median(ref)
    scale = max(1.4826 * np.median(np.abs(ref - med)), 0.5 * ref.std(), 1e-3 * abs(med), 1e-9)
    return direction * (v - med) / scale


def maxz_scores(F, H, cal):
    """F: dict ten -> mang (n,) cho P.BASE_FEATS; H: histogram (n,144); cal: chi so anh OK dung hieu chuan."""
    Z = [_z(F[f], cal, P.DIRECTION[f]) for f in P.BASE_FEATS]
    s, calset = H[cal].sum(0), set(cal.tolist())
    hd = np.zeros(len(H))
    for i in range(len(H)):
        ref = (s - H[i]) / (len(cal) - 1) if i in calset else s / len(cal)
        hd[i] = cv2.compareHist(H[i], ref.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
    Z.append(_z(hd, cal, 1))
    return np.max(Z, axis=0)
