"""Phan tich LOI cho bao cao: ghep anh anh loi BI SOT (FN), anh dat BI BAO NHAM (FP) va anh loi bat dung (TP).
Doc benchmark_<bo>_scores.csv + anh chu thich trong outputs/annotated_v2/<bo>/ (chay benchmark.py truoc).
Chay:  python scripts/report_figs.py hazelnut   [--method patch_knn --fpr 0.10 --n 6]"""
import sys, csv, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from src.config import OUT_DIR

ap = argparse.ArgumentParser()
ap.add_argument('dataset')
ap.add_argument('--method', default=None, help='mac dinh: phuong phap tot nhat luc chay benchmark')
ap.add_argument('--fpr', type=float, default=0.10)
ap.add_argument('--n', type=int, default=6)
a = ap.parse_args()
name = a.dataset
meta = json.loads((OUT_DIR / ('benchmark_%s.json' % name)).read_text())
method = a.method or meta.get('annot_method')
rows = list(csv.DictReader(open(OUT_DIR / ('benchmark_%s_scores.csv' % name), encoding='utf-8')))
files = [r['file'] for r in rows]
lab = np.array([1 if r['label'] == 'defect' else 0 for r in rows])
sc = np.array([float(r[method]) for r in rows])
thr = float(np.quantile(sc[lab == 0], 1 - a.fpr))
pred = sc > thr
tp, fn = np.where(pred & (lab == 1))[0], np.where(~pred & (lab == 1))[0]
fp, tn = np.where(pred & (lab == 0))[0], np.where(~pred & (lab == 0))[0]
print('%s | phuong phap=%s | nguong cho FPR=%.0f%%: %.4f' % (name, method, a.fpr * 100, thr))
print('TP=%d FN=%d FP=%d TN=%d | recall=%.2f precision=%.2f' % (
    len(tp), len(fn), len(fp), len(tn), len(tp) / max(len(tp) + len(fn), 1), len(tp) / max(len(tp) + len(fp), 1)))
print('Anh loi BI SOT (FN):', ', '.join(files[i] for i in fn))
print('Anh dat BI BAO NHAM (FP):', ', '.join(files[i] for i in fp))

H = 230


def tile(i, tag):
    img = cv2.imread(str(OUT_DIR / 'annotated_v2' / name / files[i]))
    img = cv2.resize(img, (int(img.shape[1] * H / img.shape[0]), H))
    cap = np.full((26, img.shape[1], 3), 30, np.uint8)
    cv2.putText(cap, '%s %s s=%.2f' % (tag, files[i][:14], sc[i]), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return np.vstack([img, cap])


def row(idx, tag, title):
    tiles = [tile(i, tag) for i in idx[:a.n]]
    banner = np.full((28, 10, 3), (60, 60, 160), np.uint8)
    if not tiles:
        tiles = [np.full((H + 26, 260, 3), 30, np.uint8)]
        cv2.putText(tiles[0], '(khong co)', (80, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    r = np.hstack(tiles)
    b = np.full((28, r.shape[1], 3), (70, 70, 70), np.uint8)
    cv2.putText(b, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return np.vstack([b, r])


blocks = [row(fn[np.argsort(sc[fn])], 'FN', 'ANH LOI BI SOT (diem thap nhat truoc)'),
          row(fp[np.argsort(-sc[fp])], 'FP', 'ANH DAT BI BAO NHAM (diem cao nhat truoc)'),
          row(tp[np.argsort(-sc[tp])], 'TP', 'ANH LOI BAT DUNG (diem cao nhat truoc)')]
W = max(b.shape[1] for b in blocks)
blocks = [cv2.copyMakeBorder(b, 0, 0, 0, W - b.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)) for b in blocks]
out = OUT_DIR / 'figures' / ('errors_%s.png' % name)
cv2.imwrite(str(out), np.vstack(blocks))
print('Da luu:', out)
