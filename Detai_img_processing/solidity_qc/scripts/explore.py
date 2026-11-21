"""BUOC 1: nhin bang mat xem phan vung co dung khong.
Cot: anh goc | mask Otsu | contour (xanh/do) + hull (xanh duong) + vung do be mat (vang) | canh Canny
Chay:  python scripts/explore.py hazelnut        (hoac biscuits, pills)   [so anh moi lop, mac dinh 4]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from src.config import DATASETS, OUT_DIR
from src.dataset import list_images
from src import pipeline as P

name = sys.argv[1] if len(sys.argv) > 1 else 'biscuits'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
cfg = DATASETS[name]
items = list_images(name)
pick = [i for i in items if i[1] == 'ok'][:N] + [i for i in items if i[1] == 'defect'][:N]

H = 240
rows = []
for path, label in pick:
    img = P.load(path, cfg['crop'])
    r = P.extract(img, cfg)
    if r is None:
        print('[CANH BAO] phan vung that bai:', path.name)
        continue
    mask, _ = P.segment(P.to_channel(img, cfg['channel']), cfg['object_bright'])
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(g, 50, 120)
    edges[r['interior'] == 0] = 0
    vis = P.draw(img, r, 'S=%.3f' % r['solidity'], (0, 0, 255) if label == 'defect' else (0, 200, 0))
    cols = [img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), vis, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)]
    cols = [cv2.resize(c, (int(c.shape[1] * H / c.shape[0]), H)) for c in cols]
    row = np.hstack(cols)
    cv2.putText(row, '%s | %s | otsu=%d' % (label, path.name, r['otsu']), (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    rows.append(row)
    print('%-7s %-26s solidity=%.4f circ=%.3f edge=%.4f sobel=%.1f std=%.1f' % (
        label, path.name, r['solidity'], r['circularity'], r['edge_density'], r['sobel_mean'], r['gray_std']))

W = max(r.shape[1] for r in rows)
rows = [cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
out = OUT_DIR / 'figures' / ('explore_%s.png' % name)
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), np.vstack(rows))
print('Da luu:', out)
