"""Phan tich DO NHAY tham so cua phuong phap theo o (patch_knn): kich thuoc o x so lang gieng k.
Day la phuong phap MOT LOP (khong dung anh loi de huan luyen) nen it co nguy co hoc vet. Tuy vay, neu CHON tham so tot nhat
tren chinh du lieu nay thi ket qua se lac quan -> trong bao cao hay trinh bay nhu mot bang do nhay, khong phai 'tinh chinh'.
Chay:  python scripts/sensitivity.py hazelnut   [--patches 16,24,32,48 --ks 1,3,5 --reps 5]"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.config import DATASETS, OUT_DIR
from src.dataset import list_images
from src import pipeline as P
from src import patches as PT

ap = argparse.ArgumentParser()
ap.add_argument('dataset', choices=list(DATASETS))
ap.add_argument('--patches', default='16,24,32,48')
ap.add_argument('--ks', default='1,3,5')
ap.add_argument('--splits', type=int, default=5)
ap.add_argument('--reps', type=int, default=5)
a = ap.parse_args()
name, cfg = a.dataset, DATASETS[a.dataset]
sizes = [int(v) for v in a.patches.split(',')]
ks = [int(v) for v in a.ks.split(',')]

recs = []
for path, label in list_images(name):
    img = P.load(path, cfg['crop'])
    r = P.extract(img, cfg)
    if r is None:
        continue
    ds = {s: PT.patch_descriptors(img, r, patch=s) for s in sizes}
    if any(v is None for v in ds.values()):
        print('[CANH BAO] bo qua (o qua lon / vat qua nho):', path.name)
        continue
    recs.append((label, ds))
y = np.array([1 if l == 'defect' else 0 for l, _ in recs])
n = len(y)
splits = min(a.splits, int(y.sum()), int((y == 0).sum()))
STATS = {'p_max': 0, 'p_top5': 1, 'p_p90': 2, 'p_mean': 3}
auc = {(s, k, st): [] for s in sizes for k in ks for st in STATS}
for rep in range(a.reps):
    folds = list(StratifiedKFold(splits, shuffle=True, random_state=rep).split(np.zeros(n), y))
    for s in sizes:
        descs = [d[s]['desc'] for _, d in recs]
        for k in ks:
            oof = {st: np.zeros(n) for st in STATS}
            for tr, te in folds:
                bank = PT.PatchBank(k).fit([descs[i] for i in tr[y[tr] == 0]])
                for i in te:
                    summ = PT.summarize(bank.score(descs[i]))
                    for st, j in STATS.items():
                        oof[st][i] = summ[j]
            for st in STATS:
                auc[(s, k, st)].append(roc_auc_score(y, oof[st]))

lines = ['=== Do nhay tham so | %s | ok=%d defect=%d | %d-fold x %d lan ===' % (name, int((y == 0).sum()), int(y.sum()), splits, a.reps), '']
for st in STATS:
    lines.append('AUC cua thong ke %s (hang: kich thuoc o, cot: k)' % st)
    lines.append('        ' + ''.join('k=%-7d' % k for k in ks))
    for s in sizes:
        lines.append('o=%-5d ' % s + ''.join('%.3f   ' % np.mean(auc[(s, k, st)]) for k in ks))
    lines.append('')
lines.append('Mac dinh cua pipeline: o=24, k=3, thong ke p_top5. Chenh lech < 0.03 giua cac o thuong nam trong sai so do it anh.')
text = '\n'.join(lines)
print(text)
(OUT_DIR / ('sensitivity_%s.txt' % name)).write_text(text, encoding='utf-8')

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for axx, st in zip(ax, ['p_top5', 'p_p90']):
    M = np.array([[np.mean(auc[(s, k, st)]) for k in ks] for s in sizes])
    im = axx.imshow(M, vmin=0.5, vmax=1.0, cmap='viridis')
    axx.set_xticks(range(len(ks)))
    axx.set_xticklabels(['k=%d' % k for k in ks])
    axx.set_yticks(range(len(sizes)))
    axx.set_yticklabels(['o=%d' % s for s in sizes])
    for i in range(len(sizes)):
        for j in range(len(ks)):
            axx.text(j, i, '%.2f' % M[i, j], ha='center', va='center', color='w' if M[i, j] < .85 else 'k')
    axx.set_title('%s - AUC (%s)' % (name, st))
plt.tight_layout()
(OUT_DIR / 'figures').mkdir(parents=True, exist_ok=True)
fp = OUT_DIR / 'figures' / ('sens_%s.png' % name)
plt.savefig(fp, dpi=150)
print('\nDa luu:', fp, 'va sensitivity_%s.txt' % name)
