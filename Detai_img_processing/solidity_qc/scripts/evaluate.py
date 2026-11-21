"""BUOC 2: danh gia phat hien khuyet tat theo kieu MOT LOP (one-class).
 - Chia anh OK: ~60% de HIEU CHUAN (tinh trung vi + MAD), phan con lai + toan bo anh LOI de KIEM TRA.
 - Moi dac trung -> diem z (cao = bat thuong). Diem anh = max z cua cac dac trung trong bo.
 - Lap nhieu lan chia ngau nhien (mac dinh 30) de ket qua on dinh khi du lieu it.
Chay:  python scripts/evaluate.py hazelnut     (biscuits | hazelnut | pills)
Tuy chon: --cal 0.6  --reps 30  --k 3.0"""
import sys, csv, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.config import DATASETS, OUT_DIR
from src.dataset import list_images
from src import pipeline as P

ap = argparse.ArgumentParser()
ap.add_argument('dataset', choices=list(DATASETS))
ap.add_argument('--cal', type=float, default=0.6, help='ti le anh OK dung hieu chuan')
ap.add_argument('--reps', type=int, default=30, help='so lan chia ngau nhien')
ap.add_argument('--k', type=float, default=3.0, help='nguong z de ket luan LOI')
a = ap.parse_args()
name, cfg = a.dataset, DATASETS[a.dataset]
(OUT_DIR / 'figures').mkdir(parents=True, exist_ok=True)
ann_dir = OUT_DIR / 'annotated' / name
ann_dir.mkdir(parents=True, exist_ok=True)

# ---------- 1. Trich dac trung ----------
recs = []
for path, label in list_images(name):
    r = P.extract(P.load(path, cfg['crop']), cfg)
    if r is None:
        print('[CANH BAO] phan vung that bai, bo qua:', path.name)
        continue
    r.update(path=path, file=path.name, label=label)
    recs.append(r)
y = np.array([1 if r['label'] == 'defect' else 0 for r in recs])
ok_idx, def_idx = np.where(y == 0)[0], np.where(y == 1)[0]
if len(ok_idx) < 8 or len(def_idx) < 1:
    sys.exit('Can it nhat 8 anh ok va 1 anh defect (hien co ok=%d, defect=%d)' % (len(ok_idx), len(def_idx)))
vals = {f: np.array([r[f] for r in recs]) for f in P.BASE_FEATS}
H = np.array([r['hist'] for r in recs], np.float32)
n_cal = min(max(5, int(round(a.cal * len(ok_idx)))), len(ok_idx) - 3)

SETS = {'shape': P.SHAPE_FEATS, 'surface': P.SURFACE_FEATS, 'all': P.ALL_FEATS}


# ---------- 2. Ham chuc nang ----------
def zscore(v, cal, direction):
    ref = v[cal]
    med = np.median(ref)
    scale = max(1.4826 * np.median(np.abs(ref - med)), 0.5 * ref.std(), 1e-3 * abs(med), 1e-9)
    return direction * (v - med) / scale


def all_z(cal):
    """Diem z cua moi dac trung cho TAT CA anh, hieu chuan tren tap cal."""
    Z = {f: zscore(vals[f], cal, P.DIRECTION[f]) for f in P.BASE_FEATS}
    s, calset = H[cal].sum(0), set(cal.tolist())
    hd = np.zeros(len(H))
    for i in range(len(H)):                     # anh trong cal: bo chinh no ra khi tinh mau chuan
        ref = (s - H[i]) / (len(cal) - 1) if i in calset else s / len(cal)
        hd[i] = cv2.compareHist(H[i], ref.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
    Z['hist_dist'] = zscore(hd, cal, 1)
    for sname, fl in SETS.items():
        Z[sname] = np.max([Z[f] for f in fl], axis=0)
    return Z


def auc(s_def, s_ok):
    d, o = s_def[:, None], s_ok[None, :]
    return float(((d > o).sum() + 0.5 * (d == o).sum()) / (d.size * o.size))


def prf(s_def, s_ok, k):
    tp, fp = int((s_def > k).sum()), int((s_ok > k).sum())
    fn, tn = len(s_def) - tp, len(s_ok) - fp
    pre = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
    return dict(precision=pre, recall=rec, f1=f1, fpr=fp / (fp + tn))


# ---------- 3. Lap nhieu lan chia ----------
keys = P.ALL_FEATS + list(SETS)
res_auc = {kk: [] for kk in keys}
res_prf = {s: {m: [] for m in ('precision', 'recall', 'f1', 'fpr')} for s in SETS}
canon = None
for rep in range(a.reps):
    rng = np.random.default_rng(rep)
    cal = rng.choice(ok_idx, n_cal, replace=False)
    test_ok = np.setdiff1d(ok_idx, cal)
    Z = all_z(cal)
    if rep == 0:
        canon = (cal, Z)
    for kk in keys:
        res_auc[kk].append(auc(Z[kk][def_idx], Z[kk][test_ok]))
    for s in SETS:
        for m, v in prf(Z[s][def_idx], Z[s][test_ok], a.k).items():
            res_prf[s][m].append(v)

ms = lambda arr: (float(np.mean(arr)), float(np.std(arr)))
lines = []
P_ = lines.append
P_('=== %s | %s ===' % (name, cfg['role']))
P_('Anh ok=%d, defect=%d | hieu chuan=%d anh ok/lan, kiem tra=%d ok + %d defect | %d lan chia | k=%.1f'
   % (len(ok_idx), len(def_idx), n_cal, len(ok_idx) - n_cal, len(def_idx), a.reps, a.k))
P_('')
P_('AUC theo tung dac trung (0.5 = doan mo, 1.0 = tach hoan toan):')
for f in P.ALL_FEATS:
    m, s = ms(res_auc[f])
    P_('  %-14s %.3f +- %.3f' % (f, m, s))
P_('')
P_('Bo dac trung (diem = max z):   AUC            precision  recall   F1      FPR')
for sname in SETS:
    am, asd = ms(res_auc[sname])
    pr = {m: ms(res_prf[sname][m])[0] for m in res_prf[sname]}
    P_('  %-8s                      %.3f +- %.3f   %.2f       %.2f     %.2f    %.2f' % (
        sname, am, asd, pr['precision'], pr['recall'], pr['f1'], pr['fpr']))
P_('')
P_('Luu y: ket qua la trung binh tren nhieu lan chia ngau nhien; du lieu it nen do lech chuan van lon.')
text = '\n'.join(lines)
print(text)
(OUT_DIR / ('summary_%s.txt' % name)).write_text(text, encoding='utf-8')

summary = dict(dataset=name, role=cfg['role'], n_ok=int(len(ok_idx)), n_defect=int(len(def_idx)), k=a.k,
               auc={kk: dict(zip(('mean', 'std'), ms(res_auc[kk]))) for kk in keys},
               prf={s: {m: ms(res_prf[s][m])[0] for m in res_prf[s]} for s in SETS})
(OUT_DIR / ('summary_%s.json' % name)).write_text(json.dumps(summary, indent=2), encoding='utf-8')

# ---------- 4. CSV + anh chu thich (lan chia chuan: rep 0) ----------
cal, Z = canon
calset = set(cal.tolist())
with open(OUT_DIR / ('results_%s.csv' % name), 'w', newline='', encoding='utf-8') as f:
    cols = ['file', 'label', 'split'] + P.BASE_FEATS + ['hist_dist', 'score_all', 'pred']
    w = csv.writer(f)
    w.writerow(cols)
    for i, r in enumerate(recs):
        split = 'cal' if i in calset else 'test'
        pred = 'defect' if Z['all'][i] > a.k else 'ok'
        w.writerow([r['file'], r['label'], split] + ['%.5f' % r[ff] for ff in P.BASE_FEATS]
                   + ['%.5f' % Z['hist_dist'][i], '%.2f' % Z['all'][i], pred])
        bad = Z['all'][i] > a.k
        txt = '%s z=%.1f%s' % ('DEFECT' if bad else 'OK', Z['all'][i], ' (cal)' if split == 'cal' else '')
        img = P.load(r['path'], cfg['crop'])
        cv2.imwrite(str(ann_dir / ('%s__%s' % (r['label'], r['file']))), P.draw(img, r, txt, (0, 0, 255) if bad else (0, 200, 0)))

# ---------- 5. Bieu do ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
names = P.ALL_FEATS + list(SETS)
means = [ms(res_auc[n])[0] for n in names]
errs = [ms(res_auc[n])[1] for n in names]
colors = ['tab:blue'] * len(P.SHAPE_FEATS) + ['tab:orange'] * len(P.SURFACE_FEATS) + ['tab:green'] * 3
ax[0].bar(range(len(names)), means, yerr=errs, color=colors, capsize=3)
ax[0].axhline(0.5, ls='--', c='k', lw=1)
ax[0].set_xticks(range(len(names)))
ax[0].set_xticklabels(names, rotation=40, ha='right')
ax[0].set_ylim(0, 1.05)
ax[0].set_ylabel('AUC')
ax[0].set_title('%s - AUC theo dac trung' % name)
test_ok = np.setdiff1d(ok_idx, cal)
sc = Z['all']
hi = max(sc[def_idx].max(), sc[test_ok].max(), a.k) * 1.05
bins = np.linspace(min(0, sc.min()), hi, 25)
ax[1].hist(sc[test_ok], bins, alpha=.7, color='tab:green', label='ok (kiem tra)')
ax[1].hist(sc[def_idx], bins, alpha=.7, color='tab:red', label='defect')
ax[1].axvline(a.k, ls='--', c='k', label='k = %.1f' % a.k)
ax[1].set_xlabel('Diem bat thuong (max z, bo "all")')
ax[1].legend()
ax[1].set_title('Phan bo diem - lan chia chuan')
plt.tight_layout()
fp = OUT_DIR / 'figures' / ('eval_%s.png' % name)
plt.savefig(fp, dpi=150)
print('\nDa luu: %s, results_%s.csv, summary_%s.txt/json, anh chu thich trong outputs/annotated/%s/' % (fp.name, name, name, name))
