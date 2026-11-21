"""Danh gia NANG CAO: so sanh nhieu phuong phap tren CUNG cac lan chia kiem tra cheo (stratified k-fold lap lai).
  baseline_maxz : max-z dac trung toan cuc (mot lop)               [giao trinh + thong ke]
  patch_knn     : diem bat thuong theo o (PatchCore thu cong, mot lop, khong dung anh loi)
  sup_G_lr      : hoi quy logistic tren dac trung toan cuc (co giam sat)     [doi chung: chua co o]
  sup_GS_lr     : hoi quy logistic: toan cuc + tom tat diem theo o
  sup_GAS_rf    : rung ngau nhien: toan cuc + dac trung vat + tom tat diem theo o
Moi phuong phap chi duoc hieu chuan/huan luyen tren tap train cua tung fold -> khong ro ri nhan.
Chay:  python scripts/benchmark.py hazelnut   [--splits 5 --reps 10 --k 3]"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from src.config import DATASETS, OUT_DIR
from src.dataset import list_images
from src import pipeline as P
from src import patches as PT
from src.oneclass import maxz_scores

ap = argparse.ArgumentParser()
ap.add_argument('dataset', choices=list(DATASETS))
ap.add_argument('--splits', type=int, default=5)
ap.add_argument('--reps', type=int, default=10)
ap.add_argument('--k', type=int, default=3, help='so lang gieng gan nhat khi cham diem o')
a = ap.parse_args()
name, cfg = a.dataset, DATASETS[a.dataset]
(OUT_DIR / 'figures').mkdir(parents=True, exist_ok=True)
ann_dir = OUT_DIR / 'annotated_v2' / name
ann_dir.mkdir(parents=True, exist_ok=True)

# ---------- 1. Trich dac trung ----------
recs = []
for path, label in list_images(name):
    img = P.load(path, cfg['crop'])
    r = P.extract(img, cfg)
    pd = PT.patch_descriptors(img, r) if r is not None else None
    if r is None or pd is None:
        print('[CANH BAO] phan vung that bai, bo qua:', path.name)
        continue
    r.update(path=path, file=path.name, label=label, pd=pd, img=img)
    recs.append(r)
n = len(recs)
y = np.array([1 if r['label'] == 'defect' else 0 for r in recs])
n_def, n_ok = int(y.sum()), int((y == 0).sum())
splits = min(a.splits, n_def, n_ok)
if splits < 2:
    sys.exit('Khong du anh (ok=%d, defect=%d)' % (n_ok, n_def))
F = {f: np.array([r[f] for r in recs]) for f in P.BASE_FEATS}
H = np.array([r['hist'] for r in recs], np.float32)
descs = [r['pd']['desc'] for r in recs]
G = np.stack([F[f] for f in P.BASE_FEATS], 1)                    # 5 dac trung toan cuc
A = np.stack([d.mean(0) for d in descs], 0)                      # 12 dac trung trung binh cua vat
fnames = P.BASE_FEATS + ['mean_' + d for d in PT.DESC_NAMES] + PT.PATCH_FEATS

METHODS = ['baseline_maxz', 'patch_knn', 'sup_G_lr', 'sup_GS_lr', 'sup_GAS_rf']


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight='balanced', max_iter=3000))


def rf(seed):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced_subsample',
                                  random_state=seed, n_jobs=-1)


def tpr_at(fpr_t, yy, s):
    f, t, _ = roc_curve(yy, s)
    return float(np.interp(fpr_t, f, t))


def prec_at(fpr_t, yy, s):
    thr = np.quantile(s[yy == 0], 1 - fpr_t)
    pred = s > thr
    tp, fp = int((pred & (yy == 1)).sum()), int((pred & (yy == 0)).sum())
    return tp / (tp + fp) if tp + fp else 0.0


# ---------- 2. Kiem tra cheo lap lai ----------
res = {m: dict(auc=[], tpr5=[], tpr10=[], tpr20=[], prec10=[]) for m in METHODS}
canon = None
for rep in range(a.reps):
    rng = np.random.default_rng(rep)
    oof = {m: np.zeros(n) for m in METHODS}
    S_all = np.zeros((n, 4))
    patch_sc, patch_thr = {}, {}
    for tr, te in StratifiedKFold(splits, shuffle=True, random_state=rep).split(np.zeros(n), y):
        tr_ok, tr_def = tr[y[tr] == 0], tr[y[tr] == 1]
        bank = PT.PatchBank(a.k).fit([descs[i] for i in tr_ok])
        S = np.zeros((n, 4))
        for i in np.concatenate([te, tr_def]):                    # anh test + anh loi train: cham bang ngan hang day du
            sc = bank.score(descs[i])
            S[i] = PT.summarize(sc)
            if rep == 0 and i in set(te.tolist()):
                patch_sc[i] = sc
        ok_patch_scores = []
        groups = np.array_split(rng.permutation(tr_ok), 5)        # anh OK train: cham cheo (khong tu cham minh)
        for gi, g in enumerate(groups):
            others = np.concatenate([groups[j] for j in range(len(groups)) if j != gi])
            b2 = PT.PatchBank(a.k).fit([descs[i] for i in others])
            for i in g:
                sc = b2.score(descs[i])
                S[i] = PT.summarize(sc)
                ok_patch_scores.append(sc)
        thr_patch = float(np.percentile(np.concatenate(ok_patch_scores), 99))
        for i in te:
            patch_thr[i] = thr_patch
        S_all[te] = S[te]

        oof['baseline_maxz'][te] = maxz_scores(F, H, tr_ok)[te]
        oof['patch_knn'][te] = S[te, 1]                           # trung binh 5 o bat thuong nhat
        for m, X, model in [('sup_G_lr', G, lr()), ('sup_GS_lr', np.hstack([G, S]), lr()),
                            ('sup_GAS_rf', np.hstack([G, A, S]), rf(rep))]:
            model.fit(X[tr], y[tr])
            oof[m][te] = model.predict_proba(X[te])[:, 1]
    for m in METHODS:
        s = oof[m]
        res[m]['auc'].append(roc_auc_score(y, s))
        res[m]['tpr5'].append(tpr_at(0.05, y, s))
        res[m]['tpr10'].append(tpr_at(0.10, y, s))
        res[m]['tpr20'].append(tpr_at(0.20, y, s))
        res[m]['prec10'].append(prec_at(0.10, y, s))
    if rep == 0:
        canon = (oof, patch_sc, patch_thr)

ms = lambda v: (float(np.mean(v)), float(np.std(v)))
best = max(METHODS, key=lambda m: np.mean(res[m]['auc']))   # phuong phap tot nhat (dung de ve anh chu thich)
lines = ['=== %s | %s ===' % (name, cfg['role']),
         'Anh ok=%d, defect=%d | %d-fold x %d lan lap | %d o/anh trung binh' % (
             n_ok, n_def, splits, a.reps, int(np.mean([len(d) for d in descs]))), '',
         'Phuong phap       AUC            Recall@FPR5%  Recall@FPR10%  Recall@FPR20%  Precision@FPR10%']
for m in METHODS:
    r = res[m]
    lines.append('%-16s  %.3f +- %.3f   %.2f          %.2f           %.2f           %.2f' % (
        m, *ms(r['auc']), ms(r['tpr5'])[0], ms(r['tpr10'])[0], ms(r['tpr20'])[0], ms(r['prec10'])[0]))
lines += ['', 'Recall@FPR10% = ty le anh loi bat duoc khi cho phep bao nham 10% anh dat (nguong lay tu diem anh dat).',
          'Moi so la trung binh tren cac lan chia; khong co phuong phap nao duoc dung nhan loi cua anh kiem tra de huan luyen.']

# do quan trong dac trung (huan luyen tren toan bo du lieu, chi de DIEN GIAI)
S_full = np.zeros((n, 4))
bank = PT.PatchBank(a.k).fit([descs[i] for i in np.where(y == 0)[0]])
ok_idx = np.where(y == 0)[0]
groups = np.array_split(np.random.default_rng(0).permutation(ok_idx), 5)
for gi, g in enumerate(groups):
    b2 = PT.PatchBank(a.k).fit([descs[i] for j in range(5) if j != gi for i in groups[j]])
    for i in g:
        S_full[i] = PT.summarize(b2.score(descs[i]))
for i in np.where(y == 1)[0]:
    S_full[i] = PT.summarize(bank.score(descs[i]))
imp = rf(0).fit(np.hstack([G, A, S_full]), y).feature_importances_
order = np.argsort(imp)[::-1][:8]
lines += ['', 'Top dac trung quan trong (rung ngau nhien, chi de dien giai):'] + [
    '  %-18s %.3f' % (fnames[i], imp[i]) for i in order]
text = '\n'.join(lines)
print(text)
(OUT_DIR / ('benchmark_%s.txt' % name)).write_text(text, encoding='utf-8')
json.dump({'dataset': name, 'n_ok': n_ok, 'n_defect': n_def, 'annot_method': best,
           'methods': {m: {k: ms(v) for k, v in res[m].items()} for m in METHODS}},
          open(OUT_DIR / ('benchmark_%s.json' % name), 'w'), indent=2)

# ---------- 3. Anh chu thich (du doan KIEM TRA CHEO cua lan chia 0, nen trung thuc) ----------
oof, patch_sc, patch_thr = canon
op_thr = float(np.quantile(oof[best][y == 0], 0.90))              # nguong ung voi bao nham ~10% anh dat
for i, r in enumerate(recs):
    bad = oof[best][i] > op_thr
    col = (0, 0, 255) if bad else (0, 200, 0)
    out = P.draw(r['img'], r, '%s %.2f' % ('DEFECT' if bad else 'OK', oof[best][i]), col)
    sc, s_, pd = patch_sc[i], r['pd']['scale'], r['pd']
    for (x, yy), v in zip(pd['xy'], sc):
        if v > patch_thr[i]:                                        # o bat thuong -> khoanh do
            h = int(PT.PATCH / s_ / 2)
            cx, cy = int(x / s_), int(yy / s_)
            cv2.rectangle(out, (cx - h, cy - h), (cx + h, cy + h), (0, 0, 255), 1)
    cv2.putText(out, 'GT:%s' % r['label'], (5, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(str(ann_dir / ('%s__%s' % (r['label'], r['file']))), out)
import csv
with open(OUT_DIR / ('benchmark_%s_scores.csv' % name), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['file', 'label'] + METHODS)
    for i, r in enumerate(recs):
        w.writerow(['%s__%s' % (r['label'], r['file']), r['label']] + ['%.4f' % oof[m][i] for m in METHODS])

# ---------- 4. Bieu do ----------
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.5))
col = ['tab:gray', 'tab:orange', 'tab:blue', 'tab:cyan', 'tab:green']
ax[0].bar(range(len(METHODS)), [ms(res[m]['auc'])[0] for m in METHODS],
          yerr=[ms(res[m]['auc'])[1] for m in METHODS], color=col, capsize=3)
ax[0].axhline(0.5, ls='--', c='k', lw=1)
ax[0].set_xticks(range(len(METHODS)))
ax[0].set_xticklabels(METHODS, rotation=30, ha='right')
ax[0].set_ylim(0, 1.05)
ax[0].set_ylabel('AUC')
ax[0].set_title('%s - AUC (kiem tra cheo)' % name)
for m, c in zip(METHODS, col):
    f_, t_, _ = roc_curve(y, oof[m])
    ax[1].plot(f_, t_, color=c, label='%s (%.2f)' % (m, roc_auc_score(y, oof[m])))
ax[1].plot([0, 1], [0, 1], 'k--', lw=1)
ax[1].set_xlabel('Ti le bao nham (FPR)')
ax[1].set_ylabel('Ti le bat loi (Recall)')
ax[1].legend(fontsize=8, loc='lower right')
ax[1].set_title('Duong ROC - lan chia 0')
plt.tight_layout()
fp = OUT_DIR / 'figures' / ('bench_%s.png' % name)
plt.savefig(fp, dpi=150)
print('\nDa luu: %s, benchmark_%s.txt/json/_scores.csv, anh chu thich (%s) trong outputs/annotated_v2/%s/' % (
    fp.name, name, best, name))
