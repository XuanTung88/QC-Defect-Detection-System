"""
web_bridge.py - Cầu nối Python CLI cho Hệ thống Web Kiểm tra Khuyết tật Hình dạng.
GIỮ NGUYÊN 100% mã nguồn trong src/. Chỉ gọi đúng pipeline gốc.

LUỒNG ĐÚNG:
  Bước 1 - explore  : Kiểm tra phân vùng Otsu + trực quan Convex Hull
  Bước 2 - evaluate  : Đánh giá one-class z-score, tính AUC từng đặc trưng
  Bước 3 - benchmark : So sánh 5 phương pháp cross-validation, tạo ảnh annotated
  Bước 4 - report    : Ghép ảnh lỗi FN/FP/TP để phân tích

Giao tiếp với Node.js qua stdout (JSON).
"""
import sys
import os
import json
import argparse
import pickle
import io
import base64
from pathlib import Path
from contextlib import redirect_stdout

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import cv2
import numpy as np

from src import pipeline as P
from src import patches as PT
from src.config import DATASETS, OK_NAMES, DEFECT_NAMES, DATA_ROOT, OUT_DIR

BASE_FEATS = P.BASE_FEATS
ALL_FEATS = P.ALL_FEATS
DIRECTION = P.DIRECTION


def convert_types(obj):
    """Chuyển kiểu numpy sang python native để JSON serialize."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()
                if k not in ('contour', 'hull', 'interior', 'hist', 'img')}
    elif isinstance(obj, (list, tuple)):
        return [convert_types(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    return obj


def find_data_dir(dataset_name, custom_dir=None):
    """Tìm thư mục dữ liệu: ưu tiên custom_dir, sau đó DATA_ROOT gốc."""
    if custom_dir:
        p = Path(custom_dir)
        if p.exists():
            return p
    # Thư mục data gốc từ config.py
    default = DATA_ROOT / dataset_name
    if default.exists():
        return default
    return None


def list_images_from_dir(data_dir, exts=('.png', '.jpg', '.jpeg')):
    """Liệt kê ảnh từ thư mục có cấu trúc: data_dir/ok/ và data_dir/defect/"""
    items = []
    data_dir = Path(data_dir)
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        key = sub.name.lower()
        if key in OK_NAMES:
            label = 'ok'
        elif key in DEFECT_NAMES:
            label = 'defect'
        else:
            continue
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() in exts:
                items.append((f, label))
    return items


def parse_dataset_cfg(dataset_name=None, cfg_override=None):
    """Lấy config: từ DATASETS gốc nếu có, sau đó apply override."""
    if dataset_name and dataset_name in DATASETS:
        cfg = dict(DATASETS[dataset_name])
    else:
        cfg = dict(channel='gray', object_bright=True, crop=0, interior='filled',
                   exts=('.png', '.jpg', '.jpeg'))
    if not cfg.get('exts'):
        cfg['exts'] = ('.png', '.jpg', '.jpeg')
    if cfg_override:
        try:
            overrides = json.loads(cfg_override) if isinstance(cfg_override, str) else cfg_override
            cfg.update(overrides)
        except Exception as e:
            sys.stderr.write(f"[Warning] cfg parse error: {e}\n")
    return cfg


# =======================================================================
# BƯỚC 1: EXPLORE - Trực quan hóa phân vùng Otsu + Convex Hull
# =======================================================================
def cmd_explore(args):
    """
    Tương đương explore.py gốc nhưng trả kết quả dạng JSON + base64 ảnh.
    Ghi ảnh lưới ra file (đúng như explore.py) và trả đường dẫn + console log.
    """
    dataset_name = args.dataset_name
    cfg = parse_dataset_cfg(dataset_name, args.cfg)
    data_dir = find_data_dir(dataset_name, args.data_dir)

    if data_dir is None:
        print(json.dumps({"success": False, "error": f"Khong tim thay thu muc du lieu cho '{dataset_name}'."}))
        return

    items = list_images_from_dir(data_dir, cfg.get('exts'))
    if not items:
        print(json.dumps({"success": False, "error": "Khong tim thay anh nao trong thu muc."}))
        return

    n_per_class = int(args.n_per_class) if args.n_per_class else 4
    ok_items = [i for i in items if i[1] == 'ok'][:n_per_class]
    def_items = [i for i in items if i[1] == 'defect'][:n_per_class]
    pick = ok_items + def_items

    H = 240
    rows = []
    console_lines = []
    sample_results = []

    for path, label in pick:
        try:
            img = P.load(path, cfg.get('crop', 0))
            r = P.extract(img, cfg)
            if r is None:
                console_lines.append(f"[CANH BAO] phan vung that bai: {path.name}")
                continue
            mask, _ = P.segment(P.to_channel(img, cfg['channel']), cfg['object_bright'])
            g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            edges = cv2.Canny(g, 50, 120)
            edges[r['interior'] == 0] = 0
            color = (0, 0, 255) if label == 'defect' else (0, 200, 0)
            vis = P.draw(img, r, f"S={r['solidity']:.3f}", color)
            cols = [img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                    vis, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)]
            cols_r = [cv2.resize(c, (int(c.shape[1] * H / c.shape[0]), H)) for c in cols]
            row_img = np.hstack(cols_r)
            cv2.putText(row_img, f"{label} | {path.name} | otsu={r['otsu']}",
                        (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            rows.append(row_img)

            log = (f"{label:<7} {path.name:<26} solidity={r['solidity']:.4f} "
                   f"circ={r['circularity']:.3f} edge={r['edge_density']:.4f} "
                   f"sobel={r['sobel_mean']:.1f} std={r['gray_std']:.1f}")
            console_lines.append(log)
            sample_results.append({
                "filename": path.name, "label": label,
                "solidity": float(r['solidity']),
                "circularity": float(r['circularity']),
                "edge_density": float(r['edge_density']),
                "sobel_mean": float(r['sobel_mean']),
                "gray_std": float(r['gray_std']),
                "otsu": int(r['otsu'])
            })
        except Exception as e:
            console_lines.append(f"[LOI] {path.name}: {e}")

    # Lưu ảnh đầu ra (đúng như explore.py)
    out_fig_path = None
    custom_out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    custom_out_dir.mkdir(parents=True, exist_ok=True)

    if rows:
        W = max(r.shape[1] for r in rows)
        rows_padded = [cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1], cv2.BORDER_CONSTANT)
                       for r in rows]
        grid = np.vstack(rows_padded)
        fig_dir = custom_out_dir / 'figures'
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_fig_path = str(fig_dir / f"explore_{dataset_name}.png")
        cv2.imwrite(out_fig_path, grid)
        console_lines.append(f"Da luu: {out_fig_path}")

    print(json.dumps(convert_types({
        "success": True,
        "step": "explore",
        "dataset": dataset_name,
        "total_ok": len(ok_items),
        "total_defect": len(def_items),
        "processed": len(sample_results),
        "figure_path": out_fig_path,
        "console": "\n".join(console_lines),
        "samples": sample_results
    }), ensure_ascii=False))


# =======================================================================
# BƯỚC 2: EVALUATE - Đánh giá one-class z-score (evaluate.py)
# =======================================================================
def cmd_evaluate(args):
    """
    Tương đương evaluate.py gốc. Trả JSON kết quả và lưu file outputs.
    """
    dataset_name = args.dataset_name
    cfg = parse_dataset_cfg(dataset_name, args.cfg)
    data_dir = find_data_dir(dataset_name, args.data_dir)

    if data_dir is None:
        print(json.dumps({"success": False, "error": f"Khong tim thay du lieu cho '{dataset_name}'."}))
        return

    items = list_images_from_dir(data_dir, cfg.get('exts'))
    if not items:
        print(json.dumps({"success": False, "error": "Khong co anh nao trong thu muc."}))
        return

    # Import evaluate functions
    import cv2 as _cv2
    import numpy as _np

    recs = []
    skipped = []
    for path, label in items:
        try:
            r = P.extract(P.load(path, cfg.get('crop', 0)), cfg)
            if r is None:
                skipped.append(path.name)
                continue
            r.update(path=path, file=path.name, label=label)
            recs.append(r)
        except Exception as e:
            skipped.append(f"{path.name}({e})")

    y = _np.array([1 if r['label'] == 'defect' else 0 for r in recs])
    ok_idx = _np.where(y == 0)[0]
    def_idx = _np.where(y == 1)[0]

    if len(ok_idx) < 5 or len(def_idx) < 1:
        print(json.dumps({"success": False,
                          "error": f"Khong du anh. Hien co ok={len(ok_idx)}, defect={len(def_idx)}. Can >= 5 OK va >= 1 Defect."}))
        return

    cal_ratio = float(args.cal_ratio) if args.cal_ratio else 0.6
    reps = int(args.reps) if args.reps else 30
    k_thr = float(args.threshold_k) if args.threshold_k else 3.0

    vals = {f: _np.array([r[f] for r in recs]) for f in BASE_FEATS}
    H_arr = _np.array([r['hist'] for r in recs], _np.float32)
    n_cal = min(max(5, int(round(cal_ratio * len(ok_idx)))), len(ok_idx) - 2)

    SETS = {'shape': P.SHAPE_FEATS, 'surface': P.SURFACE_FEATS, 'all': P.ALL_FEATS}

    def zscore(v, cal, direction):
        ref = v[cal]
        med = _np.median(ref)
        scale = max(1.4826 * _np.median(_np.abs(ref - med)), 0.5 * ref.std(),
                    1e-3 * abs(med), 1e-9)
        return direction * (v - med) / scale

    def all_z(cal):
        Z = {f: zscore(vals[f], cal, DIRECTION[f]) for f in BASE_FEATS}
        s = H_arr[cal].sum(0)
        calset = set(cal.tolist())
        hd = _np.zeros(len(H_arr))
        for i in range(len(H_arr)):
            ref = (s - H_arr[i]) / (len(cal) - 1) if i in calset else s / len(cal)
            hd[i] = _cv2.compareHist(H_arr[i], ref.astype(_np.float32), _cv2.HISTCMP_BHATTACHARYYA)
        Z['hist_dist'] = zscore(hd, cal, 1)
        for sname, fl in SETS.items():
            Z[sname] = _np.max([Z[f] for f in fl], axis=0)
        return Z

    def auc_score(s_def, s_ok):
        d, o = s_def[:, None], s_ok[None, :]
        return float(((d > o).sum() + 0.5 * (d == o).sum()) / (d.size * o.size))

    def prf(s_def, s_ok, k):
        tp, fp = int((s_def > k).sum()), int((s_ok > k).sum())
        fn, tn = len(s_def) - tp, len(s_ok) - fp
        pre = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
        return dict(precision=pre, recall=rec, f1=f1, fpr=fp / (fp + tn) if (fp + tn) else 0.0)

    keys = BASE_FEATS + ['hist_dist', 'shape', 'surface', 'all']
    res_auc = {kk: [] for kk in keys}
    res_prf = {s: {m: [] for m in ('precision', 'recall', 'f1', 'fpr')} for s in SETS}

    canon = None
    for rep in range(reps):
        rng = _np.random.default_rng(rep)
        cal = rng.choice(ok_idx, n_cal, replace=False)
        test_ok = _np.setdiff1d(ok_idx, cal)
        Z = all_z(cal)
        if rep == 0:
            canon = (cal, Z)
        for kk in keys:
            res_auc[kk].append(auc_score(Z[kk][def_idx], Z[kk][test_ok]))
        for sname in SETS:
            for m, v in prf(Z[sname][def_idx], Z[sname][test_ok], k_thr).items():
                res_prf[sname][m].append(v)

    ms = lambda arr: (float(_np.mean(arr)), float(_np.std(arr)))

    auc_summary = {}
    for kk in keys:
        m, s = ms(res_auc[kk])
        auc_summary[kk] = {"mean": m, "std": s}

    prf_summary = {
        sname: {m: ms(res_prf[sname][m])[0] for m in res_prf[sname]}
        for sname in SETS
    }

    # Lưu summary JSON + TXT (đúng như evaluate.py)
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": dataset_name,
        "role": cfg.get("role", dataset_name),
        "n_ok": int(len(ok_idx)),
        "n_defect": int(len(def_idx)),
        "k": k_thr,
        "auc": auc_summary,
        "prf": prf_summary
    }
    summary_path = out_dir / f"summary_{dataset_name}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    txt_lines = [
        f"=== {dataset_name} | {cfg.get('role', dataset_name)} ===",
        f"Anh ok={len(ok_idx)}, defect={len(def_idx)} | hieu chuan={n_cal} anh ok/lan, kiem tra={len(ok_idx)-n_cal} ok + {len(def_idx)} defect | {reps} lan chia | k={k_thr:.1f}",
        "",
        "AUC theo tung dac trung:"
    ]
    for f in BASE_FEATS + ['hist_dist']:
        txt_lines.append(f"  {f:<14} {auc_summary[f]['mean']:.3f} +- {auc_summary[f]['std']:.3f}")
    txt_lines += ["", "Bo dac trung (diem = max z):   AUC            precision  recall   F1      FPR"]
    for sname in SETS:
        pr = prf_summary[sname]
        txt_lines.append(
            f"  {sname:<8}                      {auc_summary[sname]['mean']:.3f} +- {auc_summary[sname]['std']:.3f}   "
            f"{pr['precision']:.2f}       {pr['recall']:.2f}     {pr['f1']:.2f}    {pr['fpr']:.2f}"
        )
    (out_dir / f"summary_{dataset_name}.txt").write_text("\n".join(txt_lines), encoding='utf-8')

    # Lưu CSV + ảnh annotated (dùng canonical split)
    import csv
    cal_canon, Z_canon = canon
    calset_canon = set(cal_canon.tolist())
    ann_dir = out_dir / 'annotated' / dataset_name
    ann_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"results_{dataset_name}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        cols = ['file', 'label', 'split'] + BASE_FEATS + ['hist_dist', 'score_all', 'pred']
        w = csv.writer(f)
        w.writerow(cols)
        for i, r in enumerate(recs):
            split = 'cal' if i in calset_canon else 'test'
            bad = Z_canon['all'][i] > k_thr
            pred = 'defect' if bad else 'ok'
            w.writerow([r['file'], r['label'], split]
                       + [f"{r[ff]:.5f}" for ff in BASE_FEATS]
                       + [f"{Z_canon['hist_dist'][i]:.5f}", f"{Z_canon['all'][i]:.2f}", pred])
            txt = f"{'DEFECT' if bad else 'OK'} z={Z_canon['all'][i]:.1f}{' (cal)' if split == 'cal' else ''}"
            img = P.load(r['path'], cfg.get('crop', 0))
            out_img = P.draw(img, r, txt, (0, 0, 255) if bad else (0, 200, 0))
            cv2.imwrite(str(ann_dir / f"{r['label']}__{r['file']}"), out_img)

    # Vẽ biểu đồ eval_<dataset>.png
    eval_fig_path = None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        (out_dir / 'figures').mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        names = BASE_FEATS + ['hist_dist'] + list(SETS)
        means = [auc_summary[n]['mean'] for n in names]
        errs = [auc_summary[n]['std'] for n in names]
        colors = ['tab:blue'] * len(P.SHAPE_FEATS) + ['tab:orange'] * len(P.SURFACE_FEATS) + ['tab:green'] * 3
        ax[0].bar(range(len(names)), means, yerr=errs, color=colors, capsize=3)
        ax[0].axhline(0.5, ls='--', c='k', lw=1)
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=40, ha='right')
        ax[0].set_ylim(0, 1.05)
        ax[0].set_ylabel('AUC')
        ax[0].set_title(f"{dataset_name} - AUC theo dac trung")
        test_ok = _np.setdiff1d(ok_idx, cal_canon)
        sc = Z_canon['all']
        hi = max(float(sc[def_idx].max()), float(sc[test_ok].max()) if len(test_ok) else 0.0, k_thr) * 1.05
        bins = _np.linspace(min(0.0, float(sc.min())), hi, 25)
        if len(test_ok):
            ax[1].hist(sc[test_ok], bins, alpha=.7, color='tab:green', label='ok (kiem tra)')
        ax[1].hist(sc[def_idx], bins, alpha=.7, color='tab:red', label='defect')
        ax[1].axvline(k_thr, ls='--', c='k', label=f'k = {k_thr:.1f}')
        ax[1].set_xlabel('Diem bat thuong (max z, bo "all")')
        ax[1].legend()
        ax[1].set_title('Phan bo diem - lan chia chuan')
        plt.tight_layout()
        eval_fig_path = str(out_dir / 'figures' / f"eval_{dataset_name}.png")
        plt.savefig(eval_fig_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        sys.stderr.write(f"[Warning] eval plot error: {e}\n")

    print(json.dumps(convert_types({
        "success": True,
        "step": "evaluate",
        "dataset": dataset_name,
        "n_ok": int(len(ok_idx)),
        "n_defect": int(len(def_idx)),
        "skipped": skipped,
        "auc": auc_summary,
        "prf": prf_summary,
        "summary_path": str(summary_path),
        "csv_path": str(csv_path),
        "figure_path": eval_fig_path,
        "annotated_dir": str(ann_dir)
    }), ensure_ascii=False))


# =======================================================================
# BƯỚC 3: BENCHMARK - So sánh 5 phương pháp (benchmark.py)
# =======================================================================
def cmd_benchmark(args):
    """
    Tương đương benchmark.py gốc: 5 phương pháp cross-val, lưu JSON + CSV + ảnh.
    """
    dataset_name = args.dataset_name
    cfg = parse_dataset_cfg(dataset_name, args.cfg)
    data_dir = find_data_dir(dataset_name, args.data_dir)

    if data_dir is None:
        print(json.dumps({"success": False, "error": f"Khong tim thay du lieu cho '{dataset_name}'."}))
        return

    items = list_images_from_dir(data_dir, cfg.get('exts'))
    if not items:
        print(json.dumps({"success": False, "error": "Khong co anh."}))
        return

    # Chạy script benchmark.py gốc qua subprocess để giữ NGUYÊN 100%
    import subprocess
    script_path = CURRENT_DIR / 'scripts' / 'benchmark.py'
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Nếu dataset là dataset tùy chỉnh (không có trong DATASETS gốc),
    # cần thêm vào config tạm thời
    extra_env = dict(os.environ)
    if dataset_name not in DATASETS and args.data_dir:
        # Truyền qua biến môi trường để script biết dùng custom data_dir
        extra_env['WEB_CUSTOM_DATA_DIR'] = str(data_dir)
        extra_env['WEB_CUSTOM_DATASET'] = dataset_name

    splits = int(args.splits) if args.splits else 5
    reps = int(args.reps) if args.reps else 10

    result = subprocess.run(
        [sys.executable, str(script_path), dataset_name,
         '--splits', str(splits), '--reps', str(reps)],
        capture_output=True, text=True, encoding='utf-8',
        cwd=str(CURRENT_DIR), env=extra_env
    )

    if result.returncode != 0:
        # Nếu dataset không có trong DATASETS chuẩn → chạy inline
        err = result.stderr or result.stdout
        if 'invalid choice' in err or dataset_name not in DATASETS:
            # Chạy phiên bản inline cho custom dataset
            _run_benchmark_inline(dataset_name, cfg, items, out_dir, splits, reps, args)
            return
        print(json.dumps({"success": False, "error": f"Benchmark that bai: {err[:500]}"}))
        return

    # Đọc kết quả từ file JSON đã lưu
    bench_json = out_dir / f"benchmark_{dataset_name}.json"
    if not bench_json.exists():
        bench_json = OUT_DIR / f"benchmark_{dataset_name}.json"

    result_data = {}
    if bench_json.exists():
        result_data = json.loads(bench_json.read_text(encoding='utf-8'))

    bench_fig = out_dir / 'figures' / f"bench_{dataset_name}.png"
    if not bench_fig.exists():
        bench_fig = OUT_DIR / 'figures' / f"bench_{dataset_name}.png"

    print(json.dumps(convert_types({
        "success": True,
        "step": "benchmark",
        "dataset": dataset_name,
        "console": result.stdout[-2000:] if result.stdout else "",
        "result": result_data,
        "figure_path": str(bench_fig) if bench_fig.exists() else None
    }), ensure_ascii=False))


def _run_benchmark_inline(dataset_name, cfg, items, out_dir, splits, reps, args):
    """Chạy benchmark inline cho custom dataset không có trong DATASETS gốc."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, roc_curve
    from src.oneclass import maxz_scores

    recs = []
    for path, label in items:
        try:
            img = P.load(path, cfg.get('crop', 0))
            r = P.extract(img, cfg)
            pd = PT.patch_descriptors(img, r) if r else None
            if r is None or pd is None:
                continue
            r.update(path=path, file=path.name, label=label, pd=pd, img=img)
            recs.append(r)
        except Exception:
            pass

    if len(recs) < 4:
        print(json.dumps({"success": False, "error": "Khong du anh hop le de benchmark."}))
        return

    n = len(recs)
    y = np.array([1 if r['label'] == 'defect' else 0 for r in recs])
    n_def, n_ok = int(y.sum()), int((y == 0).sum())
    actual_splits = min(splits, n_def, n_ok)
    if actual_splits < 2:
        print(json.dumps({"success": False, "error": f"Khong du anh cho cross-val (ok={n_ok}, defect={n_def})."}))
        return

    F = {f: np.array([r[f] for r in recs]) for f in BASE_FEATS}
    H = np.array([r['hist'] for r in recs], np.float32)
    descs = [r['pd']['desc'] for r in recs]
    G = np.stack([F[f] for f in BASE_FEATS], 1)
    A = np.stack([d.mean(0) for d in descs], 0)

    METHODS = ['baseline_maxz', 'patch_knn', 'sup_G_lr', 'sup_GS_lr', 'sup_GAS_rf']
    res = {m: dict(auc=[], tpr5=[], tpr10=[], tpr20=[], prec10=[]) for m in METHODS}

    def lr():
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight='balanced', max_iter=3000))
    def rf(seed):
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                      class_weight='balanced_subsample', random_state=seed, n_jobs=-1)
    def tpr_at(fpr_t, yy, s):
        f_, t_, _ = roc_curve(yy, s)
        return float(np.interp(fpr_t, f_, t_))
    def prec_at(fpr_t, yy, s):
        thr = np.quantile(s[yy == 0], 1 - fpr_t)
        pred = s > thr
        tp, fp = int((pred & (yy == 1)).sum()), int((pred & (yy == 0)).sum())
        return tp / (tp + fp) if tp + fp else 0.0

    canon = None
    actual_reps = min(reps, 5)
    for rep in range(actual_reps):
        rng = np.random.default_rng(rep)
        oof = {m: np.zeros(n) for m in METHODS}
        patch_sc, patch_thr = {}, {}
        for tr, te in StratifiedKFold(actual_splits, shuffle=True, random_state=rep).split(np.zeros(n), y):
            tr_ok, tr_def = tr[y[tr] == 0], tr[y[tr] == 1]
            bank = PT.PatchBank(3).fit([descs[i] for i in tr_ok])
            S = np.zeros((n, 4))
            for i in np.concatenate([te, tr_def]):
                sc = bank.score(descs[i])
                S[i] = PT.summarize(sc)
                if rep == 0 and i in set(te.tolist()):
                    patch_sc[i] = sc
            n_grp = min(5, len(tr_ok))
            groups = np.array_split(rng.permutation(tr_ok), max(n_grp, 2))
            ok_patch_scores = []
            for gi, g in enumerate(groups):
                others = np.concatenate([groups[j] for j in range(len(groups)) if j != gi])
                b2 = PT.PatchBank(3).fit([descs[i] for i in others])
                for i in g:
                    sc = b2.score(descs[i])
                    S[i] = PT.summarize(sc)
                    ok_patch_scores.append(sc)
            thr_p = float(np.percentile(np.concatenate(ok_patch_scores), 99)) if ok_patch_scores else 999.0
            for i in te:
                patch_thr[i] = thr_p
            oof['baseline_maxz'][te] = maxz_scores(F, H, tr_ok)[te]
            oof['patch_knn'][te] = S[te, 1]
            for m, X, model in [('sup_G_lr', G, lr()), ('sup_GS_lr', np.hstack([G, S]), lr()),
                                 ('sup_GAS_rf', np.hstack([G, A, S]), rf(rep))]:
                model.fit(X[tr], y[tr])
                oof[m][te] = model.predict_proba(X[te])[:, 1]
        for m in METHODS:
            s = oof[m]
            res[m]['auc'].append(float(roc_auc_score(y, s)))
            res[m]['tpr5'].append(tpr_at(0.05, y, s))
            res[m]['tpr10'].append(tpr_at(0.10, y, s))
            res[m]['tpr20'].append(tpr_at(0.20, y, s))
            res[m]['prec10'].append(prec_at(0.10, y, s))
        if rep == 0:
            canon = (oof, patch_sc, patch_thr)

    best = max(METHODS, key=lambda m: np.mean(res[m]['auc']))
    ms = lambda v: (float(np.mean(v)), float(np.std(v)))

    out_dir.mkdir(parents=True, exist_ok=True)
    bench_data = {
        "dataset": dataset_name, "n_ok": n_ok, "n_defect": n_def, "annot_method": best,
        "methods": {m: {k: ms(v) for k, v in res[m].items()} for m in METHODS}
    }
    bench_json = out_dir / f"benchmark_{dataset_name}.json"
    bench_json.write_text(json.dumps(bench_data, indent=2), encoding='utf-8')

    # Lưu benchmark_<dataset>.txt
    txt_lines = [
        f"=== {dataset_name} | {cfg.get('role', dataset_name)} ===",
        f"Anh ok={n_ok}, defect={n_def} | {actual_splits}-fold x {actual_reps} lan lap | {int(np.mean([len(d) for d in descs]))} o/anh trung binh",
        "",
        "Phuong phap       AUC            Recall@FPR5%  Recall@FPR10%  Recall@FPR20%  Precision@FPR10%"
    ]
    for m in METHODS:
        rm = res[m]
        txt_lines.append(
            f"{m:<16}  {ms(rm['auc'])[0]:.3f} +- {ms(rm['auc'])[1]:.3f}   "
            f"{ms(rm['tpr5'])[0]:.2f}          {ms(rm['tpr10'])[0]:.2f}           "
            f"{ms(rm['tpr20'])[0]:.2f}           {ms(rm['prec10'])[0]:.2f}"
        )
    (out_dir / f"benchmark_{dataset_name}.txt").write_text("\n".join(txt_lines), encoding='utf-8')

    # Tạo ảnh annotated_v2 + CSV scores (dùng canon rep 0)
    import csv
    oof, patch_sc, patch_thr = canon
    ann_dir = out_dir / 'annotated_v2' / dataset_name
    ann_dir.mkdir(parents=True, exist_ok=True)
    op_thr = float(np.quantile(oof[best][y == 0], 0.90))
    for i, r in enumerate(recs):
        bad = oof[best][i] > op_thr
        col = (0, 0, 255) if bad else (0, 200, 0)
        out_img = P.draw(r['img'], r, f"{'DEFECT' if bad else 'OK'} {oof[best][i]:.2f}", col)
        if i in patch_sc:
            sc_p, s_, pd_i = patch_sc[i], r['pd']['scale'], r['pd']
            for (px, py), pv in zip(pd_i['xy'], sc_p):
                if pv > patch_thr.get(i, 999.0):
                    hh = int(PT.PATCH / s_ / 2)
                    cx, cy = int(px / s_), int(py / s_)
                    cv2.rectangle(out_img, (cx - hh, cy - hh), (cx + hh, cy + hh), (0, 0, 255), 1)
        cv2.putText(out_img, f"GT:{r['label']}", (5, out_img.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imwrite(str(ann_dir / f"{r['label']}__{r['file']}"), out_img)

    csv_path = out_dir / f"benchmark_{dataset_name}_scores.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['file', 'label'] + METHODS)
        for i, r in enumerate(recs):
            w.writerow([f"{r['label']}__{r['file']}", r['label']] + [f"{oof[m][i]:.4f}" for m in METHODS])

    # Vẽ biểu đồ bench_<dataset>.png
    bench_fig_path = None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        (out_dir / 'figures').mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.5))
        cols = ['tab:gray', 'tab:orange', 'tab:blue', 'tab:cyan', 'tab:green']
        ax[0].bar(range(len(METHODS)), [ms(res[m]['auc'])[0] for m in METHODS],
                  yerr=[ms(res[m]['auc'])[1] for m in METHODS], color=cols, capsize=3)
        ax[0].axhline(0.5, ls='--', c='k', lw=1)
        ax[0].set_xticks(range(len(METHODS)))
        ax[0].set_xticklabels(METHODS, rotation=30, ha='right')
        ax[0].set_ylim(0, 1.05)
        ax[0].set_ylabel('AUC')
        ax[0].set_title(f"{dataset_name} - AUC (kiem tra cheo)")
        for m, c in zip(METHODS, cols):
            f_, t_, _ = roc_curve(y, oof[m])
            ax[1].plot(f_, t_, color=c, label=f"{m} ({roc_auc_score(y, oof[m]):.2f})")
        ax[1].plot([0, 1], [0, 1], 'k--', lw=1)
        ax[1].set_xlabel('Ti le bao nham (FPR)')
        ax[1].set_ylabel('Ti le bat loi (Recall)')
        ax[1].legend(fontsize=8, loc='lower right')
        ax[1].set_title('Duong ROC - lan chia 0')
        plt.tight_layout()
        bench_fig_path = str(out_dir / 'figures' / f"bench_{dataset_name}.png")
        plt.savefig(bench_fig_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        sys.stderr.write(f"[Warning] bench plot error: {e}\n")

    print(json.dumps(convert_types({
        "success": True,
        "step": "benchmark",
        "dataset": dataset_name,
        "result": bench_data,
        "figure_path": bench_fig_path
    }), ensure_ascii=False))


# =======================================================================
# BƯỚC 4: REPORT - Phân tích lỗi FN/FP/TP (report_figs.py) + Table (make_table.py)
# =======================================================================
def _generate_summary_table_md(out_dir):
    """Sinh file summary_all.md giống make_table.py nhưng an toàn với dữ liệu thiếu prf."""
    out = []
    base = sorted(out_dir.glob('summary_*.json'))
    if base:
        out += ['### Co so (dac trung giao trinh, hieu chuan mot lop)', '',
                '| Bo du lieu | ok / defect | AUC hinh dang | AUC be mat | AUC tat ca | Recall (k=3) | FPR (k=3) |',
                '|---|---|---|---|---|---|---|']
        for p in base:
            s = json.loads(p.read_text(encoding='utf-8'))
            a = s.get('auc', {})
            prf_all = s.get('prf', {}).get('all', {})
            rec_str = f"{prf_all['recall']:.2f}" if 'recall' in prf_all else '—'
            fpr_str = f"{prf_all['fpr']:.2f}" if 'fpr' in prf_all else '—'
            out.append('| %s | %d / %d | %.2f | %.2f | %.2f | %s | %s |' % (
                s.get('dataset', p.stem), s.get('n_ok', 0), s.get('n_defect', 0),
                a.get('shape', {}).get('mean', 0), a.get('surface', {}).get('mean', 0),
                a.get('all', {}).get('mean', 0), rec_str, fpr_str))
    bench = sorted(out_dir.glob('benchmark_*.json'))
    if bench:
        out += ['', '### Nang cao (kiem tra cheo): AUC va Recall@FPR10%', '']
        methods = list(json.loads(bench[0].read_text(encoding='utf-8')).get('methods', {}).keys())
        if methods:
            out += ['| Bo du lieu | ' + ' | '.join(methods) + ' |', '|---|' + '---|' * len(methods)]
            for p in bench:
                s = json.loads(p.read_text(encoding='utf-8'))
                cells = [
                    '%.2f / %.2f' % (s['methods'][m]['auc'][0], s['methods'][m]['tpr10'][0])
                    if m in s.get('methods', {}) else '—'
                    for m in methods
                ]
                out.append('| %s | %s |' % (s.get('dataset', p.stem), ' | '.join(cells)))
            out += ['', 'Moi o: AUC / Recall@FPR10%.']
    text = '\n'.join(out)
    (out_dir / 'summary_all.md').write_text(text, encoding='utf-8')
    return str(out_dir / 'summary_all.md')


def cmd_report(args):
    """
    Tương đương report_figs.py + make_table.py:
    đọc benchmark CSV + ảnh annotated_v2, tạo ảnh báo cáo lỗi và bảng summary_all.md.
    """
    import subprocess
    dataset_name = args.dataset_name
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR

    # Kiểm tra file tiên quyết từ bước benchmark
    bench_json = out_dir / f"benchmark_{dataset_name}.json"
    if not bench_json.exists():
        bench_json = OUT_DIR / f"benchmark_{dataset_name}.json"
    if not bench_json.exists():
        print(json.dumps({"success": False,
                          "error": "Chua co ket qua benchmark. Hay chay buoc benchmark truoc."}))
        return

    # Chạy report_figs.py gốc
    script_path = CURRENT_DIR / 'scripts' / 'report_figs.py'
    result = subprocess.run(
        [sys.executable, str(script_path), dataset_name, '--fpr', '0.10', '--n', '6'],
        capture_output=True, text=True, encoding='utf-8', cwd=str(CURRENT_DIR)
    )

    if result.returncode != 0:
        err = (result.stderr or result.stdout or '').strip()
        print(json.dumps({"success": False, "error": f"Report loi: {err[:500]}"}))
        return

    # Sinh bảng tổng hợp summary_all.md (make_table.py)
    table_script = CURRENT_DIR / 'scripts' / 'make_table.py'
    table_res = subprocess.run(
        [sys.executable, str(table_script)],
        capture_output=True, text=True, encoding='utf-8', cwd=str(CURRENT_DIR)
    )
    if table_res.returncode == 0:
        table_path = str(OUT_DIR / 'summary_all.md')
    else:
        table_path = _generate_summary_table_md(OUT_DIR)

    errors_fig = out_dir / 'figures' / f"errors_{dataset_name}.png"
    if not errors_fig.exists():
        errors_fig = OUT_DIR / 'figures' / f"errors_{dataset_name}.png"

    meta = {}
    if bench_json.exists():
        meta = json.loads(bench_json.read_text(encoding='utf-8'))

    output = result.stdout or ""
    stats = {}
    for line in output.split('\n'):
        if 'TP=' in line:
            stats['summary'] = line.strip()

    print(json.dumps(convert_types({
        "success": True,
        "step": "report",
        "dataset": dataset_name,
        "console": output[:3000],
        "errors_figure_path": str(errors_fig) if errors_fig.exists() else None,
        "table_path": table_path,
        "best_method": meta.get("annot_method", ""),
        "stats_summary": stats.get("summary", "")
    }), ensure_ascii=False))


# =======================================================================
# PHÂN LOẠI THỰC TẾ - Dùng sau khi đã qua đủ 4 bước
# =======================================================================
def cmd_inspect_batch(args):
    """
    Phân loại ảnh mới dùng mô hình đã huấn luyện từ bước evaluate/benchmark.
    Nếu chưa có file model .pkl, tự động build từ ảnh OK của dataset đã evaluate.
    """
    dataset_name = args.dataset_name
    cfg = parse_dataset_cfg(dataset_name, args.cfg)
    data_dir = find_data_dir(dataset_name, args.data_dir)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold_k = float(args.threshold_k) if args.threshold_k else 3.0

    if not images_dir.exists():
        print(json.dumps({"success": False, "error": f"Khong tim thay thu muc anh: {images_dir}"}))
        return

    # Lấy hoặc build calibration từ ảnh OK của dataset gốc
    model_path = Path(args.model_path) if args.model_path else None
    if model_path and model_path.exists():
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        calibration = model_data['calibration']
        ref_hist = model_data['ref_hist']
        bank = None
        if model_data.get('patch_bank_path') and Path(model_data['patch_bank_path']).exists():
            with open(model_data['patch_bank_path'], 'rb') as f:
                bank = pickle.load(f)
    elif data_dir and data_dir.exists():
        # Build trực tiếp từ ảnh OK trong data_dir (đúng với triết lý one-class)
        ok_items = [(p, l) for p, l in list_images_from_dir(data_dir, cfg.get('exts')) if l == 'ok']
        if len(ok_items) < 5:
            print(json.dumps({"success": False,
                              "error": "Khong du anh OK de xay dung calibration. Can >= 5 anh OK."}))
            return

        ok_recs = []
        ok_descs = []
        for path, _ in ok_items:
            try:
                img = P.load(path, cfg.get('crop', 0))
                r = P.extract(img, cfg)
                if r is None:
                    continue
                pd = PT.patch_descriptors(img, r)
                if pd is None:
                    continue
                ok_recs.append(r)
                ok_descs.append(pd['desc'])
            except Exception:
                pass

        F_ok = {f: np.array([r[f] for r in ok_recs]) for f in BASE_FEATS}
        H_ok = np.array([r['hist'] for r in ok_recs], np.float32)
        calibration = {}
        for f in BASE_FEATS:
            ref = F_ok[f]
            med = float(np.median(ref))
            mad = float(np.median(np.abs(ref - med)))
            scale = float(max(1.4826 * mad, 0.5 * ref.std(), 1e-3 * abs(med), 1e-9))
            calibration[f] = {"median": med, "scale": scale, "direction": DIRECTION[f]}

        ref_hist = H_ok.mean(0).astype(np.float32)
        hd_ok = np.array([float(cv2.compareHist(H_ok[i], ref_hist, cv2.HISTCMP_BHATTACHARYYA))
                          for i in range(len(H_ok))], dtype=np.float32)
        hd_med = float(np.median(hd_ok))
        hd_scale = float(max(1.4826 * np.median(np.abs(hd_ok - hd_med)), 0.5 * np.std(hd_ok), 1e-6))
        calibration['hist_dist'] = {"median": hd_med, "scale": hd_scale, "direction": 1}

        bank = PT.PatchBank(k=3).fit(ok_descs) if ok_descs else None
    else:
        print(json.dumps({"success": False,
                          "error": "Khong co model va khong tim thay du lieu goc de xay dung calibration."}))
        return

    # Phân loại từng ảnh
    img_exts = cfg.get('exts', ('.png', '.jpg', '.jpeg'))
    image_files = sorted([f for f in images_dir.iterdir()
                          if f.is_file() and f.suffix.lower() in img_exts])

    if not image_files:
        print(json.dumps({"success": False, "error": "Khong co anh nao trong thu muc can kiem tra."}))
        return

    results = []
    ok_count = defect_count = 0

    for fpath in image_files:
        try:
            img = P.load(fpath, cfg.get('crop', 0))
            r = P.extract(img, cfg)
            if r is None:
                results.append({"filename": fpath.name, "prediction": "error",
                                 "error": "Phan vung that bai", "annotated_path": None,
                                 "anomaly_score": None, "features": {}})
                continue

            # Tính z-score
            z_scores = {}
            for f in BASE_FEATS:
                c = calibration[f]
                z_scores[f] = float(c['direction'] * (r[f] - c['median']) / c['scale'])

            hist_d = float(cv2.compareHist(r['hist'], ref_hist, cv2.HISTCMP_BHATTACHARYYA))
            c_h = calibration['hist_dist']
            z_scores['hist_dist'] = float((hist_d - c_h['median']) / c_h['scale'])
            max_z = float(max(z_scores.values()))

            # Patch anomaly
            p_summary = [0.0, 0.0, 0.0, 0.0]
            patch_scores = None
            pd = PT.patch_descriptors(img, r)
            if pd and bank:
                patch_scores = bank.score(pd['desc'])
                p_summary = PT.summarize(patch_scores)

            is_defect = max_z > threshold_k
            pred_label = "defect" if is_defect else "ok"
            if is_defect:
                defect_count += 1
            else:
                ok_count += 1

            # Vẽ ảnh annotated
            color = (0, 0, 255) if is_defect else (0, 200, 0)
            tag = f"{pred_label.upper()} z={max_z:.2f}"
            ann_img = P.draw(img, r, text=tag, color=color)

            # Khoanh vùng ô bất thường nếu có
            if pd and patch_scores is not None:
                p_thr = float(np.percentile(patch_scores, 95))
                for (px, py), psc in zip(pd['xy'], patch_scores):
                    if psc >= p_thr and psc > 1.5:
                        h = int(PT.PATCH / pd['scale'] / 2)
                        cx, cy = int(px / pd['scale']), int(py / pd['scale'])
                        cv2.rectangle(ann_img, (cx - h, cy - h), (cx + h, cy + h), (0, 0, 255), 1)

            ann_name = f"annotated_{fpath.name}"
            ann_path = out_dir / ann_name
            cv2.imwrite(str(ann_path), ann_img)

            results.append({
                "filename": fpath.name,
                "annotated_path": str(ann_path),
                "annotated_filename": ann_name,
                "prediction": pred_label,
                "anomaly_score": max_z,
                "features": {
                    "solidity": float(r['solidity']),
                    "circularity": float(r['circularity']),
                    "edge_density": float(r['edge_density']),
                    "sobel_mean": float(r['sobel_mean']),
                    "gray_std": float(r['gray_std']),
                    "hist_dist": hist_d,
                    "patch_max": float(p_summary[0]),
                    "patch_top5": float(p_summary[1]),
                    "patch_p90": float(p_summary[2]),
                    "patch_mean": float(p_summary[3])
                },
                "z_scores": z_scores
            })
        except Exception as e:
            results.append({"filename": fpath.name, "prediction": "error",
                             "error": str(e), "annotated_path": None,
                             "anomaly_score": None, "features": {}})

    print(json.dumps(convert_types({
        "success": True,
        "step": "inspect",
        "dataset": dataset_name,
        "total_images": len(image_files),
        "ok_count": ok_count,
        "defect_count": defect_count,
        "threshold_used": threshold_k,
        "results": results
    }), ensure_ascii=False))


# =======================================================================
# MODULE MỚI: DEFECT QC — Phát hiện sứt mẻ / vỡ / khuyết tật cấu trúc
# Độc lập hoàn toàn, không ảnh hưởng pipeline cũ.
# =======================================================================

from src import defect_qc as DQ   # import ở đây để lazy (không load khi dùng các lệnh cũ)

_DEFECT_EXTS = ('.png', '.jpg', '.jpeg')


def _dq_list_images(folder: Path) -> list:
    """Liệt kê ảnh trong 1 thư mục phẳng."""
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in _DEFECT_EXTS)


def _dq_parse_cfg(args) -> dict:
    obj_bright = str(getattr(args, 'object_bright', 'true')).lower() not in ('false', '0', 'no')
    return dict(
        channel=getattr(args, 'channel', 'gray'),
        object_bright=obj_bright
    )


# -----------------------------------------------------------------------
# defect_scan: Quét toàn bộ tập dữ liệu (ok/ + defect/) → phân tích + JSON
# -----------------------------------------------------------------------
def cmd_defect_scan(args):
    """
    Quét toàn bộ dataset có cấu trúc: data_dir/ok/ và data_dir/defect/
    Trả JSON kết quả gồm: metrics per image, AUC, confusion stats, ảnh annotated.
    """
    cfg = _dq_parse_cfg(args)
    threshold_k = float(args.threshold_k)
    reps = int(args.reps)
    data_dir = Path(args.data_dir)
    session_name = args.session_name or data_dir.name

    ok_dir = data_dir / 'ok'
    defect_dir = data_dir / 'defect'

    # Thư mục lưu ảnh annotated
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = OUT_DIR / 'defect_qc' / session_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = out_dir / 'annotated'
    ann_dir.mkdir(parents=True, exist_ok=True)

    ok_files = _dq_list_images(ok_dir)
    defect_files = _dq_list_images(defect_dir)
    if not ok_files:
        print(json.dumps({"success": False, "error": f"Không tìm thấy ảnh OK trong: {ok_dir}"}))
        return

    # --- Trích đặc trưng tất cả ảnh ---
    recs = []
    skipped = []
    for path, label in [(f, 'ok') for f in ok_files] + [(f, 'defect') for f in defect_files]:
        bgr = cv2.imread(str(path))
        if bgr is None:
            skipped.append(str(path.name))
            continue
        m = DQ.extract(bgr, cfg['channel'], cfg['object_bright'])
        if m is None:
            skipped.append(str(path.name))
            continue
        m.update(path=path, file=path.name, label=label)
        recs.append(m)

    if len([r for r in recs if r['label'] == 'ok']) < 3:
        print(json.dumps({"success": False, "error": "Cần ít nhất 3 ảnh OK để hiệu chuẩn."}))
        return

    n = len(recs)
    labels = np.array([1 if r['label'] == 'defect' else 0 for r in recs])
    ok_idx = np.where(labels == 0)[0]
    def_idx = np.where(labels == 1)[0]

    feats_dict = {f: np.array([r[f] for r in recs], dtype=np.float64) for f in DQ.DEFECT_FEATS}

    # --- Lặp nhiều lần chia ngẫu nhiên (one-class) ---
    n_cal = max(3, int(0.6 * len(ok_idx)))
    auc_runs = []
    for rep in range(reps):
        rng = np.random.default_rng(rep)
        cal = rng.choice(ok_idx, n_cal, replace=False)
        test_ok = np.setdiff1d(ok_idx, cal)
        scores = DQ.maxz_scores_defect(feats_dict, cal)
        s_def = scores[def_idx]
        s_ok_test = scores[test_ok]
        if len(s_def) > 0 and len(s_ok_test) > 0:
            d, o = s_def[:, None], s_ok_test[None, :]
            auc_val = float(((d > o).sum() + 0.5 * (d == o).sum()) / (d.size * o.size + 1e-9))
            auc_runs.append(auc_val)

    auc_mean = float(np.mean(auc_runs)) if auc_runs else 0.5
    auc_std = float(np.std(auc_runs)) if auc_runs else 0.0

    # --- Chấm điểm cuối (dùng toàn bộ OK làm calibration) ---
    final_cal = ok_idx
    final_scores = DQ.maxz_scores_defect(feats_dict, final_cal)

    # --- Vẽ ảnh annotated & gom kết quả ---
    results = []
    for i, r in enumerate(recs):
        sc = float(final_scores[i])
        pred = 'defect' if sc > threshold_k else 'ok'
        is_bad = pred == 'defect'
        color = (0, 0, 220) if is_bad else (0, 200, 0)
        txt = f"{'DEFECT' if is_bad else 'OK'} z={sc:.1f}"
        bgr = cv2.imread(str(r['path']))
        ann_img = DQ.draw(bgr, r, txt, color)
        ann_fname = f"{r['label']}__{r['file']}"
        cv2.imwrite(str(ann_dir / ann_fname), ann_img)
        results.append({
            "file": r['file'],
            "label": r['label'],
            "pred": pred,
            "score": round(sc, 4),
            "annotated": str(ann_dir / ann_fname),
            "annotated_url_rel": f"annotated/{ann_fname}",
            **{f: round(float(r[f]), 5) for f in DQ.DEFECT_FEATS}
        })

    # Thống kê
    res_ok = [x for x in results if x['label'] == 'ok']
    res_def = [x for x in results if x['label'] == 'defect']
    tp = sum(1 for x in res_def if x['pred'] == 'defect')
    fn = sum(1 for x in res_def if x['pred'] == 'ok')
    fp = sum(1 for x in res_ok  if x['pred'] == 'defect')
    tn = sum(1 for x in res_ok  if x['pred'] == 'ok')
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    print(json.dumps({
        "success": True,
        "session_name": session_name,
        "n_ok": len(ok_idx),
        "n_defect": len(def_idx),
        "n_total": n,
        "skipped": skipped,
        "auc_mean": round(auc_mean, 4),
        "auc_std": round(auc_std, 4),
        "threshold_k": threshold_k,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "annotated_dir": str(ann_dir),
        "out_dir": str(out_dir),
        "results": results
    }, ensure_ascii=False))


# -----------------------------------------------------------------------
# defect_inspect: Phân loại batch ảnh mới (chưa có nhãn) từ ảnh OK chuẩn
# -----------------------------------------------------------------------
def cmd_defect_inspect(args):
    """
    Phân loại hàng loạt ảnh mới (không biết nhãn) dựa trên ảnh OK chuẩn.
    ok_dir: thư mục ảnh OK để hiệu chuẩn
    images_dir: thư mục ảnh cần phân loại
    out_dir: lưu ảnh annotated
    """
    cfg = _dq_parse_cfg(args)
    threshold_k = float(args.threshold_k)
    ok_dir = Path(args.ok_dir)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok_files = _dq_list_images(ok_dir)
    if len(ok_files) < 3:
        print(json.dumps({"success": False, "error": "Cần ít nhất 3 ảnh OK để hiệu chuẩn."}))
        return

    # --- Hiệu chuẩn từ ảnh OK ---
    ok_feats = []
    skipped_ok = []
    for f in ok_files:
        bgr = cv2.imread(str(f))
        if bgr is None:
            skipped_ok.append(f.name)
            continue
        m = DQ.extract(bgr, cfg['channel'], cfg['object_bright'])
        if m is None:
            skipped_ok.append(f.name)
            continue
        ok_feats.append(m)

    if len(ok_feats) < 3:
        print(json.dumps({"success": False, "error": "Không đủ ảnh OK có thể xử lý được để hiệu chuẩn."}))
        return

    cal = DQ.calibrate(ok_feats)

    # --- Phân loại ảnh mới ---
    new_files = _dq_list_images(images_dir)
    results = []
    skipped_new = []
    for f in new_files:
        bgr = cv2.imread(str(f))
        if bgr is None:
            skipped_new.append(f.name)
            continue
        m = DQ.extract(bgr, cfg['channel'], cfg['object_bright'])
        if m is None:
            skipped_new.append(f.name)
            continue
        sc = DQ.score_one(m, cal)
        pred = 'defect' if sc > threshold_k else 'ok'
        is_bad = pred == 'defect'
        color = (0, 0, 220) if is_bad else (0, 200, 0)
        txt = f"{'DEFECT' if is_bad else 'OK'} z={sc:.1f}"
        ann_img = DQ.draw(bgr, m, txt, color)
        ann_fname = f"{f.stem}_annotated{f.suffix}"
        ann_path = out_dir / ann_fname
        cv2.imwrite(str(ann_path), ann_img)
        results.append({
            "file": f.name,
            "pred": pred,
            "score": round(float(sc), 4),
            "annotated_path": str(ann_path),
            "annotated_fname": ann_fname,
            **{feat: round(float(m[feat]), 5) for feat in DQ.DEFECT_FEATS}
        })

    ok_count = sum(1 for r in results if r['pred'] == 'ok')
    defect_count = sum(1 for r in results if r['pred'] == 'defect')

    print(json.dumps({
        "success": True,
        "total": len(results),
        "ok_count": ok_count,
        "defect_count": defect_count,
        "threshold_k": threshold_k,
        "skipped": skipped_new + skipped_ok,
        "results": results
    }, ensure_ascii=False))


# =======================================================================
# MAIN
# =======================================================================
def main():
    parser = argparse.ArgumentParser(description="Web Bridge CLI - He thong phat hien khuyet tat hinh dang")
    sub = parser.add_subparsers(dest="command", required=True)

    # Bước 1: explore
    p1 = sub.add_parser("explore")
    p1.add_argument("--dataset_name", required=True)
    p1.add_argument("--data_dir", default=None)
    p1.add_argument("--cfg", default=None)
    p1.add_argument("--out_dir", default=None)
    p1.add_argument("--n_per_class", default=4)

    # Bước 2: evaluate
    p2 = sub.add_parser("evaluate")
    p2.add_argument("--dataset_name", required=True)
    p2.add_argument("--data_dir", default=None)
    p2.add_argument("--cfg", default=None)
    p2.add_argument("--out_dir", default=None)
    p2.add_argument("--cal_ratio", default=0.6)
    p2.add_argument("--reps", default=30)
    p2.add_argument("--threshold_k", default=3.0)

    # Bước 3: benchmark
    p3 = sub.add_parser("benchmark")
    p3.add_argument("--dataset_name", required=True)
    p3.add_argument("--data_dir", default=None)
    p3.add_argument("--cfg", default=None)
    p3.add_argument("--out_dir", default=None)
    p3.add_argument("--splits", default=5)
    p3.add_argument("--reps", default=10)

    # Bước 4: report
    p4 = sub.add_parser("report")
    p4.add_argument("--dataset_name", required=True)
    p4.add_argument("--out_dir", default=None)

    # Phân loại thực tế (sau khi đã qua đủ 4 bước)
    p5 = sub.add_parser("inspect_batch")
    p5.add_argument("--dataset_name", required=True)
    p5.add_argument("--images_dir", required=True)
    p5.add_argument("--output_dir", required=True)
    p5.add_argument("--model_path", default=None)
    p5.add_argument("--data_dir", default=None)
    p5.add_argument("--cfg", default=None)
    p5.add_argument("--threshold_k", default=None)

    # ── MODULE MỚI: Defect QC (vật thể khép kín, sứt mẻ) ──────────────────
    p_dscan = sub.add_parser("defect_scan")
    p_dscan.add_argument("--data_dir",      required=True, help="Thư mục chứa ok/ và defect/")
    p_dscan.add_argument("--out_dir",       default=None)
    p_dscan.add_argument("--channel",       default="gray",  choices=["gray", "sat", "val"])
    p_dscan.add_argument("--object_bright", default="true")
    p_dscan.add_argument("--threshold_k",   default="3.0",   type=str)
    p_dscan.add_argument("--reps",          default="20",    type=str)
    p_dscan.add_argument("--session_name",  default=None)

    p_dinsp = sub.add_parser("defect_inspect")
    p_dinsp.add_argument("--images_dir",   required=True, help="Thư mục chứa ảnh cần phân tích")
    p_dinsp.add_argument("--ok_dir",       required=True, help="Thư mục ảnh OK chuẩn để hiệu chuẩn")
    p_dinsp.add_argument("--out_dir",      required=True, help="Thư mục lưu ảnh annotated")
    p_dinsp.add_argument("--channel",      default="gray",  choices=["gray", "sat", "val"])
    p_dinsp.add_argument("--object_bright",default="true")
    p_dinsp.add_argument("--threshold_k",  default="3.0",   type=str)

    args = parser.parse_args()
    cmds = {
        "explore": cmd_explore,
        "evaluate": cmd_evaluate,
        "benchmark": cmd_benchmark,
        "report": cmd_report,
        "inspect_batch": cmd_inspect_batch,
        # module mới (ĐỘC LẬP, không đụng pipeline cũ)
        "defect_scan": cmd_defect_scan,
        "defect_inspect": cmd_defect_inspect,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
