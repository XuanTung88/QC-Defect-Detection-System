"""Gop ket qua thanh bang cho bao cao. Chay sau khi da evaluate.py va/hoac benchmark.py."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUT_DIR

out = []
base = sorted(OUT_DIR.glob('summary_*.json'))
if base:
    out += ['### Co so (dac trung giao trinh, hieu chuan mot lop)', '',
            '| Bo du lieu | ok / defect | AUC hinh dang | AUC be mat | AUC tat ca | Recall (k=3) | FPR (k=3) |',
            '|---|---|---|---|---|---|---|']
    for p in base:
        s = json.loads(p.read_text(encoding='utf-8')); a = s['auc']
        out.append('| %s | %d / %d | %.2f | %.2f | %.2f | %.2f | %.2f |' % (
            s['dataset'], s['n_ok'], s['n_defect'], a['shape']['mean'], a['surface']['mean'], a['all']['mean'],
            s['prf']['all']['recall'], s['prf']['all']['fpr']))
bench = sorted(OUT_DIR.glob('benchmark_*.json'))
if bench:
    out += ['', '### Nang cao (kiem tra cheo): AUC va Recall@FPR10%', '']
    methods = list(json.loads(bench[0].read_text())['methods'])
    out += ['| Bo du lieu | ' + ' | '.join(methods) + ' |', '|---|' + '---|' * len(methods)]
    for p in bench:
        s = json.loads(p.read_text())
        cells = ['%.2f / %.2f' % (s['methods'][m]['auc'][0], s['methods'][m]['tpr10'][0]) for m in methods]
        out.append('| %s | %s |' % (s['dataset'], ' | '.join(cells)))
    out += ['', 'Moi o: AUC / Recall@FPR10%.']
text = '\n'.join(out)
print(text)
(OUT_DIR / 'summary_all.md').write_text(text, encoding='utf-8')
