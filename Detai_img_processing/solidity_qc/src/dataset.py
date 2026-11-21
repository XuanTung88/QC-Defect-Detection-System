from pathlib import Path
from .config import DATA_ROOT, DATASETS, OK_NAMES, DEFECT_NAMES


def list_images(name):
    """Tra ve [(path, 'ok'|'defect')] cho 1 bo du lieu."""
    cfg = DATASETS[name]
    root = DATA_ROOT / name
    items = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        key = sub.name.lower()
        if key in OK_NAMES:
            label = 'ok'
        elif key in DEFECT_NAMES:
            label = 'defect'
        else:
            continue
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() in cfg['exts']:
                items.append((f, label))
    return items
