### Co so (dac trung giao trinh, hieu chuan mot lop)

| Bo du lieu | ok / defect | AUC hinh dang | AUC be mat | AUC tat ca | Recall (k=3) | FPR (k=3) |
|---|---|---|---|---|---|---|
| biscuits | 48 / 67 | 0.87 | 0.84 | 0.93 | — | — |
| hazelnut | 22 / 17 | 0.60 | 0.85 | 0.83 | — | — |
| pills | 26 / 26 | 0.87 | 0.60 | 0.88 | 0.81 | 0.16 |

### Nang cao (kiem tra cheo): AUC va Recall@FPR10%

| Bo du lieu | baseline_maxz | patch_knn | sup_G_lr | sup_GS_lr | sup_GAS_rf |
|---|---|---|---|---|---|
| biscuits | 0.90 / 0.67 | 0.86 / 0.71 | 0.76 / 0.55 | 0.85 / 0.70 | 0.93 / 0.81 |
| hazelnut | 0.83 / 0.58 | 0.98 / 0.91 | 0.83 / 0.66 | 0.95 / 0.87 | 0.96 / 0.86 |
| pills | 0.88 / 0.76 | 0.56 / 0.09 | 0.89 / 0.77 | 0.89 / 0.77 | 0.88 / 0.80 |

Moi o: AUC / Recall@FPR10%.