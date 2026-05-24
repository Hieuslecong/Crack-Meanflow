# OmniCrack30K Split Report

Seed: 42
Subset size: 5000
Policy: stratified random by `prefix + empty/nonempty`, ratios 70/15/15.

| Split | Count | Empty | Nonempty |
|---|---:|---:|---:|
| train | 3502 | 476 | 3026 |
| val | 753 | 102 | 651 |
| test | 745 | 101 | 644 |

## Files
- `splits/omnicrack30k/omnicrack30k_5k_train.csv`
- `splits/omnicrack30k/omnicrack30k_5k_val.csv`
- `splits/omnicrack30k/omnicrack30k_5k_test.csv`
- `splits/omnicrack30k/omnicrack30k_5k_all.csv`

## Prefix counts
### train
{'BCL': 1289, 'TopoDS': 841, 'Khanh11k': 574, 'CrSpEE': 141, 'LCW': 135, 'S2DS': 86, 'DIC': 62, 'DeepCrack': 61, 'CRACK500': 58, 'GAPS384': 45, 'Stone331': 39, 'CrackLS315': 37, 'CrackTree260': 30, 'Masonry': 28, 'CSSC': 22, 'CFD': 14, 'CRKWH100': 12, 'Ceramic': 12, 'UAV75': 9, 'AEL': 7}

### val
{'BCL': 276, 'TopoDS': 180, 'Khanh11k': 123, 'CrSpEE': 30, 'LCW': 29, 'S2DS': 19, 'DIC': 14, 'DeepCrack': 13, 'CRACK500': 12, 'GAPS384': 10, 'CrackLS315': 8, 'Stone331': 8, 'CrackTree260': 7, 'Masonry': 6, 'CSSC': 5, 'CFD': 3, 'CRKWH100': 3, 'Ceramic': 3, 'AEL': 2, 'UAV75': 2}

### test
{'BCL': 276, 'TopoDS': 180, 'Khanh11k': 123, 'CrSpEE': 30, 'LCW': 28, 'S2DS': 19, 'DIC': 13, 'DeepCrack': 13, 'CRACK500': 12, 'GAPS384': 9, 'CrackLS315': 8, 'Stone331': 8, 'CrackTree260': 6, 'Masonry': 6, 'CSSC': 4, 'CFD': 3, 'CRKWH100': 2, 'Ceramic': 2, 'UAV75': 2, 'AEL': 1}
