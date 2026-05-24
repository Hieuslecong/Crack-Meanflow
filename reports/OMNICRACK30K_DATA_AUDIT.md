# OmniCrack30K Data Audit

Image dir: `/home/hieulc/avitech_13/omnicrack30k_data/images/training`
Mask dir: `/home/hieulc/avitech_13/omnicrack30k_data/annotations_resize`
Matched pairs: **29884**
Images: 29884 | Masks: 29884
Unmatched images: 0 | Unmatched masks: 0

## Mask summary
- Empty masks: 4064
- Nonempty masks: 25820
- Near-empty positive <=0.001: 1964
- Empty ratio: 0.1360
- Unique values: `{0: 29884, 255: 25820}`
- Bad mask value examples: 0

## Ratio stats
- min: 0.000000
- p01: 0.000000
- p05: 0.000000
- p50: 0.008865
- p95: 0.054733
- p99: 0.102572
- max: 0.477539
- mean: 0.016268

## Modes / sizes
- Image modes: `{'L': 1994, 'LA': 1, 'P': 77, 'RGB': 27712, 'RGBA': 100}`
- Mask modes: `{'L': 29752, 'RGB': 76, 'RGBA': 56}`
- Top image sizes: `{'256x256': 18810, '448x448': 4904, '1024x1024': 1076, '512x512': 490, '544x384': 472, '2560x1440': 428, '1920x1080': 384, '224x224': 240, '800x600': 223, '768x768': 186, '1000x750': 169, '480x320': 120, '1002x752': 105, '640x480': 88, '960x720': 65, '1002x751': 54, '1002x750': 48, '384x544': 47, '3264x2448': 42, '1600x1200': 41}`
- Top mask sizes: `{'256x256': 29884}`

## Source prefixes
| Prefix | Count | Empty |
|---|---:|---:|
| BCL | 11000 | 3195 |
| TopoDS | 7180 | 0 |
| Khanh11k | 4904 | 49 |
| CrSpEE | 1203 | 0 |
| LCW | 1145 | 245 |
| S2DS | 743 | 511 |
| DIC | 530 | 0 |
| DeepCrack | 519 | 0 |
| CRACK500 | 493 | 0 |
| GAPS384 | 384 | 2 |
| Stone331 | 331 | 0 |
| CrackLS315 | 315 | 5 |
| CrackTree260 | 260 | 0 |
| Masonry | 240 | 0 |
| CSSC | 186 | 56 |
| CFD | 118 | 0 |
| CRKWH100 | 100 | 0 |
| Ceramic | 100 | 0 |
| UAV75 | 75 | 0 |
| AEL | 58 | 1 |

## Samples
Montage: `outputs/omnicrack30k/audit_montage.png`
