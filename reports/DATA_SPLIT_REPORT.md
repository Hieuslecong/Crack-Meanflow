# CrackMeanFlow Data Split Report

Image dir: `/home/hieulc/avitech_13/omnicrack30k_data/images/training`
Mask dir: `/home/hieulc/avitech_13/omnicrack30k_data/annotations_resize`
Matched pairs: 5000
Train pairs: 3502
Val pairs: 753
Test pairs: 745
Split policy: configured_csv_split
Seed: 42

## Leakage audit
- train contains `_test_`: 4
- test contains `_train_`: 9
- val contains `_test_`: 0
- val contains `_train_`: 10

## First train names
- AEL_Im_GT_AIGLE_RN_F04aor
- AEL_Im_GT_AIGLE_RN_F06bor
- AEL_Im_GT_AIGLE_RN_F08aor
- AEL_Im_GT_AIGLE_RN_F10aor
- AEL_Im_GT_AIGLE_RN_F15aor
- AEL_Im_GT_ESAR_34a
- AEL_Im_GT_LCMS_23aor
- BCL_c1003
- BCL_c1026
- BCL_c1028

## First val names
- AEL_Im_GT_AIGLE_RN_F16bor
- AEL_Im_GT_LCMS_38cor
- BCL_c1061
- BCL_c1075
- BCL_c1078
- BCL_c108
- BCL_c1191
- BCL_c1215
- BCL_c1220
- BCL_c1232

## First test names
- AEL_Im_GT_AIGLE_RN_F08bor
- BCL_c1081
- BCL_c1103
- BCL_c1127
- BCL_c1209
- BCL_c1249
- BCL_c1253
- BCL_c1288
- BCL_c1303
- BCL_c1315
