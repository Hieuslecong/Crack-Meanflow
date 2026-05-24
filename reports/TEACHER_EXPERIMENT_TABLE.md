# TEACHER_EXPERIMENT_TABLE

Diagnostic supervised teacher only. Not CrackMeanFlow main metric.

| Run | Loss | Img | Batch | Epochs | Best val F1/Dice | Val th | Test F1/Dice | IoU | Precision | Recall | Status | Artifacts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| TEACHER01_BCE_DICE_sanity | BCE+Dice | 128 | 8 | 3 | 0.553852 | 0.600 | 0.541304 | 0.371087 | 0.535826 | 0.546895 | PASS sanity | `outputs/teacher/TEACHER01_BCE_DICE_sanity/metrics.json` |
| TEACHER01_BCE_DICE_full | BCE+Dice | 128 | 8 | 30 | 0.634659 | 0.500 | 0.648507 | 0.479845 | 0.631968 | 0.665936 | PASS teacher gate >=0.60 | `outputs/teacher/TEACHER01_BCE_DICE_full/metrics.json` |

Decision:
- Teacher upper-bound gate passed: test F1/Dice `0.648507` >= `0.60`.
- Continue to MF endpoint calibration.
- Teacher < `0.70`; teacher distillation optional later, not immediate.
