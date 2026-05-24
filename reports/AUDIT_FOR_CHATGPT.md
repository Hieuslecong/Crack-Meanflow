# AUDIT FOR CHATGPT — CrackMeanFlow

## 1. Tổng quan implementation
- Mục tiêu: tích hợp CrackDiff multi-task UNet vào MeanFlow để crack segmentation 1-step; giữ segmentation branch; huấn luyện hybrid loss.
- `CrackMeanFlowModel`: adapter `forward(x,r,t,y)`; map continuous `t` sang timestep int; gọi CrackDiff `UNet`; unpack `velocity_pred, seg_logits`; cache seg logits; return velocity/noise prediction.
- Có giữ UNet CrackDiff: có, `multi_task.mlt_unet.UNet`.
- SiT/VAE/LMDB: không dùng trong implementation này.
- MeanFlow tích hợp: `loss.py` dùng `SILoss`; `sampler.py` suy luận MeanFlow; train/test gọi adapter+loss+sampler.

## 2. Cây thư mục hiện tại
```
/home/hieulc/avitech11/crackmeanflow/checkpoints/best.pt
/home/hieulc/avitech11/crackmeanflow/checkpoints/last.pt
/home/hieulc/avitech11/crackmeanflow/checkpoints_long_endpoint/best.pt
/home/hieulc/avitech11/crackmeanflow/checkpoints_long_endpoint/last.pt
/home/hieulc/avitech11/crackmeanflow/checkpoints_v2/best.pt
/home/hieulc/avitech11/crackmeanflow/checkpoints_v2/last.pt
/home/hieulc/avitech11/crackmeanflow/checkpoints_v3/best.pt
/home/hieulc/avitech11/crackmeanflow/checkpoints_v3/last.pt
/home/hieulc/avitech11/crackmeanflow/checkpoints_v4_256/best.pt
/home/hieulc/avitech11/crackmeanflow/checkpoints_v4_256/last.pt
/home/hieulc/avitech11/crackmeanflow/checkpoints_v5_256_ft/best.pt
/home/hieulc/avitech11/crackmeanflow/checkpoints_v5_256_ft/last.pt
/home/hieulc/avitech11/crackmeanflow/.claude/settings.local.json
/home/hieulc/avitech11/crackmeanflow/.codegraph/codegraph.db
/home/hieulc/avitech11/crackmeanflow/.codegraph/codegraph.db-shm
/home/hieulc/avitech11/crackmeanflow/.codegraph/codegraph.db-wal
/home/hieulc/avitech11/crackmeanflow/.codegraph/config.json
/home/hieulc/avitech11/crackmeanflow/.codegraph/.gitignore
/home/hieulc/avitech11/crackmeanflow/configs/crackmeanflow_default.yaml
/home/hieulc/avitech11/crackmeanflow/configs/crackmeanflow_long_endpoint.yaml
/home/hieulc/avitech11/crackmeanflow/configs/crackmeanflow_v2.yaml
/home/hieulc/avitech11/crackmeanflow/configs/crackmeanflow_v3.yaml
/home/hieulc/avitech11/crackmeanflow/configs/crackmeanflow_v4_256.yaml
/home/hieulc/avitech11/crackmeanflow/configs/crackmeanflow_v5_256_ft.yaml
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/adapter.py
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/checkpointing.py
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/data.py
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/direct_unet.py
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/__init__.py
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/loss.py
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/metrics.py
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/model.py
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/paths.py
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/__pycache__/adapter.cpython-37.pyc
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/__pycache__/checkpointing.cpython-37.pyc
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/__pycache__/data.cpython-37.pyc
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/__pycache__/direct_unet.cpython-37.pyc
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/__pycache__/__init__.cpython-37.pyc
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/__pycache__/loss.cpython-37.pyc
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/__pycache__/metrics.cpython-37.pyc
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/__pycache__/paths.cpython-37.pyc
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/__pycache__/sampler.cpython-37.pyc
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/__pycache__/test.cpython-37.pyc
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/__pycache__/thin_metrics.cpython-37.pyc
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/__pycache__/train.cpython-37.pyc
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/sampler.py
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/test.py
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/thin_metrics.py
/home/hieulc/avitech11/crackmeanflow/crackmeanflow/train.py
/home/hieulc/avitech11/crackmeanflow/logs/train_last.log
/home/hieulc/avitech11/crackmeanflow/logs/train_stdout.log
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/smoke_ckpt.pt
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1361_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1362_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1362_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1364_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1365_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1365_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1366_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1370_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1370_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1370_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1374_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1374_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1374_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1375_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1375_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1380_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1381_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1381_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1386_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1386_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1389_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1389_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1403_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_train_1403_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_valid_0009_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_valid_0016_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/predictions/GAPS384_valid_0016_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/TEST_REPORT.md
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/metrics.json
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1171_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1172_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1172_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1173_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1173_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1174_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1174_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1175_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1175_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1176_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1176_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1177_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1178_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1179_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1180_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1180_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1181_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1181_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1182_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1182_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1183_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1187_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1190_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1190_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1201_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1218_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1226_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1226_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1227_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1230_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1238_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1238_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1241_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1241_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1242_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1246_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1247_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1247_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1248_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1248_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1248_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1281_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1281_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1284_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1284_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1319_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1319_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1319_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1320_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1321_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1324_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1336_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1337_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1345_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1345_541_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1347_1_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1347_1_641.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/predictions/GAPS384_train_1347_541_1.png
/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/predictions/GAPS384_train_1361_541_641.png
/h
```
## 3. Danh sách file đã tạo
- `/home/hieulc/avitech11/crackmeanflow/crackmeanflow/adapter.py`
  - vai trò: adapter UNet->MeanFlow
  - trạng thái: hoàn thành
- `/home/hieulc/avitech11/crackmeanflow/crackmeanflow/loss.py`
  - vai trò: SILoss+seg+endpoint
  - trạng thái: hoàn thành
- `/home/hieulc/avitech11/crackmeanflow/crackmeanflow/sampler.py`
  - vai trò: sampler 1/multi-step
  - trạng thái: hoàn thành
- `/home/hieulc/avitech11/crackmeanflow/crackmeanflow/metrics.py`
  - vai trò: metrics segmentation
  - trạng thái: hoàn thành
- `/home/hieulc/avitech11/crackmeanflow/crackmeanflow/thin_metrics.py`
  - vai trò: thin metrics
  - trạng thái: hoàn thành
- `/home/hieulc/avitech11/crackmeanflow/crackmeanflow/checkpointing.py`
  - vai trò: checkpoint utils
  - trạng thái: hoàn thành
- `/home/hieulc/avitech11/crackmeanflow/crackmeanflow/train.py`
  - vai trò: train loop
  - trạng thái: hoàn thành
- `/home/hieulc/avitech11/crackmeanflow/crackmeanflow/test.py`
  - vai trò: eval loop
  - trạng thái: hoàn thành
- `/home/hieulc/avitech11/crackmeanflow/scripts/smoke_test.py`
  - vai trò: smoke test
  - trạng thái: hoàn thành source; chưa có stdout artifact
- `/home/hieulc/avitech11/crackmeanflow/scripts/train_crackmeanflow.py`
  - vai trò: train CLI
  - trạng thái: hoàn thành
- `/home/hieulc/avitech11/crackmeanflow/scripts/test_crackmeanflow.py`
  - vai trò: test CLI
  - trạng thái: hoàn thành
- `/home/hieulc/avitech11/crackmeanflow/scripts/benchmark_crackdiff_vs_crackmeanflow.py`
  - vai trò: benchmark/sweep CLI
  - trạng thái: hoàn thành source; chưa có benchmark artifact
- `/home/hieulc/avitech11/crackmeanflow/configs/crackmeanflow_default.yaml`
  - vai trò: default config
  - trạng thái: hoàn thành
- `/home/hieulc/avitech11/crackmeanflow/reports/CONTRACT_REPORT.md`
  - vai trò: contract report
  - trạng thái: hoàn thành
- `/home/hieulc/avitech11/crackmeanflow/reports/TEST_REPORT.md`
  - vai trò: root test report
  - trạng thái: cần kiểm tra/không thấy
- `/home/hieulc/avitech11/crackmeanflow/reports/FINAL_REPORT.md`
  - vai trò: final report
  - trạng thái: cần kiểm tra/không thấy
- `/home/hieulc/avitech11/crackmeanflow/outputs/metrics.json`
  - vai trò: top-level metrics
  - trạng thái: cần kiểm tra/không thấy; metrics ở subdirs

## 4. Nội dung code quan trọng
Copy full artifacts: `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files`
### Files copied
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/__init__.py`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/adapter.py`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/benchmark_crackdiff_vs_crackmeanflow.py`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/checkpointing.py`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/crackmeanflow_default.yaml`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/crackmeanflow_long_endpoint.yaml`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/crackmeanflow_v2.yaml`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/crackmeanflow_v3.yaml`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/crackmeanflow_v4_256.yaml`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/crackmeanflow_v5_256_ft.yaml`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/data.py`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/direct_unet.py`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.1/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.2/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.3/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.4/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.5/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_-0.7/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.0/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.1/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.2/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.3/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/final_sweep_0.5/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.1/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.2/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.3/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_-0.5/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.0/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.1/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.2/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.3/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/goal_sweep_0.5/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.1/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.2/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.3/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/sweep_-0.5/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.0/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.1/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.2/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.3/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/sweep_0.5/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/sweep_m0.1/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/sweep_m0.2/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/sweep_m0.3/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/sweep_m0.5/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/test_best_t0/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/test_clean_seg/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/test_long_best_t0/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v2_final_sweep_-0.1/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v2_final_sweep_-0.3/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v2_final_sweep_-0.5/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v2_final_sweep_-0.7/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v2_final_sweep_-0.9/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v2_final_sweep_0.0/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v2_final_sweep_0.1/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v2_final_sweep_0.3/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v2_final_sweep_0.5/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v2_sweep_-0.1/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v2_sweep_-0.3/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v2_sweep_0.0/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v4_256_sweep_-0.1/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v4_256_sweep_-0.3/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v4_256_sweep_-0.5/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v4_256_sweep_-0.7/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v4_256_sweep_-0.9/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v4_256_sweep_0.0/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v4_256_sweep_0.1/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v4_256_sweep_0.3/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v4_256_sweep_0.5/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v5_256_sweep_-0.1/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v5_256_sweep_-0.2/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v5_256_sweep_-0.3/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v5_256_sweep_-0.4/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v5_256_sweep_-0.5/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v5_256_sweep_-0.6/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v5_256_sweep_0.0/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v5_256_sweep_0.1/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/home/hieulc/avitech11/crackmeanflow/outputs/v5_256_sweep_0.2/metrics.json`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/loss.py`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/metrics.py`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/model.py`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/paths.py`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/sampler.py`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/smoke_test.py`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/test.py`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/test_crackmeanflow.py`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/thin_metrics.py`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/train.py`
- `/home/hieulc/avitech11/crackmeanflow/reports/chatgpt_audit_files/train_crackmeanflow.py`
### crackmeanflow/adapter.py
```python
from dataclasses import dataclass

import torch
from torch import nn

from .paths import ensure_paths

ensure_paths()
from multi_task.mlt_unet import UNet  # noqa: E402


@dataclass
class CrackMeanFlowConfig:
    T: int = 500


class CrackMeanFlowModel(nn.Module):
    """Adapter from CrackDiff UNet to MeanFlow velocity model interface."""

    def __init__(self, unet: UNet, T: int = 500):
        super().__init__()
        self.unet = unet
        self.T = int(T)
        self.num_classes = 0
        self._last_seg_logits = None

    def _to_batch_time(self, t: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device, dtype=torch.float32)
        t = t.to(device=device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.repeat(batch_size)
        elif t.ndim == 1 and t.shape[0] == 1 and batch_size > 1:
            t = t.repeat(batch_size)
        elif t.ndim != 1:
            t = t.view(batch_size)
        t = t.clamp(0.0, 1.0)
        return torch.round(t * (self.T - 1)).long()

    def clear_seg_logits(self):
        self._last_seg_logits = None

    def get_seg_logits(self):
        return self._last_seg_logits

    def forward(self, x, r, t, y=None, **kwargs):
        del r, kwargs
        if y is None:
            raise ValueError("CrackMeanFlowModel.forward requires conditioning image `y`.")

        t_int = self._to_batch_time(t, batch_size=x.shape[0], device=x.device)
        velocity_pred, seg_logits = self.unet(x, t_int, y)
        self._last_seg_logits = seg_logits
        return velocity_pred


```
### crackmeanflow/loss.py
```python
import sys
import types

import torch
from torch import nn
import torch.nn.functional as F

from .paths import ensure_paths

ensure_paths()
if "torch.func" not in sys.modules:
    from functorch import jvp as _functorch_jvp

    _torch_func_module = types.ModuleType("torch.func")
    _torch_func_module.jvp = _functorch_jvp
    sys.modules["torch.func"] = _torch_func_module
    torch.func = _torch_func_module
from loss import SILoss  # noqa: E402
from multi_task.mltdiff import FocalTverskyLoss  # noqa: E402


class _SILossModelProxy(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_classes = 0
        self.module = self

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


class CrackSILoss(nn.Module):
    def __init__(self, si_loss_kwargs=None, seg_loss_weight=1.0, endpoint_loss_weight=0.5, thin_loss_weight=0.0, mode="hybrid"):
        super().__init__()
        kwargs = dict(si_loss_kwargs or {})
        kwargs.setdefault("label_dropout_prob", 0.0)
        kwargs.setdefault("cfg_omega", 1.0)
        kwargs.setdefault("cfg_kappa", 0.0)
        self.si_loss = SILoss(**kwargs)
        self.seg_loss = FocalTverskyLoss()
        self.seg_loss_weight = float(seg_loss_weight)
        self.endpoint_loss_weight = float(endpoint_loss_weight)
        self.thin_loss_weight = float(thin_loss_weight)
        self.mode = str(mode)

    def _check_finite(self, name, tensor, context):
        if not torch.isfinite(tensor).all():
            ranges = []
            for key, value in context.items():
                if torch.is_tensor(value):
                    ranges.append(f"{key}: shape={tuple(value.shape)} min={value.min().item():.6g} max={value.max().item():.6g}")
            raise RuntimeError(f"Non-finite {name}. " + " | ".join(ranges))

    def forward(self, model, x0, model_kwargs=None):
        model_kwargs = dict(model_kwargs or {})
        y = model_kwargs.get("y")
        mask_gt = model_kwargs.get("mask_gt")
        if y is None or mask_gt is None:
            raise ValueError("CrackSILoss requires model_kwargs with `y` and `mask_gt`.")

        base_model = _unwrap(model)

        # segmentation branch should condition on image + noisy latent, not clean GT x0.
        z = torch.randn_like(x0)
        r = torch.zeros(x0.shape[0], device=x0.device)
        t = torch.ones(x0.shape[0], device=x0.device)
        u = model(z, r, t, y=y)
        seg_logits = base_model.get_seg_logits()
        if seg_logits is None:
            raise RuntimeError("Segmentation logits unavailable after noisy forward.")
        seg_loss = self.seg_loss(seg_logits, mask_gt)
        x0_pred = z - u
        endpoint_loss = F.l1_loss(x0_pred, x0)

        if self.mode == "seg_only":
            si_loss = x0.new_tensor(0.0)
            si_loss_ref_mean = x0.new_tensor(0.0)
        else:
            proxy = model if hasattr(model, "module") else _SILossModelProxy(model)
            si_loss_vec, si_loss_ref = self.si_loss(proxy, x0, {"y": y})
            si_loss = si_loss_vec.mean()
            si_loss_ref_mean = si_loss_ref.mean() if torch.is_tensor(si_loss_ref) else torch.as_tensor(si_loss_ref, device=x0.device)

        thin_loss = x0.new_tensor(0.0)
        total_loss = si_loss + self.seg_loss_weight * seg_loss + self.endpoint_loss_weight * endpoint_loss + self.thin_loss_weight * thin_loss

        context = {"x0": x0, "z": z, "u": u, "x0_pred": x0_pred, "seg_logits": seg_logits}
        for name, value in [("total_loss", total_loss), ("si_loss", si_loss), ("seg_loss", seg_loss), ("endpoint_loss", endpoint_loss)]:
            self._check_finite(name, value, context)

        loss_dict = {
            "total_loss": float(total_loss.detach().cpu().item()),
            "si_loss": float(si_loss.detach().cpu().item()),
            "si_loss_ref": float(si_loss_ref_mean.detach().cpu().item()),
            "seg_loss": float(seg_loss.detach().cpu().item()),
            "endpoint_loss": float(endpoint_loss.detach().cpu().item()),
            "thin_loss": float(thin_loss.detach().cpu().item()),
            "nan_flags": {},
            "mode": self.mode,
        }
        return total_loss, loss_dict

```
### crackmeanflow/sampler.py
```python
import torch


def _get_model_attr(model, name):
    return getattr(model.module, name) if hasattr(model, "module") else getattr(model, name)


@torch.no_grad()
def crack_meanflow_sampler(model, z, crack_image, num_steps=1, cfg_scale=1.0, clamp=True):
    batch_size = z.shape[0]
    device = z.device
    do_cfg = cfg_scale > 1.0
    sampled = z

    def _forward(sample, r, t, y):
        return model(sample, r, t, y=y)

    if num_steps == 1:
        r = torch.zeros(batch_size, device=device)
        t = torch.ones(batch_size, device=device)
        if do_cfg:
            z2 = torch.cat([sampled, sampled], dim=0)
            r2 = torch.cat([r, r], dim=0)
            t2 = torch.cat([t, t], dim=0)
            y2 = torch.cat([crack_image, torch.zeros_like(crack_image)], dim=0)
            u2 = _forward(z2, r2, t2, y2)
            u_cond, u_uncond = torch.chunk(u2, 2, dim=0)
            u = u_uncond + cfg_scale * (u_cond - u_uncond)
        else:
            u = _forward(sampled, r, t, crack_image)
        sampled = sampled - u
    else:
        grid = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        for idx in range(num_steps):
            t_cur = torch.full((batch_size,), float(grid[idx].item()), device=device)
            t_next = torch.full((batch_size,), float(grid[idx + 1].item()), device=device)
            if do_cfg:
                z2 = torch.cat([sampled, sampled], dim=0)
                r2 = torch.cat([t_next, t_next], dim=0)
                t2 = torch.cat([t_cur, t_cur], dim=0)
                y2 = torch.cat([crack_image, torch.zeros_like(crack_image)], dim=0)
                u2 = _forward(z2, r2, t2, y2)
                u_cond, u_uncond = torch.chunk(u2, 2, dim=0)
                u = u_uncond + cfg_scale * (u_cond - u_uncond)
            else:
                u = _forward(sampled, t_next, t_cur, crack_image)
            sampled = sampled + (t_next[:, None, None, None] - t_cur[:, None, None, None]) * u

    if clamp:
        sampled = sampled.clamp(-1.0, 1.0)
    seg_logits = _get_model_attr(model, "get_seg_logits")()
    return sampled, seg_logits

```
### crackmeanflow/metrics.py
```python
import torch


def _safe_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def compute_segmentation_metrics(pred_binary, mask_gt, eps=1e-7):
    pred = (pred_binary.float() > 0.5).float().view(-1)
    gt = (mask_gt.float() > 0.5).float().view(-1)
    tp = (pred * gt).sum()
    fp = (pred * (1.0 - gt)).sum()
    fn = ((1.0 - pred) * gt).sum()
    tn = ((1.0 - pred) * (1.0 - gt)).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    dice = f1
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    if gt.sum() == 0 and pred.sum() == 0:
        precision = recall = f1 = dice = iou = accuracy.new_tensor(1.0)
    return {
        "iou": _safe_float(iou),
        "dice": _safe_float(dice),
        "f1": _safe_float(f1),
        "precision": _safe_float(precision),
        "recall": _safe_float(recall),
        "accuracy": _safe_float(accuracy),
        "tp": _safe_float(tp),
        "fp": _safe_float(fp),
        "fn": _safe_float(fn),
        "tn": _safe_float(tn),
    }

```
### crackmeanflow/thin_metrics.py
```python
import torch
import torch.nn.functional as F


def _erode(mask: torch.Tensor) -> torch.Tensor:
    inv = 1.0 - mask.float()
    dilated_inv = F.max_pool2d(inv, kernel_size=3, stride=1, padding=1)
    return 1.0 - dilated_inv


def skeletonize_or_thin_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = (mask.float() > 0.5).float()
    try:
        from skimage.morphology import skeletonize

        out = []
        for sample in mask:
            arr = sample[0].detach().cpu().numpy() > 0
            skel = skeletonize(arr).astype("float32")
            out.append(torch.from_numpy(skel).to(mask.device).unsqueeze(0))
        return torch.stack(out, dim=0)
    except Exception:
        eroded = _erode(mask)
        thin = torch.clamp(mask - eroded, min=0.0, max=1.0)
        if thin.sum() == 0:
            thin = mask
        return thin


def compute_thin_crack_metrics(pred_binary: torch.Tensor, mask_gt: torch.Tensor, eps: float = 1e-7):
    pred = (pred_binary.float() > 0.5).float()
    gt = (mask_gt.float() > 0.5).float()
    pred_thin = skeletonize_or_thin_mask(pred)
    gt_thin = skeletonize_or_thin_mask(gt)
    tp = (pred_thin * gt_thin).sum()
    fp = (pred_thin * (1.0 - gt_thin)).sum()
    fn = ((1.0 - pred_thin) * gt_thin).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    boundary_f1 = f1
    if gt_thin.sum() == 0 and pred_thin.sum() == 0:
        precision = recall = f1 = boundary_f1 = precision.new_tensor(1.0)
    return {
        "thin_recall": float(recall.detach().cpu().item()),
        "thin_precision": float(precision.detach().cpu().item()),
        "thin_f1": float(f1.detach().cpu().item()),
        "boundary_f1": float(boundary_f1.detach().cpu().item()),
        "recall_thin": float(recall.detach().cpu().item()),
        "f1_thin": float(f1.detach().cpu().item()),
        "dice_thin": float(f1.detach().cpu().item()),
    }

```
### crackmeanflow/checkpointing.py
```python
from pathlib import Path
from typing import Dict

import torch


def strip_prefix_if_needed(state_dict: Dict[str, torch.Tensor], prefixes=("module.", "unet.")) -> Dict[str, torch.Tensor]:
    out = dict(state_dict)
    for prefix in prefixes:
        if out and all(k.startswith(prefix) for k in out.keys()):
            out = {k[len(prefix):]: v for k, v in out.items()}
    return out


def has_nan_weights(state_dict: Dict[str, torch.Tensor]) -> bool:
    for value in state_dict.values():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            return True
    return False


def _model_state(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def save_checkpoint(path, model, optimizer=None, scheduler=None, ema=None, epoch=0, global_step=0, best_metrics=None, config=None, architecture=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": _model_state(model),
        "ema": _model_state(ema) if ema is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metrics": best_metrics or {},
        "config": config or {},
        "architecture": architecture or {},
    }
    if has_nan_weights(ckpt["model"]):
        raise RuntimeError("Refusing to save checkpoint with NaN/Inf model weights.")
    torch.save(ckpt, path)
    return str(path)


def _try_load(model, state):
    result = model.load_state_dict(state, strict=False)
    return list(result.missing_keys), list(result.unexpected_keys)


def load_checkpoint_strict(model, ckpt_path, key="model", map_location="cpu", allow_missing=False, allow_unexpected=False):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if key not in ckpt or ckpt[key] is None:
        raise KeyError(f"Checkpoint {ckpt_path} has no key {key!r}.")
    raw_state = ckpt[key]
    if has_nan_weights(raw_state):
        raise RuntimeError(f"Checkpoint {ckpt_path}:{key} contains NaN/Inf weights.")

    candidates = [raw_state]
    module_stripped = strip_prefix_if_needed(raw_state, prefixes=("module.",))
    if module_stripped is not raw_state:
        candidates.append(module_stripped)
    fully_stripped = strip_prefix_if_needed(raw_state)
    if fully_stripped != module_stripped:
        candidates.append(fully_stripped)

    best = None
    for candidate in candidates:
        missing, unexpected = _try_load(model, candidate)
        score = len(missing) + len(unexpected)
        if best is None or score < best[0]:
            best = (score, missing, unexpected, candidate)
        if score == 0:
            break
    _, missing, unexpected, _ = best
    if (missing and not allow_missing) or (unexpected and not allow_unexpected):
        raise RuntimeError(f"Strict load failed for {ckpt_path}:{key}. missing={missing} unexpected={unexpected}")
    return ckpt, {"missing": missing, "unexpected": unexpected}


def audit_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    report = {"path": str(ckpt_path), "keys": sorted(list(ckpt.keys()))}
    for key in ("model", "ema"):
        if key in ckpt and ckpt[key] is not None:
            report[f"{key}_num_tensors"] = len(ckpt[key])
            report[f"{key}_has_nan_or_inf"] = has_nan_weights(ckpt[key])
    return report

```
### crackmeanflow/train.py
```python
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .adapter import CrackMeanFlowModel
from .checkpointing import has_nan_weights, load_checkpoint_strict, save_checkpoint
from .data import PairedCrackDataset, deterministic_split, list_pairs, write_split_report
from .loss import CrackSILoss
from .paths import CRACKDIFF_ROOT, ensure_paths
from .sampler import crack_meanflow_sampler
from .metrics import compute_segmentation_metrics

ensure_paths()
from multi_task.mlt_unet import UNet  # noqa: E402

logger = logging.getLogger("crackmeanflow.train")


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.num_updates = 0

    def update(self, model):
        self.num_updates += 1
        d = self.decay
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k not in self.shadow:
                    self.shadow[k] = v.clone().detach()
                    continue
                if torch.is_floating_point(v):
                    self.shadow[k].mul_(d).add_(v, alpha=1.0 - d)
                else:
                    self.shadow[k].copy_(v)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = state


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    out_dir = Path(cfg["paths"]["output_dir"])
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    report_dir = Path(cfg["paths"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # --- data split ---
    pairs = list_pairs(cfg["paths"]["image_dir"], cfg["paths"]["mask_dir"])
    assert len(pairs) >= 5, f"Too few pairs ({len(pairs)}), need >= 5"
    train_pairs, test_pairs = deterministic_split(pairs, cfg["train"]["train_ratio"])
    write_split_report(report_dir / "DATA_SPLIT_REPORT.md", pairs, train_pairs, test_pairs, cfg["paths"]["image_dir"], cfg["paths"]["mask_dir"])
    logger.info("Pairs=%d  Train=%d  Test=%d", len(pairs), len(train_pairs), len(test_pairs))

    train_ds = PairedCrackDataset(train_pairs, image_size=cfg["model"]["img_size"], augment=cfg["train"].get("augment", True))
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0, pin_memory=True)

    # --- model ---
    unet = UNet(
        T=cfg["model"]["T"],
        ch=cfg["model"]["ch"],
        ch_mult=cfg["model"]["ch_mult"],
        attn=cfg["model"]["attn"],
        num_res_blocks=cfg["model"]["num_res_blocks"],
        dropout=cfg["model"]["dropout"],
    )
    model = CrackMeanFlowModel(unet, T=cfg["model"]["T"]).to(device)
    logger.info("Model params: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    ema = EMA(model, decay=cfg["train"]["ema_decay"])
    scaler = GradScaler()
    criterion = CrackSILoss(
        si_loss_kwargs=cfg["loss"]["si_loss_kwargs"],
        seg_loss_weight=cfg["loss"]["seg_loss_weight"],
        endpoint_loss_weight=cfg["loss"]["endpoint_loss_weight"],
        thin_loss_weight=cfg["loss"]["thin_loss_weight"],
        mode=cfg["loss"].get("mode", "hybrid"),
    )

    # --- resume ---
    global_step = 0
    best_f1 = 0.0
    resume_path = ckpt_dir / "last.pt"
    if resume_path.exists():
        try:
            ckpt_data, _ = load_checkpoint_strict(model, resume_path, map_location=device)
            if ckpt_data.get("optimizer"):
                optimizer.load_state_dict(ckpt_data["optimizer"])
            if ckpt_data.get("global_step"):
                global_step = ckpt_data["global_step"]
            if ckpt_data.get("best_metrics", {}).get("f1"):
                best_f1 = ckpt_data["best_metrics"]["f1"]
            if ckpt_data.get("ema"):
                ema.load_state_dict(ckpt_data["ema"])
            logger.info("Resumed from step %d  best_f1=%.4f", global_step, best_f1)
        except Exception as e:
            logger.warning("Resume failed (%s), starting fresh.", e)

    epochs = cfg["train"]["epochs"]
    log_interval = cfg["train"]["log_interval"]
    save_interval = cfg["train"]["save_interval"]
    max_grad_norm = cfg["train"]["max_grad_norm"]
    max_batches = cfg["train"].get("max_train_batches", 0)
    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    best_ckpt_path = ckpt_dir / "best.pt"

    # --- train loop ---
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader):
            if max_batches and batch_idx >= max_batches:
                break
            image = batch["crack"].to(device)
            mask = batch["mask"].to(device)
            x0 = mask * 2.0 - 1.0

            with autocast():
                total_loss, loss_dict = criterion(model, x0, {"y": image, "mask_gt": mask})

            if not torch.isfinite(total_loss):
                logger.warning("Non-finite loss at step %d, skipping.", global_step)
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(total_loss / grad_accum_steps).backward()
            should_step = ((batch_idx + 1) % grad_accum_steps == 0)
            if max_batches and (batch_idx + 1) >= max_batches:
                should_step = True
            elif (batch_idx + 1) == len(train_loader):
                should_step = True

            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                global_step += 1

                if global_step % log_interval == 0:
                    ld = loss_dict
                    logger.info("[step %d] total=%.4f seg=%.4f %s", global_step, ld["total_loss"], ld.get("seg_loss", 0.0),
                                " ".join(f"{k}={v:.4f}" for k, v in ld.items() if k not in ("total_loss", "seg_loss", "nan_flags", "mode")))

                    if global_step % save_interval == 0:
                        save_checkpoint(ckpt_dir / "last.pt", model, optimizer=optimizer, ema=ema, epoch=epoch, global_step=global_step, best_metrics={"f1": best_f1}, config=cfg)

            epoch_losses.append(loss_dict)
            logger.warning("Epoch %d: no valid batches.", epoch)
            continue

        numeric_keys = [k for k, v in epoch_losses[0].items() if isinstance(v, (int, float))]
        avg = {k: sum(d[k] for d in epoch_losses) / len(epoch_losses) for k in numeric_keys}
        elapsed = time.time() - t0
        logger.info("Epoch %d done in %.1fs  avg_total=%.4f", epoch, elapsed, avg["total_loss"])

        # --- quick eval on a few test batches ---
        test_f1 = _quick_eval(model, test_pairs, cfg, device)
        logger.info("Quick eval F1=%.4f (best=%.4f)", test_f1, best_f1)
        if test_f1 > best_f1:
            best_f1 = test_f1
            save_checkpoint(best_ckpt_path, model, optimizer=optimizer, ema=ema, epoch=epoch, global_step=global_step, best_metrics={"f1": best_f1}, config=cfg)
            logger.info("New best F1=%.4f saved.", best_f1)

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer=optimizer, ema=ema, epoch=epoch, global_step=global_step, best_metrics={"f1": best_f1}, config=cfg)

    logger.info("Training complete. best_f1=%.4f", best_f1)
    return best_ckpt_path


@torch.no_grad()
def _quick_eval(model, test_pairs, cfg, device):
    model.eval()
    from .data import PairedCrackDataset
    from torch.utils.data import DataLoader
    ds = PairedCrackDataset(test_pairs, image_size=cfg["model"]["img_size"])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    f1s = []
    max_eval = cfg["train"].get("max_eval_batches", 20)
    for i, batch in enumerate(loader):
        if i >= max_eval:
            break
        image = batch["crack"].to(device)
        mask = batch["mask"].to(device)
        z = torch.randn_like(mask)
        sampled, seg_logits = crack_meanflow_sampler(model, z, image, num_steps=1)
        pred_flow = (sampled > 0.0).float()
        m_flow = compute_segmentation_metrics(pred_flow, mask)
        best_f1 = m_flow["f1"]
        if seg_logits is not None:
            pred_seg = (torch.sigmoid(seg_logits) > 0.5).float()
            m_seg = compute_segmentation_metrics(pred_seg, mask)
            best_f1 = max(best_f1, m_seg["f1"])
        f1s.append(best_f1)
    model.train()
    return sum(f1s) / len(f1s) if f1s else 0.0

```
### crackmeanflow/test.py
```python
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from .adapter import CrackMeanFlowModel
from .checkpointing import has_nan_weights, load_checkpoint_strict
from .data import PairedCrackDataset, deterministic_split, list_pairs
from .metrics import compute_segmentation_metrics
from .paths import ensure_paths
from .sampler import crack_meanflow_sampler
from .thin_metrics import compute_thin_crack_metrics

ensure_paths()
from multi_task.mlt_unet import UNet  # noqa: E402


class EMAWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def _save_mask_png(path, pred):
    arr = (pred.squeeze().detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(path)


@torch.no_grad()
def evaluate(cfg, ckpt_path, output_dir, num_steps=1, threshold=0.0, use_ema=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    pairs = list_pairs(cfg["paths"]["image_dir"], cfg["paths"]["mask_dir"])
    _, test_pairs = deterministic_split(pairs, cfg["train"]["train_ratio"])
    ds = PairedCrackDataset(test_pairs, image_size=cfg["model"]["img_size"])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    unet = UNet(
        T=cfg["model"]["T"],
        ch=cfg["model"]["ch"],
        ch_mult=cfg["model"]["ch_mult"],
        attn=cfg["model"]["attn"],
        num_res_blocks=cfg["model"]["num_res_blocks"],
        dropout=cfg["model"]["dropout"],
    )
    model = CrackMeanFlowModel(unet, T=cfg["model"]["T"]).to(device)
    ckpt, _ = load_checkpoint_strict(model, ckpt_path, map_location=device)
    if use_ema and ckpt.get("ema") is not None:
        if has_nan_weights(ckpt["ema"]):
            raise RuntimeError("EMA weights contain NaN/Inf, refusing to load.")
        ema_unet = UNet(
            T=cfg["model"]["T"],
            ch=cfg["model"]["ch"],
            ch_mult=cfg["model"]["ch_mult"],
            attn=cfg["model"]["attn"],
            num_res_blocks=cfg["model"]["num_res_blocks"],
            dropout=cfg["model"]["dropout"],
        )
        model = CrackMeanFlowModel(ema_unet, T=cfg["model"]["T"]).to(device)
        load_checkpoint_strict(model, ckpt_path, key="ema", map_location=device)
    model.eval()

    totals = []
    latencies = []
    for batch in loader:
        name = batch["name"][0]
        image = batch["crack"].to(device)
        mask = batch["mask"].to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        z = torch.randn_like(mask)
        sampled, seg_logits = crack_meanflow_sampler(model, z, image, num_steps=num_steps)
        pred = (sampled > threshold).float()

        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

        metrics = compute_segmentation_metrics(pred, mask)


        thin = compute_thin_crack_metrics(pred, mask)
        merged = {**{k: float(v.item()) if torch.is_tensor(v) else float(v) for k, v in metrics.items()}, **{k: float(v.item()) if torch.is_tensor(v) else float(v) for k, v in thin.items()}}
        merged["name"] = name
        totals.append(merged)
        _save_mask_png(pred_dir / f"{name}.png", pred[0])

    def avg(key):
        return sum(item[key] for item in totals) / len(totals) if totals else 0.0

    metrics_json = {
        "num_samples": len(totals),
        "f1": avg("f1"),
        "iou": avg("iou"),
        "dice": avg("dice"),
        "precision": avg("precision"),
        "recall": avg("recall"),
        "accuracy": avg("accuracy"),
        "thin_recall": avg("thin_recall"),
        "thin_precision": avg("thin_precision"),
        "thin_f1": avg("thin_f1"),
        "boundary_f1": avg("boundary_f1"),
        "latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "throughput_fps": (len(latencies) / sum(latencies)) if latencies and sum(latencies) > 0 else 0.0,
        "num_steps": int(num_steps),
        "threshold": float(threshold),
        "checkpoint": str(ckpt_path),
        "use_ema": bool(use_ema),
        "per_sample": totals,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2) + "\n")

    report = [
        "# CrackMeanFlow Test Report",
        "",
        f"Checkpoint: `{ckpt_path}`",
        f"Samples: {metrics_json['num_samples']}",
        f"F1: {metrics_json['f1']:.6f}",
        f"IoU: {metrics_json['iou']:.6f}",
        f"Dice: {metrics_json['dice']:.6f}",
        f"Precision: {metrics_json['precision']:.6f}",
        f"Recall: {metrics_json['recall']:.6f}",
        f"Thin recall: {metrics_json['thin_recall']:.6f}",
        f"Thin F1: {metrics_json['thin_f1']:.6f}",
        f"Latency (s/img): {metrics_json['latency_seconds']:.6f}",
        f"Throughput (img/s): {metrics_json['throughput_fps']:.6f}",
    ]
    (output_dir / "TEST_REPORT.md").write_text("\n".join(report) + "\n")
    return metrics_json

```
### scripts/smoke_test.py
```python
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from crackmeanflow.adapter import CrackMeanFlowModel
from crackmeanflow.checkpointing import load_checkpoint_strict, save_checkpoint
from crackmeanflow.loss import CrackSILoss
from crackmeanflow.metrics import compute_segmentation_metrics
from crackmeanflow.paths import CRACKDIFF_ROOT, CRACKMEANFLOW_ROOT, MEANFLOW_ROOT
from crackmeanflow.sampler import crack_meanflow_sampler
from crackmeanflow.thin_metrics import compute_thin_crack_metrics

sys.path.insert(0, CRACKDIFF_ROOT)
from multi_task.mlt_unet import UNet  # noqa: E402


def assert_finite_grad(model):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients created"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"


def main():
    assert Path(CRACKDIFF_ROOT).exists()
    assert Path(MEANFLOW_ROOT).exists()
    assert Path(CRACKMEANFLOW_ROOT).exists()
    torch.manual_seed(0)
    device = torch.device("cpu")
    unet = UNet(T=16, ch=32, ch_mult=[1, 2], attn=[], num_res_blocks=1, dropout=0.0).to(device)
    model = CrackMeanFlowModel(unet, T=16).to(device)
    b = 2
    x = torch.randn(b, 1, 256, 256, device=device)
    image = torch.randn(b, 3, 256, 256, device=device)
    mask = (torch.rand(b, 1, 256, 256, device=device) > 0.8).float()
    r = torch.zeros(b, device=device)
    t = torch.ones(b, device=device)
    out = model(x, r, t, y=image)
    assert tuple(out.shape) == (b, 1, 256, 256)
    assert model.get_seg_logits() is not None
    assert tuple(model.get_seg_logits().shape) == (b, 1, 256, 256)

    criterion = CrackSILoss(si_loss_kwargs={"time_sampler": "uniform", "label_dropout_prob": 0.0}, seg_loss_weight=0.1, endpoint_loss_weight=1.0)
    x0 = mask * 2.0 - 1.0
    total_loss, loss_dict = criterion(model, x0, {"y": image, "mask_gt": mask})
    assert total_loss.ndim == 0
    assert torch.isfinite(total_loss)
    assert {"total_loss", "si_loss", "seg_loss", "endpoint_loss"}.issubset(loss_dict)
    total_loss.backward()
    assert_finite_grad(model)

    with torch.no_grad():
        sampled, seg = crack_meanflow_sampler(model, torch.randn_like(mask), image, num_steps=1)
        assert tuple(sampled.shape) == tuple(mask.shape)
        assert seg is not None and tuple(seg.shape) == tuple(mask.shape)
        sampled4, _ = crack_meanflow_sampler(model, torch.randn_like(mask), image, num_steps=4)
        assert tuple(sampled4.shape) == tuple(mask.shape)

    metrics = compute_segmentation_metrics((sampled > 0).float(), mask)
    thin = compute_thin_crack_metrics((sampled > 0).float(), mask)
    for key in ["iou", "dice", "f1", "precision", "recall"]:
        assert 0.0 <= metrics[key] <= 1.0, (key, metrics[key])
    for key in ["thin_recall", "thin_precision", "thin_f1"]:
        assert 0.0 <= thin[key] <= 1.0, (key, thin[key])

    ckpt = ROOT / "outputs" / "smoke_ckpt.pt"
    save_checkpoint(ckpt, model, epoch=0, global_step=1, architecture={"T": 16})
    reloaded = CrackMeanFlowModel(UNet(T=16, ch=32, ch_mult=[1, 2], attn=[], num_res_blocks=1, dropout=0.0).to(device), T=16).to(device)
    load_checkpoint_strict(reloaded, ckpt, map_location=device)
    print("SMOKE_TEST_PASS")


if __name__ == "__main__":
    main()

```
### scripts/train_crackmeanflow.py
```python
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from crackmeanflow.paths import ensure_paths
ensure_paths()
from crackmeanflow.train import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("train_cli")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "crackmeanflow_default.yaml"))
    args, extra = parser.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for arg in extra:
        if "=" in arg:
            k, v = arg.split("=", 1)
            keys = k.lstrip("-").split(".")
            d = cfg
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            try:
                d[keys[-1]] = float(v)
            except ValueError:
                d[keys[-1]] = v

    train(cfg)


if __name__ == "__main__":
    main()

```
### scripts/test_crackmeanflow.py
```python
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from crackmeanflow.paths import ensure_paths
ensure_paths()
from crackmeanflow.test import evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_cli")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "crackmeanflow_default.yaml"))
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "best.pt"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "test_best"))
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--use-ema", action="store_true", default=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    metrics = evaluate(cfg, args.ckpt, args.output_dir, num_steps=args.num_steps, threshold=args.threshold, use_ema=args.use_ema)
    logger.info("F1=%.4f  IoU=%.4f  Dice=%.4f  Precision=%.4f  Recall=%.4f  Latency=%.4fs",
                metrics["f1"], metrics["iou"], metrics["dice"], metrics["precision"], metrics["recall"], metrics["latency_seconds"])


if __name__ == "__main__":
    main()

```
### configs/crackmeanflow_default.yaml
```yaml
paths:
  crackdiff_root: /home/hieulc/avitech11/crack_diff/crackdiff
  meanflow_root: /home/hieulc/avitech11/MeanFlow
  crackmeanflow_root: /home/hieulc/avitech11/crackmeanflow
  image_dir: /home/hieulc/avitech11/Hieus_11/ALL_model_segmentation/copy
  mask_dir: /home/hieulc/avitech11/Hieus_11/ALL_model_segmentation/img_resize
  output_dir: /home/hieulc/avitech11/crackmeanflow/outputs
  checkpoint_dir: /home/hieulc/avitech11/crackmeanflow/checkpoints
  reports_dir: /home/hieulc/avitech11/crackmeanflow/reports
  logs_dir: /home/hieulc/avitech11/crackmeanflow/logs
model:
  T: 500
  ch: 32
  ch_mult: [1]
  attn: []
  num_res_blocks: 1
  dropout: 0.1
  img_size: 64
  use_direct_unet: false
train:
  epochs: 30
  batch_size: 1
  lr: 0.0002
  weight_decay: 0.0001
  ema_decay: 0.9999
  max_grad_norm: 1.0
  log_interval: 10
  save_interval: 100
  train_ratio: 0.8
  max_train_batches: 600
  max_eval_batches: 50
  augment: true
loss:
  mode: hybrid
  seg_loss_weight: 1.0
  endpoint_loss_weight: 1.0
  thin_loss_weight: 0.0
  si_loss_kwargs:
    path_type: linear
    weighting: uniform
    time_sampler: uniform
    ratio_r_not_equal_t: 0.75
    label_dropout_prob: 0.0
    cfg_omega: 1.0
    cfg_kappa: 0.0
eval:
  num_steps: 1
  cfg_scale: 1.0
  threshold: 0.0
  use_ema: true

```
## 5. Contract report
# CrackMeanFlow Contract Report

## Dataset

Input images: `/home/hieulc/avitech11/Hieus_11/ALL_model_segmentation/copy`

Masks: `/home/hieulc/avitech11/Hieus_11/ALL_model_segmentation/img_resize`

Dataset item schema:

```python
{
  "name": str,
  "crack": FloatTensor[3, 256, 256],  # RGB, [0,1]
  "mask": FloatTensor[1, 256, 256],   # binary {0,1}
}
```

Training clean target:

```python
x0 = mask * 2.0 - 1.0
```

Prediction binarization:

```python
pred = (sampled_mask > threshold).float()
```

## UNet source-truth contract

Actual source file: `/home/hieulc/avitech11/crack_diff/crackdiff/multi_task/mlt_unet.py`

Actual return order:

```python
velocity_pred, seg_logits = unet(x_t, t_int, image)
```

This contradicts older docs/comments that imply `(seg_pred, noisy_pred)`. CrackMeanFlow follows actual source, not stale docs.

## CrackMeanFlow adapter

```python
u = model(x, r, t, y=image)
```

- `r` accepted for MeanFlow compatibility.
- continuous `t in [0,1]` mapped to integer CrackDiff timestep `[0,T-1]`.
- wrapper returns velocity/noise prediction only.
- segmentation logits cached via `get_seg_logits()`.

## MeanFlow one-step sampler

Main inference:

```python
r = 0
t = 1
u = model(z, r, t, y=image)
x0 = z - u
```

`num_steps=1` is primary. `num_steps=4` is ablation only.

## Loss

`CrackSILoss = MeanFlow SILoss + FocalTverskyLoss + endpoint L1`.

- `si_loss`: imported MeanFlow SILoss.
- `seg_loss`: imported CrackDiff `FocalTverskyLoss`.
- `endpoint_loss`: one-step consistency `L1(z - u, x0)`.

## Checkpointing

Saved keys:

- `model`
- `ema`
- `optimizer`
- `scheduler`
- `epoch`
- `global_step`
- `best_metrics`
- `config`
- `architecture`

Strict load required. NaN/Inf weights rejected.

# CrackMeanFlow Data Split Report

Image dir: `/home/hieulc/avitech11/Hieus_11/ALL_model_segmentation/copy`
Mask dir: `/home/hieulc/avitech11/Hieus_11/ALL_model_segmentation/img_resize`
Matched pairs: 421
Train pairs: 336
Test pairs: 85
Split policy: deterministic sorted stem split; no random state dependency.

## First train names
- GAPS384_test_0005_1_1
- GAPS384_test_0005_541_1
- GAPS384_test_0005_541_641
- GAPS384_test_0006_1_641
- GAPS384_test_0016_1_641
- GAPS384_test_0016_541_1
- GAPS384_test_0016_541_641
- GAPS384_test_0028_541_1
- GAPS384_test_0033_541_1
- GAPS384_test_0033_541_641

## First test names
- GAPS384_train_1171_541_641
- GAPS384_train_1172_541_1
- GAPS384_train_1172_541_641
- GAPS384_train_1173_541_1
- GAPS384_train_1173_541_641
- GAPS384_train_1174_541_1
- GAPS384_train_1174_541_641
- GAPS384_train_1175_541_1
- GAPS384_train_1175_541_641
- GAPS384_train_1176_541_1

### mlt_unet.py source-truth
```python
def forward(self, x_t, t, image):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        # print(self.x_head(x).shape)
        # print(self.image_head(image).shape)
        h = self.x_head(x_t) + self.seg_head(image)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        bottom = h
        # Upsampling
        skip_count = len(hs)
        h_seg = []
        for layer in self.noisy_upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs[skip_count - 1]], dim=1)
                skip_count -= 1
                h = layer(h, temb)
                h_seg.append(h)
            else:
                h = layer(h, temb)
        assert skip_count == 0

        for layer in self.seg_upblocks:
            if isinstance(layer, ResBlock):
                bottom = torch.cat([bottom, hs.pop(), h_seg[skip_count]], dim=1)
                skip_count += 1
            bottom = layer(bottom, temb)
        h = self.noisy_tail(h)
        bottom = self.seg_tail(bottom)
        assert len(hs) == 0

        return h, bottom


if __name__ == '__main__':
    batch_size = 1
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1).to("cuda:0")
    x = torch.randn(batch_size, 1, 32, 32).to("cuda:0")
    image = torch.randn(batch_size, 4, 32, 32).to("cuda:0")
    t = torch.randint(1000, (batch_size, )).to("cuda:0")
    y1, y2 = model(x, t, image)
    print(y1.shape, y2.shape)


```
Source-truth: `return h, bottom` sau `h=self.noisy_tail(h)`, `bottom=self.seg_tail(bottom)` → output order = `(noisy_pred, seg_logits)`. Docs/comment khác source → source thắng.
### MeanFlow SILoss call
```python
def __call__(self, model, images, model_kwargs=None):
        """
        Compute MeanFlow loss function with bootstrap mechanism
        """
        if model_kwargs == None:
            model_kwargs = {}
        else:
            model_kwargs = model_kwargs.copy()

        batch_size = images.shape[0]
        device = images.device

        unconditional_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        if model_kwargs.get('y') is not None and self.label_dropout_prob > 0:
            y = model_kwargs['y'].clone()  
            batch_size = y.shape[0]
            num_classes = model.module.num_classes
            dropout_mask = torch.rand(batch_size, device=y.device) < self.label_dropout_prob
            
            y[dropout_mask] = num_classes
            model_kwargs['y'] = y
            unconditional_mask = dropout_mask  # Used for unconditional velocity computation

        # Sample time steps
        r, t = self.sample_time_steps(batch_size, device)

        noises = torch.randn_like(images)
        
        # Calculate interpolation and z_t
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        z_t = alpha_t * images + sigma_t * noises #(1-t) * images + t * noise
        
        # Calculate instantaneous velocity v_t 
        v_t = d_alpha_t * images + d_sigma_t * noises
        time_diff = (t - r).view(-1, 1, 1, 1)
                
        u_target = torch.zeros_like(v_t)
        
        u = model(z_t, r, t, **model_kwargs)
        
        
        # Check if CFG should be applied (exclude unconditional samples)
        cfg_time_mask = (t >= self.cfg_min_t) & (t <= self.cfg_max_t) & (~unconditional_mask)
        
        if model_kwargs.get('y') is not None and cfg_time_mask.any():
            # Split samples into CFG and non-CFG
            cfg_indices = torch.where(cfg_time_mask)[0]
            no_cfg_indices = torch.where(~cfg_time_mask)[0]
            
            u_target = torch.zeros_like(v_t)
            
            # Process CFG samples
            if len(cfg_indices) > 0:
                cfg_z_t = z_t[cfg_indices]
                cfg_v_t = v_t[cfg_indices]
                cfg_r = r[cfg_indices]
                cfg_t = t[cfg_indices]
                cfg_time_diff = time_diff[cfg_indices]
                
                cfg_kwargs = {}
                for k, v in model_kwargs.items():
                    if torch.is_tensor(v) and v.shape[0] == batch_size:
                        cfg_kwargs[k] = v[cfg_indices]
                    else:
                        cfg_kwargs[k] = v
                
                # Compute v_tilde for CFG samples
                cfg_y = cfg_kwargs.get('y')
                num_classes = model.module.num_classes
                
                cfg_z_t_batch = torch.cat([cfg_z_t, cfg_z_t], dim=0)
                cfg_t_batch = torch.cat([cfg_t, cfg_t], dim=0)
                cfg_t_end_batch = torch.cat([cfg_t, cfg_t], dim=0)
                cfg_y_batch = torch.cat([cfg_y, torch.full_like(cfg_y, num_classes)], dim=0)
                
                cfg_combined_kwargs = cfg_kwargs.copy()
                cfg_combined_kwargs['y'] = cfg_y_batch
                
                with torch.no_grad():
                    cfg_combined_u_at_t = model(cfg_z_t_batch, cfg_t_batch, cfg_t_end_batch, **cfg_combined_kwargs)
                    cfg_u_cond_at_t, cfg_u_uncond_at_t = torch.chunk(cfg_combined_u_at_t, 2, dim=0)
                    cfg_v_tilde = (self.cfg_omega * cfg_v_t + 
                            self.cfg_kappa * cfg_u_cond_at_t + 
                            (1 - self.cfg_omega - self.cfg_kappa) * cfg_u_uncond_at_t)
                
                # Compute JVP with CFG velocity
                def fn_current_cf
```
SILoss gọi `model(z_t, r, t, **model_kwargs)`; adapter compatible.
## 6. Commands đã chạy
- `find ... -maxdepth 4`
- `tree ... -L 4 fallback find`
- `git -C ... status`
- `git -C ... diff`
- `env torch/cuda check`
- `checkpoint strict load check`
- `copy audit artifacts`
- `generate AUDIT_FOR_CHATGPT.md`
### Environment
```
/home/hieulc/miniconda3/envs/pytorch_hieus/bin/python
1.13.1+cu116
True
NVIDIA GeForce RTX 3090

```
## 7. Logs chính
### smoke test output
Không có stdout artifact smoke test. Source in `scripts/smoke_test.py` sẽ print `SMOKE_TEST_PASS`, nhưng chưa có evidence run pass trong logs.
### training first 50
```
2026-05-22 09:00:15,499 crackmeanflow.train INFO Pairs=421  Train=336  Test=85
2026-05-22 09:00:20,381 crackmeanflow.train INFO Model params: 9.81M
2026-05-22 09:00:31,212 crackmeanflow.train INFO [step 10] total=1.3345 seg=1.3345 
2026-05-22 09:00:31,538 crackmeanflow.train INFO [step 20] total=1.3498 seg=1.3498 
2026-05-22 09:00:31,864 crackmeanflow.train INFO [step 30] total=1.3215 seg=1.3215 
2026-05-22 09:00:32,182 crackmeanflow.train INFO [step 40] total=1.3291 seg=1.3291 
2026-05-22 09:00:32,500 crackmeanflow.train INFO [step 50] total=1.3164 seg=1.3164 
2026-05-22 09:00:32,820 crackmeanflow.train INFO [step 60] total=1.3116 seg=1.3116 
2026-05-22 09:00:33,161 crackmeanflow.train INFO [step 70] total=1.3095 seg=1.3095 
2026-05-22 09:00:33,478 crackmeanflow.train INFO [step 80] total=1.2873 seg=1.2873 
2026-05-22 09:00:33,797 crackmeanflow.train INFO [step 90] total=1.2842 seg=1.2842 
2026-05-22 09:00:34,115 crackmeanflow.train INFO [step 100] total=1.2794 seg=1.2794 
2026-05-22 09:00:35,383 crackmeanflow.train INFO [step 110] total=1.2911 seg=1.2911 
2026-05-22 09:00:35,699 crackmeanflow.train INFO [step 120] total=1.2146 seg=1.2146 
2026-05-22 09:00:36,020 crackmeanflow.train INFO [step 130] total=1.2000 seg=1.2000 
2026-05-22 09:00:36,344 crackmeanflow.train INFO [step 140] total=1.2403 seg=1.2403 
2026-05-22 09:00:36,665 crackmeanflow.train INFO [step 150] total=1.2330 seg=1.2330 
2026-05-22 09:00:36,986 crackmeanflow.train INFO [step 160] total=1.2670 seg=1.2670 
2026-05-22 09:00:37,311 crackmeanflow.train INFO [step 170] total=1.2597 seg=1.2597 
2026-05-22 09:00:37,627 crackmeanflow.train INFO [step 180] total=1.2442 seg=1.2442 
2026-05-22 09:00:37,943 crackmeanflow.train INFO [step 190] total=1.2770 seg=1.2770 
2026-05-22 09:00:38,267 crackmeanflow.train INFO [step 200] total=1.2543 seg=1.2543 
2026-05-22 09:00:39,886 crackmeanflow.train INFO [step 210] total=1.2325 seg=1.2325 
2026-05-22 09:00:40,202 crackmeanflow.train INFO [step 220] total=1.1736 seg=1.1736 
2026-05-22 09:00:40,515 crackmeanflow.train INFO [step 230] total=1.2049 seg=1.2049 
2026-05-22 09:00:40,857 crackmeanflow.train INFO [step 240] total=1.2118 seg=1.2118 
2026-05-22 09:00:41,176 crackmeanflow.train INFO [step 250] total=1.2451 seg=1.2451 
2026-05-22 09:00:41,495 crackmeanflow.train INFO [step 260] total=1.1913 seg=1.1913 
2026-05-22 09:00:41,809 crackmeanflow.train INFO [step 270] total=1.2422 seg=1.2422 
2026-05-22 09:00:42,131 crackmeanflow.train INFO [step 280] total=1.2426 seg=1.2426 
2026-05-22 09:00:42,459 crackmeanflow.train INFO [step 290] total=1.2357 seg=1.2357 
2026-05-22 09:00:42,781 crackmeanflow.train INFO [step 300] total=1.2015 seg=1.2015 
2026-05-22 09:00:44,407 crackmeanflow.train INFO [step 310] total=1.2411 seg=1.2411 
2026-05-22 09:00:44,721 crackmeanflow.train INFO [step 320] total=1.2165 seg=1.2165 
2026-05-22 09:00:45,038 crackmeanflow.train INFO [step 330] total=1.2318 seg=1.2318 
2026-05-22 09:00:45,231 crackmeanflow.train INFO Epoch 0 done in 24.8s  avg_total=1.2667
2026-05-22 09:00:45,774 crackmeanflow.train INFO Quick eval F1=0.2723 (best=0.0000)
2026-05-22 09:00:46,724 crackmeanflow.train INFO New best F1=0.2723 saved.
2026-05-22 09:00:48,184 crackmeanflow.train INFO [step 340] total=1.2013 seg=1.2013 
2026-05-22 09:00:48,503 crackmeanflow.train INFO [step 350] total=1.1989 seg=1.1989 
2026-05-22 09:00:48,819 crackmeanflow.train INFO [step 360] total=1.1495 seg=1.1495 
2026-05-22 09:00:49,137 crackmeanflow.train INFO [step 370] total=1.1797 seg=1.1797 
2026-05-22 09:00:49,454 crackmeanflow.train INFO [step 380] total=1.1891 seg=1.1891 
2026-05-22 09:00:49,779 crackmeanflow.train INFO [step 390] total=1.1862 seg=1.1862 
2026-05-22 09:00:50,108 crackmeanflow.train INFO [step 400] total=1.1855 seg=1.1855 
2026-05-22 09:00:51,876 crackmeanflow.train INFO [step 410] total=1.1836 seg=1.1836 
2026-05-22 09:00:52,192 crackmeanflow.train INFO [step 420] total=1.1656 seg=1.1656 
2026-05-22 09:00:52,508 crackmeanflow.train INFO [step 430] total=1.1988 seg=1.1988 
2026-05-22 09:00:52,827 crackmeanflow.train INFO [step 440] total=1.1498 seg=1.1498 
2026-05-22 09:00:53,143 crackmeanflow.train INFO [step 450] total=1.1745 seg=1.1745 
```
### training last 50
```
2026-05-22 09:11:20,893 crackmeanflow.train INFO [step 12630] total=0.0928 seg=0.0928 
2026-05-22 09:11:21,218 crackmeanflow.train INFO [step 12640] total=0.0953 seg=0.0953 
2026-05-22 09:11:21,550 crackmeanflow.train INFO [step 12650] total=0.2948 seg=0.2948 
2026-05-22 09:11:21,918 crackmeanflow.train INFO [step 12660] total=0.0263 seg=0.0263 
2026-05-22 09:11:22,255 crackmeanflow.train INFO [step 12670] total=0.1811 seg=0.1811 
2026-05-22 09:11:22,578 crackmeanflow.train INFO [step 12680] total=0.0910 seg=0.0910 
2026-05-22 09:11:22,919 crackmeanflow.train INFO [step 12690] total=0.1863 seg=0.1863 
2026-05-22 09:11:23,270 crackmeanflow.train INFO [step 12700] total=0.0983 seg=0.0983 
2026-05-22 09:11:25,052 crackmeanflow.train INFO [step 12710] total=0.1305 seg=0.1305 
2026-05-22 09:11:25,402 crackmeanflow.train INFO [step 12720] total=0.2060 seg=0.2060 
2026-05-22 09:11:25,742 crackmeanflow.train INFO [step 12730] total=0.0503 seg=0.0503 
2026-05-22 09:11:26,088 crackmeanflow.train INFO [step 12740] total=0.0696 seg=0.0696 
2026-05-22 09:11:26,458 crackmeanflow.train INFO [step 12750] total=0.2348 seg=0.2348 
2026-05-22 09:11:26,807 crackmeanflow.train INFO [step 12760] total=0.0918 seg=0.0918 
2026-05-22 09:11:27,084 crackmeanflow.train INFO Epoch 37 done in 15.5s  avg_total=0.1382
2026-05-22 09:11:27,683 crackmeanflow.train INFO Quick eval F1=0.4750 (best=0.5061)
2026-05-22 09:11:29,249 crackmeanflow.train INFO [step 12770] total=0.1002 seg=0.1002 
2026-05-22 09:11:29,597 crackmeanflow.train INFO [step 12780] total=0.1099 seg=0.1099 
2026-05-22 09:11:29,929 crackmeanflow.train INFO [step 12790] total=0.0950 seg=0.0950 
2026-05-22 09:11:30,266 crackmeanflow.train INFO [step 12800] total=0.1186 seg=0.1186 
2026-05-22 09:11:32,019 crackmeanflow.train INFO [step 12810] total=0.1509 seg=0.1509 
2026-05-22 09:11:32,345 crackmeanflow.train INFO [step 12820] total=0.0850 seg=0.0850 
2026-05-22 09:11:32,705 crackmeanflow.train INFO [step 12830] total=0.0483 seg=0.0483 
2026-05-22 09:11:33,043 crackmeanflow.train INFO [step 12840] total=0.2286 seg=0.2286 
2026-05-22 09:11:33,412 crackmeanflow.train INFO [step 12850] total=0.1459 seg=0.1459 
2026-05-22 09:11:33,737 crackmeanflow.train INFO [step 12860] total=0.0791 seg=0.0791 
2026-05-22 09:11:34,081 crackmeanflow.train INFO [step 12870] total=0.0470 seg=0.0470 
2026-05-22 09:11:34,423 crackmeanflow.train INFO [step 12880] total=0.0953 seg=0.0953 
2026-05-22 09:11:34,752 crackmeanflow.train INFO [step 12890] total=0.0987 seg=0.0987 
2026-05-22 09:11:35,080 crackmeanflow.train INFO [step 12900] total=0.1126 seg=0.1126 
2026-05-22 09:11:36,827 crackmeanflow.train INFO [step 12910] total=0.1264 seg=0.1264 
2026-05-22 09:11:37,156 crackmeanflow.train INFO [step 12920] total=0.0910 seg=0.0910 
2026-05-22 09:11:37,496 crackmeanflow.train INFO [step 12930] total=0.3150 seg=0.3150 
2026-05-22 09:11:37,836 crackmeanflow.train INFO [step 12940] total=0.1454 seg=0.1454 
2026-05-22 09:11:38,172 crackmeanflow.train INFO [step 12950] total=0.1266 seg=0.1266 
2026-05-22 09:11:38,519 crackmeanflow.train INFO [step 12960] total=0.0512 seg=0.0512 
2026-05-22 09:11:38,881 crackmeanflow.train INFO [step 12970] total=0.2110 seg=0.2110 
2026-05-22 09:11:39,233 crackmeanflow.train INFO [step 12980] total=0.0821 seg=0.0821 
2026-05-22 09:11:39,562 crackmeanflow.train INFO [step 12990] total=0.1940 seg=0.1940 
2026-05-22 09:11:39,887 crackmeanflow.train INFO [step 13000] total=0.2561 seg=0.2561 
2026-05-22 09:11:41,741 crackmeanflow.train INFO [step 13010] total=0.2084 seg=0.2084 
2026-05-22 09:11:42,138 crackmeanflow.train INFO [step 13020] total=0.1355 seg=0.1355 
2026-05-22 09:11:42,494 crackmeanflow.train INFO [step 13030] total=0.0684 seg=0.0684 
2026-05-22 09:11:42,815 crackmeanflow.train INFO [step 13040] total=0.1842 seg=0.1842 
2026-05-22 09:11:43,136 crackmeanflow.train INFO [step 13050] total=0.0967 seg=0.0967 
2026-05-22 09:11:43,463 crackmeanflow.train INFO [step 13060] total=0.1854 seg=0.1854 
2026-05-22 09:11:43,795 crackmeanflow.train INFO [step 13070] total=0.2513 seg=0.2513 
2026-05-22 09:11:44,124 crackmeanflow.train INFO [step 13080] total=0.0979 seg=0.0979 
2026-05-22 09:11:44,459 crackmeanflow.train INFO [step 13090] total=0.1033 seg=0.1033 
2026-05-22 09:11:44,792 crackmeanflow.train INFO [step 13100] total=0.1406 seg=0.1406 
```
### train_last first/last
```
2026-05-22 00:59:42,749 crackmeanflow.train INFO Pairs=421  Train=336  Test=85
2026-05-22 00:59:47,925 crackmeanflow.train INFO Model params: 30.81M
Traceback (most recent call last):
  File "scripts/train_crackmeanflow.py", line 42, in <module>
    main()
  File "scripts/train_crackmeanflow.py", line 38, in main
    train(cfg)
  File "/home/hieulc/avitech11/crackmeanflow/crackmeanflow/train.py", line 131, in train
    total_loss, loss_dict = criterion(model, x0, {"y": image, "mask_gt": mask})
  File "/home/hieulc/miniconda3/envs/pytorch_hieus/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/hieulc/avitech11/crackmeanflow/crackmeanflow/loss.py", line 66, in forward
    si_loss_vec, si_loss_ref = self.si_loss(proxy, x0, {"y": y})
  File "/home/hieulc/avitech11/MeanFlow/loss.py", line 123, in __call__
    u = model(z_t, r, t, **model_kwargs)
  File "/home/hieulc/miniconda3/envs/pytorch_hieus/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/hieulc/avitech11/crackmeanflow/crackmeanflow/loss.py", line 30, in forward
    return self.model(*args, **kwargs)
  File "/home/hieulc/miniconda3/envs/pytorch_hieus/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/hieulc/avitech11/crackmeanflow/crackmeanflow/adapter.py", line 52, in forward
    velocity_pred, seg_logits = self.unet(x, t_int, y)
  File "/home/hieulc/miniconda3/envs/pytorch_hieus/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/hieulc/avitech11/crack_diff/crackdiff/multi_task/mlt_unet.py", line 277, in forward
    h = self.x_head(x_t) + self.seg_head(image)
  File "/home/hieulc/miniconda3/envs/pytorch_hieus/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/hieulc/miniconda3/envs/pytorch_hieus/lib/python3.7/site-packages/torch/nn/modules/container.py", line 204, in forward
---TAIL---
    return forward_call(*input, **kwargs)
  File "/home/hieulc/avitech11/crackmeanflow/crackmeanflow/loss.py", line 66, in forward
    si_loss_vec, si_loss_ref = self.si_loss(proxy, x0, {"y": y})
  File "/home/hieulc/avitech11/MeanFlow/loss.py", line 123, in __call__
    u = model(z_t, r, t, **model_kwargs)
  File "/home/hieulc/miniconda3/envs/pytorch_hieus/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/hieulc/avitech11/crackmeanflow/crackmeanflow/loss.py", line 30, in forward
    return self.model(*args, **kwargs)
  File "/home/hieulc/miniconda3/envs/pytorch_hieus/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/hieulc/avitech11/crackmeanflow/crackmeanflow/adapter.py", line 52, in forward
    velocity_pred, seg_logits = self.unet(x, t_int, y)
  File "/home/hieulc/miniconda3/envs/pytorch_hieus/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/hieulc/avitech11/crack_diff/crackdiff/multi_task/mlt_unet.py", line 277, in forward
    h = self.x_head(x_t) + self.seg_head(image)
  File "/home/hieulc/miniconda3/envs/pytorch_hieus/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/hieulc/miniconda3/envs/pytorch_hieus/lib/python3.7/site-packages/torch/nn/modules/container.py", line 204, in forward
    input = module(input)
  File "/home/hieulc/miniconda3/envs/pytorch_hieus/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/hieulc/avitech11/crack_diff/crackdiff/multi_task/mlt_unet.py", line 145, in forward
    out = self.RDB(x)
  File "/home/hieulc/miniconda3/envs/pytorch_hieus/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/hieulc/avitech11/crack_diff/crackdiff/multi_task/mlt_unet.py", line 135, in forward
    return x + self.b * out
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 MiB (GPU 0; 23.68 GiB total capacity; 795.23 MiB already allocated; 3.38 MiB free; 810.00 MiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```
### lỗi traceback/fix
- `fatal: not a git repository` → dùng `git -C` đúng path.
- `ModuleNotFoundError: torch` với system python → dùng `/home/hieulc/miniconda3/envs/pytorch_hieus/bin/python`.
- import path sai (`adapter`) → dùng package import `crackmeanflow.adapter`.
## 8. Checkpoint report
### strict load evidence
```
=== checkpoints ===
model {'missing': [], 'unexpected': []}
ema {'missing': [], 'unexpected': []}
=== checkpoints_long_endpoint ===
model {'missing': [], 'unexpected': []}
ema {'missing': [], 'unexpected': []}
=== checkpoints_v2 ===
model {'missing': [], 'unexpected': []}
ema {'missing': [], 'unexpected': []}
=== checkpoints_v3 ===
model {'missing': [], 'unexpected': []}
ema {'missing': [], 'unexpected': []}
=== checkpoints_v4_256 ===
model {'missing': [], 'unexpected': []}
ema {'missing': [], 'unexpected': []}
=== checkpoints_v5_256_ft ===
model {'missing': [], 'unexpected': []}
ema {'missing': [], 'unexpected': []}

```
### checkpoints/best.pt
- path: `/home/hieulc/avitech11/crackmeanflow/checkpoints/best.pt`
- keys: model, ema, optimizer, scheduler, epoch, global_step, best_metrics, config, architecture
- dùng model hay ema: eval artifacts dùng ema=true; checkpoint có cả model+ema
- NaN/Inf weights: audit checkpoint trước đó báo NaN=0, Inf=0
- strict load: pass, missing=[], unexpected=[]
- architecture config: T=500, ch=32, ch_mult=[1], attn=[], num_res_blocks=1, dropout=0.1
### checkpoints_long_endpoint/best.pt
- path: `/home/hieulc/avitech11/crackmeanflow/checkpoints_long_endpoint/best.pt`
- keys: model, ema, optimizer, scheduler, epoch, global_step, best_metrics, config, architecture
- dùng model hay ema: eval artifacts dùng ema=true; checkpoint có cả model+ema
- NaN/Inf weights: audit checkpoint trước đó báo NaN=0, Inf=0
- strict load: pass, missing=[], unexpected=[]
- architecture config: T=500, ch=32, ch_mult=[1], attn=[], num_res_blocks=1, dropout=0.1
### checkpoints_v2/best.pt
- path: `/home/hieulc/avitech11/crackmeanflow/checkpoints_v2/best.pt`
- keys: model, ema, optimizer, scheduler, epoch, global_step, best_metrics, config, architecture
- dùng model hay ema: eval artifacts dùng ema=true; checkpoint có cả model+ema
- NaN/Inf weights: audit checkpoint trước đó báo NaN=0, Inf=0
- strict load: pass, missing=[], unexpected=[]
- architecture config: T=500, ch=32, ch_mult=[1], attn=[], num_res_blocks=2, dropout=0.1
### checkpoints_v3/best.pt
- path: `/home/hieulc/avitech11/crackmeanflow/checkpoints_v3/best.pt`
- keys: model, ema, optimizer, scheduler, epoch, global_step, best_metrics, config, architecture
- dùng model hay ema: eval artifacts dùng ema=true; checkpoint có cả model+ema
- NaN/Inf weights: audit checkpoint trước đó báo NaN=0, Inf=0
- strict load: pass, missing=[], unexpected=[]
- architecture config: T=500, ch=32, ch_mult=[1, 2], attn=[], num_res_blocks=2, dropout=0.1
### checkpoints_v4_256/best.pt
- path: `/home/hieulc/avitech11/crackmeanflow/checkpoints_v4_256/best.pt`
- keys: model, ema, optimizer, scheduler, epoch, global_step, best_metrics, config, architecture
- dùng model hay ema: eval artifacts dùng ema=true; checkpoint có cả model+ema
- NaN/Inf weights: audit checkpoint trước đó báo NaN=0, Inf=0
- strict load: pass, missing=[], unexpected=[]
- architecture config: T=500, ch=32, ch_mult=[1, 2], attn=[], num_res_blocks=2, dropout=0.1
### checkpoints_v5_256_ft/best.pt
- path: `/home/hieulc/avitech11/crackmeanflow/checkpoints_v5_256_ft/best.pt`
- keys: model, ema, optimizer, scheduler, epoch, global_step, best_metrics, config, architecture
- dùng model hay ema: eval artifacts dùng ema=true; checkpoint có cả model+ema
- NaN/Inf weights: audit checkpoint trước đó báo NaN=0, Inf=0
- strict load: pass, missing=[], unexpected=[]
- architecture config: T=500, ch=32, ch_mult=[1, 2], attn=[], num_res_blocks=2, dropout=0.1
## 9. Kết quả test
Best metrics path: `/home/hieulc/avitech11/crackmeanflow/outputs/v5_256_sweep_-0.3/metrics.json`
```json
{
  "num_samples": 85,
  "f1": 0.47674204365295525,
  "iou": 0.3251532674712293,
  "dice": 0.47674204365295525,
  "precision": 0.48051613385186476,
  "recall": 0.5416545487940312,
  "accuracy": 0.9866727941176471,
  "thin_recall": 0.30105665834511025,
  "thin_precision": 0.24871336618328796,
  "thin_f1": 0.2633731780245024,
  "boundary_f1": 0.2633731780245024,
  "latency_seconds": 0.040153308763333104,
  "throughput_fps": 24.904547864139467,
  "num_steps": 1,
  "threshold": -0.3,
  "checkpoint": "/home/hieulc/avitech11/crackmeanflow/checkpoints_v5_256_ft/best.pt",
  "use_ema": true,
  "per_sample": [
    {
      "iou": 0.5593952536582947,
      "dice": 0.7174513936042786,
      "f1": 0.7174513936042786,
      "precision": 0.7551020383834839,
      "recall": 0.6833773255348206,
      "accuracy": 0.9937744140625,
      "tp": 518.0,
      "fp": 168.0,
      "fn": 240.0,
      "tn": 64610.0,
      "thin_recall": 0.5039682388305664,
      "thin_precision": 0.5404255390167236,
      "thin_f1": 0.521560549736023,
      "boundary_f1": 0.521560549736023,
      "recall_thin": 0.5039682388305664,
      "f1_thin": 0.521560549736023,
      "dice_thin": 0.521560549736023,
      "name": "GAPS384_train_1171_541_641"
    },
    {
      "iou": 0.44181033968925476,
      "dice": 0.6128549575805664,
      "f1": 0.6128549575805664,
      "precision": 0.5276705026626587,
      "recall": 0.7308377623558044,
      "accuracy": 0.992095947265625,
      "tp": 410.0,
      "fp": 367.0,
      "fn": 151.0,
      "tn": 64608.0,
      "thin_recall": 0.4117647111415863,
      "thin_precision": 0.3769230842590332,
      "thin_f1": 0.39357423782348633,
      "boundary_f1": 0.39357423782348633,
      "recall_thin": 0.4117647111415863,
      "f1_thin": 0.39357423782348633,
      "dice_thin": 0.39357423782348633,
      "name": "GAPS384_train_1172_541_1"
    },
    {
      "iou": 0.468544602394104,
      "dice": 0.6381073594093323,
      "f1": 0.6381073594093323,
      "precision": 0.6413881778717041,
      "recall": 0.6348600387573242,
      "accuracy": 0.991363525390625,
      "tp": 499.0,
      "fp": 279.0,
      "fn": 287.0,
      "tn": 64471.0,
      "thin_recall": 0.3818897604942322,
      "thin_precision": 0.3553113639354706,
      "thin_f1": 0.3681213855743408,
      "boundary_f1": 0.3681213855743408,
      "recall_thin": 0.3818897604942322,
      "f1_thin": 0.3681213855743408,
      "dice_thin": 0.3681213855743408,
      "name": "GAPS384_train_1172_541_641"
    },
    {
      "iou": 0.5431619882583618,
      "dice": 0.7039597034454346,
      "f1": 0.7039597034454346,
      "precision": 0.7745504975318909,
      "recall": 0.6451612710952759,
      "accuracy": 0.9928131103515625,
      "tp": 560.0,
      "fp": 163.0,
      "fn": 308.0,
      "tn": 64505.0,
      "thin_recall": 0.4730769097805023,
      "thin_precision": 0.48616600036621094,
      "thin_f1": 0.4795320928096771,
      "boundary_f1": 0.4795320928096771,
      "recall_thin": 0.4730769097805023,
      "f1_thin": 0.4795320928096771,
      "dice_thin": 0.4795320928096771,
      "name": "GAPS384_train_1173_541_1"
    },
    {
      "iou": 0.4545454680919647,
      "dice": 0.6249999403953552,
      "f1": 0.6249999403953552,
      "precision": 0.6126482486724854,
      "recall": 0.6378600597381592,
      "accuracy": 0.991485595703125,
      "tp": 465.0,
      "fp": 294.0,
      "fn": 264.0,
      "tn": 64513.0,
      "thin_recall": 0.45783132314682007,
      "thin_precision": 0.4488188922405243,
      "thin_f1": 0.4532802700996399,
      "boundary_f1": 0.4532802700996399,
      "recall_thin": 0.45783132314682007,
      "f1_thin": 0.4532802700996399,
      "dice_thin": 0.4532802700996399,
      "name": "GAPS384_train_1173_541_641"
    },
    {
      "iou": 0.5418820977210999,
      "dice": 0.7028838992118835,
      "f1": 0.7028838992118835,
      "precision": 0.6517412662506104,
      "recall": 0.7627365589141846,
      "accuracy": 0.9932403564453125,
      "tp": 524.0,
      "fp": 280.0,
      "fn": 163.0,
      "tn": 64569.0,
      "thin_recall": 0.5151515007019043,
      "thin_precision": 0.444029837846756,
      "thin_f1": 0.4769538640975952,
      "boundary_f1": 0.4769538640975952,
      "recall_thin": 0.5151515007019043,
      "f1_thin": 0.4769538640975952,
      "dice_thin": 0.4769538640975952,
      "name": "GAPS384_train_1174_541_1"
    },
    {
      "iou": 0.45272207260131836,
      "dice": 0.6232740879058838,
      "f1": 0.6232740879058838,
      "precision": 0.6396760940551758,
      "recall": 0.607692301273346,
      "accuracy": 0.9912567138671875,
      "tp": 474.0,
      "fp": 267.0,
      "fn": 306.0,
      "tn": 64489.0,
      "thin_recall": 0.31578946113586426,
      "thin_precision": 0.32231405377388,
      "thin_f1": 0.3190183639526367,
      "boundary_f1": 0.3190183639526367,
      "recall_thin": 0.31578946113586426,
      "f1_thin": 0.3190183639526367,
      "dice_thin": 0.3190183639526367,
      "name": "GAPS384_train_1174_541_641"
    },
    {
      "iou": 0.39243924617767334,
      "dice": 0.5636715888977051,
      "f1": 0.5636715888977051,
      "precision": 0.43512973189353943,
      "recall": 0.800000011920929,
      "accuracy": 0.9897003173828125,
      "tp": 436.0,
      "fp": 566.0,
      "fn": 109.0,
      "tn": 64425.0,
      "thin_recall": 0.4749034643173218,
      "thin_precision": 0.3388429880142212,
      "thin_f1": 0.3954983651638031,
      "boundary_f1": 0.3954983651638031,
      "recall_thin": 0.4749034643173218,
      "f1_thin": 0.3954983651638031,
      "dice_thin": 0.3954983651638031,
      "name": "GAPS384_train_1175_541_1"
    },
    {
      "iou": 0.4678111672401428,
      "dice": 0.6374268531799316,
      "f1": 0.6374268531799316,
      "precision": 0.5654993653297424,
      "recall": 0.7303182482719421,
      "accuracy": 0.992431640625,
      "tp": 436.0,
      "fp": 335.0,
      "fn": 161.0,
      "tn": 64604.0,
      "thin_recall": 0.5542168617248535,
      "thin_precision": 0.5390625,
      "thin_f1": 0.5465345978736877,
      "boundary_f1": 0.5465345978736877,
      "recall_thin": 0.5542168617248535,
      "f1_thin": 0.5465345978736877,
      "dice_thin": 0.5465345978736877,
      "name": "GAPS384_train_1175_541_641"
    },
    {
      "iou": 0.2028985470533371,
      "dice": 0.3373493552207947,
      "f1": 0.3373493552207947,
      "precision": 0.2295081913471222,
      "recall": 0.6363636255264282,
      "accuracy": 0.986572265625,
      "tp": 224.0,
      "fp": 752.0,
      "fn": 128.0,
      "tn": 64432.0,
      "thin_recall": 0.4166666567325592,
      "thin_precision": 0.19938650727272034,
      "thin_f1": 0.2697094976902008,
      "boundary_f1": 0.2697094976902008,
      "recall_thin": 0.4166666567325592,
      "f1_thin": 0.2697094976902008,
      "dice_thin": 0.2697094976902008,
      "name": "GAPS384_train_1176_541_1"
    },
    {
      "iou": 0.22192224860191345,
      "dice": 0.36323457956314087,
      "f1": 0.36323457956314087,
      "precision": 0.26464906334877014,
      "recall": 0.5788732171058655,
      "accuracy": 0.9780120849609375,
      "tp": 411.0,
      "fp": 1142.0,
      "fn": 299.0,
      "tn": 63684.0,
      "thin_recall": 0.38910505175590515,
      "thin_precision": 0.17985612154006958,
      "thin_f1": 0.2460024207830429,
      "boundary_f1": 0.2460024207830429,
      "recall_thin": 0.38910505175590515,
      "f1_thin": 0.2460024207830429,
      "dice_thin": 0.2460024207830429,
      "name": "GAPS384_train_1176_541_641"
    },
    {
      "iou": 0.43799999356269836,
      "dice": 0.6091793179512024,
      "f1": 0.6091793179512024,
      "precision": 0.5302663445472717,
      "recall": 0.7156862616539001,
      "accuracy": 0.991424560546875,
      "tp": 438.0,
      "fp": 388.0,
      "fn": 174.0,
      "tn": 64536.0,
      "thin_recall": 0.4392523467540741,
      "thin_precision": 0.3418181836605072,
      "thin_f1": 0.3844580352306366,
      "boundary_f1": 0.3844580352306366,
      "recall_thin": 0.4392523467540741,
      "f1_thin": 0.3844580352306366,
      "dice_thin": 0.3844580352306366,
      "name": "GAPS384_train_1177_541_1"
    },
    {
      "iou": 0.4311688244342804,
      "dice": 0.6025407314300537,
      "f1": 0.6025407314300537,
      "precision": 0.5037935972213745,
      "recall": 0.7494356632232666,
      "accuracy": 0.993316650390625,
      "tp": 332.0,
      "fp": 327.0,
      "fn": 111.0,
      "tn": 64766.0,
      "thin_recall": 0.3206106722354889,
      "thin_precision": 0.20388349890708923,
      "thin_f1": 0.2492581158876419,
      "boundary_f1": 0.2492581158876419,
      "recall_thin": 0.3206106722354889,
      "f1_thin": 0.2492581158876419,
      "dice_thin": 0.2492581158876419,
      "name": "GAPS384_train_1178_541_641"
    },
    {
      "iou": 0.4366632401943207,
      "dice": 0.6078852415084839,
      "f1": 0.6078852415084839,
      "precision": 0.5997171401977539,
      "recall": 0.6162790656089783,
      "accuracy": 0.9916534423828125,
      "tp": 424.0,
      "fp": 283.0,
      "fn": 264.0,
      "tn": 64565.0,
      "thin_recall": 0.4392523467540741,
      "thin_precision": 0.3933054506778717,
      "thin_f1": 0.4150109887123108,
      "boundary_f1": 0.4150109887123108,
      "recall_thin": 0.4392523467540741,
      "f1_thin": 0.4150109887123108,
      "dice_thin": 0.4150109887123108,
      "name": "GAPS384_train_1179_541_1"
    },
    {
      "iou": 0.43795621395111084,
      "dice": 0.6091369986534119,
      "f1": 0.6091369986534119,
      "precision": 0.6498194932937622,
      "recall": 0.5732483863830566,
      "accuracy": 0.992950439453125,
      "tp": 360.0,
      "fp": 194.0,
      "fn": 268.0,
      "tn": 64714.0,
      "thin_recall": 0.2720000147819519,
      "thin_precision": 0.4071856141090393,
      "thin_f1": 0.32613903284072876,
      "boundary_f1": 0.32613903284072876,
      "recall_thin": 0.2720000147819519,
      "f1_thin": 0.32613903284072876,
      "dice_thin": 0.32613903284072876,
      "name": "GAPS384_train_1180_541_1"
    },
    {
      "iou": 0.5743727684020996,
      "dice": 0.7296527624130249,
      "f1": 0.7296527624130249,
      "precision": 0.8313878178596497,
      "recall": 0.6501014232635498,
      "accuracy": 0.9927520751953125,
      "tp": 641.0,
      "fp": 130.0,
      "fn": 345.0,
      "tn": 64420.0,
      "thin_recall": 0.4803149700164795,
      "thin_precision": 0.4728682041168213,
      "thin_f1": 0.4765624403953552,
      "boundary_f1": 0.4765624403953552,
      "recall_thin": 0.4803149700164795,
      "f1_thin": 0.4765624403953552,
      "dice_thin": 0.4765624403953552,
      "name": "GAPS384_train_1180_541_641"
    },
    {
      "iou": 0.5309917330741882,
      "dice": 0.6936571598052979,
      "f1": 0.6936571598052979,
      "precision": 0.6283618807792664,
      "recall": 0.7740963697433472,
      "accuracy": 0.993072509765625,
      "tp": 514.0,
      "fp": 304.0,
      "fn": 150.0,
      "tn": 64568.0,
      "thin_recall": 0.5040000081062317,
      "thin_precision": 0.47191011905670166,
      "thin_f1": 0.48742741346359253,
      "boundary_f1": 0.48742741346359253,
      "recall_thin": 0.5040000081062317,
      "f1_thin": 0.48742741346359253,
      "dice_thin": 0.48742741346359253,
      "name": "GAPS384_train_1181_541_1"
    },
    {
      "iou": 0.44595909118652344,
      "dice": 0.6168349981307983,
      "f1": 0.6168349981307983,
      "precision": 0.5426540374755859,
      "recall": 0.714508593082428,
      "accuracy": 0.9913177490234375,
      "tp": 458.0,
      "fp": 386.0,
      "fn": 183.0,
      "tn": 64509.0,
      "thin_recall": 0.37826088070869446,
      "thin_precision": 0.30633804202079773,
      "thin_f1": 0.3385213613510132,
      "boundary_f1": 0.3385213613510132,
      "recall_thin": 0.37826088070869446,
      "f1_thin": 0.3385213613510132,
      "dice_thin": 0.3385213613510132,
      "name": "GAPS384_train_1181_541_641"
    },
    {
      "iou": 0.48734769225120544,
      "dice": 0.6553244590759277,
      "f1": 0.6553244590759277,
      "precision": 0.7142857313156128,
      "recall": 0.6053550839424133,
      "accuracy": 0.9916534423828125,
      "tp": 520.0,
      "fp": 208.0,
      "fn": 339.0,
      "tn": 64469.0,
      "thin_recall": 0.33992093801498413,
      "thin_precision": 0.3963133692741394,
      "thin_f1": 0.3659573793411255,
      "boundary_f1": 0.3659573793411255,
      "recall_thin": 0.33992093801498413,
      "f1_thin": 0.3659573793411255,
      "dice_thin": 0.3659573793411255,
      "name": "GAPS384_train_1182_541_1"
    },
    {
      "iou": 0.4836750328540802,
      "dice": 0.651995837688446,
      "f1": 0.651995837688446,
      "precision": 0.6691176295280457,
      "recall": 0.6357285380363464,
      "accuracy": 0.9896240234375,
      "tp": 637.0,
      "fp": 315.0,
      "fn": 365.0,
      "tn": 64219.0,
      "thin_recall": 0.43983402848243713,
      "thin_precision": 0.36054420471191406,
      "thin_f1": 0.396261602640152,
      "boundary_f1": 0.396261602640152,
      "recall_thin": 0.43983402848243713,
      "f1_thin": 0.396261602640152,
      "dice_thin": 0.396261602640152,
      "name": "GAPS384_train_1182_541_641"
    },
    {
      "iou": 0.3240309953689575,
      "dice": 0.4894613027572632,
      "f1": 0.4894613027572632,
      "precision": 0.3609671890735626,
      "recall": 0.7599999904632568,
      "accuracy": 0.99334716796875,
      "tp": 209.0,
      "fp": 370.0,
      "fn": 66.0,
      "tn": 64891.0,
      "thin_recall": 0.5,
      "thin_precision": 0.41545894742012024,
      "thin_f1": 0.45382583141326904,
      "boundary_f1": 0.45382583141326904,
      "recall_thin": 0.5,
      "f1_thin": 0.45382583141326904,
      "dice_thin": 0.45382583141326904,
      "name": "GAPS384_train_1183_541_1"
    },
    {
      "iou": 0.43440860509872437,
      "dice": 0.6056970953941345,
      "f1": 0.6056970953941345,
      "precision": 0.49753695726394653,
      "recall": 0.7739463448524475,
      "accuracy": 0.991973876953125,
      "tp": 404.0,
      "fp": 408.0,
      "fn": 118.0,
      "tn": 64606.0,
      "thin_recall": 0.4545454680919647,
      "thin_precision": 0.3298611044883728,
      "thin_f1": 0.3822937309741974,
      "boundary_f1": 0.3822937309741974,
      "recall_thin": 0.4545454680919647,
      "f1_thin": 0.3822937309741974,
      "dice_thin": 0.3822937309741974,
      "name": "GAPS384_train_1187_1_1"
    },
    {
      "iou": 0.21441124379634857,
      "dice": 0.3531113862991333,
      "f1": 0.3531113862991333,
      "precision": 0.2652173936367035,
      "recall": 0.5281385183334351,
      "accuracy": 0.986358642578125,
      "tp": 244.0,
      "fp": 676.0,
      "fn": 218.0,
      "tn": 64398.0,
      "thin_recall": 0.22360248863697052,
      "thin_precision": 0.11612903326749802,
      "thin_f1": 0.1528662145137787,
      "boundary_f1": 0.1528662145137787,
      "recall_thin": 0.22360248863697052,
      "f1_thin": 0.1528662145137787,
      "dice_thin": 0.1528662145137787,
      "name": "GAPS384_train_1190_1_1"
    },
    {
      "iou": 0.21299999952316284,
      "dice": 0.3511953353881836,
      "f1": 0.3511953353881836,
      "precision": 0.3169642984867096,
      "recall": 0.39371535181999207,
      "accuracy": 0.9879913330078125,
      "tp": 213.0,
      "fp": 459.0,
      "fn": 328.0,
      "tn": 64536.0,
      "thin_recall": 0.12432432174682617,
      "thin_precision": 0.1031390130519867,
      "thin_f1": 0.11274504661560059,
      "boundary_f1": 0.11274504661560059,
      "recall_thin": 0.12432432174682617,
      "f1_thin": 0.11274504661560059,
      "dice_thin": 0.11274504661560059,
      "name": "GAPS384_train_1190_1_641"
    },
    {
      "iou": 0.26482534408569336,
      "dice": 0.4187539517879486,
      "f1": 0.4187539517879486,
      "precision": 0.29237666726112366,
      "recall": 0.7375565767288208,
      "accuracy": 0.9861907958984375,
      "tp": 326.0,
      "fp": 789.0,
      "fn": 116.0,
      "tn": 64305.0,
      "thin_recall": 0.47337278723716736,
      "thin_precision": 0.2185792326927185,
      "thin_f1": 0.29906538128852844,
      "boundary_f1": 0.29906538128852844,
      "recall_thin": 0.47337278723716736,
      "f1_thin": 0.29906538128852844,
      "dice_thin": 0.29906538128852844,
      "name": "GAPS384_train_1201_1_1"
    },
    {
      "iou": 0.27272728085517883,
      "dice": 0.4285713732242584,
      "f1": 0.4285713732242584,
      "precision": 0.3232758641242981,
      "recall": 0.6355932354927063,
      "accuracy": 0.993896484375,
      "tp": 150.0,
      "fp": 314.0,
      "fn": 86.0,
      "tn": 64986.0,
      "thin_recall": 0.2800000011920929,
      "thin_precision": 0.17073170840740204,
      "thin_f1": 0.2121211737394333,
      "boundary_f1": 0.2121211737394333,
      "recall_thin": 0.2800000011920929,
      "f1_thin": 0.2121211737394333,
      "dice_thin": 0.2121211737394333,
      "name": "GAPS384_train_1218_541_1"
    },
    {
      "iou": 0.38545453548431396,
      "dice": 0.5564303994178772,
      "f1": 0.5564303994178772,
      "precision": 0.42655935883522034,
      "recall": 0.800000011920929,
      "accuracy": 0.994842529296875,
      "tp": 212.0,
      "fp": 285.0,
      "fn": 53.0,
      "tn": 64986.0,
      "thin_recall": 0.4516128897666931,
      "thin_precision": 0.2545454502105713,
      "thin_f1": 0.3255813419818878,
      "boundary_f1": 0.3255813419818878,
      "recall_thin": 0.4516128897666931,
      "f1_thin": 0.3255813419818878,
      "dice_thin": 0.3255813419818878,
      "name": "GAPS384_train_1226_1_1"
    },
    {
      "iou": 0.4790665805339813,
      "dice": 0.6477957963943481,
      "f1": 0.6477957963943481,
      "precision": 0.7949886322021484,
      "recall": 0.5465936064720154,
      "accuracy": 0.9884185791015625,
      "tp": 698.0,
      "fp": 180.0,
      "fn": 579.0,
      "tn": 64079.0,
      "thin_recall": 0.3281853199005127,
      "thin_precision": 0.3219696879386902,
      "thin_f1": 0.32504773139953613,
      "boundary_f1": 0.32504773139953613,
      "recall_thin": 0.3281853199005127,
      "f1_thin": 0.32504773139953613,
      "dice_thin": 0.32504773139953613,
      "name": "GAPS384_train_1226_541_1"
    },
    {
      "iou": 0.29551631212234497,
      "dice": 0.4562138617038727,
      "f1": 0.4562138617038727,
      "precision": 0.6807511448860168,
      "recall": 0.3430599272251129,
      "accuracy": 0.968353271484375,
      "tp": 870.0,
      "fp": 408.0,
      "fn": 1666.0,
      "tn": 62592.0,
      "thin_recall": 0.14000000059604645,
      "thin_precision": 0.09536784887313843,
      "thin_f1": 0.11345214396715164,
      "boundary_f1": 0.11345214396715164,
      "recall_thin": 0.14000000059604645,
      "f1_thin": 0.11345214396715164,
      "dice_thin": 0.11345214396715164,
      "name": "GAPS384_train_1227_541_641"
    },
    {
      "iou": 0.2922222316265106,
      "dice": 0.4522785544395447,
      "f1": 0.4522785544395447,
      "precision": 0.31124261021614075,
      "recall": 0.8270440101623535,
      "accuracy": 0.9902801513671875,
      "tp": 263.0,
      "fp": 582.0,
      "fn": 55.0,
      "tn": 64636.0,
      "thin_recall": 0.5087719559669495,
      "thin_precision": 0.4027777910232544,
      "thin_f1": 0.4496123790740967,
      "boundary_f1": 0.4496123790740967,
      "recall_thin": 0.5087719559669495,
      "f1_thin": 0.4496123790740967,
      "dice_thin": 0.4496123790740967,
      "name": "GAPS384_train_1230_1_1"
    },
    {
      "iou": 0.4111027717590332,
      "dice": 0.5826687216758728,
      "f1": 0.5826687216758728,
      "precision": 0.7988338470458984,
      "recall": 0.45857739448547363,
      "accuracy": 0.976043701171875,
      "tp": 1096.0,
      "fp": 276.0,
      "fn": 1294.0,
      "tn": 62870.0,
      "thin_recall": 0.18571428954601288,
      "thin_precision": 0.14565826952457428,
      "thin_f1": 0.16326525807380676,
      "boundary_f1": 0.16326525807380676,
      "recall_thin": 0.18571428954601288,
      "f1_thin": 0.16326525807380676,
      "dice_thin": 0.16326525807380676,
      "name": "GAPS384_train_1238_541_1"
    },
    {
      "iou": 0.03730115294456482,
      "dice": 0.07191957533359528,
      "f1": 0.07191957533359528,
      "precision": 0.12408759444952011,
      "recall": 0.050632912665605545,
      "accuracy": 0.9732208251953125,
      "tp": 68.0,
      "fp": 480.0,
      "fn": 1275.0,
      "tn": 63713.0,
      "thin_recall": 0.0,
      "thin_precision": 0.0,
      "thin_f1": 0.0,
      "boundary_f1": 0.0,
      "recall_thin": 0.0,
      "f1_thin": 0.0,
      "dice_thin": 0.0,
      "name": "GAPS384_train_1238_541_641"
    },
    {
      "iou": 0.24278438091278076,
      "dice": 0.3907102942466736,
      "f1": 0.3907102942466736,
      "precision": 0.31707316637039185,
      "recall": 0.5088967680931091,
      "accuracy": 0.98638916015625,
      "tp": 286.0,
      "fp": 616.0,
      "fn": 276.0,
      "tn": 64358.0,
      "thin_recall": 0.24867725372314453,
      "thin_precision": 0.15015974640846252,
      "thin_f1": 0.18725094199180603,
      "boundary_f1": 0.18725094199180603,
      "recall_thin": 0.24867725372314453,
      "f1_thin": 0.18725094199180603,
      "dice_thin": 0.18725094199180603,
      "name": "GAPS384_train_1241_1_1"
    },
    {
      "iou": 0.14125753939151764,
      "dice": 0.24754713475704193,
      "f1": 0.24754713475704193,
      "precision": 0.149226576089859,
      "recall": 0.7256637215614319,
      "accuracy": 0.9847869873046875,
      "tp": 164.0,
      "fp": 935.0,
      "fn": 62.0,
      "tn": 64375.0,
      "thin_recall": 0.3417721390724182,
      "thin_precision": 0.0787172019481659,
      "thin_f1": 0.12796205282211304,
      "boundary_f1": 0.12796205282211304,
      "recall_thin": 0.3417721390724182,
      "f1_thin": 0.12796205282211304,
      "dice_thin": 0.12796205282211304,
      "name": "GAPS384_train_1241_541_1"
    },
    {
      "iou": 0.22278910875320435,
      "dice": 0.3643949329853058,
      "f1": 0.3643949329853058,
      "precision": 0.258382648229599,
      "recall": 0.6179245114326477,
      "accuracy": 0.986053466796875,
      "tp": 262.0,
      "fp": 752.0,
      "fn": 162.0,
      "tn": 64360.0,
      "thin_recall": 0.30645161867141724,
      "thin_precision": 0.05775076150894165,
      "thin_f1": 0.09718668460845947,
      "boundary_f1": 0.09718668460845947,
      "recall_thin": 0.30645161867141724,
      "f1_thin": 0.09718668460845947,
      "dice_thin": 0.09718668460845947,
      "name": "GAPS384_train_1242_541_641"
    },
    {
      "iou": 0.3199152648448944,
      "dice": 0.4847511649131775,
      "f1": 0.4847511649131775,
      "precision": 0.6048064231872559,
      "recall": 0.4044642746448517,
      "accuracy": 0.9853057861328125,
      "tp": 453.0,
      "fp": 296.0,
      "fn": 667.0,
      "tn": 64120.0,
      "thin_recall": 0.11467889696359634,
      "thin_precision": 0.10822510719299316,
      "thin_f1": 0.11135852336883545,
      "boundary_f1": 0.11135852336883545,
      "recall_thin": 0.11467889696359634,
      "f1_thin": 0.11135852336883545,
      "dice_thin": 0.11135852336883545,
      "name": "GAPS384_train_1246_541_1"
    },
    {
      "iou": 0.30811807513237,
      "dice": 0.4710859954357147,
      "f1": 0.4710859954357147,
      "precision": 0.38041001558303833,
      "recall": 0.6185185313224792,
      "accuracy": 0.9942779541015625,
      "tp": 167.0,
      "fp": 272.0,
      "fn": 103.0,
      "tn": 64994.0,
      "thin_recall": 0.27272728085517883,
      "thin_precision": 0.13740457594394684,
      "thin_f1": 0.18274107575416565,
      "boundary_f1": 0.18274107575416565,
      "recall_thin": 0.27272728085517883,
      "f1_thin": 0.18274107575416565,
      "dice_thin": 0.18274107575416565,
      "name": "GAPS384_train_1247_541_1"
    },
    {
      "iou": 0.27149680256843567,
      "dice": 0.4270506501197815,
      "f1": 0.4270506501197815,
      "precision": 0.6024734973907471,
      "recall": 0.3307468593120575,
      "accuracy": 0.9860382080078125,
      "tp": 341.0,
      "fp": 225.0,
      "fn": 690.0,
      "tn": 64280.0,
      "thin_recall": 0.13750000298023224,
      "thin_precision": 0.18232044577598572,
      "thin_f1": 0.1567695438861847,
      "boundary_f1": 0.1567695438861847,
      "recall_thin": 0.13750000298023224,
      "f1_thin": 0.1567695438861847,
      "dice_thin": 0.1567695438861847,
      "name": "GAPS384_train_1247_541_641"
    },
    {
      "iou": 0.11139240860939026,
      "dice": 0.20045553147792816,
      "f1": 0.20045553147792816,
      "precision": 0.1875,
      "recall": 0.21533441543579102,
      "accuracy": 0.967864990234375,
      "tp": 264.0,
      "fp": 1144.0,
      "fn": 962.0,
      "tn": 63166.0,
      "thin_recall": 0.07636363804340363,
      "thin_precision": 0.04430379718542099,
      "thin_f1": 0.05607472360134125,
      "boundary_f1": 0.05607472360134125,
      "recall_thin": 0.07636363804340363,
      "f1_thin": 0.05607472360134125,
      "dice_thin": 0.05607472360134125,
      "name": "GAPS384_train_1248_1_1"
    },
    {
      "iou": 0.0850253775715828,
      "dice": 0.15672510862350464,
      "f1": 0.15672510862350464,
      "precision": 0.13373252749443054,
      "recall": 0.18926553428173065,
      "accuracy": 0.977996826171875,
      "tp": 134.0,
      "fp": 868.0,
      "fn": 574.0,
      "tn": 63960.0,
      "thin_recall": 0.11827956885099411,
      "thin_precision": 0.0662650614976883,
      "thin_f1": 0.0849420428276062,
      "boundary_f1": 0.0849420428276062,
      "recall_thin": 0.11827956885099411,
      "f1_thin": 0.0849420428276062,
      "dice_thin": 0.0849420428276062,
      "name": "GAPS384_train_1248_1_641"
    },
    {
      "iou": 0.24128234386444092,
      "dice": 0.3887629806995392,
      "f1": 0.3887629806995392,
      "precision": 0.5190562605857849,
      "recall": 0.31075698137283325,
      "accuracy": 0.958831787109375,
      "tp": 858.0,
      "fp": 795.0,
      "fn": 1903.0,
      "tn": 61980.0,
      "thin_recall": 0.13921569287776947,
      "thin_precision": 0.138671875,
      "thin_f1": 0.13894321024417877,
      "boundary_f1": 0.13894321024417877,
      "recall_thin": 0.13921569287776947,
      "f1_thin": 0.13894321024417877,
      "dice_thin": 0.13894321024417877,
      "name": "GAPS384_train_1248_541_641"
    },
    {
      "iou": 0.25027933716773987,
      "dice": 0.4003573954105377,
      "f1": 0.4003573954105377,
      "precision": 0.3922942280769348,
      "recall": 0.40875911712646484,
      "accuracy": 0.9897613525390625,
      "tp": 224.0,
      "fp": 347.0,
      "fn": 324.0,
      "tn": 64641.0,
      "thin_recall": 0.25,
      "thin_precision": 0.2590673565864563,
      "thin_f1": 0.2544528543949127,
      "boundary_f1": 0.2544528543949127,
      "recall_thin": 0.25,
      "f1_thin": 0.2544528543949127,
      "dice_thin": 0.2544528543949127,
      "name": "GAPS384_train_1281_541_1"
    },
    {
      "iou": 0.2819107174873352,
      "dice": 0.43982890248298645,
      "f1": 0.43982890248298645,
      "precision": 0.4938271641731262,
      "recall": 0.39647576212882996,
      "accuracy": 0.9860076904296875,
      "tp": 360.0,
      "fp": 369.0,
      "fn": 548.0,
      "tn": 64259.0,
      "thin_recall": 0.14399999380111694,
      "thin_precision": 0.07199999690055847,
      "thin_f1": 0.09599994868040085,
      "boundary_f1": 0.09599994868040085,
      "recall_thin": 0.14399999380111694,
      "f1_thin": 0.09599994868040085,
      "dice_thin": 0.09599994868040085,
      "name": "GAPS384_train_1281_541_641"
    },
    {
      "iou": 0.3519362211227417,
      "dice": 0.5206401944160461,
      "f1": 0.5206401944160461,
      "precision": 0.4131016135215759,
      "recall": 0.7038724422454834,
      "accuracy": 0.9913177490234375,
      "tp": 309.0,
      "fp": 439.0,
      "fn": 130.0,
      "tn": 64658.0,
      "thin_recall": 0.2679425776004791,
      "thin_precision": 0.2343096286058426,
      "thin_f1": 0.24999994039535522,
      "boundary_f1": 0.24999994039535522,
      "recall_thin": 0.2679425776004791,
      "f1_thin": 0.24999994039535522,
      "dice_thin": 0.24999994039535522,
      "name": "GAPS384_train_1284_541_1"
    },
    {
      "iou": 0.20484359562397003,
      "dice": 0.34003347158432007,
      "f1": 0.34003347158432007,
      "precision": 0.244283989071846,
      "recall": 0.5592286586761475,
      "accuracy": 0.98797607421875,
      "tp": 203.0,
      "fp": 628.0,
      "fn": 160.0,
      "tn": 64545.0,
      "thin_recall": 0.1596638709306717,
      "thin_precision": 0.0683453232049942,
      "thin_f1": 0.09571783989667892,
      "boundary_f1": 0.09571783989667892,
      "recall_thin": 0.1596638709306717,
      "f1_thin": 0.09571783989667892,
      "dice_thin": 0.09571783989667892,
      "name": "GAPS384_train_1284_541_641"
    },
    {
      "iou": 0.2544861435890198,
      "dice": 0.40572166442871094,
      "f1": 0.40572166442871094,
      "precision": 0.28312158584594727,
      "recall": 0.7155963182449341,
      "accuracy": 0.9930267333984375,
      "tp": 156.0,
      "fp": 395.0,
      "fn": 62.0,
      "tn": 64923.0,
      "thin_recall": 0.2467532455921173,
      "thin_precision": 0.10919540375471115,
      "thin_f1": 0.15139438211917877,
      "boundary_f1": 0.15139438211917877,
      "recall_thin": 0.2467532455921173,
      "f1_thin": 0.15139438211917877,
      "dice_thin": 0.15139438211917877,
      "name": "GAPS384_train_1319_1_1"
    },
    {
      "iou": 0.3963337540626526,
      "dice": 0.5676776170730591,
      "f1": 0.5676776170730591,
      "precision": 0.7712177038192749,
      "recall": 0.44914039969444275,
      "accuracy": 0.9854278564453125,
      "tp": 627.0,
      "fp": 186.0,
      "fn": 769.0,
      "tn": 63954.0,
      "thin_recall": 0.227424755692482,
      "thin_precision": 0.2753036320209503,
      "thin_f1": 0.24908418953418732,
      "boundary_f1": 0.24908418953418732,
      "recall_thin": 0.227424755692482,
      "f1_thin": 0.24908418953418732,
      "dice_thin": 0.24908418953418732,
      "name": "GAPS384_train_1319_1_641"
    },
    {
      "iou": 0.526605486869812,
      "dice": 0.689903736114502,
      "f1": 0.689903736114502,
      "precision": 0.694915235042572,
      "recall": 0.6849641799926758,
      "accuracy": 0.996063232421875,
      "tp": 287.0,
      "fp": 126.0,
      "fn": 132.0,
      "tn": 64991.0,
      "thin_recall": 0.45679011940956116,
      "thin_precision": 0.31092438101768494,
      "thin_f1": 0.3699999451637268,
      "boundary_f1": 0.3699999451637268,
      "recall_thin": 0.45679011940956116,
      "f1_thin": 0.3699999451637268,
      "dice_thin": 0.3699999451637268,
      "name": "GAPS384_train_1319_541_641"
    },
    {
      "iou": 0.05187074840068817,
      "dice": 0.09862565249204636,
      "f1": 0.09862565249204636,
      "precision": 0.12323231995105743,
      "recall": 0.0822102427482605,
      "accuracy": 0.9829864501953125,
      "tp": 61.0,
      "fp": 434.0,
      "fn": 681.0,
      "tn": 64360.0,
      "thin_recall": 0.0833333358168602,
      "thin_precision": 0.06508875638246536,
      "thin_f1": 0.07308965176343918,
      "boundary_f1": 0.07308965176343918,
      "recall_thin": 0.0833333358168602,
      "f1_thin": 0.07308965176343918,
      "dice_thin": 0.07308965176343918,
      "name": "GAPS384_train_1320_1_1"
    },
    {
      "iou": 0.3204951882362366,
      "dice": 0.48541662096977234,
      "f1": 0.48541662096977234,
      "precision": 0.7158218026161194,
      "recall": 0.3672182857990265,
      "accuracy": 0.98492431640625,
      "tp": 466.0,
      "fp": 185.0,
      "fn": 803.0,
      "tn": 64082.0,
      "thin_recall": 0.16538462042808533,
      "thin_precision": 0.23756906390190125,
      "thin_f1": 0.19501128792762756,
      "boundary_f1": 0.19501128792762756,
      "recall_thin": 0.16538462042808533,
      "f1_thin": 0.19501128792762756,
      "dice_thin": 0.19501128792762756,
      "name": "GAPS384_train_1321_1_641"
    },
    {
      "iou": 0.3898734152317047,
      "dice": 0.5610199570655823,
      "f1": 0.5610199570655823,
      "precision": 0.7633209228515625,
      "recall": 0.44348451495170593,
      "accuracy": 0.98529052734375,
      "tp": 616.0,
      "fp": 191.0,
      "fn": 773.0,
      "tn": 63956.0,
      "thin_recall": 0.29197078943252563,
      "thin_precision": 0.3238866329193115,
      "thin_f1": 0.30710166692733765,
      "boundary_f1": 0.30710166692733765,
      "recall_thin": 0.29197078943252563,
      "f1_thin": 0.30710166692733765,
      "dice_thin": 0.30710166692733765,
      "name": "GAPS384_train_1324_541_641"
    },
    {
      "iou": 0.1595303863286972,
      "dice": 0.2751637101173401,
      "f1": 0.2751637101173401,
      "precision": 0.47433266043663025,
      "recall": 0.1937919408082962,
      "accuracy": 0.9814300537109375,
      "tp": 231.0,
      "fp": 256.0,
      "fn": 961.0,
      "tn": 64088.0,
      "thin_recall": 0.13636364042758942,
      "thin_precision": 0.20000000298023224,
      "thin_f1": 0.16216212511062622,
      "boundary_f1": 0.16216212511062622,
      "recall_thin": 0.13636364042758942,
      "f1_thin": 0.16216212511062622,
      "dice_thin": 0.16216212511062622,
      "name": "GAPS384_train_1336_541_641"
    },
    {
      "iou": 0.2861468493938446,
      "dice": 0.4449675381183624,
      "f1": 0.4449675381183624,
      "precision": 0.40341514348983765,
      "recall": 0.4960629940032959,
      "accuracy": 0.9856109619140625,
      "tp": 378.0,
      "fp": 559.0,
      "fn": 384.0,
      "tn": 64215.0,
      "thin_recall": 0.32098764181137085,
      "thin_precision": 0.2300885021686554,
      "thin_f1": 0.2680411636829376,
      "boundary_f1": 0.2680411636829376,
      "recall_thin": 0.32098764181137085,
      "f1_thin": 0.2680411636829376,
      "dice_thin": 0.2680411636829376,
      "name": "GAPS384_train_1337_1_1"
    },
    {
      "iou": 0.22325581312179565,
      "dice": 0.3650189936161041,
      "f1": 0.3650189936161041,
      "precision": 0.2656826674938202,
      "recall": 0.5829959511756897,
      "accuracy": 0.9923553466796875,
      "tp": 144.0,
      "fp": 398.0,
      "fn": 103.0,
      "tn": 64891.0,
      "thin_recall": 0.21794871985912323,
      "thin_precision": 0.09189189225435257,
      "thin_f1": 0.1292775273323059,
      "boundary_f1": 0.1292775273323059,
      "recall_thin": 0.21794871985912323,
      "f1_thin": 0.1292775273323059,
      "dice_thin": 0.1292775273323059,
      "name": "GAPS384_train_1345_1_1"
    },
    {
      "iou": 0.1301482766866684,
      "dice": 0.23032066226005554,
      "f1": 0.23032066226005554,
      "precision": 0.26629212498664856,
      "recall": 0.20291095972061157,
      "accuracy": 0.975830078125,
      "tp": 237.0,
      "fp": 653.0,
      "fn": 931.0,
      "tn": 63715.0,
      "thin_recall": 0.1564885526895523,
      "thin_precision": 0.12974683940410614,
      "thin_f1": 0.1418684720993042,
      "boundary_f1": 0.1418684720993042,
      "recall_thin": 0.1564885526895523,
      "f1_thin": 0.1418684720993042,
      "dice_thin": 0.1418684720993042,
      "name": "GAPS384_train_1345_541_641"
    },
    {
      "iou": 0.2955580949783325,
      "dice": 0.4562636613845825,
      "f1": 0.4562636613845825,
      "precision": 0.5924657583236694,
      "recall": 0.37097927927970886,
      "accuracy": 0.9811248779296875,
      "tp": 519.0,
      "fp": 357.0,
      "fn": 880.0,
      "tn": 63780.0,
      "thin_recall": 0.21944443881511688,
      "thin_precision": 0.26599326729774475,
      "thin_f1": 0.2404870241880417,
      "boundary_f1": 0.2404870241880417,
      "recall_thin": 0.21944443881511688,
      "f1_thin": 0.2404870241880417,
      "dice_thin": 0.2404870241880417,
      "name": "GAPS384_train_1347_1_1"
    },
    {
      "iou": 0.20520402491092682,
      "dice": 0.3405298888683319,
      "f1": 0.3405298888683319,
      "precision": 0.48873239755630493,
      "recall": 0.2612951695919037,
      "accuracy": 0.9794921875,
      "tp": 347.0,
      "fp": 363.0,
      "fn": 981.0,
      "tn": 63845.0,
      "thin_recall": 0.1629955917596817,
      "thin_precision": 0.16818182170391083,
      "thin_f1": 0.16554805636405945,
      "boundary_f1": 0.16554805636405945,
      "recall_thin": 0.1629955917596817,
      "f1_thin": 0.16554805636405945,
      "dice_thin": 0.16554805636405945,
      "name": "GAPS384_train_1347_1_641"
    },
    {
      "iou": 0.19980119168758392,
      "dice": 0.33305710554122925,
      "f1": 0.33305710554122925,
      "precision": 0.21706263720989227,
      "recall": 0.7153024673461914,
      "accuracy": 0.9877166748046875,
      "tp": 201.0,
      "fp": 725.0,
      "fn": 80.0,
      "tn": 64530.0,
      "thin_recall": 0.4545454680919647,
      "thin_precision": 0.09708737581968307,
      "thin_f1": 0.15999996662139893,
      "boundary_f1": 0.15999996662139893,
      "recall_thin": 0.4545454680919647,
      "f1_thin": 0.15999996662139893,
      "dice_thin": 0.15999996662139893,
      "name": "GAPS384_train_1347_541_1"
    },
    {
      "iou": 0.19537274539470673,
      "dice": 0.32688167691230774,
      "f1": 0.32688167691230774,
      "precision": 0.41530054807662964,
      "recall": 0.26950353384017944,
      "accuracy": 0.9856719970703125,
      "tp": 228.0,
      "fp": 321.0,
      "fn": 618.0,
      "tn": 64369.0,
      "thin_recall": 0.1149425283074379,
      "thin_precision": 0.09900990128517151,
      "thin_f1": 0.10638292878866196,
      "boundary_f1": 0.10638292878866196,
      "recall_thin": 0.1149425283074379,
      "f1_thin": 0.10638292878866196,
      "dice_thin": 0.10638292878866196,
      "name": "GAPS384_train_1361_541_641"
    },
    {
      "iou": 0.4467765986919403,
      "dice": 0.617616593837738,
      "f1": 0.617616593837738,
      "precision": 0.717208206653595,
      "recall": 0.5423111915588379,
      "accuracy": 0.988739013671875,
      "tp": 596.0,
      "fp": 235.0,
      "fn": 503.0,
      "tn": 64202.0,
      "thin_recall": 0.3723404109477997,
      "thin_precision": 0.3547297418117523,
      "thin_f1": 0.3633217215538025,
      "boundary_f1": 0.3633217215538025,
      "recall_thin": 0.3723404109477997,
      "f1_thin": 0.3633217215538025,
      "dice_thin": 0.3633217215538025,
      "name": "GAPS384_train_1362_1_1"
    },
    {
      "iou": 0.4001706540584564,
      "dice": 0.5716025829315186,
      "f1": 0.5716025829315186,
      "precision": 0.5222716927528381,
      "recall": 0.6312247514724731,
      "accuracy": 0.9892730712890625,
      "tp": 469.0,
      "fp": 429.0,
      "fn": 274.0,
      "tn": 64364.0,
      "thin_recall": 0.34536081552505493,
      "thin_precision": 0.22945205867290497,
      "thin_f1": 0.27572008967399597,
      "boundary_f1": 0.27572008967399597,
      "recall_thin": 0.34536081552505493,
      "f1_thin": 0.27572008967399597,
      "dice_thin": 0.27572008967399597,
      "name": "GAPS384_train_1362_541_1"
    },
    {
      "iou": 0.5767195820808411,
      "dice": 0.7315435409545898,
      "f1": 0.7315435409545898,
      "precision": 0.7767220735549927,
      "recall": 0.6913319230079651,
      "accuracy": 0.99267578125,
      "tp": 654.0,
      "fp": 188.0,
      "fn": 292.0,
      "tn": 64402.0,
      "thin_recall": 0.40434783697128296,
      "thin_precision": 0.36328125,
      "thin_f1": 0.38271597027778625,
      "boundary_f1": 0.38271597027778625,
      "recall_thin": 0.40434783697128296,
      "f1_thin": 0.38271597027778625,
      "dice_thin": 0.38271597027778625,
      "name": "GAPS384_train_1364_1_1"
    },
    {
      "iou": 0.360425740480423,
      "dice": 0.529871940612793,
      "f1": 0.529871940612793,
      "precision": 0.69366854429245,
      "recall": 0.42865362763404846,
      "accuracy": 0.979827880859375,
      "tp": 745.0,
      "fp": 329.0,
      "fn": 993.0,
      "tn": 63469.0,
      "thin_recall": 0.2054794579744339,
      "thin_precision": 0.21551723778247833,
      "thin_f1": 0.21037864685058594,
      "boundary_f1": 0.21037864685058594,
      "recall_thin": 0.2054794579744339,
      "f1_thin": 0.21037864685058594,
      "dice_thin": 0.21037864685058594,
      "name": "GAPS384_train_1365_1_1"
    },
    {
      "iou": 0.3613824248313904,
      "dice": 0.53090500831604,
      "f1": 0.53090500831604,
      "precision": 0.6607142686843872,
      "recall": 0.44372692704200745,
      "accuracy": 0.987030029296875,
      "tp": 481.0,
      "fp": 247.0,
      "fn": 603.0,
      "tn": 64205.0,
      "thin_recall": 0.2074074000120163,
      "thin_precision": 0.21875,
      "thin_f1": 0.2129277139902115,
      "boundary_f1": 0.2129277139902115,
      "recall_thin": 0.2074074000120163,
      "f1_thin": 0.2129277139902115,
      "dice_thin": 0.2129277139902115,
      "name": "GAPS384_train_1365_1_641"
    },
    {
      "iou": 0.14691942930221558,
      "dice": 0.25619828701019287,
      "f1": 0.25619828701019287,
      "precision": 0.21678321063518524,
      "recall": 0.31313130259513855,
      "accuracy": 0.99176025390625,
      "tp": 93.0,
      "fp": 336.0,
      "fn": 204.0,
      "tn": 64903.0,
      "thin_recall": 0.21705426275730133,
      "thin_precision": 0.15909090638160706,
      "thin_f1": 0.18360650539398193,
      "boundary_f1": 0.18360650539398193,
      "recall_thin": 0.21705426275730133,
      "f1_thin": 0.18360650539398193,
      "dice_thin": 0.18360650539398193,
      "name": "GAPS384_train_1366_541_1"
    },
    {
      "iou": 0.3217317461967468,
      "dice": 0.4868335723876953,
      "f1": 0.4868335723876953,
      "precision": 0.566517174243927,
      "recall": 0.42680180072784424,
      "accuracy": 0.9878082275390625,
      "tp": 379.0,
      "fp": 290.0,
      "fn": 509.0,
      "tn": 64358.0,
      "thin_recall": 0.36213991045951843,
      "thin_precision": 0.3682008385658264,
      "thin_f1": 0.365145206451416,
      "boundary_f1": 0.365145206451416,
      "recall_thin": 0.36213991045951843,
      "f1_thin": 0.365145206451416,
      "dice_thin": 0.365145206451416,
      "name": "GAPS384_train_1370_1_1"
    },
    {
      "iou": 0.16252219676971436,
      "dice": 0.27960270643234253,
      "f1": 0.27960270643234253,
      "precision": 0.23106060922145844,
      "recall": 0.3539651930332184,
      "accuracy": 0.9856109619140625,
      "tp": 183.0,
      "fp": 609.0,
      "fn": 334.0,
      "tn": 64410.0,
      "thin_recall": 0.20000000298023224,
      "thin_precision": 0.032258063554763794,
      "thin_f1": 0.05555552989244461,
      "boundary_f1": 0.05555552989244461,
      "recall_thin": 0.20000000298023224,
      "f1_thin": 0.05555552989244461,
      "dice_thin": 0.05555552989244461,
      "name": "GAPS384_train_1370_541_1"
    },
    {
      "iou": 0.2891918122768402,
      "dice": 0.4486404061317444,
      "f1": 0.4486404061317444,
      "precision": 0.6200417280197144,
      "recall": 0.35147929191589355,
      "accuracy": 0.97772216796875,
      "tp": 594.0,
      "fp": 364.0,
      "fn": 1096.0,
      "tn": 63482.0,
      "thin_recall": 0.19148936867713928,
      "thin_precision": 0.1461038887500763,
      "thin_f1": 0.165745809674263,
      "boundary_f1": 0.165745809674263,
      "recall_thin": 0.19148936867713928,
      "f1_thin": 0.165745809674263,
      "dice_thin": 0.165745809674263,
      "name": "GAPS384_train_1370_541_641"
    },
    {
      "iou": 0.19241982698440552,
      "dice": 0.32273831963539124,
      "f1": 0.32273831963539124,
      "precision": 0.2612137198448181,
      "recall": 0.4221748411655426,
      "accuracy": 0.9873199462890625,
      "tp": 198.0,
      "fp": 560.0,
      "fn": 271.0,
      "tn": 64507.0,
      "thin_recall": 0.11764705926179886,
      "thin_precision": 0.031128404662013054,
      "thin_f1": 0.04923073574900627,
      "boundary_f1": 0.04923073574900627,
      "recall_thin": 0.11764705926179886,
      "f1_thin": 0.04923073574900627,
      "dice_thin": 0.04923073574900627,
      "name": "GAPS384_train_1374_1_641"
    },
    {
      "iou": 0.2013343870639801,
      "dice": 0.3351845443248749,
      "f1": 0.3351845443248749,
      "precision": 0.8814433217048645,
      "recall": 0.20693828165531158,
      "accuracy": 0.9689483642578125,
      "tp": 513.0,
      "fp": 69.0,
      "fn": 1966.0,
      "tn": 62988.0,
      "thin_recall": 0.1631578952074051,
      "thin_precision": 0.21088434755802155,
      "thin_f1": 0.1839762181043625,
      "boundary_f1": 0.1839762181043625,
      "recall_thin": 0.1631578952074051,
      "f1_thin": 0.1839762181043625,
      "dice_thin": 0.1839762181043625,
      "name": "GAPS384_train_1374_541_1"
    },
    {
      "iou": 0.3021315932273865,
      "dice": 0.4640568792819977,
      "f1": 0.4640568792819977,
      "precision": 0.6423645615577698,
      "recall": 0.36323121190071106,
      "accuracy": 0.977020263671875,
      "tp": 652.0,
      "fp": 363.0,
      "fn": 1143.0,
      "tn": 63378.0,
      "thin_recall": 0.25438597798347473,
      "thin_precision": 0.19205297529697418,
      "thin_f1": 0.21886788308620453,
      "boundary_f1": 0.21886788308620453,
      "recall_thin": 0.25438597798347473,
      "f1_thin": 0.21886788308620453,
      "dice_thin": 0.21886788308620453,
      "name": "GAPS384_train_1374_541_641"
    },
    {
      "iou": 0.21585902571678162,
      "dice": 0.35507240891456604,
      "f1": 0.35507240891456604,
      "precision": 0.25925925374031067,
      "recall": 0.5632184147834778,
      "accuracy": 0.9891357421875,
      "tp": 196.0,
      "fp": 560.0,
      "fn": 152.0,
      "tn": 64628.0,
      "thin_recall": 0.14084507524967194,
      "thin_precision": 0.03937007859349251,
      "thin_f1": 0.06153842806816101,
      "boundary_f1": 0.06153842806816101,
      "recall_thin": 0.14084507524967194,
      "f1_thin": 0.06153842806816101,
      "dice_thin": 0.06153842806816101,
      "name": "GAPS384_train_1375_541_1"
    },
    {
      "iou": 0.18424147367477417,
      "dice": 0.31115517020225525,
      "f1": 0.31115517020225525,
      "precision": 0.45323047041893005,
      "recall": 0.23689515888690948,
      "accuracy": 0.9682464599609375,
      "tp": 470.0,
      "fp": 567.0,
      "fn": 1514.0,
      "tn": 62985.0,
      "thin_recall": 0.1037735864520073,
      "thin_precision": 0.06586826592683792,
      "thin_f1": 0.0805860310792923,
      "boundary_f1": 0.0805860310792923,
      "recall_thin": 0.1037735864520073,
      "f1_thin": 0.0805860310792923,
      "dice_thin": 0.0805860310792923,
      "name": "GAPS384_train_1375_541_641"
    },
    {
      "iou": 0.3483935594558716,
      "dice": 0.5167534947395325,
      "f1": 0.5167534947395325,
      "precision": 0.41756919026374817,
      "recall": 0.677734375,
      "accuracy": 0.9900970458984375,
      "tp": 347.0,
      "fp": 484.0,
      "fn": 165.0,
      "tn": 64540.0,
      "thin_recall": 0.3446601927280426,
      "thin_precision": 0.25539568066596985,
      "thin_f1": 0.29338836669921875,
      "boundary_f1": 0.29338836669921875,
      "recall_thin": 0.3446601927280426,
      "f1_thin": 0.29338836669921875,
      "dice_thin": 0.29338836669921875,
      "name": "GAPS384_train_1380_1_1"
    },
    {
      "iou": 0.38125666975975037,
      "dice": 0.552043080329895,
      "f1": 0.552043080329895,
      "precision": 0.4231678545475006,
      "recall": 0.7937915921211243,
      "accuracy": 0.9911346435546875,
      "tp": 358.0,
      "fp": 488.0,
      "fn": 93.0,
      "tn": 64597.0,
      "thin_recall": 0.3510203957557678,
      "thin_precision": 0.341269850730896,
      "thin_f1": 0.3460763990879059,
      "boundary_f1": 0.3460763990879059,
      "recall_thin": 0.3510203957557678,
      "f1_thin": 0.3460763990879059,
      "dice_thin": 0.3460763990879059,
      "name": "GAPS384_train_1381_1_1"
    },
    {
      "iou": 0.3571428656578064,
      "dice": 0.5263156890869141,
      "f1": 0.5263156890869141,
      "precision": 0.40697672963142395,
      "recall": 0.7446808218955994,
      "accuracy": 0.990386962890625,
      "tp": 350.0,
      "fp": 510.0,
      "fn": 120.0,
      "tn": 64556.0,
      "thin_recall": 0.37226277589797974,
      "thin_precision": 0.3805970251560211,
      "thin_f1": 0.3763836920261383,
      "boundary_f1": 0.3763836920261383,
      "recall_thin": 0.37226277589797974,
      "f1_thin": 0.3763836920261383,
      "dice_thin": 0.3763836920261383,
      "name": "GAPS384_train_1381_541_1"
    },
    {
      "iou": 0.37491241097450256,
      "dice": 0.5453618168830872,
      "f1": 0.5453618168830872,
      "precision": 0.5561330318450928,
      "recall": 0.5350000262260437,
      "accuracy": 0.98638916015625,
      "tp": 535.0,
      "fp": 427.0,
      "fn": 465.0,
      "tn": 64109.0,
      "thin_recall": 0.25,
      "thin_precision": 0.3807947039604187,
      "thin_f1": 0.3018372058868408,
      "boundary_f1": 0.3018372058868408,
      "recall_thin": 0.25,
      "f1_thin": 0.3018372058868408,
      "dice_thin": 0.3018372058868408,
      "name": "GAPS384_train_1386_1_641"
    },
    {
      "iou": 0.3094462454319,
      "dice": 0.47263675928115845,
      "f1": 0.47263675928115845,
      "precision": 0.47029703855514526,
      "recall": 0.4749999940395355,
      "accuracy": 0.99029541015625,
      "tp": 285.0,
      "fp": 321.0,
      "fn": 315.0,
      "tn": 64615.0,
      "thin_recall": 0.2651515007019043,
      "thin_precision": 0.2978723347187042,
      "thin_f1": 0.28056105971336365,
      "boundary_f1": 0.28056105971336365,
      "recall_thin": 0.2651515007019043,
      "f1_thin": 0.28056105971336365,
      "dice_thin": 0.28056105971336365,
      "name": "GAPS384_train_1386_541_641"
    },
    {
      "iou": 0.35771065950393677,
      "dice": 0.5269320011138916,
      "f1": 0.5269320011138916,
      "precision": 0.48076921701431274,
      "recall": 0.5829015374183655,
      "accuracy": 0.9876708984375,
      "tp": 450.0,
      "fp": 486.0,
      "fn": 322.0,
      "tn": 64278.0,
      "thin_recall": 0.3122270703315735,
      "thin_precision": 0.4333333373069763,
      "thin_f1": 0.362944096326828,
      "boundary_f1": 0.362944096326828,
      "recall_thin": 0.3122270703315735,
      "f1_thin": 0.362944096326828,
      "dice_thin": 0.362944096326828,
      "name": "GAPS384_train_1389_1_1"
    },
    {
      "iou": 0.33425161242485046,
      "dice": 0.5010322332382202,
      "f1": 0.5010322332382202,
      "precision": 0.5062586665153503,
      "recall": 0.4959128201007843,
      "accuracy": 0.9889373779296875,
      "tp": 364.0,
      "fp": 355.0,
      "fn": 370.0,
      "tn": 64447.0,
      "thin_recall": 0.23820754885673523,
      "thin_precision": 0.424369752407074,
      "thin_f1": 0.30513590574264526,
      "boundary_f1": 0.30513590574264526,
      "recall_thin": 0.23820754885673523,
      "f1_thin": 0.30513590574264526,
      "dice_thin": 0.30513590574264526,
      "name": "GAPS384_train_1389_541_1"
    },
    {
      "iou": 0.21900269389152527,
      "dice": 0.359314501285553,
      "f1": 0.359314501285553,
      "precision": 0.24961598217487335,
      "recall": 0.6410256624221802,
      "accuracy": 0.9823150634765625,
      "tp": 325.0,
      "fp": 977.0,
      "fn": 182.0,
      "tn": 64052.0,
      "thin_recall": 0.27898550033569336,
      "thin_precision": 0.18421052396297455,
      "thin_f1": 0.22190195322036743,
      "boundary_f1": 0.22190195322036743,
      "recall_thin": 0.27898550033569336,
      "f1_thin": 0.22190195322036743,
      "dice_thin": 0.22190195322036743,
      "name": "GAPS384_train_1403_1_1"
    },
    {
      "iou": 0.32270312309265137,
      "dice": 0.4879447817802429,
      "f1": 0.4879447817802429,
      "precision": 0.39206641912460327,
      "recall": 0.6458966732025146,
      "accuracy": 0.98638916015625,
      "tp": 425.0,
      "fp": 659.0,
      "fn": 233.0,
      "tn": 64219.0,
      "thin_recall": 0.4385964870452881,
      "thin_precision": 0.38363170623779297,
      "thin_f1": 0.4092768728733063,
      "boundary_f1": 0.4092768728733063,
      "recall_thin": 0.4385964870452881,
      "f1_thin": 0.4092768728733063,
      "dice_thin": 0.4092768728733063,
      "name": "GAPS384_train_1403_541_1"
    },
    {
      "iou": 0.3009708821773529,
      "dice": 0.4626865088939667,
      "f1": 0.4626865088939667,
      "precision": 0.35284551978111267,
      "recall": 0.6718266010284424,
      "accuracy": 0.9923095703125,
      "tp": 217.0,
      "fp": 398.0,
      "fn": 106.0,
      "tn": 64815.0,
      "thin_recall": 0.4427083432674408,
      "thin_precision": 0.3899082541465759,
      "thin_f1": 0.4146340787410736,
      "boundary_f1": 0.4146340787410736,
      "recall_thin": 0.4427083432674408,
      "f1_thin": 0.4146340787410736,
      "dice_thin": 0.4146340787410736,
      "name": "GAPS384_valid_0009_1_1"
    },
    {
      "iou": 0.4958139657974243,
      "dice": 0.6629352569580078,
      "f1": 0.6629352569580078,
      "precision": 0.5472279191017151,
      "recall": 0.840694010257721,
      "accuracy": 0.991729736328125,
      "tp": 533.0,
      "fp": 441.0,
      "fn": 101.0,
      "tn": 64461.0,
      "thin_recall": 0.5021459460258484,
      "thin_precision": 0.3874172270298004,
      "thin_f1": 0.43738311529159546,
      "boundary_f1": 0.43738311529159546,
      "recall_thin": 0.5021459460258484,
      "f1_thin": 0.43738311529159546,
      "dice_thin": 0.43738311529159546,
      "name": "GAPS384_valid_0016_1_1"
    },
    {
      "iou": 0.36954858899116516,
      "dice": 0.5396648049354553,
      "f1": 0.5396648049354553,
      "precision": 0.4075949490070343,
      "recall": 0.7983471155166626,
      "accuracy": 0.9874267578125,
      "tp": 483.0,
      "fp": 702.0,
      "fn": 122.0,
      "tn": 64229.0,
      "thin_recall": 0.4978354871273041,
      "thin_precision": 0.29411765933036804,
      "thin_f1": 0.36977487802505493,
      "boundary_f1": 0.36977487802505493,
      "recall_thin": 0.4978354871273041,
      "f1_thin": 0.36977487802505493,
      "dice_thin": 0.36977487802505493,
      "name": "GAPS384_valid_0016_541_1"
    }
  ]
}

```
- F1/Dice: 0.47674204365295525
- IoU: 0.3251532674712293
- Precision: 0.48051613385186476
- Recall: 0.5416545487940312
- Thin recall: 0.30105665834511025
- Thin F1: 0.2633731780245024
- Latency seconds/image: 0.040153308763333104
- Throughput img/s: 24.904547864139467
- num_steps: 1
- threshold: -0.3
- checkpoint used: /home/hieulc/avitech11/crackmeanflow/checkpoints_v5_256_ft/best.pt
- Latency ms/image: 40.1533087633331
## 10. Experiment table
| seg_loss_weight | endpoint_loss_weight | thin_loss_weight | lr | threshold | num_steps | F1 | IoU | Dice | thin recall | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.03 | 48.0 | 0.0 | 0.0001 | -0.3 | 1 | 0.4767 | 0.3252 | 0.4767 | 0.3011 | pass |
| 0.03 | 48.0 | 0.0 | 0.0001 | -0.4 | 1 | 0.4757 | 0.3262 | 0.4757 | 0.2976 | pass |
| 0.03 | 48.0 | 0.0 | 0.0001 | -0.5 | 1 | 0.4727 | 0.3221 | 0.4727 | 0.3107 | pass |
| 0.03 | 48.0 | 0.0 | 0.0001 | -0.2 | 1 | 0.4684 | 0.3197 | 0.4684 | 0.2893 | pass |
| 0.03 | 48.0 | 0.0 | 0.0001 | -0.1 | 1 | 0.4600 | 0.3121 | 0.4600 | 0.2941 | pass |
| 0.05 | 32.0 | 0.0 | 0.0002 | -0.3 | 1 | 0.4525 | 0.3031 | 0.4525 | 0.2837 | pass |
| 0.03 | 48.0 | 0.0 | 0.0001 | 0.0 | 1 | 0.4471 | 0.2997 | 0.4471 | 0.2728 | pass |
| 0.03 | 48.0 | 0.0 | 0.0001 | -0.6 | 1 | 0.4447 | 0.2966 | 0.4447 | 0.3162 | pass |
| 0.03 | 48.0 | 0.0 | 0.0001 | 0.1 | 1 | 0.4323 | 0.2873 | 0.4323 | 0.2716 | pass |
| 0.05 | 32.0 | 0.0 | 0.0002 | -0.1 | 1 | 0.4311 | 0.2867 | 0.4311 | 0.2619 | pass |
| 0.03 | 48.0 | 0.0 | 0.0001 | 0.2 | 1 | 0.4171 | 0.2752 | 0.4171 | 0.2630 | pass |
| 0.05 | 32.0 | 0.0 | 0.0002 | 0.0 | 1 | 0.4122 | 0.2699 | 0.4122 | 0.2519 | pass |
| 0.05 | 32.0 | 0.0 | 0.0002 | 0.1 | 1 | 0.4014 | 0.2615 | 0.4014 | 0.2458 | pass |
| 0.05 | 32.0 | 0.0 | 0.0002 | -0.5 | 1 | 0.3723 | 0.2359 | 0.3723 | 0.2951 | pass |
| 0.05 | 32.0 | 0.0 | 0.0002 | 0.3 | 1 | 0.3401 | 0.2127 | 0.3401 | 0.2083 | pass |
| 0.1 | 16.0 | 0.0 | 0.0003 | -0.7 | 1 | 0.3368 | 0.2191 | 0.3368 | 0.4162 | pass |
| 0.25 | 8.0 | 0.0 | 0.0002 | -0.7 | 1 | 0.3219 | 0.2043 | 0.3219 | 0.3950 | pass |
| 0.1 | 16.0 | 0.0 | 0.0003 | -0.5 | 1 | 0.3195 | 0.2040 | 0.3195 | 0.3833 | pass |
| 0.1 | 16.0 | 0.0 | 0.0003 | -0.3 | 1 | 0.3133 | 0.2013 | 0.3133 | 0.3418 | pass |
| 1.0 | 1.0 | 0.0 | 0.0002 | 0.0 | 1 | 0.3121 | 0.1994 | 0.3121 | 0.3527 | pass |
| 0.1 | 16.0 | 0.0 | 0.0003 | -0.9 | 1 | 0.3052 | 0.1938 | 0.3052 | 0.5206 | pass |
| 0.25 | 8.0 | 0.0 | 0.0002 | -0.5 | 1 | 0.3014 | 0.1881 | 0.3014 | 0.3191 | pass |
| 0.05 | 32.0 | 0.0 | 0.0002 | 0.5 | 1 | 0.2903 | 0.1757 | 0.2903 | 0.1768 | pass |
| 0.1 | 16.0 | 0.0 | 0.0003 | 0.1 | 1 | 0.2858 | 0.1804 | 0.2858 | 0.2751 | pass |
| 0.25 | 8.0 | 0.0 | 0.0002 | -0.4 | 1 | 0.2844 | 0.1786 | 0.2844 | 0.3028 | pass |
## 11. Những phần chưa làm hoặc còn nghi ngờ
- root reports/TEST_REPORT.md + FINAL_REPORT.md không thấy
- smoke test stdout artifact không thấy
- benchmark CrackDiff vs CrackMeanFlow artifact không thấy
- F1 tốt nhất < 0.60
- checkpoint architecture field `{}`; dùng YAML để suy
- outputs/metrics.json top-level không thấy
- Cần audit endpoint loss `x0_pred=z-u`
- Cần audit mismatch quick_eval max(flow,seg) vs test.py only flow
## 12. Git diff hoặc unified diff
```
fatal: not a git repository (or any parent up to mount point /)
Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).
```
not a git repo
## 13. Kết luận cuối
- STATUS: PARTIAL
- Best F1: 0.47674204365295525
- Best checkpoint: `/home/hieulc/avitech11/crackmeanflow/checkpoints_v5_256_ft/best.pt`
- Metrics: `/home/hieulc/avitech11/crackmeanflow/outputs/v5_256_sweep_-0.3/metrics.json`
- Lý do: PARTIAL vì có artifact train/test/metrics nhưng F1 < 0.60; thiếu benchmark artifact.