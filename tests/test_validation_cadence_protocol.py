from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]

def test_canonical_validation_cadence_and_eval_batch_are_locked():
    for p in list((ROOT/'configs/conference').glob('*.yaml'))+list((ROOT/'configs/journal').glob('*.yaml')):
        cfg=yaml.safe_load(p.read_text());ev=cfg['eval']
        assert ev['checkpoint_validation_interval_epochs']==5,p
        assert ev['batch_size']==2,p
