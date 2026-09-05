from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]

def test_all_executable_configs_use_multiseed_source_checkpoint_selection():
    for p in (ROOT/'configs').rglob('*.yaml'):
        cfg=yaml.safe_load(p.read_text())
        if not isinstance(cfg,dict) or 'track' not in cfg or 'backbone' not in cfg: continue
        ev=cfg['eval']
        assert ev['checkpoint_selection_seeds']==[0,1,2], p
        assert ev['checkpoint_use_final_threshold_grid'] is True, p
        assert len(ev['final_threshold_calibration_seeds'])==5, p

def test_canonical_cfd_configs_use_no_replacement_sampling():
    for p in list((ROOT/'configs/conference').glob('*.yaml'))+list((ROOT/'configs/journal').glob('*.yaml')):
        cfg=yaml.safe_load(p.read_text())
        sb=cfg['train']['sample_balance']
        assert sb['enabled'] is False, p
        assert sb['unit']=='uniform_crop_without_replacement', p
