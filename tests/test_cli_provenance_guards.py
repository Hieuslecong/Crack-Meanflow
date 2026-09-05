from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_train_records_group_regex_override_into_effective_config_path():
    text=(ROOT/'scripts/train_journal.py').read_text()
    assert "cfg.setdefault('train',{})['parent_group_regex']=str(a.group_regex)" in text

def test_dataset_label_guards_exist_on_train_eval_threshold_and_target_lock():
    files=['scripts/train_journal.py','scripts/evaluate_journal.py','scripts/freeze_source_threshold.py','scripts/validate_target_dataset.py']
    for f in files:
        text=(ROOT/f).read_text()
        assert 'must be non-empty' in text or 'non-empty provenance labels' in text
