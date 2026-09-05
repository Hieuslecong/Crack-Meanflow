from pathlib import Path

def test_evaluator_records_config_taint_contract():
    text=(Path(__file__).resolve().parents[1]/'scripts/evaluate_journal.py').read_text()
    assert 'DIAGNOSTIC_CONFIG_MISMATCH' in text and 'TAINTED_RESUME_CONFIG_CHANGE' in text

def test_trainer_records_resume_config_taint_contract():
    text=(Path(__file__).resolve().parents[1]/'scripts/train_journal.py').read_text()
    assert 'resume_config_mismatch' in text and 'TAINTED_RESUME_CONFIG_CHANGE' in text
