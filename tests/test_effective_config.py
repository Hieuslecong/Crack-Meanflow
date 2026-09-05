from pathlib import Path

def test_trainer_emits_effective_config_for_runtime_overrides():
    text=(Path(__file__).resolve().parents[1]/'scripts/train_journal.py').read_text()
    assert 'EFFECTIVE_CONFIG.yaml' in text and 'max_optimizer_steps' in text
