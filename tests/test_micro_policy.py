from pathlib import Path

def test_micro_overfit_has_no_hard_absolute_f1_gate_by_default():
    text=(Path(__file__).resolve().parents[1]/'scripts/micro_overfit.py').read_text()
    assert "default=None" in text and "min-final-f1" in text
    assert "final['f1']>=.80" not in text
