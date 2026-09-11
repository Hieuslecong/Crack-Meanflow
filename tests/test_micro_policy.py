from pathlib import Path
import subprocess
import sys

def test_micro_overfit_has_no_hard_absolute_f1_gate_by_default():
    text=(Path(__file__).resolve().parents[1]/'scripts/micro_overfit.py').read_text()
    assert "default=None" in text and "min-final-f1" in text
    assert "final['f1']>=.80" not in text


def test_micro_overfit_exposes_runtime_accumulation_policy():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/micro_overfit.py", "--help"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "--runtime-grad-accum-steps" in result.stdout
