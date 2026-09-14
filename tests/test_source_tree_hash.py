from pathlib import Path
from crackmeanflow.common.checkpointing import source_tree_hash, source_tree_manifest

def test_source_tree_hash_changes_with_source_content(tmp_path):
    (tmp_path/'crackmeanflow').mkdir(); (tmp_path/'scripts').mkdir(); (tmp_path/'configs').mkdir()
    f=tmp_path/'crackmeanflow'/'x.py'; f.write_text('x=1\n')
    h1=source_tree_hash(tmp_path); f.write_text('x=2\n'); h2=source_tree_hash(tmp_path)
    assert h1!=h2


def test_source_tree_manifest_includes_all_execution_runners_but_not_tooling(tmp_path):
    (tmp_path / "crackmeanflow").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "crackmeanflow" / "x.py").write_text("x=1\n")
    execution_files = (
        "train_journal.py",
        "train_paper_v3.py",
        "train_generalization_screen_v3.py",
        "evaluate_journal.py",
        "freeze_source_threshold.py",
        "execute_experiment_queue.py",
        "smoke_preflight_v3.py",
        "smoke_gate_v3.py",
        "branch_coverage_probe_v3.py",
        "resume_equivalence_v3.py",
    )
    for name in execution_files:
        (tmp_path / "scripts" / name).write_text("x=1\n")
    (tmp_path / "scripts" / "audit_report.py").write_text("x=1\n")
    paths = {row["path"] for row in source_tree_manifest(tmp_path)}
    assert {f"scripts/{name}" for name in execution_files} <= paths
    assert "scripts/audit_report.py" not in paths

    initial = source_tree_hash(tmp_path)
    (tmp_path / "scripts" / "train_generalization_screen_v3.py").write_text("x=2\n")
    assert source_tree_hash(tmp_path) != initial
