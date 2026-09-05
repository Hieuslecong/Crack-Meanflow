from crackmeanflow.common import environment_info
from pathlib import Path

def test_environment_info_records_runtime_dependency_versions():
    x=environment_info()
    assert x['torch']
    assert x['python'] and x['python_executable']
    for k in ('numpy','Pillow','PyYAML','scipy','scikit-image'):
        assert k in x['package_versions'] and x['package_versions'][k]
    assert 'pythonhashseed' in x

def test_trainer_writes_standalone_environment_snapshot():
    text=(Path(__file__).resolve().parents[1]/'scripts/train_journal.py').read_text()
    assert 'ENVIRONMENT.json' in text and "'environment':environment_info()" in text
