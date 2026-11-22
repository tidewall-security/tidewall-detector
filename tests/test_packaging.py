"""A fresh wheel must contain the inference library, never evaluation data/tests."""
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def test_fresh_wheel_build_excludes_evaluation_data(tmp_path):
    source=Path(__file__).resolve().parents[1]
    staging=tmp_path/'source'
    shutil.copytree(source,staging,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','*.egg-info','build','dist'))
    wheels=tmp_path/'wheels';wheels.mkdir()
    result=subprocess.run([sys.executable,'-c',
        'from setuptools import build_meta; import sys; build_meta.build_wheel(sys.argv[1])',str(wheels)],
        cwd=staging,capture_output=True,text=True,timeout=30,check=False)
    assert result.returncode==0,result.stderr
    files=list(wheels.glob('*.whl'));assert len(files)==1
    with zipfile.ZipFile(files[0]) as wheel:
        names=wheel.namelist()
    assert 'tidewall_lora/ensemble-manifest.json' in names
    assert 'tidewall_lora/model-manifest.json' in names
    assert 'tidewall_lora/recipe-provenance.json' in names
    assert 'tidewall_lora/api.py' in names
    assert 'tidewall_lora/prepare.py' in names
    with zipfile.ZipFile(files[0]) as wheel:
        metadata = wheel.read(next(name for name in names if name.endswith('.dist-info/METADATA'))).decode()
    assert 'Tidewall detector — public preview candidate' in metadata
    assert 'not a calibrated probability' in metadata
    assert any(name.endswith('/licenses/LICENSE') for name in names)
    assert any(name.endswith('/licenses/NOTICE') for name in names)
    assert not any(name.startswith(('evaluation/','tests/')) for name in names)


def test_source_distribution_keeps_readme_linked_documentation(tmp_path):
    source = Path(__file__).resolve().parents[1]
    staging = tmp_path / 'source'
    shutil.copytree(source, staging, ignore=shutil.ignore_patterns(
        '.git', '.venv', '__pycache__', '.pytest_cache', '*.egg-info', 'build', 'dist'))
    output = tmp_path / 'dist'
    output.mkdir()
    result = subprocess.run([sys.executable, '-c',
        'from setuptools import build_meta; import sys; build_meta.build_sdist(sys.argv[1])',
        str(output)], cwd=staging, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    with tarfile.open(next(output.glob('*.tar.gz'))) as archive:
        names = {str(Path(name).relative_to(Path(name).parts[0])) for name in archive.getnames()}
    assert {'README.md', 'LICENSE', 'NOTICE', 'CONTRIBUTING.md',
            'CHANGELOG.md', 'SECURITY.md', 'docs/model-scope.md', 'docs/api.md'} <= names
    assert not any(name.startswith('evaluation/') for name in names)
    assert not any(name.endswith(('.safetensors', '.gguf', '.key', '.pem')) for name in names)
