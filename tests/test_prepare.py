"""Standalone preparation must preserve assets and never replace credentials."""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from tidewall_lora import classifier, recipe
from tidewall_lora.prepare import load_settings, prepare


@pytest.fixture
def models(tmp_path, monkeypatch):
    root = tmp_path / 'existing models'
    (root / 'base').mkdir(parents=True)
    (root / 'adapter').mkdir()
    files = {}
    for name in ('base/model.safetensors', 'adapter/adapter_model.safetensors'):
        value = name.encode()
        (root / name).write_bytes(value)
        files[name] = {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({
        'prompt_sha256': hashlib.sha256(recipe.SYSTEM_PROMPT.encode()).hexdigest(),
        'files': files,
    }))
    monkeypatch.setattr(classifier, 'MANIFEST_PATH', manifest)
    for name in tuple(os.environ):
        if name.startswith('TIDEWALL_'):
            monkeypatch.delenv(name)
    return root


def test_prepare_private_state_reuses_models(tmp_path, models):
    state = tmp_path / 'private state'
    path = prepare(state, models, 'cuda')
    settings = load_settings(path)
    assert settings.model_dir == str(models.resolve())
    assert settings.device == 'cuda'
    assert settings.profile == 'tidewall.prompt-injection.1'
    assert settings.validate() == (state / 'classifier.key').read_text().strip()
    assert len(settings.validate()) == 64
    assert set(p.name for p in state.iterdir()) == {'service.json', 'classifier.key'}
    assert stat.S_IMODE(state.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE((state / 'classifier.key').stat().st_mode) == 0o600
    assert settings.validate() not in path.read_text()


def test_existing_state_and_symlink_are_preserved(tmp_path, models):
    state = tmp_path / 'state'
    prepare(state, models)
    before = {p.name: p.read_bytes() for p in state.iterdir()}
    with pytest.raises(FileExistsError):
        prepare(state, models)
    link = tmp_path / 'linked-state'
    link.symlink_to(state, target_is_directory=True)
    with pytest.raises(FileExistsError):
        prepare(link, models)
    assert before == {p.name: p.read_bytes() for p in state.iterdir()}


def test_bad_model_does_not_create_state(tmp_path, models):
    (models / 'base/model.safetensors').write_bytes(b'tampered')
    state = tmp_path / 'state'
    with pytest.raises(classifier.ArtifactError):
        prepare(state, models)
    assert not state.exists()


def test_bad_device_does_not_create_state(tmp_path, models):
    with pytest.raises(ValueError):
        prepare(tmp_path / 'state', models, 'metal')
    assert not (tmp_path / 'state').exists()


@pytest.mark.parametrize('field,value', [
    ('schema_version', True), ('schema_version', 2), ('device', 'metal'),
    ('model_dir', 'relative'), ('api_key_file', 'relative'),
    ('tenant_id', 42), ('unexpected', 'value'), ('profile', 'other'),
])
def test_invalid_config_rejected(tmp_path, models, field, value):
    path = prepare(tmp_path / 'state', models)
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_settings(path)


def test_duplicate_fields_and_nonobject_rejected(tmp_path, models):
    path = prepare(tmp_path / 'state', models)
    original = path.read_text()
    for bad in ('[]', original[:-2] + ', "device": "cuda"}\n'):
        path.write_text(bad)
        with pytest.raises(ValueError):
            load_settings(path)


@pytest.mark.parametrize('name', ['TIDEWALL_TENANT_ID', 'TIDEWALL_TENANT_ID_FILE',
    'TIDEWALL_API_KEY_FILE', 'TIDEWALL_MODEL_DIR', 'TIDEWALL_DEVICE',
    'TIDEWALL_PROFILE', 'TIDEWALL_CACHE_SIZE', 'TIDEWALL_CACHE_TTL_SECONDS'])
def test_environment_conflict_rejected(tmp_path, models, monkeypatch, name):
    path = prepare(tmp_path / 'state', models)
    monkeypatch.setenv(name, 'conflicting-value')
    with pytest.raises(ValueError, match='environment'):
        load_settings(path)


def test_config_wires_server_without_environment(tmp_path, models, monkeypatch):
    from tidewall_lora import __main__, api
    import uvicorn
    config = prepare(tmp_path / 'state', models, 'cuda')
    captured = {}
    monkeypatch.setattr(api, 'create_app', lambda settings: captured.setdefault('settings', settings))
    monkeypatch.setattr(uvicorn, 'run', lambda app, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(sys, 'argv', ['tidewall_lora', '--config', str(config)])
    __main__.main()
    assert captured['settings'].device == 'cuda'
    assert captured['host'] == '127.0.0.1'
    assert captured['workers'] == 1
    assert os.environ['HF_HUB_OFFLINE'] == '1'


def test_cli_help_and_invalid_config_do_not_expose_contents(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    help_result = subprocess.run([sys.executable, '-m', 'tidewall_lora.prepare', '--help'],
        env=env, capture_output=True, text=True, timeout=15)
    assert help_result.returncode == 0
    assert '--models-from' in help_result.stdout
    config = tmp_path / 'bad.json'
    config.write_text('{SECRET_SENTINEL')
    result = subprocess.run([sys.executable, '-m', 'tidewall_lora', '--config', str(config)],
        env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 2
    assert 'SECRET_SENTINEL' not in result.stderr
    assert 'Invalid service configuration' in result.stderr
