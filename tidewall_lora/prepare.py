"""Prepare private standalone service state; model retrieval is a separate step."""
import argparse
import json
import os
from pathlib import Path
import secrets
from uuid import uuid4

from .classifier import ArtifactError, verify_manifest
from .schemas import PROFILE

SETTING_ENV = (
    'TIDEWALL_TENANT_ID', 'TIDEWALL_TENANT_ID_FILE', 'TIDEWALL_API_KEY_FILE',
    'TIDEWALL_MODEL_DIR', 'TIDEWALL_DEVICE', 'TIDEWALL_PROFILE',
    'TIDEWALL_CACHE_SIZE', 'TIDEWALL_CACHE_TTL_SECONDS',
)
FIELDS = {'schema_version', 'tenant_id', 'api_key_file', 'model_dir', 'device', 'profile'}


def _write_private(path: Path, value: str):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(value)


def prepare(state: Path, models_from: Path, device: str = 'cpu') -> Path:
    """Verify existing weights and create new state without copying model assets."""
    if device not in ('cpu', 'mps', 'cuda'):
        raise ValueError('Unsupported device')
    models = Path(models_from).resolve()
    # Keep the last path component unresolved so an existing symlink is rejected.
    state = Path(os.path.abspath(state))
    if state.resolve().is_relative_to(models):
        raise ValueError('Service state must be outside the model directory')
    verify_manifest(models)
    state.mkdir(mode=0o700, exist_ok=False)
    key_file = state / 'classifier.key'
    _write_private(key_file, secrets.token_hex(32) + '\n')
    config = {
        'schema_version': 1,
        'tenant_id': str(uuid4()),
        'api_key_file': str(key_file),
        'model_dir': str(models),
        'device': device,
        'profile': PROFILE,
    }
    path = state / 'service.json'
    _write_private(path, json.dumps(config, indent=2) + '\n')
    return path


def _unique_fields(pairs):
    data = {}
    for key, value in pairs:
        if key in data:
            raise ValueError('Duplicate configuration field')
        data[key] = value
    return data


def load_settings(path: Path):
    """Strict operator config; never silently mix it with environment settings."""
    if any(name in os.environ for name in SETTING_ENV):
        raise ValueError('Do not combine --config with inference-setting environment variables')
    with Path(path).open('r') as stream:
        content = stream.read(16385)
    if len(content) > 16384:
        raise ValueError('Configuration too large')
    data = json.loads(content, object_pairs_hook=_unique_fields)
    if not isinstance(data, dict) or set(data) != FIELDS:
        raise ValueError('Invalid configuration fields')
    if type(data['schema_version']) is not int or data['schema_version'] != 1:
        raise ValueError('Unsupported configuration version')
    if any(type(data[name]) is not str for name in FIELDS - {'schema_version'}):
        raise ValueError('Invalid configuration value')
    if data['profile'] != PROFILE or data['device'] not in ('cpu', 'mps', 'cuda'):
        raise ValueError('Unsupported standalone profile/device')
    if not all(Path(data[name]).is_absolute() for name in ('model_dir', 'api_key_file')):
        raise ValueError('Configuration requires absolute paths')
    from .api import InferenceSettings
    settings = InferenceSettings(**{k: v for k, v in data.items() if k != 'schema_version'})
    settings.validate()
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True, help='New private state directory')
    parser.add_argument('--models-from', type=Path, required=True, help='Existing pinned model directory')
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    args = parser.parse_args()
    try:
        config = prepare(args.state, args.models_from, args.device)
    except (OSError, ValueError, ArtifactError):
        parser.error('Preparation failed: use valid pinned models and a new state directory outside the models')
    print('Prepared service configuration: ' + str(config))


if __name__ == '__main__':
    main()
