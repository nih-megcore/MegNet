import json
import os.path as op

import pytest

from MEGnet import megnet_init


def _set_model_paths(monkeypatch, tmp_path):
    weights_path = tmp_path / 'model_v2k3'
    monkeypatch.setattr(megnet_init, 'weights_path', str(weights_path))
    monkeypatch.setattr(megnet_init, 'model_path', str(weights_path / 'model_v2.keras'))
    monkeypatch.setattr(megnet_init, 'config_path', str(weights_path / 'config.json'))
    return weights_path


def test_check_weights_requires_config_json(monkeypatch, tmp_path):
    weights_path = _set_model_paths(monkeypatch, tmp_path)
    weights_path.mkdir()
    (weights_path / 'model_v2.keras').touch()

    assert megnet_init._check_weights() is False


def test_check_weights_requires_model_file(monkeypatch, tmp_path):
    weights_path = _set_model_paths(monkeypatch, tmp_path)
    weights_path.mkdir()
    with open(op.join(weights_path, 'config.json'), 'w', encoding='utf-8') as fid:
        json.dump({'model_version': 'v2.2'}, fid)

    assert megnet_init._check_weights() is False


def test_check_weights_rejects_model_version_below_min_version(monkeypatch, tmp_path):
    weights_path = _set_model_paths(monkeypatch, tmp_path)
    weights_path.mkdir()
    (weights_path / 'model_v2.keras').touch()
    monkeypatch.setattr(megnet_init, 'min_model_version', 'v2.2')
    with open(op.join(weights_path, 'config.json'), 'w', encoding='utf-8') as fid:
        json.dump({'model_version': 'v2.1'}, fid)

    assert megnet_init._check_weights() is False


@pytest.mark.parametrize('model_version', ['v2.2', 'v2.3'])
def test_check_weights_accepts_compatible_model_version(
        monkeypatch, tmp_path, model_version):
    weights_path = _set_model_paths(monkeypatch, tmp_path)
    weights_path.mkdir()
    (weights_path / 'model_v2.keras').touch()
    monkeypatch.setattr(megnet_init, 'min_model_version', 'v2.2')
    with open(op.join(weights_path, 'config.json'), 'w', encoding='utf-8') as fid:
        json.dump({'model_version': model_version}, fid)

    assert megnet_init._check_weights() is True
