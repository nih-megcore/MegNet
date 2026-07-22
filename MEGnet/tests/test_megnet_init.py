import json
import os.path as op

from MEGnet import megnet_init


def _set_config_path(monkeypatch, tmp_path):
    weights_path = tmp_path / 'model_v2k3'
    monkeypatch.setattr(megnet_init, 'weights_path', str(weights_path))
    monkeypatch.setattr(megnet_init, 'config_path', str(weights_path / 'config.json'))
    return weights_path


def test_check_weights_requires_config_json(monkeypatch, tmp_path):
    _set_config_path(monkeypatch, tmp_path).mkdir()

    assert megnet_init._check_weights() is False


def test_check_weights_rejects_model_version_not_greater_than_v2_1(monkeypatch, tmp_path):
    weights_path = _set_config_path(monkeypatch, tmp_path)
    weights_path.mkdir()
    with open(op.join(weights_path, 'config.json'), 'w', encoding='utf-8') as fid:
        json.dump({'model_version': 'v2.1'}, fid)

    assert megnet_init._check_weights() is False


def test_check_weights_accepts_model_version_greater_than_v2_1(monkeypatch, tmp_path):
    weights_path = _set_config_path(monkeypatch, tmp_path)
    weights_path.mkdir()
    with open(op.join(weights_path, 'config.json'), 'w', encoding='utf-8') as fid:
        json.dump({'model_version': 'v2.2'}, fid)

    assert megnet_init._check_weights() is True
