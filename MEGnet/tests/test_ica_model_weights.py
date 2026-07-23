import pytest

from MEGnet.prep_inputs import ICA


def test_main_requires_results_dir():
    with pytest.raises(TypeError, match='results_dir'):
        ICA.main('input.fif')


def test_cmdline_requires_results_dir(monkeypatch):
    monkeypatch.setattr(
        'sys.argv',
        ['ICA.py', '-filename', 'input.fif', '-line_freq', '60'],
    )

    with pytest.raises(SystemExit) as error:
        ICA.cmdline()

    assert error.value.code == 2


def test_require_model_weights_returns_compatible_model(monkeypatch, tmp_path):
    model_path = tmp_path / 'model_v2k3' / 'model_v2.keras'

    monkeypatch.setattr(ICA.megnet_init, 'model_path', str(model_path))
    monkeypatch.setattr(ICA.megnet_init, '_check_weights', lambda: True)

    assert ICA._require_model_weights() == str(model_path)


def test_require_model_weights_rejects_invalid_install(monkeypatch, tmp_path):
    model_path = tmp_path / 'model_v2k3' / 'model_v2.keras'
    monkeypatch.setattr(ICA.megnet_init, 'model_path', str(model_path))
    monkeypatch.setattr(ICA.megnet_init, '_check_weights', lambda: False)

    with pytest.raises(RuntimeError, match='Run `megnet_init`'):
        ICA._require_model_weights()
