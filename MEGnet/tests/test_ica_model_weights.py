import pytest
import numpy as np

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


def test_classify_ica_saves_classification_vector(monkeypatch, tmp_path):
    results_root = tmp_path / 'results'
    output_dir = results_root / 'sample'
    output_dir.mkdir(parents=True)
    (output_dir / 'ICATimeSeries.mat').touch()
    for index in range(1, 21):
        (output_dir / f'component{index}.mat').touch()

    class FakeKerasModels:
        @staticmethod
        def load_model(model_path, compile=False):
            return object()

    class FakeKeras:
        models = FakeKerasModels

    monkeypatch.setattr(ICA, '_require_model_weights', lambda: 'model_v2.keras')
    monkeypatch.setitem(__import__('sys').modules, 'keras', FakeKeras)
    monkeypatch.setattr(
        ICA,
        'fPredictChunkAndVoting_parrallel',
        lambda model, arrTS, arrSP: (
            np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ]),
            None,
        ),
    )

    def fake_loadmat(path):
        if str(path).endswith('ICATimeSeries.mat'):
            return {'arrICATimeSeries': np.ones((10, 3))}
        return {'array': np.ones((2, 2, 3))}

    monkeypatch.setattr('scipy.io.loadmat', fake_loadmat)

    result = ICA.classify_ica(
        results_dir=str(results_root),
        outbasename='sample',
        filename='ignored.fif',
    )

    classification_path = output_dir / 'megnet_classification.npy'
    assert classification_path.is_file()
    np.testing.assert_array_equal(np.load(classification_path), np.array([0, 1, 2]))
    np.testing.assert_array_equal(result['classes'], np.array([0, 1, 2]))
