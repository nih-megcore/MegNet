"""Regression tests for MEGnet's custom ICA topomap rendering."""

from importlib import import_module

import numpy as np
from scipy.io import loadmat


def test_ica_module_imports():
    """The preprocessing module imports with supported MNE versions."""
    module = import_module("MEGnet.prep_inputs.ICA")
    assert callable(module.circle_plot)


def test_circle_plot_writes_rgb_ica_image(tmp_path):
    """Render the ICA image path used for classifier spatial inputs."""
    circle_plot = import_module("MEGnet.prep_inputs.ICA").circle_plot
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    radii = np.resize([0.35, 0.55, 0.75, 0.9], angles.size)
    positions = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    data = np.linspace(-1.0, 1.0, len(positions))
    image_path = tmp_path / "component1.png"

    circle_plot(circle_pos=positions, data=data, out_fname=image_path)

    assert image_path.is_file()
    image_array = loadmat(image_path.with_suffix(".mat"))["array"]
    assert image_array.ndim == 3
    assert image_array.shape[2] == 3
    assert image_array.dtype == np.uint8
    assert np.ptp(image_array) > 0
