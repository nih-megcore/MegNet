#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 18:51:35 2022

@author: jstout
"""

import numpy as np
import mne
from ..ICA.py import _make_ica, sensor_pos2circle


def test_circle_plot():
    from pathlib import Path
    download_path = '/tmp/test'
    path = mne.datasets.sample.data_path(download_path)
    filename = Path(path) / 'MEG/sample/sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(filename).crop(60.0).pick_types(meg='mag')
    ica = _make_ica(filename)
    pos_new = sensor_pos2circle(raw, ica)
    
    tmp_ = np.read('./circle_plot_pos_new.npy')
    assert np.allclose(pos_new, tmp_)
    # create_circle_plot(raw, ica)
