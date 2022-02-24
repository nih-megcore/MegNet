#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 18:51:35 2022

@author: jstout
Minimal testing - needs to be improved with abstraction of tmpdir
Also needs more comparitive testing with results
"""

import numpy as np
import mne
from ..ICA import sensor_pos2circle
from ..ICA import read_raw, raw_preprocess, calc_ica
from pathlib import Path
import os
from ..ICA import main
import shutil

download_path = '/tmp/test'
path = mne.datasets.sample.data_path(download_path)
megin_filename = Path(path) / 'MEG/sample/sample_audvis_raw.fif'
ctf_filename = '/fast/BIDS_HV_V1/bids/sub-ON02747/ses-01/meg/sub-ON02747_ses-01_task-airpuff_run-01_meg.ds'

tests_array_dir = os.path.dirname(os.path.abspath(__file__))

def test_circle_plot():
    raw = mne.io.read_raw_fif(megin_filename).crop(60.0).pick_types(meg='mag').load_data()
    ica = raw_preprocess(raw, mains_freq=60)
    pos_new = sensor_pos2circle(raw, ica)
    tmp_ = np.load(os.path.join(tests_array_dir,
                                'circle_plot_pos_new.npy'))
    assert np.allclose(pos_new, tmp_)
    
def test_reader():
    raw = read_raw(ctf_filename)
    assert raw.compensation_grade == 3
    assert isinstance(raw, mne.io.ctf.ctf.RawCTF)
    
    raw = read_raw(megin_filename)
    assert isinstance(raw, mne.io.fiff.raw.Raw)

def test_megin():
    filename = '/tmp/test/MNE-sample-data/MEG/sample/sample_audvis_raw.fif'
    results_dir = '/tmp/test/results'
    if os.path.exists(results_dir) : shutil.rmtree(results_dir)
    os.mkdir(results_dir)
    main(
        filename,
        mains_freq=60,
        save_preproc=True,
        save_ica=True,
        seedval=0,
        results_dir=results_dir
        )
    
def test_ctf():
    results_dir = '/tmp/test/results'
    if os.path.exists(results_dir) : shutil.rmtree(results_dir)
    os.mkdir(results_dir)
    main(
        ctf_filename,
        mains_freq=60,
        save_preproc=True,
        save_ica=True,
        seedval=0,
        results_dir=results_dir
        )
    
    
