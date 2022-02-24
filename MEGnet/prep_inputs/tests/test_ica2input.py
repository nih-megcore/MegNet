#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 18:51:35 2022

@author: jstout
"""

import numpy as np
import mne
from ..ICA import sensor_pos2circle
from ..ICA import read_raw, raw_preprocess, calc_ica
from pathlib import Path
import os

download_path = '/tmp/test'
path = mne.datasets.sample.data_path(download_path)
filename = Path(path) / 'MEG/sample/sample_audvis_raw.fif'



def test_circle_plot():
    raw = mne.io.read_raw_fif(filename).crop(60.0).pick_types(meg='mag').load_data()
    ica = raw_preprocess(raw, mains_freq=60)
    pos_new = sensor_pos2circle(raw, ica)
    tmp_ = np.load('./circle_plot_pos_new.npy')
    assert np.allclose(pos_new, tmp_)
    
def test_reader():
    filename = '/fast/BIDS_HV_V1/bids/sub-ON02747/ses-01/meg/sub-ON02747_ses-01_task-airpuff_run-01_meg.ds'
    raw = read_raw(filename)
    assert raw.compensation_grade == 3
    assert isinstance(raw, mne.io.ctf.ctf.RawCTF)
    
    filename = '/tmp/test/MNE-sample-data/MEG/sample/sample_audvis_raw.fif'
    raw = read_raw(filename)
    assert isinstance(raw, mne.io.fiff.raw.Raw)


# def test_megin():
#     filename = '/tmp/test/MNE-sample-data/MEG/sample/sample_audvis_raw.fif'
#     results_dir = '/tmp/test/results'
#     if not os.path.exists(results_dir) : os.mkdir(results_dir)
#     raw = read_raw(filename)
#     raw = raw_preprocess(raw, 60)
    
#     file_base = os.path.basename(filename)
#     file_base = os.path.splitext(file_base)[0]
    
#     out_fname = os.path.join(results_dir, file_base+'_250srate_meg.fif')
#     raw.save(out_fname, overwrite=True) #Save with EEG
#     raw.pick_types(meg=True, eeg=False, ref_meg=False)
    
#     ica = calc_ica(raw, file_base=file_base, results_dir=results_dir,
#                    save=True, seedval=0)
    
#     mag_idxs = mne.pick_types(raw.info, meg='mag')
#     mag_raw = raw.copy()
    
#     circle_pos = sensor_pos2circle(raw, ica)
#     circle_pos = circle_pos[mag_idxs]
    
#     for comp in np.arange(0,ica.n_components,1):
#         data = np.dot(ica.mixing_matrix_[:,comp].T,
#                       ica.pca_components_[:ica.n_components_])
#         mag_data = data[mag_idxs]
#         out_fname = f'{results_dir}/{file_base}-ica-{str(comp)}.png'
#         circle_plot(circle_pos=circle_pos, 
#                     data=mag_data, 
#                     out_fname=out_fname)    
    
  


# # =============================================================================
# # Not done
# # =============================================================================
# def test_main():


#     raw = read_raw(filename)
#     assert raw.compensation_grade == 3
#     assert isinstance(raw, mne.io.ctf.ctf.RawCTF)
    
#     filename = '/tmp/test/MNE-sample-data/MEG/sample/sample_audvis_raw.fif'
#     raw = read_raw(filename)
#     assert isinstance(raw, mne.io.fiff.raw.Raw)
    
