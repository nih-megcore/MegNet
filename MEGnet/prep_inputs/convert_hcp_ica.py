#!/usr/bin/env python
import numpy as np
import mne
import hcp 
import os, os.path as op
import copy
from MEGnet.prep_inputs.ICA import sensor_pos2circle
from MEGnet.prep_inputs.ICA import circle_plot
from scipy.io import savemat
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-hcp_root')
parser.add_argument('-subject')
parser.add_argument('-task', 
                    choices=['rest','task_motor','task_working_memory',
                             'task_story_math'] #, 'noise_empty_room', 'noise_subject']
                    )
parser.add_argument('-run', type=int)
parser.add_argument('-results_dir')
args=parser.parse_args()

hcp_root = args.hcp_root # '/fast/OPEN/HCP/'
subject = args.subject #'100307' #sys.argv[1]
task = args.task #'rest'
run = args.run #1
results_dir = args.results_dir

def unprocessed_path(subject=None,
                     data_type=None,
                     run_index=None,
                     hcp_path=None):
    'Get the path to the raw data in 4D format'
    data_type=data_type.lower()
    if data_type=='rest':
        data_type='restin'
    if data_type=='task_working_memory':
        data_type='Wrkmem'
    if data_type=='task_story_math':
        data_type='StoryM'
    if data_type=='task_motor':
        data_type='Motor'
    data_type=data_type[0].upper()+data_type[1:]
    if data_type=='Storym':
        data_type='StoryM'
    run_dict={'Restin':2,
              'Wrkmem':5,
              'StoryM':7}
    run_index=copy.copy(run_index)
    run_index=str(run_dict[data_type]+int(run_index))
    
    outpath=op.join(hcp_path, subject, 'unprocessed', 'MEG', 
                    f'{run_index}-{data_type}', '4D','c,rfDC')
    return outpath


def main(args):
    hcp_root = args.hcp_root # '/fast/OPEN/HCP/'
    subject = args.subject #'100307' #sys.argv[1]
    task = args.task #'rest'
    run = args.run #1    
    
    data_dict = dict(subject=subject,
                     data_type=task,
                     run_index=run,
                     hcp_path=hcp_root)
    
    unproc_raw_path = unprocessed_path(**data_dict)
    unproc_raw = mne.io.read_raw_bti(unproc_raw_path, head_shape_fname=None, 
                                     rename_channels=False, sort_by_ch_name=False)

    raw = hcp.read_raw(**data_dict)
    annot = hcp.read_annot(**data_dict) 
    ecg_eog_ic = annot['ica']['ecg_eog_ic']
    
    #Verify that the naming in the fieldtrip format is the same as the MNE
    for i,j in zip(unproc_raw.ch_names, raw.ch_names):
        assert i==j
    
    #Preproc data
    raw.load_data()
    resample_freq = 250
    #notch_freqs = range(mains_freq, int(resample_freq * 2/3), mains_freq)
    raw.notch_filter([60]) #notch_freqs)
    raw.resample(resample_freq)
    raw.filter(1.0, 100)
    
    ica_mat = hcp.read_ica(**data_dict)
    # ica_mat['unmixing'].tolist()
    exclude=annot['ica']['ecg_eog_ic']
    
    ch_names = ica_mat['topolabel'].tolist().tolist()
    picks = mne.pick_channels(raw.info['ch_names'], include=ch_names)
    assert ch_names == [raw.ch_names[p] for p in picks]
    unmixing_matrix = np.array(ica_mat['unmixing'].tolist())
    
    n_components, n_channels = unmixing_matrix.shape
    mixing = np.array(ica_mat['topo'].tolist())
    
    # proj_mat = (np.eye(n_channels) - np.dot(
    #         mixing[:, exclude], unmixing_matrix[exclude]))
    # raw._data *= 1e15
    # raw._data[picks] = np.dot(proj_mat, raw._data[picks])
    # raw._data /= 1e15
    
    ica_traces = np.dot(unmixing_matrix, raw._data[picks])
    # import pylab
    # pylab.plot(ica_traces[exclude,0:1000].T)
    
    # HACK - Force the ICA weights into a evoked structure to get the topomap out
    epo = mne.make_fixed_length_epochs(unproc_raw.copy().crop(0,100), preload=True)
    evk = epo.average()
    evk.pick_channels(ch_names)
    
    psuedo_tmax = evk.times[mixing.shape[1]]
    evk.crop(None, psuedo_tmax, include_tmax=False)
    evk._data[:,:mixing.shape[1]]=mixing
    
    # =============================================================================
    # Save the outputs
    # =============================================================================
    # Topography only works with mags - get index for display
    # mag_idxs = mne.pick_types(raw.info, meg='mag')
    _tmp = f'{unproc_raw_path.split("/")[-6]}_{unproc_raw_path.split("/")[-3]}'
    _tmp = _tmp.replace('-','_')
    out_subjdir = f'{results_dir}/{_tmp}'
    if not op.exists(out_subjdir): os.mkdir(out_subjdir)
    
    circle_pos = sensor_pos2circle(evk, evk)
    
    for comp in np.arange(0,evk._data.shape[1],1):
        data = evk._data[:,comp]     #np.dot(ica.mixing_matrix_[:,comp].T,
                     # ica.pca_components_[:ica.n_components_])
        
        out_fname = f'{out_subjdir}/component{str(comp+1)}.png' #'{file_base}-ica-{str(comp)}.png'
        circle_plot(circle_pos=circle_pos, 
                    data=data, 
                    out_fname=out_fname)
    
    # Save ICA timeseries as input for classification
    # Currently inputs to classification are matlab arrays
    ica_ts = ica_traces #ica.get_sources(raw)._data.T
    outfname = f'{out_subjdir}/ICATimeSeries.mat' #'{file_base}-ica-ts.mat'
    savemat(outfname, 
            {'arrICATimeSeries':ica_ts})


main(args)
# # =============================================================================
# # Standard ICA
# # =============================================================================
# ica = mne.preprocessing.read_ica('/fast/results_ica/sub-ON02747_ses-01_task-rest_run-01_meg/sub-ON02747_ses-01_task-rest_run-01_meg_0-ica.fif')










