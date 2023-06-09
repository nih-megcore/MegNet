#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:59:39 2023

@author: jstout
"""

import mne
import numpy as np
import pandas as pd
import glob 
import os.path as op, os
from scipy.io import loadmat
import sys

topdir = sys.argv[1]
os.chdir(topdir)

# Compile and clean dataset list
dsets = glob.glob(op.join(topdir, '*', 'probabilities.npy'))


def load_SP(topdir):
    sp_list = []
    for i in range(1,21):
        tmp = loadmat(op.join(topdir, f'component{i}.mat'))['array']
        sp_list.append(tmp)
    output = np.stack(sp_list)
    assert output.shape == (20,120,120,3)
    return output
    
# =============================================================================
# Setup Filenames
# =============================================================================

def get_idx(call_matrix, idxval):
    #Made a function because of the try/except statement
    try:
        tmp = np.argwhere(call_matrix==idxval)[0]
        return tmp
    except:
        return None

def get_example_cases(call_matrix):
    idx_na = get_idx(call_matrix, 0.)
    idx_veog = get_idx(call_matrix, 1.)
    idx_ecg = get_idx(call_matrix, 2.)
    idx_heog = get_idx(call_matrix, 3.)
    return idx_na, idx_veog, idx_ecg, idx_heog


def write_call_matrix(dset):
    dset_dir = op.dirname(dset)
    starts_fname = op.join(dset_dir, 't_starts.npy')
    arrTS_fname = op.join(dset_dir, 'ICATimeSeries.mat')
    
    # =============================================================================
    # Load data
    # =============================================================================
    # Component x ProbClass x Chunk
    prob = np.load(dset)
    # 20 ICA x X x Y x Color  (20 x 120 x 120 x 3)  
    arrSP = load_SP(dset_dir)
    # 20 ICA x Time
    arrTS = loadmat(arrTS_fname)['arrICATimeSeries'].T
    # Starts
    starts = np.load(starts_fname)
    starts = starts[starts >= 0]
    
    call_matrix = np.zeros([20, prob.shape[-1]])
    for chunk_idx in range(prob.shape[-1]):
        chunk = prob[:,:,chunk_idx]
        call = np.argmax(chunk, axis=1)    
        stable_call = [j if chunk[i,j]>0.8 else np.nan for i,j in enumerate(call)]
        call_matrix[:, chunk_idx] = np.array(stable_call).T

    np.save(op.join(dset_dir, 'chunk_call_matrix.npy'), call_matrix)
    print(call_matrix)
        
    outdir = op.join(dset_dir, 'extra')
    if not op.exists(outdir): os.mkdir(outdir)
    idx_na, idx_veog, idx_ecg, idx_heog = get_example_cases(call_matrix)
    
    if idx_na is not None:
        na_trace = arrTS[idx_na[0], starts[idx_na[1]]:starts[idx_na[1]]+15000]
        na_topo = arrSP[idx_na[0],:,:,:]
        np.save(op.join(outdir, 'na_tr.npy'), na_trace)
        np.save(op.join(outdir, 'na_sp.npy'), na_topo)
    
    if idx_veog is not None:
        veog_trace = arrTS[idx_veog[0], starts[idx_veog[1]]:starts[idx_veog[1]]+15000]
        veog_topo = arrSP[idx_veog[0],:,:,:]
        np.save(op.join(outdir, 'veog_tr.npy'), veog_trace)
        np.save(op.join(outdir, 'veog_sp.npy'), veog_topo)
    
    if idx_ecg is not None:
        ecg_trace = arrTS[idx_ecg[0], starts[idx_ecg[1]]:starts[idx_ecg[1]]+15000]
        ecg_topo = arrSP[idx_ecg[0],:,:,:]
        np.save(op.join(outdir, 'ecg_tr.npy'), ecg_trace)
        np.save(op.join(outdir, 'ecg_sp.npy'), ecg_topo)
    
    if idx_heog is not None:
        heog_trace = arrTS[idx_heog[0], starts[idx_heog[1]]:starts[idx_heog[1]]+15000]
        heog_topo = arrSP[idx_heog[0],:,:,:]
        np.save(op.join(outdir, 'heog_tr.npy'), heog_trace)
        np.save(op.join(outdir, 'heog_sp.npy'), heog_topo)
        
        
        
        
    

for dset in dsets:
    write_call_matrix(dset)













