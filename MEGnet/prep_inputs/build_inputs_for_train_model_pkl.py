#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:23:50 2023

@author: jstout
"""


import pandas as pd
import MEGnet
from os import path as op
import os
import glob
from scipy.io import loadmat
import numpy as np
import copy
import pickle
from MEGnet.megnet_utilities import fGetStartTimesOverlap
from numpy.random import randint
from numpy.random import shuffle
import copy

tmp=MEGnet.__path__[0]


train_dir = op.join(MEGnet.__path__[0], 'prep_inputs','training')
pkl_topdir = op.join(train_dir, 'Inputs', 'ICAs_pkl_all')
# os.mkdir(pkl_topdir)
# extract_and_pickle_all_datasets(final, pkl_topdir)
# dframe = pd.read_csv('/home/jstout/src/MegNET2022/MEGnet/prep_inputs/training/final_subjICA_dframe.csv')

# dframe_fname = op.join(pkl_topdir, 'dframe_train.csv') #'final_dframe.csv')
# dframe= pd.read_csv(dframe_fname)
# dframe = dframe[~ dframe.duplicated(['Site','TaskType', 'participant_id'])]

# dframe = dframe[dframe.Site != 'NYU']
# dframe.reset_index(inplace=True)
# #KEY indexes original subj_stack
# dframe.drop(['idx','index','Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)

# =============================================================================
# Just load the proper dframe from pkl dir
# =============================================================================
dframe = pd.read_csv(op.join(pkl_topdir, 'final_dframe.csv'))

# =============================================================================
# Assemble time series from the 
# =============================================================================

arrTS = []
arrSP = []
arrCL = []

for idx, row in dframe.iterrows():
    input_vec = row
    indir = op.join(pkl_topdir, f'{input_vec.Site}_{input_vec.subjid}_{input_vec.TaskType}')
    print(indir)
    with open(op.join(indir, 'ts.pkl'),'rb') as f:
        ts = pickle.load(f)
    sp_fname = op.join(indir, 'sp.npy')
    cl_fname = op.join(indir, 'cl.npy')
    sp = np.load(sp_fname)
    cl = np.load(cl_fname)
    arrTS.append(ts)
    arrSP.append(sp)
    arrCL.append(cl)

assert len(arrTS) == len(dframe)
    
def prep_ts_blocks(arrTimeSeries, overlap=3750, normalize=True):
    '''Cut the data into 1min blocks and temporally normalize each block
    Currently assuming all datasets are the same number of samples'''
    # Chunk Data
    # num_blocks = 1 + (arrTimeSeries.shape[1] - 15000) // overlap
    # starts = np.arange(num_blocks)*3750
    starts = fGetStartTimesOverlap(arrTimeSeries.shape[1])
    blocks = np.zeros([arrTimeSeries.shape[0], 15000, len(starts)], dtype=float)
    for i,start in enumerate(starts): 
        blocks[:,:, i] = copy.deepcopy(arrTimeSeries[:,start:start+15000]) 
    
    if normalize==True:
        testTimeSeries = blocks #copy.deepcopy(arrScanTimeSeries[intStartTime:intStartTime+intModelLen])
        min_vals = np.min(testTimeSeries, axis=1, keepdims=True)
        max_vals = np.max(testTimeSeries, axis=1, keepdims=True)
        scaling_factors = 8 / (max_vals - min_vals)
        mean_vals = np.mean(testTimeSeries, axis=1, keepdims=True)
        testTimeSeries = testTimeSeries - mean_vals
        testTimeSeries = testTimeSeries * scaling_factors 
    return testTimeSeries

subj_stack = {}
for idx, _ in enumerate(arrTS): 
    current_TS = arrTS[idx]
    current_sp = arrSP[idx]
    current_cl = arrCL[idx]
    #Chop temporal series into blocks
    ts_tmp = prep_ts_blocks(current_TS)
    ts_out = np.vstack([ ts_tmp[:,:,i] for i in range(ts_tmp.shape[2])])
    repeats = ts_tmp.shape[2]
    #Repeat the spatial maps and class    
    sp_out = np.vstack([ current_sp for i in range(repeats) ]) 
    cl_out = np.tile(current_cl, repeats)
    input_series = dframe.loc[idx]
    subj_stack[idx] = {'ts':ts_out,
                       'sp':sp_out,
                       'cl':cl_out,
                       'id':f'{input_series.Site}_{input_series.subjid}_{input_series.TaskType}'
                       }
    
#Make the surrogate data - must complete the iteration above
from MEGnet.prep_inputs.training.ica_augmenter import group_ts_augmenter
for idx, _ in enumerate(arrTS): 
    current_TS = arrTS[idx]
    current_sp = arrSP[idx]
    current_cl = arrCL[idx]
    #Chop temporal series into blocks
    ts_tmp = prep_ts_blocks(current_TS)
    ts_out = np.vstack([ ts_tmp[:,:,i] for i in range(ts_tmp.shape[2])])
    repeats = ts_tmp.shape[2]
    #Repeat the spatial maps and class    
    sp_out = np.vstack([ current_sp for i in range(repeats) ]) 
    cl_out = np.tile(current_cl, repeats)
    input_series = dframe.loc[idx]
    #Generate the augmented data
    aug_ts, aug_sp, aug_cl = group_ts_augmenter(ts_out, sp_out, cl_out)
    aug_ts2, aug_sp2, aug_cl2 = group_ts_augmenter(ts_out, sp_out, cl_out)
    ts_out = np.concat([ts_out, aug_ts, aug_ts2], axis=0)
    sp_out = np.concat([sp_out, aug_sp, aug_sp2], axis=0)
    cl_out = np.concat([cl_out, aug_cl, aug_cl2], axis=0)
    
    subj_stack[idx] = {'ts':ts_out,
                       'sp':sp_out,
                       'cl':cl_out,
                       'id':f'{input_series.Site}_{input_series.subjid}_{input_series.TaskType}'
                       }
assert len(subj_stack)==len(dframe)



del arrTS, arrSP, arrCL

def make_shuffled_nparrs(dframe_in, subj_stack):
    '''This code takes the full subject list w/pkl data and returns arrays'''
    idxs = dframe_in.key.to_list()
    shuffle(idxs)
    dframe_in['Shuffle_idx']=idxs
    dframe_in.sort_values(['Shuffle_idx'], inplace=True)
    
    shuff_ts = []
    shuff_sp = []
    shuff_cl = []
    
    for idx in dframe_in.key.to_list():
        shuff_ts.append(subj_stack[idx]['ts'])
        shuff_sp.append(subj_stack[idx]['sp'])
        shuff_cl.append(subj_stack[idx]['cl'])
    
    arrTS = np.vstack(shuff_ts)
    arrSP = np.vstack(shuff_sp)
    arrCL = np.hstack(shuff_cl)
    return arrTS, arrSP, arrCL

dframe_train=pd.DataFrame(dframe[dframe.Fold0=='Train'])
dframe_test=pd.DataFrame(dframe[dframe.Fold0=='Test'])

tr_arrTS, tr_arrSP, tr_arrCL = make_shuffled_nparrs(dframe_train, subj_stack) 
te_arrTS, te_arrSP, te_arrCL = make_shuffled_nparrs(dframe_test, subj_stack) 

#Holdout has already been processed and saved out as pickle data
# hd_arrTS, hd_arrSP, hd_arrCL = make_shuffled_nparrs(dframe_hold, subj_stack)

out_topdir = op.join(train_dir, 'Inputs', 'FULL_nparr_surrogate4', 'train')
np.save(op.join(out_topdir,'tr_arrTS.npy'), tr_arrTS)
np.save(op.join(out_topdir,'tr_arrSP.npy'), tr_arrSP)
np.save(op.join(out_topdir,'tr_arrCL.npy'), tr_arrCL)

np.save(op.join(out_topdir,'te_arrTS.npy'), te_arrTS)
np.save(op.join(out_topdir,'te_arrSP.npy'), te_arrSP)
np.save(op.join(out_topdir,'te_arrCL.npy'), te_arrCL)

    