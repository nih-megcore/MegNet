#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:49:45 2023

@author: jstout
"""

import os.path as op, os
import MEGnet
from MEGnet.megnet_utilities import fPredictChunkAndVoting

#Run on CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from tensorflow import keras


import pandas as pd
import numpy as np
import copy
from scipy.io import loadmat



tmp=MEGnet.__path__[0]
# dframe_path = op.join(tmp, 'prep_inputs','training', 'final_longform_dframe.csv')
dframe_path = op.join(tmp, 'prep_inputs','training', 'final_subjICA_dframe.csv')
dframe = pd.read_csv(dframe_path)

# # All loaded vectors are 45000 samples in duration    
# train_dir = op.join(MEGnet.__path__[0], 'prep_inputs','training')
# np_arr_topdir = op.join(train_dir, 'Inputs')
# arrTS_fname = op.join(np_arr_topdir, 'arrTS.npy')
# arrSP_fname = op.join(np_arr_topdir, 'arrSP.npy')
# arrC_ID_fname = op.join(np_arr_topdir, 'arrC_ID.npy')

# # Load data and verify everything makes sense
# arrTimeSeries = np.load(arrTS_fname)
# arrSpatialMap = np.load(arrSP_fname)
# class_ID = np.load(arrC_ID_fname)  
# assert arrTimeSeries.shape[0] == arrSpatialMap.shape[0]
# assert class_ID.shape[0] == arrTimeSeries.shape[0]
# assert dframe.__len__() == arrTimeSeries.shape[0]
# assert np.alltrue(dframe.ClassID == class_ID)

# =============================================================================
# Helper Funcs
# =============================================================================
def get_inputs(dataset_info):
    '''
    Load the MNE created datasets for each line of the dataframe corresponding
    to a single MEG acquisition

    Parameters
    ----------
    subj_info : pandas.series
        A line from the fully merged dataframe.

    Returns
    -------
    ts_ : numpy.ndarray
        ICA timeseries - ICA# X Samples
    spat_resized : numpy.ndarray
        ICA Spatial topography map - cropped by bounding box. 
        Shape of (20, 120, 120, 3)  -- (ICA, X, Y, Color)

    '''
    data_dir = dataset_info['dirname']
    ts_fname = op.join(data_dir, 'ICATimeSeries.mat')
    ts_ = loadmat(ts_fname)['arrICATimeSeries'].T
    assert type(ts_) is np.ndarray
    
    spat_ = [] 
    for i in range(1,21): spat_.append(loadmat(op.join(data_dir, f'component{str(i)}.mat'))['array'])
    assert len(spat_)==20
    spat_ = np.stack(spat_)
    assert spat_.shape == (20,120, 120, 3)
    
    spat_resized = spat_ #[:,25:-35,16:-14,:]
    class_vec = make_classification_vector(dataset_info)
    
    return ts_ , spat_resized, class_vec

def make_classification_vector(input_vec):
    '''Convert the separate labelled columns into a 20X1 vector of labels'''
    output = np.zeros(20, dtype=int) #Number of ICAs
    VEOG =  _convert_strlist2intlist(input_vec.eyeblink)
    HEOG = _convert_strlist2intlist(input_vec.Saccade)
    EKG = _convert_strlist2intlist(input_vec.EKG)
    output[VEOG] = 1 
    output[HEOG] = 3
    output[EKG] = 2
    return output

def extract_all_datasets(dframe):
    '''
    Loop over all datasets
    Load the spatial, temporal, and classIDs into numpy matrcies

    Parameters
    ----------
    dframe : pd.DataFrame
        Subject level dataframe (each row is 1 subjectXrun).

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    TS_test, SP_test, class_vec = [], [], []
    for idx,input_vec in dframe.iterrows():
        print(idx)
        print(input_vec)
        TS_tmp, SP_tmp, CLid_tmp = get_inputs(input_vec)
        failed=[]
        if TS_tmp.shape[1] < 40000: 
            failed.append(input_vec) 
        TS_test.append(TS_tmp[:,:45000]) 
        
        SP_test.append(SP_tmp)
        class_vec.append(CLid_tmp) 
        for i in failed:
            print(i)
    return np.vstack(TS_test), np.vstack(SP_test), np.stack(class_vec).flatten()


def _convert_strlist2intlist(strlist):
    '''Hack to fix formatting'''
    tmp_ = strlist.replace('"','').replace('[','').replace(' ','').replace(']','').replace("'","").split(',')
    if (tmp_=='') | (tmp_==[]) | (tmp_==['']):
        return []
    return [int(i) for i in tmp_]
    
def _change_dirname(dirname):
    return op.join('/fast/MEGNET/ICA_megnet/RAW/HCP_reprocessed_newtopo', op.basename(dirname))


def make_longform_dframe(dframe, class_ID=None):
    '''
    Convert the subject level dataframe to component level dataframe.  Indexes
    will match the 0-axis/row of the numpy array.

    Parameters
    ----------
    dframe : pd.DataFrame
        Subject level dataframe (merged demographics and ICA info).
    classID : np.array
        Class IDs for the rows of ICAs.  Used as a sanity check (assert).

    Returns
    -------
    dframe_longform : pd.DataFrame
        ICA level dataframe.

    '''
    init_cols = dframe.columns    
    ica_dummy_var = [f'ICA0{str(i)}' for i in range(10)] +  [f'ICA{str(i)}' for i in range(10,20)]
    dframe_wideform = copy.deepcopy(dframe)
    for idx, row in dframe_wideform.iterrows():
        dframe_wideform.loc[idx,ica_dummy_var]=make_classification_vector(row)
    
    dframe_longform = pd.melt(dframe_wideform, id_vars=init_cols, value_vars=ica_dummy_var,
                             var_name='ICAnum', value_name='ClassID')
    
    dframe_longform.sort_values(by=['key','ICAnum'], ascending=True, inplace=True)
    dframe_longform.reset_index(drop=True, inplace=True)
    
    
    dframe_longform = dframe_longform.drop(columns=['eyeblink', 'Saccade', 'EKG', 'other', 'Unnamed: 7', 'Unnamed: 6', 'type_y','type_x'])
    dframe_longform.ClassID = dframe_longform.ClassID.astype(int)
    
    match_idxs = [i*20 for i in range(len(dframe))]
    tmp = dframe.participant_id.values == dframe_longform.loc[match_idxs, 'participant_id'].values
    assert False not in tmp
    assert np.alltrue(class_ID==dframe_longform.ClassID)
    return dframe_longform


# =============================================================================
# Get the HCP data
# =============================================================================
hcp_idx = dframe[dframe.Site=='SLU'].index.values #[20:40]
hcp_dframe = dframe.loc[hcp_idx].reset_index(drop=True)
hcp_dframe.dirname = hcp_dframe.dirname.apply(_change_dirname)
ts, sp , cl = extract_all_datasets(hcp_dframe)

hcp_arrTS, hcp_arrSP = ts, sp


# hcp_arrTS = arrTimeSeries[hcp_idx,:]
# hcp_arrSP = arrSpatialMap[hcp_idx,:,:,:]
# hcp_arrC_ID = class_ID[hcp_idx]
hcp_dframe = dframe.loc[hcp_idx].reset_index(drop=True)
# del arrTimeSeries, arrSpatialMap, class_ID
hcp_dframe_longform = make_longform_dframe(hcp_dframe, class_ID=cl)


import tensorflow_addons
model_fname = op.join(MEGnet.__path__[0], 'model/MEGnet_final_model.h5')
kModel = keras.models.load_model(model_fname) #, compile=False)

#np.zeros([hcp_arrTS.shape[0]])
output = fPredictChunkAndVoting(kModel, hcp_arrTS, hcp_arrSP, cl, intModelLen=15000, intOverlap=3750)
arrPredicionsVote, arrGTVote, arrPredictionsChunk, arrGTChunk = output

#
# np.zeros((20,3))
output = fPredictChunkAndVoting(kModel, hcp_arrTS, hcp_arrSP, np.zeros((hcp_arrTS.shape[0],3)), intModelLen=15000, intOverlap=3750)
arrPredicionsVote, arrGTVote, arrPredictionsChunk, arrGTChunk = output

sample_output = arrPredicionsVote[:,0,:].argmax(axis=1)


from sklearn.metrics import confusion_matrix
confusion_matrix(cl, sample_output)
