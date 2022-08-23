#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:49:16 2022

@author: jstout
"""

import pandas as pd
import MEGnet
from os import path as op
import os
import glob
from scipy.io import loadmat
import numpy as np

# =============================================================================
# Currently dropping smt datasets from CAMCAN - also need to add rest datasets
# =============================================================================
tmp=MEGnet.__path__[0]
class_table_path = op.join(op.dirname(tmp), 'ICA_combined_participants.tsv')
class_table = pd.read_csv(class_table_path, sep='\t')
if 'Unnamed: 0' in class_table.columns:
    class_table.drop('Unnamed: 0', axis=1, inplace=True)
if 'idx' in class_table.columns:
    class_table.drop('idx', axis=1, inplace=True)



dataset_path = '/fast/MEGNET/ICAs'
dsets = glob.glob(op.join(dataset_path, '*_meg'))
datasets = pd.DataFrame(dsets, columns=['dirname'])

def get_subjid(dirname):
    tmp = op.basename(dirname)
    return tmp.split('_')[0]

def get_type(dirname):
    '''Extract the task type from the dataset name'''
    return [i[5:] for i in op.basename(dirname).split('_') if i[0:4]=='task'][0]
    

datasets['subjid'] = datasets.dirname.apply(get_subjid)
datasets['type'] = datasets.dirname.apply(get_type)
# FIX! Needs to label smt datasets !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

final = pd.merge(class_table, datasets, left_on=['participant_id', 'type'], right_on=['subjid','type'])

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
    data_dir = dataset_info.dirname
    ts_fname = op.join(data_dir, 'ICATimeSeries.mat')
    ts_ = loadmat(ts_fname)['arrICATimeSeries'].T
    assert type(ts_) is np.ndarray
    
    spat_ = [] 
    for i in range(1,21): spat_.append(loadmat(op.join(data_dir, f'component{str(i)}.mat'))['array'])
    assert len(spat_)==20
    spat_ = np.stack(spat_)
    assert spat_.shape == (20,180, 150, 3)
    
    spat_resized = spat_[:,25:-35,16:-14,:]
    return ts_ , spat_resized

def get_default_hcp():
    '''Load and return the hcp ICA dataset'''
    data_dir = op.join(MEGnet.__path__[0], 'example_data/HCP/100307/@rawc_rfDC_8-StoryM_resample_notch_band/ICA202DDisc')
    ts_ = loadmat(op.join(data_dir, 'ICATimeSeries.mat'))['arrICATimeSeries'].T
    
    # lSpatial.append(loadmat(os.path.join(strDataPathSpatial,f'component{intComp}.mat'))['array'][30:-30,15:-14,:])
    spat_ = []
    for i in range(1,21): spat_.append(loadmat(op.join(data_dir, f'component{str(i)}.mat'))['array'][30:-30,15:-15,:])
    spat_ = np.stack(spat_)
    return ts_, spat_

def test_fPredict():
    arrTimeSeries, arrSpatialMap = get_default_hcp()
    output = fPredictChunkAndVoting(kModel, 
                                    arrTimeSeries, 
                                    arrSpatialMap, 
                                    np.zeros((20,3)), #the code expects the Y values as it was used for performance, just put in zeros as a place holder.
                                    intModelLen=15000, 
                                    intOverlap=3750)
    arrPredicionsVote, arrGTVote, arrPredictionsChunk, arrGTChunk = output
    correct_out = [2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0]
    actual_out = arrPredicionsVote[:,0,:].argmax(axis=1)
    assert_vals = [np.equal(i,j) for i,j in zip(correct_out, actual_out)]
    assert False not in assert_vals

from tensorflow import keras
model_fname = '/home/jstout/src/MegNET2022/MEGnet/model/MEGnet_final_model.h5'
kModel = keras.models.load_model(model_fname, compile=False)

arrTimeSeries, arrSpatialMap = get_inputs(final.loc[0])

#use the vote chunk prediction function to make a prediction on each input
from MEGnet.label_ICA_components import fPredictChunkAndVoting
output = fPredictChunkAndVoting(kModel, 
                                arrTimeSeries, 
                                arrSpatialMap, 
                                np.zeros((20,3)), #the code expects the Y values as it was used for performance, just put in zeros as a place holder.
                                intModelLen=15000, 
                                intOverlap=3750)
arrPredicionsVote, arrGTVote, arrPredictionsChunk, arrGTChunk = output
    
    
    get_default_hcp()
    
    
    
