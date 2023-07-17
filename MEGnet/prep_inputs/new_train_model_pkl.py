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

tmp=MEGnet.__path__[0]


train_dir = op.join(MEGnet.__path__[0], 'prep_inputs','training')
pkl_topdir = op.join(train_dir, 'Inputs', 'PKL')
# os.mkdir(pkl_topdir)
# extract_and_pickle_all_datasets(final, pkl_topdir)
# dframe = pd.read_csv('/home/jstout/src/MegNET2022/MEGnet/prep_inputs/training/final_subjICA_dframe.csv')

dframe_fname = op.join(pkl_topdir, 'final_dframe.csv')
dframe= pd.read_csv(dframe_fname)
dframe = dframe[~ dframe.duplicated(['Site','TaskType', 'participant_id'])]

dframe = dframe[dframe.Site != 'NYU']
dframe.reset_index(inplace=True)
dframe.key=dframe.index.to_numpy()
dframe.drop(['idx','index','Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)

arrTS = []
arrSP = []
arrCL = []

for idx, row in dframe.iterrows():
    input_vec = row
    # print(indir)
    indir = op.join(pkl_topdir, f'{input_vec.Site}_{input_vec.subjid}_{input_vec.TaskType}')
    with open(op.join(indir, 'ts.pkl'),'rb') as f:
        ts = pickle.load(f)
    sp_fname = op.join(indir, 'sp.npy')
    cl_fname = op.join(indir, 'cl.npy')
    sp = np.load(sp_fname)
    cl = np.load(cl_fname)
    arrTS.append(ts)
    arrSP.append(cl)

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
for idx, current_TS in enumerate(arrTS): 
    ts_tmp = prep_ts_blocks(current_TS)
    ts_out = np.vstack([ ts_tmp[:,:,i] for i in range(ts_tmp.shape[2])])
    repeats = ts_tmp.shape[2]
    
    sp_out = np.vstack([ sp for i in range(repeats) ]) 
    cl_out = np.tile(cl, repeats)
    subj_stack[idx] = {'ts':ts_out,
                       'sp':sp_out,
                       'cl':cl_out
                       }


assert len(subj_stack)==len(dframe)

dframe_train = pd.DataFrame(dframe[dframe.HoldOut==False])
dframe_train.reset_index(inplace=True, drop=True)


from numpy.random import randint
from numpy.random import shuffle

import copy
## Reset index 
idxs = dframe_train.key.to_list() #list(range(len(dframe_train)))
shuffle(idxs)
dframe_train['Shuffle_idx']=idxs
dframe_train.sort_values(['Shuffle_idx'], inplace=True)


shuff_ts = []
shuff_sp = []
shuff_cl = []

train_stack = []

for idx in dframe.key.to_list():
    shuff_ts.append(subj_stack[idx]['ts'])
    shuff_sp.append(subj_stack[idx]['sp'])
    shuff_cl.append(subj_stack[idx]['cl'])

arrTS = np.vstack(shuff_ts)
arrSP = np.vstack(shuff_sp)
arrCL = np.hstack(shuff_cl)



# =============================================================================
# 
# =============================================================================
assert arrTS.shape[0]==arrSP.shape[0]
assert arrCL.shape[0]==arrSP.shape[0]


# =============================================================================
# 
# =============================================================================
NB_EPOCH = 3 #40
BATCH_SIZE = 1000  
VERBOSE = 1

from sklearn.metrics import confusion_matrix
import sklearn
from tensorflow_addons.metrics import F1Score
import tensorflow as tf
from tensorflow import one_hot
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.array([0,1,2,3]), y=arrCL)

model_fname = op.join(MEGnet.__path__[0], 'model/MEGnet_final_model.h5')
kModel = keras.models.load_model(model_fname, compile=False)

# class_weights = {0:1, 1:15, 2:15, 3:15}
from numpy.random import uniform
for idx,layer in enumerate(kModel.layers):
    weights = layer.get_weights()
    if weights==[]:
        continue
    layer_rand_w=[]
    for sublayer in weights:
        layer_rand_w.append(uniform(low=-1.0, high=1.0, size=sublayer.shape))
    layer.set_weights(layer_rand_w)


f1_score=F1Score(4, average='macro')
kModel.compile(
    loss=keras.losses.CategoricalCrossentropy(), 
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), 
    metrics=[f1_score, 'accuracy']
    )

earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                 patience=4,
                                 restore_best_weights=True)







def save_weights_and_history(history, kModel, cv_num):
    # for idx,epoch in enumerate(history):
    epo_dir = op.join(output_dir, f'epoch{str(cv_num)}')
    if not os.path.exists(epo_dir): os.mkdir(epo_dir)
    with open(f'{epo_dir}/history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    kModel.save(f'{epo_dir}/model')
    # with open(f'{epo_dir}/score', 'wb') as file_sc:
    #     pickle.dump(score[idx], file_sc)
              
history_tmp = kModel.fit(x=dict(spatial_input=arrSP, temporal_input=arrTS), y=one_hot(arrCL,4),
                     batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,  
                     # validation_data=(dict(spatial_input=SP_te, temporal_input=TS_te), one_hot(CL_te,4)),
                     class_weight=class_weights, callbacks=[earlystop])

output_dir = op.join(MEGnet.__path__[0], 'model_nih')
save_weights_and_history(history_tmp, kModel, 0)



    