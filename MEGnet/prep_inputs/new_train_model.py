#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:49:16 2022

@author: jstout

    output[VEOG] = 1 
    output[HEOG] = 3
    output[EKG] = 2


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
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt 
from sklearn.metrics import confusion_matrix
import sklearn
from tensorflow_addons.metrics import F1Score
import tensorflow as tf
from tensorflow import one_hot
from tensorflow import keras
from MEGnet.megnet_utilities import fPredictChunkAndVoting, fGetStartTimesOverlap

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('-output_dir')
    # parser.add_argument('-normalize',
    #                     action='store_true',
    #                     help='''Apply min/max normalization on time series
    #                     This will be set to +/- 15 which is the tested mean'''
    #                     )
    parser.add_argument('-weights',
                        default=3,
                        help='Minority weighting for loss calc')
    parser.add_argument('-extended_model',
                        action='store_true',
                        help='Extended model has another dropout layer')
    parser.add_argument('-burn_in',
                        action='store_true',
                        help='''Run a batch with high learning rate with middle
                        layers of the model frozen'''
                        )
    parser.add_argument('-full_frozen',
                        action='store_true',
                        help='Freeze middle layers during all of training'
                        )
    args = parser.parse_args()
    output_dir = args.output_dir
    # if args.normalize==True:
    #     NORMALIZE=True
    # else:
    #    NORMALIZE=False
    if args.extended_model==True:
        MODEL_EXT=True
    else:
        MODEL_EXT=False
    if args.burn_in==True:
        BURN_IN=True
    else:
        BURN_IN=False
    if args.full_frozen==True:
        FREEZE_MIDS = True
    else:
        FREEZE_MIDS = False
    w_ = int(args.weights)
    class_weights={0:1, 1:w_, 2:w_, 3:int(w_)+5}
    if not os.path.exists(output_dir): os.mkdir(output_dir)

# To initialize these not from commandline
# NORMALIZE=True ; MODEL_EXT=False ; BURN_IN = False ; FREEZE_MIDS  = False; class_weights={0:1, 1:15, 2:15, 3:15}
tmp=MEGnet.__path__[0]
dframe_path = op.join(tmp, 'prep_inputs','training', 'Inputs','NIH_CAM_HCP','Final_3site.csv')
dframe = pd.read_csv(dframe_path)

# All loaded vectors are 45000 samples in duration    
train_dir = op.join(MEGnet.__path__[0], 'prep_inputs','training')
np_arr_topdir = op.join(train_dir, 'Inputs', 'NIH_CAM_HCP')
arrTS_fname = op.join(np_arr_topdir, 'arrTS.npy')
arrSP_fname = op.join(np_arr_topdir, 'arrSP.npy')
arrC_ID_fname = op.join(np_arr_topdir, 'arrC_ID.npy')

# Load data and verify everything makes sense
arrTimeSeries = np.load(arrTS_fname)
arrSpatialMap = np.load(arrSP_fname)
class_ID = np.load(arrC_ID_fname)  
assert arrTimeSeries.shape[0] == arrSpatialMap.shape[0]
assert class_ID.shape[0] == arrTimeSeries.shape[0]
assert dframe.__len__() == arrTimeSeries.shape[0]
assert np.alltrue(dframe.ClassID == class_ID)

# Load the keras model
if MODEL_EXT==False:
    model_fname = op.join(MEGnet.__path__[0], 'model/MEGnet_final_model.h5')
    kModel = keras.models.load_model(model_fname, compile=False)
else:
    model_fname = op.join(MEGnet.__path__[0], 'model_extended2')
    kModel = keras.models.load_model(model_fname, compile=False)

# =============================================================================
# Training parameters
# =============================================================================
NB_EPOCH = 200
BATCH_SIZE = 500  
VERBOSE = 1

f1_score=F1Score(4, average='macro')
kModel.compile(
    loss=keras.losses.CategoricalCrossentropy(), 
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), 
    metrics=[f1_score, 'accuracy']
    )

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                 patience=4,
                                 restore_best_weights=True)


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
        scaling_factors = 10 / (max_vals - min_vals)
        mean_vals = np.mean(testTimeSeries, axis=1, keepdims=True)
        testTimeSeries = testTimeSeries - mean_vals
        testTimeSeries = testTimeSeries * scaling_factors 
    return testTimeSeries
    
    
def get_train_test_idx(dframe, fold_number=0):
    '''
    dframe: input dataframe
    fold_number: int
    Returns dictionary {train:train_idx, test:test_idx}
    '''
    fold_str='Fold'+str(fold_number)
    train_idx = dframe[dframe[fold_str]=='Train'].index.values
    test_idx = dframe[dframe[fold_str]=='Test'].index.values
    return {'train':train_idx, 'test':test_idx}


cv = {i:get_train_test_idx(dframe,i) for i in range(7)}
history=[]

# =============================================================================
# Convert multistep sections of data to separate test
# =============================================================================
arrTS_blocks = prep_ts_blocks(arrTimeSeries, overlap=3750, normalize=True)
num_steps = arrTS_blocks.shape[-1]
arrTS_multi = arrTS_blocks.reshape([arrTS_blocks.shape[0]*arrTS_blocks.shape[-1], -1])
arrSP_multi = np.repeat(arrSpatialMap, repeats=num_steps, axis=0)
clID_multi = np.repeat(class_ID, repeats=num_steps, axis=0)

dframe_list = []
for i in range(num_steps):
    tmp= copy.deepcopy(dframe)
    tmp['ts_block']=i
    dframe_list.append(tmp)
dframe_multi = pd.concat(dframe_list)
dframe_multi.reset_index(drop=True, inplace=True)

assert arrTS_multi.shape[0] ==  arrSP_multi.shape[0] == clID_multi.shape[0] == len(dframe_multi)
# =============================================================================
# Burnin trainable layers - 1 round of initialization before possibly unfreeze
# =============================================================================
def freeze_mid_layers(model):
    for idx,layer in enumerate(model.layers[:-2]):
        if idx in [0,1,2,3,10,11,12,13]:
            layer.trainable=True
        else:
            layer.trainable=False
    return model

if BURN_IN==True:
    kModel=freeze_mid_layers(kModel)
    kModel.compile(
        loss=keras.losses.CategoricalCrossentropy(), 
        optimizer=keras.optimizers.Adam(), 
        metrics=[f1_score, 'accuracy']
        )
    
    tr = dframe_multi[dframe_multi['Fold0']=='Train'].index.values  
    te = dframe_multi[dframe_multi['Fold0']=='Test'].index.values 
    
    SP_tr, TS_tr, CL_tr   =  arrSP_multi[tr,:,:,:], arrTS_multi[tr,:], clID_multi[tr]
    SP_te, TS_te, CL_te   =  arrSP_multi[te,:,:,:], arrTS_multi[te,:], clID_multi[te]
                   
    history_tmp = kModel.fit(x=dict(spatial_input=SP_tr, temporal_input=TS_tr), y=one_hot(CL_tr,4),
                         batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,  
                         validation_data=(dict(spatial_input=SP_te, temporal_input=TS_te), one_hot(CL_te,4)),
                         class_weight=class_weights, callbacks=[earlystop])


# =============================================================================
# Handle the FREEZE option
# =============================================================================
if FREEZE_MIDS == False:
    #Make all layers trainable - in case BURN_IN was set to true
    for layer in kModel.layers:
        layer.trainable=True
else:
    kModel=freeze_mid_layers(kModel)

# =============================================================================
#
#         --------- TRAIN THE MODEL -------------
#
# =============================================================================
hold_idx = dframe[dframe.HoldOut==True].index.values
hold_sp, hold_ts, hold_clID = arrSpatialMap[hold_idx,:,:,:], arrTimeSeries[hold_idx,:], class_ID[hold_idx]
# arrTimeSeries=arrTimeSeries[:,:15000] #Temporary Hack

l_rate=4e-4
kModel.compile(
    loss=keras.losses.CategoricalCrossentropy(), 
    optimizer=keras.optimizers.Adam(learning_rate=l_rate), 
    metrics=[f1_score, 'accuracy']
    )

def save_weights_and_history(history, kModel, cv_num):
    # for idx,epoch in enumerate(history):
    epo_dir = op.join(output_dir, f'epoch{str(cv_num)}')
    if not os.path.exists(epo_dir): os.mkdir(epo_dir)
    with open(f'{epo_dir}/history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    kModel.save(f'{epo_dir}/model')
    # with open(f'{epo_dir}/score', 'wb') as file_sc:
    #     pickle.dump(score[idx], file_sc)

score_history=[]
for cv_num in cv.keys():
    fold = f'Fold{cv_num}'
    tr = dframe_multi[dframe_multi[fold]=='Train'].index.values  #cv[cv_num]['train'] #Train numpy indices axis0
    te = dframe_multi[dframe_multi[fold]=='Test'].index.values #cv[cv_num]['test'] #Test numpy indices axis0
    
    SP_tr, TS_tr, CL_tr   =  arrSP_multi[tr,:,:,:], arrTS_multi[tr,:], clID_multi[tr]
    SP_te, TS_te, CL_te   =  arrSP_multi[te,:,:,:], arrTS_multi[te,:], clID_multi[te]
                   
    history_tmp = kModel.fit(x=dict(spatial_input=SP_tr, temporal_input=TS_tr), y=one_hot(CL_tr,4),
                         batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,  
                         validation_data=(dict(spatial_input=SP_te, temporal_input=TS_te), one_hot(CL_te,4)),
                         class_weight=class_weights, callbacks=[earlystop])
    save_weights_and_history(history_tmp, kModel, cv_num)
    history.append(history_tmp)
    l_rate/=2 #Update the learning rate on each crossval
    kModel.compile(
            loss=keras.losses.CategoricalCrossentropy(), 
            optimizer=keras.optimizers.Adam(learning_rate=l_rate), 
            metrics=[f1_score, 'accuracy']
            )
    
    
    # score_history.append(fPredictChunkAndVoting(kModel, hold_ts, hold_sp, hold_clID))

#%% Save outputs

# save_weights_and_history(history)
# kModel.save(f'{output_dir}/model')
    
# =============================================================================
#     Plot and save history
# =============================================================================

# i=0;  j=0
# fig, axes = plt.subplots(3,3)
# for epo in range(8):
#     with open(f'{output_dir}/epoch{str(epo)}/trainHistoryDict', mode='rb') as w:
#         history = pickle.load(w)
#     if i==3:
#         i=0
#         j+=1
#     axes[j,i].plot(history['accuracy'], 'r')
#     axes[j,i].plot(history['f1_score'], 'g')    
#     axes[j,i].plot(history['val_accuracy'], 'b+')
#     axes[j,i].plot(history['val_f1_score'], 'k+')
#     axes[j,i].set_title(f'Epoch{str(epo)}')
#     i+=1
# fig.suptitle('TrAcc(red) TeAcc(blue) TrF1(green) TeF1(black)')
# fig.tight_layout()
# fig.savefig('TestTrain_graph.png', dpi=300) 

# score= fPredictChunkAndVoting(kModel, hold_ts, hold_sp, hold_clID)
# with open(f'score', 'wb') as file_sc:
#     pickle.dump(score, file_sc)

# y_hat = kModel.predict(x=dict(spatial_input=hold_sp, temporal_input=hold_ts))
# y_pred = y_hat.argmax(axis=1)
# matrix = confusion_matrix(hold_clID, y_hat.argmax(axis=1))
# np.save(f'{output_dir}/confusion_mat.npy', matrix)


