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
from sklearn import preprocessing
from matplotlib import pyplot as plt 
from sklearn.metrics import confusion_matrix
import sklearn
from tensorflow_addons.metrics import F1Score
import tensorflow as tf
from tensorflow import one_hot
from tensorflow import keras
from MEGnet.megnet_utilities import fPredictChunkAndVoting

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('-output_dir')
    parser.add_argument('-normalize',
                        action='store_true',
                        help='''Apply min/max normalization on time series
                        This will be set to +/- 15 which is the tested mean'''
                        )
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
    if args.normalize==True:
        NORMALIZE=True
    else:
       NORMALIZE=False
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
    class_weights={0:1, 1:w_, 2:w_, 3:w_}
    if not os.path.exists(output_dir): os.mkdir(output_dir)

# To initialize these not from commandline
# NORMALIZE=True ; MODEL_EXT=False ; BURN_IN = False ; FREEZE_MIDS  = False; class_weights={0:1, 1:10, 2:10, 3:10}

tmp=MEGnet.__path__[0]
dframe_path = op.join(tmp, 'prep_inputs','training', 'final_longform_dframe.csv')
dframe = pd.read_csv(dframe_path)

# All loaded vectors are 45000 samples in duration    
train_dir = op.join(MEGnet.__path__[0], 'prep_inputs','training')
np_arr_topdir = op.join(train_dir, 'Inputs')
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
NB_EPOCH = 400
BATCH_SIZE = 700  
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

if NORMALIZE==True:
    print('Normalizing the data')
    #MinMax timeseries
    mm_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    arrTimeSeries = mm_scaler.fit_transform(arrTimeSeries.T).T * 15
    #DO NOT Normalize Spatial -- Issues with this INT -> Float

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


cv = {i:get_train_test_idx(dframe,i) for i in range(8)}
history=[]

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
    tr = cv[0]['train'] #Train numpy indices axis0
    te = cv[0]['test'] #Test numpy indices axis0
    
    SP_tr, TS_tr, CL_tr   =  arrSpatialMap[tr,:,:,:], arrTimeSeries[tr,:], class_ID[tr]
    SP_te, TS_te, CL_te   =  arrSpatialMap[te,:,:,:], arrTimeSeries[te,:], class_ID[te] 
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
arrTimeSeries=arrTimeSeries[:,:15000] #Temporary Hack

kModel.compile(
    loss=keras.losses.CategoricalCrossentropy(), 
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), 
    metrics=[f1_score, 'accuracy']
    )

score_history=[]
for cv_num in cv.keys():
    tr = cv[cv_num]['train'] #Train numpy indices axis0
    te = cv[cv_num]['test'] #Test numpy indices axis0
    
    SP_tr, TS_tr, CL_tr   =  arrSpatialMap[tr,:,:,:], arrTimeSeries[tr,:], class_ID[tr]
    SP_te, TS_te, CL_te   =  arrSpatialMap[te,:,:,:], arrTimeSeries[te,:], class_ID[te]  
                   
    history_tmp = kModel.fit(x=dict(spatial_input=SP_tr, temporal_input=TS_tr), y=one_hot(CL_tr,4),
                         batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,  
                         validation_data=(dict(spatial_input=SP_te, temporal_input=TS_te), one_hot(CL_te,4)),
                         class_weight=class_weights, callbacks=[earlystop])
    history.append(history_tmp)
    # score_history.append(fPredictChunkAndVoting(kModel, hold_ts, hold_sp, hold_clID))

#%% Save outputs
def save_weights_and_history(history):
    for idx,epoch in enumerate(history):
        epo_dir = op.join(output_dir, f'epoch{idx}')
        os.mkdir(epo_dir)
        with open(f'{epo_dir}/trainHistoryDict', 'wb') as file_pi:
            pickle.dump(epoch.history, file_pi)
        # with open(f'{epo_dir}/score', 'wb') as file_sc:
        #     pickle.dump(score[idx], file_sc)

save_weights_and_history(history)
kModel.save(f'{output_dir}/model')
    
# =============================================================================
#     Plot and save history
# =============================================================================

i=0;  j=0
fig, axes = plt.subplots(3,3)
for epo in range(8):
    with open(f'{output_dir}/epoch{str(epo)}/trainHistoryDict', mode='rb') as w:
        history = pickle.load(w)
    if i==3:
        i=0
        j+=1
    axes[j,i].plot(history['accuracy'], 'r')
    axes[j,i].plot(history['f1_score'], 'g')    
    axes[j,i].plot(history['val_accuracy'], 'b+')
    axes[j,i].plot(history['val_f1_score'], 'k+')
    axes[j,i].set_title(f'Epoch{str(epo)}')
    i+=1
fig.suptitle('TrAcc(red) TeAcc(blue) TrF1(green) TeF1(black)')
fig.tight_layout()
fig.savefig('TestTrain_graph.png', dpi=300) 

score= fPredictChunkAndVoting(kModel, hold_ts, hold_sp, hold_clID)
with open(f'score', 'wb') as file_sc:
    pickle.dump(score, file_sc)

y_hat = kModel.predict(x=dict(spatial_input=hold_sp, temporal_input=hold_ts))
y_pred = y_hat.argmax(axis=1)
matrix = confusion_matrix(hold_clID, y_hat.argmax(axis=1))
np.save(f'{output_dir}/confusion_mat.npy', matrix)


