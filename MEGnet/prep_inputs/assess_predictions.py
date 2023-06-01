#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:13:22 2023

@author: jstout
"""

import os, os.path as op
import copy
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from tensorflow import keras
import numpy as np
import pandas as pd
import pickle
import MEGnet
from MEGnet.megnet_utilities import fPredictChunkAndVoting, fGetStartTimesOverlap
import tensorflow_addons
from sklearn.metrics import confusion_matrix, f1_score
import sys






# with open('history.pkl','rb') as f:
#     history = pickle.load(f)
    
# import pylab
# pylab.plot(history['val_accuracy'])


train_dir = op.join(MEGnet.__path__[0], 'prep_inputs','training')
np_arr_topdir = op.join(train_dir, 'Inputs','NIH_CAM_HCP')
arrTS_fname = op.join(np_arr_topdir, 'arrTS.npy')
arrSP_fname = op.join(np_arr_topdir, 'arrSP.npy')
arrC_ID_fname = op.join(np_arr_topdir, 'arrC_ID.npy')
dframe_fname = op.join(np_arr_topdir, 'Final_3site.csv')

## Load the datasets
arrTS = np.load(arrTS_fname)
arrSP = np.load(arrSP_fname)
cl = np.load(arrC_ID_fname)
dframe = pd.read_csv(dframe_fname)

## Generate the holdout
hold_idx = dframe[dframe.HoldOut==True].index
arrTS = arrTS[hold_idx, :]
arrSP = arrSP[hold_idx, :,:,:]
cl = cl[hold_idx]



lTimeSeries = arrTS
arrSpatialMap = arrSP
arrY = cl
intModelLen=15000
intOverlap=3750

def fPredictChunkAndVoting(kModel, lTimeSeries, arrSpatialMap, arrY, intModelLen=15000, intOverlap=3750):
    """
    This function is designed to take in ICA time series and a spatial map pair and produce a prediction useing a trained model.
    The time series will be split into multiple chunks and the final prediction will be a weighted vote of each time chunk.
    The weight for the voting will be determined by the manout of time and overlap each chunk has with one another.
    For example if the total lenght of the scan is 50 seconds, and the chunks are 15 seconds long with a 5 second overlap:
        The first chunk will be the only chunk to use the first 10 seconds, and one of two chunks to use the next 5 seconds.
            Thus   

    :param kModel: The model that will be used for the predictions on each chunk. It should have two inputs the spatial map and time series respectivley
    :type kModel: a keras model
    :param lTimeSeries: The time series for each scan (can also be an array if all scans are the same lenght)
    :type lTimeSeries: list or array (if each scan is a different length, then it needs to be a list)
    :param arrSpatialMap: The spatial maps (one per scan)
    :type arrSpatialMap: numpy array
    :param intModelLen: The lenght of the time series in the model, defaults to 15000
    :type intModelLen: int, optional
    :param intOverlap: The lenght of the overlap between scans, defaults to 3750
    :type intOverlap: int, optional
    """
    #empty list to hold the prediction for each component pair
    lPredictionsVote = []
    lGTVote = []

    lPredictionsChunk = []
    lGTChunk = []

    i = 0
    num_subjs = arrSpatialMap.shape[0]//20
    arrSpatialMap = arrSpatialMap.reshape(arrSpatialMap.shape[0]//20, 20, 120,120,3)
    lTimeSeries = lTimeSeries.reshape(lTimeSeries.shape[0]//20, 20, -1)
    
    for subj_idx in range(num_subjs):
    # for arrScanTimeSeries, arrScanSpatialMap, arrScanY in zip(lTimeSeries, arrSpatialMap, arrY):
        arrScanTimeSeries=lTimeSeries[subj_idx, :,:]
        arrScanSpatialMap = arrSpatialMap[subj_idx,:,:,:,:]
        intTimeSeriesLen = lTimeSeries.shape[-1]
        lStartTimes = fGetStartTimesOverlap(intTimeSeriesLen, intModelLen=intModelLen, intOverlap=intOverlap)

        if lStartTimes[-1]+intModelLen <= intTimeSeriesLen:
            lStartTimes.append(arrScanTimeSeries.shape[0]-intModelLen)
            
        lTimeChunks = [[x,x+intModelLen] for x in lStartTimes]
        dctTimeChunkVotes = dict([[x,0] for x in lStartTimes])
        for intT in range(intTimeSeriesLen):
            lChunkMatches = [x <= intT < x+intModelLen for x in dctTimeChunkVotes.keys()]
            intInChunks = np.sum(lChunkMatches)
            for intStartTime, bTruth in zip(dctTimeChunkVotes.keys(), lChunkMatches):
                if bTruth:
                    dctTimeChunkVotes[intStartTime]+=1.0/intInChunks

        #predict
        dctWeightedPredictions = {}
        dctTimeChunkVotes = {i:j for i,j in dctTimeChunkVotes.items() if i>0}
        for intStartTime in dctTimeChunkVotes.keys():
            testTimeSeries = copy.deepcopy(arrScanTimeSeries[:, intStartTime:intStartTime+intModelLen])
            min_vals = np.min(testTimeSeries, axis=1, keepdims=True)
            max_vals = np.max(testTimeSeries, axis=1, keepdims=True)
            scaling_factors = 8 / (max_vals - min_vals)
            mean_vals = np.mean(testTimeSeries, axis=1, keepdims=True)
            testTimeSeries = testTimeSeries - mean_vals
            testTimeSeries = testTimeSeries * scaling_factors 
            
            lPrediction = kModel.predict([arrScanSpatialMap,testTimeSeries])
            lPredictionsChunk.append(lPrediction)
            
            dctWeightedPredictions[intStartTime] = lPrediction*dctTimeChunkVotes[intStartTime]

        arrScanPrediction = np.stack(dctWeightedPredictions.values())
        arrScanPrediction = arrScanPrediction.mean(axis=0)
        arrScanPrediction = arrScanPrediction/arrScanPrediction.sum()
        lPredictionsVote.append(arrScanPrediction)
        
        #print(f"{i}/{arrY.shape[0]}")
        i+=1
    lPredictionsVote = np.stack(lPredictionsVote)
    lPredictionsVote = lPredictionsVote.reshape(lPredictionsVote.shape[0]*lPredictionsVote.shape[1],-1)
    return lPredictionsVote #, np.stack(lGTVote) #, np.stack(lPredictionsChunk), np.stack(lGTChunk)

# =============================================================================
# Run the prediction scoring
# =============================================================================

topdir = sys.argv[1] #'/tmp/TESTing_053023/extended_w15_wBurnIn'



def calc_stats( epo_num):
    dirname = op.join(topdir,f'epoch{epo_num}') #/tmp/TESTing_053023/extended_w15_wBurnIn/epoch5'
    os.chdir(dirname)

    kModel = keras.models.load_model('./model', compile=False)
    arrPredicionsVote = fPredictChunkAndVoting(kModel, arrTS, arrSP, cl, 
                                               intModelLen=15000, 
                                               intOverlap=3750)
    ypred = arrPredicionsVote.argmax(axis=1)
    return ypred

epoch_eval={}    
for epo_num in [0,1,2,3,4,5, 6]:
    epoch_eval[f'epoch{epo_num}'] = calc_stats(epo_num)
    pred_fname = op.join(topdir, f'epoch{epo_num}', 'predictions.npy')
    np.save(pred_fname, epoch_eval[f'epoch{epo_num}'])

out_fname = op.join(topdir, 'results.txt')
with open(out_fname, 'w') as f:
    for epo_num in [0,1,2,3,4,5, 6]:
        ypred = epoch_eval[f'epoch{epo_num}']
        conf_mat = confusion_matrix(cl, ypred)
        np.save(op.join(topdir, f'epoch{epo_num}', 'confusion_mat.npy') ,conf_mat)
        acc = np.sum(cl==ypred)/len(cl)
        ax1_sum = conf_mat.sum(axis=1)
        diag_vals = copy.deepcopy(np.diag(conf_mat))
        np.fill_diagonal(conf_mat, 0)
        positive_pred = diag_vals / ax1_sum
        false_neg = conf_mat.sum(axis=1)/ax1_sum
        # false_pos = conf_mat.sum(axis=0)/ax1_sum
        f1score = f1_score(cl, ypred, average='macro')
        
        print(f'######### EPOCH{epo_num} #########')
        print(f'Accuracy:  {acc}')
        print(f'PosPred: {positive_pred}')
        print(f'FalseNeg: {false_neg}')
        print(f'F1score: {f1score}')
        f.write(f'######### EPOCH{epo_num} #########\n')
        f.write(f'Accuracy:  {acc}\n')
        f.write(f'PosPred: {positive_pred}\n')
        f.write(f'FalseNeg: {false_neg}\n')
        f.write(f'F1score: {f1score}\n')
        
        







