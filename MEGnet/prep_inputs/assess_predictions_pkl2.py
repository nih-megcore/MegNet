#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 12:55:01 2023

@author: jstout
"""

import os, os.path as op
import glob
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


pkl_dir = '/home/jstout/src/MegNET2022/MEGnet/prep_inputs/training/Inputs/PKL'
pkl_topdir = pkl_dir
dframe_hold = pd.read_csv(op.join(pkl_topdir, 'dframe_holdout.csv'))

# =============================================================================
# assess predictions
# =============================================================================

def fPredictChunkAndVoting(kModel, lTimeSeries, arrSpatialMap, intModelLen=15000, intOverlap=3750):
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
    allChunkPredictions = []

    i = 0
    num_subjs = arrSpatialMap.shape[0]//20
    arrSpatialMap = arrSpatialMap.reshape(arrSpatialMap.shape[0]//20, 20, 120,120,3)
    lTimeSeries = lTimeSeries.reshape(lTimeSeries.shape[0]//20, 20, -1)
    
    for subj_idx in range(num_subjs):
    # for arrScanTimeSeries, arrScanSpatialMap, arrScanY in zip(lTimeSeries, arrSpatialMap, arrY):
        lPredictionsChunk=[]
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
        allChunkPredictions.append(np.stack(lPredictionsChunk, axis=-1))
        #print(f"{i}/{arrY.shape[0]}")
        i+=1
    lPredictionsVote = np.stack(lPredictionsVote)
    lPredictionsVote = lPredictionsVote.reshape(lPredictionsVote.shape[0]*lPredictionsVote.shape[1],-1)
    return lPredictionsVote , np.stack(allChunkPredictions) #, np.stack(lGTChunk)

# =============================================================================
# Run the prediction scoring
# =============================================================================


kModel_fname = op.join(MEGnet.__path__[0], 'model_v2')
kModel = keras.models.load_model(kModel_fname, compile=False)

holdout = {}
for idx, input_vec in dframe_hold.iterrows():
    outdir = op.join(pkl_dir, f'{input_vec.Site}_{input_vec.subjid}_{input_vec.TaskType}')
    print(outdir)
    ts_pklfile = op.join(outdir, 'ts.pkl')
    sp_file = op.join(outdir, 'sp.npy')
    cl_file = op.join(outdir, 'cl.npy')
    holdout[idx]={'ts':None, 'sp':None, 'cl':None}
    with open(ts_pklfile, 'rb') as f: holdout[idx]['ts'] = pickle.load(f)
    with open(sp_file, 'rb') as g: holdout[idx]['sp'] = np.load(g)
    with open(cl_file, 'rb') as h: holdout[idx]['cl'] = np.load(h)


for idx in holdout.keys():  #These are the keys to the dictionary not dataframe
    arrTS = holdout[idx]['ts']
    arrSP = holdout[idx]['sp']
    arrPredicionsVote, arrProb = fPredictChunkAndVoting(kModel, arrTS, arrSP,  
                                           intModelLen=15000, 
                                           intOverlap=3750)
    holdout[idx]['arrPred'] = arrPredicionsVote.argmax(axis=1)
    holdout[idx]['arrProb'] = arrProb


ypred = np.hstack([holdout[i]['arrPred'] for i in holdout.keys()])
cl = np.hstack([holdout[i]['cl'] for i in holdout.keys()])

conf_mat = confusion_matrix(cl, ypred)
acc = np.sum(cl==ypred)/len(cl)
ax1_sum = conf_mat.sum(axis=1)
diag_vals = copy.deepcopy(np.diag(conf_mat))
np.fill_diagonal(conf_mat, 0)
positive_pred = diag_vals / ax1_sum
false_neg = conf_mat.sum(axis=1)/ax1_sum
# false_pos = conf_mat.sum(axis=0)/ax1_sum
f1score = f1_score(cl, ypred, average='macro')

# print(f'######### {dirname} #########')
print(f'Accuracy:  {acc}')
print(f'PosPred: {positive_pred}')
print(f'FalseNeg: {false_neg}')
print(f'F1score: {f1score}')
# f.write(f'######### {dirname} #########\n')
# f.write(f'Accuracy:  {acc}\n')
# f.write(f'PosPred: {positive_pred}\n')
# f.write(f'FalseNeg: {false_neg}\n')
# f.write(f'F1score: {f1score}\n')


# =============================================================================
# Done
# =============================================================================
# epoch_eval={}    
# for dirname in glob.glob('epo*'): #epo_num in [0,1,2,3,4,5, 6]:
#     pred, prob = calc_stats(dirname)
#     epoch_eval[f'{dirname}'] = pred
#     epoch_eval[f'{dirname}_prob'] = prob
#     pred_fname = op.join(topdir, f'{dirname}', 'predictions.npy')
#     prob_fname = op.join(topdir, f'{dirname}', 'probabilities.npy')
#     np.save(pred_fname, epoch_eval[f'{dirname}'])
#     np.save(prob_fname, epoch_eval[f'{dirname}_prob'])
#     conf_mat = confusion_matrix(cl, pred)
#     np.save(op.join(topdir, f'{dirname}', 'confusion_mat.npy') ,conf_mat)

# os.chdir(topdir)
# out_fname = op.join(topdir, 'results.txt')
# with open(out_fname, 'w') as f:
#     for dirname in glob.glob('epo*'): #[0,1,2,3,4,5, 6]:
#         ypred = epoch_eval[f'{dirname}']
#         conf_mat = confusion_matrix(cl, ypred)
#         acc = np.sum(cl==ypred)/len(cl)
#         ax1_sum = conf_mat.sum(axis=1)
#         diag_vals = copy.deepcopy(np.diag(conf_mat))
#         np.fill_diagonal(conf_mat, 0)
#         positive_pred = diag_vals / ax1_sum
#         false_neg = conf_mat.sum(axis=1)/ax1_sum
#         # false_pos = conf_mat.sum(axis=0)/ax1_sum
#         f1score = f1_score(cl, ypred, average='macro')
        
#         print(f'######### {dirname} #########')
#         print(f'Accuracy:  {acc}')
#         print(f'PosPred: {positive_pred}')
#         print(f'FalseNeg: {false_neg}')
#         print(f'F1score: {f1score}')
#         f.write(f'######### {dirname} #########\n')
#         f.write(f'Accuracy:  {acc}\n')
#         f.write(f'PosPred: {positive_pred}\n')
#         f.write(f'FalseNeg: {false_neg}\n')
#         f.write(f'F1score: {f1score}\n')
        
        