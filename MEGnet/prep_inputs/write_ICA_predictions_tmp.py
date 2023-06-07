#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:13:22 2023

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
from MEGnet.megnet_utilities import fGetStartTimesOverlap
import tensorflow_addons
from sklearn.metrics import confusion_matrix, f1_score
import sys
from scipy.io import loadmat



topdir = sys.argv[1]
os.chdir(topdir)
arrTS_fname = op.join(topdir, 'ICATimeSeries.mat')
arrTS = loadmat(arrTS_fname)['arrICATimeSeries'].T

model_fname = op.join(MEGnet.__path__[0], 'model', 'MEGnet_final_model.h5')
kModel = keras.models.load_model(model_fname)

def load_SP(topdir):
    sp_list = []
    for i in range(1,21):
        tmp = loadmat(op.join(topdir, f'component{i}.mat'))['array']
        sp_list.append(tmp)
    output = np.stack(sp_list)
    assert output.shape == (20,120,120,3)
    return output
    
arrSP = load_SP(topdir)

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
    
    # if arrY==None:
    #     arrY=np.zeros(len(arrSpatialMap))

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
    return lPredictionsVote , np.stack(allChunkPredictions), lStartTimes  #, np.stack(lGTChunk)

# =============================================================================
# 
# =============================================================================
ypred, yprob, starts = fPredictChunkAndVoting(kModel, arrTS, arrSP)
yprob = yprob.squeeze() #Only 1 subject - so remove leading singleton

pred_fname = op.join(topdir, 'predictions.npy')
prob_fname = op.join(topdir, 'probabilities.npy')
starts_fname = op.join(topdir, 't_starts.npy')
np.save(pred_fname, ypred)
np.save(prob_fname, yprob)
np.save(starts_fname, starts)







