#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 21:58:33 2023

@author: jstout
"""

idxs = final.index[final.Scanner.isin(['BTI','CTF'])]

arrTimeSeries, arrSpatialMap, class_ID = load_all_inputs()

arrTimeSeries=arrTimeSeries[idxs, :] 
arrSpatialMap=arrSpatialMap[idxs, :, : ,:]
class_ID = class_ID[idxs]
final=final.loc[idxs]


# def zscore(ins):
#     m = np.mean(ins)
#     s = np.std(ins)
#     return (ins-m) / s


#Mask and zscore images 
from scipy.stats import zscore

def per_subj_zscore(ins):
    '''Normalize on a per subject basis'''
    tmp = ins.reshape(int(ins.shape[0]/20), 20*ins.shape[-1])
    return zscore(tmp, axis=-1).reshape(ins.shape)

def mask_zscore(ins, mask=None):
    '''Mask out the circular ROI and perform zscore across subjects and
    pixels
    
    Spatial maps: ICA x X x Y x Color
    mask: 3D mask
    '''
    ins = copy.deepcopy(ins)
    spShape = ins.shape
    flat_mask = mask.reshape([-1])
    test = ins.reshape([spShape[0], -1])
    masked_test = test[:, flat_mask==1]
    test[:, flat_mask==1] = zscore(masked_test, axis=-1) 
    return test.reshape(spShape)


# def per_subj_minmax(ins):
#     mm_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
#     ins = copy.deepcopy(ins)
#     #Use a loop to do within subject normalization
#     for idxs in range(int(len(ins)/20)):
#         start=0+idxs*20
#         end=20+idxs*20
#         tmp = ins[start:end,:]
#         ins[start:end,:] = mm_scaler.fit_transform(tmp)
#     return ins

# def mask_subj_minmax(ins):
#     mm_scaler = preprocessing.MinMaxScaler(range(-1,1))
#     ins = copy.deepcopy(ins)
#     spShape = ins.shape
#     flat_mask = mask.reshape([-1])
#     test = ins.reshape([spShape[0], -1])
#     masked_test = test[:, flat_mask==1]
#     test[:, flat_mask==1] = zscore(masked_test, axis=-1) 
#     return test.reshape(spShape)    


# =============================================================================
# Normalize data
# =============================================================================
    
MEGNET_path = MEGnet.__path__[0]
train_path = op.join(MEGNET_path, 'prep_inputs', 'training')
mask_3d = op.join(train_path, 'mask3.npy')
mask3 = np.load(mask_3d)

#Temporal zscore
arrTimeSeries = zscore(arrTimeSeries, axis=1)

#Spatial map zscore
spShape = arrSpatialMap.shape
flat_mask = mask3.reshape([-1])
test = arrSpatialMap.reshape([spShape[0], -1])
# Mask the data
masked_test = test[:, flat_mask==1]


#MinMax timeseries
mm_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
arrTimeSeries = mm_scaler.fit_transform(arrTimeSeries.T).T

#MinMax images
mm_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
tmp = arrSpatialMap.reshape([arrSpatialMap.shape[0],-1])
arrSpatialMap = mm_scaler.fit_transform(tmp.T).T.reshape(arrSpatialMap.shape)
