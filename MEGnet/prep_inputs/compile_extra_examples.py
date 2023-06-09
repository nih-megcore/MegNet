#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:25:57 2023

@author: jstout
"""

import pandas as pd
import numpy as np
import glob
import os, os.path as op
import sys

tmp=sys.argv[1]

topdir = op.dirname(tmp)
site = op.basename(tmp)

# topdir = '/tmp'
# site = 'Aston'
# dsets = glob.glob(f'{topdir}/{site}/*/extra')




na_dsets = glob.glob(f'{topdir}/{site}/*/extra/na_tr.npy')
veog_dsets = glob.glob(f'{topdir}/{site}/*/extra/veog_tr.npy')
ecg_dsets = glob.glob(f'{topdir}/{site}/*/extra/ecg_tr.npy')
heog_dsets = glob.glob(f'{topdir}/{site}/*/extra/heog_tr.npy')

dsets = na_dsets + veog_dsets + ecg_dsets + heog_dsets
dframe = pd.DataFrame(dsets, columns=['fname'])
dframe['Site']=site

def get_class_ID(fname):
    tmp = op.basename(fname)
    if tmp=='na_tr.npy':
        return 0
    elif tmp=='veog_tr.npy':
        return 1
    elif tmp=='ecg_tr.npy':
        return 2
    elif tmp=='heog_tr.npy':
        return 3
    else:
        raise()

def get_subjid(fname):
    tmp=fname.split('/')[-3].split('_')
    return [i for i in tmp if 'sub-' in i][0]

def get_task(fname):
    tmp=fname.split('/')[-3].split('_')
    return [i for i in tmp if 'task-' in i][0]

def get_run(fname):
    tmp=fname.split('/')[-3].split('_')
    tmp = [i for i in tmp if 'run-' in i]
    if len(tmp)==0:
        return None
    else:
        return tmp[0]

def get_session(fname):
    tmp=fname.split('/')[-3].split('_')
    tmp = [i for i in tmp if 'ses-' in i]
    if len(tmp)==0:
        return None
    else:
        return tmp[0]


dframe['ClassID']=dframe.fname.apply(get_class_ID)
dframe['subject']=dframe.fname.apply(get_subjid)
dframe['task']=dframe.fname.apply(get_task)
dframe['session']=dframe.fname.apply(get_session)
dframe['run'] = dframe.fname.apply(get_run)

dframe.sort_values(by=['subject','task','session','run'], inplace=True)
dframe.reset_index(inplace=True, drop=True)


arrTS = np.zeros([len(dframe), 15000])
arrSP = np.zeros([len(dframe), 120,120,3], dtype=np.int) 
    

def create_np_array(dframe):
    for idx,row in dframe.iterrows():
        arrTS[idx,:] = np.load(row.fname)
        arrSP[idx,:,:,:] = np.load(row.fname[:-6]+'sp.npy')

create_np_array(dframe)
# =============================================================================
# Save outputs
# =============================================================================
dframe.to_csv(op.join(topdir, site, 'extra_test_dsets.csv'))
np.save(op.join(topdir, site, 'extra_test_arrTS.npy'), arrTS)
np.save(op.join(topdir, site, 'extra_test_arrSP.npy'), arrSP)



