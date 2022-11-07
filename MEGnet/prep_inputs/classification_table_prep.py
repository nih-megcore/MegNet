#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 09:49:15 2022

@author: jstout

"""

import os, os.path as op
import pandas as pd
import datetime
import numpy as np

topdir = '/home/jstout/src/MegNET2022'
os.chdir(topdir)
class_xlsx_fname = op.join(topdir, 'ICA-1.xlsx')

#Results in a dict with sheet names as keys
all_sheets = pd.read_excel(class_xlsx_fname, sheet_name=None)
['ICA_1', 'results_mous', 'NIH_y', 'NIH_hvprotocol', 'results_camcan', 'results_camcan_batch2']

# =============================================================================
# # Dataset information from Manual Classification XLS file
# =============================================================================
# Scanner type: CTF, ELEKTA, 4D, KIT
test = pd.DataFrame(columns=['SheetName','Scanner','Site','TaskType'])
test.loc[len(test)]=['ICA_1','CTF','NIH',None]
test.loc[len(test)]=['results_mous','CTF','Donders',None]
test.loc[len(test)]=['NIH_y', 'CTF','NIH',None]
test.loc[len(test)]=['NIH_hvprotocol', 'CTF','NIH',None]
test.loc[len(test)]=['results_camcan', 'Elekta','Cambridge',None]
test.loc[len(test)]=['results_camcan_batch2', 'Elekta','Cambridge',None]
test.loc[len(test)]=['results_camcan_rest', 'Elekta','Cambridge',None]

#Combine into 1 sheet
sheet_list = []
for key,value in all_sheets.items():
    cols = test.columns.values
    if key=='results_camcan_batch2':
        key='results_camcan'
    if key=='results_camcan_rest':
        key='results_camcan'
    append_vals = test[test['SheetName']==key].values
    value.loc[:,cols]=append_vals[0]
    sheet_list.append(value)

ica_data = pd.concat(sheet_list)
ica_data.reset_index(drop=True, inplace=True)


# =============================================================================
# Get age/gender from participants.tsv
# Download the all_demographics folder from biowulf enigma/bids/all_demographics
# =============================================================================
demo_dir = '/tmp/all_demographics'
demo=dict(ICA_1='NIHstr_participants.tsv',
      results_mous='mous_participants.tsv',
      NIH_y='NIHyan_participants.tsv',
      NIH_hvprotocol='NIHhv_participants.tsv')

def load_demo(fname):
    return pd.read_csv(op.join(demo_dir, fname), sep='\t')

demo_list = []
column_sel = ['participant_id', 'age', 'sex','SheetName']

#NIHstr data
tmp = load_demo('NIHstr_participants.tsv')
tmp['SheetName']='ICA_1'
tmp.rename(columns=dict(participant_age='age'), inplace=True)
demo_list.append(tmp.loc[:,column_sel])

#NIHy data
tmp = load_demo('NIHyan_participants.tsv')
tmp['SheetName']='NIH_y'
tmp.rename(columns=dict(participant_age='age'), inplace=True)
demo_list.append(tmp.loc[:,column_sel])

#NIHhv data
tmp = load_demo('NIHhv_participants.tsv')
tmp['SheetName']='NIH_hvprotocol'
demo_list.append(tmp.loc[:,column_sel])

#Mous data
tmp = load_demo('mous_participants.tsv')
tmp['SheetName']='results_mous'
demo_list.append(tmp.loc[:,column_sel])

#Camcan data
tmp = load_demo('cam_participants.tsv')
tmp['SheetName']='results_camcan'
demo_list.append(tmp.loc[:,column_sel])

demo_final = pd.concat(demo_list)
demo_final.reset_index(drop=True, inplace=True)



# =============================================================================
# Merge demographic and ICA results
# =============================================================================
#ICAColumns
#['idx', 'sub', 'type', 'eyeblink', 'Saccade', 'EKG', 'other',
#       'SheetName', 'Scanner', 'Site', 'TaskType', 'Unnamed: 6']

def munge_subjid(subj):
    subj=str(subj)
    if subj[0:4]!='sub-':
        subj='sub-'+subj
    return subj

def fix_nihy_subjids(dframe):
    'Specifically parse datasets from NIH_y - subject IDs need to conform'
    for idx, row in dframe.iterrows():
        if row.SheetName=='NIH_y':
            print(f"Fixing ID {dframe.loc[idx]['participant_id']}")
            if len(str(dframe.loc[idx]['participant_id']))==1:
                dframe.loc[idx, 'participant_id']='sub-000'+str(dframe.loc[idx]['participant_id'])
            elif  len(str(dframe.loc[idx,'participant_id']))==2:
                dframe.loc[idx, 'participant_id']='sub-00'+str(dframe.loc[idx]['participant_id'])
            else:
                print(f'Error with {dframe.loc[idx]["participant_id"]}')
    

ica_data.rename(columns=dict(sub='participant_id'), inplace=True)
fix_nihy_subjids(ica_data)
ica_data['participant_id']=ica_data['participant_id'].apply(munge_subjid)

all_dat = pd.merge(ica_data, demo_final, on=['participant_id','SheetName'])

all_dat[['participant_id', 'type']]
#Cleanup
#Drop subjects without age
all_dat=all_dat.dropna(subset=['age'])
all_dat['age']=all_dat['age'].astype(int)
all_dat['age'].hist()



#Change coding of M/F for hv protocol
hv_m_idx = all_dat[(all_dat.SheetName=='NIH_hvprotocol') & (all_dat.sex==1)].index
hv_f_idx = all_dat[(all_dat.SheetName=='NIH_hvprotocol') & (all_dat.sex==2)].index
all_dat.loc[hv_m_idx,'sex']='M'
all_dat.loc[hv_f_idx,'sex']='F'


#Use single letter for gender
all_dat['sex']=all_dat['sex'].str[0].apply(str.upper)


#Remove run number from mmi task id
all_dat.loc[all_dat.type.str[0:4]=='mmi3','type']='mmi3'


# =============================================================================
# Demographic level stats
# =============================================================================
group_demo_data=all_dat.drop_duplicates(subset=['participant_id','SheetName'])
group_demo_data.age.hist(density=True)

group_demo_data.sex.hist()

group_demo_data.groupby(['Site','sex'])['age'].mean()
group_demo_data.groupby(['Site'])['age'].hist(density=True, alpha=0.5)
# =============================================================================
# ICA stats
# =============================================================================

def sync_class(val):
    if type(val)==datetime.datetime:
        val=[]
    if type(val)==int:
        val=list(str(val))
    if type(val)==float:
        if np.isnan(val):
            val=[]
        else:
            val=list(str(int(val)))
    if ',' in val:
        val = list(val.split(','))
        val = [str(i) for i in val]  # this is possibly not necessary
    return val
    

# Munge the ICA classes
all_dat['eyeblink']=all_dat.eyeblink.apply(sync_class)
all_dat['Saccade']=all_dat.Saccade.apply(sync_class)
all_dat['EKG']=all_dat.EKG.apply(sync_class)

all_dat['eyeblink_ct']=all_dat.eyeblink.apply(len)
all_dat['saccade_ct']=all_dat.Saccade.apply(len)
all_dat['ekg_ct']=all_dat.EKG.apply(len)

#Get rid of datasets with more than 4 components in one category # these are just entry label errors
all_dat = all_dat[all_dat.eyeblink_ct < 5]
all_dat = all_dat[all_dat.ekg_ct < 5]
all_dat = all_dat[all_dat.saccade_ct < 5]


all_dat.groupby('sex').ekg_ct.hist(density=True)
all_dat.groupby('sex').eyeblink_ct.hist(density=True)
all_dat.groupby('sex').saccade_ct.hist(density=True) 
all_dat.groupby('sex').saccade_ct.plot(kind='hist', density=1, bins=5, stacked=False, alpha=.5)


# =============================================================================
# Write out final dataset
# =============================================================================
all_dat.to_csv('ICA_combined_participants.tsv', sep='\t', index=False) 
