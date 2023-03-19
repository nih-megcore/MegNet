#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 13:06:00 2023

@author: jstout
"""

from MEGnet.prep_inputs.cvSplits import main as cvSplitter
import MEGnet
import pandas as pd
import os.path as op, os
input_fname = op.join(MEGnet.__path__[0], 'prep_inputs', 'training', 'ICA_combined_participants.tsv')

dframe = pd.read_csv(input_fname, sep='\t')

def test_cvSplitter():
    check=cvSplitter(kfolds=5,
               foldNormFields=['Site','sex','age','scanner'],
               data_dframe=dframe)