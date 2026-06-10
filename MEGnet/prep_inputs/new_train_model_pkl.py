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
from numpy.random import randint
from numpy.random import shuffle
import copy

tmp=MEGnet.__path__[0]


train_dir = op.join(MEGnet.__path__[0], 'prep_inputs','training')
pkl_topdir = op.join(train_dir, 'Inputs', 'ICAs_pkl_all')


# =============================================================================
# Just load the proper dframe from pkl dir
# =============================================================================
dframe = pd.read_csv(op.join(pkl_topdir, 'final_dframe.csv'))

# =============================================================================
# Load the already saved arrays
# =============================================================================
out_topdir = op.join(train_dir, 'Inputs', 'FULL_nparr_surrogate', 'train')
tr_arrTS = np.load(op.join(out_topdir,'tr_arrTS.npy'))
tr_arrSP = np.load(op.join(out_topdir,'tr_arrSP.npy'))
tr_arrCL = np.load(op.join(out_topdir, 'tr_arrCL.npy'))

#The test data should not be augmented - Load from the original folder
out_topdir = op.join(train_dir, 'Inputs', 'FULL_nparr', 'test')
te_arrTS = np.load(op.join(out_topdir,'te_arrTS.npy'))
te_arrSP = np.load(op.join(out_topdir,'te_arrSP.npy'))
te_arrCL = np.load(op.join(out_topdir, 'te_arrCL.npy'))

assert tr_arrTS.shape[0]==tr_arrSP.shape[0]
assert tr_arrCL.shape[0]==tr_arrSP.shape[0]


# =============================================================================
# Training
# =============================================================================
NB_EPOCH = 30 #40
BATCH_SIZE = 300  
VERBOSE = 1

from sklearn.metrics import confusion_matrix
import sklearn
import os
os.environ['KERAS_BACKEND'] = 'torch'
import keras
from keras import ops

class MacroF1Score(keras.metrics.Metric):
    def __init__(self, num_classes, class_weights=None, name='macro_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.class_weights = self._format_class_weights(class_weights)
        self.true_positives  = self.add_weight(shape=(num_classes,), name='tp', initializer='zeros')
        self.false_positives = self.add_weight(shape=(num_classes,), name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(shape=(num_classes,), name='fn', initializer='zeros')

    def _format_class_weights(self, class_weights):
        if class_weights is None:
            return None
        if isinstance(class_weights, dict):
            class_weights = [class_weights[idx] for idx in range(self.num_classes)]
        if len(class_weights) != self.num_classes:
            raise ValueError(
                f'class_weights must have {self.num_classes} entries; got {len(class_weights)}'
            )
        return tuple(float(weight) for weight in class_weights)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_pred: softmax output (batch, num_classes)
        # y_true: one-hot (batch, num_classes) or integer labels
        y_pred = ops.one_hot(ops.argmax(y_pred, axis=-1), self.num_classes)
        y_pred = ops.cast(y_pred, 'float32')
        y_true = ops.cast(y_true, 'float32')

        tp = ops.sum(y_true * y_pred, axis=0)
        fp = ops.sum((1 - y_true) * y_pred, axis=0)
        fn = ops.sum(y_true * (1 - y_pred), axis=0)

        self.true_positives.assign(self.true_positives + tp)
        self.false_positives.assign(self.false_positives + fp)
        self.false_negatives.assign(self.false_negatives + fn)

    def result(self):
        p = self.true_positives / (self.true_positives + self.false_positives + 1e-7)
        r = self.true_positives / (self.true_positives + self.false_negatives + 1e-7)
        f1_per_class = 2 * (p * r) / (p + r + 1e-7)
        if self.class_weights is not None:
            class_weights = ops.convert_to_tensor(self.class_weights, dtype=f1_per_class.dtype)
            return ops.sum(f1_per_class * class_weights) / (ops.sum(class_weights) + 1e-7)
        return ops.mean(f1_per_class)

    def reset_state(self):
        self.true_positives.assign(ops.zeros((self.num_classes,)))
        self.false_positives.assign(ops.zeros((self.num_classes,)))
        self.false_negatives.assign(ops.zeros((self.num_classes,)))



# import tensorflow as tf
# from tensorflow import one_hot
# from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

clw = compute_class_weight('balanced', classes=np.array([0,1,2,3]), y=tr_arrCL)
clw/=clw.min()
class_weights={i:j for i,j in enumerate(clw)}

model_fname = op.join(MEGnet.__path__[0], 'model_v2k3/model_v2.keras')
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

from sklearn.metrics import f1_score
# f1_score=F1Score(4, average='macro')
kModel.compile(
    loss=keras.losses.CategoricalCrossentropy(), 
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), 
    metrics=['accuracy', MacroF1Score(num_classes=4, class_weights=class_weights)]
    )

earlystop = keras.callbacks.EarlyStopping(monitor='loss',
                                 patience=6,
                                 restore_best_weights=True)



# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder()#sparse_output=False)
# encoded_X = encoder.fit_transform(tr_arrCL)

from sklearn.preprocessing import LabelBinarizer
# Initialize the binarizer
tr_lb = LabelBinarizer()
# Fit and transform to One Hot Encoding
tr_encoded = tr_lb.fit_transform(tr_arrCL)

te_lb = LabelBinarizer()
te_encoded = te_lb.fit_transform(te_arrCL)


def save_weights_and_history(history, kModel, cv_num):
    # for idx,epoch in enumerate(history):
    epo_dir = op.join(output_dir, f'epoch{str(cv_num)}')
    if not os.path.exists(epo_dir): os.mkdir(epo_dir)
    with open(f'{epo_dir}/history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    kModel.save(f'{epo_dir}/model')
    # with open(f'{epo_dir}/score', 'wb') as file_sc:
    #     pickle.dump(score[idx], file_sc)
              
history_tmp = kModel.fit(x=dict(spatial_input=tr_arrSP, temporal_input=tr_arrTS), y=tr_encoded,   #one_hot(tr_arrCL,4),
                     batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,  
                     validation_data=(dict(spatial_input=te_arrSP, temporal_input=te_arrTS), te_encoded), #one_hot(te_arrCL,4)),
                     class_weight=class_weights, callbacks=[earlystop])

# epochs = range(len(history_tmp.history['f1_score']))
# pylab.plot(epochs, history_tmp.history['f1_score'], '+', epochs, history_tmp.history['val_f1_score'],'.')
# pylab.legend(['F1_train','F1_val'])

output_dir='/vf/users/EnigmaMeg/MEGNET/KEEP_TEST_NPY2/MODELFIT/v2_surrogate1'
if not op.exists(output_dir): os.makedirs(output_dir)
with open(f'{output_dir}/history.pkl', 'wb') as file_pi:
    pickle.dump(history_tmp.history, file_pi)
kModel.save(f'{output_dir}/model.keras')


# output_dir = op.join(MEGnet.__path__[0], 'model_nih')
# save_weights_and_history(history_tmp, kModel, 0)



    
