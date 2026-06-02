#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 11:55:19 2025

@author: jstout
"""


import MEGnet
import os, os.path as op

_megnet_path = MEGnet.__path__[0]
weights_path = op.join(_megnet_path, 'model_v2_k3')

def _check_weights():
    if op.exists(weights_path):
        return True
    else:
        return False
        

def main():
    """
    Check for existence of model and perform the download if not present
    """
    
    if _check_weights():
        print('Model weights present - check successful')
    else:
        print(f'''Model weights were not found in:  
              {weights_path}
              Performing download from huggingface repository''')
              
        # Download the data
        from huggingface_hub import snapshot_download
        
        try:
            snapshot_download(
                repo_id='jstout211/MEGnetV2',
                local_dir= _megnet_path,
                local_dir_use_symlinks=False,  
                revision='c6f12ab',
                allow_patterns=["model_v2/*"]
            )
        except BaseException as e:
            print('Could not download the weights for classification')
            print('This is likely an issue with network access to the huggingface repository')
            print('')
            print(f'Error:  {e}')
            
