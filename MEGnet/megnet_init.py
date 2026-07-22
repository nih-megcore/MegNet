#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 11:55:19 2025

@author: jstout
"""


import json
import re
import MEGnet
import os, os.path as op

_megnet_path = MEGnet.__path__[0]
weights_path = op.join(_megnet_path, 'model_v2k3')
config_path = op.join(weights_path, 'config.json')
min_model_version = 'v2.1'


def _version_tuple(version):
    version_match = re.match(r'^v?(\d+(?:\.\d+)*)$', version)
    if not version_match:
        return None
    return tuple(int(i) for i in version_match.group(1).split('.'))


def _check_weights():
    if not op.exists(config_path):
        return False

    try:
        with open(config_path, encoding='utf-8') as fid:
            config = json.load(fid)
    except (OSError, json.JSONDecodeError):
        return False

    model_version = config.get('model_version')
    if not isinstance(model_version, str):
        return False

    model_version_tuple = _version_tuple(model_version)
    min_model_version_tuple = _version_tuple(min_model_version)
    if model_version_tuple is None or min_model_version_tuple is None:
        return False

    return model_version_tuple > min_model_version_tuple


def _download_weights():
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id='jstout211/MEGnetV2',
        local_dir=_megnet_path,
        local_dir_use_symlinks=False,
        allow_patterns=["model_v2k3/*"],
        force_download=True
    )


def main():
    """
    Check for existence of model and perform the download if not present
    """
    
    if _check_weights():
        print('Model weights present - check successful')
    else:
        print(f'''Model weights are missing or out of date in:
              {weights_path}
              Pulling newest weights from huggingface repository''')

        try:
            _download_weights()
        except BaseException as e:
            print('Could not download the weights for classification')
            print('This is likely an issue with network access to the huggingface repository')
            print('')
            print(f'Error:  {e}')
            
