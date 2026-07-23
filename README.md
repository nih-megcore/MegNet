# MEGNET
[![megnet-tests](https://github.com/nih-megcore/MegNet/actions/workflows/megnet-actions.yml/badge.svg)](https://github.com/nih-megcore/MegNet/actions/workflows/megnet-actions.yml)

This repository is a fork of the code listed below in the original code reference.  This repository adds an automated processing wrapper and python package installation around the original codebase.  The current codebase utilizes mne python to preprocess the data, generate the infomax ICA components (n=20), circular topography maps, and timeseries outputs.  The architecture of neural net has been preserved, however, the weights have been reset to uniform distribution and retrained using repository data from MEGIN, CTF, 4D, and KIT systems.

## Install (pytorch based version) 
Tested compatability: (mne 1.10/1.11/1.12 & python 3.11/3.12/3.13/3.14)
```
conda create -n megnet 'mne>=1.10' 'python>3.10'
conda activate megnet
pip install MEGnet-neuro
megnet_init  #Download model weights from hugging face
```

## Install (tensorflow based version) 
Tested compatability: (mne 1.10/1.11/1.12  & python 3.10/3.11/3.12/3.13) (py restricted by tensorflow builds)
```
conda create -n megnet 'mne>=1.10' 'python<3.14'
conda activate megnet
pip install 'MEGnet-neuro-tf'
megnet_init #Download model weights from hugging face
```

## Original Code Repository
https://github.com/DeepLearningForPrecisionHealthLab/MegNET_2020 <br>
Manuscript available: https://pubmed.ncbi.nlm.nih.gov/34274419/ <br>
DOI: https://doi.org/10.1016/j.neuroimage.2021.118402 <br>
PMID: 34274419 <br>

