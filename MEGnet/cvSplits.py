#! /usr/bin/env python

import pandas as pd
import numpy as np
import pandas as pd
import pickle
# import matplotlib.pyplot as plt
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold #https://github.com/trent-b/iterative-stratification 

'''
Example usage
./cvSplits.py -tsv '/path/to/file/file.tsv' -kfold 4 -fields '[Scanner,Site,age]'

*by default fields is set to Scanner, Site, sex, age
'''

def foolist(astring):
    alist=astring.strip('[').strip(']').split(',')
    alist = [a.strip() for a in alist if len(a)]
    print(alist)
    return alist

def main():

    # MultilabelStratifiedKFold allows to stratify data based on multiple labels
    # For MEGnet, use input fields by user of the default ['Scanner','Site','sex','age'] as labels 
    
    mskf = MultilabelStratifiedKFold(n_splits=kfolds,shuffle=True, random_state=0)

    # MultilabelStratifiedKFold requires a numpy array as input to make splits
    # --> categorical variables will be number-coded
    # --> continuous variables will be split in n groups [n=4] @@to change
    
    # 1 | make dictionary that includes all field-id codes
    
    codeDict = {}
    count = -1
    if targetCol_con:
        cont_Cat = np.zeros([data.shape[0], len(targetCol_con)])

    for col in data.columns:

        if col in targetCol_cat: 
            
            tempDict = {}
            id = -1  
            for l in data.loc[:,col].unique().tolist():
                id +=1
                tempDict[l] = id
            codeDict[col] = tempDict

        elif col in targetCol_con:
            count +=1
            cont_ = data.loc[:,col].values

            pcnt = np.percentile(cont_,np.array([25,50,75])) # @@4 groups
            pcnt = np.insert(pcnt,0,cont_.min())
            pcnt = np.insert(pcnt,len(pcnt),cont_.max())

            

            tmp = {}
            for p in np.arange(len(pcnt)-1):
                #print(pcnt[p])
                cont_Cat[np.where(cont_ > pcnt[p])[0],count] = p
                tmp[str(int(pcnt[p]))+'-'+str(int(pcnt[p+1]))] = p

            codeDict[col] = tmp

    
    print('code dictionary')
    for key, value in codeDict.items():
        print(key, value)


    # 2 | make numpy array for selected fields
    
    # 2.1 | Targets

    targetCols = targetCol_cat + targetCol_con
    y = np.zeros([data.shape[0], len(targetCols)]) # ['Scanner','Site','sex','age']
    count = -1
    count2 = -1
    for col in targetCols:
        count +=1
        if col in targetCol_cat:       
            for key in codeDict[col].keys():
                ind = np.where((data.loc[:,col]==key)==True)[0]
                y[ind,count] = codeDict[col][key]
        else: # continous
            count2 +=1
            y[:,count] = cont_Cat[:,count2]
            
    y = y.astype(int)

    # 2.2 | Model
    X = data.idx.values

    # 3 | The folds are made by preserving the percentage of samples for each label.
        # Check whether that's the case

    check = {}
    count =-1
    for train_index, test_index in mskf.split(X=X, y=y):
        count +=1
        temp = {}
        temp['train_indx'] = train_index
        temp['test_indx'] = test_index

        check[count] = temp

    # print fold partitions
    print('\n---')
    print('check if fold partitions have equal distributions across selected categories')
    for kf in np.arange(kfolds):
        print('FOLD ' + str(kf) + '\n')

        for col in data.columns:
            
            df_ = data.iloc[check[kf]['train_indx']] # subset of data frame with training indices
            
            if col in targetCols: #['Scanner','Site','sex','age']
                print('-' + col)
                cT = []
                cS = []
                for k,v in data.groupby(col): # loop over unique fields per column
                    cTotal = v.shape[0]
                    cT.append(cTotal) # get total number of examples  
                
                for k,v in df_.groupby(col):
                    cSplit = v.shape[0]
                    cS.append(cSplit) # get number of examples in current split 
                
                ii = -1
                for k in codeDict[col].keys():
                    ii +=1
                    print('   ['+ k + '] Training (percent of data):',
                            100*(np.array(cS[ii])/np.array(cT[ii])))

            df_ = data.iloc[check[kf]['test_indx']] # subset of original dataframe with test data indices
            if col in targetCols:
                cT = []
                cS = []
                for k,v in data.groupby(col):
                    cTotal = v.shape[0]
                    cT.append(cTotal)
                
                for k,v in df_.groupby(col):
                    cSplit = v.shape[0]
                    cS.append(cSplit)
                
                ii = -1
                for k in codeDict[col].keys():
                    ii +=1
                    print('   ['+ k + '] Testing (percent of data):',
                            100*(np.array(cS[ii])/np.array(cT[ii])))
                
                
        print('\n')     

    # save check as pickle
    print('saving results to ' + output_path+output_name)
    f = open(output_path+output_name,"wb")# create a binary pickle file 
    pickle.dump(check,f)# write pickle file
    f.close()
    


if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-tsv', help='''Set full path (including filename) for tsv file''', required = True)
    parser.add_argument('-kfold', type = int, help='''define number of folds for cross validation [kfold=5]''', required=False,
                        default=5)
    parser.add_argument('-fields', help='''List of columns in tsv file to run stratification of cross-validated folds ['Scanner','Site','sex','age']''',
                        required=False, default=['Scanner','Site','sex','age'],
                        type=foolist,nargs='*')

    args = parser.parse_args()
    tsv = args.tsv
    kfolds = args.kfold
    fields = args.fields[0]
    print('fields', fields)

    print('---')
    print('reading tsv file...\n')
    data = pd.read_csv(tsv,sep='\t')
    print('tsv column names: \n', data.columns.values.tolist())
    print('\n---')
    print('checking category of selected fields...')

    catFields = data.select_dtypes(exclude='number').columns
    contFields = data.select_dtypes(include='number').columns
    
    targetCol_cat = []
    targetCol_con = []

    for field in fields:
        if field in catFields:
            print(field +' is categorical')
            targetCol_cat.append(field)
        elif field in contFields:
            print(field + ' is continuous')
            targetCol_con.append(field)

    output_path = tsv[:-len(args.tsv.split('/')[-1])]
    output_name = 'cv_' + str(kfolds) + '_FoldInd.pkl'

    print('\n---')
    main()


