import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.impute import SimpleImputer as imputer
import glob

def stack_list(theList):
    result = []
    
    for i in theList:
        for row in i.values:
            result.append(row)
            
            
    imp = imputer(missing_values = np.nan, strategy = 'mean')

    
    return imp.fit_transform(np.array(result))

def stack_labels(theList):
    labels = []
    
    for i in theList:
        for row in i.values:
            labels.append(row)
    
    return np.array(labels)

def return_shapes(theList):
    result = []
    
    for i in theList:
        result.append(i.shape[0])
        
    return np.array(result)


def return_to_original(data, original):
    result = []
    
    for i in range(original.shape[0]):
        temp = []
        for j in range(original[i]):
            temp.append(data[j])
        result.append(np.array(temp))
    
    return result

def split_probs(probs):
    result = []
    
    for i in probs:
        result.append(i[:,0])
        
    return result
    
    
    
def combine_label_and_data(data, labels):
    result = []
    
    for i,f in enumerate(data):
        temp = labels[i].reshape(labels[i].shape[0],1)
        result.append(np.append(f,temp, axis=1))
        
    return result
        
        
        