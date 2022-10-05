#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:19:28 2020

@author: scoobydoo
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from ismlldataset.datasets import get_dataset
import pickle as pkl
from utils import _map
import os
random = np.random.RandomState(92)

def ptp(X):
    param_nos_to_extract = X.shape[1]
    domain = np.zeros((param_nos_to_extract, 2))
    for i in range(param_nos_to_extract):
        domain[i, 0] = np.min(X[:, i])
        domain[i, 1] = np.max(X[:, i])
    X = (X - domain[:, 0]) / np.ptp(domain, axis=1)
    return X

def flatten(x,y):
    '''
    Genearte x_i,y_j for all i,j \in |x|

    Parameters
    ----------
    x : numpy.array
        predictors; shape = (n,m)
    y : numpy.array
        targets; shape = (n,t)

    Returns
    -------
    numpy.array()
        shape ((n\times m\times t)\times 2).

    '''
    x_stack = []
    for c in range(y.shape[1]):
        c_label = np.tile(y[:,c],reps=[x.shape[1]]).transpose().reshape(-1,1)
        x_stack.append(np.concatenate([x.transpose().reshape(-1,1),c_label],axis=-1))
    return np.vstack(x_stack)

class Dataset(object):
    
    def __init__(self,dataset_ids,transformation,metafeatures_df, configuration):
        # batch properties
        self.N       = configuration['N']
        self.C       = configuration['C']
        self.M       = configuration['M']
        self.offline_metafeatures            = configuration["offline-metafeatures"]
        
        # data 
        self.data = []
        # labels 
        self.labels = []
        # hyperparameter settings and response
        self.lambdas = []
        # metafeatures
        self.D2V = []         
        self.MF2 = []
        self.MF1 = []
        # hyperparameter space
        self.Lambda    = None
        rootdir     = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(rootdir, "pickled", configuration["metadataset"] + ".pkl"), "rb") as f:
                hpo_data = pkl.load(f)        
        for dataset_id in dataset_ids:

            dataset = get_dataset(dataset_id=dataset_id)
            data,labels = dataset.get_folds(split=0,return_valid=True)
            train_x,valid_x,test_x = data
            train_y,valid_y,test_y = labels
            
            self.data.append({"train":train_x,"valid":valid_x,"test":test_x})
            self.labels.append({"train":train_y,"valid":valid_y,"test":test_y})
            
            # metadataset = get_metadataset(dataset_id)
            # metadataset.apply_special_transformation(transformation)
            # metadataset.normalize_response()
            # hyperparameters,response = metadataset.get_meta_data()
            # hyperparameters = hyperparameters.drop_duplicates()
            # response = response.loc[hyperparameters.index.tolist()]
            # hyperparameters = np.array(pd.get_dummies(pd.DataFrame(hyperparameters)))
            # response = np.array(response).reshape(-1,1)
            
            self.D2V.append(metafeatures_df["d2v"].loc[_map[dataset_id]].ravel())
            self.MF1.append(metafeatures_df["mf1"].loc[_map[dataset_id]].ravel())
            self.MF2.append(metafeatures_df["mf2"].loc[_map[dataset_id]].ravel())
            # normalize hyperparameters
            self.Lambda = hpo_data[_map[dataset_id]]["X"] if self.Lambda is None else self.Lambda#ptp(hyperparameters) if self.Lambda is None else self.Lambda
            response = hpo_data[_map[dataset_id]]["Y"].reshape(-1,1)
            # join hyperparmater and response
            task =  np.concatenate([response,self.Lambda],axis=1)     
            self.lambdas.append(task)
        self.cardinality,self.D    = self.Lambda.shape
    def sample_batch(self,data,labels,N,C,M):
        '''
        Sample a batch from the dataset of size (N,M)
        and a corresponding label of shape (N,C).

        Parameters
        ----------
        data : numpy.array
            dataset; shape (N,F) with N >= nisntanc and F >= M
        labels : numpy.array
            categorical labels; shape (N,) with N >= nisntanc
        N : int
            Number of instances in the output batch.
        C : int
            Number of classes in the output label.
        M : int
            Number of features in the output batch.

        Returns
        -------
        data : numpy.array
            subset of the original dataset.
        labels : numpy.array
            one-hot encoded label representation of the classes in the subset

        '''
        # Create the one-hot encoder
        ohc           = OneHotEncoder(categories = [range(len(np.unique(labels)))],sparse=False)
        d = {ni: indi for indi, ni in enumerate(np.unique(labels))}
        # process the labels
        labels        = np.asarray([d[ni] for ni in labels.reshape(-1)]).reshape(-1)
        # transform the labels to one-hot encoding
        labels        = ohc.fit_transform(labels.reshape(-1,1))
            
        # Ne should be less than or equal to the dataset size
        N            = random.choice(np.arange(0,data.shape[0]),size=np.minimum(N,data.shape[0]),replace=False)
        # M should be less than or equal to the dataset size
        M         = random.choice(np.arange(0,data.shape[1]),size=np.minimum(M,data.shape[1]),replace=False)
        # C should be less than or equal to the total number of labels
        C         = random.choice(np.arange(0,labels.shape[1]),size=np.minimum(C,labels.shape[1]),replace=False)
        # extract data at selected instances
        data          = data[N]
        # extract labels at selected instances
        labels        = labels[N]
        # extract selected features from the data
        data          = data[:,M]
        # extract selected labels from the data
        labels        = labels[:,C]
        return data,labels

    def _instance(self,targetdataset,config,fold,**kwags):
        data    = self.data
        # select labels list
        labels  = self.labels
        # select surrogate list
        lambdas = self.lambdas
        # sample batch from the train-split of the pos data
        x,y = self.sample_batch(data[targetdataset][fold],labels[targetdataset][fold],**kwags)        
        # get surrogate value of surr task of the positive dataset
        l = lambdas[targetdataset][config][0]
        return x,y,l
    
    def single(self,target,config,fold,N=None,C=None,M=None):

        # check if Ne is provided
        N = N if N is not None else self.N
        # check if Ne is provided
        C = C if C is not None else self.C
        # check if Ne is provided
        M = M if M is not None else self.M        
        # format 
        instance_x,instance_i = [],[]

        x,y,l = self._instance(targetdataset=target,
                                           config=config,fold=fold,
                                           N=N,C=C,M=M)

        instance_i.append(x.shape+(y.shape[1],)+(target,))
        instance_x.append(flatten(x,y))
        # remove x,y
        del x,y
        # stack x values
        instance = {}
        instance["x"] = np.vstack(instance_x)
        # stack N
        instance["N"] = np.vstack(instance_i)[:,0][:,None]
        # stack Ms
        instance["M"] = np.vstack(instance_i)[:,1][:,None]
        # stack C
        instance["C"] = np.vstack(instance_i)[:,2][:,None]
        # get task description of surr task
        instance["lambda"] = self.Lambda[config]
        instance["l"] = l
        return instance

    def single_meta_features(self,target,config,fold,N=None,C=None,M=None):

        # check if Ne is provided
        N = N if N is not None else self.N
        # check if Ne is provided
        C = C if C is not None else self.C
        # check if Ne is provided
        M = M if M is not None else self.M        
        
        if self.offline_metafeatures == "d2v":
            mfe = self.D2V
        elif self.offline_metafeatures=="mf2":
            mfe = self.MF2
        elif self.offline_metafeatures=="mf1":
            mfe = self.MF1
        
        instance_x,instance_i = [],[]

        x,y,l = self._instance(targetdataset=target,
                                           config=config,fold=fold,
                                           N=N,C=C,M=M)

        instance_i.append(x.shape+(y.shape[1],)+(target,))
        instance_x.append(mfe[target])
        # remove x,y
        del x,y
        # stack x values
        instance = {}
        instance["x"] = np.vstack(instance_x)
        # stack N
        instance["N"] = np.vstack(instance_i)[:,0][:,None]
        # stack Ms
        instance["M"] = np.vstack(instance_i)[:,1][:,None]
        # stack C
        instance["C"] = np.vstack(instance_i)[:,2][:,None]
        # get task description of surr task
        instance["lambda"] = self.Lambda[config]
        instance["l"] = l
        return instance