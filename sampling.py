#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:21:42 2020

@author: scoobydoo
"""

import tensorflow as tf
import numpy as np
random = np.random.RandomState(3718)
tf.random.set_seed(0)
def pool(n,ntotal,shuffle):
    '''
    A function that eliminates an element from a range(ntotal)
    and returns the output shuffled. 
    '''
    _pool = [_ for _ in list(range(ntotal)) if _!= n]
    if shuffle:
        random.shuffle(_pool)
    return _pool

class Batch(object):
    
    def __init__(self,batch_size):
        
        self.batch_size = batch_size
        
        self.clear()
    
    def clear(self):
        '''
        Reference Equation 10. 
        
        x --> hierarchical dataset representation [None,2]
        N --> number of instance in the dataset
        M --> number of features in the dataset
        C --> number of classes  in the dataset
        
        l --> true hyperparameter response
        lmbda --> hyperparameter setting
        
        Returns
        -------
        None.

        '''
        # from the paper: , input to e_1. input shape [None,2]
        self.x = []
        # number of instances per item in triplets
        self.N = []
        # number of features per item in triplets
        self.M = []
        # number of classes per item in triplets
        self.C = []
        # surrogate value of the postive item in the triplets
        self.l = []
        # surrogate task 
        self.lmbda = []     
        # model input
        self.input = None
        
    def append(self,instance):
        
        if len(self.x)==self.batch_size:
            
            self.clear()
            
        self.x.append(instance["x"])
        self.N.append(instance["N"])
        self.M.append(instance["M"])
        self.C.append(instance["C"])
        self.l.append(instance["l"])
        self.lmbda.append(instance["lambda"])
        
    def collect(self):
        '''
        Prepare sampled instances in a single batch for the model.

        '''
        if len(self.x)!= self.batch_size:
            raise(f'Batch formation incomplete!\n{len(self.x)}!={self.batch_size}')
            
        self.input = (tf.concat(self.x,axis=0),
                      tf.cast(tf.transpose(tf.concat(self.C,axis=0)),dtype=tf.int32),
                      tf.cast(tf.transpose(tf.concat(self.M,axis=0)),dtype=tf.int32),
                      tf.cast(tf.transpose(tf.concat(self.N,axis=0)),dtype=tf.int32),
                      tf.stack(self.lmbda),
                      )
        self.output = {'response'  :tf.cast(tf.concat([self.l],axis=0),dtype=tf.float32),
                       'similarity':tf.concat([tf.ones(self.batch_size),tf.zeros(self.batch_size)],axis=0),
                       }
        
class Sampler(object):
    def __init__(self,dataset,source_dataset,source_equal_query,use_mf = False):
        
        self.query_dataset          = dataset
        self.source_dataset          = source_dataset
        self.source_equal_query = source_equal_query
        if use_mf:
            self.get_instance = self.query_dataset.single_meta_features
            self.get_source_instance = self.source_dataset.single_meta_features
        else:
            self.get_instance = self.query_dataset.single
            self.get_source_instance = self.source_dataset.single
        
    def sample(self,batchlist,targetdataset=None,hyperparameters = None,sourcedataset=None):

        batch_size = batchlist[0].batch_size
        
        # if the target dataset not provided
        if targetdataset is None:
            # sample randomly
            ntarget  = len(self.query_dataset.data)
            
            targetdataset = random.choice(ntarget)
            
            targetdataset = [targetdataset]*batch_size

        elif type(targetdataset)==int or type(targetdataset)==np.int64:
            
            targetdataset = [targetdataset]*batch_size
            
        else:
            
            assert(len(targetdataset)==batch_size)
            
        # if hyperparameter are not provided
        if hyperparameters is None:
            # sample randomly
            hyperparameters = random.choice(np.arange(self.query_dataset.cardinality),size=batch_size,replace=False)
        else:
            # sample from available hyperparameters
            if type(hyperparameters) is not int and len(hyperparameters)==batch_size:
                pass
            else:
                hyperparameters = random.choice(hyperparameters,size=batch_size,replace=True if len(hyperparameters)<batch_size else False)
            
        # clear batch
        [batch.clear()  for batch in batchlist]
        
        # if the complimentary datasets not provided
        if sourcedataset is None:
            # sample randomly from the source split
            nsource  = len(self.source_dataset.data)            
            swimmingpool  = pool(targetdataset,nsource,shuffle=True) if self.source_equal_query else pool(-1,nsource,shuffle=True)
            sourcedataset = random.choice(swimmingpool,batch_size,replace=True if batch_size > nsource else False)
            
        elif type(sourcedataset)==int or type(sourcedataset)==np.int64:
            
            sourcedataset = [sourcedataset]*batch_size
            
        else:
            
            assert(len(sourcedataset)==batch_size)

        for source,hyperparameter,target in zip(sourcedataset,hyperparameters,targetdataset):
            # we sample from the training partition of the target dataset
            instance = self.get_instance(target,hyperparameter,fold="train")
            batchlist[0].append(instance)
            if len(batchlist)==3:
                # we sample from the validation partition of the target dataset
                # a batch to act as the similar dataset to tthe target dataset (s_{t,q}=1)
                instance = self.get_instance(target,hyperparameter,fold="valid")
                batchlist[1].append(instance)
                # we sample from the training partition of the source dataset
                # a batch to act as the dissimilar dataset to tthe source dataset (s_{t,q}=0)
                instance = self.get_source_instance(source,hyperparameter,fold="train")
                batchlist[2].append(instance)
            
        return batchlist