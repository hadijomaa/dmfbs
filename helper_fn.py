#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 00:20:41 2020

@author: scoobydoo
"""

import numpy as np

# helper functions

def distance_to_minimum(results,targetdataset,dataset,approximations=None):

    if not approximations:
        # this is only during meta-learning the initial parameters
        # during fine-tuning to the target dataset, the approximations
        # are provided from the sequential process
        approximations = np.argsort(np.asarray(results['targetresponse']))[::-1]
    
    trueresponse = dataset.lambdas[targetdataset][:,0]
    opt = max(trueresponse)
    # evaluate selected predictions
    regret = [opt - trueresponse[_] for _ in approximations]
    incumbent = regret[0]
    output = []
    for _ in regret:
        # check current vs incumbent
        if _<incumbent:
            # assign incumbent to current
            incumbent = _
        # append incumbent to results
        output.append(incumbent)
    # return results
    return output

def evaluateTarget(model,batch,sampler,targetdataset,hyperparameter_ids):
    
    batch = sampler.sample([batch],
                           targetdataset=targetdataset,hyperparameters=hyperparameter_ids)[0]
    
    batch.collect()
    
    targety = model.model(batch.input)
    
    summary = pd.DataFrame()
    
    summary["response"]       = batch.output["response"].numpy()
    summary["targetresponse"] = targety["targetresponse"].numpy()
    summary["error"] = (summary["targetresponse"]-summary["response"])**2
    return summary
    
import pandas as pd
def evaluate(model,batch,sampler,dataset):
    
    results = []
    
    for targetdataset in range(len(dataset.data)):
        
        summary=evaluateTarget(model,batch,sampler,targetdataset,hyperparameter_ids = range(dataset.lambdas[targetdataset].shape[0]))
        
        results.append(summary)
        
    mse     = [_["error"].mean() for _ in results]
    d2m     = [distance_to_minimum(_,targetdataset,dataset=dataset) for targetdataset,_ in enumerate(results)]
    return np.vstack(d2m).mean(axis=0),np.stack(mse)