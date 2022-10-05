#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 12:49:08 2021

@author: scoobydoo
"""

from dataset import Dataset
from sampling import Batch,Sampler
import tensorflow as tf
import copy
import json
from model import Model
from helper_fn import evaluateTarget
import numpy as np
import argparse
import pandas as pd
import os
from baselines.helper_fn import warm_start,regret
# set random seeds
tf.random.set_seed(0)
rng = np.random.RandomState(42)
# create parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--meta_split', help='Select training fold', type=int,default=0)
parser.add_argument('--meta_fold', help='Select meta-fold to evaluate on ', choices=['meta-test','meta-valid'],type=str,default='meta-test')
parser.add_argument('--targetdataset', help='index', type=int,default=0)
parser.add_argument('--metadataset', help='Select metadataset',choices=['searchspace-b','searchspace-a','searchspace-c'], type=str,default='searchspace-a')
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--offline_metafeatures', help='Metafeatures used', choices=['mf1','mf2','d2v',"None"])
parser.add_argument('--alpha_bdi', help='batch identification task hyperparameter',type=float,default=0.1)
parser.add_argument('--init', help='Metafeatures used', choices=['aaai','tstr','d2v','random','transferable'],default='random')
parser.add_argument('--alpha_mr', help='regularization task hyperparameter',type=float,default=10)
parser.add_argument('--seed', help='Select seed', type=int)
parser.add_argument('--k', help='number of starting points', type=int,default=3)

args        = parser.parse_args()

args.offline_metafeatures = None if args.offline_metafeatures=="None" else args.offline_metafeatures
rootdir       = os.path.dirname(os.path.realpath(__file__))

directory     = os.path.join(rootdir,f"checkpoints{'-v2' if args.metadataset !='searchspace-b' else ''}",f"{args.metadataset}",f"split-{args.meta_split}","metalearn",
                         f"{'dmfbs' if not args.offline_metafeatures else 'MFBS({})'.format(args.offline_metafeatures)}",f"alpha_mr-{args.alpha_mr}-alpha_bdi-{args.alpha_bdi}")

configuration = json.load(open(os.path.join(directory,"configuration.txt"),'r'))
assert(configuration["alpha_bdi"]==args.alpha_bdi and configuration["alpha_mr"]==args.alpha_mr)
# load training sets
transformation = {"searchspace-a":"Layout Md",
               "searchspace-b":"Regularization Md",
               "searchspace-c":"Optimization Md"}

training_dataset_ids = pd.read_csv(os.path.join(rootdir, "dataset_id_splits.csv"), index_col=0)[f"train-{args.meta_split}"].dropna().astype(int).ravel()
test_dataset_ids = [pd.read_csv(os.path.join(rootdir, "dataset_id_splits.csv"), index_col=0)[f"test-{args.meta_split}"].dropna().astype(int).ravel()[args.targetdataset]]
metafeatures_df = {"d2v":pd.read_csv(os.path.join(rootdir, "metafeatures", f"d2v-{args.meta_split}.csv"), index_col=0),
                   "mf1":pd.read_csv(os.path.join(rootdir, "metafeatures", "mf1.csv"), index_col=0),
                   "mf2":pd.read_csv(os.path.join(rootdir, "metafeatures", "mf2.csv"), index_col=0), }

# create Dataset
# create Dataset
loaded = False
while not loaded:
    try:
        metatrain_dataset         = Dataset(dataset_ids=training_dataset_ids,transformation=transformation[args.metadataset],metafeatures_df=metafeatures_df,configuration=configuration)
        metatest_dataset         = Dataset(dataset_ids=test_dataset_ids,transformation=transformation[args.metadataset],metafeatures_df=metafeatures_df,configuration=configuration)
        loaded = True
    except Exception:
        pass

model     = Model(configuration,rootdir=rootdir,for_eval=True)

validation_perf = pd.read_csv(os.path.join(directory,"validation-error.csv"),header=0,index_col=0).mean(axis=1).ravel()
load_iter = np.argmin(validation_perf)
model.directory = os.path.join(directory,args.meta_fold,f"targetdataset-{args.targetdataset}")

os.makedirs(model.directory,exist_ok=True)

test_sampler     = Sampler(dataset=metatest_dataset,source_dataset=metatrain_dataset,source_equal_query=False,use_mf=configuration["offline-metafeatures"])

testconfiguration = copy.deepcopy(configuration)
testconfiguration['batch_size'] = metatest_dataset.cardinality

testmodel   = Model(testconfiguration,rootdir=rootdir,for_eval=True)
testbatch   = Batch(testconfiguration['batch_size'])
testmodel.set_weights(model.get_weights())
file = pd.read_csv(os.path.join(rootdir, "dataset_name_splits.csv"), index_col=0)[f"test-{args.meta_split}"].dropna().ravel()[args.targetdataset]
if args.seed is None:
    # NOTE TARGETDATASET IS SET TO 0 because the metatest_dataset has only one task !!!
    initialResponse = evaluateTarget(testmodel, testbatch,test_sampler,targetdataset=0,hyperparameter_ids=np.arange(metatest_dataset.cardinality))    
    initialResponse     = np.argsort(np.asarray(initialResponse["targetresponse"]))[::-1] #large to small
    x   = initialResponse[:args.k].tolist()    
else:
    random = np.random.RandomState(args.seed)
    sourcefiles= pd.read_csv(os.path.join(rootdir, "dataset_name_splits.csv"), index_col=0)[f"train-{args.meta_split}"].dropna().ravel().tolist()
    x     = warm_start(method=args.init,dataset=metatest_dataset,k=args.k,file=file,sourcefiles=sourcefiles,metafeatures=metafeatures_df,fold="test",generator=random).tolist()

targetdataset = 0
response             =  metatest_dataset.lambdas[targetdataset][:,0]

y             = [response[_] for _ in x]

batchlist = [Batch(configuration['batch_size']) for _ in range(3)]

optimizer        = tf.keras.optimizers.Adam(configuration['learning_rate'])

training_datasets = np.arange(len(training_dataset_ids))

rng.shuffle(training_datasets)
import tqdm
bar = tqdm.tqdm(range(args.n_iters))
for iters in bar:
    sol = [response[_] for _ in x]
    if max(response) in sol:
        break
    error    = np.inf
    epochs = 0
    patience = 0
    epochsBar = tqdm.tqdm(range(500))
    model.model.load_weights(os.path.join(directory,f"iteration-{load_iter}","weights","weights"), by_name=False, skip_mismatch=False)
    for epochs in epochsBar:
        
        for sourcedataset in training_datasets.reshape(-1,16):
            
            batchlist = test_sampler.sample(batchlist,targetdataset=targetdataset,sourcedataset=sourcedataset,hyperparameters=x)
        
            [batch.collect() for batch in batchlist]
        
            metrics = model.train_step(optimizer=optimizer,batchlist=batchlist)
            
        testmodel.set_weights(model.model.get_weights())
        
        testresults = evaluateTarget(testmodel, testbatch,test_sampler,targetdataset=0,hyperparameter_ids=np.arange(metatest_dataset.cardinality))
        
        currenterror = np.mean((testresults["targetresponse"].loc[x]-testresults["response"].loc[x]).ravel()**2)
        if error >= currenterror:
            weights = copy.deepcopy(model.model.get_weights())
            error = currenterror        
        if np.allclose(currenterror,0,atol=1e-2):
            break
    
    testmodel.set_weights(weights)
    
    testresults = evaluateTarget(testmodel, testbatch,test_sampler,targetdataset=0,hyperparameter_ids=np.arange(metatest_dataset.cardinality))
    
    candidates          = [_ for _ in np.argsort(testresults['targetresponse'].ravel())[::-1] if _ not in x]
    
    x += [candidates[0]]    
            
    y = [response[_] for _ in x]
    
    rng.shuffle(training_datasets)
    
y = [response[_] for _ in x]

results            = regret(y,response)
results['indices'] = np.asarray(x).reshape(-1,)

savedir     = os.path.join(rootdir,"results",f"init-{args.k}",f"seed-{args.seed}",f"{'dmfbs' if not args.offline_metafeatures else 'MFBS({})'.format(args.offline_metafeatures)}-alpha_mr-{args.alpha_mr}-alpha_bdi-{args.alpha_bdi}",args.metadataset,f"split-{args.meta_split}","test")
os.makedirs(savedir,exist_ok=True)
results.to_csv(os.path.join(savedir,f"{file}.csv"))