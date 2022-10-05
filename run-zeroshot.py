#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:03:32 2021

@author: hsjomaa
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
from baselines.helper_fn import regret
# set random seeds
tf.random.set_seed(0)
rng = np.random.RandomState(42)
# create parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--meta_split', help='Select training fold', type=int,default=0)
parser.add_argument('--meta_fold', help='Select meta-fold to evaluate on ', choices=['meta-test','meta-valid'],type=str,default='meta-test')
parser.add_argument('--metadataset', help='Select metadataset',choices=['searchspace-b','searchspace-a','searchspace-c'], type=str,default='searchspace-a')
parser.add_argument('--offline_metafeatures', help='Metafeatures used', choices=['mf1','mf2','d2v',"None"])
parser.add_argument('--alpha_bdi', help='batch identification task hyperparameter',type=float,default=0.1)
parser.add_argument('--alpha_mr', help='regularization task hyperparameter',type=float,default=10.)

args        = parser.parse_args()

args.offline_metafeatures = None if args.offline_metafeatures=="None" else args.offline_metafeatures
rootdir       = os.path.dirname(os.path.realpath(__file__))

directory     = os.path.join(rootdir,"checkpoints",f"{args.metadataset}",f"split-{args.meta_split}","metalearn",
                         f"{'dmfbs' if not args.offline_metafeatures else 'MFBS({})'.format(args.offline_metafeatures)}","alpha_mr-{int(args.alpha_mr)}-alpha_bdi-{float(args.alpha_bdi) if args.alpha_bdi==0 else args.alpha_bdi}")

configuration = json.load(open(os.path.join(directory,"configuration.txt"),'r'))
assert(configuration["alpha_bdi"]==args.alpha_bdi and configuration["alpha_mr"]==args.alpha_mr)
# load training sets
transformation = {"searchspace-a":"Layout Md",
               "searchspace-b":"Regularization Md",
               "searchspace-c":"Optimization Md"}

training_dataset_ids = pd.read_csv(os.path.join(rootdir, "dataset_id_splits.csv"), index_col=0)[f"train-{args.meta_split}"].dropna().astype(int).ravel()
test_dataset_ids = pd.read_csv(os.path.join(rootdir, "dataset_id_splits.csv"), index_col=0)[f"test-{args.meta_split}"].dropna().astype(int).ravel()
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
assert(len(validation_perf)==501)
load_iter = np.argmin(validation_perf)
model.model.load_weights(os.path.join(directory,f"iteration-{load_iter}","weights","weights"), by_name=False, skip_mismatch=False)

test_sampler     = Sampler(dataset=metatest_dataset,source_dataset=metatrain_dataset,source_equal_query=False,use_mf=configuration["offline-metafeatures"])

testconfiguration = copy.deepcopy(configuration)
testconfiguration['batch_size'] = metatest_dataset.cardinality

testmodel   = Model(testconfiguration,rootdir=rootdir,for_eval=True)
testbatch   = Batch(testconfiguration['batch_size'])
testmodel.set_weights(model.get_weights())

files = pd.read_csv(os.path.join(rootdir, "dataset_name_splits.csv"), index_col=0)[f"test-{args.meta_split}"].dropna().ravel()
savedir     = os.path.join(rootdir,"results","zero-shot",f"{'dmfbs' if not args.offline_metafeatures else 'MFBS({})'.format(args.offline_metafeatures)}-alpha_mr-{args.alpha_mr}-alpha_bdi-{args.alpha_bdi}",args.metadataset,f"split-{args.meta_split}","test")
os.makedirs(savedir,exist_ok=True)
for targetdataset,file in enumerate(files):
    initialResponse     = evaluateTarget(testmodel, testbatch,test_sampler,targetdataset=targetdataset,hyperparameter_ids=np.arange(metatest_dataset.cardinality))    
    initialResponse     = np.argsort(np.asarray(initialResponse["targetresponse"]))[::-1] #large to small
    x   = initialResponse.tolist()    
    
    response             =  metatest_dataset.lambdas[targetdataset][:,0]
    
    y             = [response[_] for _ in x]
    results            = regret(y,response)
    results['indices'] = np.asarray(x).reshape(-1,)
    
    results.to_csv(os.path.join(savedir,f"{file}.csv"))