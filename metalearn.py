#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 00:22:26 2020

@author: scoobydoo
"""

from dataset  import Dataset
from sampling import Batch,Sampler
import tensorflow as tf
import copy
import json
from model import Model
from helper_fn import evaluate
import numpy as np
import argparse
import os
import pandas as pd
import tqdm
# set random seeds
tf.random.set_seed(143)
#random.seed(42)
random = np.random.RandomState(783)
# create parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--meta_split', help='Select training fold', type=int,default=1)
parser.add_argument('--metadataset', help='Select metadataset',choices=['searchspace-b','searchspace-a','searchspace-c'], type=str,default='searchspace-a')
parser.add_argument('--learning_rate', help='Learning rate',type=float,default=1e-3)
parser.add_argument('--backend_learning_rate', help='Learning rate',type=float,default=1e-2)
parser.add_argument('--alpha_bdi', help='batch identification task hyperparameter',type=float,default=0.1)
parser.add_argument('--alpha_mr', help='regularization task hyperparameter',type=float,default=10)
parser.add_argument('--inner_steps', help='reptile-steps',type=int,default=5)
parser.add_argument('--offline_metafeatures', help='Metafeatures used', choices=['mf1','mf2','d2v',"None"])
args    = parser.parse_args()

rootdir        = os.path.dirname(os.path.realpath(__file__))
networkconfig_file     = os.path.join(rootdir, "configurations", "predictionnetwork.json")
phiconfig_file = os.path.join(rootdir, "configurations", "phi.json")
info_file       = os.path.join(rootdir, "md_info.json")
searchspaceinfo = json.load(open(info_file,'r'))

configuration = json.load(open(networkconfig_file,'r'))
configuration.update(json.load(open(phiconfig_file,'r')))
configuration.update(searchspaceinfo[args.metadataset.replace("searchspace-","")])
args.offline_metafeatures = None if args.offline_metafeatures == "None" else args.offline_metafeatures
epochs = 500

# update with shared configurations with specifics
config_specs = {
    'meta-split':	args.meta_split,
    'metadataset':	args.metadataset,
    'learning_rate':	args.learning_rate,
    'alpha_mr':	args.alpha_mr,
    'alpha_bdi':	    args.alpha_bdi,
    'inner_steps':	args.inner_steps,
    'batch_size':	16,
    'backend_learning_rate':	args.backend_learning_rate,
    'offline-metafeatures':args.offline_metafeatures,
    'log-frequency':5
    }
configuration.update(config_specs)

transformation = {"searchspace-a":"Layout Md",
               "searchspace-b":"Regularization Md",
               "searchspace-c":"Optimization Md"}

training_dataset_ids = pd.read_csv(os.path.join(rootdir, "dataset_id_splits.csv"), index_col=0)[f"train-{args.meta_split}"].dropna().astype(int).ravel()
validation_dataset_ids = pd.read_csv(os.path.join(rootdir, "dataset_id_splits.csv"), index_col=0)[f"valid-{args.meta_split}"].dropna().astype(int).ravel()
metafeatures_df = {"d2v":pd.read_csv(os.path.join(rootdir, "metafeatures", f"d2v-{args.meta_split}.csv"), index_col=0),
                   "mf1":pd.read_csv(os.path.join(rootdir, "metafeatures", "mf1.csv"), index_col=0),
                   "mf2":pd.read_csv(os.path.join(rootdir, "metafeatures", "mf2.csv"), index_col=0), }

# create Dataset
loaded = False
while not loaded:
    try:
        metatrain_dataset         = Dataset(dataset_ids=training_dataset_ids,transformation=transformation[args.metadataset],metafeatures_df=metafeatures_df,configuration=configuration)
        metavalid_dataset         = Dataset(dataset_ids=validation_dataset_ids,transformation=transformation[args.metadataset],metafeatures_df=metafeatures_df,configuration=configuration)
        loaded=True
    except Exception as e:
        print(e)
    
# load training sets
metatrain = len(metatrain_dataset.data)
metavalid = len(metavalid_dataset.data)
backendoptimizer = tf.keras.optimizers.SGD(configuration['backend_learning_rate'])

model     = Model(configuration,rootdir=rootdir)

testconfiguration = copy.deepcopy(configuration)
testconfiguration['batch_size'] = metatrain_dataset.cardinality
testmodel   = Model(testconfiguration,rootdir=rootdir,for_eval=True)
testbatch   = Batch(testconfiguration['batch_size'])

# define list/csv tracking
print(model.model.summary())

sampler     = Sampler(dataset=metatrain_dataset,source_dataset=metatrain_dataset,source_equal_query=True,use_mf=configuration["offline-metafeatures"])
validation_sampler     = Sampler(dataset=metavalid_dataset,source_dataset=metatrain_dataset,source_equal_query=False,use_mf=configuration["offline-metafeatures"])

metatrain_ids = np.arange(metatrain)
random.shuffle(metatrain_ids)

# create triple batches (target dataset, similar dataset, dissimilar dataset)
batchlist = [Batch(configuration['batch_size']) for _ in range(3)]

optimizer        = tf.keras.optimizers.Adam(configuration['learning_rate'],beta_1=0.)
metabatchId=0
validationPerformance = []
validationError = []
train_log_dir = os.path.join(model.directory,"train")
valid_log_dir = os.path.join(model.directory,"valid")
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
epoch = 0
train_curloss = np.inf
losses = None
epochsbar = tqdm.tqdm(range(epochs+1), desc='epochs',leave=True)
##### epoch ends after sampling all meta-train datasets
for epoch in epochsbar:
    #### loop over meta-train datasets 
    for taskids in tqdm.tqdm(range(5), desc='taskids',leave=True):
        
        metaBatch    = metatrain_ids[taskids*16:(taskids+1)*16]
        
        metaBatchBar = tqdm.tqdm(metaBatch, desc='metaBatch',leave=True)
        
        model.store_weights() 
        
        theta_i = []
        
        batchloss = None
        #### loop over task ids in the metabatch
        for targetdataset in metaBatchBar:
            
            model.set_weights(model.theta)    
            
            for inner_steps in range(configuration["inner_steps"]):
        
                batchlist = sampler.sample(batchlist,targetdataset=targetdataset)
                        
                [batch.collect() for batch in batchlist]
    
                metrics = model.train_step(optimizer=optimizer,batchlist=batchlist)
    
                if not batchloss:
                    batchloss = {}
                    for k,v in metrics.items():
                        batchloss[k] = [v.numpy()]
                else:
                    for k,v in metrics.items():
                        batchloss[k].append(v.numpy())
                
            theta_i.append(model.model.get_weights())
    
        theta = [tf.reduce_mean(tf.stack(_,axis=0),axis=0) for _ in zip(*theta_i)]
            
        model.set_weights(theta)
    
        model.backend_train_step(backendoptimizer)
        
        model.set_weights()

        if not losses:
            losses = {}
            for k,v in batchloss.items():
                losses[k] = [np.mean(v)]
        else:
            for k,v in batchloss.items():
                losses[k].append(np.mean(v))

    random.shuffle(metatrain_ids)
    with train_summary_writer.as_default():
        tloss = 0.
        for key,val in losses.items():
            tf.summary.scalar(f"{key}", np.mean(val), step=epoch)
            if key=="Manifold Regularization":
                tloss += args.alpha_mr*np.mean(val)
            elif key=="Dataset Batch Identification":
                tloss += args.alpha_bdi*np.mean(val)
            elif key=="Quadratic Loss":
                tloss += np.mean(val)                
        tf.summary.scalar("total", tloss, step=epoch)
######################## ADDED VALIDATION 
    validbatchloss = None
    for validationdataset in range(metavalid):
        batchlist = validation_sampler.sample(batchlist,targetdataset=validationdataset)
        [batch.collect() for batch in batchlist]    
        validmetrics = model.validate_step(batchlist)

        if not validbatchloss:
            validbatchloss = {}
            for k,v in validmetrics.items():
                validbatchloss[k] = [v.numpy()]
        else:
            for k,v in validmetrics.items():
                validbatchloss[k].append(v.numpy())
    validlosses = {}
    for k,v in validbatchloss.items():
        validlosses[k] = [np.mean(v)]

    with valid_summary_writer.as_default():
        vloss = 0.
        for key,val in validlosses.items():
            tf.summary.scalar(f"{key}", np.mean(val), step=epoch)
            if key=="Manifold Regularization":
                vloss += args.alpha_mr*np.mean(val)
            elif key=="Dataset Batch Identification":
                vloss += args.alpha_bdi*np.mean(val)
            elif key=="Quadratic Loss":
                vloss += np.mean(val)                
        tf.summary.scalar("total", vloss, step=epoch)      
########################                 
    
    losses = None
    
    model.save_weights(iteration=epoch)
    
    if abs(train_curloss-tloss)<1e-4:
        break
    # testmodel.set_weights(model.get_weights())
    
    # min20,error         = evaluate(testmodel, testbatch,validation_sampler, metavalid_dataset)
    
    # validationError.append(error)
    # validationPerformance.append(min20)        
    # pd.DataFrame(validationError).to_csv(os.path.join(model.directory,"validation-error.csv"))
    # pd.DataFrame(validationPerformance).to_csv(os.path.join(model.directory,"validation.csv"))
    # break