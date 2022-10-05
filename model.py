#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:58:58 2020

@author: scoobydoo
"""

import tensorflow as tf
import json
import os
from modules import FunctionE1,FunctionE3,FunctionE2,PoolE1,PoolE2,FunctionR2,FunctionR1
tf.random.set_seed(452)


class Model(object):
    '''
    Model
    '''
    def __init__(self,configuration,rootdir,for_eval=False):

        # data shape
        self.batch_size    = configuration['batch_size']
        self.meta_split         = configuration['meta-split']
        self.metadataset   = configuration['metadataset']
        self.D             = configuration['D']

        self.nonlinearity_d2v  = configuration['nonlinearity_d2v']
        self.nonlinearity_rx   = configuration['nonlinearity_rx']
        # Function F
        self.units_f        = configuration['units_f']
        self.nhidden_f      = configuration['nhidden_f']
        self.architecture_f = configuration['architecture_f']
        self.resblocks_f    = configuration['resblocks_f']

        # Function G
        self.units_g        = configuration['units_g']
        self.nhidden_g      = configuration['nhidden_g']
        self.architecture_g = configuration['architecture_g']
        
        # Function H
        self.units_h        = configuration['units_h']
        self.nhidden_h      = configuration['nhidden_h']
        self.architecture_h = configuration['architecture_h']
        self.resblocks_h    = configuration['resblocks_h']
        # PoolH
        self.output_dim      = configuration['output_dim']
        self.units_r2        = configuration['units_r2']
        self.nhidden_r2      = configuration['nhidden_r2']
        self.architecture_r2 = configuration['architecture_r2']
        
        # Embedding
        self.units_r1        = configuration['units_r1']
        self.nhidden_r1      = configuration['nhidden_r1']
        self.architecture_r1 = configuration['architecture_r1']

        # optimization
        self.alpha_mr  = configuration['alpha_mr']
        self.alpha_bdi = configuration['alpha_bdi']
        
        self.offline_metafeatures = configuration["offline-metafeatures"]

        if self.offline_metafeatures:
            if configuration["offline-metafeatures"]=="d2v":
                self.dim_metafeatures = 32
            elif configuration["offline-metafeatures"]=="mf1":
                # several metafeatures are zeros, so dropped leaving 41 instead
                # of 46 
                self.dim_metafeatures = 41
            elif configuration["offline-metafeatures"]=="mf2":
                self.dim_metafeatures = 22
        self.backend = None        
        self.model,self.dataset2vec = self.create_model()
        self.backend,_              = self.create_model()
        
        self.backend.set_weights(self.model.get_weights())            
        if not for_eval:
            self.directory = self._create_dir(rootdir)
            self._save_configuration(configuration)

    def _create_dir(self,rootdir):
        import datetime
        directory = os.path.join(rootdir, "checkpoints-v2",f"{self.metadataset}",f"split-{self.meta_split}","metalearn",
                                 f"{'dmfbs' if not self.offline_metafeatures else 'MFBS({})'.format(self.offline_metafeatures)}",datetime.datetime.now().strftime("train-%Y-%m-%d-%H-%M-%S-%f"))
        os.makedirs(directory)
        return directory
    
    @tf.function
    def similarity(self,layer1,layer2):
        '''

        Parameters
        ----------
        layer : tf.Tensor
            Extracted metafeatures; shape = [None,3,units_hh].
        positive_pair : bool
            indicator of similarity expected (between positive pair or negative pair).

        Returns
        -------
        tf.Tensor
            Similarity between metafeatures.

        '''
        # check if requires reshape
        return tf.exp(-self.distance(layer1,layer2))
    
    @tf.function
    def distance(self,layer1,layer2):
        '''
        Return the cosine similarity between dataset metafeatures

        Parameters
        ----------
        layer : tf.Tensor
            metafeatures.

        Returns
        -------
        cos : tf.Tensor
            Cosine similarity.

        '''
        dist = tf.norm(layer1-layer2,axis=1)
        return dist
    
    @tf.function
    def similarityloss(self,target_y,layer1,layer2,layer3):
        '''
        Compute the similarity log_loss between positive-pair metafeatures
        and negative-pair metafeatures.

        Parameters
        ----------
        target_y : tf.Tensor
            Similarity indicator.
        predicted_y : tf.Tensor
            Extracted metafeatures; shape = [None,3,units_hh].

        Returns
        -------
        tf.Tensor

        '''
        negative_prob   = self.similarity(layer1,layer3)
        
        positive_prob   = self.similarity(layer1,layer2)
        
        logits          = tf.concat([positive_prob,negative_prob],axis=0)
        # create weight
        return tf.keras.losses.BinaryCrossentropy()(y_true=target_y[:,None],y_pred=logits[:,None])
    
    @tf.function
    def reconstructionloss(self,target_y,predicted_y):
        '''
        Computes the mean squared error between the target_y and predicted_y

        Parameters
        ----------
        target_y : tf.Tensor
            Actual Surrogate Value.
        predicted_y : tf.Tensor
            Predicted Surrogate Value.
        sample_weights : bool
            apply weighted reconstruction loss
        Returns
        -------
        loss : tf.Tensor
            Mean Squared Error.

        '''
        loss          = tf.keras.losses.MeanSquaredError()(y_true=target_y[:,None],y_pred=predicted_y)
        return loss        
    
    @tf.function
    def regularizationloss(self,target_y,predicted_y,sample_weights):
        '''
        Compute the regulization loss, i.e. weighted mean squared error
        between target_y and predicted_y

        Parameters
        ----------
        target_y : tf.Tensor
            Actual Surrogate Value of (negative) dataset.
        predicted_y : tf.Tensor
            Predicted Surrogate Value.
        sample_weights : tf.Tensor
            Cosine Similarity .
        Returns
        -------
        loss : tf.Tensor

        '''
        loss = tf.keras.losses.MeanSquaredError()(y_true=target_y[:,None],y_pred=predicted_y,sample_weight=sample_weights[:,None])
        return loss
   
    @tf.function
    def loss(self,target_y,output,pos_mf,neg_mf,neg_res):
        '''
        Compute the total loss of the network.

        Parameters
        ----------
        target_y : tuple(tf.Tensor)
            (Similarty Indicator,targetresponse)
        output : tuple
            Output of the model.
        training : bool
            important to specify keys of metrics dict.
        Returns
        -------
        loss : tf.Tensor

        '''
        # add prefix
        similaritytarget = target_y['similarity']
        # create metrics placeholder
        losses  = {}
        l       = None
        # Compute Similarity Loss
        targetresponse       = output['targetresponse']
        response             = target_y['response']
        losses.update({'Dataset Batch Identification':self.similarityloss(similaritytarget,layer1=output['metafeatures']
                                                            ,layer2=pos_mf
                                                            ,layer3=neg_mf)})
        l = self.alpha_bdi*losses['Dataset Batch Identification']
        
        losses.update({'Quadratic Loss':self.reconstructionloss(target_y=response,predicted_y=targetresponse)})
        l += losses['Quadratic Loss']
        sample_weights = self.similarity(layer1=output["metafeatures"],layer2=neg_mf)
        losses.update({'Manifold Regularization':self.regularizationloss(target_y=neg_res,predicted_y=targetresponse,sample_weights=sample_weights)})
        l += self.alpha_mr*losses['Manifold Regularization']
        return l,losses
       
    def train_step(self,batchlist,optimizer,clip=True):
        with tf.GradientTape() as tape:
            output      = self.model(batchlist[0].input, training=True)
            output_pos  = self.model(batchlist[1].input)
            output_neg  = self.model(batchlist[2].input)            
            loss,metrics = self.loss(batchlist[0].output, output,output_pos["metafeatures"],output_neg["metafeatures"],output_neg["targetresponse"])
                
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # clip gradients
        if clip:
            gradients = [tf.clip_by_value(t,clip_value_min=-0.5,clip_value_max=0.5) for t in gradients]
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables ))
        return metrics

    def validate_step(self,batchlist):
        output      = self.model(batchlist[0].input)
        output_pos  = self.model(batchlist[1].input)
        output_neg  = self.model(batchlist[2].input)            
        _,metrics = self.loss(batchlist[0].output, output,output_pos["metafeatures"],output_neg["metafeatures"],output_neg["targetresponse"])
        return metrics
    
    @tf.function
    def backend_train_step(self,optimizer):
        # update the weights of the original model
        meta_grads = [tf.subtract(old, new) for old,new in zip(self.backend.trainable_variables,
                                                                self.model.trainable_variables)]
        # \phi = \phi - 1/n (\phi-W)
        # call the optimizer on the original vars based on the meta gradient
        optimizer.apply_gradients(zip(meta_grads, self.backend.trainable_variables))

        
    def _save_configuration(self,configuration):
        configuration.update({"savedir":self.directory})
        filepath = os.path.join(self.directory,"configuration.txt")
        with open(filepath, 'w') as json_file:
          json.dump(configuration, json_file)        
        
    def save_weights(self,iteration=None):
        '''
        Save weights of model with provided description
        
        Parameters
        ----------
        description: str
            name of weights to save.
        Returns
        -------
        None.

        '''
        # define filepath
        iteration = f"-{iteration}" if iteration is not None else ''
        filepath = os.path.join(self.directory,f"iteration{iteration}","weights")
        os.makedirs(filepath,exist_ok=True)
        # save internal model weights
        self.model.save_weights(filepath=os.path.join(filepath,"weights"))
    
    def set_weights(self,weights=None):
        '''
        Update the weights of the internal model with backend model
        weights or with provided weights.
        
        Parameters
        ----------
        weights : List[tf.Variable], optional
            Weights of the trainable variables. The default is None.

        '''
        # get weights
        weights = self.backend.get_weights() if weights is None else weights
        # set weights
        self.model.set_weights(weights=weights)
        
    def get_weights(self,internal=True):
        '''
        Return weights of the (internal) model

        Parameters
        ----------
        internal : bool, optional
            indicator of type of model for which we want 
            to get weights. The default is True.

        Returns
        -------
        weights : list(tf.Tensor)

        '''
        # get weights
        weights = self.model.get_weights() if internal else self.backend.get_weights()
        return weights
    
    def dataset2vecmodel(self,trainable):
        # input two dataset2vec shape = [None,2], i.e. flattened tabular batch
        x      = tf.keras.Input(shape=(2),dtype=tf.float32)
        # Number of sampled classes from triplets
        nclasses = tf.keras.Input(shape=(self.batch_size),dtype=tf.int32,batch_size=1)
        # Number of sampled features from triplets
        nfeature = tf.keras.Input(shape=(self.batch_size),dtype=tf.int32,batch_size=1)
        # Number of sampled instances from triplets
        ninstanc = tf.keras.Input(shape=(self.batch_size),dtype=tf.int32,batch_size=1)
        # Encode the predictor target relationship across all instances
        layer    = FunctionE1(units = self.units_f,nhidden = self.nhidden_f,nonlinearity = self.nonlinearity_d2v,architecture=self.architecture_f,resblocks=self.resblocks_f,trainable=trainable)(x)
        # Average over instances
        layer    = PoolE1(units=self.units_f)(layer,nclasses[0],nfeature[0],ninstanc[0])
        # Encode the interaction between features and classes across the latent space
        layer    = FunctionE2(units = self.units_g,nhidden   = self.nhidden_g,nonlinearity = self.nonlinearity_d2v,architecture = self.architecture_g,trainable=trainable)(layer)
        # Average across all instances
        layer    = PoolE2(units=self.units_g)(layer,nclasses[0],nfeature[0])
        # Extract the metafeatures
        metafeatures    = FunctionE3(units = self.units_h,nhidden   = self.nhidden_h, nonlinearity = self.nonlinearity_d2v,architecture=self.architecture_h,trainable=trainable,resblocks=self.resblocks_h)(layer)
        
        output = {'metafeatures':metafeatures}
        
        dataset2vec     = tf.keras.Model(inputs=[x,nclasses,nfeature,ninstanc], outputs=output)
        return dataset2vec
        
    def create_model(self):
        '''
        Create Model

        Returns
        -------
        tf.keras.Model.

        '''
        trainable = True
        if self.offline_metafeatures:
            assert(hasattr(self,"dim_metafeatures"))
            x      = tf.keras.Input(shape=(self.dim_metafeatures),dtype=tf.float32)
            nclasses = tf.keras.Input(shape=(self.batch_size),dtype=tf.int32,batch_size=1)
            nfeature = tf.keras.Input(shape=(self.batch_size),dtype=tf.int32,batch_size=1)
            ninstanc = tf.keras.Input(shape=(self.batch_size),dtype=tf.int32,batch_size=1)            
            output = {'metafeatures':x}
            dataset2vec     = tf.keras.Model(inputs=[x,nclasses,nfeature,ninstanc], outputs=output)
        else:
            dataset2vec = self.dataset2vecmodel(trainable)
        tasks    = tf.keras.Input(shape=(self.D),dtype=tf.float32)        
        layer         = dataset2vec.output["metafeatures"]
        embeddinghead = FunctionR1(units = self.units_r1,nhidden   = self.nhidden_r1,nonlinearity = self.nonlinearity_rx,architecture = self.architecture_r1,trainable=trainable)
        layer = embeddinghead(layer,tasks)
        rhead = FunctionR2(units = self.units_r2,nhidden   = self.nhidden_r2,nonlinearity = self.nonlinearity_rx,output_dim = self.output_dim,architecture=self.architecture_r2,trainable=True)
        targetresponse = rhead(layer)
        outputs = {'targetresponse':targetresponse}
        outputs.update(dataset2vec.output)
        return tf.keras.Model(inputs=[dataset2vec.input,tasks], outputs=outputs),dataset2vec
    
    def store_weights(self):
        self.theta = self.model.get_weights()