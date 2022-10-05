#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 00:02:51 2020

@author: scoobydoo
"""

import tensorflow as tf
tf.random.set_seed(452)
ARCHITECTURES = ['SQU','ASC','DES','SYM','ENC']
def get_units(idx,neurons,architecture,layers=None):
    assert architecture in ARCHITECTURES
    if architecture == 'SQU':
        return neurons
    elif architecture == 'ASC':
        return (2**idx)*neurons
    elif architecture == 'DES':
        return (2**(layers-1-idx))*neurons    
    elif architecture=='SYM':
        assert (layers is not None and layers > 2)
        if layers%2==1:
            return neurons*2**(int(layers/2) - abs(int(layers/2)-idx))
        else:
            x = int(layers/2)
            idx  = idx if idx < x else 2*x-idx -1
            return neurons*2**(int(layers/2) - abs(int(layers/2)-idx))
        
    elif architecture=='ENC':
        assert (layers is not None and layers > 2)
        if layers%2==0:
            x = int(layers/2)
            idx  = idx if idx < x else 2*x-idx -1            
            return neurons*2**(int(layers/2)-1 -idx)
        else:
            x = int(layers/2)
            idx  = idx if idx < x else 2*x-idx
            return neurons*2**(int(layers/2) -idx)

class DenseLayerwActivation(tf.keras.layers.Layer):
    
    def __init__(self, units, nonlinearity="relu"):
        
        super(DenseLayerwActivation, self).__init__()    
        self.units        = units        
        self.nonlinearity = tf.keras.layers.Activation(nonlinearity)
        self.dense        = tf.keras.layers.Dense(units=self.units)
        
    def call(self,x):
        return self.nonlinearity(self.dense(x))
    
class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, units, nhidden, nonlinearity,architecture,trainable):
        super(ResidualBlock, self).__init__()    
        self.n            = nhidden
        self.units        = units        
        self.nonlinearity = tf.keras.layers.Activation(nonlinearity)
        self.block        = [tf.keras.layers.Dense(units=get_units(_,self.units,architecture,self.n),trainable=trainable) \
                             for _ in range(self.n)]
            
    def call(self, x):
        e = x + 0
        for i, layer in enumerate(self.block):
            e = layer(e)
            if i < (self.n - 1):
                e = self.nonlinearity(e)
        return self.nonlinearity(e + x)
        

class FunctionE1(tf.keras.layers.Layer):
    
    def __init__(self, units, nhidden, nonlinearity,architecture,trainable, resblocks=0):
        # m number of residual blocks
        super(FunctionE1, self).__init__()    
        self.n            = nhidden
        self.units        = units                
        if resblocks>0:
            self.block        = [DenseLayerwActivation(units=self.units,nonlinearity=nonlinearity)]
            assert(architecture=="SQU")
            self.block       += [ResidualBlock(units=self.units,architecture=architecture,nhidden=self.n,trainable=trainable,nonlinearity=nonlinearity) \
                                 for _ in range(resblocks)]                
            self.block       += [DenseLayerwActivation(units=self.units,nonlinearity=nonlinearity)]
        else:
            self.block        = [DenseLayerwActivation(units=get_units(_,self.units,architecture,self.n),nonlinearity=nonlinearity) \
                                 for _ in range(self.n)]            
        
    def call(self, x):
        e = x
        
        for i,fc in enumerate(self.block):
            
            e = fc(e)
                
        return e

class PoolE1(tf.keras.layers.Layer):

    def __init__(self,units):    
        super(PoolE1, self).__init__()
        
        self.units = units
        
    def call(self,x,nclasses,nfeature,ninstanc):
        
        s = tf.multiply(nclasses,tf.multiply(nfeature,ninstanc))
        
        x           = tf.split(x,num_or_size_splits=s,axis=0)
        
        e  = []
        
        for i,bx in enumerate(x):
            
            te     = tf.reshape(bx,shape=(1,nclasses[i],nfeature[i],ninstanc[i],self.units))
            
            te     = tf.reduce_mean(te,axis=3)
            e.append(tf.reshape(te,shape=(nclasses[i]*nfeature[i],self.units)))
            
        e = tf.concat(e,axis=0)
        
        return e
    
class FunctionE2(tf.keras.layers.Layer):
    def __init__(self, units, nhidden, nonlinearity,architecture,trainable):
        
        # m number of residual blocks
        super(FunctionE2, self).__init__()    
        self.n            = nhidden
        self.units        = units        
        self.block        = [DenseLayerwActivation(units=get_units(_,self.units,architecture,self.n),nonlinearity=nonlinearity) \
                             for _ in range(self.n)]
            
    def call(self, x):
        e = x
        
        for fc in self.block:
            
            e = fc(e)
            
        return e

class PoolE2(tf.keras.layers.Layer):

    def __init__(self,units):    
        super(PoolE2, self).__init__()
        
        self.units = units
        
    def call(self, x,nclasses,nfeature):
        
        s = tf.multiply(nclasses, nfeature)      
        
        x = tf.split(x,num_or_size_splits=s,axis=0)
        
        e  = []
        
        for i,bx in enumerate(x):
            
            te     = tf.reshape(bx,shape=(1,nclasses[i]*nfeature[i],self.units))
            
            te     = tf.reduce_mean(te,axis=1)
            
            e.append(te)
            
        e = tf.concat(e,axis=0)

        return e
    

class FunctionE3(tf.keras.layers.Layer):
    def __init__(self, units, nhidden, nonlinearity,architecture,trainable, resblocks=0):
        super(FunctionE3, self).__init__()    
        # override function with residual blocks
        self.n            = nhidden
        self.units        = units               
        if resblocks>0:
            self.block        = [DenseLayerwActivation(units=self.units,nonlinearity=nonlinearity)]
            assert(architecture=="SQU")
            self.block       += [ResidualBlock(units=self.units,architecture=architecture,nhidden=self.n,trainable=trainable,nonlinearity=nonlinearity) \
                                 for _ in range(resblocks)]                
            self.block       += [tf.keras.layers.Dense(units=self.units)]
        else:
            units = [get_units(_,self.units,architecture,self.n) for _ in range(self.n)]
            self.block = []
            for _ in units[:-1]:
                self.block.append(DenseLayerwActivation(units=_,nonlinearity=nonlinearity))
            self.block.append(tf.keras.layers.Dense(units=units[:-1]))
        
    def call(self,x):
        
        e = x
        
        for i,fc in enumerate(self.block):
            
            # make sure activation only applied once!
            e = fc(e)

        return e

class FunctionR1(tf.keras.layers.Layer):
    """
    Joint feature hyperparameter-metafeature embeddings
    """

    def __init__(self, units, nhidden, nonlinearity,architecture,trainable):
        super(FunctionR1, self).__init__()    
        
        self.n            = nhidden
        self.units        = units        
        self.block        = [DenseLayerwActivation(units=get_units(_,self.units,architecture,self.n),nonlinearity=nonlinearity) \
                             for _ in range(self.n)]          
        # modules
        self.block2 = [DenseLayerwActivation(units=get_units(_,self.units,architecture,self.n),nonlinearity=nonlinearity) \
                             for _ in range(self.n)] 
            
    def call(self, x,t):
        ex = x
        
        for fc in self.block:
            
            ex = fc(ex)
            
        et = t
        for fc in self.block2:
            
            et = fc(et)
            
        e  = tf.keras.layers.concatenate([ex,et],axis=-1)
        return e
    
class FunctionR2(tf.keras.layers.Layer):
    """
    Prediction Network
    """

    def __init__(self, units, nhidden, nonlinearity,architecture,trainable,output_dim):
        super(FunctionR2, self).__init__()    

        self.n            = nhidden
        self.units        = units        
        self.block        = [DenseLayerwActivation(units=get_units(_,self.units,architecture,self.n),nonlinearity=nonlinearity) \
                             for _ in range(self.n)]     
            
        self.output_dim = output_dim
        self.fc_final   = tf.keras.layers.Dense(units=self.output_dim,trainable=trainable)
        
    def call(self, x):
        e = x
        # residual block
        for fc in self.block:
            
            e = fc(e)
            
        e = self.fc_final(e)
        return e
