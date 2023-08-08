## Imports + seed

# Common
import random
import numpy as np 
import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import concatenate, Input, Conv2D, BatchNormalization, UpSampling2D, Add, Activation
import keras.backend as kb

seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed


# Image size
SIZE = 512
INPUT_SHAPE = (SIZE, SIZE, 1)


## DEEP RES-U-NET MODEL LAYERS

def bn_activ(x, activ=True):
    bn = BatchNormalization() (x)
    if(activ):
        bn = Activation("relu") (bn)
    
    return bn

def conv_block(x, filters, kernel_size=(3,3), padding = "same", strides = 1):
    c = bn_activ(x)
    c = Conv2D(filters, kernel_size, padding = padding, strides = strides, kernel_initializer='he_normal') (c)
    
    return c

def res_init_block(x, filters, kernel_size=(3,3), padding = "same", strides = 1):
    c = Conv2D(filters, kernel_size, padding = padding, strides = strides, kernel_initializer='he_normal') (x)
    conv = conv_block(c, filters)

    sc = Conv2D(filters, kernel_size=(1,1), padding = padding, strides = strides, kernel_initializer='he_normal') (x)
    sc = bn_activ(sc, activ=False)
    
    out = Add() ([conv, sc])
    
    return out

def res_block(x, filters, padding = "same", strides = 2):
    conv = conv_block(x, filters, strides=strides)
    conv = conv_block(conv, filters, strides=1)
    
    sc = Conv2D(filters, kernel_size=(1,1), padding = padding, strides = strides, kernel_initializer='he_normal') (x)
    sc = bn_activ(sc, activ=False)
    
    out = Add() ([conv, sc])
    
    return out

def ups_con_block(x, skip, ch):
    up = UpSampling2D((2,2), interpolation='bilinear') (x)
    con = concatenate([up, skip])
    
    out = res_block(con, ch, strides=1)
    
    return out



## Deep Residual U-Net Model

def DeepResUNet():
    f = [16, 32, 64, 128, 256, 512]
    
    inputs = Input(INPUT_SHAPE)
    
    ## Initial block
    e0 = res_init_block(inputs, f[0])
    
    ## Encoder
    e1 = res_block(e0, f[1])
    e2 = res_block(e1, f[2])
    e3 = res_block(e2, f[3])
    e4 = res_block(e3, f[4])
    
    ## Bridge
    bn = res_block(e4, f[5])

    ## Decoder
    d0 = ups_con_block(bn, e4, f[4])
    d1 = ups_con_block(d0, e3, f[3])
    d2 = ups_con_block(d1, e2, f[2])
    d3 = ups_con_block(d2, e1, f[1])
    d4 = ups_con_block(d3, e0, f[0])
    
    ## Last convolution with sigmoid
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid") (d4)
    
    model = Model(inputs, outputs, name='DeepResUNet')
    
    return model