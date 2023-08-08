## Imports + seed

# Common
import random
import numpy as np 
import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Concatenate, MaxPool2D, Conv2D, Input, UpSampling2D, add, multiply, BatchNormalization
import keras.backend as kb

seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed


# Image size
SIZE = 512
INPUT_SHAPE = (SIZE, SIZE, 1)


## ATTENTION U-NET MODEL LAYERS

## Attention block
# From https://arxiv.org/abs/1804.03999
def attention_block(x, gating, inter_shape):
    x_shape = kb.int_shape(x)
    
    theta_x = Conv2D(inter_shape, (1,1), strides = (2,2), padding = "same") (x)
    
    phi_g = Conv2D(inter_shape, (1,1), padding = "same") (gating)
    
    con_xg = add([phi_g, theta_x])
    
    act_xg = keras.layers.Activation("relu") (con_xg)
    
    psi = Conv2D(1, (1,1), padding = "same") (act_xg)
    
    sig_xg = keras.layers.Activation("sigmoid") (psi)
    sig_xg_shape = kb.int_shape(sig_xg)
    
    up_psi = UpSampling2D(size = (x_shape[1] // sig_xg_shape[1], x_shape[2] // sig_xg_shape[2]), interpolation='bilinear') (sig_xg)
    
    y = multiply([up_psi, x])
    
    out = Conv2D(x_shape[3], (1,1), padding = "same") (y)
    
    return out

def down_block(x, filters, kernel_size=(3,3), padding = "same", strides = 1):
    ## Convolution with 3x3 kernel
    c = Conv2D(filters, kernel_size, padding = padding, strides = strides, activation = "relu") (x)
    c = Conv2D(filters, kernel_size, padding = padding, strides = strides, activation = "relu") (c)
    
    ## Maxpooling by 2x2 kernel
    p = MaxPool2D((2,2), (2,2)) (c)
    
    return c, p

def att_up_block(x, skip, filters, kernel_size=(3,3), padding = "same", strides = 1):
    ## Upsampling
    us = UpSampling2D((2,2), interpolation='bilinear') (x)
    
    att = attention_block(skip, x, filters)
    
    ## Concatenation with attention output
    concat = Concatenate()([us, att])
    
    ## Convolution
    c = Conv2D(filters, kernel_size, padding = padding, strides = strides, activation = "relu") (concat)
    c = Conv2D(filters, kernel_size, padding = padding, strides = strides, activation = "relu") (c)
    
    return c

## Bottom of the UNet -> The lowest layer
def bottleneck(x, filters, kernel_size=(3,3), padding = "same", strides = 1):
    ## Convolution
    c = Conv2D(filters, kernel_size, padding = padding, strides = strides, activation = "relu") (x)
    c = Conv2D(filters, kernel_size, padding = padding, strides = strides, activation = "relu") (c)   
    
    return c

## ATTENTION U-NET MODEL

def Att_UNet():

    f = [16, 32, 64, 128, 256]
    
    inputs = Input(INPUT_SHAPE)
    
    p0 = inputs
    
    ## Encoder
    c1, p1 = down_block(p0, f[0])
    c2, p2 = down_block(p1, f[1])
    c3, p3 = down_block(p2, f[2])
    c4, p4 = down_block(p3, f[3])
    
    bn = bottleneck(p4, f[4])  ## Bottleneck
    
    ## Decoder
    u1 = att_up_block(bn, c4, f[3])
    u2 = att_up_block(u1, c3, f[2])
    u3 = att_up_block(u2, c2, f[1])
    u4 = att_up_block(u3, c1, f[0])
    
    ## Last convolution with sigmoid
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid") (u4)
    
    model = Model(inputs, outputs)
    
    return model