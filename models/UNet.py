## Imports + seed

# Common
import random
import numpy as np 
import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Concatenate, MaxPool2D, Conv2D, Input, UpSampling2D

seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed


# Image size
SIZE = 512
INPUT_SHAPE = (SIZE, SIZE, 1)


## U-NET MODEL LAYERS

def down_block(x, filters, kernel_size=(3,3), padding = "same", strides = 1):
    ## Convolution with 3x3 kernel
    c = Conv2D(filters, kernel_size, padding = padding, strides = strides, activation = "relu") (x)
    c = Conv2D(filters, kernel_size, padding = padding, strides = strides, activation = "relu") (c)
    
    ## Maxpooling by 2x2 kernel
    p = MaxPool2D((2,2), (2,2)) (c)
    
    return c, p

def up_block(x, skip, filters, kernel_size=(3,3), padding = "same", strides = 1):
    ## Upsampling
    us = UpSampling2D((2,2), interpolation='bilinear') (x)
    
    ## Concatenation with opposing encoder
    concat = Concatenate()([us, skip])
    
    ## Convolution
    c = Conv2D(filters, kernel_size, padding = padding, strides = strides, activation = "relu") (concat)
    c = Conv2D(filters, kernel_size, padding = padding, strides = strides, activation = "relu") (c)
    
    return c

## Bottleneck
def bottleneck(x, filters, kernel_size=(3,3), padding = "same", strides = 1):
    ## Convolution
    c = Conv2D(filters, kernel_size, padding = padding, strides = strides, activation = "relu") (x)
    c = Conv2D(filters, kernel_size, padding = padding, strides = strides, activation = "relu") (c)   
    
    return c

## U-NET MODEL

def UNet():

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
    u1 = up_block(bn, c4, f[3])
    u2 = up_block(u1, c3, f[2])
    u3 = up_block(u2, c2, f[1])
    u4 = up_block(u3, c1, f[0])
    
    ## Last convolution with sigmoid
    outputs = Conv2D(1, 1, padding="same", activation='sigmoid') (u4)
    
    model = Model(inputs=inputs, outputs=outputs, name='UNet')
    
    return model