## Imports + seed

# Common
import random
import numpy as np 
import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import concatenate, MaxPool2D, Input, Conv2D, UpSampling2D

seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed


# Image size
SIZE = 512
INPUT_SHAPE = (SIZE, SIZE, 1)


## Conv block

def conv_block(x, kernels, kernel_size=(3, 3), strides=1, padding='same', is_relu=True, n=2):
    for i in range(1, n + 1):
        x = Conv2D(filters=kernels, kernel_size=kernel_size, padding=padding, strides=strides) (x)
        if is_relu:
            x = keras.activations.relu(x)

    return x

## UNET 3+ MODEL
# https://github.com/hamidriasat/UNet-3-Plus

def UNet3P(INPUT_SHAPE = INPUT_SHAPE, OUTPUT_CHANNELS = 1):

    filters = [16, 32, 64, 128, 256]

    input_layer = Input(INPUT_SHAPE, name = 'input')

    """ Encoder"""
    # block 1
    e1 = conv_block(input_layer, filters[0])

    # block 2
    e2 = MaxPool2D(pool_size=(2, 2)) (e1)
    e2 = conv_block(e2, filters[1])                

    # block 3
    e3 = MaxPool2D(pool_size=(2, 2)) (e2)  
    e3 = conv_block(e3, filters[2])               

    # block 4
    e4 = MaxPool2D(pool_size=(2, 2)) (e3)  
    e4 = conv_block(e4, filters[3])                 

    # block 5
    # bottleneck layer
    e5 = MaxPool2D(pool_size=(2, 2)) (e4)  
    e5 = conv_block(e5, filters[4])                 

    """ Decoder """
    cat_channels = filters[0]
    cat_blocks = len(filters)
    upsample_channels = cat_blocks * cat_channels

    """ d4 """
    e1_d4 = MaxPool2D(pool_size=(8, 8)) (e1)                        
    e1_d4 = conv_block(e1_d4, cat_channels, n=1)                       

    e2_d4 = MaxPool2D(pool_size=(4, 4)) (e2)                        
    e2_d4 = conv_block(e2_d4, cat_channels, n=1)                       

    e3_d4 = MaxPool2D(pool_size=(2, 2)) (e3)                       
    e3_d4 = conv_block(e3_d4, cat_channels, n=1)                       

    e4_d4 = conv_block(e4, cat_channels, n=1)                       

    e5_d4 = UpSampling2D(size=(2, 2), interpolation='bilinear') (e5) 
    e5_d4 = conv_block(e5_d4, cat_channels, n=1)                       

    d4 = concatenate([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4])
    d4 = conv_block(d4, upsample_channels, n=1)                        

    """ d3 """
    e1_d3 = MaxPool2D(pool_size=(4, 4)) (e1)   
    e1_d3 = conv_block(e1_d3, cat_channels, n=1)       

    e2_d3 = MaxPool2D(pool_size=(2, 2)) (e2)    
    e2_d3 = conv_block(e2_d3, cat_channels, n=1)      

    e3_d3 = conv_block(e3, cat_channels, n=1)        

    d4_d3 = UpSampling2D(size=(2, 2), interpolation='bilinear') (d4)      
    d4_d3 = conv_block(d4_d3, cat_channels, n=1)        

    e5_d3 = UpSampling2D(size=(4, 4), interpolation='bilinear') (e5)     
    e5_d3 = conv_block(e5_d3, cat_channels, n=1)       

    d3 = concatenate([e1_d3, e2_d3, e3_d3, d4_d3, e5_d3])
    d3 = conv_block(d3, upsample_channels, n=1)        

    """ d2 """
    e1_d2 = MaxPool2D(pool_size=(2, 2)) (e1)    
    e1_d2 = conv_block(e1_d2, cat_channels, n=1)       

    e2_d2 = conv_block(e2, cat_channels, n=1)           

    d3_d2 = UpSampling2D(size=(2, 2), interpolation='bilinear') (d3)    
    d3_d2 = conv_block(d3_d2, cat_channels, n=1)        

    d4_d2 = UpSampling2D(size=(4, 4), interpolation='bilinear') (d4)     
    d4_d2 = conv_block(d4_d2, cat_channels, n=1)        

    e5_d2 = UpSampling2D(size=(8, 8), interpolation='bilinear') (e5)      
    e5_d2 = conv_block(e5_d2, cat_channels, n=1)       

    d2 = concatenate([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2])
    d2 = conv_block(d2, upsample_channels, n=1)         

    """ d1 """
    e1_d1 = conv_block(e1, cat_channels, n=1)      

    d2_d1 = UpSampling2D(size=(2, 2), interpolation='bilinear') (d2)      
    d2_d1 = conv_block(d2_d1, cat_channels, n=1)        

    d3_d1 = UpSampling2D(size=(4, 4), interpolation='bilinear') (d3)      
    d3_d1 = conv_block(d3_d1, cat_channels, n=1)        

    d4_d1 = UpSampling2D(size=(8, 8), interpolation='bilinear') (d4)      
    d4_d1 = conv_block(d4_d1, cat_channels, n=1)        

    e5_d1 = UpSampling2D(size=(16, 16), interpolation='bilinear') (e5)    
    e5_d1 = conv_block(e5_d1, cat_channels, n=1)        

    d1 = concatenate([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1, ])
    d1 = conv_block(d1, upsample_channels, n=1)         

    # last layer does not have batchnorm and relu
    d = conv_block(d1, OUTPUT_CHANNELS, n=1, is_relu=False)
    
    output = keras.activations.sigmoid(d)

    return Model(inputs=input_layer, outputs=output, name='UNet3P')