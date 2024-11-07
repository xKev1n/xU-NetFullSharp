## Imports + seed

# Common
import random
import numpy as np 
import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation

seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed

# Image size
SIZE = 512
INPUT_SHAPE = (SIZE, SIZE, 1)
OUTPUT_CHANNELS = 1

## Conv block
def conv_block(x, kernels, kernel_size=(4, 4), strides=(1,1), padding='same'):
    c = Conv2D(kernels, kernel_size=kernel_size, padding=padding, strides=strides) (x)
    c = Activation("relu") (c)

    return c

def up_block(x, kernels, kernel_size=(4, 4), strides=(1,1), padding='same'):
    c = Conv2DTranspose(kernels, kernel_size=kernel_size, padding=padding, strides=strides) (x)
    c = Activation("relu") (c)

    return c
    
## KALISZ AUTOENCODER MODEL

def KaliszAE(INPUT_SHAPE = INPUT_SHAPE, OUTPUT_CHANNELS = OUTPUT_CHANNELS):

    filters = [32, 48]


    input_layer = Input(INPUT_SHAPE, name = 'input')
    
    e1 = conv_block(input_layer, filters[0])
    e2 = conv_block(e1, filters[0], strides=(2,2))
    
    e3 = conv_block(e2, filters[1], strides=(2,2))
    e4 = conv_block(e3, filters[1], strides=(2,2))
    
    ## ---------------------------------------- ##
    
    d1 = up_block(e4, filters[1], strides=(2,2))
    d2 = up_block(d1, filters[1], strides=(2,2))
    
    d3 = up_block(d2, filters[0], strides=(2,2))
    d4 = up_block(d3, filters[0])
    
    output = Conv2D(OUTPUT_CHANNELS, 1, activation='sigmoid', padding='same') (d4)
    
    model = Model(inputs=input_layer, outputs=output, name='KaliszAE')
    
    return model