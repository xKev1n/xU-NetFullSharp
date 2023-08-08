## Imports + seed

# Common
import random
import numpy as np 
import tensorflow as tf

import keras
from keras.layers import concatenate, MaxPool2D, Input, Conv2D, UpSampling2D, multiply, DepthwiseConv2D, Activation
from keras.models import Model

seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed

# Image size
SIZE = 512
INPUT_SHAPE = (SIZE, SIZE, 1)
OUTPUT_CHANNELS = 1

## Custom Activation function
## xUnit: https://blog.paperspace.com/xunit-spatial-activation/
def x_unit(x, kernel_size=9):
    res = Activation("relu") (x)
    
    res = DepthwiseConv2D(kernel_size=kernel_size, padding='same') (res)
    
    res = Activation("sigmoid") (res)
    
    res = multiply([res, x])
    
    return res


## Layers
def conv_block(x, f, kernel_size=(3,3), padding = "same", strides = 1, dilation_rate = (1,1)):
    ## Convolution block with activation
    c = Conv2D(f, kernel_size=kernel_size, padding=padding, strides=strides, dilation_rate=dilation_rate, kernel_initializer='he_uniform') (x)
    c = x_unit(c)
    
    return c

def up_block(x, skip, filters, kernel_size=(3,3), padding = "same", strides = 1, dilation_rate = (1,1)):
    concat = concatenate([x, skip])
    c = conv_block(concat, filters, kernel_size=kernel_size, padding=padding, strides=strides, dilation_rate=dilation_rate)
    
    return c
    

## Blocks
def Encoder_block(x, filters, kernel_size=(3,3), padding = "same", strides = 1, level=1):                                                  ## level:  1  --->  2  --->  3
    if level==3:
        c1 = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides, dilation_rate=(2,2))                             ##  (1,1)    (1,1)    (2,2)
    else:
        c1 = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
                                                                                                                                        
    c2 = conv_block(c1, filters, kernel_size=kernel_size, padding=padding, strides=strides, dilation_rate=(2**(level-1),2**(level-1)))          ##  (1,1)    (2,2)    (4,4)
    c3 = conv_block(c2, filters, kernel_size=kernel_size, padding=padding, strides=strides, dilation_rate=(2**(level-1),2**(level-1)))          ##  (1,1)    (2,2)    (4,4)
    
    bn = conv_block(c3, filters, kernel_size=kernel_size, padding=padding, strides=strides, dilation_rate=(2**(level+1),2**(level+1)))          ##  (4,4)    (8,8)    (16,16)
    
    d3 = up_block(bn, c3, filters, kernel_size=kernel_size, padding=padding, strides=strides, dilation_rate=(2**(level-1),2**(level-1)))        ##  (1,1)    (2,2)    (4,4)
    d2 = up_block(d3, c2, filters, kernel_size=kernel_size, padding=padding, strides=strides, dilation_rate=(2**(level-1),2**(level-1)))        ##  (1,1)    (2,2)    (4,4)
    
    if level==3:
        d1 = up_block(d2, c1, filters, kernel_size=kernel_size, padding=padding, strides=strides, dilation_rate=(2,2))                          ##  (1,1)    (1,1)    (2,2)
    else:
        d1 = up_block(d2, c1, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    return d1


def xUNetFS(INPUT_SHAPE = INPUT_SHAPE, OUTPUT_CHANNELS=OUTPUT_CHANNELS):

    f = [8, 16, 32, 64, 128]

    inputs = Input(INPUT_SHAPE, name = 'input')

    conv1_1 = Encoder_block(inputs, f[0], level=1)
    ## DOWN CONNECTIONS
    pool1_1_1 = MaxPool2D((2,2), (2,2), name='maxpool1_1_1') (conv1_1)
    pool1_1_2 = MaxPool2D((4,4), (4,4), name='maxpool1_1_2') (conv1_1)
    pool1_1_3 = MaxPool2D((8,8), (8,8), name='maxpool1_1_3') (conv1_1)
    
    conv2_1 = Encoder_block(pool1_1_1, f[1], level=1)
    ## DOWN CONNECTIONS
    pool2_1_1 = MaxPool2D((2,2), (2,2), name='maxpool2_1_1') (conv2_1)
    pool2_1_2 = MaxPool2D((4,4), (4,4), name='maxpool2_1_2') (conv2_1)

    up1_2 = UpSampling2D((2,2), interpolation='bilinear', name='up12') (conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12')
    conv1_2 = conv_block(conv1_2,  f=f[0])
    ## DOWN CONNECTIONS
    pool1_2_1 = MaxPool2D((2,2), (2,2), name='maxpool1_2_1') (conv1_2)
    pool1_2_2 = MaxPool2D((4,4), (4,4), name='maxpool1_2_2') (conv1_2)
    
    conv3_1 = Encoder_block(pool2_1_1, f[2], level=2)
    ## DOWN CONNECTION
    pool3 = MaxPool2D((2,2), (2,2), name='maxpool3') (conv3_1)
    ## UP CONNECTION
    conv3_1_1_3 = UpSampling2D((4,4), interpolation='bilinear') (conv3_1)

    up2_2 = UpSampling2D((2,2), interpolation='bilinear', name='up22') (conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1, pool1_1_1], name='merge22')
    conv2_2 = conv_block(conv2_2, f=f[1])
    ## DOWN CONNECTION
    pool2_2_1 = MaxPool2D((2,2), (2,2), name='maxpool2_2_1') (conv2_2)

    up1_3 = UpSampling2D((2,2), interpolation='bilinear', name='up13') (conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2, conv3_1_1_3], name='merge13')
    conv1_3 = conv_block(conv1_3, f=f[0])
    ## DOWN CONNECTION
    pool1_3_1 = MaxPool2D((2,2), (2,2), name='maxpool1_3_1') (conv1_3)

    conv4_1 = Encoder_block(pool3, f[3], level=2)
    ## UP CONNECTIONS
    conv4_1_2_3 = UpSampling2D((4,4), interpolation='bilinear') (conv4_1)
    conv4_1_1_4 = UpSampling2D((8,8), interpolation='bilinear') (conv4_1)
    pool4 = MaxPool2D((2,2), (2,2), name='maxpool4') (conv4_1)
    
    up3_2 = UpSampling2D((2,2), interpolation='bilinear', name='up32') (conv4_1)
    conv3_2_1 = concatenate([pool1_1_2, pool2_1_1], name='SKIP3_2_1')
    conv3_2_1 = conv_block(conv3_2_1, f[2]*2)
    conv3_2 = concatenate([up3_2, conv3_1, conv3_2_1], name='merge32')
    conv3_2 = conv_block(conv3_2, f=f[2])
    ## UP CONNECTION
    conv3_2_1_4 = UpSampling2D((4,4), interpolation='bilinear') (conv3_2)

    up2_3 = UpSampling2D((2,2), interpolation='bilinear', name='up23') (conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2, conv4_1_2_3, pool1_2_1], name='merge23')
    conv2_3 = conv_block(conv2_3, f=f[1])

    up1_4 = UpSampling2D((2,2), interpolation='bilinear', name='up14') (conv2_3)
    conv1_4_1 = concatenate([conv4_1_1_4, conv3_2_1_4], name='SKIP1_4_1')
    conv1_4_1 = conv_block(conv1_4_1, f[0]*2)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3, conv1_4_1], name='merge14')
    conv1_4 = conv_block(conv1_4, f=f[0])

    conv5_1 = Encoder_block(pool4, f[4], level=3)
    ## UP CONNECTIONS
    conv5_1_3_3 = UpSampling2D((4,4), interpolation='bilinear') (conv5_1)
    conv5_1_2_4 = UpSampling2D((8,8), interpolation='bilinear') (conv5_1)
    conv5_1_1_5 = UpSampling2D((16,16), interpolation='bilinear') (conv5_1)

    up4_2 = UpSampling2D((2,2), interpolation='bilinear', name='up42') (conv5_1)
    conv4_2_1 = concatenate([pool3, pool2_1_2, pool1_1_3], name='SKIP4_2_1')
    conv4_2_1 = conv_block(conv4_2_1, f[3]*3)
    conv4_2 = concatenate([up4_2, conv4_1, conv4_2_1], name='merge42')
    conv4_2 = Encoder_block(conv4_2, f[3], level=2)
    ## UP CONNECTIONS
    conv4_2_2_4 = UpSampling2D((4,4), interpolation='bilinear') (conv4_2)
    conv4_2_1_5 = UpSampling2D((8,8), interpolation='bilinear') (conv4_2)

    up3_3 = UpSampling2D((2,2), interpolation='bilinear', name='up33') (conv4_2)
    conv3_3_1 = concatenate([pool2_2_1, pool1_2_2], name='SKIP3_3_1')
    conv3_3_1 = conv_block(conv3_3_1, f[2]*2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2, conv5_1_3_3, conv3_3_1], name='merge33')
    conv3_3 = Encoder_block(conv3_3, f[2], level=2)
    ## UP CONNECTION
    conv3_3_1_5 = UpSampling2D((4,4), interpolation='bilinear') (conv3_3)

    up2_4 = UpSampling2D((2,2), interpolation='bilinear', name='up24') (conv3_3)
    conv2_4_1 = concatenate([conv4_2_2_4, conv5_1_2_4], name='SKIP2_4_1')
    conv2_4_1 = conv_block(conv2_4_1, f[1]*2)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3, conv2_4_1, pool1_3_1], name='merge24')
    conv2_4 = Encoder_block(conv2_4, f[1], level=1)

    up1_5 = UpSampling2D((2,2), interpolation='bilinear', name='up15') (conv2_4)
    conv1_5_1 = concatenate([conv3_3_1_5, conv4_2_1_5, conv5_1_1_5], name='SKIP1_5_1')
    conv1_5_1 = conv_block(conv1_5_1, f[0]*3)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4, conv1_5_1], name='merge15')
    conv1_5 = Encoder_block(conv1_5, f[0], level=1)

    output = Conv2D(OUTPUT_CHANNELS, kernel_size=(1,1), padding = 'same', activation='sigmoid' , name='OUT') (conv1_5)
    
    model = Model(inputs=inputs, outputs=output, name='xU-NetFullSharp')
    
    return model