## U-NET# MODEL

## Imports + seed

# Common
import random
import numpy as np 
import tensorflow as tf

import keras
from keras.layers import concatenate, Activation, Conv2D, Input, MaxPool2D, UpSampling2D
from keras.models import Model

seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed

# Image size
SIZE = 512
INPUT_SHAPE = (SIZE, SIZE, 1)

# Blocks
def conv_batchnorm_relu_block(tensor, f):

    x = Conv2D(f, (3, 3), padding='same', strides=1) (tensor)
    x = Activation('relu') (x)
    x = Conv2D(f, (5, 5), padding='same') (x)
    x = Activation('relu') (x)
    x = Conv2D(f, (3, 3), padding='same', strides=1) (x)
    x = Activation('relu') (x)
    x = Conv2D(f, (5, 5), padding='same') (x)
    x = Activation('relu') (x)
    
    return x

# U-Net# Model
def UNetSharp(INPUT_SHAPE = INPUT_SHAPE):

    f = [8, 16, 32, 64, 128]

    inputs = Input(INPUT_SHAPE, name = 'input')

    conv1_1 = conv_batchnorm_relu_block(inputs, f=f[0])
    pool1 = MaxPool2D((2,2), (2,2), name='maxpool1') (conv1_1)

    conv2_1 = conv_batchnorm_relu_block(pool1, f=f[1])
    pool2 = MaxPool2D((2,2), (2,2), name='maxpool2') (conv2_1)

    up1_2 = UpSampling2D((2,2), interpolation='bilinear', name='up12') (conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12')
    conv1_2 = conv_batchnorm_relu_block(conv1_2,  f=f[0])

    conv3_1 = conv_batchnorm_relu_block(pool2, f=f[2])
    pool3 = MaxPool2D((2,2), (2,2), name='maxpool3') (conv3_1)
    conv3_1_1_3 = UpSampling2D((4,4), interpolation='bilinear') (conv3_1)

    up2_2 = UpSampling2D((2,2), interpolation='bilinear', name='up22') (conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22')
    conv2_2 = conv_batchnorm_relu_block(conv2_2, f=f[1])

    up1_3 = UpSampling2D((2,2), interpolation='bilinear', name='up13') (conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2, conv3_1_1_3], name='merge13')
    conv1_3 = conv_batchnorm_relu_block(conv1_3, f=f[0])

    conv4_1 = conv_batchnorm_relu_block(pool3, f=f[3])
    conv4_1_2_3 = UpSampling2D((4,4), interpolation='bilinear') (conv4_1)
    conv4_1_1_4 = UpSampling2D((8,8), interpolation='bilinear') (conv4_1)
    pool4 = MaxPool2D((2,2), (2,2), name='maxpool4') (conv4_1)
    
    up3_2 = UpSampling2D((2,2), interpolation='bilinear', name='up32') (conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32')
    conv3_2 = conv_batchnorm_relu_block(conv3_2, f=f[2])
    conv3_2_1_4 = UpSampling2D((4,4), interpolation='bilinear') (conv3_2)

    up2_3 = UpSampling2D((2,2), interpolation='bilinear', name='up23') (conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2, conv4_1_2_3], name='merge23')
    conv2_3 = conv_batchnorm_relu_block(conv2_3, f=f[1])

    up1_4 = UpSampling2D((2,2), interpolation='bilinear', name='up14') (conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3, conv4_1_1_4, conv3_2_1_4], name='merge14')
    conv1_4 = conv_batchnorm_relu_block(conv1_4, f=f[0])

    conv5_1 = conv_batchnorm_relu_block(pool4, f=f[4])
    conv5_1_3_3 = UpSampling2D((4,4), interpolation='bilinear') (conv5_1)
    conv5_1_2_4 = UpSampling2D((8,8), interpolation='bilinear') (conv5_1)
    conv5_1_1_5 = UpSampling2D((16,16), interpolation='bilinear') (conv5_1)

    up4_2 = UpSampling2D((2,2), interpolation='bilinear', name='up42') (conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42')
    conv4_2 = conv_batchnorm_relu_block(conv4_2, f=f[3])
    conv4_2_2_4 = UpSampling2D((4,4), interpolation='bilinear') (conv4_2)
    conv4_2_1_5 = UpSampling2D((8,8), interpolation='bilinear') (conv4_2)

    up3_3 = UpSampling2D((2,2), interpolation='bilinear', name='up33') (conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2, conv5_1_3_3], name='merge33')
    conv3_3 = conv_batchnorm_relu_block(conv3_3, f=f[2])
    conv3_3_1_5 = UpSampling2D((4,4), interpolation='bilinear') (conv3_3)

    up2_4 = UpSampling2D((2,2), interpolation='bilinear', name='up24') (conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3, conv4_2_2_4, conv5_1_2_4], name='merge24')
    conv2_4 = conv_batchnorm_relu_block(conv2_4, f=f[1])

    up1_5 = UpSampling2D((2,2), interpolation='bilinear', name='up15') (conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4, conv3_3_1_5, conv4_2_1_5, conv5_1_1_5], name='merge15')
    conv1_5 = conv_batchnorm_relu_block(conv1_5, f=f[0])

    # Last convolution with sigmoid
    output = Conv2D(1, 1, activation='sigmoid', padding='same') (conv1_5)
    
    model = Model(inputs=inputs, outputs=output, name='UNetSharp')
    
    return model