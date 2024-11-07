## Imports + seed

# Common
import random
import numpy as np 
import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import concatenate, MaxPool2D, Input, Conv2D, UpSampling2D, add, multiply, DepthwiseConv2D, Conv2DTranspose, Activation, BatchNormalization
import keras.backend as kb

seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed


# Image size
SIZE = 512
INPUT_SHAPE = (SIZE, SIZE, 1)

## Custom Activation function
## xUnit: https://blog.paperspace.com/xunit-spatial-activation/
def x_unit(x, kernel_size=9):
    # res = BatchNormalization() (x)        # Optional Batch Normalization layer
    res = Activation("relu") (x)
    res = DepthwiseConv2D(kernel_size=kernel_size, padding='same') (res)
    # res = BatchNormalization() (res)      # Optional Batch Normalization layer
    res = Activation("sigmoid") (res)
    
    res = multiply([res, x])
    
    return res

## Attention block
# From https://arxiv.org/abs/1804.03999
def attention_block(x, gating, inter_shape):
    x_shape = kb.int_shape(x)
    
    theta_x = Conv2D(inter_shape, (1,1), strides = (2,2), padding = "same", kernel_initializer='he_uniform') (x)
    
    phi_g = Conv2D(inter_shape, (1,1), padding = "same", kernel_initializer='he_uniform') (gating)
    
    con_xg = add([phi_g, theta_x])
    
    act_xg = Activation("relu") (con_xg)
    
    psi = Conv2D(1, (1,1), padding = "same") (act_xg)
    
    sig_xg = Activation("sigmoid") (psi)
    sig_xg_shape = kb.int_shape(sig_xg)
    
    up_psi = UpSampling2D(size = (x_shape[1] // sig_xg_shape[1], x_shape[2] // sig_xg_shape[2]), interpolation = 'bilinear') (sig_xg)
    
    y = multiply([up_psi, x])
    
    out = Conv2D(x_shape[3], (1,1), padding = "same", kernel_initializer='he_uniform') (y)

    return out

## Layers

def conv_block(x, filters, kernel_size=(3,3), padding = "same", strides = 1, dilation_rate = (1,1)):
    ## Convolution block and activation
    c = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, dilation_rate=dilation_rate, kernel_initializer='he_uniform') (x)
    c = x_unit(c)
    
    return c

def up_block(x, skip, filters, kernel_size=(3,3), padding = "same", strides = 1, dilation_rate = (1,1)):
    concat = concatenate([x, skip])
    c = conv_block(concat, filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, dilation_rate=dilation_rate)
    
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


def Att_xUNetFS(INPUT_SHAPE = INPUT_SHAPE, OUTPUT_CHANNELS = 1):

    f = [8, 16, 32, 64, 128]

    inputs = Input(INPUT_SHAPE, name = 'INPUT')

    conv1_1 = Encoder_block(inputs, f[0], level=1)
    pool1 = MaxPool2D((2,2), strides=2, name='POOL1') (conv1_1)

    conv2_1 = Encoder_block(pool1, f[1], level=1)
    pool2 = MaxPool2D((2,2), strides=2, name='POOL2') (conv2_1)
    
    att1_1 = attention_block(conv1_1, conv2_1, f[0])
    ## Down connections
    down1_1_2_2 = MaxPool2D((2,2), strides=2) (att1_1)
    down1_1_3_2 = MaxPool2D((4,4), strides=4) (att1_1)
    down1_1_4_2 = MaxPool2D((8,8), strides=8) (att1_1)

    up1_2 = UpSampling2D((2,2), interpolation='bilinear', name='UP1_2') (conv2_1)
    conv1_2 = concatenate([up1_2, att1_1], name='SKIP1_2')
    conv1_2 = conv_block(conv1_2, f[0])

    conv3_1 = Encoder_block(pool2, f[2], level=2)
    pool3 = MaxPool2D((2,2), strides=2, name='POOL3') (conv3_1)

    att2_1 = attention_block(conv2_1, conv3_1, f[1])
    ## Down connections
    down2_1_3_2 = MaxPool2D((2,2), strides=2) (att2_1)
    down2_1_4_2 = MaxPool2D((4,4), strides=4) (att2_1)
    
    up2_2 = UpSampling2D((2,2), interpolation='bilinear', name='UP2_2') (conv3_1)
    conv2_2 = concatenate([up2_2, att2_1, down1_1_2_2], name='SKIP2_2')
    conv2_2 = conv_block(conv2_2, f[1])
    
    att1_2 = attention_block(conv1_2, conv2_2, f[0])
    ## Down connections
    down1_2_2_3 = MaxPool2D((2,2), strides=2) (att1_2)
    down1_2_3_3 = MaxPool2D((4,4), strides=4) (att1_2)

    up1_3 = UpSampling2D((2,2), interpolation='bilinear', name='UP1_3') (conv2_2)

    conv4_1 = Encoder_block(pool3, f[3], level=2)
    ## Up connections
    conv4_1_2_3 = UpSampling2D((4,4), interpolation='bilinear') (conv4_1)
    conv4_1_1_4 = UpSampling2D((8,8), interpolation='bilinear') (conv4_1)
    
    pool4 = MaxPool2D((2,2), strides=2, name='POOL4') (conv4_1)
    conv5_1 = Encoder_block(pool4, f[4], level=3)
    ## Up connections
    conv5_1_3_3 = UpSampling2D((4,4), interpolation='bilinear') (conv5_1)
    conv5_1_2_4 = UpSampling2D((8,8), interpolation='bilinear') (conv5_1)
    conv5_1_1_5 = UpSampling2D((16,16), interpolation='bilinear') (conv5_1)
    
    att3_1 = attention_block(conv3_1, conv4_1, f[2])
    ## Up connection
    att3_1_1_3 = UpSampling2D((4,4), interpolation='bilinear') (att3_1)
    ## Down connection
    down3_1_4_2 = MaxPool2D((2,2), strides=2) (att3_1)
    
    att4_1 = attention_block(conv4_1, conv5_1, f[3])
    
    up4_2 = UpSampling2D((2,2), interpolation='bilinear', name='UP4_2') (conv5_1)
    conv4_2_1 = concatenate([down1_1_4_2, down2_1_4_2, down3_1_4_2], name='SKIP4_2_1')
    conv4_2_1 = conv_block(conv4_2_1, f[3]*3)
    conv4_2 = concatenate([up4_2, att4_1, conv4_2_1], name='SKIP4_2')
    conv4_2 = Encoder_block(conv4_2, f[3], level=2)
    ## Up connections
    conv4_2_2_4 = UpSampling2D((4,4), interpolation='bilinear') (conv4_2)
    conv4_2_1_5 = UpSampling2D((8,8), interpolation='bilinear') (conv4_2)
    
    conv1_3 = concatenate([up1_3, att1_1, att1_2, att3_1_1_3], name='SKIP1_3')
    conv1_3 = conv_block(conv1_3, f[0])
    
    up3_2 = UpSampling2D((2,2), interpolation='bilinear', name='UP3_2') (conv4_1)
    conv3_2_1 = concatenate([down1_1_3_2, down2_1_3_2], name='SKIP3_2_1')
    conv3_2_1 = conv_block(conv3_2_1, f[2]*2)
    conv3_2 = concatenate([up3_2, att3_1, conv3_2_1], name='SKIP3_2')
    conv3_2 = conv_block(conv3_2, f[2])

    att2_2 = attention_block(conv2_2, conv3_1, f[1])
    ## Down connection
    down2_2_3_3 = MaxPool2D((2,2), strides=2) (att2_2)
    
    att3_2 = attention_block(conv3_2, conv4_2, f[2])
    ## Up connection
    att3_2_1_4 = UpSampling2D((4,4), interpolation='bilinear') (att3_2)
    
    up2_3 = UpSampling2D((2,2), interpolation='bilinear', name='UP2_3') (conv3_2)
    conv2_3 = concatenate([up2_3, att2_1, att2_2, conv4_1_2_3, down1_2_2_3], name='SKIP2_3')
    conv2_3 = conv_block(conv2_3, f[1])
    
    att1_3 = attention_block(conv1_3, conv2_3, f[0])
    ## Down connection
    down1_3_2_4 = MaxPool2D((2,2), strides=2) (att1_3)

    up1_4 = UpSampling2D((2,2), interpolation='bilinear', name='UP1_4') (conv2_3)
    conv1_4 = concatenate([up1_4, att1_1, att1_2, att1_3, att3_2_1_4, conv4_1_1_4], name='SKIP1_4')
    conv1_4 = conv_block(conv1_4, f[0])


    up3_3 = UpSampling2D((2,2), interpolation='bilinear', name='UP3_3') (conv4_2)
    conv3_3_1 = concatenate([down1_2_3_3, down2_2_3_3], name='SKIP3_3_1')
    conv3_3_1 = conv_block(conv3_3_1, f[2]*2)
    conv3_3 = concatenate([up3_3, att3_1, att3_2, conv5_1_3_3, conv3_3_1], name='SKIP3_3')
    conv3_3 = Encoder_block(conv3_3, f[2], level=2)
    ## Up connection
    conv3_3_1_5 = UpSampling2D((4,4), interpolation='bilinear') (conv3_3)
    
    att2_3 = attention_block(conv2_3, conv3_3, f[1])

    up2_4 = UpSampling2D((2,2), interpolation='bilinear', name='UP2_4') (conv3_3)
    conv2_4_1 = concatenate([conv4_2_2_4, conv5_1_2_4], name='SKIP2_4_1')
    conv2_4_1 = conv_block(conv2_4_1, f[1]*2)
    conv2_4 = concatenate([up2_4, att2_1, att2_2, att2_3, conv2_4_1, down1_3_2_4], name='SKIP2_4')
    conv2_4 = Encoder_block(conv2_4, f[1], level=1)
    
    att1_4 = attention_block(conv1_4, conv2_4, f[0])

    up1_5 = UpSampling2D((2,2), interpolation='bilinear', name='UP1_5') (conv2_4)
    conv1_5_1 = concatenate([conv3_3_1_5, conv4_2_1_5, conv5_1_1_5], name='SKIP1_5_1')
    conv1_5_1 = conv_block(conv1_5_1, f[0]*3)
    conv1_5 = concatenate([up1_5, att1_1, att1_2, att1_3, att1_4, conv1_5_1], name='SKIP1_5')
    conv1_5 = Encoder_block(conv1_5, f[0], level=1)

    #### OUTPUT #####

    out = Conv2D(OUTPUT_CHANNELS, kernel_size=(1,1), padding = 'same', activation='sigmoid' , name='OUT') (conv1_5)
    
    model = Model(inputs=inputs, outputs=out, name='Att_xU-NetFullSharp')
    
    return model