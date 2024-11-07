import random
import numpy as np 
import tensorflow as tf

import keras
from keras.layers import concatenate, Activation, Conv2D, Input, MaxPool2D, UpSampling2D, add, multiply
import keras.backend as kb
from keras.models import Model

seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed


# Image size
SIZE = 512
INPUT_SHAPE = (SIZE, SIZE, 1)

def conv_batchnorm_relu_block(tensor, f):

    x = Conv2D(f, (3, 3), padding='same') (tensor)
    #x = BatchNormalization() (x)
    x = Activation('relu') (x)
    x = Conv2D(f, (3, 3), padding='same') (x)
    #x = BatchNormalization() (x)
    x = Activation('relu') (x)

    return x

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
    
    up_psi = UpSampling2D(size = (x_shape[1] // sig_xg_shape[1], x_shape[2] // sig_xg_shape[2]), interpolation = 'bilinear') (sig_xg)
    
    y = multiply([up_psi, x])
    
    out = Conv2D(x_shape[3], (1,1), padding = "same") (y)
    
    return out


## ATTENTION U-NET++ MODEL

def Att_UNetPP(INPUT_SHAPE = INPUT_SHAPE, DS = False):

    f = [16, 32, 64, 128, 256]

    inputs = Input(INPUT_SHAPE, name = 'input')

    conv1_1 = conv_batchnorm_relu_block(inputs, f=f[0])
    pool1 = MaxPool2D((2,2), (2,2), name='maxpool1') (conv1_1)

    conv2_1 = conv_batchnorm_relu_block(pool1, f=f[1])
    pool2 = MaxPool2D((2,2), (2,2), name='maxpool2') (conv2_1)
    
    att1_1 = attention_block(conv1_1, conv2_1, f[0])

    up1_2 = UpSampling2D((2,2), interpolation = 'bilinear', name='up12') (conv2_1)
    conv1_2 = concatenate([up1_2, att1_1], name='merge12')
    conv1_2 = conv_batchnorm_relu_block(conv1_2,  f=f[0])

    conv3_1 = conv_batchnorm_relu_block(pool2, f=f[2])
    pool3 = MaxPool2D((2,2), (2,2), name='maxpool3') (conv3_1)

    att2_1 = attention_block(conv2_1, conv3_1, f[1])
    
    up2_2 = UpSampling2D((2,2), interpolation = 'bilinear', name='up22') (conv3_1)
    conv2_2 = concatenate([up2_2, att2_1], name='merge22')
    conv2_2 = conv_batchnorm_relu_block(conv2_2, f=f[1])
    
    att1_2 = attention_block(conv1_2, conv2_2, f[0])

    up1_3 = UpSampling2D((2,2), interpolation = 'bilinear', name='up13') (conv2_2)
    conv1_3 = concatenate([up1_3, att1_1, att1_2], name='merge13')
    conv1_3 = conv_batchnorm_relu_block(conv1_3, f=f[0])

    conv4_1 = conv_batchnorm_relu_block(pool3, f=f[3])
    pool4 = MaxPool2D((2,2), (2,2), name='maxpool4') (conv4_1)
    
    att3_1 = attention_block(conv3_1, conv4_1, f[2])
    
    up3_2 = UpSampling2D((2,2), interpolation = 'bilinear', name='up32') (conv4_1)
    conv3_2 = concatenate([up3_2, att3_1], name='merge32')
    conv3_2 = conv_batchnorm_relu_block(conv3_2, f=f[2])

    att2_2 = attention_block(conv2_2, conv3_1, f[1])
    
    up2_3 = UpSampling2D((2,2), interpolation = 'bilinear', name='up23') (conv3_2)
    conv2_3 = concatenate([up2_3, att2_1, att2_2], name='merge23')
    conv2_3 = conv_batchnorm_relu_block(conv2_3, f=f[1])
    
    att1_3 = attention_block(conv1_3, conv2_3, f[0])

    up1_4 = UpSampling2D((2,2), interpolation = 'bilinear', name='up14') (conv2_3)
    conv1_4 = concatenate([up1_4, att1_1, att1_2, att1_3], name='merge14')
    conv1_4 = conv_batchnorm_relu_block(conv1_4, f=f[0])

    conv5_1 = conv_batchnorm_relu_block(pool4, f=f[4])
    
    att4_1 = attention_block(conv4_1, conv5_1, f[3])

    up4_2 = UpSampling2D((2,2), interpolation = 'bilinear', name='up42') (conv5_1)
    conv4_2 = concatenate([up4_2, att4_1], name='merge42')
    conv4_2 = conv_batchnorm_relu_block(conv4_2, f=f[3])
    
    att3_2 = attention_block(conv3_2, conv4_2, f[2])

    up3_3 = UpSampling2D((2,2), interpolation = 'bilinear', name='up33') (conv4_2)
    conv3_3 = concatenate([up3_3, att3_1, att3_2], name='merge33')
    conv3_3 = conv_batchnorm_relu_block(conv3_3, f=f[2])
    
    att2_3 = attention_block(conv2_3, conv3_3, f[1])

    up2_4 = UpSampling2D((2,2), interpolation = 'bilinear', name='up24') (conv3_3)
    conv2_4 = concatenate([up2_4, att2_1, att2_2, att2_3], name='merge24')
    conv2_4 = conv_batchnorm_relu_block(conv2_4, f=f[1])
    
    att1_4 = attention_block(conv1_4, conv2_4, f[0])

    up1_5 = UpSampling2D((2,2), interpolation = 'bilinear', name='up15') (conv2_4)
    conv1_5 = concatenate([up1_5, att1_1, att1_2, att1_3, att1_4], name='merge15')
    conv1_5 = conv_batchnorm_relu_block(conv1_5, f=f[0])


    output1 = Conv2D(1, 1, activation='sigmoid', padding='same') (conv1_2)
    output2 = Conv2D(1, 1, activation='sigmoid', padding='same') (conv1_3)
    output3 = Conv2D(1, 1, activation='sigmoid', padding='same') (conv1_4)
    output4 = Conv2D(1, 1, activation='sigmoid', padding='same') (conv1_5)

    if DS:
        outputs = [output1, output2, output3, output4]
    else:
        outputs = output4
    
    model = Model(inputs=inputs, outputs=outputs, name='Att_UNetPP')
    return model