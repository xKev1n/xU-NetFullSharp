import random
import numpy as np 

import tensorflow as tf
from keras.optimizers import Adam
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

from utils import l1_ssim, ms_ssim, ssim, mse, mae, psnr

seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed

BACKBONES = ['resnet18', 'efficientnetb0']    ## ResNet-18, EfficientNet-B0'
SIZE=512
INPUT_SHAPE=(SIZE, SIZE, 3)     ## RGB for pretrained U-Net and FPN models
OUT_CHANNELS = 1

## https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265691
def DeBoNet(COMPILE=True, NAME='UNET_RES18'):
    
    modelfpn_ef0 = sm.FPN(BACKBONES[1], input_shape=INPUT_SHAPE, classes=OUT_CHANNELS, activation='sigmoid')
    modelunet_res18 = sm.Unet(BACKBONES[0], input_shape=INPUT_SHAPE, classes=OUT_CHANNELS, activation='sigmoid')
    modelfpn_res18 = sm.FPN(BACKBONES[0], input_shape=INPUT_SHAPE, classes=OUT_CHANNELS, activation='sigmoid')
    
    if NAME == 'UNET_RES18':
        model = modelunet_res18
    elif NAME == 'FPN_RES18':
        model = modelfpn_res18
    elif NAME == 'FPN_EF0':
        model = modelfpn_ef0
    else:
        print(f"Warning: Unknown model name {NAME}!")
        return
    
    if COMPILE:
        model.compile(optimizer=Adam(lr=0.001), 
                    loss=l1_ssim, 
                    metrics=[ms_ssim, ssim, mse, mae, psnr]) 
    
        model.summary()
    
    return model