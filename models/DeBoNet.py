import tensorflow as tf
from keras.optimizers import Adam
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

BACKBONES = ['resnet18', 'efficientnetb0']    ## ResNet-18, EfficientNet-B0'
SIZE=512
INPUT_SHAPE=(SIZE, SIZE, 3)     ## RGB for pretrained U-Net and FPN models
OUT_CHANNELS = 1

def mae(y_true, y_pred):
    return tf.math.reduce_mean(tf.keras.losses.MAE(y_true, y_pred))

# MSE

def mse(y_true, y_pred):
    return tf.math.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))

# PSNR
def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

# SSIM

def ssim(y_true, y_pred):
    _ssim = tf.math.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1))
    return _ssim

# MSSIM
def ms_ssim(y_true, y_pred):
    _mssim = tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=1))
    return _mssim

def ssim_loss(y_true, y_pred):
    _ssim_loss = tf.image.ssim(y_true, y_pred, max_val=1)
    _ssim_loss = tf.math.subtract(tf.constant(1.0), _ssim_loss)
    _ssim_loss = tf.reduce_mean(_ssim_loss)

    return _ssim_loss

def l1_ssim(y_true, y_pred):
    ALPHA = 0.84
    DELTA = 1 - ALPHA
    
    _ssim_loss = ssim_loss(y_true, y_pred)
    _ssim_loss = tf.math.multiply(_ssim_loss, ALPHA)
    
    _mae = mae(y_true, y_pred)
    _mae = tf.math.multiply(_mae, DELTA)
    
    _loss = tf.math.add(_mae, _ssim_loss)
    
    return _loss


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
        print("Warning: Unknown model name %s" % NAME)
        return
    
    if COMPILE:
        model.compile(optimizer=Adam(lr=0.001), 
                  loss=l1_ssim, 
                  metrics=[ms_ssim, ssim, mse, mae, psnr]) 
    
        model.summary()
    
    return model