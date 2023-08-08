# Core
import tensorflow as tf
import keras
import sys
import math
from tensorflow.python.client import device_lib
import cv2 as cv
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
import os
import glob

# Image manipulation
from generator import DataGen, getIds, preprocess_image
from tensorflow.keras.utils import img_to_array
import keras.backend as kb
# Statistics
from sewar import full_ref as fr
import xlsxwriter
import numpy as np

seed = 2019
np.random.seed = seed

# Image size
SIZE = 512
INPUT_SHAPE = (SIZE, SIZE, 1)
BATCH_SIZE = 10

## LOSS FUNCTIONS

# MAE
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

def ssim_loss(y_true, y_pred):
    _ssim_loss = tf.image.ssim(y_true, y_pred, max_val=1)
    _ssim_loss = tf.math.subtract(tf.constant(1.0), _ssim_loss)
    _ssim_loss = tf.reduce_mean(_ssim_loss)

    return _ssim_loss

## USED LOSS FUNCTION
def l1_ssim(y_true, y_pred):
    ALPHA = 0.84
    DELTA = 1 - ALPHA
    
    _ssim_loss = ssim_loss(y_true, y_pred)
    _ssim_loss = tf.math.multiply(_ssim_loss, ALPHA)
    
    _mae = mae(y_true, y_pred)
    _mae = tf.math.multiply(_mae, DELTA)
    
    _loss = tf.math.add(_mae, _ssim_loss)
    
    return _loss

# MS-SSIM
def ms_ssim(y_true, y_pred):
    _mssim = tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=1))
    return _mssim

    
## Check capabilities
def check_capabilities():
    ## GPU Check

    print(device_lib.list_local_devices())

    print("Num of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    tf.test.is_built_with_cuda()

    print(tf.version.VERSION)

    sys.version
    print(tf.__version__)
    print(keras.__version__)


## Saving the model and weights
def save_model(model, model_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open("saved_models/"+model_name, "w") as json_file:
        json_file.write(model_json)

    print("Saved model to disk")

def save_weights(model, file_name):
    model.save_weights("saved_models/"+file_name)

    print("Saved weights to disk")

def load_model(file_name):
    ## Loading the model
    # Load json and create model

    json_file = open('saved_models/'+file_name, 'r')
    model_json = json_file.read()
    json_file.close()

    model = tf.keras.models.model_from_json(model_json)

    print("Loaded model from disk")

    return model

def load_weights(model, file_name):
    # Load weights into new model
    model.load_weights(file_name)

    print("Loaded weights from disk")

## INTERNAL TEST SET
def get_test_data(RGB=False):
    t_path = "ribs_suppresion/new/augmented/test/"

    test_path = t_path + "JSRT/"

    test_ids = getIds(test_path)
    print(len(test_ids))

    test_gen = DataGen(test_ids, t_path, image_size=SIZE, batch_size=BATCH_SIZE, RGB=RGB)
    return test_gen, test_ids

## EXTERNAL TEST SET
def get_unseen_data(random=True, RGB=False):
    t_path = "ribs_suppresion/test_inverted/"

    ids = getIds(t_path)
    print(len(ids))
    test_ids = []
    
    ## Return 10 random images or the whole test set
    if (random):
        for i in range(0, 10):
            num = np.random.randint(0, len(ids))
            print(ids[num])
            test_ids.append(ids[num])
    else:
        for i in range(len(ids)):
            print(ids[i])
            test_ids.append(ids[i])
    
    test_data = []
    for id in test_ids:
        data = t_path + id

        if RGB:
            img = cv.imread(data)
            if (img.shape != (SIZE, SIZE, 3)):
                img = cv.resize(img, (SIZE, SIZE))
        else:
            img = cv.imread(data,0)
            img = np.expand_dims(img, axis = -1)
            if (img.shape != (SIZE, SIZE, 1)):
                img = cv.resize(img, (SIZE, SIZE))
        
        img = preprocess_image(img)
        img = img_to_array(img)
        # Convert to 0--1 interval  
        img = img / 255.0
        
        test_data.append(img)
    
    test_data = np.array(test_data)
    
    return test_data, test_ids
    
def test_model(model):   
    ## Testing the model's predictions
    results = []
    data, ids = get_unseen_data(random=False)

    for i in range (len(data)//BATCH_SIZE):
        temp = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        print(temp.shape)
        result = model.predict(temp)
        result = np.array(result)
        results.append(result)
    results = np.array(results)
    results = np.reshape(results, (len(data), SIZE, SIZE, 1))
    return results, ids

def eval_results(results, ids, model_name):
    for i in range(0, len(results)):
        cv.imwrite("test_predictions/"+model_name+"/{}_predicted.png".format(ids[i]), results[i]*255)
    
def eval_test_results(model, model_name, RGB=False):
    test_gen, test_ids = get_test_data(RGB=RGB)
    print(len(test_gen))
    result = model.predict(test_gen)

    metrics = ["IMAGE", "SSIM", "MS-SSIM", "MSE", "MAE", "PSNR", "UQI", "CORRELATION", "INTERSECTION", "CHI_SQUARED", "BHATTACHARYYA"]

    predicted_ssim = []
    predicted_mssim = []
    predicted_mse = []
    predicted_mae = []
    predicted_psnr = []
    predicted_uqi = []
    predicted_corr = []
    predicted_inter = []
    predicted_chisq = []
    predicted_bhatta = []
    
    workbook = xlsxwriter.Workbook("predictions/512/"+model_name+"/"+model_name+"_predictions_eval.xlsx")
    f = workbook.add_worksheet()

    for col_num, data in enumerate(metrics):
        f.write(0, col_num, data)

    for i in range(0, len(test_gen)):

        source, target = test_gen.__getitem__(i)
        target = np.array(target).astype('float32')
        temp_result = result[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
        temp_result = np.array(temp_result.astype('float32'))

        print(target.shape, temp_result.shape)
        
        for j in range(BATCH_SIZE):
            temp_ssim = ssim(target[j], temp_result[j]).numpy() 
            temp_mssim = ms_ssim(target[j], temp_result[j]).numpy()  
            temp_mse = mse(target[j], temp_result[j]).numpy()
            temp_mae = mae(target[j], temp_result[j]).numpy()
            temp_psnr = psnr(target[j], temp_result[j]).numpy()  
            temp_uqi = fr.uqi(target[j], temp_result[j])  
            
            ## Convert to grayscale
            if target[j].shape[2] == 3:
                target[j] = cv.cvtColor(target[j],cv.COLOR_BGR2GRAY)
            if temp_result[j].shape[2] == 3:
                temp_result[j] = cv.cvtColor(temp_result[j],cv.COLOR_BGR2GRAY)
            
            img_g = target[j]*255
            img_p = temp_result[j]*255
            
            ## Calculate histograms
            hist_g = cv.calcHist([img_g],[0],None,[256],[0,256])
            hist_p = cv.calcHist([img_p],[0],None,[256],[0,256])
            hist_gn = cv.normalize(hist_g, hist_g).flatten()
            hist_pn = cv.normalize(hist_p, hist_p).flatten()
            
            ## Comparison
            temp_corr = cv.compareHist(hist_gn, hist_pn, cv.HISTCMP_CORREL)
            temp_inter = cv.compareHist(hist_gn, hist_pn, cv.HISTCMP_INTERSECT)
            temp_chisq = cv.compareHist(hist_gn, hist_pn, cv.HISTCMP_CHISQR)
            temp_bhatta = cv.compareHist(hist_gn, hist_pn, cv.HISTCMP_BHATTACHARYYA)
            
            
            predicted_ssim.append(temp_ssim)
            predicted_mssim.append(temp_mssim)
            predicted_mse.append(temp_mse)
            predicted_mae.append(temp_mae)
            predicted_psnr.append(temp_psnr)
            predicted_uqi.append(temp_uqi)
            predicted_corr.append(temp_corr)
            predicted_inter.append(temp_inter)
            predicted_chisq.append(temp_chisq)
            predicted_bhatta.append(temp_bhatta)

            vals = [test_ids[i*BATCH_SIZE+j].strip(".png"), temp_ssim, temp_mssim, temp_mse, temp_mae, temp_psnr, temp_uqi, temp_corr, temp_inter, temp_chisq, temp_bhatta]
            
            for col_num, data in enumerate(vals):
                f.write(i*BATCH_SIZE+j+1, col_num, data)

            ## Save results
            cv.imwrite("predictions/512/"+model_name +"/{0}_pred.png".format(test_ids[i*BATCH_SIZE+j].strip(".png")), temp_result[j]*255)
    workbook.close()

## FOR DEBONET    
def eval_test_results_woPred(pred_path, target_path, model_name):
    pred_ids = sorted(glob.glob(pred_path + "*.png"))
    target_ids = sorted(glob.glob(target_path + "*.png"))
    print(pred_ids[0], target_ids[0])
    print(len(pred_ids), len(target_ids))

    metrics = ["IMAGE", "SSIM", "MS-SSIM", "MSE", "MAE", "PSNR", "UQI", "CORRELATION", "INTERSECTION", "CHI_SQUARED", "BHATTACHARYYA"]

    predicted_ssim = []
    predicted_mssim = []
    predicted_mse = []
    predicted_mae = []
    predicted_psnr = []
    predicted_uqi = []
    predicted_corr = []
    predicted_inter = []
    predicted_chisq = []
    predicted_bhatta = []
    
    workbook = xlsxwriter.Workbook(
        "predictions/512/"+model_name+"/"+model_name+"_predictions_eval.xlsx")
    f = workbook.add_worksheet()

    for col_num, data in enumerate(metrics):
        f.write(0, col_num, data)

    for i in range(0, len(target_ids)):

        pred = cv.imread(pred_ids[i], 0)
        pred = np.expand_dims(pred, axis = -1)
        img_p = pred
        pred = np.array(pred).astype('float32')
        pred /= 255.0
        
        target = cv.imread(target_ids[i], 0)
        target = np.expand_dims(target, axis = -1)
        img_g = target
        target = np.array(target).astype('float32')
        target /= 255.0

        print(target.shape, pred.shape)
        
        temp_ssim = ssim(target, pred).numpy() 
        temp_mssim = ms_ssim(target, pred).numpy()  
        temp_mse = mse(target, pred).numpy()
        temp_mae = mae(target, pred).numpy()
        temp_psnr = psnr(target, pred).numpy()  
        temp_uqi = fr.uqi(target, pred)  
        
        hist_g = cv.calcHist([img_g],[0],None,[256],[0,256])
        hist_p = cv.calcHist([img_p],[0],None,[256],[0,256])
        hist_gn = cv.normalize(hist_g, hist_g).flatten()
        hist_pn = cv.normalize(hist_p, hist_p).flatten()
        
        temp_corr = cv.compareHist(hist_gn, hist_pn, cv.HISTCMP_CORREL)
        temp_inter = cv.compareHist(hist_gn, hist_pn, cv.HISTCMP_INTERSECT)
        temp_chisq = cv.compareHist(hist_gn, hist_pn, cv.HISTCMP_CHISQR)
        temp_bhatta = cv.compareHist(hist_gn, hist_pn, cv.HISTCMP_BHATTACHARYYA)
            
            
        predicted_ssim.append(temp_ssim)
        predicted_mssim.append(temp_mssim)
        predicted_mse.append(temp_mse)
        predicted_mae.append(temp_mae)
        predicted_psnr.append(temp_psnr)
        predicted_uqi.append(temp_uqi)
        predicted_corr.append(temp_corr)
        predicted_inter.append(temp_inter)
        predicted_chisq.append(temp_chisq)
        predicted_bhatta.append(temp_bhatta)

        vals = [target_ids[i].strip(".png"), temp_ssim, temp_mssim, temp_mse, temp_mae, temp_psnr, temp_uqi, temp_corr, temp_inter, temp_chisq, temp_bhatta]
            
        for col_num, data in enumerate(vals):
            f.write(i+1, col_num, data)

    workbook.close()


initial_lr = 0.001
epochs = 100
decay = initial_lr / epochs

def lr_time_based_decay(epoch, lr):
    return lr * 1 / (1 + decay * epoch)

def compile_model(model):
    ## COMPILE

    model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr),
    loss = [l1_ssim],
    metrics = [ms_ssim, ssim, mse, mae, psnr]
    )
    model.summary()
    
    return model

def train_model(model):
    ## TRAINING
    
    # System paths
    path = "ribs_suppresion/new/augmented/"
    source_path = path+"train/"
    valid_path = path+"val/"
    
    ## Validation / Training data
    #val_data_size = 720

    train_ids = getIds(source_path+"JSRT/")
    valid_ids = getIds(valid_path+"JSRT/")
    
    #valid_ids = train_ids[:val_data_size] 
    print("Validation:")
    print(len(valid_ids))

    #train_ids = train_ids[val_data_size:]
    print("Training:")
    print(len(train_ids))

    ## Training data generation
    train_gen = DataGen(train_ids, source_path, image_size=SIZE, batch_size=BATCH_SIZE)

    ## Validation data generation
    valid_gen = DataGen(valid_ids, valid_path, image_size=SIZE, batch_size=BATCH_SIZE)


    train_steps = len(train_ids) // BATCH_SIZE
    valid_steps = len(valid_ids) // BATCH_SIZE
    print(train_steps)
    print(valid_steps)
    
    model_name = 'XUNETFS'
    filepath=".tf_checkpoints/512/"+model_name+"/"+model_name+"_b10_f128_best_weights_{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_weights_only=True, monitor='val_loss', save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_time_based_decay, verbose=1)
    earlyStopping = EarlyStopping(monitor='val_loss', 
                               patience=10, 
                               verbose=1, 
                               mode='min')
    
    callbacks_list = [checkpoint, lr_scheduler, earlyStopping]
    
    history = model.fit(train_gen,
                    epochs = epochs,
                    validation_data=valid_gen,
                    steps_per_epoch=train_steps,
                    validation_steps=valid_steps,
                    callbacks=callbacks_list,
                    verbose=2)
    return history

def train_debonet(model):
    ## TRAINING OF THE DeBoNet ENSEMBLE FROM
    # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265691
    
    # System paths
    path = "ribs_suppresion/new/augmented/"
    source_path = path+"train/"
    valid_path = path+"val/"

    train_ids = getIds(source_path+"JSRT/")
    valid_ids = getIds(valid_path+"JSRT/")
    
    print("Validation:")
    print(len(valid_ids))

    print("Training:")
    print(len(train_ids))

    ## Training data generation
    # The backbones require RGB input
    train_gen = DataGen(train_ids, source_path, image_size=SIZE, batch_size=BATCH_SIZE, RGB=True)

    ## Validation data generation
    valid_gen = DataGen(valid_ids, valid_path, image_size=SIZE, batch_size=BATCH_SIZE, RGB=True)

    ## STEPS
    train_steps = len(train_ids) // BATCH_SIZE
    valid_steps = len(valid_ids) // BATCH_SIZE
    print(train_steps)
    print(valid_steps)
    
    ## NAMES: UNET_RES18, FPN_RES18, FPN_EF0
    model_name = 'UNET_RES18'
    filepath=".tf_checkpoints/512/DEBONET/"+model_name+"/"+model_name+"_b10_f256_best_weights_{epoch:02d}.hdf5"
    
    ## SETUP
    checkpoint = ModelCheckpoint(filepath, 
                             monitor='val_loss', 
                             verbose=1, 
                             save_weights_only=True,
                             save_best_only=True, 
                             mode='min') 
    earlyStopping = EarlyStopping(monitor='val_loss', 
                               patience=10, 
                               verbose=1, 
                               mode='min')
    tensor_board = TensorBoard(log_dir='.logs/', 
                           histogram_freq=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.5, 
                              patience=10,
                              verbose=1, 
                              mode='min', 
                              min_lr=0.00001)
    callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]
    
    ## TRAINING
    history = model.fit(train_gen,
                    epochs = epochs,
                    validation_data=valid_gen,
                    steps_per_epoch=train_steps,
                    validation_steps=valid_steps,
                    callbacks=callbacks_list,
                    verbose=2)
    return history

## LR SCHEDULER FOR KALISZ MARCZYK MODEL
def scheduler(epoch, lr):
    if epoch <= 100:
        return lr
    else:
        initial_lrate = 0.001
        drop = 0.5
        epochs_drop = 50.0
        lrate = initial_lrate * math.pow(drop, math.floor((epoch-100)/epochs_drop))
        return lrate

def train_kalisz(model):
    ## TRAINING OF THE KALISZ MARCZYK AUTOENCODER FROM
    # https://ieeexplore.ieee.org/abstract/document/9635451
    
    # System paths
    path = "ribs_suppresion/new/augmented/"
    source_path = path+"train/"
    valid_path = path+"val/"
    
    ## Validation / Training data
    train_ids = getIds(source_path+"JSRT/")
    valid_ids = getIds(valid_path+"JSRT/")
    
    print("Validation:")
    print(len(valid_ids))

    print("Training:")
    print(len(train_ids))

    ## Training data generation
    train_gen = DataGen(train_ids, source_path, image_size=SIZE, batch_size=BATCH_SIZE)

    ## Validation data generation
    valid_gen = DataGen(valid_ids, valid_path, image_size=SIZE, batch_size=BATCH_SIZE)

    ## Steps
    train_steps = len(train_ids) // BATCH_SIZE
    valid_steps = len(valid_ids) // BATCH_SIZE
    print(train_steps)
    print(valid_steps)
    
    ## SETUP
    model_name = 'KALISZ_AE'
    filepath=".tf_checkpoints/512/"+model_name+"/"+model_name+"_b10_best_weights_{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, 
                             monitor='val_loss', 
                             verbose=1, 
                             save_weights_only=True,
                             save_best_only=True, 
                             mode='min') 
    lr_scheduler = LearningRateScheduler(scheduler)
    callbacks_list = [checkpoint, lr_scheduler]
    
    ## TRAINING
    history = model.fit(train_gen,
                    epochs = 300,
                    validation_data=valid_gen,
                    steps_per_epoch=train_steps,
                    validation_steps=valid_steps,
                    callbacks=callbacks_list,
                    verbose=2)
    
    ## RETURN RESULTS
    return history
