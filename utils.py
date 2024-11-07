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
from tqdm import tqdm

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
    mae = tf.keras.losses.MeanAbsoluteError()
    return mae(y_true, y_pred)
# MSE
def mse(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred)

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

    sys.version
    print(tf.__version__)
    print(keras.__version__)


## Saving the model and weights
def save_model(model, model_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join('saved_models', model_name), "w") as json_file:
        json_file.write(model_json)

    print("Saved model to disk")

def save_weights(model, file_name):
    model.save_weights(os.path.join('saved_models', file_name))

    print("Saved weights to disk")

def load_model(file_name):
    ## Loading the model
    # Load json and create model

    json_file = open(os.path.join('saved_models', file_name), 'r')
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
def get_test_data(t_path, RGB=False):

    test_path = os.path.join(t_path, 'JSRT')

    test_ids = getIds(test_path)
    print(f'Number of test images: {len(test_ids)}')

    test_gen = DataGen(test_ids, t_path, image_size=SIZE, batch_size=BATCH_SIZE, RGB=RGB)
    return test_gen, test_ids

def get_flops(model, model_inputs) -> float:
        """
        Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model
        in inference mode. It uses tf.compat.v1.profiler under the hood.
        Reference: https://github.com/wandb/wandb/blob/latest/wandb/integration/keras/keras.py#L1025-L1073
        """

        if not isinstance(
            model, (tf.keras.models.Sequential, tf.keras.models.Model)
        ):
            raise ValueError(
                "Calculating FLOPS is only supported for "
                "`tf.keras.Model` and `tf.keras.Sequential` instances."
            )

        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph,
        )

        # Compute FLOPs for one sample
        batch_size = 1
        inputs = [
            tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
            for inp in model_inputs
        ]

        # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
        real_model = tf.function(model).get_concrete_function(inputs)
        frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

        # Calculate FLOPs with tf.profiler
        run_meta = tf.compat.v1.RunMetadata()
        opts = (
            tf.compat.v1.profiler.ProfileOptionBuilder(
                tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
            )
            .with_empty_output()
            .build()
        )

        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
        )

        tf.compat.v1.reset_default_graph()

        # convert to GFLOPs
        return (flops.total_float_ops / 1e9)/2

## EXTERNAL TEST SET
def get_unseen_data(t_path, random=False, RGB=False):

    ids = getIds(t_path)
    print(f'Number of test images: {len(ids)}')
    test_ids = []
    
    ## Return 10 random images or the whole test set
    if (random):
        for i in range(0, 10):
            num = np.random.randint(0, len(ids))
            test_ids.append(ids[num])
    else:
        for i in range(len(ids)):
            test_ids.append(ids[i])
    
    test_data = []
    for id in tqdm(test_ids, desc='Loading test data'):
        data = os.path.join(t_path, id)

        if RGB:
            img = cv.imread(data)
            if (img.shape != (SIZE, SIZE, 3)):
                img = cv.resize(img, (SIZE, SIZE), interpolation=cv.INTER_LANCZOS4)
        else:
            img = cv.imread(data, 0)
            img = np.expand_dims(img, axis = -1)
            if (img.shape != (SIZE, SIZE, 1)):
                img = cv.resize(img, (SIZE, SIZE))
        
        img = preprocess_image(img)
        img = img_to_array(img)
        # Convert to 0--1 interval  
        img /= 255.0
        
        test_data.append(img)
    
    test_data = np.array(test_data)
    
    return test_data, test_ids
    
def test_model(model, t_path, RGB=False, random=False):   
    ## Testing the model's predictions
    results = []
    times = []
    data, ids = get_unseen_data(t_path, random=random, RGB=RGB)
    import time
    for i in tqdm(range(0, len(data), BATCH_SIZE), desc='Performing inference'):
        batch_data = data[i:i+BATCH_SIZE]
        start = time.time()
        result = model.predict(batch_data)
        end = time.time()
        results.extend(result)
        times.append(end-start)
    
    print(f'Mean inference time for a batch of size {BATCH_SIZE}: {np.mean(times) * 1000:.2f} ms')
    print(f'Median inference time for a batch of size {BATCH_SIZE}: {np.median(times) * 1000:.2f} ms')
    print(f'Min inference time for a batch of size {BATCH_SIZE}: {np.min(times) * 1000:.2f} ms')
    print(f'Max inference time for a batch of size {BATCH_SIZE}: {np.max(times) * 1000:.2f} ms')
    print(f'Std inference time for a batch of size {BATCH_SIZE}: {np.std(times) * 1000:.2f} ms')

    results = np.array(results)
    results = np.reshape(results, (len(data), SIZE, SIZE, 1))

    return results, ids

def eval_results(results, ids, model_name):
    os.makedirs(os.path.join("outputs", "external", model_name), exist_ok=True)
    
    for i in tqdm(range(0, len(results)), desc='Saving predictions'):
        cv.imwrite(os.path.join("outputs", "external", model_name, f"{ids[i]}_predicted.png"), results[i]*255)
    
def eval_test_results(model, model_name, t_path, RGB=False):
    test_gen, test_ids = get_test_data(t_path, RGB=RGB)
    print(f'Test batches: {len(test_gen)}')
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
    
    out_path = os.path.join("outputs", "internal", model_name)
    os.makedirs(out_path, exist_ok=True)
    
    workbook = xlsxwriter.Workbook(os.path.join(out_path, f"{model_name}_predictions_eval.xlsx"))
    f = workbook.add_worksheet()

    for col_num, data in enumerate(metrics):
        f.write(0, col_num, data)

    for i in tqdm(range(0, len(test_gen)), desc='Evaluating results'):

        source, target = test_gen.__getitem__(i)
        target = np.array(target).astype('float32')
        temp_result = result[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
        temp_result = np.array(temp_result.astype('float32'))
        
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
            cv.imwrite(os.path.join(out_path, f"{test_ids[i*BATCH_SIZE+j].strip('.png')}_pred.png"), temp_result[j]*255)
    workbook.close()

## FOR DEBONET ENSEMBLE, MATLAB SCRIPT FOR GENERATING THE COMBINED OUTPUT IS AVAILABLE AT: https://github.com/sivaramakrishnan-rajaraman/Bone-Suppresion-Ensemble/blob/main/bone_suppression_ensemble.py    
def eval_test_results_woPred(pred_path, target_path, model_name):
    pred_ids = sorted(glob.glob(pred_path + "*.png"))
    target_ids = sorted(glob.glob(target_path + "*.png"))

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
    
    out_path = os.path.join("outputs", "internal", model_name)
    os.makedirs(out_path, exist_ok=True)
    
    workbook = xlsxwriter.Workbook(
        os.path.join(out_path, f"{model_name}_predictions_eval.xlsx"))
    f = workbook.add_worksheet()

    for col_num, data in enumerate(metrics):
        f.write(0, col_num, data)

    for i in tqdm(range(0, len(target_ids)), desc='Evaluating results'):

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

def train_model(model, path, model_name):
    ## TRAINING
    
    # System paths
    source_path = os.path.join(path, "train")
    valid_path = os.path.join(path, "val")
    
    ## Validation / Training data
    #val_data_size = 720

    train_ids = getIds(os.path.join(source_path, "JSRT"))
    valid_ids = getIds(os.path.join(valid_path, "JSRT"))
    
    #valid_ids = train_ids[:val_data_size] 
    print(f"Validation: {len(valid_ids)}")

    #train_ids = train_ids[val_data_size:]
    print(f"Training: {len(train_ids)}")

    ## Training data generation
    train_gen = DataGen(train_ids, source_path, image_size=SIZE, batch_size=BATCH_SIZE)

    ## Validation data generation
    valid_gen = DataGen(valid_ids, valid_path, image_size=SIZE, batch_size=BATCH_SIZE)


    train_steps = len(train_ids) // BATCH_SIZE
    valid_steps = len(valid_ids) // BATCH_SIZE
    print(f'Training steps: {train_steps}')
    print(f'Validation steps: {valid_steps}')
    
    os.makedirs(os.path.join(".tf_checkpoints", model_name), exist_ok=True)
    filepath = os.path.join(".tf_checkpoints", model_name, f"{model_name}_b{BATCH_SIZE}_best_weights_{epoch:02d}.hdf5")
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

def train_debonet(model, path, model_name):
    ## TRAINING OF THE DeBoNet ENSEMBLE FROM
    # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265691
    
    # System paths
    source_path = os.path.join(path, "train")
    valid_path = os.path.join(path, "val")

    train_ids = getIds(os.path.join(source_path, "JSRT"))
    valid_ids = getIds(os.path.join(valid_path, "JSRT"))
    
    print(f"Validation: {len(valid_ids)}")

    print(f"Training: {len(train_ids)}")

    ## Training data generation
    # The backbones require RGB input
    train_gen = DataGen(train_ids, source_path, image_size=SIZE, batch_size=BATCH_SIZE, RGB=True)

    ## Validation data generation
    valid_gen = DataGen(valid_ids, valid_path, image_size=SIZE, batch_size=BATCH_SIZE, RGB=True)

    ## STEPS
    train_steps = len(train_ids) // BATCH_SIZE
    valid_steps = len(valid_ids) // BATCH_SIZE
    print(f'Training steps: {train_steps}')
    print(f'Validation steps: {valid_steps}')
    
    ## NAMES: UNET_RES18, FPN_RES18, FPN_EF0
    os.makedirs(os.path.join(".tf_checkpoints", "DEBONET", model_name), exist_ok=True)
    filepath = os.path.join(".tf_checkpoints", "DEBONET", model_name, f"{model_name}_b{BATCH_SIZE}_best_weights_{epoch:02d}.hdf5")
    
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
    tensor_board = TensorBoard(log_dir='.logs/', histogram_freq=0)
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

def train_kalisz(model, path, model_name="KALISZ_AE"):
    ## TRAINING OF THE KALISZ MARCZYK AUTOENCODER FROM
    # https://ieeexplore.ieee.org/abstract/document/9635451
    
    # System paths
    source_path = os.path.join(path, "train")
    valid_path = os.path.join(path, "val")
    
    ## Validation / Training data
    train_ids = getIds(os.path.join(source_path, "JSRT"))
    valid_ids = getIds(os.path.join(valid_path, "JSRT"))
    
    print(f"Validation: {len(valid_ids)}")

    print(f"Training: {len(train_ids)}")

    ## Training data generation
    train_gen = DataGen(train_ids, source_path, image_size=SIZE, batch_size=BATCH_SIZE)

    ## Validation data generation
    valid_gen = DataGen(valid_ids, valid_path, image_size=SIZE, batch_size=BATCH_SIZE)

    ## Steps
    train_steps = len(train_ids) // BATCH_SIZE
    valid_steps = len(valid_ids) // BATCH_SIZE
    print(f'Training steps: {train_steps}')
    print(f'Validation steps: {valid_steps}')
    
    ## SETUP
    os.makedirs(os.path.join(".tf_checkpoints", model_name), exist_ok=True)
    filepath = os.path.join(".tf_checkpoints", model_name, f"{model_name}_b{BATCH_SIZE}_best_weights_{epoch:02d}.hdf5")
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