## Imports + seed

# Common
import os
import sys
import random
import numpy as np 
from glob import glob
import tensorflow as tf
import keras

# Images 
import cv2 as cv

# Augmentation
import albumentations as A
from tqdm import tqdm


def preprocess_image(image):
    #img = cv.bitwise_not(img)    # INVERT THE IMAGE
    image = np.expand_dims(image, axis = -1)
    if (type(image) == tf.uint16):
        image /= 65535.0
    elif(type(image) == tf.uint8):
        image /= 255.0
    
    return np.array(image)


seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed

# Image size
SIZE = 512
INPUT_SHAPE = (SIZE, SIZE, 1)

print(tf.__version__)
print(keras.__version__)

path = 'C:\\new\\'
source_image_path = path+'train\\JSRT\\'
target_image_path = path+'train\\BSE_JSRT\\'

source_val_image_path = path+'val\\JSRT\\'
target_val_image_path = path+'val\\BSE_JSRT\\'

source_test_image_path = path+'test\\JSRT\\'
target_test_image_path = path+'test\\BSE_JSRT\\'

# Get Images
source_image_names = sorted(glob(source_image_path + "*.png"))
target_image_names = sorted(glob(target_image_path + "*.png"))

source_val_image_names = sorted(glob(source_val_image_path + "*.png"))
target_val_image_names = sorted(glob(target_val_image_path + "*.png"))

source_test_image_names = sorted(glob(source_test_image_path + "*.png"))
target_test_image_names = sorted(glob(target_test_image_path + "*.png"))

print(len(source_image_names))
print(len(target_image_names))

print(len(source_val_image_names))
print(len(target_val_image_names))

print(len(source_test_image_names))
print(len(target_test_image_names))

print(source_image_names[0], target_image_names[0])
print(source_val_image_names[0], target_val_image_names[0])
print(source_test_image_names[0], target_test_image_names[0])

train_transform = A.Compose(
    [
        A.GaussNoise(p=0.3),
        A.MultiplicativeNoise(p=0.15),
    ],
)

test_transform = A.Compose(
    [
        A.InvertImg(p=1.0),
        #A.CLAHE(clip_limit=1.6, tile_grid_size=(10,10), p=1.0),
    ],
    additional_targets={'image0': 'image'}
)


transform = A.Compose(
    [
        A.InvertImg(),
        A.CLAHE(clip_limit=[1,2], tile_grid_size=(10,10), p=0.6),
        A.UnsharpMask(p=0.35),
        A.RandomCropFromBorders(p=0.6),
        A.ElasticTransform(p=0.25),
        A.GridDistortion(p=0.25),
        A.Perspective(p=0.35),
        A.HorizontalFlip(p=0.35),
        A.ShiftScaleRotate(p=0.7, shift_limit_y=0.1, shift_limit_x=0.1),
        A.RandomBrightnessContrast(p=0.35),
    ],
    additional_targets={'image0': 'image'}
)

val_transform = A.Compose(
    [
        A.InvertImg(),
        A.CLAHE(clip_limit=[1,2], tile_grid_size=(10,10), p=0.6),
        A.UnsharpMask(p=0.35),
        A.HorizontalFlip(p=0.45),
        A.RandomBrightnessContrast(p=0.45),
    ],
    additional_targets={'image0': 'image'}
)

target_train_path = "C:\\new\\augmented\\train\\"
target_val_path = "C:\\new\\augmented\\val\\"
target_test_path = "C:\\new\\augmented\\test\\"

for times in range(50):
    for i in tqdm(range (len(source_image_names))):
        image = cv.imread(source_image_names[i], 0)
        image0 = cv.imread(target_image_names[i], 0)

        image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
        image0 = cv.normalize(image0, None, 0, 255, cv.NORM_MINMAX)
        
        train_trans = train_transform(image=image)
        trans = transform(image=train_trans['image'], image0=image0)
        
        image = cv.resize(trans['image'], (SIZE, SIZE), cv.INTER_CUBIC)
        image0 = cv.resize(trans['image0'], (SIZE, SIZE), cv.INTER_CUBIC)
        
        image = np.array(image)
        image0 = np.array(image0)

        cv.imwrite(target_train_path+"JSRT\\"+source_image_names[i].split('\\')[-1][:-4]+"_"+str(times)+".png", image)
        cv.imwrite(target_train_path+"BSE_JSRT\\"+source_image_names[i].split('\\')[-1][:-4]+"_"+str(times)+".png", image0)

for times in range(12):
    for i in tqdm(range (len(source_val_image_names))):
        image = cv.imread(source_val_image_names[i], 0)
        image0 = cv.imread(target_val_image_names[i], 0)

        image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
        image0 = cv.normalize(image0, None, 0, 255, cv.NORM_MINMAX)
        
        trans = val_transform(image=image, image0=image0)
        
        image = cv.resize(trans['image'], (SIZE, SIZE), cv.INTER_CUBIC)
        image0 = cv.resize(trans['image0'], (SIZE, SIZE), cv.INTER_CUBIC)
        
        image = np.array(image)
        image0 = np.array(image0)

        cv.imwrite(target_val_path+"JSRT\\"+source_val_image_names[i].split('\\')[-1][:-4]+"_"+str(times)+".png", image)
        cv.imwrite(target_val_path+"BSE_JSRT\\"+source_val_image_names[i].split('\\')[-1][:-4]+"_"+str(times)+".png", image0)

for times in range(1):
    for i in tqdm(range (len(source_test_image_names))):
        image = cv.imread(source_test_image_names[i], 0)
        image0 = cv.imread(target_test_image_names[i], 0)

        image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
        image0 = cv.normalize(image0, None, 0, 255, cv.NORM_MINMAX)
        
        #trans = test_transform(image=image, image0=image0)
        #image = cv.resize(trans['image'], (IMAGE_SIZE, IMAGE_SIZE), cv.INTER_CUBIC)
        #image0 = cv.resize(trans['image0'], (IMAGE_SIZE, IMAGE_SIZE), cv.INTER_CUBIC)
        
        image = cv.resize(image, (SIZE, SIZE), cv.INTER_CUBIC)
        image0 = cv.resize(image0, (SIZE, SIZE), cv.INTER_CUBIC)
        
        image = np.array(image)
        image0 = np.array(image0)

        cv.imwrite(target_test_path+"JSRT\\"+source_test_image_names[i].split('\\')[-1][:-4]+"_"+str(times)+".png", image)
        cv.imwrite(target_test_path+"BSE_JSRT\\"+source_test_image_names[i].split('\\')[-1][:-4]+"_"+str(times)+".png", image0)