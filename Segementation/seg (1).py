# -*- coding: utf-8 -*-
"""
Created on Fri May 19 18:28:25 2023

@author: user
"""

import os
import cv2
import glob
import PIL
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage 
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize


# neural imaging
import nilearn as nl
import nibabel as nib
import nilearn.plotting as nlplt
from mayavi import mlab


# ml libs
import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers.experimental import preprocessing


# dice loss as defined above for 4 classes
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
   #     K.print_tensor(loss, message='loss value for class {} : '.format(SEGMENT_CLASSES[i]))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
#    K.print_tensor(total_loss, message=' total dice coef: ')
    return total_loss


 
# define per class evaluation of dice coef
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,1])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,2] * y_pred[:,:,:,2]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,2])) + K.sum(K.square(y_pred[:,:,:,2])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,3])) + K.sum(K.square(y_pred[:,:,:,3])) + epsilon)



# Computing Precision 
def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    
# Computing Sensitivity      
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


# Computing Specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def predictByCase(case):
    model = keras.models.load_model('model/seg_model.h5', 
                                       custom_objects={ 'accuracy' : tf.keras.metrics.MeanIoU(num_classes=4),
                                                       "dice_coef": dice_coef,
                                                       "precision": precision,
                                                       "sensitivity":sensitivity,
                                                       "specificity":specificity,
                                                       "dice_coef_necrotic": dice_coef_necrotic,
                                                       "dice_coef_edema": dice_coef_edema,
                                                       "dice_coef_enhancing": dice_coef_enhancing
                                                      }, compile=False)
    case_path=f"D:/my_GP/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{case}"
    files = next(os.walk(case_path))[2]
    X = np.empty((100, 128, 128, 2))
    
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_flair.nii');
    flair=nib.load(vol_path).get_fdata()
    
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_t1ce.nii');
    ce=nib.load(vol_path).get_fdata() 
    
    for j in range(100):
        X[j,:,:,0] = cv2.resize(flair[:,:,j+22], (128,128))
        X[j,:,:,1] = cv2.resize(ce[:,:,j+22], (128,128))
        
    return model.predict(X/np.max(X), verbose=1)
#p=predictByCase('003')

def Show3DTumor(case):
    p=predictByCase(case)
    data1 =p[:,:,:,1]
    data2= p[:,:,:,2]
    data3 =p[:,:,:,3]
    
    data = np.concatenate((data1, data2, data3), axis=1)
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(800, 800))
    mlab.contour3d(data, contours=6, transparent=True)
    vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(data), vmin=np.min(data), vmax=np.max(data))
    mlab.view(azimuth=180, elevation=180, distance='auto')
    label_names = {1: 'necrosis', 2: 'edema', 3: 'enhancing'}
    for i, label in enumerate([1,2,3]):
        mlab.text(0.7, 0.7 - i*0.25, label_names[label], width=0.2)
    fig.scene.save('plot.png')




    # Show the figure
    mlab.show()
#Show3DTumor(p)
        
def show3Dbrain(case):

    TrainDatasetPath = f'D:/my_GP/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{case}/'
    
    # Load the data and mask arrays from the .npy files
    data1= nib.load(TrainDatasetPath + f'BraTS20_Training_{case}_t2.nii').get_fdata()
    data2=nib.load(TrainDatasetPath + f'BraTS20_Training_{case}_t1.nii').get_fdata()
    data3=nib.load(TrainDatasetPath + f'BraTS20_Training_{case}_t1ce.nii').get_fdata()
    data4=nib.load(TrainDatasetPath + f'BraTS20_Training_{case}_flair.nii').get_fdata()
    
    # Create a new NIfTI image with the concatenated data
    mask=nib.load(TrainDatasetPath + f'BraTS20_Training_{case}_seg.nii').get_fdata()
    mask=mask.astype(np.uint8)
    
    # Crop the mask array to the same size as the data array
    mask_cropped = mask[:data1.size]
    
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(800, 800))
    
    # Create a volume visualization of the masked image data
    #vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(data1), vmin=np.min(data1), vmax=np.max(data1))
    vo3 = mlab.pipeline.volume(mlab.pipeline.scalar_field(data3), vmin=np.min(data3), vmax=np.max(data3))
    vo4 = mlab.pipeline.volume(mlab.pipeline.scalar_field(data4), vmin=np.min(data4), vmax=np.max(data4))
    #vo5 = mlab.pipeline.volume(mlab.pipeline.scalar_field(data2), vmin=np.min(data2), vmax=np.max(data2))
    
    vol2 = mlab.pipeline.volume(mlab.pipeline.scalar_field(mask_cropped), vmin=np.min(mask_cropped), vmax=np.max(mask_cropped))
    
    
    # Adjust the camera position and orientation
    mlab.view(azimuth=180, elevation=180, distance='auto')
    
    # Show the figure
    mlab.show()


#show3Dbrain('002')
VOLUME_START_AT=22
IMG_SIZE=128
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC', 
    2 : 'EDEMA',
    3 : 'ENHANCING' 
}

def showPredictsByCase(case, start_slice = 60):
    path = f"D:/my_GP/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{case}"
    gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
    origImage = nib.load(os.path.join(path, f'BraTS20_Training_{case}_flair.nii')).get_fdata()
    p = predictByCase(case)

    core = p[:,:,:,1]
    edema= p[:,:,:,2]
    enhancing = p[:,:,:,3]

    plt.figure(figsize=(18, 50))
    f, axarr = plt.subplots(1,6, figsize = (18, 50)) 

    for i in range(6): # for each image, add brain background
        axarr[i].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray", interpolation='none')
    
    axarr[0].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
    axarr[0].title.set_text('Original image flair')
    curr_gt=cv2.resize(gt[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_NEAREST)
    axarr[1].imshow(curr_gt,  interpolation='none', alpha=0.3) # ,alpha=0.3,cmap='Reds'
    axarr[1].title.set_text('Ground truth')
    axarr[2].imshow(p[start_slice,:,:,1:4],cmap="Reds", interpolation='none', alpha=0.3)
    axarr[2].title.set_text('all classes')
    axarr[3].imshow(edema[start_slice,:,:],  interpolation='none', alpha=0.3)
    axarr[3].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
    axarr[4].imshow(core[start_slice,:,:], interpolation='none', alpha=0.3)
    axarr[4].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
    axarr[5].imshow(enhancing[start_slice,:,:],  interpolation='none', alpha=0.3)
    axarr[5].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')

    filepath = f'seg/predictions_{case}.png'
    plt.savefig(filepath)

    # Close the plot to free up memory
    plt.close()
    return filepath

 
#a=showPredictsByCase('350')