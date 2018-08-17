import glob
import keras
import scipy
import nibabel as nib
import numpy as np
import os.path
import random
import imageio
import re
import tensorflow as tf
import keras.backend as K

from config import *

# List avaiable sequences
def prepare_data(directory):
    return glob.glob(directory)
       
def open(path):
    image = scipy.misc.imread(path).astype(np.float)
    subimages = np.split(image / 1000.0, input + output, axis=1)
    return [np.stack(augment(subimages[output:]), axis=-1) * 2 - 1, np.stack(subimages[:output], axis=-1)]

def selective_crossentropy(y_true, y_pred):
    _epsilon=tf.convert_to_tensor(K.epsilon(), dtype=y_pred.dtype.base_dtype)
    y_pred=tf.clip_by_value(y_pred, _epsilon, 1.- _epsilon)
    return - tf.reduce_sum(y_true * tf.log(y_pred), len(y_pred.get_shape())-1)
    
# Load image sequences
def load(patient, size):

    seg0 = nib.load(patient.replace("t1.nii.gz", "seg0.nii.gz")).get_data().astype(np.float)
    seg0 = np.pad(seg0, ((8, 8), (8, 8), (2, 3)), 'constant')
    
    seg1 = nib.load(patient.replace("t1.nii.gz", "seg1.nii.gz")).get_data().astype(np.float)
    seg1 = np.pad(seg1, ((8, 8), (8, 8), (2, 3)), 'constant')
    
    seg2 = nib.load(patient.replace("t1.nii.gz", "seg2.nii.gz")).get_data().astype(np.float)
    seg2 = np.pad(seg2, ((8, 8), (8, 8), (2, 3)), 'constant')
    
    seg3 = nib.load(patient.replace("t1.nii.gz", "seg3.nii.gz")).get_data().astype(np.float)
    seg3 = np.pad(seg3, ((8, 8), (8, 8), (2, 3)), 'constant')
    
    t1 = nib.load(patient).get_data().astype(np.float)
    t1 = np.pad(t1, ((8, 8), (8, 8), (2, 3)), 'constant')
    
    t1ce = nib.load(patient.replace("t1.nii.gz", "t1ce.nii.gz")).get_data().astype(np.float)
    t1ce = np.pad(t1ce, ((8, 8), (8, 8), (2, 3)), 'constant')
    
    t2 = nib.load(patient.replace("t1.nii.gz", "t2.nii.gz")).get_data().astype(np.float)
    t2 = np.pad(t2, ((8, 8), (8, 8), (2, 3)), 'constant')
    
    flair = nib.load(patient.replace("t1.nii.gz", "flair.nii.gz")).get_data().astype(np.float)
    flair = np.pad(flair, ((8, 8), (8, 8), (2, 3)), 'constant')

    combined = np.stack((t1, t1ce, t2, flair), -1)
    
    seg = np.stack((seg0, seg1, seg2, seg3), -1)
    
    x = []
    y = []
    ind = []
    
    for i in range(256 // size):
        x_begin = i * size
        x_end = x_begin + size
        for j in range(256 // size):
            y_begin = j * size
            y_end = y_begin + size
            for k in range(160 // size):
                z_begin = k * size
                z_end = z_begin + size
                x.append(combined[np.newaxis, x_begin:x_end, y_begin:y_end, z_begin:z_end])
                y.append(seg[np.newaxis, x_begin:x_end, y_begin:y_end, z_begin:z_end])
                ind.append([x_begin, x_end, y_begin, y_end, z_begin, z_end])
    
    #fxs = np.split(combined, 8, 0)
    #fys = np.split(seg, 8, 0)
    #for i in range(len(fxs)):
    #    sxs = np.split(fxs[i], 8, 1)
    #    sys = np.split(fys[i], 8, 1)
    #    for j in range(len(sxs)):
    #        txs = np.split(sxs[j], 5, 2)
    #        tys = np.split(sys[j], 5, 2)
    #        for _X in txs:
    #            x.append(_X[np.newaxis, :,:,:,:])
    #        for _Y in tys:
    #            y.append(_Y[np.newaxis, :,:,:,np.newaxis])
        
    return x, y, ind
    
def load2(patient, size):
    
    t1 = nib.load(patient).get_data().astype(np.float)
    t1 = np.pad(t1, ((8, 8), (8, 8), (2, 3)), 'constant')
    
    t1ce = nib.load(patient.replace("t1.nii.gz", "t1ce.nii.gz")).get_data().astype(np.float)
    t1ce = np.pad(t1ce, ((8, 8), (8, 8), (2, 3)), 'constant')
    
    t2 = nib.load(patient.replace("t1.nii.gz", "t2.nii.gz")).get_data().astype(np.float)
    t2 = np.pad(t2, ((8, 8), (8, 8), (2, 3)), 'constant')
    
    flair = nib.load(patient.replace("t1.nii.gz", "flair.nii.gz")).get_data().astype(np.float)
    flair = np.pad(flair, ((8, 8), (8, 8), (2, 3)), 'constant')

    combined = np.stack((t1, t1ce, t2, flair), -1)
    
    x = []
    ind = []
    
    for i in range(256 // size):
        x_begin = i * size
        x_end = x_begin + size
        for j in range(256 // size):
            y_begin = j * size
            y_end = y_begin + size
            for k in range(160 // size):
                z_begin = k * size
                z_end = z_begin + size
                x.append(combined[np.newaxis, x_begin:x_end, y_begin:y_end, z_begin:z_end])
                ind.append([x_begin, x_end, y_begin, y_end, z_begin, z_end])
        
    return x, ind
            
def store(patient, y, idx):
    
    full = np.ones((256, 256, 160))
    
    for i in range(len(idx)):
        p = idx[i]
        #full[p[0]:p[1], p[2]:p[3], p[4]:p[5]] = np.squeeze(y[i], axis=-1)
        full[p[0]:p[1], p[2]:p[3], p[4]:p[5]] = np.argmax(np.squeeze(y[i]), axis=-1)

    #out = np.clip(np.sign(full[8:-8, 8:-8, 2:-3]) + 1, 0, 1).astype(np.int16)
    out=full[8:-8, 8:-8, 2:-3].astype(np.int16)
    nib.save(nib.Nifti1Image(out, None), patient.replace("t1.nii.gz", "out.nii.gz"))
            
def augment(sequence):
    return [apply_contrast(apply_gaussian_noise(s)) for s in sequence]
    
def resize(image):
    offset = 8

    h1 = int(np.ceil(np.random.uniform(1e-2, offset)))
    w1 = int(np.ceil(np.random.uniform(1e-2, offset)))
    
    out = np.zeros((size + 2*offset, size + 2*offset))
    out[offset:offset+size, offset:offset+size] = image
    return out[h1:h1+size, w1:w1+size]
    
def apply_contrast(image):
    # Apply random brightness but keep values in [0, 1]
    # We apply a quadratic function with the form y = ax^2 + bx
    # Visualization: https://www.desmos.com/calculator/zzz75gguna
    delta = random.uniform(-0.04, 0.04)
    a = -4 * delta
    b = 1 - a
    return a * (image*image) + b * (image)
    
def apply_gaussian_noise(image):
    # Apply gaussian noise but keep values in [0, 1]
    random_value = random.uniform(-0.01, 0.01)
    return np.clip(image + (random_value), 0.0, 1.0)

def strip(arr):
    return arr[:, sequence_crop:-sequence_crop]
    
def re_shape(arr):
    return np.reshape(arr, (1, sequence_total, size, size, output))
    
def save_image(input, gt, generated, path):
    all = np.concatenate((input, gt, generated), axis=4)
    all = np.squeeze(all)
    all = np.squeeze(np.concatenate(np.split(all, 4, axis=0), axis=1))
    all = np.squeeze(np.concatenate(np.split(all, 12, axis=2), axis=1))
    imageio.imwrite(path, (np.clip(all, 0.0, 1.0) * 255).astype(np.uint8))
    
def convert_image(image, size, border, rotation):
    image = np.rot90(image, 4 - rotation)
    
    temp = np.ones((size, size)) * 0.5
    temp[border[0] : border[1], border[2] : border[3]] = image
    
    image = np.reshape(temp, (size, size, 1))
    image = np.concatenate([image, image, image], axis=2)
    return (image * 255).astype(np.uint8)
