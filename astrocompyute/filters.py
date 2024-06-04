# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 17:46:11 2023

@author: champ
"""

import time

import cv2

import numpy as np
import numpy.random as npr

import scipy as sp

import skimage

import cupy as cp
import cupyx as cpx
from cupyx.scipy import ndimage as cpxndimage
from cupyx.scipy.signal import medfilt2d as cp_medfilt2d


def maximumBoxFilter(n, img):
  
    # Creates the shape of the kernel
    size = (n,n)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)
    
    # Applies the maximum filter with kernel NxN
    imgResult = cv2.dilate(img, kernel)

    return imgResult

def maximumBoxFilterRC(nr, nc, img):
  
    # Creates the shape of the kernel
    size = (nr,nc)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)
    
    # Applies the maximum filter with kernel NxN
    imgResult = cv2.dilate(img, kernel)

    return imgResult

def maximumDiskFilter(n, img):
  
    # Creates the shape of the kernel
    size = (n,n)
    shape = cv2.MORPH_ELLIPSE
    kernel = cv2.getStructuringElement(shape, size)
    
    # Applies the maximum filter with kernel NxN
    imgResult = cv2.dilate(img, kernel)

    return imgResult

def maximumDiskFilterRC(nr, nc, img):
  
    # Creates the shape of the kernel
    size = (nr,nc)
    shape = cv2.MORPH_ELLIPSE
    kernel = cv2.getStructuringElement(shape, size)
    
    # Applies the maximum filter with kernel NxN
    imgResult = cv2.dilate(img, kernel)

    return imgResult


def minimumBoxFilter(n, img):
    # Creates the shape of the kernel
    size = (n, n)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    # Applies the minimum filter with kernel NxN
    imgResult = cv2.erode(img, kernel)

    return imgResult

  
def RollMax(n, img):
    
    local_max_bool = np.ones(img.shape, dtype = bool)
    
    for row_shift in range(-n, n+1, 1 ):
        for col_shift in range(-n, n+1, 1 ):
            
            local_max_bool &= (img >= np.roll(img, (row_shift, col_shift), axis = (0,1)))
            
    return local_max_bool
            
            
            
def cuda_RollMax(n, cpu_univariate_image):
    
    gpu_image = cp.asarray(cpu_univariate_image)
    
    local_max_bool = cp.ones(cpu_univariate_image.shape, dtype = bool)
    
    for row_shift in range(-n, n+1, 1 ):
        for col_shift in range(-n, n+1, 1 ):
            
            local_max_bool &= (gpu_image >= cp.roll(gpu_image, (row_shift, col_shift), axis = (0,1)))
    
    return  cp.asnumpy(local_max_bool)


    
def cuda_maxFilter(n, cpu_univariate_image):
    
    gpu_image = cp.asarray(cpu_univariate_image)
    
    imgResult = cpxndimage.maximum_filter(gpu_image, size=n, footprint=None, output=None, mode='nearest', cval=0.0, origin=0)
        
    return  cp.asnumpy(imgResult)

def cuda_maxFilter_disk(n, cpu_univariate_image):
    
    gpu_image = cp.asarray(cpu_univariate_image)
    
    star_footprint = np.array(skimage.morphology.disk(n), dtype = bool)
    
    imgResult = cpxndimage.maximum_filter(gpu_image, size=None, footprint=star_footprint, output=None, mode='nearest', cval=0.0, origin=0)
        
    return  cp.asnumpy(imgResult)


def cuda_minFilter(n, cpu_univariate_image):
    
    gpu_image = cp.asarray(cpu_univariate_image)
    
    imgResult = cpxndimage.minimum_filter(gpu_image, size=n, footprint=None, output=None, mode='nearest', cval=0.0, origin=0)
        
    return  cp.asnumpy(imgResult)
    
def cuda_uniformFilter(n, cpu_univariate_image):
    
    gpu_image = cp.asarray(cpu_univariate_image)
    
    imgResult = cpxndimage.uniform_filter(gpu_image, size=n, output=None, mode='nearest', cval=0.0, origin=0)
        
    return  cp.asnumpy(imgResult)
    
def cuda_gaussianFilter(n, cpu_univariate_image):
    
    gpu_image = cp.asarray(cpu_univariate_image)
    
    imgResult = cpxndimage.gaussian_filter(gpu_image, n, order=0, output=None, mode='nearest', cval=0.0, truncate=3.0)
         
    return  cp.asnumpy(imgResult)

def sp_maxFilter(n, cpu_univariate_image):
    
    imgResult = sp.ndimage.maximum_filter(cpu_univariate_image, size=n, footprint=None, output=None, mode='nearest', cval=0.0, origin=0)
        
    return imgResult

def cuda_rankfilt(nR, nC, cpu_univariate_image, rank):
    
    imgResult = cpxndimage.rank_filter(cp.asarray(cpu_univariate_image), rank , size = (nR,nC), mode = 'reflect')

    return imgResult
    
    
def cuda_rankfilt_dual(long_dim, short_dim, cpu_univariate_image, rank):
    
    imgResult = cpxndimage.rank_filter(cp.asarray(cpu_univariate_image), rank , size = (long_dim,short_dim), mode = 'reflect')
    imgResult = cpxndimage.rank_filter(imgResult, rank , size = (short_dim,long_dim), mode = 'reflect')

    return cp.asnumpy(imgResult)


def cuda_medfilt2d(cpu_univariate_image, kernel_size, fix_zeros = False):
    ks = kernel_size
    im_r, im_c =  cpu_univariate_image.shape
    
    padded_image = np.zeros((im_r+2*ks, im_c+2*ks), dtype = cpu_univariate_image.dtype)
    
    # insert the image data
    padded_image[ks:ks+im_r, ks:ks+im_c] = cpu_univariate_image
    
    # top
    padded_image[:ks, ks:ks+im_c] = cpu_univariate_image[:ks][::-1]
    
    # bottom
    padded_image[-ks:, ks:ks+im_c] = cpu_univariate_image[-ks:][::-1]
    
    # left
    padded_image[ks:ks+im_r, :ks] = cpu_univariate_image[:,:ks][:, ::-1]
    
    # right
    padded_image[ks:ks+im_r, -ks:] = cpu_univariate_image[:,-ks:][:, ::-1]
    
    # upper left
    padded_image[:ks,:ks] = cpu_univariate_image[:ks,:ks][::-1, ::-1]
    
    # upper right
    padded_image[:ks,-ks:] = cpu_univariate_image[:ks,-ks:][::-1, ::-1]
    
    # lower left
    padded_image[-ks:,:ks] = cpu_univariate_image[-ks:,:ks][::-1, ::-1]
    
    # lower right
    padded_image[-ks:,-ks:] = cpu_univariate_image[-ks:,-ks:][::-1, ::-1]
    
    gpu_image = cp.asarray(padded_image)
    filt_image = cp_medfilt2d(gpu_image, kernel_size = kernel_size)
     
    return  cp.asnumpy(filt_image)[ks:ks+im_r, ks:ks+im_c]



def cuda_medfiltRGB(cpu_RGB, kernel_size, fix_zeros = False):
    
    
    RGB_filtered = np.stack([cuda_medfilt2d(cpu_RGB[:,:,0], kernel_size, fix_zeros = fix_zeros),
                               cuda_medfilt2d(cpu_RGB[:,:,1], kernel_size, fix_zeros = fix_zeros),
                               cuda_medfilt2d(cpu_RGB[:,:,2], kernel_size, fix_zeros = fix_zeros)], axis = 2)
    
    return RGB_filtered