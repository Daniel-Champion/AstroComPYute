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
from cupyx.scipy.signal import fftconvolve as cp_fftconvolve

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
    
    local_max_bool_cpu = cp.asnumpy(local_max_bool)
    
    del local_max_bool
    del gpu_image

    cp._default_memory_pool.free_all_blocks()
    return  local_max_bool_cpu


    
def cuda_maxFilter(n, cpu_univariate_image):
    
    gpu_image = cp.asarray(cpu_univariate_image)
    
    imgResult = cpxndimage.maximum_filter(gpu_image, size=n, footprint=None, output=None, mode='nearest', cval=0.0, origin=0)
        
    imgResult_cpu = cp.asnumpy(imgResult)
    
    del imgResult

    cp._default_memory_pool.free_all_blocks()
         
    return  imgResult_cpu

def cuda_maxFilter_disk(n, cpu_univariate_image):
    
    gpu_image = cp.asarray(cpu_univariate_image)
    
    star_footprint = np.array(skimage.morphology.disk(n), dtype = bool)
    
    imgResult = cpxndimage.maximum_filter(gpu_image, size=None, footprint=star_footprint, output=None, mode='nearest', cval=0.0, origin=0)
        
    imgResult_cpu = cp.asnumpy(imgResult)
    
    del imgResult

    cp._default_memory_pool.free_all_blocks()
         
    return  imgResult_cpu


def cuda_minFilter(n, cpu_univariate_image):
    
    gpu_image = cp.asarray(cpu_univariate_image)
    
    imgResult = cpxndimage.minimum_filter(gpu_image, size=n, footprint=None, output=None, mode='nearest', cval=0.0, origin=0)
        
    imgResult_cpu = cp.asnumpy(imgResult)
    
    del imgResult

    cp._default_memory_pool.free_all_blocks()
         
    return  imgResult_cpu
    
def cuda_uniformFilter(n, cpu_univariate_image):
    
    gpu_image = cp.asarray(cpu_univariate_image)
    
    imgResult = cpxndimage.uniform_filter(gpu_image, size=n, output=None, mode='nearest', cval=0.0, origin=0)
        
    imgResult_cpu = cp.asnumpy(imgResult)
    
    del imgResult

    cp._default_memory_pool.free_all_blocks()
         
    return  imgResult_cpu
    
def cuda_gaussianFilter(n, cpu_univariate_image):
    
    gpu_image = cp.asarray(cpu_univariate_image)
    
    imgResult = cpxndimage.gaussian_filter(gpu_image, n, order=0, output=None, mode='nearest', cval=0.0, truncate=3.0)
         
    imgResult_cpu = cp.asnumpy(imgResult)
    
    del imgResult

    cp._default_memory_pool.free_all_blocks()
         
    return  imgResult_cpu

def cuda_gaussianFilterRGB(n, cpu_RGB_image):
    
    gpu_image = cp.asarray(cpu_RGB_image)
    imgResult = cp.zeros(gpu_image.shape, dtype = gpu_image.dtype)
    imgResult[:,:,0] = cpxndimage.gaussian_filter(gpu_image[:,:,0], n, order=0, output=None, mode='nearest', cval=0.0, truncate=5.0)
    imgResult[:,:,1] = cpxndimage.gaussian_filter(gpu_image[:,:,1], n, order=0, output=None, mode='nearest', cval=0.0, truncate=5.0)
    imgResult[:,:,2] = cpxndimage.gaussian_filter(gpu_image[:,:,2], n, order=0, output=None, mode='nearest', cval=0.0, truncate=5.0)
    
    imgResult_cpu = cp.asnumpy(imgResult)
    
    del imgResult

    cp._default_memory_pool.free_all_blocks()
         
    return  imgResult_cpu



def sp_maxFilter(n, cpu_univariate_image):
    
    imgResult = sp.ndimage.maximum_filter(cpu_univariate_image, size=n, footprint=None, output=None, mode='nearest', cval=0.0, origin=0)
        
    imgResult_cpu = cp.asnumpy(imgResult)
    
    del imgResult

    cp._default_memory_pool.free_all_blocks()
    
    return imgResult_cpu

def cuda_rankfilt(nR, nC, cpu_univariate_image, rank):
    
    imgResult = cpxndimage.rank_filter(cp.asarray(cpu_univariate_image), rank , size = (nR,nC), mode = 'reflect')

    imgResult_cpu = cp.asnumpy(imgResult)
    
    del imgResult

    cp._default_memory_pool.free_all_blocks()

    return imgResult_cpu
    
    
def cuda_rankfilt_dual(long_dim, short_dim, cpu_univariate_image, rank):
    
    imgResult = cpxndimage.rank_filter(cp.asarray(cpu_univariate_image), rank , size = (long_dim,short_dim), mode = 'reflect')
    imgResult = cpxndimage.rank_filter(imgResult, rank , size = (short_dim,long_dim), mode = 'reflect')


    imgResult_cpu = cp.asnumpy(imgResult)
    
    del imgResult

    cp._default_memory_pool.free_all_blocks()
    
    
    return imgResult_cpu


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
    
    filt_image_cpu = cp.asnumpy(filt_image)[ks:ks+im_r, ks:ks+im_c]
    
    del filt_image
    del gpu_image

    cp._default_memory_pool.free_all_blocks()
    
    return  filt_image_cpu



def cuda_medfiltRGB(cpu_RGB, kernel_size, fix_zeros = False):
    
    
    RGB_filtered = np.stack([cuda_medfilt2d(cpu_RGB[:,:,0], kernel_size, fix_zeros = fix_zeros),
                               cuda_medfilt2d(cpu_RGB[:,:,1], kernel_size, fix_zeros = fix_zeros),
                               cuda_medfilt2d(cpu_RGB[:,:,2], kernel_size, fix_zeros = fix_zeros)], axis = 2)
    
    return RGB_filtered


#%%

def FFTUniformFilter(InputImage, footprint):
    convImage = sp.signal.fftconvolve(InputImage, footprint, mode='same')
    return convImage

def FootprintMedianFilter(InputImage, footprint):
    return sp.ndimage.median_filter(InputImage, size=None, footprint=footprint)

def FilterMSE(InputImage, footprint, FilteredImage):
    footN = footprint.sum()
    UI2 = FFTUniformFilter(InputImage**2, footprint)
    UI = FFTUniformFilter(InputImage, footprint)
    MSE_filt_F = (UI2 - 2.0*UI*FilteredImage + footN*FilteredImage**2)/footN
    return MSE_filt_F




def cuda_FFTUniformFilter(InputImageGPU, footprint):
    
    #print('debug1:', InputImageGPU.dtype, type(InputImageGPU))
    #print('debug2:', footprint.dtype, type(footprint))
    convImage = cp_fftconvolve(InputImageGPU, footprint, mode='same')
    return convImage

def cuda_FootprintMedianFilter(InputImageGPU, footprint):
    return cpxndimage.median_filter(InputImageGPU, size=None, footprint=footprint)

def cuda_FilterMSE(InputImageGPU, footprint, FilteredImageGPU):
    footN = footprint.sum()
    UI2 = cuda_FFTUniformFilter(InputImageGPU**2, footprint)
    UI = cuda_FFTUniformFilter(InputImageGPU, footprint)
    MSE_filt_F = (UI2 - 2.0*UI*FilteredImageGPU + footN*FilteredImageGPU**2)/footN
    return MSE_filt_F





def AdaptiveMedianFilter(InputImageRGB, footprint_length_scale = 4):
    
    ###########################################################################
    #### Step 1: Construct the filter footprint "Zoo".  These are all shapes
    #### and sizes of approximately the same size footprint 
    
    footN = footprint_length_scale

    diskFootprint = skimage.morphology.disk(footN)

    # ellipses 
    semiminor = int(round(footN/2))

    semimajor = semiminor

    while skimage.morphology.ellipse(semiminor, semimajor).sum() < diskFootprint.sum():
        semimajor += 1
        
    ellipseVert = skimage.morphology.ellipse(semiminor, semimajor)
    ellipseHoiz = skimage.morphology.ellipse(semimajor, semiminor)

    ellipseVertL45 = sp.ndimage.rotate(ellipseVert, 45, reshape=True)
    ellipseVertR45 = sp.ndimage.rotate(ellipseVert, -45, reshape=True)

    ## the footprint zoo consists of:
    ##    A.  5x disk-like footprints (some are shifted relative to the footprint origin)
    ##    B.  5x ellipse-like verticle orientation footprints (some are shifted relative to the footprint origin)
    ##    C.  5x ellipse-like horizontal orientation footprints (some are shifted relative to the footprint origin)
    ##    D.  6x +/-45-degree tilted ellipse footrpints (some are shifted relative to the footprint origin)
    
    FootprintZoo = [diskFootprint,                                              # centered 
                    np.vstack([diskFootprint, 0*diskFootprint[:2*footN-1,:]]),    # above
                    np.vstack([0*diskFootprint[:2*footN-1,:], diskFootprint]),    # below
                    np.hstack([diskFootprint, 0*diskFootprint[:, :2*footN-1]]),    # left
                    np.hstack([0*diskFootprint[:, :2*footN-1], diskFootprint]),    # right
                    ellipseVert,
                    np.vstack([ellipseVert, 0*ellipseVert[:2*semimajor-1,:]]),
                    np.vstack([0*ellipseVert[:2*semimajor-1,:], ellipseVert]),
                    np.hstack([ellipseVert, 0*ellipseVert[:,:2*semiminor-1]]),
                    np.hstack([0*ellipseVert[:,:2*semiminor-1], ellipseVert]),    
                    ellipseHoiz,
                    np.vstack([ellipseHoiz, 0*ellipseHoiz[:2*semiminor-1,:]]),
                    np.vstack([0*ellipseHoiz[:2*semiminor-1,:], ellipseHoiz]),
                    np.hstack([ellipseHoiz, 0*ellipseHoiz[:,:2*semimajor-1]]),
                    np.hstack([0*ellipseHoiz[:,:2*semimajor-1], ellipseHoiz]),  
                    ellipseVertL45,
                    np.pad(ellipseVertL45, ((0,ellipseVertR45.shape[0]-5),(0, ellipseVertR45.shape[0]-5))),
                    np.pad(ellipseVertL45, ((ellipseVertR45.shape[0]-5, 0),(ellipseVertR45.shape[0]-5, 0))),
                    ellipseVertR45,
                    np.pad(ellipseVertR45, ((0,ellipseVertR45.shape[0]-5),(ellipseVertR45.shape[0]-5, 0))),
                    np.pad(ellipseVertR45, ((ellipseVertR45.shape[0]-5,0),(0, ellipseVertR45.shape[0]-5))),
                    ]

    ###########################################################################
    #### Step 2: initialize input and output image data structures
    
    FootprintZooGPU = [cp.asarray(ftprint) for ftprint in FootprintZoo]


    InputImageRGB_gpu = cp.asarray(np.array(InputImageRGB, dtype = np.float32))
    
    BestMSE = cp.zeros(InputImageRGB.shape, dtype = np.float32) + 999999 

    AdaptiveFilterRGB = cp.zeros(InputImageRGB.shape, dtype = np.float32)
    
    ###########################################################################
    #### Step 3: Main loop over image channels and footprints in the zoo
    
    for chnl in range(3):
        for ifoot, _footprint in enumerate(FootprintZoo):
            
            #print('testing footprint:', ifoot, len(FootprintZoo))
            
            _footprint_gpu = FootprintZooGPU[ifoot]
            
            gFilterImage = cuda_FootprintMedianFilter(InputImageRGB_gpu[:,:,chnl], _footprint)
            
            gFilter_MSE_foot = cuda_FilterMSE(InputImageRGB_gpu[:,:,chnl], _footprint_gpu, gFilterImage)
            
            better_bool = gFilter_MSE_foot < BestMSE[:,:,chnl]
            
            AdaptiveFilterRGB[:,:,chnl][better_bool] = gFilterImage[better_bool]
            BestMSE[:,:,chnl][better_bool] = gFilter_MSE_foot[better_bool]
            
    return cp.asnumpy(AdaptiveFilterRGB)
    


def AdaptiveMedianFilter_LowMem(InputImageRGB, footprint_length_scale = 4):
    
    ###########################################################################
    #### Step 1: Construct the filter footprint "Zoo".  These are all shapes
    #### and sizes of approximately the same size footprint 
    
    footN = footprint_length_scale

    diskFootprint = skimage.morphology.disk(footN)

    # ellipses 
    semiminor = int(round(footN/2))

    semimajor = semiminor

    while skimage.morphology.ellipse(semiminor, semimajor).sum() < diskFootprint.sum():
        semimajor += 1
        
    ellipseVert = skimage.morphology.ellipse(semiminor, semimajor)
    ellipseHoiz = skimage.morphology.ellipse(semimajor, semiminor)

    ellipseVertL45 = sp.ndimage.rotate(ellipseVert, 45, reshape=True)
    ellipseVertR45 = sp.ndimage.rotate(ellipseVert, -45, reshape=True)

    ## the footprint zoo consists of:
    ##    A.  5x disk-like footprints (some are shifted relative to the footprint origin)
    ##    B.  5x ellipse-like verticle orientation footprints (some are shifted relative to the footprint origin)
    ##    C.  5x ellipse-like horizontal orientation footprints (some are shifted relative to the footprint origin)
    ##    D.  6x +/-45-degree tilted ellipse footrpints (some are shifted relative to the footprint origin)
    
    FootprintZoo = [diskFootprint,                                              # centered 
                    np.vstack([diskFootprint, 0*diskFootprint[:2*footN-1,:]]),    # above
                    np.vstack([0*diskFootprint[:2*footN-1,:], diskFootprint]),    # below
                    np.hstack([diskFootprint, 0*diskFootprint[:, :2*footN-1]]),    # left
                    np.hstack([0*diskFootprint[:, :2*footN-1], diskFootprint]),    # right
                    ellipseVert,
                    np.vstack([ellipseVert, 0*ellipseVert[:2*semimajor-1,:]]),
                    np.vstack([0*ellipseVert[:2*semimajor-1,:], ellipseVert]),
                    np.hstack([ellipseVert, 0*ellipseVert[:,:2*semiminor-1]]),
                    np.hstack([0*ellipseVert[:,:2*semiminor-1], ellipseVert]),    
                    ellipseHoiz,
                    np.vstack([ellipseHoiz, 0*ellipseHoiz[:2*semiminor-1,:]]),
                    np.vstack([0*ellipseHoiz[:2*semiminor-1,:], ellipseHoiz]),
                    np.hstack([ellipseHoiz, 0*ellipseHoiz[:,:2*semimajor-1]]),
                    np.hstack([0*ellipseHoiz[:,:2*semimajor-1], ellipseHoiz]),  
                    ellipseVertL45,
                    np.pad(ellipseVertL45, ((0,ellipseVertR45.shape[0]-5),(0, ellipseVertR45.shape[0]-5))),
                    np.pad(ellipseVertL45, ((ellipseVertR45.shape[0]-5, 0),(ellipseVertR45.shape[0]-5, 0))),
                    ellipseVertR45,
                    np.pad(ellipseVertR45, ((0,ellipseVertR45.shape[0]-5),(ellipseVertR45.shape[0]-5, 0))),
                    np.pad(ellipseVertR45, ((ellipseVertR45.shape[0]-5,0),(0, ellipseVertR45.shape[0]-5))),
                    ]

    ###########################################################################
    #### Step 2: initialize input and output image data structures
    
    FootprintZooGPU = [cp.asarray(ftprint) for ftprint in FootprintZoo]


    #InputImageRGB_gpu = cp.asarray(np.array(InputImageRGB, dtype = np.float32))
    gFilterImage_chnl = cp.zeros((InputImageRGB.shape[0], InputImageRGB.shape[1]), dtype = np.float32)
    InputImage_chnl = cp.zeros((InputImageRGB.shape[0], InputImageRGB.shape[1]), dtype = np.float32)
    UI2 = cp.zeros((InputImageRGB.shape[0], InputImageRGB.shape[1]), dtype = np.float32)
    UI = cp.zeros((InputImageRGB.shape[0], InputImageRGB.shape[1]), dtype = np.float32)
    gFilter_MSE_foot = cp.zeros((InputImageRGB.shape[0], InputImageRGB.shape[1]), dtype = np.float32)
    better_bool = cp.zeros((InputImageRGB.shape[0], InputImageRGB.shape[1]), dtype = bool)
    BestMSE = np.zeros(InputImageRGB.shape, dtype = np.float32) + 999999 

    AdaptiveFilterRGB = np.zeros(InputImageRGB.shape, dtype = np.float32)
    
    ###########################################################################
    #### Step 3: Main loop over image channels and footprints in the zoo
    
    for chnl in range(3):
        
        InputImage_chnl = cp.asarray(np.array(InputImageRGB[:,:,chnl], dtype = np.float32))
        
        for ifoot, _footprint in enumerate(FootprintZoo):
            
            #print('testing footprint:', ifoot, len(FootprintZoo))
            
            #gFilterImage = cuda_FootprintMedianFilter(InputImageRGB_gpu[:,:,chnl], _footprint)
            
            gFilterImage_chnl[:,:] = cpxndimage.median_filter(InputImage_chnl, size=None, footprint=FootprintZooGPU[ifoot])
            
            footN = FootprintZooGPU[ifoot].sum()

            UI2[:,:] = cp_fftconvolve(InputImage_chnl, FootprintZooGPU[ifoot], mode='same')
            UI[:,:] = cp_fftconvolve(InputImage_chnl, FootprintZooGPU[ifoot], mode='same')
            
            gFilter_MSE_foot = (UI2 - 2.0*UI*gFilterImage_chnl + footN*gFilterImage_chnl**2)/footN
            
            better_bool_cpu = cp.asnumpy(gFilter_MSE_foot) < BestMSE[:,:,chnl]
            #better_bool_cpu = cp.asnumpy(better_bool)
            
            AdaptiveFilterRGB[:,:,chnl][better_bool_cpu] = cp.asnumpy(gFilterImage_chnl)[better_bool_cpu]
            BestMSE[:,:,chnl][better_bool_cpu] = cp.asnumpy(gFilter_MSE_foot)[better_bool_cpu]
    
    
    del gFilter_MSE_foot
    del better_bool
    del UI
    del UI2
    del gFilterImage_chnl
    del InputImage_chnl
    del FootprintZooGPU
    
    
    

    cp._default_memory_pool.free_all_blocks()
    
    return cp.asnumpy(AdaptiveFilterRGB)
    






