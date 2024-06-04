# -*- coding: utf-8 -*-
"""
Created on Sat May 25 10:02:33 2024

@author: champ
"""

from copy import deepcopy
import numpy as np

import skimage.io as skio
from astropy.io import fits
import rawpy
import cv2

import numpy.random as npr
import astrocompyute.imagemath as imagemath

import matplotlib.pyplot as plt
import os

def ReadASI_TIF_FITS(asi_tif_fp, 
                     salt = True, 
                     scalarRGB = False, 
                     Red_Corr = 1.7566594340058168, 
                     Green_Corr = 1.0, 
                     Blue_Corr = 1.664086416513601):
    
    if asi_tif_fp[-3:] == 'tif':
        image_bayer = skio.imread(asi_tif_fp)
        image_rgb = cv2.cvtColor(image_bayer, 48) # 48=cv2.COLOR_BAYER_RG2BGR
        image_rgb = np.array(image_rgb, dtype = float) 
        
        
    elif (asi_tif_fp[-3:] == 'cr2') or (asi_tif_fp[-3:] == 'cr3'):
        
        rawIM16 = rawpy.imread(asi_tif_fp)
        image_rgb = rawIM16.postprocess(#gamma=(1,1), # default is (2.222, 4.5)
                                      no_auto_bright=True, 
                                      output_bps=16, 
                                      use_camera_wb  = True, 
                                      #use_auto_wb = True,
                                      exp_shift = 1.0,
                                      exp_preserve_highlights = 1.0)

        image_rgb = np.array(image_rgb, dtype = float)#/2**16
        image_bayer = None

    else:
        hdul = fits.open(asi_tif_fp)
        image_bayer = deepcopy(hdul[0].data)
        image_rgb = cv2.cvtColor(image_bayer, 48) # 48=cv2.COLOR_BAYER_RG2BGR
        image_rgb = np.array(image_rgb, dtype = float) 
        
    # image_rgb = cv2.cvtColor(image_bayer, 48) # 48=cv2.COLOR_BAYER_RG2BGR
    # image_rgb = np.array(image_rgb, dtype = float) 
    
    ## add salt
    if salt:
        image_rgb = (image_rgb + npr.rand(image_rgb.shape[0], image_rgb.shape[1], image_rgb.shape[2])) / 2**16
    else:
        image_rgb /= 2**16
        
    if scalarRGB:
        image_rgb[:,:,0] *= Red_Corr
        image_rgb[:,:,1] *= Green_Corr
        image_rgb[:,:,2] *= Blue_Corr
        image_rgb[image_rgb > 1.0] = 1.0
    
    return image_rgb, image_bayer






def SaveImage2Disk(SaveImage, 
                   OutputDir,
                   starless_version = None, 
                   description = '', 
                   save_token = ''):
    
    SaveImage[np.isnan(SaveImage)] = 0.0
    ## Create a 2x2 binned smaller resolution image
    SaveImage_2x2 = imagemath.DownsampleImage(SaveImage, 
                                           bin_size = 2, 
                                           resolution = (int(SaveImage.shape[0]/2), int(SaveImage.shape[1]/2)))
    

    ## create a deepcopy of the image to be saved so that we can clip the intensities
    out_im = deepcopy(SaveImage)
       
    out_im[out_im > ((2**16-1)/(2**16))] =  ((2**16-1)/(2**16))
    out_im[out_im <0] = 0
    
    ## create a deepcopy of the 2x2 image to be saved so that we can clip the intensities
    out_im_2x2 = deepcopy(SaveImage_2x2)
       
    out_im_2x2[out_im_2x2 > ((2**16-1)/(2**16))] =  ((2**16-1)/(2**16))
    out_im_2x2[out_im_2x2 <0] = 0
    
    if type(starless_version) != type(None):
        out_im_starless = deepcopy(starless_version)
        out_im_starless[out_im_starless > ((2**16-1)/(2**16))] =  ((2**16-1)/(2**16))
        out_im_starless[out_im_starless <0] = 0
    
    # save a full resolution png
    plt.imsave(os.path.join(OutputDir, description + '_'+save_token+'_plt.png'), out_im) 
    
    # save a 2x2 binned png
    plt.imsave(os.path.join(OutputDir, description + '_'+save_token+'_plt_2x2.png'), out_im_2x2) 
    
    # save the starless if provided (png)
    if type(starless_version) != type(None):        
        plt.imsave(os.path.join(OutputDir, description + '_'+save_token+'_plt_starless.png'), out_im_starless) 
        
    # save a full resolution jpg
    plt.imsave(os.path.join(OutputDir, description + '_'+save_token+'_plt.jpg'), out_im) 
    # save a 2x2 binned jpg
    plt.imsave(os.path.join(OutputDir, description + '_'+save_token+'_plt_2x2.jpg'), out_im_2x2) 
    
    os.path.join(OutputDir, )
    
    ## OpenCV can do a high quality 16-bit tif save.  But we need to swap some channels
    out_im_swap = deepcopy(out_im)
    out_im_swap[:,:,0] = deepcopy(out_im[:,:,2])
    out_im_swap[:,:,2] = deepcopy(out_im[:,:,0])
    print('ggg', out_im_swap.min(), out_im_swap.max())
    out_im_swap = np.array(np.round(out_im_swap*65536), dtype = np.uint16)
    print('ggg', out_im_swap.min(), out_im_swap.max())
    cv2.imwrite(os.path.join(OutputDir, description + '_'+save_token+'.tif'), out_im_swap)
    
    

    
    
    




































