# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:00:14 2024

@author: champ
"""
import os

from copy import deepcopy
import numpy as np
import cupy as cp
import cv2
import skimage.io as skio
import astrocompyute.color as mycolor
import astrocompyute.filters as myfilters
#L = 0.2126*R + 0.7152*G + 0.0722*B


def CustomPseudoLogStretch(ImDataRGB, BP_S = 0.0, BP_T = 0.01, ExpN = 16, zz = 0.5, StretchChannel = ''):
    
    if StretchChannel == 'Luminance':
        
        ImDataHLS = mycolor.RGB2HLS(ImDataRGB)
        ImData = ImDataHLS[:,:,1]
        
    elif StretchChannel == 'Value':
        ImDataHSV = mycolor.RGB2HSV(ImDataRGB)
        ImData = ImDataHSV[:,:,2]
        
    elif StretchChannel == 'Intensity':
        ImDataHSI = mycolor.RGB2HSI(ImDataRGB)
        ImData = ImDataHSI[:,:,2]
        
    else: # unlinked stretch
        ImData = ImDataRGB
        
    StretchData = np.zeros(ImData.shape, ImData.dtype)
    
    StrBool = ImData > BP_S
    
    StretchData[StrBool] = np.log2(2**ExpN * (ImData[StrBool] - BP_S) + 1.0)/ExpN + BP_T

    aa = BP_T**(1.0/zz)
    bb = (1.0/(zz*BP_T))*(2**ExpN / (ExpN * np.log(2)))

    StretchData[~StrBool] = (-aa / (bb*(ImData[~StrBool] - BP_S) - 1))**zz
    
    if StretchChannel == 'Luminance':
        ImDataHLS[:,:,1] = StretchData
        StretchData = mycolor.HLS2RGB(ImDataHLS)
        
    elif StretchChannel == 'Value':
        ImDataHSV[:,:,2] = StretchData
        StretchData = mycolor.HSV2RGB(ImDataHSV)
        
    elif StretchChannel == 'Intensity':
        ImDataHSI[:,:,2] = StretchData
        StretchData = mycolor.HSI2RGB(ImDataHSI)
        
    else:
        pass

    return StretchData



def CustomPseudoLogStretch_cuda(ImDataRGB, BP_S = 0.0, BP_T = 0.01, ExpN = 16, zz = 0.5, StretchChannel = ''):
    
    if StretchChannel == 'Luminance':
        
        ImDataHLS = mycolor.RGB2HLS(ImDataRGB)
        ImData = ImDataHLS[:,:,1]
        
    elif StretchChannel == 'Value':
        ImDataHSV = mycolor.RGB2HSV(ImDataRGB)
        ImData = ImDataHSV[:,:,2]
        
    elif StretchChannel == 'Intensity':
        ImDataHSI = mycolor.RGB2HSI(ImDataRGB)
        ImData = ImDataHSI[:,:,2]
        
    else: # unlinked stretch
        ImData = ImDataRGB  
    
    StretchData = cp.zeros(ImData.shape, ImData.dtype)
    
    ImData_gpu = cp.asarray(ImData)
    
    StrBool = ImData_gpu > BP_S
    
    StretchData[StrBool] = cp.log2(2**ExpN * (ImData_gpu[StrBool] - BP_S) + 1.0)/ExpN + BP_T

    aa = BP_T**(1.0/zz)
    bb = (1.0/(zz*BP_T))*(2**ExpN / (ExpN * np.log(2)))

    StretchData[~StrBool] = (-aa / (bb*(ImData_gpu[~StrBool] - BP_S) - 1))**zz
    StretchData = cp.asnumpy(StretchData)

    if StretchChannel == 'Luminance':
        ImDataHLS[:,:,1] = StretchData
        StretchData = mycolor.HLS2RGB(ImDataHLS)
        
    elif StretchChannel == 'Value':
        ImDataHSV[:,:,2] = StretchData
        StretchData = mycolor.HSV2RGB(ImDataHSV)
        
    elif StretchChannel == 'Intensity':
        ImDataHSI[:,:,2] = StretchData
        StretchData = mycolor.HSI2RGB(ImDataHSI)
        
    else:
        pass

    return StretchData






def SaturationValueBoost(imRGB, saturation_boost = 1.0, value_boost = 1.0, boost_method = 'linear', return_HSV = False):
    
    # Convert the image from BGR to HSV color space
    image = cv2.cvtColor(imRGB, cv2.COLOR_RGB2HSV)
    
    if saturation_boost != 1.0:
        if boost_method == 'linear':
            image[:, :, 1] = image[:, :, 1] * saturation_boost
        else:
            image[:, :, 1] = image[:, :, 1] ** saturation_boost

    if value_boost != 1.0:
        if boost_method == 'linear':
            image[:, :, 2] = image[:, :, 2] * value_boost
        else:
            image[:, :, 2] = image[:, :, 2] ** value_boost
        
    # Convert the image back to RGB color space
    imageRGB = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    
    if return_HSV:
        return imageRGB, image
    else:
        return imageRGB
    

   
def StarnetPP(stretched_image_RGB, starnet_pp_path = 'C:/Users/champ/Astronomy/Software/StarNetv2CLI_Win/StarNetv2CLI_Win/'):
    
    temp_proc_fp = starnet_pp_path + 'inputRGB.tif'
    temp_proc_fp__starless = starnet_pp_path + 'outputStarless.tif'
    
    
    out_im = deepcopy(stretched_image_RGB)
    
    out_im[out_im > ((2**16-1)/(2**16))] =  ((2**16-1)/(2**16))
    out_im[out_im <0] = 0
    
    
    out_im_swap = deepcopy(out_im)
    
    out_im_swap[:,:,0] = deepcopy(out_im[:,:,2])
    out_im_swap[:,:,2] = deepcopy(out_im[:,:,0])

    cv2.imwrite(temp_proc_fp, np.array(out_im_swap*2**16, dtype = np.uint16))
    
    starnet_cmd = starnet_pp_path + 'starnet++.exe' 
    starnet_cmd += ' ' + temp_proc_fp + ' ' + temp_proc_fp__starless

    cur_pat = os.getcwd()
    os.chdir(starnet_pp_path)
    os.system(starnet_cmd)
    os.chdir(cur_pat)
    
    starless_RGB = skio.imread(temp_proc_fp__starless)/2**16
        
    return starless_RGB


def GAddImages(ImA, ImB):
    
    return 1.0 - (1.0 - ImA)*(1.0 - ImB)

    
    
def GSubtractImages(ImA, ImB):
    
    return 1.0 - (1.0 - ImA)/(1.0 - ImB)

    
def GaussianStarReduction(StarOnlyImageRGB, 
                          StarReductionFactor = 2.0,
                          ReductionChannel = 'Luminance',
                          StarFilterDiam = 3):
    
    
    if ReductionChannel == 'Luminance':
        starOnly_HLS = mycolor.RGB2HLS(StarOnlyImageRGB)
        starOnly_X = starOnly_HLS[:,:,1]
        
    else:
        starOnly_HSV = mycolor.RGB2HSV(StarOnlyImageRGB)
        starOnly_X = starOnly_HSV[:,:,2]
        
    star_mags = myfilters.cuda_maxFilter_disk(StarFilterDiam, starOnly_X)

    shrink_stars_value = starOnly_X ** StarReductionFactor
    
    shrink_stars_value *= (star_mags / (star_mags ** StarReductionFactor))
    
    if ReductionChannel == 'Luminance':
        starOnly_HLS[:,:,1] = shrink_stars_value
        StarReducedRGB = mycolor.HLS2RGB(starOnly_HLS)
        
    else:
        starOnly_HSV[:,:,2] = shrink_stars_value
        StarReducedRGB = mycolor.HSV2RGB(starOnly_HSV)        
        
    return StarReducedRGB




    





































