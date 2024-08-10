# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:00:14 2024

@author: champ
"""
import os

from copy import deepcopy
import numpy as np
import scipy as sp
import cupy as cp
import cv2
import skimage.io as skio
import astrocompyute.color as mycolor
import astrocompyute.filters as myfilters
import astrocompyute.imagemath as imagemath
#L = 0.2126*R + 0.7152*G + 0.0722*B
from astrocompyute.visualization import ShowImage, ShowImageRGB, QuickInspectRGB, PlotHistRGB, ContourMultiPlot, MultiScatterPlot


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
    StretchData_cpu = cp.asnumpy(StretchData)

    if StretchChannel == 'Luminance':
        ImDataHLS[:,:,1] = StretchData_cpu
        StretchData_cpu = mycolor.HLS2RGB(ImDataHLS)
        
    elif StretchChannel == 'Value':
        ImDataHSV[:,:,2] = StretchData_cpu
        StretchData_cpu = mycolor.HSV2RGB(ImDataHSV)
        
    elif StretchChannel == 'Intensity':
        ImDataHSI[:,:,2] = StretchData_cpu
        StretchData_cpu = mycolor.HSI2RGB(ImDataHSI)
        
    else:
        pass

    del StretchData
    del ImData_gpu
    del StrBool
    
    cp._default_memory_pool.free_all_blocks()

    return StretchData_cpu



def solveBoostParamsLogLinear(linear_boost_val, log_transition_intesity, verbose = 0):
    
    def gamma_funk(gamma):
        
        return gamma / ((np.log((1.0 - log_transition_intesity / linear_boost_val)*gamma+1.0)) * (1.0 + log_transition_intesity / (1.0-log_transition_intesity))) - linear_boost_val
    

    sol = sp.optimize.root_scalar(gamma_funk, x0 = 1.0, x1 = 1.1, method = 'secant')

    
    gamma = sol.root
    
    xc = log_transition_intesity/linear_boost_val
    beta = log_transition_intesity * np.log( (1.0 - xc)*gamma + 1.0)/(1.0 - log_transition_intesity)
    
    def transform_funtion(input_intensity):
        
        t_bool = input_intensity > xc
        output_intensity = np.zeros(input_intensity.shape, input_intensity.dtype)
        output_intensity[t_bool] = (np.log((input_intensity[t_bool] - xc)*gamma + 1.0) + beta) / (np.log((1.0 - xc)*gamma+1.0) + beta)
        output_intensity[~t_bool] = linear_boost_val * input_intensity[~t_bool]
        return output_intensity
        
        
    if verbose > 0:
        X = np.linspace(0.0, 1.0, num = 1000)
        Y = transform_funtion(X)
        
        MultiScatterPlot([X,X], [X, Y], 
                         Labels = ['original', 'transformed'], 
                         XLabel = 'input intensity', 
                         YLabel = 'output intensity')
        
    return transform_funtion
        

def solveBoostParamsLog(linear_boost_val, verbose = 0):
    
    def simple_log_boost(gamma):
        return gamma / (np.log(gamma+1)) - linear_boost_val
    
    sol = sp.optimize.root_scalar(simple_log_boost, x0 = 1.0)
    gamma = sol.root
    
    def transform_funtion(input_intensity):
        return np.log(gamma*input_intensity + 1) / np.log(gamma+1)
        
    if verbose > 0:
        X = np.linspace(0.0, 1.0, num = 1000)
        Y = transform_funtion(X)
        
        MultiScatterPlot([X,X], [X, Y], 
                         Labels = ['original', 'transformed'], 
                         XLabel = 'input intensity', 
                         YLabel = 'output intensity')
    return transform_funtion
        
        
        

def SaturationValueBoost(imRGB, saturation_boost = 1.0, value_boost = 1.0, transition_int = 0.99, two_point = None, boost_method = 'linear', return_HSV = False):
    
    # Convert the image from BGR to HSV color space
    image = cv2.cvtColor(imRGB, cv2.COLOR_RGB2HSV)
    
    if saturation_boost != 1.0:
        #print('sat range:', np.percentile(image[:, :, 1], [0, 10, 20, 30, 40, 50, 60, 70, 80 ,90, 100]))
        
        if boost_method == 'linear':
            image[:, :, 1] = image[:, :, 1] * saturation_boost
            
        elif boost_method == 'loglinear':
            
            t_funk = solveBoostParamsLogLinear(saturation_boost, transition_int, verbose = 0)
            image[:, :, 1] = t_funk(image[:, :, 1])
            
        elif boost_method == 'log':
            t_funk = solveBoostParamsLog(saturation_boost, verbose = 1)
            image[:, :, 1] = t_funk(image[:, :, 1])
            

        else:
            image[:, :, 1] = image[:, :, 1] ** saturation_boost

    if value_boost != 1.0:
        #print('val range:', np.percentile(image[:, :, 2], [0, 10, 20, 30, 40, 50, 60, 70, 80 ,90, 100]))
        if boost_method == 'linear':
            image[:, :, 2] = image[:, :, 2] * value_boost
        elif boost_method == 'loglinear':
            
            t_funk = solveBoostParamsLogLinear(value_boost, transition_int, verbose = 0)
            image[:, :, 2] = t_funk(image[:, :, 2])
            
        elif boost_method == 'log':
            t_funk = solveBoostParamsLog(value_boost, verbose = 1)
            image[:, :, 2] = t_funk(image[:, :, 2])
            
        elif boost_method == 'two-point':
            
            iFun = sp.interpolate.interp1d([-9999, 0.0, two_point[0,0], two_point[1,0], 1.0, 9999], 
                                           [0.0, 0.0, two_point[0,1], two_point[1,1], 1.0, 1.0])
            
            image[:, :, 2] = iFun(image[:, :, 2])
            
        else: # gamma style adjustment
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
    shrink_stars_value[np.isnan(shrink_stars_value)] = 0.0
    
    if ReductionChannel == 'Luminance':
        starOnly_HLS[:,:,1] = shrink_stars_value
        StarReducedRGB = mycolor.HLS2RGB(starOnly_HLS)
        
    else:
        starOnly_HSV[:,:,2] = shrink_stars_value
        StarReducedRGB = mycolor.HSV2RGB(starOnly_HSV)        
        
    return StarReducedRGB



def GaussianStarReductionUnlinked(StarOnlyImageRGB, 
                          StarReductionRGBFactors = [3.0,2.0,3.0],
                          StarFilterDiam = 3):
    
    StarReducedRGB = np.zeros(StarOnlyImageRGB.shape, StarOnlyImageRGB.dtype)
    
    for chan in range(3):
        
        star_mags = myfilters.cuda_maxFilter_disk(StarFilterDiam, StarOnlyImageRGB[:,:,chan])
    
        shrink_stars_value = StarOnlyImageRGB[:,:,chan] ** StarReductionRGBFactors[chan]
        
        shrink_stars_value *= (star_mags / (star_mags ** StarReductionRGBFactors[chan]))
        shrink_stars_value[np.isnan(shrink_stars_value)] = 0.0
    
        StarReducedRGB[:,:,chan] = shrink_stars_value      
        
    return StarReducedRGB





#### Project sampled star colors onto the plane in RGB space that contains the 
#### astrometric stellar classification table data
def ForceStarColors(RGB_data, star_perp = np.array([ 0.3978311 , -0.66305184,  0.23206814])):
    
    proj_perp = star_perp[0] * RGB_data[:,0] + star_perp[1] * RGB_data[:,1] + star_perp[2] * RGB_data[:,2]

    proj_R_data = RGB_data[:,0] - proj_perp * star_perp[0]
    proj_G_data = RGB_data[:,1] - proj_perp * star_perp[1]
    proj_B_data = RGB_data[:,2] - proj_perp * star_perp[2]
    
    return np.vstack([proj_R_data, proj_G_data, proj_B_data]).T    


## Pre-compute a rastered array of airy disk data (expensive scipy function call)
dists_raster = np.linspace(-1, 100, num = 1000) + 0.0001
airy_raster = (2.0 * (sp.special.jn(1, dists_raster) / (dists_raster)))**2 
airy_interp = sp.interpolate.interp1d( dists_raster, airy_raster)

def airy_disk(dists, length_scale):
    aa = 2.0 / length_scale
    return (2.0 * (sp.special.jn(1, dists * aa) / (dists*aa)))**2

def fast_airy_disk(dists, length_scale):
    aa = 2.0 / length_scale
    
    return airy_interp(dists * aa)

def superfast_airy_disk(dists, length_scale):
    SI = np.searchsorted(dists_raster, (2.0 / length_scale)*dists)
    return airy_raster[SI]



def GenSyntheticStars(RGB_image_with_stars, 
                      star_waist_radius = 2.0 ,
                      max_sigmage = 8.0 , 
                      star_stretch_exponent = 4,
                      verbose = 2,
                      starnoise = 0.5):
    
    final_int_image = RGB_image_with_stars.mean(axis = 2)

    #### Locate star indices within the image
    [StarRowIndices, 
     StarColIndices, 
     Star_peak_mags, 
     Star_prominances, 
     Star_floors, 
     Star_flux_proportion] = imagemath.cuda_StarLocations(final_int_image, 
                                                     analysis_blur_sigma = 1,
                                                     local_max_window = 3, 
                                                     background_window = 11, 
                                                     minimum_prominance = 10/2**16)
                                                          
    if verbose > 1:
        #### Generate a plot with a magenta pixel at each star location to 
        #### verify star location performance
        StarReplace = np.stack([final_int_image, final_int_image, final_int_image], axis = 2)
        p_low, p_high = np.percentile(final_int_image, [5, 95])
        StarReplace -= p_low
        StarReplace /= (p_high - p_low)
        StarReplace[StarReplace <0] = 0
        StarReplace[StarReplace > 1.0] = 1.0
        #StarReplace = deepcopy(RGB_image_with_stars)
    
        StarReplace[StarRowIndices, StarColIndices, :] = np.array([1.0, 0.0, 1.0])
        
        ShowImageRGB(StarReplace)
        
        
    # Gen pixel coordinates for the whole image
    ImageRowsF, ImageColsF = imagemath.GetImageCoords(AnalysisImage = final_int_image)
    
    # estimate sub-pixel resolution star centroids
    starCentWeights = myfilters.cuda_uniformFilter(2, final_int_image)                                                       
    starCenWeightSumRows =  myfilters.cuda_uniformFilter(2, final_int_image*ImageRowsF)
    starCenWeightSumCols =  myfilters.cuda_uniformFilter(2, final_int_image*ImageColsF)
                                              
    star_centroids = np.vstack([(starCenWeightSumRows/starCentWeights)[StarRowIndices, StarColIndices], 
                                (starCenWeightSumCols/starCentWeights)[StarRowIndices, StarColIndices]]).T
    
    
    ## sample the apparent star colors using a narrow uniform filter
    starColorR = myfilters.cuda_uniformFilter(2, RGB_image_with_stars[:,:,0])
    starColorG = myfilters.cuda_uniformFilter(2, RGB_image_with_stars[:,:,1])
    starColorB = myfilters.cuda_uniformFilter(2, RGB_image_with_stars[:,:,2])
        
    star_colors = np.vstack([starColorR[StarRowIndices, StarColIndices],
                             starColorG[StarRowIndices, StarColIndices],
                             starColorB[StarRowIndices, StarColIndices]]).T                                                    
    ## project the apparent star colors to the 2D plane in RGB space containing 
    ## actual star colors
    star_colors = ForceStarColors(star_colors)
    

    ## pre compute the zero-based set of indicies that will contain the airy disk 
    ## data for each star
    psf_r, psf_c = imagemath.KernelCenterCoords(int(round(max_sigmage*star_waist_radius)) + 1)    
    
    psf_b = (psf_r**2 + psf_c**2) <= (int(round(max_sigmage*star_waist_radius)) + 1)**2
    
    psf_r_1d = psf_r[psf_b]
    psf_c_1d = psf_c[psf_b]
    
    ## initialize the star-only image
    synth_star_only = np.zeros(RGB_image_with_stars.shape, RGB_image_with_stars.dtype)

    ## main loop over stars to populate airy disks
    for istar in range(len(star_centroids)):
        
        int_r = int(round(star_centroids[istar, 0]))
        int_c = int(round(star_centroids[istar, 1]))
        
        # shift the zero-based indices to absolute star indicies
        mod_ind_r = psf_r_1d + int_r
        mod_ind_c = psf_c_1d + int_c
        
        # compute the pixel distances to the star centroid
        pix_radial_dists = np.sqrt(( mod_ind_r - star_centroids[istar, 0]  )**2 + (mod_ind_c - star_centroids[istar, 1])**2)
        
        # add some random noise to the distances
        pix_radial_dists += starnoise * np.random.randn(len(pix_radial_dists))
        
        ## evaluate airy disk function
        #I_data = airy_disk(np.sqrt(( mod_ind_r - star_centroids[istar, 0]  )**2 + (mod_ind_c - star_centroids[istar, 1])**2), star_waist_radius)
        #I_data = fast_airy_disk(np.sqrt(( mod_ind_r - star_centroids[istar, 0]  )**2 + (mod_ind_c - star_centroids[istar, 1])**2), star_waist_radius)
        I_data = superfast_airy_disk(pix_radial_dists, star_waist_radius)
    
        # determine if any indices are outside of the image bounds
        mod_bool = (mod_ind_r >=0) & (mod_ind_r < ImageRowsF.shape[0]) & (mod_ind_c >=0) & (mod_ind_c < ImageRowsF.shape[1])
    
        ## accumulate star airy disk data
        for chan in range(3):
            synth_star_only[mod_ind_r[mod_bool], mod_ind_c[mod_bool], chan] += star_colors[istar, chan] * I_data[mod_bool]
    
    ## perform a simple log stretch of the star only data
    stretched_star_only =    np.array( np.log2(1.0 + 2**star_stretch_exponent*synth_star_only)/star_stretch_exponent, np.float32)  
    
    ### you can stretch the star-only image to improve the star rendering
    # ssN = 4                                                 
    # stretched_star_only =    np.array( np.log2(1.0 + 2**ssN*synth_star_only)/ssN, np.float32)  
    # stretched_star_only =  SaturationValueBoost(stretched_star_only, saturation_boost = 1.3, value_boost = 1.5, boost_method = 'linear')                
    
    return synth_star_only, stretched_star_only



















