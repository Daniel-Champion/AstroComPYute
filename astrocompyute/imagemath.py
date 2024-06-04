# -*- coding: utf-8 -*-
"""
Created on Sat May 25 10:59:22 2024

@author: champ
"""


import os

from copy import deepcopy


import math

import numpy as np
import scipy as sp
from scipy import signal
import skimage.filters as filters
import skimage.morphology as morph
import skimage.io as skio
from skimage.measure import block_reduce
import skimage
import cv2

import cupy as cp
import cupyx as cpx


from cupyx.scipy import ndimage as cpxndimage


"""
Image Calculations
    KernelCenterCoords
    HannKernel (PSF)
    StarLocations    
    StarnetPP ???
    StarRadialInfluence
    StarMask
    NeightborhoodStats
    GetImageCoords
    SyntheticFlatField
    
    AffineGaussian
    AffineGaussianLinear
    AffineGaussianLinearBi
    GaussianBi
    GaussianBiEdge
    PointLocalBivGaussianRegression
    PointLocalBivGaussianRegression_zero
    PointLocalGaussianRegression
    StarExtraction
"""
    
    
def DownsampleImage(full_image,  bin_size = 3, resolution = (1200, 1920)):
    print('orig image:', full_image.shape)
    reduced_image = block_reduce(full_image, block_size = (bin_size,bin_size,1), func = np.mean)
    print('block reduce:', reduced_image.shape)
    
    if type(resolution) != type(None):
        ul_row = int(0.5*(reduced_image.shape[0] - resolution[0]))
        ul_col = int(0.5*(reduced_image.shape[1] - resolution[1]))
        print('new ranges:')
        print('\trow:', ul_row, ul_row + resolution[0])
        print('\tcol:', ul_col, ul_col + resolution[1])
        reduced_image = reduced_image[ul_row:ul_row + resolution[0], 
                                          ul_col:ul_col + resolution[1]]
        
    return reduced_image




def KernelCenterCoords(semimajor_size):
    psf_c, psf_r = np.meshgrid(np.arange(-semimajor_size, semimajor_size + 1, 1, dtype = int), np.arange(-semimajor_size, semimajor_size + 1, 1, dtype = int))
    return psf_r, psf_c


def HannKernel(semimajor_size):
    
    hann_1d = signal.windows.hann(2*semimajor_size + 1)
    
    
    hann_1d_interp = sp.interpolate.interp1d(np.linspace(-semimajor_size, semimajor_size, num = int(round(2*semimajor_size+1))),
                                             hann_1d, bounds_error = False, fill_value = 0.0)
    
    
    
    psf_r, psf_c = KernelCenterCoords(semimajor_size)
    
    
    psf_radius = np.sqrt(psf_r**2 + psf_c**2)
    
    hann_psf = hann_1d_interp(psf_radius)
    hann_psf /= hann_psf.sum()
    
    return hann_psf


def ImageQuantileTransform(InputImage, mode = 'unlinked'):
    NMO  = InputImage.shape[0]*InputImage.shape[1]
    quants = np.linspace(0.0, 1.0, num = NMO)
    
    def QTransf(inDat):
        SI = np.argsort(inDat.ravel())
        SI_inv = np.argsort(SI)
        return quants[SI_inv].reshape(inDat.shape)
    
    QuantImage = np.zeros(InputImage.shape, dtype = float)
    
    if len(InputImage.shape) > 2:
        for chan in range(InputImage.shape[2]):
            QuantImage[:,:,chan] = QTransf(InputImage[:,:,chan])
    
    else:
        
        QuantImage = QTransf(InputImage)
        
    return QuantImage
        
        
    
    
    
    
    


def cuda_StarLocations(UnivariateImageWithStars, analysis_blur_sigma = 3, background_window = 11, minimum_prominance = 10.0/2**16):
    
    
    gpu_image = cp.asarray(UnivariateImageWithStars)
    
    if analysis_blur_sigma > 0:
        gpu_image = cpxndimage.gaussian_filter(gpu_image, analysis_blur_sigma, order=0, output=None, mode='nearest', cval=0.0, truncate=3.0)
    
    gpu_imageMax = cpxndimage.maximum_filter(gpu_image, size=background_window, footprint=None, output=None, mode='nearest', cval=0.0, origin=0)
    gpu_imageMin = cpxndimage.minimum_filter(gpu_image, size=background_window, footprint=None, output=None, mode='nearest', cval=0.0, origin=0)
    gpu_imageSum = cpxndimage.uniform_filter(gpu_image, size=background_window, output=None, mode='nearest', cval=0.0, origin=0)
    
    RowIndices, ColIndices = (gpu_imageMax == gpu_image).nonzero()
    
    star_peak_mags = gpu_image[RowIndices, ColIndices]
    
    star_floors = gpu_imageMin[RowIndices, ColIndices]
    
    star_prominances = star_peak_mags - star_floors
    
    bbb = star_prominances > minimum_prominance
    
    flux_proportion = ((gpu_imageSum[RowIndices, ColIndices] - star_floors)*(background_window**2)) / ( (star_peak_mags - star_floors) * (background_window**2))
    
    
    RowIndices = cp.asnumpy(RowIndices[bbb])
    ColIndices = cp.asnumpy(ColIndices[bbb])
    star_peak_mags = cp.asnumpy(star_peak_mags[bbb])
    star_prominances = cp.asnumpy(star_prominances[bbb])
    star_floors = cp.asnumpy(star_floors[bbb])
    flux_proportion = cp.asnumpy(flux_proportion[bbb])
    
    return RowIndices, ColIndices, star_peak_mags, star_prominances, star_floors, flux_proportion




def CudaStarMask(star_loc_rows, star_loc_cols, footprint_size, image_shape_tuple):
    
    
    ## Create the synthetic star images for registration
    ref_synthStar_gpu = np.zeros( (image_shape_tuple[0], image_shape_tuple[1]), dtype = np.float32)

    ref_synthStar_gpu[star_loc_rows, star_loc_cols] = 1.0

    ## pass the synhthetic star images to the device
    ref_synthStar_gpu = cp.asarray(ref_synthStar_gpu)
    
    ## Apply a maximum filter to the synthetic star images to spread out the star footprints
    star_footprint = np.array(skimage.morphology.disk(footprint_size), dtype = bool)
    
    ref_synthStar_gpu = cpx.scipy.ndimage.maximum_filter(ref_synthStar_gpu, footprint=star_footprint, output=None, mode='nearest', cval=0.0, origin=0)

    return cp.asnumpy(ref_synthStar_gpu) > 0





def StarRadialInfluence(star_prominances,
                        dropoff_intensity = 2**-16,
                        star_sigma = 2.5):
    
    return np.sqrt(-star_sigma**2*np.log(dropoff_intensity/star_prominances))
    
def StarMask(star_loc_rows, star_loc_cols, star_radii, image_shape_tuple):
    
    star_mask = np.zeros((image_shape_tuple[0],image_shape_tuple[1]) , dtype = bool)
    
    max_radius = np.max(star_radii)
    
    psf_r, psf_c = KernelCenterCoords(max_radius)
    
    psf_r = psf_r.ravel()
    psf_c = psf_c.ravel()
    
    for ii in range(len(psf_r)):
        
        rc_rad = math.sqrt(psf_r[ii]**2 + psf_c[ii]**2)
        
        mask_update_bool = rc_rad <= star_radii
        
        update_rows = star_loc_rows[mask_update_bool]+psf_r[ii]
        update_cols = star_loc_cols[mask_update_bool]+psf_c[ii]
        
        update_rows[update_rows < 0] = 0
        update_rows[update_rows > (image_shape_tuple[0]-1)] = (image_shape_tuple[0]-1)

        update_cols[update_cols < 0] = 0
        update_cols[update_cols > (image_shape_tuple[1]-1)] = (image_shape_tuple[1]-1)        
        
        
        star_mask[update_rows, update_cols ] = True
        
    return star_mask


def NeightborhoodStats(loc_rows, loc_cols, radius, image_data, image_data_mask = None):
    
    # neb_mean = np.zeros((image_data[0],image_data[1]) , dtype = float)
    
    # neb_mean = np.zeros((image_data[0],image_data[1]) , dtype = float)

    image_shape_tuple = image_data.shape
    
    if type(image_data_mask) == type(None):
        
        image_data_mask = np.ones(image_data.shape , dtype = bool)
        
        
    psf_r, psf_c = KernelCenterCoords(int(round(radius)))
    
    psf_r = psf_r.ravel()
    psf_c = psf_c.ravel()
    
    psf_rad_bool = (psf_r**2 + psf_c**2) <= radius**2
    
    psf_r = psf_r[psf_rad_bool]
    psf_c = psf_c[psf_rad_bool]
    
    sample_array = np.zeros( (len(loc_rows), len(psf_r)) , dtype = image_data.dtype)
    sample_array[:,:] = np.nan
    
    loc_index = np.arange(len(loc_rows), dtype = int)
    
    for ii in range(len(psf_r)):
        
        update_rows = loc_rows+psf_r[ii]
        update_cols = loc_cols+psf_c[ii]
        
        bounds_err_bool = (update_rows < 0) | (update_rows > (image_shape_tuple[0]-1))
        bounds_err_bool |= (update_cols < 0) | (update_cols > (image_shape_tuple[1]-1))
        bounds_err_bool = ~bounds_err_bool
        
        update_rows = update_rows[bounds_err_bool]
        update_cols = update_cols[bounds_err_bool]
        
        update_bool = image_data_mask[update_rows, update_cols]
        
        sample_vals = image_data[update_rows[update_bool], update_cols[update_bool]]
        
        sample_indices = (loc_index[bounds_err_bool])[update_bool]
        
        sample_array[sample_indices, ii] = sample_vals
        
    not_nan_bool = ~np.isnan(sample_array)
    
    return sample_array, not_nan_bool










    
def GetImageCoords(AnalysisImage = None, sensor_shape = (4176, 6248)):
    
    if type(AnalysisImage) == type(None):
        ImageCols, ImageRows  = np.meshgrid( np.arange(sensor_shape[1]), np.arange(sensor_shape[0]))
    else:
        ImageCols, ImageRows  = np.meshgrid( np.arange(AnalysisImage.shape[1]), np.arange(AnalysisImage.shape[0]))

    return ImageRows, ImageCols
    

        
        
def FindRectangle2(non_negative_image, count_requirement = None):
    
    if type(count_requirement) == type(None):
        count_requirement = non_negative_image.max()    
        
    ImageRows, ImageCols = GetImageCoords(AnalysisImage = non_negative_image)
    
    ReqBool = (non_negative_image >= count_requirement).ravel()
    
    Row1D = ImageRows.ravel()[ReqBool]
    
    Col1D = ImageCols.ravel()[ReqBool]
    
    prod_RC = (Row1D+1) * (Col1D+1)
    
    UL_i = np.argmin(prod_RC)
    
    LR_i = np.argmax(prod_RC)
    
    row_UL = Row1D[UL_i]
    col_UL = Col1D[UL_i]
    
    num_rows = Row1D[LR_i] - row_UL
    num_cols = Col1D[LR_i] - col_UL
        
    opt_rect = np.array([row_UL, col_UL, num_rows, num_cols], dtype = int)
    
    return opt_rect
    
    

