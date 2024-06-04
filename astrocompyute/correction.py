# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:56:52 2024

@author: champ
"""


from copy import deepcopy

import numpy as np
import scipy as sp
import cupy as cp
from cupyx.scipy import ndimage as cpxndimage
          
import time

import skimage.filters as skfilters

import cv2

import astrocompyute.filters as myfilters
import astrocompyute.imagemath as imagemath
import numpy.random as npr
from astrocompyute.visualization import MultiScatterPlot3D

## Flat FIled and Dark filed correction function
# construct the master dark and flat using the astrocompyute.GenFlatDark script
def FFDF_correct(raw_image_rgb, 
                 proc_flat_field,
                 proc_dark_field = None,
                 FFnorm2unity = True, # setting this to False will use the relative intensities in the RGB flat field to color correct the channel bias.. I think this is like a grey field correction.
                 norm2unity = True, 
                 offset = 700.):
    
    if type(proc_dark_field) == type(None):
        proc_dark_field = offset/2**16
    
    if FFnorm2unity:
        FF_rgb_unity = deepcopy(proc_flat_field)
        FF_rgb_unity[:,:,0] = (FF_rgb_unity[:,:,0])/(FF_rgb_unity[:,:,0].max())
        FF_rgb_unity[:,:,1] = (FF_rgb_unity[:,:,1])/(FF_rgb_unity[:,:,1].max())
        FF_rgb_unity[:,:,2] = (FF_rgb_unity[:,:,2])/(FF_rgb_unity[:,:,2].max())

        spatial_norm_RGB = ( raw_image_rgb - proc_dark_field ) / FF_rgb_unity
        
    else:
        
        spatial_norm_RGB = ( raw_image_rgb - proc_dark_field ) / proc_flat_field
        
        
    spatial_norm_RGB[spatial_norm_RGB <= 1.0/2**16] =  1.0/2**16 
    spatial_norm_RGB[spatial_norm_RGB > 1.0] =  1.0 - 1.0/2**16 

    if norm2unity:
        spatial_norm_RGB[:,:,0] /= spatial_norm_RGB[:,:,0].max()
        spatial_norm_RGB[:,:,1] /= spatial_norm_RGB[:,:,1].max()
        spatial_norm_RGB[:,:,2] /= spatial_norm_RGB[:,:,2].max()
        
    return spatial_norm_RGB




###############################################################################
###############################################################################
######## Gradient/Background Removal

def polyfit2d(x, y, f, deg):
    
    if type(deg) == type(4):
        
        deg = np.array([deg, deg])
    x = np.asarray(x)
    y = np.asarray(y)
    deg = np.asarray(deg)
    vander = np.polynomial.polynomial.polyvander2d(x, y, deg)
    vander = vander.reshape((-1,vander.shape[-1]))
    f = f.reshape((vander.shape[0],))
    c = np.linalg.lstsq(vander, f)[0]
    return c.reshape(deg+1)


def UnivariateBGEst_poly(UnivImage,
                         not_star_mask,
                         kernel_size = 41,
                         upshift_perc = 0.1,
                         poly_regression_downsample = 0.001,
                         ADU_outlier_thresh = 10.,
                         poly_order = 4,
                         verbose = 0):

    t0 = time.time()
    print('preparing image and kernels', time.time() - t0)
    # Construct the kernels that will be used in the star removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    kernel_h = np.ones((3,3*kernel_size))
    kernel_v = np.ones((3*kernel_size,3))
    
    ## skimage rank filters require uint16 input (sheesh), we perform that conversion here
    Lum2 = UnivImage*2**16
    Lum2[Lum2 >= 2**16] = 2**16-1
    Lum2 = np.array(Lum2, dtype = np.uint16)
    
    ## We apply masked median filters in a geometric sequence to fill in the 
    ## star locations with masked median estimates (why dont we just sample from the not_star_mask??)
    print('applying median filters', time.time() - t0)
    BG_filled_stars = skfilters.rank.median(Lum2, kernel_h, mask = not_star_mask)  
    BG_filled_stars = skfilters.rank.median(BG_filled_stars, kernel_v, mask = BG_filled_stars>0)  
    BG_filled_stars = skfilters.rank.median(BG_filled_stars, kernel, mask = BG_filled_stars>0)  
    
    ## convert the BG estimate back to a float data type so that we can estimate
    ## BG params in that space
    BG_filled_stars  = np.array(BG_filled_stars, dtype = float)
    
    print('applying gaussian filter', time.time() - t0)
    ## We apply a gaussian blur filter here to recover from the uint16 loss of 
    # fidelity
    BG_filled_stars = myfilters.cuda_gaussianFilter(50, BG_filled_stars)
    
    print('sampling image', time.time() - t0)
    ## Determine the X and Y coordinates of the image and sample from them randomly
    ImageRows, ImageCols = imagemath.GetImageCoords(AnalysisImage = BG_filled_stars)

    rrr = npr.random(BG_filled_stars.shape)
    
    SampleBool = rrr < poly_regression_downsample
    
    nR = BG_filled_stars.shape[0]
    
    ## When we downsample, we scale the R and C coords by the row count so that 
    ## our independant coordinates vary nominaly between 0 and just over 1.  This
    ## will prevent overflow errors that I have observed when performing higher
    ## order polynomial fits. 
    R_coords_sample = ImageRows[SampleBool]/nR
    C_coords_sample = ImageCols[SampleBool]/nR
    Z_coords_sample = BG_filled_stars[SampleBool]
    
    print('Stage 1 poly fit', time.time() - t0)
    ## Stage 1 polynomial fit
    poly4 = polyfit2d(R_coords_sample, 
                      C_coords_sample, 
                      Z_coords_sample, 
                      poly_order)
    
    poly4_eval = np.polynomial.polynomial.polyval2d(R_coords_sample, C_coords_sample, poly4)
    
    # we bootstrap improve our sample points by filtering out the extreme outliers. 
    # we identify the surviving sample points here
    second_pass_mask = np.abs(poly4_eval - Z_coords_sample) < ADU_outlier_thresh

    ## Second stage polynomial fit. 
    print('Stage 2 poly fit', time.time() - t0)
    poly4 = polyfit2d(R_coords_sample[second_pass_mask], 
                      C_coords_sample[second_pass_mask], 
                      Z_coords_sample[second_pass_mask], 
                      poly_order)
    
    poly4_eval = np.polynomial.polynomial.polyval2d(R_coords_sample, C_coords_sample, poly4)
    
    if verbose > 1:
    
        MultiScatterPlot3D([R_coords_sample, R_coords_sample], 
                              [C_coords_sample, C_coords_sample], 
                              [Z_coords_sample, poly4_eval], 
                              Labels = ['data', str(poly_order) + 'th order fit'],
                          Colors = ['b', 'k'], 
                          Sizes = [2, 2], 
                          Title = '',
                          XLabel = 'Row',
                          YLabel = 'Col',
                          ZLabel = 'Intensity',
                          aspect = 'garbage')
        
        
    ## compute the upshift percentile, this is how we determine the zero point. 
    shift_z = np.percentile(Z_coords_sample - poly4_eval, upshift_perc)
    print('Evaluating poly on the entire image', time.time() - t0)
    ## Evaluate the recovery polynomial model on the entire image
    fast_auto_BG = np.polynomial.polynomial.polyval2d(ImageRows.ravel()/nR, ImageCols.ravel()/nR, poly4)
    
    fast_auto_BG += shift_z
    fast_auto_BG /= 2**16
    fast_auto_BG = fast_auto_BG.reshape(Lum2.shape)
    print('BG estimate complete', time.time() - t0)
    return fast_auto_BG


#%%

def SuperSimpleUnivariateBGEst_poly(UnivImage,
                         not_star_mask,
                         poly_regression_downsample = 0.001,
                         poly_order = 4,
                         ): # subsample_bool = None

    t0 = time.time()
    #print('sampling image', time.time() - t0)
    ## Determine the X and Y coordinates of the image and sample from them randomly
    ImageRows, ImageCols = imagemath.GetImageCoords(AnalysisImage = UnivImage)

    rrr = npr.random(UnivImage.shape)
    
    SampleBool = (rrr < poly_regression_downsample) & not_star_mask
    
    nR = UnivImage.shape[0]
    
    ## When we downsample, we scale the R and C coords by the row count so that 
    ## our independant coordinates vary nominaly between 0 and just over 1.  This
    ## will prevent overflow errors that I have observed when performing higher
    ## order polynomial fits. 
    R_coords_sample = ImageRows[SampleBool]/nR
    C_coords_sample = ImageCols[SampleBool]/nR
    Z_coords_sample = UnivImage[SampleBool]
    
    #print('Stage 1 poly fit', time.time() - t0)
    ## Stage 1 polynomial fit
    poly4 = polyfit2d(R_coords_sample, 
                      C_coords_sample, 
                      Z_coords_sample, 
                      poly_order)
    
    poly4_eval = np.polynomial.polynomial.polyval2d(R_coords_sample, C_coords_sample, poly4)
    
    #print('Evaluating poly on the entire image', time.time() - t0)
    ## Evaluate the recovery polynomial model on the entire image
    #fast_auto_BG = np.polynomial.polynomial.polyval2d(ImageRows.ravel()/nR, ImageCols.ravel()/nR, poly4)
    fast_auto_BG = polyval2d_cuda(ImageRows/nR, ImageCols/nR, poly4)
    #fast_auto_BG = fast_auto_BG.reshape(UnivImage.shape)
    #print('BG estimate complete', time.time() - t0)
    return fast_auto_BG

#%%


def WeightedUnivariateBGEst_poly(UnivImage,
                         not_star_mask,
                         PosWeightsImage,
                         poly_regression_downsample = 0.001,
                         poly_order = 4,
                         extra_edge_sampling_prop = 0.5,
                         extra_edge_sampling_width = 300,
                         ): # subsample_bool = None

    t0 = time.time()
    #print('sampling image', time.time() - t0)
    ## Determine the X and Y coordinates of the image and sample from them randomly
    ImageRows, ImageCols = imagemath.GetImageCoords(AnalysisImage = UnivImage)

    rrr = npr.random(UnivImage.shape)
    
    SampleBool = (rrr < poly_regression_downsample)
    
    rand_inds = npr.randint(0, extra_edge_sampling_width, size = int(extra_edge_sampling_prop*poly_regression_downsample*ImageRows.shape[0]*ImageRows.shape[1]))
    rand_rows = npr.randint(0, ImageRows.shape[0], size = int(extra_edge_sampling_prop*poly_regression_downsample*ImageRows.shape[0]*ImageRows.shape[1]))
    rand_cols = npr.randint(0, ImageRows.shape[1], size = int(extra_edge_sampling_prop*poly_regression_downsample*ImageRows.shape[0]*ImageRows.shape[1]))
    
    SampleBool[rand_rows, rand_inds] = True
    SampleBool[rand_rows, ImageRows.shape[1]-rand_inds-1] = True
    SampleBool[rand_inds, rand_cols] = True
    SampleBool[ImageRows.shape[0]-rand_inds-1, rand_cols] = True
    
    SampleBool &= not_star_mask
    
    nR = UnivImage.shape[0]
    
    ## When we downsample, we scale the R and C coords by the row count so that 
    ## our independant coordinates vary nominaly between 0 and just over 1.  This
    ## will prevent overflow errors that I have observed when performing higher
    ## order polynomial fits. 
    R_coords_sample = ImageRows[SampleBool]/nR
    C_coords_sample = ImageCols[SampleBool]/nR
    Z_coords_sample = UnivImage[SampleBool]
    W_sample = PosWeightsImage[SampleBool]
    
    nSamp = len(R_coords_sample)
    #print('Stage 1 poly fit', time.time() - t0)
    ## Stage 1 polynomial fit
    
    CoefMatrix = poly_features(C_coords_sample, R_coords_sample, poly_order, poly_coef = None)
   
    W = np.sqrt(np.diag(W_sample))
    X = np.linalg.lstsq(np.dot(W,CoefMatrix), np.dot(Z_coords_sample,W))[0]
    
    #print('Evaluating poly on the entire image', time.time() - t0)    
    fast_auto_BG = poly_features_cuda(ImageCols.ravel()/nR, ImageRows.ravel()/nR, poly_order, poly_coef = X).reshape(ImageCols.shape)
    
    train_pred =  poly_features_cuda(C_coords_sample, R_coords_sample, poly_order, poly_coef = X)
    
    if False:
        MultiScatterPlot3D([R_coords_sample, R_coords_sample], 
                              [C_coords_sample, C_coords_sample], 
                              [Z_coords_sample, train_pred], 
                              Labels = ['data', str(poly_order) + 'th order fit'],
                          Colors = ['b', 'k'], 
                          Sizes = [2, 2], 
                          Title = '',
                          XLabel = 'Row',
                          YLabel = 'Col',
                          ZLabel = 'Intensity',
                          aspect = 'garbage')
    
    

    #print('BG estimate complete', time.time() - t0)
    return fast_auto_BG

def WeightedRGBBGEst_poly(ImageRGB,
                                 not_star_mask,
                                 PosWeightsImage,
                                 poly_regression_downsample = 0.001,
                                 poly_order = 4,
                                 extra_edge_sampling_prop = 0.5,
                                 extra_edge_sampling_width = 300,
                                 ):
    
    WBG_est = np.zeros(ImageRGB.shape, ImageRGB.dtype)
    
    WBG_est[:,:,0] = WeightedUnivariateBGEst_poly(ImageRGB[:,:,0],
                             not_star_mask,
                             PosWeightsImage[:,:,0],
                             poly_regression_downsample = poly_regression_downsample,
                             poly_order = poly_order,
                             extra_edge_sampling_prop = extra_edge_sampling_prop,
                             extra_edge_sampling_width = extra_edge_sampling_width,
                             )
    
    WBG_est[:,:,1] = WeightedUnivariateBGEst_poly(ImageRGB[:,:,1],
                             not_star_mask,
                             PosWeightsImage[:,:,1],
                             poly_regression_downsample = poly_regression_downsample,
                             poly_order = poly_order,
                             extra_edge_sampling_prop = extra_edge_sampling_prop,
                             extra_edge_sampling_width = extra_edge_sampling_width,
                             )
    
    WBG_est[:,:,2] = WeightedUnivariateBGEst_poly(ImageRGB[:,:,2],
                             not_star_mask,
                             PosWeightsImage[:,:,2],
                             poly_regression_downsample = poly_regression_downsample,
                             poly_order = poly_order,
                             extra_edge_sampling_prop = extra_edge_sampling_prop,
                             extra_edge_sampling_width = extra_edge_sampling_width,
                             )    
    
    return WBG_est
    
    


def poly_features_cuda(x_vals, y_vals, poly_order, poly_coef = None):
    nSamp = len(x_vals)
    x_vals_gpu = cp.asarray(x_vals)
    y_vals_gpu = cp.asarray(y_vals)
    CoefMatrix = cp.zeros((nSamp, int(round(  (poly_order+2)*(poly_order+1)/2  ))), dtype = float)
    
    cidx = 0
    
    for xPow in range(0, poly_order+1, 1):
        if xPow > 0:
            xFeat = x_vals_gpu**xPow
        else:
            xFeat = cp.ones(nSamp, dtype = float)
        
        for yPow in range(0, poly_order+1, 1):
            
            if (xPow + yPow) > poly_order:
                continue
            
            if yPow > 0:
                yFeat = y_vals_gpu**xPow
            else:
                yFeat = cp.ones(nSamp, dtype = float)
                
            CoefMatrix[:,cidx] = xFeat * yFeat
            cidx += 1
    
    if type(poly_coef) != type(None):
        
        _poly_coef = cp.asarray(np.reshape(poly_coef, (CoefMatrix.shape[1],1)))
        
        poly_eval = cp.dot(CoefMatrix, _poly_coef)
        
        poly_eval = cp.asnumpy(poly_eval).ravel()
        return poly_eval
    
    return cp.asnumpy(CoefMatrix)
    
    
def poly_features(x_vals, y_vals, poly_order, poly_coef = None):
    nSamp = len(x_vals)

    CoefMatrix = np.zeros((nSamp, int(round(  (poly_order+2)*(poly_order+1)/2  ))), dtype = float)
    
    cidx = 0
    
    for xPow in range(0, poly_order+1, 1):
        if xPow > 0:
            xFeat = x_vals**xPow
        else:
            xFeat = np.ones(nSamp, dtype = float)
        
        for yPow in range(0, poly_order+1, 1):
            
            if (xPow + yPow) > poly_order:
                
                continue
            
        
            if yPow > 0:
                yFeat = y_vals**xPow
            else:
                yFeat = np.ones(nSamp, dtype = float)
                
            CoefMatrix[:,cidx] = xFeat * yFeat
            cidx += 1
            
    if type(poly_coef) != type(None):
        
        _poly_coef = np.reshape(poly_coef, (CoefMatrix.shape[1],1))
        
        poly_eval = np.dot(CoefMatrix, _poly_coef)
        
        poly_eval = poly_eval.ravel()
        return poly_eval
    
    return CoefMatrix
    

#%%


#%%
# nnn = 1000000
# xd = np.linspace(-1.0, 1.0, num = nnn)
# yd = np.linspace(-1.0, 1.0, num = nnn)

# z_data = 1.0 - 2.0*xd + 3.0 * yd - 4.0 * xd**2 + 5.0 * xd * yd - 6.0 * yd**2
# z_data += (7.0 * xd**3 - 8.0*xd**2*yd + 9.0 * xd*yd**2 - 10.0*yd**3  )
# z_data += (11.0 * xd**4 - 12.0*xd**3*yd + 13*xd**2*yd**2 - 14*xd*yd**3 + 15*yd**4)

# poly4 = polyfit2d(xd, 
#                   yd, 
#                   z_data, 
#                   4)



def polyval2d_cuda(x_data, y_data, coef_mat):
    
    x_gpu = cp.asarray(x_data)
    y_gpu = cp.asarray(y_data)
    
    z_data = cp.zeros(x_data.shape, dtype = float)
    
    
    for row_x_i in range(coef_mat.shape[0]):
        for col_y_j in range(coef_mat.shape[1]):
            z_data += float(coef_mat[row_x_i, col_y_j]) * x_gpu**row_x_i * y_gpu**col_y_j
            
    return cp.asnumpy(z_data)
            
            
    
    



#%%
def SimpleUnivGridBGEst_poly(UnivImage, 
                             BG_sample_mask, 
                             BG_sample_percentile = 5,  
                             grid_n = 20, 
                             min_sample_pix = 5,
                             verbose = 0,
                             weighted = False,
                             extraEdgeSampling = True,
                             message = ''):
    
    T1 = time.time()
    
    coarse_grid_n = grid_n
    coarse_grid_m = int(round(coarse_grid_n * (UnivImage.shape[1]/UnivImage.shape[0])))
    coarse_grid_pix = int(UnivImage.shape[0]/coarse_grid_n)-1
    coarse_grid_pix_radius = int(coarse_grid_pix/2)
    eighth  = int(round(grid_n/8))
    
    coarse_grid_sample_rows = np.array(np.round(np.linspace(coarse_grid_pix_radius+1, 
                                                            UnivImage.shape[0]-(coarse_grid_pix_radius+1), 
                                                            num = coarse_grid_n)), dtype = int)
    
    coarse_grid_sample_cols = np.array(np.round(np.linspace(coarse_grid_pix_radius+1, 
                                                            UnivImage.shape[1]-(coarse_grid_pix_radius+1), 
                                                            num = coarse_grid_m)), dtype = int)
    
    if extraEdgeSampling:
    
        coarse_grid_sample_rows = np.concatenate([coarse_grid_sample_rows[:eighth] + int(coarse_grid_pix/3), 
                                                   coarse_grid_sample_rows[:eighth] + int(2*coarse_grid_pix/3),
                                                   coarse_grid_sample_rows,
                                                   coarse_grid_sample_rows[-eighth:] - int(coarse_grid_pix/3),
                                                   coarse_grid_sample_rows[-eighth:] - int(2*coarse_grid_pix/3)])
        
        coarse_grid_sample_cols = np.concatenate([coarse_grid_sample_cols[:eighth] + int(coarse_grid_pix/3), 
                                                   coarse_grid_sample_cols[:eighth] + int(2*coarse_grid_pix/3),
                                                   coarse_grid_sample_cols,
                                                   coarse_grid_sample_cols[-eighth:] - int(coarse_grid_pix/3),
                                                   coarse_grid_sample_cols[-eighth:] - int(2*coarse_grid_pix/3)])    
        
        coarse_grid_sample_rows = np.sort(coarse_grid_sample_rows)
        coarse_grid_sample_cols = np.sort(coarse_grid_sample_cols)
    
    
    
    ## prepare the sampling footprint (the "kernel")
    k_r, k_c = imagemath.KernelCenterCoords(coarse_grid_pix_radius)
    
    circle_bool = (k_r**2 + k_c**2) <  coarse_grid_pix_radius**2
    
    k_r_samp = k_r[circle_bool].ravel()
    k_c_samp = k_c[circle_bool].ravel()
    
    sample_c2D, sample_r2D  = np.meshgrid(coarse_grid_sample_cols, coarse_grid_sample_rows)
    
    T2 = time.time()
    
    sample_int = np.zeros((sample_c2D.shape[0], sample_c2D.shape[1]))
    sample_means = np.zeros((sample_c2D.shape[0], sample_c2D.shape[1]))
    sample_bool = np.ones((sample_c2D.shape[0], sample_c2D.shape[1]), dtype = bool)
    
    for ir, r in enumerate(coarse_grid_sample_rows):
        for ic, c in enumerate(coarse_grid_sample_cols):
            
            data_samp = UnivImage[k_r_samp + r, k_c_samp + c]
            data_samp_bool = BG_sample_mask[k_r_samp + r, k_c_samp + c]
            
            data_sample_count = np.sum(data_samp_bool) 
            if data_sample_count < min_sample_pix:
                sample_bool[ir, ic] = False
                continue
            
            data_percs = float(np.percentile(data_samp[data_samp_bool], BG_sample_percentile))
            sample_int[ir, ic] = data_percs
            sample_means[ir, ic] = np.mean(data_samp[data_samp_bool])
    
    T3 = time.time()
    x_fit = sample_c2D.ravel()/UnivImage.shape[0]
    y_fit = sample_r2D.ravel()/UnivImage.shape[0]
    
    full_xy_fit = deepcopy(np.vstack((x_fit, y_fit)))
    
    z_fit = sample_int.ravel()
    
    x_fit = x_fit[sample_bool.ravel()]
    y_fit = y_fit[sample_bool.ravel()]
    z_fit = z_fit[sample_bool.ravel()]
    sigma_weights = ((sample_means - sample_int).ravel())[sample_bool.ravel()]
    
    xy_fit = np.vstack((x_fit, y_fit))
    
    def Poly4ParamEval(xy, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14):
        x = xy[0]
        y = xy[1]
        params = np.array([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14])
        z = params[0] + 0.0*x # 0th order
        z += params[1]*x + params[2]*y # 1st order
        z += params[3]*x**2 + params[4]*y**2 + params[5]*x*y # 2nd order
        z += params[6]*x**3 + params[7] * y**3 + params[8] * x*y**2 + params[9] * y*x**2  # 3rd order
        z += params[10] * x**4 + params[11] * y**4 + params[12] * x*y**3 + params[13] * y*x**3 + params[14] * x**2*y**2 # 4th order
        return z
    

    
    
    T4 = time.time()
    if weighted:
        popt, pcov = sp.optimize.curve_fit(Poly4ParamEval, 
                                           xy_fit, 
                                           z_fit, 
                                           sigma=sigma_weights, #(sample_int[:,:,50]-sample_int[:,:,5]).ravel(),
                                           p0=None,  
                                           method=None)
    else:
        popt, pcov = sp.optimize.curve_fit(Poly4ParamEval, 
                                           xy_fit, 
                                           z_fit, 
                                           #sigma=(sample_means - sample_int).ravel(), #(sample_int[:,:,50]-sample_int[:,:,5]).ravel(),
                                           p0=None,  
                                           method=None)        
    T5 = time.time()
    surf_est_full_grid = Poly4ParamEval(full_xy_fit, *popt)
    
    if verbose > 1:
        MultiScatterPlot3D([sample_c2D[sample_bool].ravel(), sample_c2D.ravel()], 
                           [sample_r2D[sample_bool].ravel(), sample_r2D.ravel()], 
                           [z_fit*2**16,  surf_est_full_grid*2**16], 
                            Colors = ['darkred', 'k'], 
                            Sizes = [20,20],
                            alpha = 0.5,
                            XLabel = 'columns',
                            YLabel = 'rows',
                            ZLabel = 'detector ADU (16-bit)',
                            Labels = ['sample percentile data', 'opt surface fit'],
                            Title = message)  
    T6 = time.time()
    ImageRows, ImageCols = imagemath.GetImageCoords(AnalysisImage = UnivImage)
    T7 = time.time()
    
    FullBG = CudaPoly4ParamEval(ImageCols/UnivImage.shape[0], ImageRows/UnivImage.shape[0], popt)
    
    
    # FullBG = Poly4ParamEval(np.vstack([ImageCols.ravel(), ImageRows.ravel()])/UnivImage.shape[0], 
    #                         *popt)
    T8 = time.time()
    if verbose > 0:
        print(message,'kernel prep', T2-T1)
        print(message,'sampling data', T3-T2)
        print(message,'prep reg inputs', T4-T3)
        print(message,'curve_fit', T5-T4)
        print(message,'eval at coarse grid', T6-T5)
        print(message,'image coords', T7-T6)
        print(message,'full eval', T8-T7)
    
    return FullBG.reshape(ImageCols.shape)


def CudaPoly4ParamEval(xCPU, yCPU, params):
    x = cp.asarray(xCPU)
    y = cp.asarray(yCPU)
    z = params[0] + 0.0*x # 0th order
    z += params[1]*x + params[2]*y # 1st order
    z += params[3]*x**2 + params[4]*y**2 + params[5]*x*y # 2nd order
    z += params[6]*x**3 + params[7] * y**3 + params[8] * x*y**2 + params[9] * y*x**2  # 3rd order
    z += params[10] * x**4 + params[11] * y**4 + params[12] * x*y**3 + params[13] * y*x**3 + params[14] * x**2*y**2 # 4th order
    return cp.asnumpy(z)
    


def SimpleRGBGridBGEst_poly(ImageRGB, 
                             BG_sample_mask, 
                             BG_sample_percentile = 5,  
                             grid_n = 20, 
                             min_sample_pix = 5,
                             verbose = 0,
                             weighted = False,
                             extraEdgeSampling = True,
                             message = ''):
    
    BGEstRGB = np.zeros(ImageRGB.shape, dtype = ImageRGB.dtype)
    
    for chan in [(0,'Red'),(1,'Green'),(2, 'Blue')]:
        BGEstRGB[:,:,chan[0]] = SimpleUnivGridBGEst_poly(ImageRGB[:,:,chan[0]], 
                                     BG_sample_mask, 
                                     BG_sample_percentile = BG_sample_percentile,  
                                     grid_n = grid_n, 
                                     min_sample_pix = min_sample_pix,
                                     verbose = verbose,
                                     weighted = weighted,
                                     extraEdgeSampling = extraEdgeSampling,
                                     message = message + ': ' + chan[1] + ' channel')
        
    return BGEstRGB
    
    
    

    
#%%

def starmapBasedGradient(ImageChannel, 
                         starmap_channel, 
                         sample_boolHA,
                         poly_order = 2,
                         sample_prop = 0.0001):
    
    # smooth the input images
    smoothed_starmap_grey = cpxndimage.median_filter(cp.asarray(starmap_channel), (101,1), mode = 'nearest')
    smoothed_starmap_grey = cpxndimage.median_filter(smoothed_starmap_grey, (1,101), mode = 'nearest')
    smoothed_starmap_grey = cp.asnumpy(cpxndimage.gaussian_filter(smoothed_starmap_grey, 30, order=0, output=None, mode='nearest', cval=0.0, truncate=3.0))

    smoothed_ImageChannel = cpxndimage.median_filter(cp.asarray(ImageChannel), (101,1), mode = 'nearest')
    smoothed_ImageChannel = cpxndimage.median_filter(smoothed_ImageChannel, (1,101), mode = 'nearest')
    smoothed_ImageChannel = cp.asnumpy(cpxndimage.gaussian_filter(smoothed_ImageChannel, 30, order=0, output=None, mode='nearest', cval=0.0, truncate=3.0))

    ImageRows, ImageCols = imagemath.GetImageCoords(AnalysisImage=ImageChannel)
    
    sample_points2D = (npr.rand(*ImageRows.shape) < sample_prop) & sample_boolHA
    
    sample_rows = ImageRows[sample_points2D]
    sample_cols = ImageCols[sample_points2D]
    sample_dataStar = smoothed_starmap_grey[sample_points2D]
    sample_dataCam = smoothed_ImageChannel[sample_points2D]

    
    cam_p5, cam_p99 = np.percentile(smoothed_ImageChannel, (5, 99))
    star_p5, star_p99 = np.percentile(smoothed_starmap_grey, (5, 99))

    const = -(star_p99-star_p5) * cam_p5 / (cam_p99-cam_p5) + star_p5
    scalar = (star_p99-star_p5) / (cam_p99-cam_p5)
    
    xy_fit = [sample_cols/ImageRows.shape[0], sample_rows/ImageRows.shape[0]]
    
    def data_starmap_transform_fm_2nd( p_vec , return_pred = False, input_xy = xy_fit, cam_data = sample_dataCam):
        p0_0, p1_x, p1_y, p2_xx, p2_xy, p2_yy , d = p_vec
        _x = input_xy[0]
        _y = input_xy[1]
        synth_Star = p0_0 + p1_x * _x + p1_y * _y 
        synth_Star += p2_xx * _x**2 + p2_xy * _x*_y + p2_yy * _y**2 
        synth_Star += d * cam_data
        
        if return_pred:
            
            poly_gradient = (synth_Star - d * cam_data)/d
            
            return synth_Star, poly_gradient
         
        rmse = np.sqrt(np.mean((synth_Star-sample_dataStar)**2))
        
        return rmse    


    def data_starmap_transform_fm_4th( p_vec , return_pred = False, input_xy = xy_fit, cam_data = sample_dataCam):
        p0_0, p1_x, p1_y, p2_xx, p2_xy, p2_yy , p3_xxx, p3_xxy, p3_xyy, p3_yyy, p4_xxxx, p4_xxxy, p4_xxyy, p4_xyyy, p4_yyyy, d = p_vec
        _x = input_xy[0]
        _y = input_xy[1]
        synth_Star = p0_0 + p1_x * _x + p1_y * _y 
        synth_Star += p2_xx * _x**2 + p2_xy * _x*_y + p2_yy * _y**2 
        synth_Star += p3_xxx * _x**3 + p3_xxy * _x**2*_y + p3_xyy * _x*_y**2 + p3_yyy * _y**3
        synth_Star += p4_xxxx *_x**4 + p4_xxxy * _x**3 * _y + p4_xxyy * _x**2 * _y**2 + p4_xyyy * _x*_y**3 + p4_yyyy * _y**4
        synth_Star += d * cam_data
        
        if return_pred:
            
            poly_gradient = (synth_Star - d * cam_data)/d
            
            return synth_Star, poly_gradient
         
        rmse = np.sqrt(np.mean((synth_Star-sample_dataStar)**2))
        
        return rmse    
    
    M1 = 0.01
    M2 = 0.01
    M3 = 0.000001
    M4 = 0.000001
    
    p02nd = np.array([const*(1.0 + .2*npr.randn()), 
                     npr.randn()*M1, npr.randn()*M1,
                     npr.randn()*M2,npr.randn()*M2,npr.randn()*M2,
                     scalar*(1.0 + .2*npr.randn())])  #, (1.0 + .1*npr.randn())])
    p04th = np.array([const*(1.0 + .2*npr.randn()), 
                     npr.randn()*M1, npr.randn()*M1,
                     npr.randn()*M2, npr.randn()*M2, npr.randn()*M2,
                     npr.randn()*M3, npr.randn()*M3, npr.randn()*M3, npr.randn()*M3,
                     npr.randn()*M4, npr.randn()*M4, npr.randn()*M4, npr.randn()*M4, npr.randn()*M4, 
                     scalar*(1.0 + .2*npr.randn())])  #, (1.0 + .1*npr.randn())])
    

    popt = sp.optimize.fmin(data_starmap_transform_fm_2nd, p02nd)
    for i in range(2):
        popt = sp.optimize.fmin(data_starmap_transform_fm_2nd, popt)
    
    p04th[:len(popt)-1] = popt[:-1]
    p04th[-1] = popt[-1]
    
    if poly_order == 4:
        popt = sp.optimize.fmin(data_starmap_transform_fm_4th, p04th)
        for i in range(10):
            popt = sp.optimize.fmin(data_starmap_transform_fm_4th, popt)
        final_rmse = data_starmap_transform_fm_4th(popt)
        print('popt', popt)
        print('final_rmse', final_rmse)
        evalFull, eval_gradient = data_starmap_transform_fm_4th( popt , 
                                             return_pred = True, 
                                             input_xy = [ImageCols/ImageRows.shape[0], ImageRows/ImageRows.shape[0]],
                                             cam_data = ImageChannel)
    else:
        evalFull, eval_gradient = data_starmap_transform_fm_2nd( popt , 
                                             return_pred = True, 
                                             input_xy = [ImageCols/ImageRows.shape[0], ImageRows/ImageRows.shape[0]],
                                             cam_data = ImageChannel)
        
    return -eval_gradient, popt

def starmapBasedGradientRGB(ImageRGB, 
                         starmap_RGB, 
                         sample_boolHA,
                         poly_order = 2,
                         sample_prop = 0.0001):
    
    SM_BGEst = np.zeros(ImageRGB.shape, dtype = ImageRGB.dtype)
    #print('xx1:', SM_BGEst.shape, ImageRGB.shape)
    
    for chan in [0,1,2]:
        SM_BGEst[:,:,chan] = starmapBasedGradient(ImageRGB[:,:,chan], 
                                 starmap_RGB[:,:,chan], 
                                 sample_boolHA,
                                 poly_order = poly_order,
                                 sample_prop = sample_prop)[0]
        
    return SM_BGEst
        
        


#%%


def starmapBasedGradientFast(ImageChannel, 
                         starmap_channel, 
                         sample_boolHA,
                         poly_order = 2,
                         sample_prop = 0.001,
                         trials = 10):
    
    # smooth the input images
    smoothed_starmap_grey = cpxndimage.median_filter(cp.asarray(starmap_channel), (101,1), mode = 'nearest')
    smoothed_starmap_grey = cpxndimage.median_filter(smoothed_starmap_grey, (1,101), mode = 'nearest')
    smoothed_starmap_grey = cp.asnumpy(cpxndimage.gaussian_filter(smoothed_starmap_grey, 30, order=0, output=None, mode='nearest', cval=0.0, truncate=3.0))

    smoothed_ImageChannel = cpxndimage.median_filter(cp.asarray(ImageChannel), (101,1), mode = 'nearest')
    smoothed_ImageChannel = cpxndimage.median_filter(smoothed_ImageChannel, (1,101), mode = 'nearest')
    smoothed_ImageChannel = cp.asnumpy(cpxndimage.gaussian_filter(smoothed_ImageChannel, 30, order=0, output=None, mode='nearest', cval=0.0, truncate=3.0))

    ImageRows, ImageCols = imagemath.GetImageCoords(AnalysisImage=ImageChannel)
    nR = ImageRows.shape[0]
    
    cam_p5, cam_p99 = np.percentile(smoothed_ImageChannel, (5, 99))
    star_p5, star_p99 = np.percentile(smoothed_starmap_grey, (5, 99))

    const = -(star_p99-star_p5) * cam_p5 / (cam_p99-cam_p5) + star_p5
    scalar = (star_p99-star_p5) / (cam_p99-cam_p5)
    scalar_opt = scalar
    
    #!#
    sample_b = (npr.rand(*ImageRows.shape) < sample_prop) & sample_boolHA
    #sN = np.sum(sample_b)
    
    rand_inds = npr.randint(0, 300, size = int(0.5*sample_prop*ImageRows.shape[0]*ImageRows.shape[1]))
    rand_rows = npr.randint(0, ImageRows.shape[0], size = int(0.5*sample_prop*ImageRows.shape[0]*ImageRows.shape[1]))
    rand_cols = npr.randint(0, ImageRows.shape[1], size = int(0.5*sample_prop*ImageRows.shape[0]*ImageRows.shape[1]))
    
    sample_b[ImageRows.shape[0] - rand_inds - 1, rand_cols] |= sample_boolHA[ImageRows.shape[0] - rand_inds - 1, rand_cols]
    sample_b[rand_inds, rand_cols] |= sample_boolHA[rand_inds, rand_cols]
    
    sample_b[rand_rows, rand_inds] |= sample_boolHA[rand_rows, rand_inds]
    sample_b[rand_rows, ImageRows.shape[1]-rand_inds-1] |= sample_boolHA[rand_rows, ImageRows.shape[1]-rand_inds-1]
    
    #ShowImage(sample_b)
    
    samp_r = ImageRows[sample_b]/nR
    samp_c = ImageCols[sample_b]/nR
    samp_cam = smoothed_ImageChannel[sample_b]
    samp_star = smoothed_starmap_grey[sample_b]
    
    def fullTestScalar(scalarD, return_poly = False):
        resultant = samp_star / scalarD - samp_cam
    
        poly4 = polyfit2d(samp_c, 
                          samp_r, 
                          resultant, 
                          4)
        gradient_est = np.polynomial.polynomial.polyval2d(samp_c, samp_r, poly4)
        
        #rmse = np.sqrt(np.mean((gradient_est - resultant)**2))
        rmse = np.sqrt(np.mean(( (gradient_est*scalarD + scalarD* samp_cam) - samp_star)**2))
        if return_poly:
            return poly4
        return rmse
    
    
    scalar_opt = sp.optimize.fmin(fullTestScalar, scalar)
    print('scalar_opt', scalar_opt)
    poly4 = fullTestScalar(float(scalar_opt), return_poly = True)
    print('poly4', poly4)            
    PolyGradient = polyval2d_cuda(ImageCols/nR, ImageRows/nR, poly4)
    
    
    if False:
    
        for itrial in range(trials):
            print(itrial, 'testing:', scalar_opt)
            resultant = smoothed_starmap_grey / scalar_opt - smoothed_ImageChannel
            
            PolyGradient = SuperSimpleUnivariateBGEst_poly(resultant,
                                                             sample_boolHA,
                                                             poly_regression_downsample = 0.001,
                                                             poly_order = 4)
            
            sample_points2D = (npr.rand(*ImageRows.shape) < sample_prop) & sample_boolHA
            
            def testScalar(scalar_est):
                resids = PolyGradient[sample_points2D] - (smoothed_starmap_grey[sample_points2D] / scalar_est -  smoothed_ImageChannel[sample_points2D])
                
                return np.sqrt(np.mean(resids**2))
            
            scalar_opt = sp.optimize.fmin(testScalar, scalar)
            print('opt_scalar:', testScalar(scalar_opt))
            try:
                scalar_opt = scalar_opt[0]
            except:
                scalar_opt = float(scalar_opt)
        
    return -PolyGradient



#%%


def RGB_BGEst_poly(RGB_image,
                         not_star_mask,
                         kernel_size = 41,
                         upshift_perc = 0.1,
                         poly_regression_downsample = 0.001,
                         ADU_outlier_thresh = 10.,
                         poly_order = 4,
                         verbose = 0):

    RedBG = UnivariateBGEst_poly(RGB_image[:,:,0],
                             not_star_mask,
                             kernel_size = kernel_size,
                             upshift_perc = upshift_perc,
                             poly_regression_downsample = poly_regression_downsample,
                             ADU_outlier_thresh = ADU_outlier_thresh,
                             poly_order = poly_order,
                             verbose = verbose)
    
    GreenBG = UnivariateBGEst_poly(RGB_image[:,:,1],
                             not_star_mask,
                             kernel_size = kernel_size,
                             upshift_perc = upshift_perc,
                             poly_regression_downsample = poly_regression_downsample,
                             ADU_outlier_thresh = ADU_outlier_thresh,
                             poly_order = poly_order,
                             verbose = verbose)
    
    BlueBG = UnivariateBGEst_poly(RGB_image[:,:,2],
                             not_star_mask,
                             kernel_size = kernel_size,
                             upshift_perc = upshift_perc,
                             poly_regression_downsample = poly_regression_downsample,
                             ADU_outlier_thresh = ADU_outlier_thresh,
                             poly_order = poly_order,
                             verbose = verbose)
    
    
    return np.stack([RedBG, GreenBG, BlueBG], axis = 2) 





































