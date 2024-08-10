# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:37:02 2024

@author: champ
"""



import time


import numpy as np
import scipy as sp
import numpy.random as npr

import skimage

import cupy as cp
import cupyx as cpx

from cupyx.scipy import ndimage as cpxndimage


from astrocompyute.visualization import ShowImage, MultiScatterPlot




def cuda_polynomial_transform(departure_image , skimage_polynomial_transform_params):
    #gpu_image = cp.asarray(departure_image)
    
    #gpu_image_footrpint = cp.ones((gpu_image.shape[0],gpu_image.shape[1]), dtype = cp.float32)
    
    xx, yy = cp.meshgrid(cp.arange(departure_image.shape[1], dtype = cp.float32), cp.arange(departure_image.shape[0], dtype = cp.float32))
    
    TX = skimage_polynomial_transform_params[0]
    TY = skimage_polynomial_transform_params[1]
    
    TransformedCols = TX[0] + TX[1]*xx + TX[2]*yy + TX[4]*xx*yy + TX[3]*xx**2 + TX[5]*yy**2
    TransformedRows = TY[0] + TY[1]*xx + TY[2]*yy + TY[4]*xx*yy + TY[3]*xx**2 + TY[5]*yy**2
    
    CoordMatrix = cp.stack([TransformedRows, TransformedCols])
    
    out_img = np.zeros(departure_image.shape, dtype=np.float32)
    if len(departure_image.shape) > 2:
        out_img[:,:,0] = cp.asnumpy(cpxndimage.map_coordinates(cp.asarray(departure_image[:,:,0]), CoordMatrix , mode='constant', cval = 0.0))
        out_img[:,:,1] = cp.asnumpy(cpxndimage.map_coordinates(cp.asarray(departure_image[:,:,1]), CoordMatrix , mode='constant', cval = 0.0))
        out_img[:,:,2] = cp.asnumpy(cpxndimage.map_coordinates(cp.asarray(departure_image[:,:,2]), CoordMatrix , mode='constant', cval = 0.0))
        #out_img = cp.stack((out_img_R, out_img_G, out_img_B), axis = 2)
    
    else:
        out_img[:,:] = cp.asnumpy(cpxndimage.map_coordinates(cp.asarray(np.array(departure_image, dtype = np.float32)), CoordMatrix , mode='constant', cval = 0.0))
        
        
    out_footprint = cp.asnumpy(cpxndimage.map_coordinates(cp.ones((departure_image.shape[0],departure_image.shape[1]), dtype = cp.float32),  CoordMatrix, mode='constant', cval = 0.0))

    del xx
    del yy
    del TransformedCols
    del TransformedRows
    del CoordMatrix

    cp._default_memory_pool.free_all_blocks()
    return out_img, out_footprint #cp.asnumpy(out_footprint)





def CudaPolynomialRegistration2(reference_image, 
                               unaligned_image, 
                               ref_star_loc_rows,
                               ref_star_loc_cols,
                               ref_star_prominances,
                               test_star_loc_rows,
                               test_star_loc_cols,
                               test_star_prominances,
                               footprint_size = 5,
                               spatial_patch_radius = 50, # increase this to be more accepting of nearest neighbors
                               corr_thresh = .8,
                               patch_sample_points = 200,
                               patch_buffer = 200,
                               sub_image_radius = 500,
                               message = '',
                               min_sigmage = 10):
    
    """
    
    
    """

    T1 = time.time()
    NR =  reference_image.shape[0]
    NC =  reference_image.shape[1]
    
    num_ref_stars = len(ref_star_loc_rows)
    num_test_stars = len(test_star_loc_rows)
    
    est_ref_bits = num_ref_stars * np.pi * footprint_size**2
    ref_bg_bits = NC*NR - est_ref_bits
    
    bg_val = 0.0 #0.1 * (- est_ref_bits / ref_bg_bits)
    
    #min_sigmage = sp.stats.norm.ppf(1.0 - 1.0/(1000*1000*(2*sub_image_radius)**2))
    print('hhh', min_sigmage)
    ## Gather image and star specifics
    image_shape = unaligned_image.shape

    ref_min_prom = ref_star_prominances.min()
    test_min_prom = test_star_prominances.min()
    
    ## Create the synthetic star images for registration
    ref_synthStar_gpu = np.zeros( (reference_image.shape[0], reference_image.shape[1]), dtype = np.float32) + bg_val
    test_synthStar_gpu = np.zeros( (unaligned_image.shape[0], unaligned_image.shape[1]), dtype = np.float32) + bg_val
    
    # ref_synthStar_gpu[ref_star_loc_rows, ref_star_loc_cols] = 1.0 #np.log2(2**5 * (ref_star_prominances - ref_min_prom) + 1)/5
    # test_synthStar_gpu[test_star_loc_rows, test_star_loc_cols] = 1.0 #np.log2(2**5 * (test_star_prominances - test_min_prom) + 1)/5
    
    
    mapped_ref_prom = ref_star_prominances - ref_min_prom
    mapped_ref_prom[mapped_ref_prom < 0] = 0.0
    mapped_ref_prom /= (2.0* ref_min_prom)
    mapped_ref_prom[mapped_ref_prom >1] = 1.0

    mapped_test_prom = test_star_prominances - test_min_prom
    mapped_test_prom[mapped_test_prom < 0] = 0.0
    mapped_test_prom /= (2.0* test_min_prom)
    mapped_test_prom[mapped_test_prom >1] = 1.0    
    
    ref_synthStar_gpu[ref_star_loc_rows, ref_star_loc_cols] = mapped_ref_prom #np.log2(2**5 * (ref_star_prominances - ref_min_prom) + 1)/5
    test_synthStar_gpu[test_star_loc_rows, test_star_loc_cols] = mapped_test_prom #np.log2(2**5 * (test_star_prominances - test_min_prom) + 1)/5
       
    
    
    ## pass the synhthetic star images to the device
    ref_synthStar_gpu = cp.asarray(ref_synthStar_gpu)
    test_synthStar_gpu = cp.asarray(test_synthStar_gpu)
    
    spatial_mask = cp.asarray(np.zeros( (reference_image.shape[0], reference_image.shape[1]), dtype = np.float32 ) )
    

    ## Apply a maximum filter to the synthetic star images to spread out the star footprints
    star_footprint = np.array(skimage.morphology.disk(footprint_size), dtype = bool)
    
    ref_synthStar_gpu = cpxndimage.maximum_filter(ref_synthStar_gpu, footprint=star_footprint, output=None, mode='nearest', cval=0.0, origin=0)
    test_synthStar_gpu = cpxndimage.maximum_filter(test_synthStar_gpu, footprint=star_footprint, output=None, mode='nearest', cval=0.0, origin=0)




    patch_pix = (2*spatial_patch_radius + 1)**2
    image_pix = reference_image.shape[0] * reference_image.shape[1]
    
    avg_patch_corr = float(cp.sum((ref_synthStar_gpu)**2))
    
    avg_patch_corr = (avg_patch_corr/image_pix) * patch_pix
    print('AVG corr:', avg_patch_corr)
    
    
    
    
    
    #ShowImage(cp.asnumpy(ref_synthStar_gpu))
    
    # this one shows the prepared synth image of the test/target
    
    
    #ShowImage(cp.asnumpy(test_synthStar_gpu), Title = message)
    
    
    ## Create the containers that will hold the correlation point correspondences
    pts_xy_ref = []
    pts_xy_test = []
    
    cc_sigmas = []
    
    ## determine the correct boundary buffer given the user's inputs
    rc_buff = max(spatial_patch_radius, patch_buffer)
    
    ## pre-compute the gpu coordinate arrays for fast argmax
    row_range_gpu = cp.arange(NR, dtype = int)
    col_range_gpu = cp.arange(NC, dtype = int)
    
    all_correlations = np.zeros((patch_sample_points, 2))
    
    edge_prop = 0.15
    for iSamp in range(patch_sample_points):

        ## Generate the test point coordinates
        patch_loc_row = npr.randint(rc_buff, NR-rc_buff-1 )
        patch_loc_col = npr.randint(rc_buff, NC-rc_buff-1 )
        
        if iSamp %3 == 0:
            row_interior_test = ( (patch_loc_row > edge_prop*NR) and (patch_loc_row < (1.0-edge_prop)*NR))
            col_interior_test = ( (patch_loc_col > edge_prop*NC) and (patch_loc_col < (1.0-edge_prop)*NC))
            while row_interior_test and col_interior_test:
                patch_loc_row = npr.randint(rc_buff, NR-rc_buff-1 )
                patch_loc_col = npr.randint(rc_buff, NC-rc_buff-1 )
                row_interior_test = ( (patch_loc_row > edge_prop*NR) and (patch_loc_row < (1.0-edge_prop)*NR))
                col_interior_test = ( (patch_loc_col > edge_prop*NC) and (patch_loc_col < (1.0-edge_prop)*NC))
        
        ## compute the sub image ranges to reduce the size of the cross correlation computation
        ref_row_start = max(0, patch_loc_row - sub_image_radius)
        ref_row_end = min(NR, patch_loc_row + sub_image_radius)
        
        ref_col_start = max(0, patch_loc_col - sub_image_radius)
        ref_col_end = min(NC, patch_loc_col + sub_image_radius)
        
        # Number of subimage rows (NSR), number of subimage columns (NSC)
        NSR = ref_row_end - ref_row_start
        NSC = ref_col_end - ref_col_start
        
        #######################
        #### Perform the circular cross-correlation
        #print('\tref patch bits:', ref_synthStar_gpu[ref_row_start:ref_row_end, ref_col_start:ref_col_end].sum(), (ref_row_end-ref_row_start)*(ref_col_end - ref_col_start))
        ## Fourier transform of the reference subimage
        # # # #rp = cp.fft.fft2(ref_synthStar_gpu[ref_row_start:ref_row_end, ref_col_start:ref_col_end])
        
        ## construction of the test subimage specific to the test point coordinates
        tp = test_synthStar_gpu[patch_loc_row-spatial_patch_radius:patch_loc_row+spatial_patch_radius+1, 
                                patch_loc_col-spatial_patch_radius:patch_loc_col+spatial_patch_radius+1]
        
        spatial_mask[patch_loc_row-spatial_patch_radius:patch_loc_row+spatial_patch_radius+1, 
                     patch_loc_col-spatial_patch_radius:patch_loc_col+spatial_patch_radius+1] =  1.0*tp
        
        #print('\ttest patch bits:', spatial_mask.sum(), (2*spatial_patch_radius+1)**2)
        
        
        
        ## compute the maximum correlation
        priori_correlation =   float(cp.sum(tp**2) ) 
        
        #print('hhg', priori_correlation, tp.shape)
        
        ## Fourier transform of the test subimage
        # # # tp2 = cp.fft.fft2(spatial_mask[ref_row_start:ref_row_end, ref_col_start:ref_col_end]).conj()
                                                                                                            
        ## complete the circular convolution
        # # # cc_image = cp.fft.fftshift(cp.fft.ifft2( rp * tp2 )).real
        
        
        cc_image = cp.fft.fftshift(cp.fft.ifft2( cp.fft.fft2(ref_synthStar_gpu[ref_row_start:ref_row_end, ref_col_start:ref_col_end]) * cp.fft.fft2(spatial_mask[ref_row_start:ref_row_end, ref_col_start:ref_col_end]).conj() )).real
        
        cc_image[int(round(0.5 * NSR)), int(round(0.5 * NSC))] = -1.

        ## repair the spatial_mask gpu array for use in the next iteration of the loop
        spatial_mask[patch_loc_row-spatial_patch_radius:patch_loc_row+spatial_patch_radius+1, 
                     patch_loc_col-spatial_patch_radius:patch_loc_col+spatial_patch_radius+1] = bg_val
        
        #######################
        #### Determine the location of the maximum correlation
        
        row_maxima = cc_image.max(axis = 1)
        col_maxima = cc_image.max(axis = 0)
        rowMax = int(row_maxima.argmax())
        colMax = int(col_maxima.argmax())
                
        #rowMax, colMax = divmod(float(cc_image.argmax()), cc_image.shape[1]) # slower?
 
        ## retrieve the max correlation
        cc_max = cc_image[int(rowMax), int(colMax)]

        ## convert the argmax location to a subtractive row/col offset
        subtractive_row_shift_test2ref =  int(round(0.5 * NSR - rowMax))
        subtractive_col_shift_test2ref =  int(round(0.5 * NSC - colMax))
        
        
        nSigma, p50, pSigma = cp.percentile(cc_image, [15.865525393145708, 50, 84.1344746068543])
        nSigma = float(nSigma)
        p50 = float(p50)
        pSigma = float(pSigma)
        cc_sigmage = (cc_max - p50) / (0.5*(pSigma - nSigma))
        #print('debug:', cc_max, nSigma, pSigma, p50, 'bg:', bg_val)
        corr_prop = (cc_max / priori_correlation)
        
        all_correlations[iSamp, 0] =  corr_prop
        all_correlations[iSamp, 1] =  cc_sigmage

        #######################
        #### Determine the sigmage of the maximum correlation   
  
        if (priori_correlation > 0.33*avg_patch_corr) and (corr_prop > corr_thresh) and (corr_prop < 1.1) and (cc_sigmage > min_sigmage): #/ True: #cc_sigmage > sigmage_thresh:
            pts_xy_ref.append([patch_loc_col-subtractive_col_shift_test2ref, patch_loc_row-subtractive_row_shift_test2ref])
            pts_xy_test.append([patch_loc_col, patch_loc_row])
            #print('Acceptable correlation:', iSamp, 'dy=', subtractive_row_shift_test2ref, 'dx=', subtractive_col_shift_test2ref, 'corr %:', 100*corr_prop, 'sigmage=', cc_sigmage)
        
        else:
            #print('Possible low quality correlation:', iSamp, 'dy=', subtractive_row_shift_test2ref, 'dx=', subtractive_col_shift_test2ref, 'corr %:', 100*corr_prop, 'sigmage=', cc_sigmage)
            pass
    #print(all_correlations)
    corr_prop_deciles = np.percentile(all_correlations[:,0], np.linspace(0, 100, num = 11))
    corr_sigmage_deciles = np.percentile(all_correlations[:,1], np.linspace(0, 100, num = 11))
    
    #print('Correlation deciles:', '(surviving:', np.sum(all_correlations[:,0] > corr_thresh),')')
    print('\tCorrelation summary:', '(surviving:', np.sum(all_correlations[:,0] > corr_thresh),'out of', len(all_correlations),')')
    for ii in range(len(corr_prop_deciles)):
        print('\t', ii*10, '%:', 'prop=', round(corr_prop_deciles[ii], 3), 'sigmage=', round(corr_sigmage_deciles[ii], 3))
        
    pts_xy_ref = np.array(pts_xy_ref)
    pts_xy_test = np.array(pts_xy_test)
    pts_xy_shifts = pts_xy_ref - pts_xy_test
    
    s_nS_x, s_med_x, s_pS_x = np.percentile(pts_xy_shifts[:,0], [15.865525393145708, 50, 84.1344746068543])
    s_nS_y, s_med_y, s_pS_y = np.percentile(pts_xy_shifts[:,1], [15.865525393145708, 50, 84.1344746068543])
    
    shift_sigma = 0.5 * ( 0.5 * (s_pS_x - s_nS_x)  + 0.5 * (s_pS_y - s_nS_y)) + .2
    
    med_shift = np.array([[s_med_x, s_med_y]])
    
    
    
    rel_lengths = np.sqrt(np.sum(((pts_xy_shifts) - med_shift)**2, axis = 1))
    
    outliers = rel_lengths > 5*shift_sigma
    
    print('\t\toutlier removal:', np.sum(outliers))
    #print('\tnum outliers:', np.sum(outliers))
    #print('\t\t)
    print('\t\tmed shift:', med_shift, ', shift stdev:', shift_sigma)
    
    if len(pts_xy_ref) > 20:
        ## Compute the polynomial transformation
        tform = skimage.transform.estimate_transform('polynomial', pts_xy_ref[~outliers], pts_xy_test[~outliers])
    
    else:
        return None, None, None, None, None
    
    print('\n\n\tRecovered shift params:', np.round(tform.params[0,0],3), np.round(tform.params[1,0], 3))
    
    ## Warp the test image into alignment with the the reference image
    #aligned_image = skimage.transform.warp(unaligned_image[:,:,1], inverse_map=tform) 
    aligned_image, aligned_image_footprint = cuda_polynomial_transform(unaligned_image , tform.params)
    T3 = time.time()
    
    print('\tTotal registration time (CUDA):', T3-T1)
    
    #del(cc_image, tp2, rp, test_synthStar_gpu, ref_synthStar_gpu)
    if False:
        MultiScatterPlot([pts_xy_ref[~outliers,0], pts_xy_test[~outliers,0]], 
                            [pts_xy_ref[~outliers,1], pts_xy_test[~outliers,1]], 
                            Colors = ['k', 'r'], 
                            Sizes = [20,5], 
                            Labels = ['ref', 'test'], 
                            Lines = False, 
                            alpha = 0.8, 
                            LineWidth = 3, 
                            Title = message, 
                            XLabel = 'Columns', 
                            YLabel = 'Rows')
        
    
    del ref_synthStar_gpu
    del test_synthStar_gpu
    del spatial_mask
    del cc_image
    
    cp._default_memory_pool.free_all_blocks()
    
    
    return aligned_image, aligned_image_footprint, tform.params, pts_xy_ref, pts_xy_test#, cp.mean(cc_sigmas)


















