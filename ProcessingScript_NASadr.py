# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:14:26 2024

@author: champ
"""
import numpy as np
#import numpy.random as npr
#import scipy as sp

from copy import deepcopy

#import skimage

import os

import time

import cv2

import matplotlib as mpl
import matplotlib.pyplot as plt
        

#%%# Import a-priori computed methods and tools
import astrocompyute.apriori as apriori 
import astrocompyute.fileio as fileio
import astrocompyute.imagemath as imagemath
import astrocompyute.registration as registration
import astrocompyute.correction as correction
import astrocompyute.visualization as astroviz
import astrocompyute.filters as myfilters
import astrocompyute.enhance as enhance
import astrocompyute.color as mycolor

from astrocompyute.visualization import ShowImage, ShowImageRGB, QuickInspectRGB, PlotHistRGB, ContourMultiPlot
#%%

    
#%%### Specify light, dark, flat data

# # Sigma 105: RMSS NA Nebula
LightFramesDir = 'C:/Users/champ/Astronomy/Data/RMSS_NA_Nebula'
MasterFLatFP = 'C:/Users/champ/Astronomy/MasterFlats/FF_Sigma105_fullSpec.npy'
MasterDarkFP = 'C:/Users/champ/Astronomy/MasterDarks/DF_5s_n10c_g200.npy' 
stack_token = 'RMSS_NA_Nebula'

# # Sigma 105:Lagoon Swan Eagle Trifid
# LightFramesDir = 'C:/Users/champ/Astronomy/Data/RMSS_Lagoon_Trifid_Swan_Eagle'
# MasterFLatFP = 'C:/Users/champ/Astronomy/MasterFlats/FF_Sigma105_fullSpec.npy'
# MasterDarkFP = 'C:/Users/champ/Astronomy/MasterDarks/DF_5s_n10c_g200.npy' 
# stack_token = 'Lagoon_Trifid_Swan_Eagle'

# Output directory for final images
output_dir = 'C:/Users/champ/Astronomy/ProcessedImages'

# generate an updated processed image at this frequncy 
astro_processing_cadence = 5

# Set this to True to active a live updating figure as the processing proceeds
live_figure = True

#%%### REQUIRED EXECUTABLE AND FILE PATHS (USER UPDATES THESE)

# File paths to ASTAP executables(obtain installer from https://www.hnsky.org/astap)
astap_cli_executable_fp = '"C:/Program Files/astap/astap_cli.exe"'
astap_executable_fp = '"C:/Program Files/astap/astap.exe"' 

# NASA All-Sky Deep Star Map "starmap" (obtain file from https://svs.gsfc.nasa.gov/4851)
GaiaDR2_all_sky_fp = 'C:/Users/champ/Astronomy/Reference/starmap_2020_16k.exr'

# H-alpha map file, obtain from:
#    https://faun.rc.fas.harvard.edu/dfink/skymaps/halpha/data/v1_1/index.html
Finkbeiner_allSky_Halpha_fp = 'C:/Users/champ/Astronomy/Reference/Halpha_map.fits'

# Starnet++ command line tool (filepaths, obtain and install "StarNetv2CLI_Win.zip" from https://www.starnetastro.com/)
# The basic install is the CPU version (which is fine), the GPU tensorfly 
# install is faster for Nvidia GPU machines. 
gpu_starnet = 'C:/Users/champ/Astronomy/Software/StarNetv2CLI_Win_GPU/StarNetv2CLI_Win/'
cpu_starnet = 'C:/Users/champ/Astronomy/Software/StarNetv2CLI_Win_CPU/StarNetv2CLI_Win/'



#%%### Initialize the image stack containers and stack quantities and survey directory

# Generate a timestamp to uniquely identify the stack
TS = str(int(time.time()))

# stack file names and paths
stack_count_fp = 'AutoStackCount_' + stack_token + '_' + TS + '.npy'
stack_sum_fp = 'AutoStackSum_'  + stack_token + '_' + TS + '.npy'
#!#stack_sqsum_fp = 'AutoStackSqSum_' + stack_token + '_' + TS + '.npy'

stack_metadata_fp = 'AutoStackMetadata_' + stack_token + '_' + TS + '.pickle'

stack_count_fp = os.path.join(LightFramesDir, stack_count_fp)
stack_sum_fp = os.path.join(LightFramesDir, stack_sum_fp)
stack_metadata_fp = os.path.join(LightFramesDir, stack_metadata_fp)

AvailableLightFrames = [os.path.join(LightFramesDir, fn) for fn in os.listdir(LightFramesDir)  if (fn[-4:] in ['.tif', 'fits', '.cr3']) and ('AutoStack' not in fn)]

# container to hold the transformation params for each sub frame
T_params = []

#%%### Load master flat/dark frames

DF_rgb = np.load(MasterDarkFP)
FF_rgb = np.load(MasterFLatFP) 

#%%### Read the first light frame and construct the stack

# read the light frame
StackedImageRGB, _imBayer0 = fileio.ReadASI_TIF_FITS(AvailableLightFrames[0], salt = False)

# initialize the stack count
StackedImageCount = np.ones((StackedImageRGB.shape[0],StackedImageRGB.shape[1]), dtype = float)

# FF and DF correction
StackedImageRGB = correction.FFDF_correct(StackedImageRGB, 
                              FF_rgb,
                              DF_rgb,
                              FFnorm2unity = True,  # setting this
                              norm2unity = False)

# ensure the stack is less than 1.0
StackedImageRGB *= (2**16-1.)/2**16

# Compute the greyscale intensity of the stack
StackImage_int = StackedImageRGB.mean(axis = 2)

#%%### Star calculations on the starting stack

# kernel sizer for star calculations
kernelSize = 21

# compute the locations of the stars in the image
[StackStarRowIndices, 
 StackStarColIndices, 
 StackStar_peak_mags, 
 StackStar_prominances, 
 StackStar_floors, 
 StackStar_flux_proportion] = imagemath.cuda_StarLocations(StackImage_int, 
                                                 analysis_blur_sigma = 1, 
                                                 background_window = kernelSize, 
                                                 minimum_prominance = 100.0/2**16)


#%%### Plate solve the initial frame and compute the starmap and HA map

# row/column coordinates of each pixel
ImageRows, ImageCols = imagemath.GetImageCoords(AnalysisImage=StackedImageRGB)

# You must provide your local path 
platesolve_solution = apriori.PlateSolve(AvailableLightFrames[0], 
                                         ra_guess = None, dec_guess = None, 
                                         fov = 8.85 , 
                                         search = 170, 
                                         astap_executable = astap_cli_executable_fp)

# compute the RA and DEC coords of each pixel
deg_RA, deg_DEC = apriori.Pixels2RADEC_Transform3D_cuda(ImageRows.ravel(), 
                                                        ImageCols.ravel(), 
                                                        platesolve_solution)

# retrieve the Gaia starmap interpolator (obtain from https://svs.gsfc.nasa.gov/4851)
EvalStarmap = apriori.GenStarmapFast(mode = 'rgb', path_to_EXR_GaiaDR2_NASA_starmap=GaiaDR2_all_sky_fp)

# evaluate the starmap interp at the image coordinates
starmap_at_image = EvalStarmap(deg_RA, deg_DEC) #.reshape(StackedImageRGB.shape)

starmap_at_image = np.stack([starmap_at_image[:,2].reshape(ImageRows.shape),
                             starmap_at_image[:,1].reshape(ImageRows.shape),
                             starmap_at_image[:,0].reshape(ImageRows.shape)], axis = 2)

starmap_grey = starmap_at_image.mean(axis = 2)

        
stretched_starmap_at_image = enhance.CustomPseudoLogStretch_cuda(starmap_at_image, 
                                                                 BP_S = 0.0, 
                                                                 BP_T = 0.01, 
                                                                 ExpN = 6, 
                                                                 zz = 0.5, 
                                                                 StretchChannel = 'Luminance')

# in case the map is flipped, repair it here with a seconday test
starmap_grey, secondary_test = apriori.SecondaryTestStarmap(starmap_grey, 
                                                                StackImage_int, 
                                                                threshold_perc = 95)

# H-alpha map file, obtain from:
#    https://faun.rc.fas.harvard.edu/dfink/skymaps/halpha/data/v1_1/index.html
# retrieve the H-alpha map interpolator
EvalHAmap = apriori.GenHAmapFast(path_to_Finkbeiner_allSky_Halpha = Finkbeiner_allSky_Halpha_fp,
                                 dec_res=8192, 
                                 ra_res=16384)

# evaluate the h-alpha map at the image coordinates
HAmap_at_image = EvalHAmap(deg_RA, deg_DEC).reshape(StackImage_int.shape)

# repair the H-alpha map if it is flipped as well
HAmap_at_image = HAmap_at_image[::secondary_test[0],::secondary_test[1]]


#%%### generate the sampling boolean indicated by the Gaia starmap and all-skyu H-alpha map

smoothed_starmap_grey = myfilters.cuda_gaussianFilter(30, starmap_grey)

p_star = np.percentile(smoothed_starmap_grey, 80) # 0.00626 seems like an OK absolute threshold

p_HA = np.percentile(HAmap_at_image, 80) # 0.1 or 0.35 is a reasonable absolute threshold

BG_DSO_STAR_bool = (smoothed_starmap_grey < p_star) & (HAmap_at_image < p_HA )

sample_boolHA =  HAmap_at_image < p_HA # 0.35 is a reasonable absolute threshold

# ShowImage(sample_boolHA)
# ShowImage(BG_DSO_STAR_bool)

#%%
## all of these work nicely to show the starmap

# ShowImageRGB(stretched_starmap_at_image)
# ShowImage(starmap_at_image[:,:,:]*10)
# QuickInspectRGB(starmap_at_image, perc_black=5, perc_white = 99.95)

#%%### generate a background sampling boolean that excludes annotated DSO's 
not_DSO_mask, annotation = apriori.PlateSolveMask(AvailableLightFrames[0], 
                    astap_executable = astap_executable_fp ,
                    dilate_n = 200,
                    return_annotation_bool = True,
                    min_BG_pix_proportion = 0.5,
                    kill_the_file = False,
                    max_attempts = 10
                    )

# ShowImage(BG_DSO_STAR_bool)
# ShowImage(not_DSO_mask)
# ShowImage(annotation)


#%%
if live_figure: # instantiate the live figure frame is selected

    live_fig, live_ax = plt.subplots(figsize=(1.75*16,1.75*10.6), facecolor='black')

    text_color = 'red'
    extra_text = '\n[f]:fullscreen, [o]:zoom, [ctrl+w]:close, [p]:pan, [h]:reset'

    mpl.rcParams['toolbar'] = 'None'
    mpl.rcParams['toolbar'] = 'toolbar2'

    live_ax.axis('off')
    #plt.tight_layout(pad = 1.00)

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    figManager = plt.get_current_fig_manager()
    #figManager.window.showMaximized()

#%%### Main stacking loop

ProcessedLightFrames = [AvailableLightFrames[0]]
AvailableLightFrames.remove(AvailableLightFrames[0])

total_count = len(ProcessedLightFrames) + len(AvailableLightFrames)

proc_count = 1



#while len(AvailableLightFrames) > 0:
while True:
    
    while (len(AvailableLightFrames) == 0):
        print('sleeping for 10 seconds to await next file...')
        time.sleep(10)
        
        AvailableLightFrames = [os.path.join(LightFramesDir, fn) for fn in os.listdir(LightFramesDir)  if (fn[-4:] in ['.tif', 'fits', '.cr3']) and ('AutoStack' not in fn)]
        AvailableLightFrames = [fp for fp in AvailableLightFrames if fp not in ProcessedLightFrames]
        
        total_count = len(ProcessedLightFrames) + len(AvailableLightFrames)
    
    fp = AvailableLightFrames[0]
    
    proc_count += 1
    print('\n\n#####################################################################')
    print('#####################################################################')

    progress_str = '(' + str(proc_count) + '/' + str(total_count) +')'
    print('Processing frame:', proc_count, fp, 'of', total_count)

    
    print('\t##############################')
    print('\tReading image', progress_str)
    try:
        FrameImage, frame_imBayer0 = fileio.ReadASI_TIF_FITS(fp, salt = False)
    except:
        continue
    
    print('\t##############################')
    print('\tFF DF correction', progress_str)
    FrameImage = correction.FFDF_correct(FrameImage, 
                                          FF_rgb,
                                          DF_rgb,
                                          FFnorm2unity = True,  # setting this
                                          norm2unity = False)
    
    FrameImage *= (2**16-1.)/2**16
    FrameImage_int = FrameImage.mean(axis = 2)
    #  QuickInspectRGB(FrameImage, perc_black=5, perc_white = 95.)
    
    print('\t##############################')
    print('\tStar locations', progress_str)
    
    [FrameStarRowIndices, 
     FrameStarColIndices, 
     FrameStar_peak_mags, 
     FrameStar_prominances, 
     FrameStar_floors, 
     FrameStar_flux_proportion] = imagemath.cuda_StarLocations(FrameImage_int, 
                                                     analysis_blur_sigma = 1, 
                                                     background_window = kernelSize, 
                                                     minimum_prominance = 100.0/2**16)
    

    print('\t##############################')
    print('\tFrame registration to stack', progress_str)

    [aligned_image, 
     aligned_image_footprint,
     tform_params, 
     pts_xy_ref, 
     pts_xy_test] = registration.CudaPolynomialRegistration2(StackedImageRGB, 
                                                             FrameImage, 
                                                             StackStarRowIndices,
                                                             StackStarColIndices,
                                                             StackStar_prominances,
                                                             FrameStarRowIndices,
                                                             FrameStarColIndices,
                                                             FrameStar_prominances,
                                                             spatial_patch_radius = 100,
                                                             corr_thresh = .7,
                                                             patch_sample_points = 200,
                                                             patch_buffer = 200,
                                                             footprint_size = 5, #bal_star_radius, #5,
                                                             sub_image_radius = 500
                                                             #nearest_neighbor_criteria = 2, # increase this to be more accepting of nearest neighbors
                                                             )



    print('\t##############################')                               
    print('\tUpdate stack', progress_str)
    T_params.append(tform_params)

    if type(aligned_image) == type(None):
                                     
        print('\tFailed frame registratrion, skipping...', progress_str)

        continue
    
    else:
        print('\tSuccessful frame registratrion', progress_str)
        StackedImageCount += aligned_image_footprint
        StackedImageRGB += aligned_image

    print('\t##############################')                               
    print('\tUpdating files', progress_str)
            
    ProcessedLightFrames.append(fp)
    AvailableLightFrames.remove(fp)
    
    AvailableLightFrames = [os.path.join(LightFramesDir, fn) for fn in os.listdir(LightFramesDir)  if (fn[-4:] in ['.tif', 'fits', '.cr3']) and ('AutoStack' not in fn)]
    AvailableLightFrames = [fp for fp in AvailableLightFrames if fp not in ProcessedLightFrames]
    
    total_count = len(ProcessedLightFrames) + len(AvailableLightFrames)
    proc_count = len(ProcessedLightFrames)
    
    
    if (proc_count%10 == 0) or (len(AvailableLightFrames)==0):
        np.save(stack_count_fp, StackedImageCount)
        np.save(stack_sum_fp, StackedImageRGB)
        # save metadata??
       
    ## Continue to the next file if this check fails, otherwise render an image
    if (proc_count%astro_processing_cadence != 0):
        continue
        
    #%%######################################################################
    print('\n\n#############################################################')
    print('#############################################################')
    print('################  Rendering Image', progress_str, '\n\n')   


    T0 = time.time()
    #######################################################################
    print('\n\n################  Computing optimal image crop')
    
    crop_n = int(0.8 * StackedImageCount.max())
    
    opt_rect = imagemath.FindRectangle2(StackedImageCount, count_requirement = crop_n) #, fuzz_iterations = 36)
    
    opt_rect = np.array(np.round(opt_rect), dtype = int)
    
    opt_rect_row_start = opt_rect[0]
    opt_rect_row_end = opt_rect[0] + opt_rect[2]
    opt_rect_col_start = opt_rect[1]
    opt_rect_col_end = opt_rect[1] + opt_rect[3]
    
    vis_crop = [opt_rect_row_start, opt_rect_row_end, opt_rect_col_start, opt_rect_col_end]
    
    interim_RGB = deepcopy(StackedImageRGB[vis_crop[0]:vis_crop[1], vis_crop[2]:vis_crop[3]] / np.stack([StackedImageCount[vis_crop[0]:vis_crop[1], vis_crop[2]:vis_crop[3]]]*3, 2))
    
    interim_apriori_BG_mask = BG_DSO_STAR_bool[vis_crop[0]:vis_crop[1], vis_crop[2]:vis_crop[3]]
    
    
    
    Lum = np.mean(interim_RGB, axis = 2)
    T1 = time.time()
    
    print('\t',round(T1 - T0, 4) , 'seconds')
    
    #######################################################################
    print('\n\n################  Computing star locations and star mask') 
    
    [FrameStarRowIndices, 
     FrameStarColIndices, 
     StackStar_peak_mags, 
     FrameStar_prominances, 
     FrameStar_floors, 
     FrameStar_flux_proportion] = imagemath.cuda_StarLocations(Lum, 
                                                     analysis_blur_sigma = 2, 
                                                     background_window = 11, #21, #kernelSize, # kernelSize=21 is nominal?
                                                     minimum_prominance = 4.0/2**16)
    
                                                     
    star_mask = ~imagemath.CudaStarMask(FrameStarRowIndices, FrameStarColIndices, 11, Lum.shape)                                                 
    #star_mask_smaller = ~imagemath.CudaStarMask(FrameStarRowIndices, FrameStarColIndices, 9, Lum.shape)                                           
    #ShowImage(star_mask)  
    #ShowImage(star_mask_smaller)  

    # ShowImage(Lum, min_color_val = np.percentile(Lum, 5), max_color_val = np.percentile(Lum, 99))  
    # ShowImage(interim_RGB[:,:,2], min_color_val = np.percentile(interim_RGB[:,:,2], 5), max_color_val = np.percentile(interim_RGB[:,:,2], 99))  
    # QuickInspectRGB(interim_RGB, perc_black=5, perc_white = 99.)
    
    T2 = time.time()
    
    print('\t',round(T2 - T1, 4) , 'seconds')
    
    #######################################################################
    print('\n\n################  Background and gradient removal') 

    BGEstMethod = 'Grid' # 'Starmap', 'Grid', 'StarWeighted'
    
    # ShowImage(interim_apriori_BG_mask & star_mask)
    
    if BGEstMethod == 'Starmap':
        ## Works well on Milky Way shots with large and dense star fields (slow)
        BG_est = correction.starmapBasedGradientRGB(interim_RGB, 
                                                    starmap_at_image[vis_crop[0]:vis_crop[1], vis_crop[2]:vis_crop[3],:], 
                                                    sample_boolHA[vis_crop[0]:vis_crop[1], vis_crop[2]:vis_crop[3]],
                                                    poly_order = 4,
                                                    sample_prop = 0.0001)
        BackgroundRemovedRGB = interim_RGB - BG_est
        BackgroundRemovedRGB -= np.percentile(BackgroundRemovedRGB, 25)


    elif BGEstMethod == 'Grid':
        ## Simple and fast 4th order polynomial BG estimation sampled far from hazards
        BG_est = correction.SimpleRGBGridBGEst_poly(interim_RGB, 
                                                    interim_apriori_BG_mask & star_mask, 
                                                    BG_sample_percentile = 5,  
                                                    grid_n = 25, 
                                                    min_sample_pix = 5,
                                                    verbose = 0,
                                                    weighted = False,
                                                    extraEdgeSampling = True)

        BackgroundRemovedRGB = interim_RGB - BG_est
        
    elif BGEstMethod == 'StarWeighted':
        
        BG_floor_est = np.stack([myfilters.cuda_rankfilt_dual(51,1,interim_RGB[:,:,chan], 5) for chan in [0,1,2]], axis = 2)
        BG_weights = 1.0 - starmap_at_image[vis_crop[0]:vis_crop[1], vis_crop[2]:vis_crop[3],:]
        p_low_R, p_high_R = np.percentile(BG_weights[:,:,0], [1,99])
        p_low_G, p_high_G = np.percentile(BG_weights[:,:,1], [1,99])
        p_low_B, p_high_B = np.percentile(BG_weights[:,:,2], [1,99])
        
        BG_weights -= np.array([[[p_low_R, p_low_G, p_low_B]]])
        BG_weights /= np.array([[[p_high_R-p_low_R, p_high_G-p_low_G, p_high_B-p_low_B]]])

        BG_weights[BG_weights < 0] = 0.0
        BG_weights[BG_weights > 1.0] = 1.0

        SSBB = (HAmap_at_image < np.percentile(HAmap_at_image, 90))[vis_crop[0]:vis_crop[1], vis_crop[2]:vis_crop[3]] 
        
        BG_est = correction.WeightedRGBBGEst_poly(BG_floor_est,
                                         SSBB & star_mask, #
                                         BG_weights**1,
                                         poly_regression_downsample = 0.001,
                                         poly_order = 4,
                                         extra_edge_sampling_prop = 0.5,
                                         extra_edge_sampling_width = 600,
                                         )
        BackgroundRemovedRGB = interim_RGB - BG_est
        
        # ShowImage(BG_weights[:,:,0]**2, Title = 'Red Deep Star Map Sampling')
        # ShowImage(BG_weights[:,:,1]**2, Title = 'Green Deep Star Map Sampling')
        # ShowImage(BG_weights[:,:,2]**2, Title = 'Blue Deep Star Map Sampling')
        
        # ShowImage(SSBB)
        # ShowImage(interim_apriori_BG_mask & star_mask)
        # ShowImage(BG_floor_est[:,:,0])
        # ShowImage(BG_floor_est[:,:,1])
        # ShowImage(BG_floor_est[:,:,2])

    
    T3 = time.time()
    
    
    
    # BackgroundRemovedRGB -= np.percentile(BackgroundRemovedRGB, 5)
    # ShowImage(BG_est.mean(axis = 2))

    
    print('\t',round(T3 - T2, 4) , 'seconds')
    ####################################################################### this can be sped up
    print('\n\n################  Color correction') 
    
    
    
    BackgroundRemovedRGB = mycolor.ColorConvolveImage(mycolor.full_spec_ASI2600MC_conv_mat, BackgroundRemovedRGB)
        
    T4 = time.time()
    #astroviz.QuickInspectRGB(BackgroundRemovedRGB, perc_black=5, perc_white = 99.5)
    print('\t',round(T4 - T3, 4) , 'seconds')
    
    
    #######################################################################
    print('\n\n################  Noise reduction (linear stage)') 
    
    color_similarity_scale = 2./2**16 # typically 1.0/2**16 , 0.5/2**16, or close to that for ASI +105mm and 100 ish frames
    
    NoiseRemovedRGB = cv2.bilateralFilter(np.array(BackgroundRemovedRGB, dtype = np.float32),  # 8-bit and float32 bit images are supported
                            20,   # 9, 15.  size of the kernel/window used in the filter, 15 takes about a second.  Smaller is faster
                            color_similarity_scale,  # 0.05 is too much, 0.02 is too low.  color proximity sigma/stdev for filter weighting (units = color units)
                            10)   # 25, 50.  spatial proximity sigma/stedv for filter weighting (units = pixels)
    #astroviz.QuickInspectRGB(NoiseRemovedRGB, perc_black=5, perc_white = 99.5)
    T5 = time.time()
    
    print('\t',round(T5 - T4, 4) , 'seconds')
    # ShowImageRGB(NoiseRemovedRGB)
    # ShowImage(NoiseRemovedRGB.mean(axis = 2))
    #######################################################################
    print('\n\n################  Histogram stretch') 
    
    #### Modified logrithmic stretching
    ## The intensities are log-stretched to produce a more accurate human eye visual resonse on the screen.
    ## The linear intensities are scaled up by multiplying by a power of 2, then the log2 trasnform takes place. 
    ## If you notice brown areas appearing in dense star regions I recommend decreasing the stretch exponent.  
    ## 14 is a good starting value, then decrease, or increase depending on the noise and BG estimation accuracy in your image
    stretch_exponent = 10# increase for a more aggressive stretch (ex: 16), 10, 12, 13, 14, 15, 16 seem to be the reasonable vals
    
    # This sets the zero point for the logarithmic stretch, negative brightens the background, postive darkens the background
    blackpoint_shift = 10./2**16  
    
    StretchedNoiseRemovedRGB = enhance.CustomPseudoLogStretch_cuda(NoiseRemovedRGB, 
                                                                   BP_S = blackpoint_shift, 
                                                                   BP_T = 0.03, 
                                                                   ExpN = stretch_exponent, 
                                                                   zz = 0.5,
                                                                   StretchChannel = '') # 'Luminance', 'Value', 'Intensity', or '' (unlinked)

    T6 = time.time()
    print('\t',round(T6 - T5, 4) , 'seconds')
    
    # ShowImageRGB(StretchedNoiseRemovedRGB)
    #astroviz.QuickInspectRGB(StretchedNoiseRemovedRGB, perc_black=5, perc_white = 99.5)
    #######################################################################
    print('\n\n################  Saturation and Value adjustment') 
    primary_boost = 1.0 # 1.2
    if primary_boost > 1.0:
        FinalFilteredRGB = enhance.SaturationValueBoost(np.array(StretchedNoiseRemovedRGB, dtype = np.float32), 
                                                        saturation_boost = primary_boost, 
                                                        value_boost = primary_boost) # works OK on Heart/Soul: saturation_boost = 1.5, value_boost = 1.8)
    else:
        FinalFilteredRGB = np.array(StretchedNoiseRemovedRGB, dtype = np.float32)
        
    T7 = time.time()
    print('\t',round(T7 - T6, 4) , 'seconds')
    

    # ShowImageRGB( enhance.SaturationValueBoost(np.array(StretchedNoiseRemovedRGB, dtype = np.float32), saturation_boost = .5, value_boost = .8, boost_method='exp'))

    #######################################################################
    print('\n\n################  Star reduction and starless processing')         

    StarReductionFactor = 2.0
    
    if StarReductionFactor > 1.0:
        
        try:
            starless_RGB = enhance.StarnetPP(FinalFilteredRGB, starnet_pp_path = gpu_starnet)
        except:
            starless_RGB = enhance.StarnetPP(FinalFilteredRGB, starnet_pp_path = cpu_starnet)
        
        #ShowImageRGB(starless_RGB)
        print('\n\n################  Noise reduction (stretched stage)') 
        starless_RGB_DN = cv2.bilateralFilter(np.array(starless_RGB, dtype = np.float32),  # 8-bit and float32 bit images are supported
                            25,   # 9, 15.  size of the kernel/window used in the filter, 15 takes about a second.  Smaller is faster
                            .035,  # 0.05 is too much, 0.02 is too low.  color proximity sigma/stdev for filter weighting (units = color units)
                            10)   # 25, 50.  spatial proximity sigma/stedv for filter weighting (units = pixels)
        
        
        
        #ShowImageRGB(starless_RGB_DN)
        #ShowImageRGB(starless_RGB)
        
        T8 = time.time()
        print('\t',round(T8 - T7, 4) , 'seconds')
        
        #######################################################################
        print('\n\n################  Star Reduction') 
        
        starsOnly_RGB = enhance.GSubtractImages(FinalFilteredRGB, starless_RGB)
        
        StarReducedRGB = enhance.GaussianStarReduction(starsOnly_RGB, 
                                                       StarReductionFactor = StarReductionFactor,
                                                       ReductionChannel = 'Luminance',
                                                       StarFilterDiam = 3)
            
        # ShowRGB(starsOnly_RGB)
        # ShowImageRGB(StarReducedRGB)
        
        StarReducedRGB = enhance.SaturationValueBoost(StarReducedRGB, saturation_boost = 1.4, value_boost = 1.0)
        
        starless_RGB_DN = enhance.SaturationValueBoost(starless_RGB_DN, saturation_boost = 1.6, value_boost = 2.5) #1.8)
        
        # this can help early in the stack with higher nosie. (dont go above 5?)
        if proc_count < 20:
            starless_RGB_DN = myfilters.cuda_medfiltRGB(starless_RGB_DN, 5)
        elif proc_count < 50:
            starless_RGB_DN = myfilters.cuda_medfiltRGB(starless_RGB_DN, 3)
            
        FinalFinalFilteredRGB = enhance.GAddImages(StarReducedRGB, starless_RGB_DN)

        
        # ShowImageRGB(starless_RGB_DN)
        # ShowImageRGB(myfilters.cuda_medfiltRGB(starless_RGB_DN, 15))
        # ShowImageRGB(StarReducedRGB)
        
        
    else:
        FinalFinalFilteredRGB = enhance.SaturationValueBoost(FinalFilteredRGB, saturation_boost = 1.7, value_boost = 2.3) 
        T8 = time.time()
    
    if live_figure:
        
        live_ax.clear()
        
        title_text = "Live processing: frame=" + ' ' + progress_str + ' ' + stack_token

        vis_image = live_ax.imshow(FinalFinalFilteredRGB) 
        
        middle_coord_x = FinalFinalFilteredRGB.shape[1]/2
        top_coord_y = 0.005 * FinalFinalFilteredRGB.shape[0]
        live_ax.text(middle_coord_x, top_coord_y, 
                     title_text+extra_text, 
                     fontsize=12,  
                     ha='center', va='top', 
                     wrap=True, color=text_color)
    
        plt.pause(2)
        plt.pause(.1)
        
    else:
        ShowImageRGB(FinalFinalFilteredRGB, Title = stack_token + '', dark_mode = True )
        plt.pause(2)
        plt.pause(.1)

#%%

if False:

    fileio.SaveImage2Disk(FinalFinalFilteredRGB, 
                          output_dir,
                          starless_version = starless_RGB_DN, 
                          description = stack_token, 
                          save_token = 'AstroComPYute_v1')




