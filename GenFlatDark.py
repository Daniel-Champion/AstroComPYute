# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 17:27:08 2024

@author: champ
"""

import numpy as np
from copy import deepcopy
import os

from astrocompyute.visualization import ContourMultiPlot, CollectUserInput, ShowImage
from astrocompyute.filters import cuda_medfiltRGB, cuda_medfilt2d
from astrocompyute.imagemath import GetImageCoords
import astrocompyute.fileio as fileio


#%%



def EmpriricalFlatFancy(flat_dir_subs, 
                        medfilt_ls = 25, 
                        normalize_intensity = 0.5, 
                        offset = 700./2**16, 
                        return_dataCube = False, 
                        max_files = 100, 
                        low_memory = False):
    
    filesproc = [fn for fn in os.listdir(flat_dir_subs) if fn[-3:] in ['tif', 'cr2', 'cr3']]
    f_nnn = min(max_files, len(filesproc))

    fp = flat_dir_subs + '/' + filesproc[0]
    imRGB, imBayer = fileio.ReadASI_TIF_FITS(fp, salt = False)
    pixN = imRGB.shape[0]*imRGB.shape[1]
    _sensor_shape_rgb = imRGB.shape

    MasterFlat_SigmaFrame_RGB = np.zeros(_sensor_shape_rgb, dtype = float)
    MasterFlat_MedianFrame_RGB = np.zeros(_sensor_shape_rgb, dtype = float)
    MasterFlat_Sum_RGB = np.zeros(_sensor_shape_rgb, dtype = float)
    MasterFlat_SquaredSum_RGB = np.zeros(_sensor_shape_rgb, dtype = float)
    
    MasterFlat_RGB = np.zeros(_sensor_shape_rgb, dtype = float)
    MasterFlat_RGB_lag = np.zeros(_sensor_shape_rgb, dtype = float)

    if low_memory:
        w_sum = 0.
        w_sum2 = 0.
        
        print('Reading flats')
        for ifn, fn in enumerate(filesproc[:f_nnn]):
            print('reading file (LM mode):', ifn, f_nnn, fn)
            fp = flat_dir_subs + '/' + fn
            imRGB, imBayer = fileio.ReadASI_TIF_FITS(fp, salt = False)
            #print(ifn, f_nnn, fn, 'b')
            w_sum += 1.0
            w_sum2 += 1.0
            
            MasterFlat_RGB_lag = deepcopy(MasterFlat_RGB)
            
            MasterFlat_RGB = MasterFlat_RGB_lag + (1. / w_sum) * (imRGB - MasterFlat_RGB_lag)
            
            MasterFlat_SigmaFrame_RGB = MasterFlat_SigmaFrame_RGB + (imRGB - MasterFlat_RGB_lag) * (imRGB - MasterFlat_RGB)
            
            #print(ifn, f_nnn, fn, 'c')
        MasterFlat_SigmaFrame_RGB = MasterFlat_SigmaFrame_RGB / w_sum
        MasterFlat_MedianFrame_RGB = deepcopy(MasterFlat_RGB)
    else:
        
        MF_R = np.zeros((f_nnn, _sensor_shape_rgb[0], _sensor_shape_rgb[1]), dtype = float)
        MF_G = np.zeros((f_nnn, _sensor_shape_rgb[0], _sensor_shape_rgb[1]), dtype = float)
        MF_B = np.zeros((f_nnn, _sensor_shape_rgb[0], _sensor_shape_rgb[1]), dtype = float)
            
        print('Reading flats')
        for ifn, fn in enumerate(filesproc[:f_nnn]):
            print('reading file:', ifn, f_nnn, fn)
            fp = flat_dir_subs + '/' + fn
            imRGB, imBayer = fileio.ReadASI_TIF_FITS(fp, salt = False)
            #print(ifn, f_nnn, fn, 'b')
            MasterFlat_RGB += imRGB
            
            MF_R[ifn, :, :] = imRGB[:,:,0]
            MF_G[ifn, :, :] = imRGB[:,:,1]
            MF_B[ifn, :, :] = imRGB[:,:,2]
            
            #print(ifn, f_nnn, fn, 'c')
        
        MasterFlat_RGB *= (1.0/f_nnn)
        
        ## Numpy
        MasterFlat_SigmaFrame_RGB[:,:,0] = np.std(MF_R, axis = 0)
        MasterFlat_SigmaFrame_RGB[:,:,1] = np.std(MF_G, axis = 0)
        MasterFlat_SigmaFrame_RGB[:,:,2] = np.std(MF_B, axis = 0)
        
        MasterFlat_MedianFrame_RGB[:,:,0] = np.median(MF_R, axis = 0)
        MasterFlat_MedianFrame_RGB[:,:,1] = np.median(MF_G, axis = 0)
        MasterFlat_MedianFrame_RGB[:,:,2] = np.median(MF_B, axis = 0) 
    
    print('spatial median filtering flats')
    # MasterFlat_Median_RGB = np.stack([MasterFlat_R_median, MasterFlat_G_median, MasterFlat_B_median], axis = -1)
    # MasterFlat_Median_RGB = cv2.medianBlur(MasterFlat_Median_RGB,medfilt_ls)
    MasterFlat_MedianSpatial_RGB = np.zeros(_sensor_shape_rgb, dtype = float)
    MasterFlat_MedianSpatial_RGB[:,:,0] = cuda_medfilt2d(MasterFlat_RGB[:,:,0], medfilt_ls)
    MasterFlat_MedianSpatial_RGB[:,:,1] = cuda_medfilt2d(MasterFlat_RGB[:,:,1], medfilt_ls)
    MasterFlat_MedianSpatial_RGB[:,:,2] = cuda_medfilt2d(MasterFlat_RGB[:,:,2], medfilt_ls)
    
    #MasterFlat_MedianSpatial_RGB[MasterFlat_MedianSpatial_RGB <= 0 ] = MasterFlat_RGB[MasterFlat_MedianSpatial_RGB <= 0]

    if normalize_intensity > 0.0:
        
        mean_adu = np.sum(MasterFlat_MedianSpatial_RGB - offset) / pixN
        norm_scalar = normalize_intensity / mean_adu
        MasterFlat_MedianSpatial_RGB = (MasterFlat_MedianSpatial_RGB - offset) * norm_scalar + offset

    if return_dataCube:
        return MasterFlat_MedianSpatial_RGB, MasterFlat_RGB, MF_R, MF_G, MF_B, MasterFlat_SigmaFrame_RGB, MasterFlat_MedianFrame_RGB#MasterFlat_RGB_Gauss

    return MasterFlat_MedianSpatial_RGB, MasterFlat_RGB, MasterFlat_SigmaFrame_RGB, MasterFlat_MedianFrame_RGB#MasterFlat_RGB_Gauss




#%%


#### Collect data from the user
collected_user_data = CollectUserInput(['DirectoryDialog', 
                                        'TextBox', 
                                        'RadioButtons', 
                                        'RadioButtons', 
                                        'RadioButtons', 
                                        'TextBox', 
                                        'TextBox',
                                        'DirectoryDialog'],  # list of either 'RadioButtons','TextBox', 'FileDialog', 'DirectoryDialog'
                          ['Images Directory',  
                           'Descriptive Name', 
                           'Processing Mode', 
                           'Low Memory Processing', 
                           'In Situ Flat', 
                           'Filter Length Scale (pixels)', 
                           'Acquisition Offset',
                           'Output Directory'], # unique names for these elements
                          ['', 
                           '', 
                           ['Dark', 'Flat'] , 
                           ['No', 'Yes'], 
                           ['No', 'Yes'], 
                           25, 
                           700,
                           ''],          # for RadioButtons, these are the list of button values, for TextBox, this is the initial fill value
                          v_space_per_element = 0.08 ,
                          left_text_space = 0.2)


## process the collected user data
collected_user_data = { lll[0]:lll[2] for lll in collected_user_data}

# location of the images
FramesDir = collected_user_data['Images Directory']

InSituFlat = collected_user_data['In Situ Flat']
InSituFlat = InSituFlat == 'Yes'

mode = collected_user_data['Processing Mode']
low_memory = collected_user_data['Low Memory Processing']

if InSituFlat:
    mode = 'Flat'
    low_memory = 'No'
    insitu = '_insitu'
else:
    insitu = ''
    
master_frame_filter_length_scale = int(round(collected_user_data['Filter Length Scale (pixels)']))
acquisition_offset = float(collected_user_data['Acquisition Offset'])

if low_memory == 'Yes':
    lm = '_LM'
    low_memory = True
else:
    lm = ''
    low_memory = False


master_frame_name = 'Master'+mode + '__' + collected_user_data['Descriptive Name'] + '__Ls=' + str(master_frame_filter_length_scale) + '_offset=' + str(int(round(acquisition_offset))) + lm + insitu + '.npy'


MasterDir = collected_user_data['Output Directory']




#%%############################################################################
#### Processing parameters
###############################################################################

bit_depth = 2**16


#%%############################################################################
#### Frames are loaded into memory and summarized/smoothed
###############################################################################


[MasterFlat_MedianSpatial_RGB, 
 MasterFlat_RGB, 
 MasterFlat_sigma_RGB, 
 MasterFlat_MedianFrame_RGB] = EmpriricalFlatFancy(FramesDir, 
                                                   medfilt_ls = master_frame_filter_length_scale, 
                                                   normalize_intensity = 0.5, 
                                                   offset = acquisition_offset/bit_depth,
                                                   low_memory = low_memory)

                                                   

#%%                                                   
                                                   
                                                   
#### Perform the correction/normalization depending on the mode of operation
                                                   
if mode == 'Dark':                                              
    MasterFlat = deepcopy(MasterFlat_MedianFrame_RGB)
elif InSituFlat:
    MasterFlat = deepcopy(MasterFlat_MedianFrame_RGB) - acquisition_offset/bit_depth
    MasterFlat = cuda_medfiltRGB(MasterFlat, master_frame_filter_length_scale)
    MasterFlat /= MasterFlat.max()
else:                                                                                                
    MasterFlat = MasterFlat_MedianSpatial_RGB - acquisition_offset/bit_depth
    MasterFlat /= MasterFlat.max()

#%%############################################################################
#### Output 
###############################################################################

master_fp = os.path.join(MasterDir, master_frame_name)
                                               
np.save(master_fp, MasterFlat)   
                                    
#%%############################################################################
#### Visualization and Inspection
###############################################################################

ImageRows, ImageCols = GetImageCoords(MasterFlat)

perc0, perc1, perc99, perc100 = np.percentile(MasterFlat*2**16, [0, 1, 99, 100])

#### Plot the master that was created for user review and inspection
ShowImage(np.hstack([MasterFlat[:,:,0], MasterFlat[:,:,1], MasterFlat[:,:,2]])*2**16, 
          min_color_val=perc1, max_color_val=perc99, 
          Title = ' 0%=' + str(round(perc0,4)) + ' 1%=' + str(round(perc1,4)) + ' 99%=' + str(round(perc99,4)) + ' 100%=' + str(round(perc100,4))  ) 

if mode == 'Flat':
    ContourMultiPlot(ImageCols, ImageRows, MasterFlat[:,:,0]/MasterFlat[:,:,0].max(), # must be in meshgrid format
    
                              x_axis_label = 'Pixel Columns',
                              y_axis_label = 'Pixel Rows',
                              z_axis_label = 'Normalized Intensity',
                              
                              num_contours = 25,
                              
                              Title = 'Red Channel',
                              aspect_style = 'equal')
                    
    ContourMultiPlot(ImageCols, ImageRows, MasterFlat[:,:,1]/MasterFlat[:,:,1].max(), # must be in meshgrid format
    
                              x_axis_label = 'Pixel Columns',
                              y_axis_label = 'Pixel Rows',
                              z_axis_label = 'Normalized Intensity',
                              
                              num_contours = 25,
                              
                              Title = 'Green Channel',
                              aspect_style = 'equal')
    
    ContourMultiPlot(ImageCols, ImageRows, MasterFlat[:,:,2]/MasterFlat[:,:,2].max(), # must be in meshgrid format
    
                              x_axis_label = 'Pixel Columns',
                              y_axis_label = 'Pixel Rows',
                              z_axis_label = 'Normalized Intensity',
                              
                              num_contours = 25,
                              
                              Title = 'Blue Channel',
                              aspect_style = 'equal')                      





print('MasterFrame: ', master_fp)












